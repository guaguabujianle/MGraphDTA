# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import torch
from torch_geometric.data import Batch
import pandas as pd
from matplotlib.colors import ListedColormap
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from IPython.display import SVG
import cairosvg
import cv2
import matplotlib.cm as cm
from tqdm import tqdm

from model import MGraphDTA
from dataset import *
from utils import *

class GradAAM():
    def __init__(self, model, module):
        self.model = model
        module.register_forward_hook(self.save_hook)
        self.target_feat = None

    def save_hook(self, md, fin, fout):
        self.target_feat = fout.x   

    def __call__(self, data):
        self.model.eval()

        output = self.model(data).view(-1)
        grad = torch.autograd.grad(output, self.target_feat)[0]
        channel_weight = torch.mean(grad, dim=0, keepdim=True)
        channel_weight = normalize(channel_weight)
        weighted_feat = self.target_feat * channel_weight
        cam = torch.sum(weighted_feat, dim=-1).detach().cpu().numpy()
        cam = normalize(cam)

        return output.detach().cpu().numpy(), cam

def clourMol(mol,highlightAtoms_p=None,highlightAtomColors_p=None,highlightBonds_p=None,highlightBondColors_p=None,sz=[400,400], radii=None):
    d2d = rdMolDraw2D.MolDraw2DSVG(sz[0], sz[1])
    op = d2d.drawOptions()
    op.dotsPerAngstrom = 40
    op.useBWAtomPalette()
    mc = rdMolDraw2D.PrepareMolForDrawing(mol)
    d2d.DrawMolecule(mc, legend='', highlightAtoms=highlightAtoms_p,highlightAtomColors=highlightAtomColors_p, highlightBonds= highlightBonds_p,highlightBondColors=highlightBondColors_p, highlightAtomRadii=radii)
    d2d.FinishDrawing()
    svg = SVG(d2d.GetDrawingText())
    res = cairosvg.svg2png(svg.data, dpi = 600, output_width=2400, output_height=2400)
    nparr = np.frombuffer(res, dtype=np.uint8)
    segment_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return segment_data

def main():
    device = torch.device('cuda:0')
    
    fpath = os.path.join('data', 'full_toxcast')
    test_df = pd.read_csv(os.path.join(fpath, 'raw', 'data_test.csv'))
    test_set = GNNDataset(fpath, train=False)

    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)
    try:
        load_model_dict(model, 'pretrained_model/epoch-173, loss-0.1468, cindex-0.8998, test_loss-0.1750.pt')
    except:
        model_dict = torch.load('pretrained_model/epoch-173, loss-0.1468, cindex-0.8998, test_loss-0.1750.pt')
        for key, val in model_dict.copy().items():
            if 'lin_l' in key:
                new_key = key.replace('lin_l', 'lin_rel')
                model_dict[new_key] = model_dict.pop(key)
            elif 'lin_r' in key:
                new_key = key.replace('lin_r', 'lin_root')
                model_dict[new_key] = model_dict.pop(key)
        model.load_state_dict(model_dict)

    gradcam = GradAAM(model, module=model.ligand_encoder.features.transition3)

    bottom = cm.get_cmap('Blues_r', 256)
    top = cm.get_cmap('Oranges', 256)
    newcolors = np.vstack([bottom(np.linspace(0.35, 0.85, 128)), top(np.linspace(0.15, 0.65, 128))])
    newcmp = ListedColormap(newcolors, name='OrangeBlue')

    smile_list = list(test_df['smiles'].unique())

    progress_bar = tqdm(total=len(smile_list))

    for idx in range(len(test_set)):
        smile = test_df.iloc[idx]['smiles']

        if len(smile_list) == 0:
            break
        if smile in smile_list:
            smile_list.remove(smile)
        else:
            continue

        data = Batch.from_data_list([test_set[idx]])
        data = data.to(device)
        _, atom_att = gradcam(data)

        mol = Chem.MolFromSmiles(smile)
        atom_color = dict([(idx, newcmp(atom_att[idx])[:3]) for idx in range(len(atom_att))])
        radii = dict([(idx, 0.2) for idx in range(len(atom_att))])
        img = clourMol(mol,highlightAtoms_p=range(len(atom_att)), highlightAtomColors_p=atom_color, radii=radii)

        cv2.imwrite(os.path.join('results', f'{idx}.png'), img)

        progress_bar.update(1)

if __name__ == '__main__':
    main()

