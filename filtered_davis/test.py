# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import argparse
from scipy.stats import spearmanr

from metrics import get_cindex
from dataset import *
from model import MGraphDTA
from utils import *

def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            pred_list.append(pred.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    epoch_cindex = get_cindex(label, pred)
    epoch_spearm = spearmanr(label, pred)[0]
    epoch_rmse = np.sqrt(running_loss.get_average())
    running_loss.reset()

    return epoch_rmse, epoch_cindex, epoch_spearm

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--model_path', required=True, type=str, help='model path ready to load')
    args = parser.parse_args()

    data_root = "data"
    DATASET = "filtered_davis"
    model_path = args.model_path

    fpath = os.path.join(data_root, DATASET)

    _, _, test_index = read_sets(fpath, 0, split_type='warm')
    dataset = GNNDataset(fpath)
    test_set = dataset[test_index]
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=8)

    device = torch.device('cuda:0')
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)

    criterion = nn.MSELoss()
    load_model_dict(model, model_path)
    test_rmse, test_cindex, test_spearm = val(model, criterion, test_loader, device)
    msg = "test_rmse:%.4f, test_cindex:%.4f, test_spearm:%.4f" % (test_rmse, test_cindex, test_spearm)
    print(msg)


if __name__ == "__main__":
    main()
