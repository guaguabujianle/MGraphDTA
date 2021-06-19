# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import argparse

from metrics import get_cindex, get_rm2
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
    epoch_r2 = get_rm2(label, pred)
    epoch_loss = running_loss.get_average()
    running_loss.reset()

    return epoch_loss, epoch_cindex, epoch_r2

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--dataset', required=True, help='davis or kiba')
    parser.add_argument('--model_path', required=True, type=str, help='model path ready to load')
    args = parser.parse_args()

    data_root = "data"
    DATASET = args.dataset
    model_path = args.model_path

    fpath = os.path.join(data_root, DATASET)

    test_set = GNNDataset(fpath, train=False)
    print("Number of test: ", len(test_set))
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8)

    device = torch.device('cuda:0')
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)

    criterion = nn.MSELoss()
    load_model_dict(model, model_path)
    test_loss, test_cindex, test_r2 = val(model, criterion, test_loader, device)
    msg = "test_loss:%.4f, test_cindex:%.4f, test_r2:%.4f" % (test_loss, test_cindex, test_r2)
    print(msg)


if __name__ == "__main__":
    main()
