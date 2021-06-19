# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse

from metrics import accuracy, precision, auc_score, recall
from dataset import *
from model import MGraphDTA
from utils import *
from log.train_logger import TrainLogger


def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    pred_cls_list = []
    label_list = []

    for data in dataloader:
        data.y = data.y.long()  
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred, data.y)
            pred_cls = torch.argmax(pred, dim=-1)

            pred_prob = F.softmax(pred, dim=-1)
            pred_prob, indices = torch.max(pred_prob, dim=-1)
            pred_prob[indices == 0] = 1. - pred_prob[indices == 0]

            pred_list.append(pred_prob.view(-1).detach().cpu().numpy())
            pred_cls_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(data.y.detach().cpu().numpy())
            running_loss.update(loss.item(), data.y.size(0))

    pred = np.concatenate(pred_list, axis=0)
    pred_cls = np.concatenate(pred_cls_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc = accuracy(label, pred_cls)
    pre = precision(label, pred_cls)
    rec = recall(label, pred_cls)
    auc = auc_score(label, pred)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss, acc, pre, rec, auc

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--dataset', required=True, help='human or celegans')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    args = parser.parse_args()

    params = dict(
        data_root="data",
        save_dir="save",
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)

    train_set = GNNDataset(fpath, types='train')
    val_set = GNNDataset(fpath, types='val')
    test_set = GNNDataset(fpath, types='test')

    logger.info(f"Number of train: {len(train_set)}")
    logger.info(f"Number of val: {len(val_set)}")
    logger.info(f"Number of test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=False, num_workers=8)

    device = torch.device('cuda:0')
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=2).to(device)

    epochs = 100
    steps_per_epoch = 10
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    global_epoch = 0

    running_loss = AverageMeter()

    model.train()

    for i in range(num_iter):
        for data in train_loader:

            global_step += 1   
            data.y = data.y.long()    
            data = data.to(device)
            pred = model(data)

            loss = criterion(pred, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), data.y.size(0)) 

            if global_step % steps_per_epoch == 0:

                global_epoch += 1

                epoch_loss = running_loss.get_average()
                running_loss.reset()

                val_loss, val_acc, val_pre, val_rec, val_auc  = val(model, criterion, val_loader, device)
                test_loss, test_acc, test_pre, test_rec, test_auc  = val(model, criterion, test_loader, device)

                msg = "epoch-%d, loss-%.4f, val_auc-%.4f, test_loss-%.4f, test_acc-%.4f, test_pre-%.4f, test_rec-%.4f, test_auc-%.4f" % (global_epoch, epoch_loss, val_auc, test_loss, test_acc, test_pre, test_rec, test_auc)
                logger.info(msg)

                if save_model:
                    save_model_dict(model, logger.get_model_dir(), msg)


if __name__ == "__main__":
    main()
