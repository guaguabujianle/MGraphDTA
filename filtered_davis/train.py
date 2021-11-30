# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import argparse

from metrics import get_cindex
from dataset import *
from model import MGraphDTA
from utils import *
from log.train_logger import TrainLogger

def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            running_loss.update(loss.item(), label.size(0))

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--fold', required=True, help='k-th fold, from 0 to 4')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    args = parser.parse_args()

    params = dict(
        data_root="data",
        save_dir="save",
        dataset="filtered_davis",
        fold_num=args.fold,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    fold_num = params.get("fold_num")
    fpath = os.path.join(data_root, DATASET)

    train_index, val_index, test_index = read_sets(fpath, fold_num, split_type='warm')

    dataset = GNNDataset(fpath)
    train_set = dataset[train_index]
    val_set = dataset[val_index]
    test_set = dataset[test_index]

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=8)

    device = torch.device('cuda:0')
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)

    epochs = 2000
    steps_per_epoch = 100
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    break_flag = False

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    global_step = 0
    global_epoch = 0
    early_stop_epoch = 100

    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    running_best_mse = BestMeter("min")

    model.train()

    for i in range(num_iter):
        if break_flag:
            break

        for data in train_loader:

            global_step += 1       
            data = data.to(device)
            pred = model(data)

            loss = criterion(pred.view(-1), data.y.view(-1))
            cindex = get_cindex(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), data.y.size(0)) 
            running_cindex.update(cindex, data.y.size(0))

            if global_step % steps_per_epoch == 0:

                global_epoch += 1

                epoch_loss = running_loss.get_average()
                epoch_cindex = running_cindex.get_average()
                running_loss.reset()
                running_cindex.reset()

                val_loss = val(model, criterion, val_loader, device)
                test_loss = val(model, criterion, test_loader, device)

                msg = "epoch-%d, loss-%.4f, cindex-%.4f, val_loss-%.4f, test_loss-%.4f" % (global_epoch, epoch_loss, epoch_cindex, val_loss, test_loss)
                logger.info(msg)

                if test_loss < running_best_mse.get_best():
                    running_best_mse.update(test_loss)
                    if save_model:
                        save_model_dict(model, logger.get_model_dir(), msg)
                else:
                    count = running_best_mse.counter()
                    if count > early_stop_epoch:
                        logger.info(f"early stop in epoch {global_epoch}")
                        break_flag = True
                        break

if __name__ == "__main__":
    main()
