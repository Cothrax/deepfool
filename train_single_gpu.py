import os
import toml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import data
from model import DF
from utils import *

def main(config_path):
    cudnn.benchmark = True

    config = toml.load(config_path)
    writer = SummaryWriter("runs/poker")

    # model
    model_crt  = DF(6, 7)
    model_last = DF(6, 7)
    for p in model_last.parameters():
        p.requires_grad = False

    # resume from a checkpoint
    if config["model"]["load"]:
        checkpoint = torch.load(config["general"]["load_path"])
        model_crt.load_state_dict(checkpoint['model'])
        model_last.load_state_dict(checkpoint['model'])
        print("successfully load model")

    model_crt.cuda()
    #model = nn.DataParallel(model)


    # data
    dataset_train = data.POKER_DATASET(model_last, 10, 4)
    dataloader_train = dataset_train
    '''
    dataloader_train = Data.DataLoader(dataset_train, batch_size=1, shuffle=False, pin_memory=True,
                        num_workers=config["general"]["num_workers"] , drop_last=False) 
    '''

    # criterion
    criterion = nn.L1Loss()

    # optim
    params = [
        {"params": model_crt.parameters(), "lr": config["hyperparameters"]["lr"]}
    ]
    optimizer = optim.Adam(params, betas=(config["hyperparameters"]["betas"], 0.999), weight_decay=config["hyperparameters"]["decay"])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(config["general"]["start_epoch"], config["general"]["start_epoch"] + config["general"]["epochs"]):
        train_package = [
            dataloader_train,
            model_crt,
            criterion,
            optimizer,
            lr_scheduler,
            writer,
            config,
            epoch
        ]
        train(train_package)

        if (epoch + 1) % config["model"]["save_iter"] == 0:
            save_checkpoint({
                "model": model_crt.state_dict(),
            }, config["general"]["save_path"] + "checkpoint_{}.pt".format(epoch+1))

        model_last.load_state_dict(model_crt.state_dict())

        #lr_scheduler.step(epoch-config["general"]["start_epoch"])

def train(package):
    [dataloader, 
     model, 
     criterion, 
     optimizer, 
     lr_scheduler,
     writer,
     config,
     epoch] = package

    model.train()
    holes, pubs, history, label = dataloader.__getitem__()
    label = label.cuda()
    holes = holes.cuda()
    pubs = pubs.cuda()
    history = history.cuda()
    predict = model(holes, pubs, history)
    loss = criterion(predict, label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print("epoch: {} loss: {:.6f}".format(epoch + 1, loss))
        writer.add_scalars("train loss", {"sum": loss}, epoch+1)

if __name__ == "__main__":
    main("./train.toml")