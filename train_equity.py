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
from agents.model import *
from utils import *

def main(config_path):
    cudnn.benchmark = True

    config = toml.load(config_path)
    writer = SummaryWriter()

    # model
    model_crt  = DF(18,6)

    # resume from a checkpoint
    if config["model"]["load"]:
        checkpoint = torch.load(config["model"]["load_path"])
        model_crt.load_state_dict(checkpoint['model'])
        print("successfully load model")

    model_crt.cuda()

    # data
    dataset_train = data.Equity_DATASET(config["general"]["data_path"])
    dataloader_train = Data.DataLoader(dataset_train, batch_size=30000, shuffle=True, pin_memory=True,
                        num_workers=6 , drop_last=False) 

    # criterion
    criterion = nn.KLDivLoss(reduction="batchmean")

    # optim
    params = [
        {"params": model_crt.parameters(), "lr": config["hyperparameters"]["lr"]}
    ]
    optimizer = optim.Adam(params, betas=(config["hyperparameters"]["betas"], 0.999), weight_decay=config["hyperparameters"]["decay"])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    if not os.path.exists(config["model"]["save_path"]):
        os.mkdir(config["model"]["save_path"])

    train_length = len(dataloader_train)
    for epoch in range(1, 1000):
        for (i, data_in) in enumerate(dataloader_train):
            train_package = [
                data_in,
                model_crt,
                criterion,
                optimizer,
                writer,
                config,
                epoch,
                i,
                train_length
            ]
            train(train_package)

        if epoch % 10 == 0:
            save_checkpoint({
                "model": model_crt.state_dict(),
            }, config["model"]["save_path"] + "checkpoint_{}.pt".format(epoch))
            print("save model successfully")
        if epoch % 20 == 0:
            lr_scheduler.step(epoch-config["general"]["start_epoch"])

def train(package):
    [data, 
     model, 
     criterion, 
     optimizer, 
     writer,
     config,
     epoch,
     i,
     train_length] = package

    model.train()
    holes , pubs, history, labels = data
    holes = holes.cuda()
    pubs = pubs.cuda()
    history = history.cuda()
    labels = labels.cuda()

    predict = model(holes, pubs, history)
    predict = torch.log(predict)

    loss = criterion(predict, labels)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    optimizer.zero_grad()

    if i % 10 == 0:
        print("iteration: {} epoch: {} loss: {:.6f}".format(i, epoch, loss))
    writer.add_scalars("train loss", {"loss": loss}, i + epoch * train_length)

if __name__ == "__main__":
    main("./train.toml")