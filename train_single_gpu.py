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
    model_crt  = DF(18, 7)
    model_last = DF(18, 7)
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
    dataset_train = data.POKER_DATASET(model_last, config["general"]["max_search_iter"], 4)
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

    iteration = 1
    while(iteration < config["general"]["max_iter"]):
        train_package = [
            dataloader_train,
            model_crt,
            criterion,
            optimizer,
            lr_scheduler,
            writer,
            config,
            iteration
        ]
        train(train_package)

        if iteration % config["model"]["save_iter"] == 0:
            save_checkpoint({
                "model": model_crt.state_dict(),
            }, config["model"]["save_path"] + "checkpoint_{}.pt".format(iteration))

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
     iteration] = package

    model.train()
    all_holes, all_pubs, all_history, all_label = dataloader.__getitem__()
    length = all_holes.shape[0]
    st = 0
    while(True):
        if st >= length:
            break
        else:
            ed = st + config["general"]["max_samples"]
            ed = ed if ed < length else length
            label = all_label[st:ed]
            holes = all_holes[st:ed]
            pubs = all_pubs[st:ed]
            history = all_history[st:ed]

            label = label.cuda()
            holes = holes.cuda()
            pubs = pubs.cuda()
            history = history.cuda()

            predict = model(holes, pubs, history)
            loss = criterion(predict, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            st = ed
            print("iteration: {} loss: {:.6f}".format(iteration, loss))
            writer.add_scalars("train loss", {"sum": loss}, iteration)
            iteration += 1

if __name__ == "__main__":
    main("./train.toml")