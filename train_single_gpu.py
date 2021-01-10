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
    global iteration
    iteration = 1
    cudnn.benchmark = True
    torch.set_num_threads(1)

    config = toml.load(config_path)
    writer = SummaryWriter()

    # model
    model_crt = DF(18, 6)
    if config["general"]["straight_sampling"]:
        model_last = DF(18, 6)
        model_last.eval()
        for m in model_last.parameters():
            m.requires_grad = False
    else:
        model_last = [DF(18, 6)] * config["general"]["num_cfr"]
        for m in model_last:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    # resume from a checkpoint
    if config["model"]["load"]:
        checkpoint = torch.load(config["model"]["load_path"])
        model_crt.load_state_dict(checkpoint)
        if config["general"]["straight_sampling"]:
            model_last.load_state_dict(checkpoint)
        else:
            for m in model_last:
                m.load_state_dict(checkpoint)
        print("successfully load model")
    
    # freeze part of model
    for name, param in model_crt.named_parameters():
        if "hist" in name or "post_process.0" in name:
            continue
        else:
            param.requires_grad = False

    # data
    dataset_train = data.POKER_DATASET(model_last, config["general"]["max_search_iter"], config["general"]["straight_sampling"])
    dataloader_train = dataset_train

    # criterion
    criterion = nn.KLDivLoss(reduction="batchmean")

    # optim
    params = [
        {"params": model_crt.parameters(), "lr": config["hyperparameters"]["lr"]}
    ]
    optimizer = optim.Adam(params, betas=(config["hyperparameters"]["betas"], 0.999), weight_decay=config["hyperparameters"]["decay"])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    while(iteration < config["general"]["max_iter"]):
        train_package = [
            dataloader_train,
            model_crt,
            criterion,
            optimizer,
            lr_scheduler,
            writer,
            config,
        ]
        train(train_package)

        if iteration % config["model"]["save_iter"] == 0:
            save_checkpoint(model_crt.state_dict(),
             config["model"]["save_path"] + "checkpoint_{}.pt".format(iteration))
            print("save model successfully")

        model_params = model_crt.state_dict()
        if config["general"]["straight_sampling"]:
            model_last.load_state_dict(model_params)
        else:
            for m in model_last:
                m.load_state_dict(model_params)

def train(package):
    global iteration
    [dataloader, 
     model, 
     criterion, 
     optimizer, 
     lr_scheduler,
     writer,
     config] = package

    model.train()
    all_holes, all_pubs, all_history, all_label = dataloader.__getitem__()
    length = all_holes.shape[0]
    print("sample length {}".format(length))
    st = 0
    loss = 0
    penalty = 0
    ctr = 0
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

            predict = model(holes, pubs, history)
            predict = torch.log(predict)
            loss_ = criterion(predict, label)
            loss_sum = loss_
            loss_sum.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad()
            st = ed
            loss += loss_.item()
            ctr += 1
    loss = loss / ctr
    print("iteration: {} loss: {:.6f} penalty: {:.6f}".format(iteration, loss, penalty))
    writer.add_scalars("train loss", {"loss": loss, "penalty": penalty}, iteration)
    iteration += 1

if __name__ == "__main__":
    main("./train.toml")