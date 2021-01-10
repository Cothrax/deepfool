#DATA

import os
import glob
import torch
import pickle as pkl
import random
import numpy as np
import torch.utils.data as Data
from cfr_py.mp_cfr import ParallelCFR
from cfr_py.pure_cfr import ParallelPureCFR

class POKER_DATASET(object):
    def __init__(self, model, max_iter, straight_sampling):
        self.straight_sampling = straight_sampling
        if self.straight_sampling:
            self.model = model
            self.cfr = ParallelCFR(6, max_iter)
            print("using straight sampling")
        else:
            self.model_list = model
            self.cfr = ParallelPureCFR(len(self.model_list), max_iter, self.model_list)
            self.ctr = 0
            self.holes = None
            self.pubs = None
            self.history = None
            self.new = None
            print("using pure cfr sampling")
    
    def __getitem__(self):
        if self.straight_sampling:
            self.cfr.parallel_search()
            for cfr in self.cfr.cfr_list:
                holes, pubs, history_ = zip(*cfr.samples)
                holes = torch.from_numpy(np.array(holes).astype(np.int64))
                pubs = torch.from_numpy(np.array(pubs).astype(np.int64))
                history = np.array(history_).astype(np.float32)
                history = torch.from_numpy(history)
                old_label = self.model(holes, pubs, history)
                cfr.strategies = old_label.numpy()

            self.cfr.parallel_run()

            all_holes, all_pubs, all_history, all_new = [], [], [], []
            for cfr in self.cfr.cfr_list:
                samples, new = zip(*cfr.labels)
                holes, pubs, history_ = zip(*samples)
                holes = torch.from_numpy(np.array(holes).astype(np.int64))
                pubs = torch.from_numpy(np.array(pubs).astype(np.int64))
                history = torch.from_numpy(np.array(history_).astype(np.float32))
                new = torch.from_numpy(np.array(new).astype(np.float32))
                all_holes.append(holes)
                all_pubs.append(pubs)
                all_history.append(history)
                all_new.append(new)
            holes = torch.cat(all_holes, dim=0)
            pubs = torch.cat(all_pubs, dim=0)
            history = torch.cat(all_history, dim=0)
            new = torch.cat(all_new, dim=0)
        else:
            if self.ctr == 0:
                torch.set_num_threads(1)
                self.cfr.parallel_run()
                torch.set_num_threads(6)

                samples, new = zip(*self.cfr.labels)
                holes, pubs, history_ = zip(*samples)
                holes = torch.from_numpy(np.array(holes).astype(np.int64))
                pubs = torch.from_numpy(np.array(pubs).astype(np.int64))
                history = torch.from_numpy(np.array(history_).astype(np.float32))
                new = torch.from_numpy(np.array(new).astype(np.float32))
                self.holes = holes
                self.pubs = pubs
                self.history = history
                self.new = new
                self.ctr += 1
            elif self.ctr <= 5:
                idx = torch.randperm(len(self.holes))
                self.holes = self.holes[idx]
                self.pubs = self.pubs[idx]
                self.history = self.history[idx]
                self.new = self.new[idx]
                self.ctr += 1
                if self.ctr == 6: self.ctr = 0

        return self.holes, self.pubs, self.history, self.new
    
    def __len__(self):
        return 10

class Equity_DATASET(Data.Dataset):
    def __init__(self, path):
        self.cards = []
        self.probs = []
        for i in range(1, 10):
            samples = pkl.load(open(path + "_{}.pkl".format(i), "rb"))[0]
            cards, prob = zip(*samples)
            self.cards.append(cards)
            self.probs.append(prob)
        self.cards = torch.from_numpy(np.array(self.cards)).view(-1, 7)
        self.probs = torch.from_numpy(np.array(self.probs)).view(-1, 6)
        print(self.cards.shape)
        print(self.probs.shape)
    
    def __getitem__(self, idx):
        st = idx*30000
        ed = st + 30000
        return self.cards[st:ed], self.probs[st:ed]
    
    def __len__(self):
        return 90