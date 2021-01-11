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
        self.cards0 = []
        self.cards1 = []
        self.history = []
        self.probs = []
        self.files = glob.glob(path + "*.pkl")

        for f in self.files:
            samples = pkl.load(open(f, "rb"))
            for s in samples:
                self.cards0.append(s[0][0])
                self.cards1.append(s[0][1])
                self.history.append(s[0][2])
                self.probs.append(s[1])
        self.cards0 = torch.LongTensor(self.cards0)
        self.cards1 = torch.LongTensor(self.cards1)
        self.history = torch.FloatTensor(self.history)
        self.probs = torch.FloatTensor(self.probs)
        print(self.cards0.shape)
        print(self.cards1.shape)
        print(self.history.shape)
        print(self.probs.shape)
    
    def __getitem__(self, idx):
        return self.cards0[idx], self.cards1[idx], self.history[idx], self.probs[idx]
    
    def __len__(self):
        return self.cards0.shape[0]