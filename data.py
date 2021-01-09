#DATA

import os
import glob
import torch
import pickle
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
        else:
            self.model_list = model
            self.cfr = ParallelPureCFR(len(self.model_list), max_iter, self.model_list)
    
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

        return holes, pubs, history, new
    
    def __len__(self):
        return 10