#DATA

import os
import glob
import torch
import pickle
import random
import numpy as np
import torch.utils.data as Data
from cfr_py.cfr import CFR

class POKER_DATASET(object):
    def __init__(self, model, max_iter, max_history):
        self.max_iter = max_iter
        self.max_history = max_history
        self.model = model
        self.cfr = CFR()
    
    def __getitem__(self):
        self.cfr.search(self.max_iter)
        samples = self.cfr.samples

        holes, pubs, history_ = zip(*samples)
        holes = torch.from_numpy(np.array(holes).astype(np.int64))
        pubs = torch.from_numpy(np.array(pubs).astype(np.int64))
        '''
        history = []
        for h in history_:
            h = h.T
            if h.shape[0] >= self.max_history:
                h = h[:self.max_history]
            else:
                h = np.pad(h, ((0, self.max_history-h.shape(0)), (0,0)))
            history.append(h)
        history = np.array(history)
        '''
        history = np.array(history_).astype(np.float32)
        history = torch.from_numpy(history)
        #input("check samples of shape {}".format(history.shape))
        old_label = self.model(holes, pubs, history)
        #print(old_label[:10])

        self.cfr.strategies = old_label.numpy()
        self.cfr.run(self.max_iter)

        samples, new = zip(*self.cfr.labels)
        holes, pubs, history_ = zip(*samples)
        holes = torch.from_numpy(np.array(holes).astype(np.int64))
        pubs = torch.from_numpy(np.array(pubs).astype(np.int64))
        '''
        history = []
        for h in history_:
            h = h.T
            if h.shape(0) >= self.max_history:
                h = h[:self.max_history]
            else:
                h = np.pad(h, ((0, self.max_history-h.shape(0)), (0,0)))
            history.append(h)
        '''
        history = np.array(history_).astype(np.float32)
        history = torch.from_numpy(history)
        #input("check samples of shape {}".format(history.shape))
        new = torch.from_numpy(np.array(new))

        return holes, pubs, history, new

    
    def __len(self):
        return 10