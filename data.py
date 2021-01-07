#DATA

import os
import glob
import torch
import pickle
import random
import numpy as np
import torch.utils.data as Data

class POKER_DATASET(object):
    def __init__(self, model):
        self.model = model
        self.cfr = CFR()# TODO import from cpp
    
    def __next__():
        while not self.cfr.ready:
            ans = self.model(self.cfr.query)
            self.cfr.search(ans)
        feature, label = self.cfr.sample

        return feature, label