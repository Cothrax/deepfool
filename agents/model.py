import math
import torch
from torch import nn
import numpy as np

class CardEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rank = nn.Embedding(13, dim)
        self.suit = nn.Embedding(4, dim)
        self.card = nn.Embedding(52, dim)
    
    def forward(self, X):
        batch_size, num_cards = X.shape
        Y = X.view(-1)

        valid = Y > 0
        Y = Y.clamp(min=0)
        embs = self.card(Y) + self.rank(Y // 4) + self.suit(Y % 4)
        #print(embs.shape)
        valid = valid.view(-1, 1)
        embs = embs * valid

        return embs.view(batch_size, num_cards, -1).sum(dim=1)

class Card(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embed1 = CardEmbedding(dim)
        self.embed2 = CardEmbedding(dim)
        self.fc = nn.Sequential(
            nn.Linear(dim*2, 192),
            nn.ReLU(True),
            nn.Linear(192, 192),
            nn.ReLU(True),
            nn.Linear(192, dim),
            nn.ReLU(True)
        )
    def forward(self, card1, card2):
        f1 = self.embed1(card1)
        f2 = self.embed2(card2)
        f3 = torch.cat([f1, f2], dim=1)
        f4 = self.fc(f3)

        return f4


class History_Act(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden_channels, 1, batch_first=True, bidirectional=True, dropout=0)#TODO dropout 0.2
        self.fc = nn.Linear(2*hidden_channels, out_channels)
        self.relu = nn.ReLU(True)
    
    def forward(self, X):
        self.gru.flatten_parameters()
        out, _ = self.gru(X) # use output instead of hidden state
        out = self.fc(out)
        out = self.relu(out).mean(dim=1)

        return out

class DF(nn.Module):
    def __init__(self, history_size, num_action, dim=64):
        super().__init__()
        self.card = Card(dim)

        #self.hist_rnn = History_Act(history_size, dim, dim)
        self.hist_fc = nn.Sequential(
            nn.Linear(4*18, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
        )

        self.post_process = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, num_action),
            nn.Softmax(dim=1)
        )

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, card1, card2, history):# history = [history of action and pot]
        #print(history[:10])
        #input("check")
        #print("history size: {}".format(history.shape))

        # card1 of shape(B, 2)
        # card2 of shape(B, 5)
        # history of shape (B, 4, 18)

        f1 = self.card(card1, card2)
        #f2 = self.hist_rnn(history)
        f2 = self.hist_fc(history.view(card1.shape[0], -1))
        f3 = torch.cat([f1, f2], dim=1)
        f4 = self.post_process(self.dropout(f3))

        return f4
    
    def ask(self, sample):
        # sample = [holes, pubs, history]
        holes, pubs, history = sample
        holes = torch.from_numpy(holes.astype(np.int64)).unsqueeze(dim=0)
        pubs = torch.from_numpy(pubs.astype(np.int64)).unsqueeze(dim=0)
        history = torch.from_numpy(history.astype(np.float32)).unsqueeze(dim=0)

        prob = self.forward(holes, pubs, history).squeeze().numpy()

        return prob

class PreTrain(nn.Module):
    def __init__(self, num_action, dim=64):
        super().__init__()
        self.card = Card(dim)

        self.post_process = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, num_action),
            nn.Softmax(dim=1)
        )

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, card1, card2):
        # card1 of shape(B, 2)
        # card2 of shape(B, 5)

        f1 = self.card(card1, card2)
        f4 = self.post_process(self.dropout(f1))

        return f4