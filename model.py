import math
import torch
from torch import nn

class CardEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rank = nn.Embedding(13, dim)
        self.suit = nn.Embedding(4, dim)
        self.card = nn.Embedding(52, dim)
    
    def forward(self, X):
        batch_size, num_cards = X.shape
        Y = X.view(-1)

        valid = Y.ge(0).float()
        Y = Y.clamp(min=0)
        embs = self.card(Y) + self.rank(x // 4) + self.suit(x % 4)
        embs = embs * valid

        return embs.view(batch_size, num_cards, -1).sum(1)

class History_Act(nn.Moudle):
    def __init__(self, in_channels, hidden_channels, out_channels):
        self.gru = nn.GRU(in_channels, hidden_channels, 1, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(2*hidden_channels, out_channels)
        self.relu = nn.ReLU(True)
    
    def forward(self, X):
        self.gru.flatten_parameters()
        out, _ = self.gru(X) # use output instead of hidden state
        out = self.fc(out)
        out = self.relu(out)

        return out

class DF(nn.Module):
    def __init__(self, num_player, num_action, dim=256):
        super().__init__()
        self.card = nn.Sequential(
            CardEmbedding(dim)
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True)
        )

        self.hist_rnn = History_Act(num_player, dim, dim)

        self.post_process = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, num_action),
            nn.ReLU(True),
            nn.Linear(dim, num_action),
            nn.Softmax()
        )

    def forward(self, card, history):# history = [history of action and pot]
        f1 = self.card(card)
        f2 = self.hist_rnn(history)
        f3 = torch.cat([f1, f2], dim=1)
        f4 = self.post_process(f3)

        return f4