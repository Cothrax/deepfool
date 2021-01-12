>"""Random player"""
import torch
import numpy as np
from .model import *
import random

from gym_env.env import Action

autplay = True  # play automatically if played against keras-rl

NUM_PLAYER = 6
class Player:
    """Mandatory class with the player methods"""

    def __init__(self, env, name='CFR'):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True
        self.env = env
        self.model = DF(18, 6).cpu()
        self.load_state_dict(torch.load("agents/checkpoint.pt", map_location=torch.device('cpu')))
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.suit = {"H":0, "D":1, "S":2, "C":3}
        self.rank = {"A":12, "T":8, "J":9, "Q":10, "K":11}
        for i in range(2, 10):
            self.rank[str(i)] = i - 2

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        card1 = self.env.players[info['player_data']['position']].cards[:]
        card1 = [self.rank[x[0]] + 13 * self.suit[x[1]] for x in card1]
        card2 = [-1] * 5
        length = len(self.env.table_cards)
        if length > 0:
            table_cards = self.env.table_cards[:length]
            table_cards = [self.rank[x[0]] + 13 * self.suit[x[1]] for x in table_cards]
            card2[:length] = table_cards
        card1 = np.array(card1)
        card2 = np.array(card2)

        order = (np.arange(6) - 5 + info["player_data"]["position"]) % 6
        if_call = np.array([info["stage_data"][i]["calls"] for i in range(4)]).astype(np.float32)
        if_raise = np.array([info["stage_data"][i]["raises"] for i in range(4)]).astype(np.float32)
        bets = np.array([info["stage_data"][i]["contribution"] for i in range(4)]).astype(np.float32)

        step = 0 if length == 0 else length - 2
        for i in range(NUM_PLAYER):
            if bets[step][i] > 0.01 and if_call[step][i] < 1e-3 and if_raise[step][i]:
                if_raise[step][i] = 1
        history = np.concatenate([if_call[:,order], if_raise[:, order], bets[:, order]], axis=1)


        card1 = torch.from_numpy(card1).unsqueeze(0)
        card2 = torch.from_numpy(card2).unsqueeze(0)
        history = torch.from_numpy(history).unsqueeze(0)

        predict = self.model(card1, card2, history).squeeze().numpy()
        print("----------------predcit-------------------")
        print(predict)

        if len(set(action_space)) == 1:
            return action_space[0]
        else:
            all_action_space = [Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_3BB, Action.RAISE_POT, Action.RAISE_2POT, Action.ALL_IN]
            mask = np.zeros(7)
            for a in action_space:
                mask[all_action_space.index(a)] = 1
            prob = predict[[0,1,1,2,3,4,5]]
            prob = prob * mask
            action = np.argmax(prob)
            if action == 6:
                prob = prob / prob.sum()
                action = np.random.choice(7, 1, p=prob)
        return action
