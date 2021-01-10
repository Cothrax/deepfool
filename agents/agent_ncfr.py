"""Random player"""
import numpy as np
import random
import pickle

from gym_env.env import Action
from cfr_py.calculator import MYCFR
from cfr_py.naive_cfr import *

autplay = True  # play automatically if played against keras-rl

 # def convert_sample(self, game):
 #     x = (game.step,
 #          int(game.win[0][game.player] * 10),
 #          int(game.win[1][game.player] * 10) if game.step >= 1 else 0,
 #          int(game.win[2][game.player] * 10) if game.step >= 2 else 0,
 #          int(game.win[3][game.player] * 10) if game.step >= 3 else 0,
 #          int(np.sum(game.if_raise[game.step]) - game.if_raise[game.step][game.player]),
 #          int(np.sum(game.if_call[game.step]) - game.if_call[game.step][game.player]),
 #          min(int((game.cur_bet - game.bets[game.step][game.player]) / game.chips[game.player] * 5), 5) if game.chips[game.player] else 5,
 #          min(int((game.pot + np.sum(game.bets)) / game.chips[game.player]), 5) if game.chips[game.player] else 5
 #     )
 #     y = (4, 11, 11, 11, 11, 6, 6, 6, 6)
 #     return encode(x, y)

#
# def encode(x, y):
#     # ((x[0] * y[1]) + x[1]) * y[2] + x[3]
#     ret = x[0]
#     for xi, yi in zip(x[1:], y[1:]):
#         ret = ret * yi + xi
#     return ret
#
# NUM_ACTION = 6
# NUM_PLAYER = 6

#
# class TabularAdvisor:
#     class Node:
#         __slots__ = ['regret', 'strategy', 'ref']
#
#         def __init__(self):
#             self.regret = np.zeros(NUM_ACTION)
#             self.strategy = np.ones(NUM_ACTION) / NUM_ACTION
#             self.ref = 0
#
#         def get_strategy(self):
#             return self.strategy
#
#         def update_strategy(self, regret):
#             self.ref += 1
#             self.regret += regret
#
#             regret_plus = np.max(np.vstack([self.regret, np.zeros(NUM_ACTION)]), axis=0)
#             tot = np.sum(regret_plus)
#             if abs(tot) > 1e-6:
#                 self.strategy = regret_plus / tot
#             else:
#                 self.strategy = np.ones(NUM_ACTION) / NUM_ACTION
#
#             # print('r+', regret_plus, '\nstr', self.strategy)
#
#     def __init__(self):
#         self.node_map = {}
#
#     def ask(self, sample):
#         if sample not in self.node_map:
#             return None
#         return self.node_map[sample].get_strategy()
#
#     def answer(self, sample, regret):
#         if sample not in self.node_map:
#             self.node_map[sample] = self.Node()
#         self.node_map[sample].update_strategy(regret)
#
# def encode(x, y):
#     # ((x[0] * y[1]) + x[1]) * y[2] + x[3]
#     ret = x[0]
#     for xi, yi in zip(x[1:], y[1:]):
#         ret = ret * yi + xi
#     return ret


class Player:
    """Mandatory class with the player methods"""

    def __init__(self, env, filepath, name='CFR'):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True
        self.env = env
        self.suit = {"H":0, "D":1, "S":2, "C":3}
        self.rank = {"A":12, "T":8, "J":9, "Q":10, "K":11}
        self.calculator = MYCFR.Calculator()
        self.advisor = pickle.load(open(filepath, 'rb'))

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

        step = 0 if length == 0 else length - 2
        win0 = self.calculator.prior_win_rate(
            int(card1[0]),
            int(card1[1]),
            int(card2[0]),
            int(card2[1]),
            int(card2[2]),
            int(card2[3]),
            int(card2[4]),
            0
        )
        win1 = self.calculator.prior_win_rate(
            int(card1[0]),
            int(card1[1]),
            int(card2[0]),
            int(card2[1]),
            int(card2[2]),
            int(card2[3]),
            int(card2[4]),
            1
        ) if length > 0 else 0

        win2 = self.calculator.prior_win_rate(
            int(card1[0]),
            int(card1[1]),
            int(card2[0]),
            int(card2[1]),
            int(card2[2]),
            int(card2[3]),
            int(card2[4]),
            2,
        ) if length >= 4 else 0

        win3 = self.calculator.prior_win_rate(
            int(card1[0]),
            int(card1[1]),
            int(card2[0]),
            int(card2[1]),
            int(card2[2]),
            int(card2[3]),
            int(card2[4]),
            3,
        ) if length >= 5 else 0

        if_call = np.array([info["stage_data"][i]["calls"] for i in range(4)]).astype(np.float32)
        if_raise = np.array([info["stage_data"][i]["raises"] for i in range(4)]).astype(np.float32)
        bets = np.array([info["stage_data"][i]["contribution"] for i in range(4)]).astype(np.float32)

        me = 5
        raises = np.sum(if_raise[step]) - if_raise[step][me]
        calls = np.sum(if_call[step]) - if_call[step][me]
        #  x = (game.step,
        #      int(game.win[0][game.player] * 10),
        #      int(game.win[1][game.player] * 10) if game.step >= 1 else 0,
        #      int(game.win[2][game.player] * 10) if game.step >= 2 else 0,
        #      int(game.win[3][game.player] * 10) if game.step >= 3 else 0,
        #      int(np.sum(game.if_raise[game.step]) - game.if_raise[game.step][game.player]),
        #      int(np.sum(game.if_call[game.step]) - game.if_call[game.step][game.player]),
        #      min(int((game.cur_bet - game.bets[game.step][game.player]) / game.chips[game.player] * 5), 5) if game.chips[game.player] else 5,
        #      min(int((game.pot + np.sum(game.bets)) / game.chips[game.player]), 5) if game.chips[game.player] else 5
        #      )
        # y = (4, 11, 11, 11, 11, 6, 6, 6, 6)

        cur_bet = np.max(bets[step])
        my_bet = bets[step][me]
        my_chips = info["player_data"]['stack'][me]
        pot = np.sum(bets)

        bet_r = min(int((cur_bet - my_bet) / my_chips * 5), 5) if my_chips > 1e-3 else 5
        pot_r = (min(int(pot / my_chips), 5)) if my_chips > 1e-3 else 5
        x = (step,
             int(win0 * 10),
             int(win1 * 10),
             int(win2 * 10),
             int(win3 * 10),
             int(raises),
             int(calls),
             bet_r,
             pot_r)
        y = (4, 11, 11, 11, 11, 6, 6, 6, 6)

        node = encode(x, y)
        predict = self.advisor.ask(node)

        print(x)
        print(y)
        print(predict)

        action = Action.FOLD
        if predict is None:
            if Action.FOLD in set(action_space):
                action = Action.FOLD
            elif Action.CHECK in set(action_space):
                action = Action.CHECK
            elif Action.CALL in set(action_space):
                action = Action.CALL
            return action


        # input('check')
        # '''
        # order = (np.arange(6) - 5 + info["player_data"]["position"]) % 6
        # if_call = np.array([info["stage_data"][i]["calls"] for i in range(4)]).astype(np.float32)
        # if_raise = np.array([info["stage_data"][i]["raises"] for i in range(4)]).astype(np.float32)
        # bets = np.array([info["stage_data"][i]["contribution"] for i in range(4)]).astype(np.float32)
        # history = np.concatenate([if_call[:,order], if_raise[:, order], bets[:, order]], axis=1)
        # '''

        # card1 = torch.from_numpy(card1).unsqueeze(0)
        # card2 = torch.from_numpy(card2).unsqueeze(0)
        # #history = torch.from_numpy(history).unsqueeze(0)

        # #predict = self.model(card1, card2, history).squeeze()
        # predict = self.model(card1, card2).squeeze().numpy()
        # print("-------------- predict ----------------------\n", predict)


        # if len(set(action_space)) == 1:
        #     return action_space[0]
        # else:
        #     all_action_space = [Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_3BB, Action.RAISE_POT, Action.RAISE_2POT, Action.ALL_IN]
        #     mask = np.zeros(7)
        #     for a in action_space:
        #         mask[all_action_space.index(a)] = 1
        #     prob = predict[[0,1,1,2,3,4,5]]
        #     prob = prob * mask
        #     action = np.argmax(prob)

        if len(set(action_space)) == 1:
            return action_space[0]
        elif(len(set(action_space))) > 2:
            predict = predict
            action = np.random.choice(len(predict), 1, p=predict)[0]
            # action = np.argmax(predict)
        else:
            action = 0 if predict[0] > predict[1] else 1
        if action >= 2:
            action += 1
        elif action == 1:
            if Action.CHECK not in set(action_space):
                action = 2
        elif action == 0 and Action.FOLD not in set(action_space):
            action = 1

        if action == Action.ALL_IN:
            input('check')

        '''
        if len(set(action_space)) == 1:
            return action_space[0]
        elif(len(set(action_space))) > 2:
            action = torch.argmax(predict).item()
        else:
            action = 0 if predict[0] > predict[1] else 1
        if action == 2:
            action += 1
        elif action == 1:
            if Action.CHECK not in set(action_space):
                action = 2
        '''

        return action
