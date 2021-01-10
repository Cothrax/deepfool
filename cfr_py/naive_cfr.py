from params import *
from game import *
from copy import deepcopy, copy
from queue import Queue
from time import time
import numpy as np
from cfr import CFR
from pure_cfr import PureCFR
from multiprocessing import Pool, Process
from itertools import chain
import pickle


def encode(x, y):
    # ((x[0] * y[1]) + x[1]) * y[2] + x[3]
    ret = x[0]
    for xi, yi in zip(x[1:], y[1:]):
        ret = ret * yi + xi
    return ret


class TabularAdvisor:
    """
    Tabular model for NaiveCFR (support parallel)
    """
    class Node:
        __slots__ = ['regret', 'strategy', 'ref']

        def __init__(self):
            self.regret = np.zeros(NUM_ACTION)
            self.strategy = np.ones(NUM_ACTION) / NUM_ACTION
            self.ref = 0

        def get_strategy(self):
            return self.strategy.copy()

        def update_strategy(self, regret):
            self.ref += 1
            self.regret += regret

            regret_plus = np.max(np.vstack([self.regret, np.zeros(NUM_ACTION)]), axis=0)
            tot = np.sum(regret_plus)
            if abs(tot) > 1e-6:
                self.strategy = regret_plus / tot
            else:
                self.strategy = np.ones(NUM_ACTION) / NUM_ACTION

            # print('r+', regret_plus, '\nstr', self.strategy)

    def __init__(self):
        self.node_map = {}

    def ask(self, sample):
        if sample not in self.node_map:
            return np.ones(NUM_ACTION) / NUM_ACTION
        return self.node_map[sample].get_strategy()

    def answer(self, sample, regret):
        if sample not in self.node_map:
            self.node_map[sample] = self.Node()
        self.node_map[sample].update_strategy(regret)


class NaiveCFR(PureCFR):
    """
    Tabular CFR
    """
    def __init__(self, advisor=None):
        if advisor is None:
            advisor = TabularAdvisor()
        super().__init__(advisor)

    def convert_sample(self, game):
        x = (game.step,
             int(game.win[0][game.player] * 10),
             int(game.win[1][game.player] * 10) if game.step >= 1 else 0,
             int(game.win[2][game.player] * 10) if game.step >= 2 else 0,
             int(game.win[3][game.player] * 10) if game.step >= 3 else 0,
             int(np.sum(game.if_raise[game.step]) - game.if_raise[game.step][game.player]),
             int(np.sum(game.if_call[game.step]) - game.if_call[game.step][game.player]),
             min(int((game.cur_bet - game.bets[game.step][game.player]) / game.chips[game.player] * 5), 5) if game.chips[game.player] else 5,
             min(int((game.pot + np.sum(game.bets)) / game.chips[game.player]), 5) if game.chips[game.player] else 5
             )
        y = (4, 11, 11, 11, 11, 6, 6, 6, 6)
        return encode(x, y)

    def update_strategy(self, node, regret, strategy):
        # self.advisor.answer(node, regret)
        self.labels.append((node, regret))

    def train(self, max_iter):
        self.cnt = 0
        start = time()
        for i in range(max_iter):
            player = 5
            game = Game(start=random.randint(0, NUM_PLAYER - 1))
            # start = time()
            self.cfr(game, player)
            # print(time() - start)

            if i % 5 == 0 and i:
                print(i, '/', max_iter, 'cfr visits', self.cnt, 'cum time', time()-start)
                self.cnt = 0
            if i % DUMP_ITER == 0:
                pickle.dump(self.advisor, open('ncfr%s.dat' % i, 'wb'))
                print('dump successful')

    def cfr_worker(self, args):
        max_iter = 20
        self.labels = []
        game, player = args
        for i in range(max_iter):
            self.cnt = 0
            self.cfr(game, player)
            # print(i, '/', max_iter, 'cfr visit', self.cnt)
        return self.labels

    def parallel_cfr(self, n_cfr):
        games = [Game(random.randint(0, NUM_PLAYER-1)) for _ in range(n_cfr)]
        players = [5] * n_cfr
        with Pool(N_CPU) as p:
            labels_list = p.map(self.cfr_worker, zip(games, players))
            for sample, regret in chain(*labels_list):
                self.advisor.answer(sample, regret)

    def parallel_train(self, n_cfr, max_iter):
        self.cnt = 0
        start = time()
        for i in range(max_iter):
            player = 5
            # start = time()
            self.parallel_cfr(n_cfr)
            # print(time() - start)
            print(i, '/', max_iter, 'cfr visits', self.cnt, 'time', time()-start)

            if i % DUMP_ITER == 0:
                pickle.dump(self.advisor, open('pncfr%d.dat' % i, 'wb'))
                print('dump successful')

        pickle.dump(self.advisor, open('pncfr.dat', 'wb'))


def test_ncfr():
    # advisor = pickle.load(open('3pncfr80.dat', 'rb'))
    cfr = NaiveCFR(advisor=None)
    cfr.parallel_train(6, 10)
    pass


if __name__ == '__main__':
    test_ncfr()
