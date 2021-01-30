from .params import *
from .game import *
from copy import deepcopy, copy
from queue import Queue
from time import time
import numpy as np
from .cfr import CFR
from .pure_cfr import PureCFR
from multiprocessing import Pool, Process
from itertools import chain
import pickle

STAIRS = np.array([0, 1/64, 1/32, 1/16, 1/8, 1/4])

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

    def mask_ref(self):
        for k in self.node_map:
            self.node_map[k].ref = -0x3f3f3f3f

    def ask(self, sample, training=True):
        if sample not in self.node_map:
            return np.ones(NUM_ACTION) / NUM_ACTION if training else None
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
        # x = (game.step,
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

        delta = game.cur_bet - game.bets[game.step][game.player]
        chip = game.chips[game.player]
        cost = INIT_CHIPS - chip
        real_pot = game.pot + np.sum(game.bets)

        # cost / INIT_CHIPS: [0, 1/64, 1/32, 1/16, 1/8, 1/4]
        # real_pot / chip  : [0, 1/64, 1/32, 1/16, 1/8, 1/4]
        # delta  / chip  : [0, 1/64, 1/32, 1/16, 1/8, 1/4]
        x = (game.step,
             int(game.win[0][game.player] * 50),
             int(game.win[1][game.player] * 10) if game.step >= 1 else 0,
             int(game.win[2][game.player] * 10) if game.step >= 2 else 0,
             int(game.win[3][game.player] * 5) if game.step >= 3 else 0,
             int(game.num) - 2,
             STAIRS.searchsorted(cost / INIT_CHIPS),
             STAIRS.searchsorted(real_pot / chip),
             STAIRS.searchsorted(delta / chip)
             )
        if x[1] > 28:
            input('check')
        y = (4, 50, 10, 10, 5, 5, 7, 7, 7)
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
        max_iter = 1000
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
            self.parallel_cfr(n_cfr)
            # print(i, '/', max_iter, 'cfr visits', self.cnt, 'time', time()-start)
            if i % 1 == 0:
                print(i, '/', max_iter, 'cfr visits', self.cnt, 'time', time()-start)
                start = time()

            if i % DUMP_ITER == 0:
                pickle.dump(self.advisor, open('bncfr%d.dat' % i, 'wb'))
                print('dump successful')

        pickle.dump(self.advisor, open('bncfr.dat', 'wb'))


class BiasedNaiveCFR(NaiveCFR):
    def __init__(self, advisor, given_prob):
        super(BiasedNaiveCFR, self).__init__(advisor)
        self.given_prob = given_prob
        self.biased_player = None

    def cfr(self, game: Game, player):
        self.cnt += 1
        if game.change_state() == GAME_OVER:
            return game.payoff()

        sample, strategy = self.get_strategy(game)
        if game.player != player:
            prob = self.given_prob if game.player == self.biased_player else strategy
            # print(game.player in self.biased_players, prob)
            a = self.get_sample_action(game, sample_prob=prob)
            next_game = deepcopy(game)
            next_game.act(a)
            util = self.cfr(next_game, player)
            return util
        else:
            legal_action = range(NUM_ACTION)
            if not game.is_raise_allowed():
                strategy[NUM_NOT_RAISE:] = 0
                tot = np.sum(strategy)
                if tot:
                    strategy /= tot
                else:
                    strategy[:NUM_NOT_RAISE] = 1 / NUM_NOT_RAISE
                legal_action = range(NUM_NOT_RAISE)

            util = np.zeros(NUM_PLAYER)
            cv_util = np.zeros(NUM_ACTION)
            for a in legal_action:
                next_game = deepcopy(game)
                next_game.act(a)
                next_util = self.cfr(next_game, player)
                cv_util[a] = next_util[player]
                util += strategy[a] * next_util

            regret = cv_util - util[player]

            if not game.is_raise_allowed():
                regret[NUM_NOT_RAISE:] = 0

            self.update_strategy(sample, regret, strategy)
            # self.strategies.append(strategy)
            return util

    def parallel_cfr(self, n_cfr):
        self.biased_player = np.random.randint(0, NUM_PLAYER-1)
        print('biased player at', self.biased_player)
        super(BiasedNaiveCFR, self).parallel_cfr(n_cfr)


class SampleGenerator:
    """
    use NaiveCFR to generate learning samples
    """
    def __init__(self, advisor):
        self.advisor = advisor
        self.naive_cfr = NaiveCFR()
        self.pure_cfr = PureCFR(None)

    def play(self, max_iter, given_prob=None):
        labels = []
        cnt = 0
        none_cnt = 0
        for t in range(max_iter):
            game = Game(start=random.randint(0, NUM_PLAYER-1))
            while game.change_state() != GAME_OVER:
                cnt += 1
                node = self.naive_cfr.convert_sample(game)
                strategy = self.advisor.ask(node, training=False)
                if strategy is None:
                    none_cnt += 1
                    if game.is_raise_allowed():
                        a = np.random.choice(range(NUM_ACTION), 1)[0]
                    else:
                        a = np.random.choice(range(NUM_NOT_RAISE), 1)[0]
                else:
                    sample = self.pure_cfr.convert_sample(game)
                    labels.append([sample, strategy])

                    if game.is_raise_allowed():
                        tot = np.sum(strategy[:NUM_NOT_RAISE])
                        if abs(tot) > 1e-3:
                            pass

                    prob = strategy if given_prob is None else given_prob
                    a = self.naive_cfr.get_sample_action(game, sample_prob=prob)
                game.act(a)
        print('strategy' if given_prob is None else 'random', ':invalid node:', none_cnt, '/', cnt)
        return labels

    def worker(self, args):
        if type(args) == int:
            return self.play(args)
        else:
            max_iter, given_prob = args
            return self.play(max_iter, given_prob)

    def generate(self, n_play, max_iter):
        """
        will generate n_play * max_iter random walk samples and
        n_play * max_iter strategy walk samples
        return one flattened list
        """
        start = time()
        with Pool(N_CPU) as p:
            labels1 = p.map(self.worker, [max_iter] * n_play)
            labels2 = p.map(self.worker, [(max_iter, init_prob)] * n_play)

            # TODO or you want to save to files?
            # NOTE that chain gives a generator, you may need list(chain(*labels1, *labels2)) to get a list
            print('COST', time() - start, 'sec')
            return chain(*labels1, *labels2)


def test_ncfr():
    # advisor = pickle.load(open('3pncfr80.dat', 'rb'))
    cfr = NaiveCFR(advisor=None)
    cfr.parallel_train(3, 1)
    pass


def test_generator(filename):
    adv = pickle.load(open(filename, 'rb'))
    gen = SampleGenerator(advisor=adv)

    labels = list(gen.generate(6, 20))
    print('generated:', len(labels))
    return labels


def test_randomized_ncfr():
    advisor = pickle.load(open('cfr_py/pncfr.dat', 'rb'))
    advisor.mask_ref()
    cfr = BiasedNaiveCFR(advisor=advisor, given_prob=np.array([0, 0, 0, 0, 0, 1]))
    cfr.parallel_train(6, 10)

    cnt = 0
    for k, v in advisor.node_map.items():
        if v.ref > 0:
            print(k, v.ref, v.regret, v.strategy)
        cnt += 1
    print(cnt)

    pass
