from .params import *
from .game import *
from copy import deepcopy, copy
from queue import Queue
from time import time
import numpy as np
from .cfr import CFR
from multiprocessing import Pool, Process


class RandomAdvisor:
    def __init__(self):
        pass

    def ask(self, sample):
        ret = np.random.random(size=NUM_ACTION)
        ret /= np.sum(ret)
        return ret


class PureCFR(CFR):
    def __init__(self, advisor):
        super().__init__()
        self.advisor = advisor

    def get_strategy(self, game):
        sample = self.convert_sample(game)
        strategy = self.advisor.ask(sample)
        return sample, strategy

    def update_strategy(self, node, regret):
        self.samples.append(node)
        self.regrets.append(regret)

    def cfr(self, game: Game, player):
        self.cnt += 1
        if game.change_state() == GAME_OVER:
            return game.payoff()

        sample, strategy = self.get_strategy(game)
        if game.player != player:
            a = self.get_sample_action(game, sample_prob=strategy)
            next_game = deepcopy(game)
            next_game.act(a)
            util = self.cfr(next_game, player)
            return util
        else:
            legal_action = range(NUM_ACTION)
            if not game.is_raise_allowed():
                strategy[NUM_NOT_RAISE:] = 0
                strategy /= np.sum(strategy)
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
                regret[NUM_NOT_RAISE:] = -float('inf')

            self.update_strategy(sample, regret)
            self.strategies.append(strategy)
            return util

    def run(self, max_iter):
        self.strategies = []
        self.regrets = []
        self.samples = []
        self.labels = []

        for i in range(max_iter):
            player = random.randint(0, NUM_PLAYER - 1)
            game = Game()
            self.cnt = 0
            self.cfr(game, player)
            print(i, '/', max_iter, ': cfr visits', self.cnt)
        self.generate_learning_samples()

        return self


class ParallelPureCFR:
    def __init__(self, num_cfr, max_iter, adv_list):
        self.num_cfr = num_cfr
        self.max_iter = max_iter
        self.cfr_list = [PureCFR(adv_list[i]) for i in range(num_cfr)]

    def worker(self, cfr: PureCFR):
        return cfr.run(self.max_iter)

    def parallel_run(self):
        with Pool(N_CPU) as p:
            self.cfr_list = p.map(self.worker, self.cfr_list)


def test_pure_cfr():

    start = time()
    advisor = RandomAdvisor()
    cfr = PureCFR(advisor)
    cfr.run(10 * 8)
    print(time() - start)
    pass


def test_parallel_pure_cfr():
    start = time()
    adv_list = [RandomAdvisor() for i in range(8)]
    cfr = ParallelPureCFR(8, 10, adv_list)
    cfr.parallel_run()
    print(time() - start)


if __name__ == '__main__':
    test_pure_cfr()
