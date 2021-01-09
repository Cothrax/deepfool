from params import *
from game import *
from copy import deepcopy, copy
from queue import Queue
from time import time
import numpy as np
from cfr import CFR
from multiprocessing import Pool, Process


class RandomAdvisor:
    def __init__(self):
        pass

    def ask(self, sample):
        ret = np.random.random(size=NUM_ACTION)
        ret /= np.sum(ret)
        return ret

    def batch_ask(self, sample):
        ret = np.random.random(size=(len(sample), NUM_ACTION))
        ret /= np.repeat(np.sum(ret, axis=1).reshape(-1, 1), NUM_ACTION, axis=1)
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
                regret[NUM_NOT_RAISE:] = 0

            self.update_strategy(sample, regret)
            self.strategies.append(strategy)
            return util

    def run(self, max_iter, if_generate=True):
        self.strategies = []
        self.regrets = []
        self.samples = []
        self.labels = []

        for i in range(max_iter):
            player = 5
            game = Game(start=random.randint(0, NUM_PLAYER - 1))
            self.cnt = 0
            self.cfr(game, player)
            print(i, '/', max_iter, ': cfr visits', self.cnt)
        if if_generate:
            self.generate_learning_samples()
            #print(i, '/', max_iter, ': cfr visits', self.cnt)
        self.generate_learning_samples()

        return self


class ParallelPureCFR:
    def __init__(self, num_cfr, max_iter, adv_list):
        self.T = 0
        self.num_cfr = num_cfr
        self.max_iter = max_iter
        self.cfr_list = [PureCFR(adv_list[i]) for i in range(num_cfr)]
        self.labels = []
        self.mapping = {}
        self.repeat_num = 0

    def worker(self, cfr: PureCFR):
        return cfr.run(self.max_iter, if_generate=False)

    def filter_regret(self, cfr):
        for sample, regret, strategy in zip(cfr.samples, cfr.regrets, cfr.strategies):
            sample_tup = (tuple(sample[0]), tuple(sample[1]), tuple(sample[2].reshape(-1)))
            if sample_tup not in self.mapping:
                self.mapping[sample_tup] = [np.zeros(NUM_ACTION), sample, strategy]
            else:
                self.repeat_num += 1
            self.mapping[sample_tup][0] += regret

    def generate_learning_samples(self):
        for _, v in self.mapping.items():
            regret, sample, old = v
            regret_plus = np.max(np.vstack([regret, np.zeros(NUM_ACTION)]), axis=0)
            tot = np.sum(regret_plus)
            if abs(tot) > 1e-3:
                k = self.T // FOLD_NUM
                new = (regret_plus / tot + old * k) / (k + 1)
                label = [sample, new]
                self.labels.append(label)

    def parallel_run(self):
        with Pool(N_CPU) as p:
            self.cfr_list = p.map(self.worker, self.cfr_list)

        self.labels = []
        self.mapping = {}
        self.repeat_num = 0
        for cfr in self.cfr_list:
            self.filter_regret(cfr)
        self.generate_learning_samples()
        print('repeat num:', self.repeat_num)


def test_pure_cfr():

    start = time()
    advisor = RandomAdvisor()
    cfr = PureCFR(advisor)
    cfr.run(5)
    print(time() - start)
    pass


def test_parallel_pure_cfr():
    start = time()
    adv_list = [RandomAdvisor() for i in range(8)]
    cfr = ParallelPureCFR(1, 10, adv_list)
    for i in range(3):
        cfr.parallel_run()
    print(time() - start)


if __name__ == '__main__':
    test_pure_cfr()
