from .game import *
from copy import deepcopy, copy
from collections import deque
import numpy as np
from time import time


class CFR:
    """
    The earliest version of our CFR Algo (using direct sampling)
    """
    # __slots__ = ['sampling_choices', 'games', 'strategies',
    #             'samples', 'regrets', 'run_ptr', 'T', 'labels']

    def __init__(self):
        self.T = 0
        self.sampling_choices = deque()
        self.games = deque()

        self.run_ptr = 0
        self.strategies = None
        self.regrets = []
        self.samples = []
        self.labels = []
        self.cnt = 0
        self.max_dep = 0

    def convert_sample(self, game):
        holes = game.holes[game.player]
        pubs = np.ones(5) * -1
        if game.step:
            num_card = game.step + 2
            pubs[:num_card] = game.pubs[:num_card]
        if_call = game.if_call.copy().astype(np.float32)
        if_raise = game.if_raise.copy().astype(np.float32)
        bets = game.bets.copy().astype(np.float32) / (100 * BIG_BLIND)
        history = np.concatenate([if_call, if_raise, bets], axis=1)
        return [holes, pubs, history]

    def submit(self, game):
        sample = self.convert_sample(game)
        self.samples.append(sample)
        # self.samples = np.append(self.samples, sample, axis=0)
  
    def generate_learning_samples(self):
        self.labels = []
        mapping = {}
        repeat_num = 0
        for sample, regret in zip(self.samples, self.regrets):
            sample_tup = (tuple(sample[0]), tuple(sample[1]), tuple(sample[2].reshape(-1)))
            if sample_tup not in mapping:
                mapping[sample_tup] = np.zeros(NUM_ACTION)
            else:
                repeat_num += 1
            mapping[sample_tup] += regret

        print('repeat num:', repeat_num, '/', len(self.samples))

        for sample, old in zip(self.samples, self.strategies):
            sample_tup = (tuple(sample[0]), tuple(sample[1]), tuple(sample[2].reshape(-1)))
            regret = mapping[sample_tup]
            if regret is not None:
                regret_plus = np.max(np.vstack([regret, np.zeros(NUM_ACTION)]), axis=0)
                tot = np.sum(regret_plus)
                if abs(tot) > 1e-3:
                    k = self.T // FOLD_NUM
                    new = (regret_plus / tot + old * k) / (k + 1)
                    label = [sample, new]
                    self.labels.append(label)
                    # print(sample)
                    # print(new)

                    # self.labels = np.append(self.labels, label, axis=0)
                mapping[sample_tup] = None

        print('-------------------- learning samples --------------------')
        output = random.choices(self.labels, k=1)
        for e in output:
            print('X:', e[0])
            print('y:', e[1])
        print('-------------------- learning samples --------------------')

        pass
        self.T += 1

    def get_strategy(self, game):
        ret = self.run_ptr
        self.run_ptr += 1
        return ret, self.strategies[ret]

    def update_strategy(self, node, regret):
        self.regrets[node] = regret

    def get_sample_action(self, game, sample_prob=init_prob):
        if game.is_raise_allowed():
            a = np.random.choice(range(NUM_ACTION), 1, p=sample_prob)
        else:
            prob = sample_prob[:NUM_NOT_RAISE]
            tot = np.sum(prob)
            if abs(tot) < 1e-3:
                prob = np.ones(NUM_NOT_RAISE) / NUM_NOT_RAISE
            else:
                prob /= tot

            a = np.random.choice(range(NUM_NOT_RAISE), 1, p=prob)

        return a


        # win = calculator.prior_win_rate(
        #     int(game.holes[game.player][0]),
        #     int(game.holes[game.player][1]),
        #     int(game.pubs[0]),
        #     int(game.pubs[1]),
        #     int(game.pubs[2]),
        #     int(game.pubs[3]),
        #     int(game.pubs[4]),
        #     game.step
        # )

        # if game.is_raise_allowed():
        #     return np.searchsorted(equity_prob, game.win[game.step][game.player])
        # else:
        #     return CHECK if game.win[game.step][game.player] > equity_prob[0] else FOLD

        # i = game.player
        # p = calculator.potential_power(
        #     int(game.holes[i][0]), int(game.holes[i][1]),
        #     int(game.pubs[0]), int(game.pubs[1]),
        #     int(game.pubs[2]), int(game.pubs[3]), int(game.pubs[4]),
        #     game.step
        # )
        # if game.is_raise_allowed():
        #     a = np.random.choice(range(NUM_ACTION), 1, p=sample_prob)
        # else:
        #     prob = sample_prob[:NUM_NOT_RAISE]/np.sum(sample_prob[:NUM_NOT_RAISE])
        #     a = np.random.choice(range(NUM_NOT_RAISE), 1, p=prob)

    def dfs(self, game: Game, player, dep):
        self.cnt += 1
        self.max_dep = max(self.max_dep, dep)
        # print(self.cnt, dep)
        if dep > 200:
            print('Warning: depth too large')

        if game.change_state() == GAME_OVER:
            return

        if game.player != player:
            a = self.get_sample_action(game)
            # print(a)
            # a = random.randint(0, NUM_ACTION-1)

            self.sampling_choices.append(a)
            next_game = deepcopy(game)
            next_game.act(a)
            self.dfs(next_game, player, dep+1)
        else:
            self.submit(game)
            legal_action = game.get_legal_action()
            for a in legal_action:
                next_game = deepcopy(game)
                next_game.act(a)
                self.dfs(next_game, player, dep+1)

    def cfr(self, game: Game, player):
        self.cnt += 1
        if game.change_state() == GAME_OVER:
            return game.payoff()

        if game.player != player:
            #     node, strategy = self.get_strategy(game, history)
            #     a = np.random.choice(range(NUM_ACTION), 1, p=strategy)[0]

            a = self.sampling_choices.popleft()
            next_game = deepcopy(game)
            next_game.act(a)

            util = self.cfr(next_game, player)
            return util
        else:
            node, strategy = self.get_strategy(game)
            legal_action = range(NUM_ACTION)
            if not game.is_raise_allowed():
                strategy[NUM_NOT_RAISE:] = 0
                tot = np.sum(strategy)
                if tot:
                    strategy /= tot
                else:
                    strategy[NUM_NOT_RAISE:] = 1/NUM_NOT_RAISE
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

            self.update_strategy(node, regret)
            return util

    def search(self, max_iter):
        self.samples = []
        self.labels = []
        begin = time()
        for i in range(max_iter):
            # start = random.randint(0, NUM_PLAYER - 1)
            player = random.randint(0, NUM_PLAYER - 1)
            game = Game()
            self.games.append((player, game))

            self.max_dep = 0
            self.cnt = 0
            self.dfs(game, player, 0)
            print(i, '/', max_iter, ': visited', self.cnt, ' max_depth', self.max_dep)

        print('COST %s sec' % (time() - begin))
        return self

    def run(self, max_iter):
        self.regrets = np.zeros(shape=(len(self.samples), NUM_ACTION))
        self.run_ptr = 0

        for i in range(max_iter):
            player, game = self.games.popleft()
            self.cnt = 0
            self.cfr(game, player)
            print(i, '/', max_iter, ': cfr visits', self.cnt)

        self.generate_learning_samples()
        return self
