from game import *
from copy import deepcopy, copy
from queue import Queue
import numpy as np


class CFR:
    __slots__ = ['sampling_choices', 'games', 'strategies',
                 'samples', 'regrets', 'run_ptr', 'T']

    def __init__(self):
        self.T = 0
        self.sampling_choices = Queue()
        self.games = Queue()

        self.run_ptr = 0
        self.strategies = None
        self.samples = None

    def submit(self, game):
        # TODO submit to samples
        sample = None
        self.samples = np.append(self.samples, [sample], axis=0)

    def generate_learning_samples(self):
        self.T += 1
        mapping = {}
        for sample, regret in zip(self.samples, self.regrets):
            if sample not in mapping:
                mapping[sample] = np.zeros(NUM_ACTION)
            else:
                mapping[sample] += self.regrets

        for sample, old in zip(self.samples, self.strategies):
            regret_plus = np.max(np.vstack([mapping[sample], np.zerosf(NUM_ACTION)]), axis=0)
            tot = np.sum(regret_plus)
            if tot:
                new = (regret_plus / tot + old * self.T) / (self.T + 1)
                # TODO (sample, new)

    def get_strategy(self, game):
        ret = self.run_ptr
        self.run_ptr += 1
        return ret, self.strategies[ret]

    def update_strategy(self, node, regret):
        self.regrets[node] = regret

    def dfs(self, game: Game, player):
        if game.change_state() == GAME_OVER:
            return game.payoff()

        if game.player != player:
            a = np.random.choice(range(NUM_ACTION), 1)[0]
            self.sampling_choices.put(a)
            next_game = deepcopy(game)
            next_game.act(a)
            self.dfs(next_game, player)
        else:
            self.submit(game)
            for a in range(NUM_ACTION):
                next_game = deepcopy(game)
                next_game.act(a)
                self.dfs(next_game, player)

    def cfr(self, game: Game, player):
        if game.change_state() == GAME_OVER:
            return game.payoff()

        if game.player != player:
            #     node, strategy = self.get_strategy(game, history)
            #     a = np.random.choice(range(NUM_ACTION), 1, p=strategy)[0]

            a = self.sampling_choices.get()
            next_game = deepcopy(game)
            next_game.act(a)

            util = self.cfr(next_game, player)
            return util
        else:
            node, strategy = self.get_strategy(game)
            util = np.zeros(NUM_ACTION)
            cv_util = np.zeros(NUM_ACTION)
            for a in range(NUM_ACTION):
                next_game = deepcopy(game)
                next_game.act(a)
                next_util = self.cfr(next_game, player)
                cv_util[a] = next_util[player]
                util += strategy[a] * next_util

            regret = cv_util - util
            self.update_strategy(node, regret)
            return util

    def train(self, max_iter):
        acc_util = np.array(NUM_ACTION)
        for i in range(max_iter):
            start = random.randint(0, NUM_PLAYER-1)
            player = random.randint(0, NUM_PLAYER-1)

            game = Game(start)
            acc_util += self.cfr(game, player)

            if i and i % 10000 == 0:
                print(i, '/', max_iter, ':', acc_util / i)

    def search(self, max_iter):
        for i in range(max_iter):
            start = random.randint(0, NUM_PLAYER - 1)
            player = random.randint(0, NUM_PLAYER - 1)
            game = Game(start)
            self.games.put((player, game))
            self.dfs(game, player)

    def run(self, max_iter):
        self.run_ptr = 0
        for i in range(max_iter):
            player, game = self.games.get()
            self.cfr(game, player)

