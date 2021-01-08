from game import *
from copy import deepcopy
from queue import Queue
import numpy as np


class CFR:
    __slots__ = ['sampling_choices', 'games', 'strategies',
                 'samples', 'regrets', 'run_ptr']

    def __init__(self):
        self.sampling_choices = Queue()
        self.games = Queue()

        self.run_ptr = 0
        self.strategies = None
        self.samples = None
        self.regrets = None

    def submit(self, game, history):
        # TODO submit to samples
        sample = None
        self.samples = np.append(self.samples, [sample], axis=0)

    def get_strategy(self, game, history):
        ret = self.run_ptr
        self.run_ptr += 1
        return ret, self.strategies[ret]

    def update_strategy(self, node, regret):
        self.regrets[node] = regret

    def dfs(self, game: Game, history, player):
        if game.change_state() == GAME_OVER:
            return game.payoff()

        if game.player != player:
            a = np.random.choice(range(NUM_ACTION), 1)[0]
            next_game = deepcopy(game)
            next_game.act(a)
            self.dfs(next_game,
                     history + str(game.player) + str(a), player)
        else:
            self.submit(game, history)
            for a in range(NUM_ACTION):
                next_game = deepcopy(game)
                next_game.act(a)
                self.dfs(next_game,
                         history + str(game.player) + str(a),
                         player)

    def cfr(self, game: Game, history, player):
        if game.change_state() == GAME_OVER:
            return game.payoff()

        if game.player != player:
            #     node, strategy = self.get_strategy(game, history)
            #     a = np.random.choice(range(NUM_ACTION), 1, p=strategy)[0]

            a = self.sampling_choices.get()
            next_game = deepcopy(game)
            next_game.act(a)
            util = self.cfr(next_game,
                            history + str(game.player) + str(a),
                            player)
            return util
        else:
            node, strategy = self.get_strategy(game, history)
            util = np.zeros(NUM_ACTION)
            cv_util = np.zeros(NUM_ACTION)
            for a in range(NUM_ACTION):
                next_game = deepcopy(game)
                next_game.act(a)
                next_util = self.cfr(next_game,
                                     history + str(game.player) + str(a),
                                     player)
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
            acc_util += self.cfr(game, '', player)

            if i and i % 10000 == 0:
                print(i, '/', max_iter, ':', acc_util / i)

    def search(self, max_iter):
        for i in range(max_iter):
            start = random.randint(0, NUM_PLAYER - 1)
            player = random.randint(0, NUM_PLAYER - 1)
            game = Game(start)
            self.games.put((player, game))
            self.dfs(game, '', player)

    def run(self, max_iter):
        self.run_ptr = 0
        for i in range(max_iter):
            player, game = self.games.get()
            self.cfr(game, '', player)

