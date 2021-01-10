from .game import *
from .cfr import CFR
from .pure_cfr import PureCFR
import random
from time import time


def test_games():
    game = Game()
    while game.change_state() != GAME_OVER:
        if not game.is_raise_allowed():
            print('[raise not allowed]')
            act = random.randint(0, 1)
        else:
            act = random.randint(0, NUM_ACTION-1)
        game.act(act)
        input()

    print(game.payoff())


def test_dfs():
    cfr = CFR()

    cfr.search(5)
    sample_size = len(cfr.samples)
    strategies = np.random.random(size=(sample_size, NUM_ACTION))
    strategies /= np.repeat(np.sum(strategies, axis=1).reshape(-1, 1), NUM_ACTION, axis=1)

    cfr.strategies = strategies
    cfr.run(5)


if __name__ == '__main__':
    start = time()
    test_dfs()
    print(time() - start)
