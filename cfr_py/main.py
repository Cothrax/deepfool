from game import *
from cfr import CFR
import random


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
    cfr.search(10)

    sample_size = len(cfr.samples)
    cfr.strategies = np.random.random(size=(sample_size, NUM_ACTION))

    cfr.strategies /= np.repeat(np.sum(cfr.strategies, axis=1).reshape(-1, 1), NUM_ACTION, axis=1)

    cfr.run(10)

if __name__ == '__main__':
    test_games()
