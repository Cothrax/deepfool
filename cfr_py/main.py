from game import *
from cfr import CFR
import random


def test_games():
    game = Game()
    while game.change_state() != GAME_OVER:
        # act = int(input('enter action: '))
        act = random.randint(0, NUM_ACTION-1)
        game.act(act)
    print(game.payoff())


def test_dfs():
    cfr = CFR()
    cfr.search(10)
    print(len(cfr.samples))


if __name__ == '__main__':
    test_dfs()
