
import numpy as np
import random


NUM_PLAYER = 6
NUM_ACTION = 7
INIT_CHIPS = 20000
SMALL_BLIND = 50
BIG_BLIND = 100

FOLD = 0
CHECK = 1
RAISE_3BB = 6
RAISE_HALF_POT = 2
RAISE_POT = 4
RAISE_2POT = 3
ALL_IN = 5
NUM_ACTION = 7

NO_CHANGE = 0
PRE_FLOP = 0
FLOP = 1
TURN = 2
RIVER = 3
GAME_OVER = 4
NUM_STATE = 5

calculator = Calculator()


class Game:
    __slots__ = ['holes', 'pubs', 'start', 'big_blind', 'bets',
                 'chips', 'folds', 'power', 'step', 'num', 'pot',
                 'cur_bet', 'player']

    def __init__(self, start):
        self.holes = -np.ones(shape=(NUM_PLAYER, 2), dtype=np.int)
        self.pubs = -np.ones(shape=5, dtype=np.int)
        self.start = start
        self.big_blind = (start + 1) % NUM_PLAYER

        self.bets = np.zeros(shape=NUM_PLAYER, dtype=np.int)
        self.chips = np.ones(shape=NUM_PLAYER, dtype=np.int) * INIT_CHIPS
        self.folds = np.zeros(shape=NUM_PLAYER, dtype=np.bool)
        self.power = np.zeros(shape=NUM_PLAYER, dtype=np.bool)

        self.step = 0
        self.num = 5
        self.pot = 0
        self.cur_bet = BIG_BLIND

        self.bets[self.start] = SMALL_BLIND
        self.chips[self.start] -= SMALL_BLIND
        self.bets[self.big_blind] = BIG_BLIND
        self.chips[self.start] -= BIG_BLIND

        self.player = (self.big_blind + 1) % NUM_PLAYER
        self.generate()

    def generate(self):
        cards = np.random.choice(range(52), NUM_PLAYER*2+5, False)
        self.holes = cards[NUM_PLAYER*2].reshape(NUM_PLAYER, 2)
        self.pubs = cards[-5:]

        for i in range(NUM_PLAYER):
            self.power[i] = calculator.power(*self.holes[NUM_PLAYER], *self.pubs)

    def bet(self, more):
        more = min(more, self.chips[self.player])
        self.chips[self.player] -= more
        self.bets[self.player] += more

    def act(self, a):
        det = self.cur_bet - self.bets[self.player]
        pot = self.pot + np.sum(self.bets)
        if a == FOLD:
            self.folds[self.player] = True
            self.num -= 1
        elif a == CHECK:
            self.bet(det)
        elif a == RAISE_3BB:
            self.bet(det + BIG_BLIND * 3)
        elif a == RAISE_POT:
            self.bet(det + pot)
        elif a == RAISE_HALF_POT:
            self.bet(det + pot // 2)
        elif a == RAISE_2POT:
            self.bet(det + pot)
        elif a == ALL_IN:
            self.bet(INIT_CHIPS)

        self.player = (self.player + 1) % NUM_PLAYER
        while self.folds[self.player]:
            self.player = (self.player + 1) % NUM_PLAYER

    def change_state(self):
        if self.num == 1:
            self.pot += np.sum(self.bets)
            self.bets[:] = 0
            return GAME_OVER

        v = self.bets[not self.folds and (self.chips > 0)]
        if np.max(v) != np.min(v):
            return NO_CHANGE

        self.player = self.big_blind
        self.pot += np.sum(self.bets)
        self.bets[:] = 0

        self.step += 1
        return self.step

    def payoff(self):
        max_p = np.max(self.power)
        win = self.power == max_p and not self.folds
        cnt = np.sum(win)
        util = (self.chips - INIT_CHIPS) + (self.pot / cnt) * win
        return util
