
import numpy as np
import random
from .calculator import MYCFR

SHOW = False


def card2str(x):
    return '<%s:%s>' % (chr(x//13+ord('a')), x%13)


NUM_PLAYER = 6
NUM_ACTION = 7
INIT_CHIPS = 20000
SMALL_BLIND = 50
BIG_BLIND = 100

FOLD = 0
CHECK = 1
RAISE_3BB = 2
RAISE_HALF_POT = 3
RAISE_POT = 4
RAISE_2POT = 5
ALL_IN = 6
NUM_ACTION = 7

# cum_prob = np.array([0.1, 0.6, 0.65, 0.8, 0.9, 0.99])
init_prob = np.array([0.1, 0.5, 0.05, 0.2, 0.1, 0.04, 0.01])

NO_CHANGE = 0
PRE_FLOP = 0
FLOP = 1
TURN = 2
RIVER = 3
GAME_OVER = 4
NUM_STATE = 5

calculator = MYCFR.Calculator()


class Game:
    __slots__ = ['holes', 'pubs', 'start', 'big_blind', 'bets',
                 'chips', 'folds', 'power', 'step', 'num', 'pot',
                 'cur_bet', 'player', 'history', 'num_act']

    def __init__(self, start=0):
        self.holes = -np.ones(shape=(NUM_PLAYER, 2), dtype=np.int)
        self.pubs = -np.ones(shape=5, dtype=np.int)
        self.start = start
        self.big_blind = (start + 1) % NUM_PLAYER

        self.bets = np.zeros(shape=NUM_PLAYER, dtype=np.int)
        self.chips = np.ones(shape=NUM_PLAYER, dtype=np.int) * INIT_CHIPS
        self.folds = np.zeros(shape=NUM_PLAYER, dtype=np.bool)
        self.power = np.zeros(shape=NUM_PLAYER, dtype=np.int)

        self.step = 0
        self.num = 6
        self.num_act = 0
        self.pot = 0
        self.cur_bet = BIG_BLIND

        self.bets[self.start] = SMALL_BLIND
        self.chips[self.start] -= SMALL_BLIND
        self.bets[self.big_blind] = BIG_BLIND
        self.chips[self.big_blind] -= BIG_BLIND

        self.player = (self.big_blind + 1) % NUM_PLAYER
        # self.history = [[] for _ in range(NUM_PLAYER)]
        self.history = -np.ones(shape=(4, NUM_PLAYER))
        self.generate()

    def generate(self):
        cards = np.random.choice(range(52), NUM_PLAYER*2+5, False)
        self.holes = cards[:NUM_PLAYER*2].reshape(NUM_PLAYER, 2)
        self.pubs = cards[-5:]

        for i in range(NUM_PLAYER):
            v = calculator.power(int(self.holes[i][0]), int(self.holes[i][1]),
                                 int(self.pubs[0]), int(self.pubs[1]),
                                 int(self.pubs[2]), int(self.pubs[3]), int(self.pubs[4]))
            self.power[i] = v

        if SHOW:
            print('pubs: ', end='')
            for x in self.pubs:
                print('%s, ' % card2str(x), end='')
            print('\nholes: ', end='')
            for each in self.holes:
                print('(%s, %s), ' % (card2str(each[0]), card2str(each[1])), end='')
            print()

    def bet(self, more):
        more = min(more, self.chips[self.player])
        self.chips[self.player] -= more
        self.bets[self.player] += more

    def act(self, a):
        if SHOW:
            print('action: %s' % a)

        if a != FOLD:
            self.num_act += 1
        det = self.cur_bet - self.bets[self.player]
        assert(det >= 0)
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

        self.cur_bet = max(self.cur_bet, self.bets[self.player])
        self.history[self.step, self.player] = a

        # self.history[self.player].append(a)
        self.player = (self.player + 1) % NUM_PLAYER
        while self.folds[self.player]:
            # self.history[self.player].append(-1)
            self.player = (self.player + 1) % NUM_PLAYER

    def _change_state(self):
        self.pot += np.sum(self.bets)
        self.bets[:] = 0
        # max_len = np.max([len(x) for x in self.history])
        # for i in range(NUM_PLAYER):
        #     if len(self.history[i]) != max_len:
        #         det = max_len - len(self.history[i])
        #         self.history[i].extend([-1] * det)
        self.player = self.big_blind
        self.step += 1

    def change_state(self):
        if self.num == 1:
            self._change_state()
            return GAME_OVER

        if self.num_act < self.num:
            return NO_CHANGE

        v = self.bets[(self.folds == 0) & (self.chips > 0)]
        if len(v) > 0 and np.max(v) != np.min(v):
            return NO_CHANGE

        self._change_state()
        return self.step

    def payoff(self):
        max_p = np.max(self.power[self.folds == 0])
        win = (self.power == max_p) & (self.folds == 0)
        cnt = np.sum(win)
        util = (self.chips - INIT_CHIPS) + (self.pot / cnt) * win
        return util
