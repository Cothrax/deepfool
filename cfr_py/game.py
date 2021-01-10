from cfr_py.params import *
import numpy as np
import random
from cfr_py.calculator import MYCFR


# C++ based calculator
calculator = MYCFR.Calculator()


class Game:
    """
    Store all info for each state
    """
    
    __slots__ = ['holes', 'pubs', 'start', 'big_blind', 'bets',
                 'chips', 'folds', 'power', 'step', 'num', 'pot',
                 'cur_bet', 'player', 'history', 'num_act',
                 'if_call', 'if_raise', 'step_counter', 'win']

    def __init__(self, start=0):
        self.holes = -np.ones(shape=(NUM_PLAYER, 2), dtype=np.int)
        self.pubs = -np.ones(shape=5, dtype=np.int)
        self.start = start
        self.big_blind = (start + 1) % NUM_PLAYER
        self.power = np.zeros(shape=NUM_PLAYER, dtype=np.int)

        self.bets = np.zeros(shape=(4, NUM_PLAYER), dtype=np.int)
        self.chips = np.ones(shape=NUM_PLAYER, dtype=np.int) * INIT_CHIPS
        self.folds = np.zeros(shape=NUM_PLAYER, dtype=np.bool)


        self.step = 0
        self.num = 6
        self.pot = 0
        self.cur_bet = BIG_BLIND

        self.bets[0][self.start] = SMALL_BLIND
        self.chips[self.start] -= SMALL_BLIND
        self.bets[0][self.big_blind] = BIG_BLIND
        self.chips[self.big_blind] -= BIG_BLIND

        self.player = (self.big_blind + 1) % NUM_PLAYER
        # self.history = [[] for _ in range(NUM_PLAYER)]
        self.history = -np.ones(shape=(4, NUM_PLAYER))

        self.if_call = np.zeros(shape=(4, NUM_PLAYER), dtype=np.bool)
        self.if_raise = np.zeros(shape=(4, NUM_PLAYER), dtype=np.bool)
        self.step_counter = 0

        self.win = np.zeros(shape=(4, NUM_PLAYER))
        self.generate()

    def debug_print(self):
        print('[GAMES]')
        print('step:', self.step, 'pot:', self.pot, 'player:', self.player)
        print('step_counter:', self.step_counter, 'cur_bet:', self.cur_bet, 'start:', self.start)
        print('holes: ', list(map(card2str, self.holes.reshape(-1))))
        print('power: ', self.power)
        print('pubs: ', list(map(card2str, self.pubs)))
        print('bets: ', self.bets)
        print('chips: ', self.chips)
        print('folds: ', self.folds)
        print('if_call: ', self.if_call)
        print('if_raise: ', self.if_raise)

    def generate(self):
        cards = np.random.choice(range(52), NUM_PLAYER*2+5, False)
        self.holes = cards[:NUM_PLAYER*2].reshape(NUM_PLAYER, 2)
        self.pubs = cards[-5:]

        for i in range(NUM_PLAYER):
            v = calculator.power(int(self.holes[i][0]), int(self.holes[i][1]),
                                 int(self.pubs[0]), int(self.pubs[1]),
                                 int(self.pubs[2]), int(self.pubs[3]), int(self.pubs[4]))
            self.power[i] = v

            for j in range(4):
                self.win[j][i] = calculator.prior_win_rate(
                    int(self.holes[i][0]),
                    int(self.holes[i][1]),
                    int(self.pubs[0]),
                    int(self.pubs[1]),
                    int(self.pubs[2]),
                    int(self.pubs[3]),
                    int(self.pubs[4]),
                    j
                )

        if SHOW:
            self.debug_print()

    def bet(self, more):
        more = min(more, self.chips[self.player])
        self.chips[self.player] -= more
        self.bets[self.step][self.player] += more

    def act(self, a):
        if not self.is_raise_allowed():
            assert a == FOLD or a == CHECK

        det = self.cur_bet - self.bets[self.step][self.player]
        assert(det >= 0)
        pot = self.pot + np.sum(self.bets[self.step])
        if a == FOLD:
            self.folds[self.player] = True
            self.num -= 1
        elif a == CHECK:
            self.bet(det)
        elif a == RAISE_3BB:
            self.bet(det + BIG_BLIND * 3)
        elif a == RAISE_POT:
            self.bet(det + pot)
        #elif a == RAISE_HALF_POT:
        #    self.bet(det + pot // 2)
        elif a == RAISE_2POT:
            self.bet(det + pot)
        elif a == ALL_IN:
            self.bet(INIT_CHIPS)
        else:
            assert False, 'Unknown action %s' % a

        self.cur_bet = max(self.cur_bet, self.bets[self.step][self.player])
        self.step_counter += 1

        if a == CHECK:
            self.if_call[self.step, self.player] |= self.cur_bet > 0
        elif a != FOLD:
            self.if_raise[self.step, self.player] = True

        self.history[self.step, self.player] = a

        # self.history[self.player].append(a)
        self.player = (self.player + 1) % NUM_PLAYER
        while self.folds[self.player]:
            self.step_counter += 1
            # self.history[self.player].append(-1)
            self.player = (self.player + 1) % NUM_PLAYER

        if SHOW:
            print('action: %s' % a)
            self.debug_print()

    def is_raise_allowed(self):
        return self.step_counter < 6

    def get_legal_action(self):
        return range(NUM_ACTION) if self.is_raise_allowed() else range(NUM_NOT_RAISE)

    def _change_state(self):
        if SHOW:
            print('enter _change_state()')

        self.pot += np.sum(self.bets[self.step])
        # self.bets[:] = 0
        # max_len = np.max([len(x) for x in self.history])
        # for i in range(NUM_PLAYER):
        #     if len(self.history[i]) != max_len:
        #         det = max_len - len(self.history[i])
        #         self.history[i].extend([-1] * det)
        self.player = self.big_blind
        self.step_counter = 0
        while self.folds[self.player]:
            self.step_counter += 1
            # self.history[self.player].append(-1)
            self.player = (self.player + 1) % NUM_PLAYER

        self.step += 1
        self.cur_bet = 0

    def change_state(self):
        if self.num == 1:
            self._change_state()
            return GAME_OVER

        if self.step_counter < 6:
            return NO_CHANGE

        v = self.bets[self.step][(self.folds == 0) & (self.chips > 0)]
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

