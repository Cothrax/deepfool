import numpy as np
# for CFR
FOLD_NUM = 100
N_CPU = 8

# for GAME
NUM_PLAYER = 6
INIT_CHIPS = 500
SMALL_BLIND = 1
BIG_BLIND = 2

FOLD = 0
CHECK = 1
RAISE_3BB = 2
# RAISE_HALF_POT = 3
RAISE_POT = 3
RAISE_2POT = 4
ALL_IN = 5

NUM_NOT_RAISE = 2
NUM_ACTION = 6

NO_CHANGE = 0
PRE_FLOP = 0
FLOP = 1
TURN = 2
RIVER = 3
GAME_OVER = 4
NUM_STATE = 5

equity_prob = np.array([0.1, 0.4, 0.7, 0.9, 1])

# for naive cfr
DUMP_ITER = 10

def card2str(x):
    return '<%s:%s>' % (chr(x//13+ord('a')), x % 13)

# for DBEUG
SHOW = False
# cum_prob = np.array([0.1, 0.6, 0.65, 0.8, 0.9, 0.99])
# init_prob = np.array([0.1, 0.3, 0.3, 0.15, 0.10, 0.05])
init_prob = np.ones(NUM_ACTION) / NUM_ACTION
# init_prob = np.array([0.1, 0.5, 0.05, 0.2, 0.1, 0.04, 0.01])
