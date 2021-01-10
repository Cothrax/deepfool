import sys
import time
import pickle as pkl
import numpy as np
import scipy.stats as st
from cfr_py.calculator import MYCFR
from multiprocessing.dummy import Pool

def generate_sample(MAX_CTR):
    calculator = MYCFR.Calculator()
    ctr = 0
    samples = []
    while ctr < MAX_CTR:
        cards = np.arange(52)
        np.random.shuffle(cards)
        cards = cards[:7]
        step = np.random.randint(4)

        if step == 0:
            cards[2:] = -1
        elif step == 1:
            cards[5:] = -1
        elif step == 2:
            cards[6:] = -1

        win = calculator.prior_win_rate(
            int(cards[0]),
            int(cards[1]),
            int(cards[2]),
            int(cards[3]),
            int(cards[4]),
            int(cards[5]),
            int(cards[6]),
            step,
            MCtimes=1000
        )
        prob = np.array(st.norm.pdf(range(6), loc=win*5))
        prob = prob / prob.sum()

        sample = [cards.astype(np.int64), prob.astype(np.float32)]
        samples.append(sample)
        ctr += 1
        if ctr % 100 == 0:
            print(ctr)

    return samples

def main():
    i = [MAX_CTR] *  NUM_CPU
    with Pool(NUM_CPU) as p:
        all_samples = p.map(generate_sample, i)

    pkl.dump(all_samples, open("./data_{}.pkl".format(sys.argv[1]), "wb"))

if __name__ == "__main__":
    MAX_CTR = 300000
    NUM_CPU = 1
    main()