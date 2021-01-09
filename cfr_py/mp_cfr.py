import numpy as np
from params import *
from multiprocessing import Pool, Process
from cfr import CFR
from time import time


class ParallelCFR:
    def __init__(self, num_cfr, max_iter):
        self.num_cfr = num_cfr
        self.max_iter = max_iter
        self.cfr_list = [CFR() for _ in range(num_cfr)]

    def worker_search(self, cfr: CFR):
        return cfr.search(self.max_iter)

    def worker_run(self, cfr):
        return cfr.run(self.max_iter)

    def parallel_run(self):
        with Pool(N_CPU) as p:
            self.cfr_list = p.map(self.worker_run, self.cfr_list)

    def parallel_search(self):
        with Pool(N_CPU) as p:
            self.cfr_list = p.map(self.worker_search, self.cfr_list)


def test_parallel():
    para_cfr = ParallelCFR(8, 50)
    para_cfr.parallel_search()
    for cfr in para_cfr.cfr_list:
        sample_size = len(cfr.samples)
        strategies = np.random.random(size=(sample_size, NUM_ACTION))
        strategies /= np.repeat(np.sum(strategies, axis=1).reshape(-1, 1), NUM_ACTION, axis=1)

        cfr.strategies = strategies

    para_cfr.parallel_run()
    for cfr in para_cfr.cfr_list:
        print('labels: ', len(cfr.labels))


if __name__ == '__main__':
    start = time()
    test_parallel()
    print(time() - start)
