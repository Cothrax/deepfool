from params import *
from game import *
from copy import deepcopy, copy
from collections import deque
from time import time
import numpy as np
from pure_cfr import PureCFR, RandomAdvisor
from multiprocessing import Pool, Process


class BfsCFR(PureCFR):
    """
    @deprecated
    Attempt using BFS to accelerate PureCFR (but doesn't work...)
    """
    
    def __init__(self, advisor):
        super().__init__(advisor)
        self.nodes = []
        self.labels = []
        self.head = None

    def search_forward(self, start):
        end = start + 1
        while end < len(self.nodes) and self.nodes[end]['depth'] == self.nodes[start]['depth']:
            end += 1

        samples = [e['sample'] for e in self.nodes[start:end]]
        result = self.advisor.batch_ask(samples)
        for strategy, node in zip(result, self.nodes[start:end]):
            node['strategy'] = strategy


    def bfs(self, init_game: Game, player):
        self.nodes = [{
            'game': init_game,
            'sample': self.convert_sample(init_game),
            'regret': None,
            'util': None,
            'strategy': None,
            'child': [],
            'depth': 0
        }]
        self.head = 0

        while self.head < len(self.nodes):
            # print(self.cnt)
            self.cnt += 1
            cur_node = self.nodes[self.head]

            if cur_node['strategy'] is None:
                self.search_forward(self.head)

            self.head += 1

            game = cur_node['game']
            if game.change_state() == GAME_OVER:
                cur_node['util'] = game.payoff()
                continue

            if game.player != player:
                a = self.get_sample_action(game, sample_prob=cur_node['strategy'])
                next_game = deepcopy(game)
                next_game.act(a)

                cur_node['child'].append(len(self.nodes))
                self.nodes.append({
                    'game': next_game,
                    'sample': self.convert_sample(next_game),
                    'util': None,
                    'regret': None,
                    'strategy': None,
                    'child': [],
                    'depth': cur_node['depth'] + 1
                })
            else:
                legal_action = range(NUM_ACTION)
                if not game.is_raise_allowed():
                    cur_node['strategy'][NUM_NOT_RAISE:] = 0
                    cur_node['strategy'] /= np.sum(cur_node['strategy'])
                    legal_action = range(NUM_NOT_RAISE)

                for a in legal_action:
                    next_game = deepcopy(game)
                    next_game.act(a)

                    cur_node['child'].append(len(self.nodes))
                    self.nodes.append({
                        'game': next_game,
                        'sample': self.convert_sample(next_game),
                        'regret': None,
                        'util': None,
                        'strategy': None,
                        'child': [],
                        'depth': cur_node['depth'] + 1
                    })

        for node in reversed(self.nodes):
            if node['util'] is not None:
                continue
            if node['game'].player != player:
                assert len(node['child']) == 1
                node['util'] = self.nodes[node['child'][0]]['util']
            else:
                legal_action = range(NUM_ACTION)
                if not node['game'].is_raise_allowed():
                    node['strategy'][NUM_NOT_RAISE:] = 0
                    node['strategy'] /= np.sum(node['strategy'])
                    legal_action = range(NUM_NOT_RAISE)

                assert(len(legal_action) == len(node['child']))
                cv_util = np.zeros(NUM_ACTION)
                node['util'] = np.zeros(NUM_ACTION)
                for a, c in zip(legal_action, node['child']):
                    next_util = self.nodes[c]['util']
                    cv_util[a] = next_util[player]
                    node['util'] += next_util * node['strategy'][a]
                node['regret'] = cv_util - node['util'][player]

                self.regrets.append(node['regret'])
                self.samples.append(node['sample'])
                self.strategies.append(node['strategy'])

    def cfr(self, game: Game, player):
        self.bfs(game, player)


def test_bfs_cfr():
    start = time()
    advisor = RandomAdvisor()
    cfr = BfsCFR(advisor)
    cfr.run(20)
    print(time() - start)
    pass

if __name__ == '__main__':
    test_bfs_cfr()
