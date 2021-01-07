//
// Created by cothrax on 1/5/21.
//

#ifndef DEEPFOOL_CFR_H
#define DEEPFOOL_CFR_H

#include "game.h"
#include "oracle.h"
#include <vector>
#include <queue>
using std::vector;
using std::queue;
using std::pair;

#define SEED 5555

class CFR {
private:
    Oracle *_oracle;
    void dfs(InfoSet &info, Game &game, vector<pair<int, int>> &history);
    queue<int> sampling_actions;

public:
    CFR(Oracle *oracle);
    bool cfr(InfoSet &info, Game &game, vector<pair<int, int>> &history, double *pi, double *util, int dep);
    void train(int iter);

    void parallel_cfr(Game &game, double *util);
    void parallel_train(int iter, int batch_size);
};

#endif //DEEPFOOL_CFR_H
