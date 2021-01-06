//
// Created by cothrax on 1/5/21.
//

#ifndef DEEPFOOL_CFR_H
#define DEEPFOOL_CFR_H

#include "game.h"
#include "oracle.h"
#include <vector>
using std::vector;
using std::pair;

#define SEED 5555

class CFR {
private:
    Oracle *_oracle;

public:
    CFR(Oracle *oracle);
    void cfr(InfoSet &info, Game &game, vector<pair<int, int>> &history, double *pi, double *util, int dep);
    // void parallel_cfr();
    void train(int iter);
};

#endif //DEEPFOOL_CFR_H
