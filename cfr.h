//
// Created by cothrax on 1/5/21.
//

#ifndef DEEPFOOL_CFR_H
#define DEEPFOOL_CFR_H

#include "game.h"
#include "oracle.h"

#define SEED 5555

class CFR {
private:
    Oracle *_oracle;
    vector<InfoSet> info_buf;
    vector<Strategy> strategy_buf;

public:
    CFR(Oracle *oracle, bool need_learn);
    void cfr(InfoSet &info, Game &game, double *pi, double *util);
    parallel_cfr();
    void train(int iter);
};

#endif //DEEPFOOL_CFR_H
