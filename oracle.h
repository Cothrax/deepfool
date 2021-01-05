//
// Created by cothrax on 1/5/21.
//

#ifndef DEEPFOOL_ORACLE_H
#define DEEPFOOL_ORACLE_H

#include "game.h"

class Oracle
{
public:
    void init() = 0;
    int
    single_query(InfoSet &info, Strategy &strategy) = 0;
    void update(int node, double *regret) = 0;

    void query(vector<InfoSet> &infos, vector<Strategy> &strategies) = 0;
    // void learn(vector<Infoset> &infos, vector<Strategy> &strategies) = 0;
};

#endif //DEEPFOOL_ORACLE_H
