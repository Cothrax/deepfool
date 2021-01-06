//
// Created by cothrax on 1/5/21.
//

#ifndef DEEPFOOL_ORACLE_H
#define DEEPFOOL_ORACLE_H

#include "game.h"
#include <unordered_map>
#include <vector>
#include <algorithm>

typedef unsigned long long ull;
using std::unordered_map;
using std::vector;

#define REGRET_FAC 1e3
//#define NOT_FOUND (ull)(1<<60)

class Oracle
{
public:
    virtual void init() = 0;
    virtual ull single_query(Game &game, InfoSet &info, Strategy &strategy) = 0;
    virtual void update(Game &game, ull node, double *regret) = 0;

    virtual void query(vector<InfoSet> &infos, vector<Strategy> &strategies) = 0;
    // void learn(vector<Infoset> &infos, vector<Strategy> &strategies) = 0;
};

class NaiveOracle: public Oracle
{
    double init_prob[NUM_ACTION];
public:
    struct Node
    {
        mutable double regret[NUM_ACTION];
        void get_strategy(Strategy &strategy, double *init_prob);
        Node() { std::fill(regret, regret + NUM_ACTION, 0); }
    };

    unordered_map<ull, Node> node_map;
    ull encode(Game &game, InfoSet &info);
    void decode(ull key, int *res);

    void init();
    void query(vector<InfoSet> &infos, vector<Strategy> &strategies) {}

    ull single_query(Game &game, InfoSet &info, Strategy &strategy) override;
    void update(Game &game, ull node, double *regret) override;
    void dump(const char *fn);
    void load(const char *fn);
};


#endif //DEEPFOOL_ORACLE_H
