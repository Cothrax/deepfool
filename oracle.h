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
using std::pair;

#define REGRET_FAC 1e3
//#define NOT_FOUND (ull)(1<<60)

class Oracle
{
public:
    virtual void init() = 0;
    virtual ull single_query(Game &game, InfoSet &info, Strategy &strategy) = 0;
    virtual void update(Game &game, ull node, double *regret) = 0;

    virtual void submit(InfoSet &info, Game &game, vector<pair<int, int>> &history) {};
    virtual void commit() {};
    virtual void learn() {};
};

class NaiveOracle: public Oracle
{
protected:
    double init_prob[NUM_ACTION];
public:
    struct Node
    {
        mutable double regret[NUM_ACTION];
        void get_strategy(Strategy &strategy, double *init_prob);
        Node() { std::fill(regret, regret + NUM_ACTION, 0); }
    };

    unordered_map<ull, Node> node_map;
    virtual ull encode(Game &game, InfoSet &info);
    virtual void decode(ull key, int *res);

    void init();
    void query(vector<InfoSet> &infos, vector<Strategy> &strategies) {}

    ull single_query(Game &game, InfoSet &info, Strategy &strategy) override;
    void update(Game &game, ull node, double *regret) override;
    void dump(const char *fn);
    virtual void load(const char *fn);
};

class NaiveOracleV2: public NaiveOracle
{
public:
    ull encode(Game &game, InfoSet &info) override;
    void decode(ull key, int *res) override;
    void load(const char *fn) override;
};


#define MAX_HIST_SIZE 128
class DeepOracle: public Oracle
{
    struct Wrap
    {
        int player;
        int holes[2];
        int pubs[5];
        int history[MAX_HIST_SIZE][2];
        double regret[NUM_ACTION];
    };

    int commit_time;

    virtual void init()
    {
        commit_time = 0;
        cache.clear();
    }

    ull single_query(Game &game, InfoSet &info, Strategy &strategy) override;

    vector<Wrap> cache;
    vector<Strategy> strategies;

    void update(Game &game, ull node, double *regret);
    virtual void submit(InfoSet &info, Game &game, vector<pair<int, int>> &history);
    virtual void commit();
    virtual void learn();
};

#endif //DEEPFOOL_ORACLE_H
