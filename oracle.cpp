//
// Created by cothrax on 1/5/21.
//

#include "oracle.h"
#include "game.h"
#include "calculator.h"
#include <cmath>
#include <fstream>

void NaiveOracle::init()
{
    init_prob[FOLD]           = 1./4;
    init_prob[CHECK]          = 1./4;
//    init_prob[RAISE_3BB]      = 1./4;
    init_prob[RAISE_QUARTER_POT]= 1./4;
    init_prob[RAISE_HALF_POT] = 1./4/4;
    init_prob[RAISE_POT]      = 1./4/4;
    init_prob[RAISE_2POT]     = 1./4/4;
    init_prob[ALL_IN]         = 1./4/4;
}

void NaiveOracle::Node::get_strategy(Strategy &strategy, double *init_prob)
{
    double sum = 0;
    for(double i : regret) sum += std::max(i, 0.);
    if(fabs(sum) < 1e-18) memcpy(strategy, init_prob, sizeof(strategy));
    else for(int i = 0; i < NUM_ACTION; i++) strategy[i] = std::max(regret[i], 0.) / sum;
}

ull NaiveOracle::single_query(Game &game, InfoSet &info, Strategy &strategy)
{
    ull key = encode(game, info);
    if(node_map.find(key) == node_map.end())
    {
        memcpy(strategy, init_prob, sizeof(strategy));
        return key;
    }
    auto &x = node_map[key];
    x.get_strategy(strategy, init_prob);
    return key;
}

void NaiveOracle::update(Game &game, ull key, double *regret)
{
    if(node_map.find(key) == node_map.end()) node_map[key] = Node();
    auto &x = node_map[key];
    for(int i = 0; i < NUM_ACTION; i++) x.regret[i] += regret[i] * REGRET_FAC;
}

ull NaiveOracle::encode(Game &game, InfoSet &info)
{
    int prev_player = info.player;
    do prev_player = prev_player ? prev_player - 1 : NUM_PLAYER - 1; while(info.folds[prev_player]);

    // (step:4, my_power, opp_power, (info.num-2):5, bet:2, pot:3)
    ull my_power = calculator.potential_power(game.holes[info.player], game.pubs, info.step);
    ull opp_power = calculator.opp_potential_power(game.holes[info.player], game.pubs, info.step);

    my_power = my_power * POWER_RANGE / MAX_POWER;
    opp_power = opp_power * POWER_RANGE / MAX_POWER;

    ull _opp = info.num - 2;

    int hold = info.chips[info.player];
    int more = std::min(info.bets[prev_player] - info.bets[info.player], hold);
    ull _bet = more * 2 < hold ? 0 : 1;
    ull _pot = info.pot * 2 < hold ? 0 : info.pot < hold ? 1 : 2;

//    ull num_bet = 0;
//    for(int i = 0; i < NUM_PLAYER; i++) num_bet += info.bets[i] != 0;

    ull ret = (((((info.step * POWER_RANGE + my_power) * POWER_RANGE + opp_power) * 5 + _opp) * 2 + _bet) * 3 + _pot);
    return ret;
}

void NaiveOracle::decode(ull key, int *res)
{
//    res[0] = key % 6; key /= 6;
    res[0] = key % 3; key /= 3;
    res[1] = key % 2; key /= 2;
    res[2] = key % 5; key /= 5;
    res[3] = key % POWER_RANGE; key /= POWER_RANGE;
    res[4] = key % POWER_RANGE; key /= POWER_RANGE;
    res[5] = key;
}

void NaiveOracle::dump(const char *fn)
{
    std::ofstream fout(fn, std::ios_base::out | std::ios_base::binary);
    int s = node_map.size();
    printf("node map size = %d\n", s);
    fout.write((char*)&s, sizeof(int));
    for(auto x: node_map)
    {
        fout.write((char*)&x.first, sizeof(x.first));
        fout.write((char*)&x.second, sizeof(x.second));
    }
    fout.close();
}

void NaiveOracle::load(const char *fn)
{
    std::ifstream fin(fn, std::ios_base::in | std::ios_base::binary);
    int s; fin.read((char*)&s, sizeof(int));
    printf("node map size = %d\n", s);

    ull key; Node node;

    int res[6];
    Strategy strategy;
    for(int i = 0; i < s; i++)
    {
        fin.read((char*)&key, sizeof(key));
        fin.read((char*)&node, sizeof(node));
        node_map[key] = node;

        decode(key, res);
        node.get_strategy(strategy, init_prob);

//        printf("(");
//        for(int i = 0; i < 6; i++) printf("%d, ", res[i]);
//        printf(") => (");
//        for(int i = 0; i < NUM_ACTION; i++) printf("%.3lf, ", strategy[i]);
//        printf(")\n");

//        if(i%1000 == 0) getchar();
    }
}


