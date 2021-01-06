//
// Created by cothrax on 1/5/21.
//

#include "oracle.h"
#include "game.h"
#include "calculator.h"
#include <cmath>

void NaiveOracle::init()
{
    init_prob[FOLD]           = 1./4;
    init_prob[CHECK]          = 1./4;
    init_prob[RAISE_3BB]      = 1./4;
    init_prob[ALL_IN]         = 1./4/4;
    init_prob[RAISE_HALF_POT] = 1./4/4;
    init_prob[RAISE_POT]      = 1./4/4;
    init_prob[RAISE_2POT]     = 1./4/4;
}

void NaiveOracle::Node::get_strategy(Strategy &strategy, double *init_prob)
{
    double sum = 0;
    for(double i : regret) sum += std::max(i, 0.);
    if(fabs(sum) < 1e-6) memcpy(strategy, init_prob, sizeof(strategy));
    else for(int i = 0; i < NUM_ACTION; i++) strategy[i] = regret[i] / sum;
}

int NaiveOracle::single_query(Game &game, InfoSet &info, Strategy &strategy)
{
    ull key = encode(game, info);
    if(node_map.find(key) == node_map.end()) node_map[key] = Node();
    auto x = node_map[key];
    x.get_strategy(strategy, init_prob);
}

void NaiveOracle::update(Game &game, ull key, double *regret)
{
    if(node_map.find(key) == node_map.end()) node_map[key] = Node();
    auto x = node_map[key];
    for(int i = 0; i < NUM_ACTION; i++) x.regret[i] += regret[i];
}

ull NaiveOracle::encode(Game &game, InfoSet &info)
{
    int prev_player = info.player;
    do prev_player = prev_player ? prev_player - 1 : NUM_PLAYER - 1; while(info.folds[prev_player]);

    // (step:4, my_power, opp_power, (info.num-2):5, bet:3, pot:3)
    ll my_power = calculator.potential_power(game.holes[info.player], game.pubs, info.step);
    ll opp_power = calculator.opp_potential_power(game.holes[info.player], game.pubs, info.step);
    ll _opp = info.num - 2;

    int hold = info.chips[info.player];
    int more = std::min(info.bets[prev_player] - info.bets[info.player], hold);
    ll _bet = more * 2 < hold ? 0 : 1;
    ll _pot = info.pot * 2 < hold ? 0 : info.pot < hold ? 1 : 2;

    ll ret = ((((info.step * POWER_RANGE + my_power) * POWER_RANGE + opp_power) * 5 + _opp) * 2 + _bet) * 3 + _pot;
    return ret;
}