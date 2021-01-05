//
// Created by cothrax on 1/5/21.
//

#include "cfr.h"
#include <cstdlib>

CFR::CFR(Oracle *oracle)
{
    _oracle = oracle;
    oracle.init();
}

void CFR::cfr(InfoSet &info, Game &game, double *pi, double *util)
{
    if(game.change_state(info) == GAME_OVER)
    {
        game.calc_results(info, util);
        return;
    }
    Strategy cur;
    int node = _oracle->single_query(info, cur);

    InfoSet next_info;
    if(info.player != game.pov)
    {
        // External Sampling
        int a;
        double cum = 0;
        double r = (rand()&32767)/32768.0;
        for(a = 0; a < NUM_ACTION; a++)
        {
            cum += cur[a];
            if(r <= cum) break;
        }

        game.act(info, a, next_info);
        double tmp = pi[info.player] * cur[a];

        swap(pi[info.player], tmp);
        cfr(next_info, game, pi, util);
        swap(pi[info.player], tmp);

        for(int i = 0; i < NUM_PLAYER; i++) util[i] /= cur[a];
    }
    else
    {
        double cv_util[NUM_ACTION];
        double next_util[NUM_PLAYER];
        double regret[NUM_PLAYER];

        fill(util, util + NUM_PLAYER, 0);
        for(int a = 0; a < NUM_ACTION; a++)
        {
            game.act(info, a, next_info);
            double tmp = pi[info.player] * cur[a];
            swap(pi[info.player], tmp);
            cfr(next_info, game, pi, next_util);
            swap(pi[info.player], tmp);

            cv_util[a] = next_util[game.pov];
            for(int i = 0; i < NUM_PLAYER; i++) util[i] += next_util[i] * cur[a];
        }

        double fac = 1; // pi_{-i}^sigma
        for(int i = 0; i < NUM_PLAYER; i++) if(i != game.pov) fac *= pi[i];
        for(int a = 0; a < NUM_ACTION; a++) regret[a] = fac * (cv_util[a] - util[a]);
        _oracle->update(node, regret);
    }
}

void CFR::train(int iter)
{
    Game game;
    srand(SEED);
    double pi[NUM_PLAYER], util[NUM_PLAYER];

    for(int i = 0; i < iter; i++)
    {
        int start = rand()%NUM_PLAYER, pov = rand()%NUM_PLAYER;
        game.generate(start, pov);
        InfoSet info(game);

        fill(pi, pi+NUM_PLAYER, 1);
        cfr(info, game, pi, util);
    }
}