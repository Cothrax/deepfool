//
// Created by cothrax on 1/5/21.
//

#include "cfr.h"
#include <cstdlib>

extern long long debug_counter;
extern int debug_max_dep;

CFR::CFR(Oracle *oracle)
{
    _oracle = oracle;
    oracle->init();
}

void CFR::cfr(InfoSet &info, Game &game, vector<pair<int, int>> &history, double *pi, double *util, int dep)
{
    debug_max_dep = std::max(debug_max_dep, dep);
    debug_counter++;
    if(dep > 150)
    {
        for(auto i: history)
        {
            printf("%d %s\n", i.first,
                   i.second == 0 ? "FOLD" :
                   i.second == 1 ? "CHECK" :
                   i.second == 2 ? "3BB" :
                   i.second == 3 ? "HALF_POT":
                   i.second == 4 ? "POT" :
                   i.second == 5 ? "2POT" :
                   "ALL_IN");
        }

    }

    if(debug_counter % 100000 == 0)
    {
        printf("debug_counter = %lld, dep = %d\n", debug_counter, dep);
        printf("step = %d, player = %d, pot = %d\n"
               "\tchip = (%d, %d, %d, %d, %d, %d)\n"
               "\tbet = (%d, %d, %d, %d ,%d ,%d)\n"
               "\tfold = (%d, %d, %d, %d, %d, %d)\n",
               info.step, info.player, info.pot,
               info.chips[0], info.chips[1], info.chips[2], info.chips[3], info.chips[4], info.chips[5],
               info.bets[0], info.bets[1], info.bets[2], info.bets[3], info.bets[4], info.bets[5],
               info.folds[0], info.folds[1], info.folds[2], info.folds[3], info.folds[4], info.folds[5]);
    }

    if(game.change_state(info) == GAME_OVER)
    {
        game.calc_result(info, util);
        return;
    }
    Strategy cur;
    ull node = _oracle->single_query(game, info, cur);

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

        std::swap(pi[info.player], tmp);
        history.push_back(std::make_pair(info.player, a));
        cfr(next_info, game, history, pi, util, dep+1);

        history.pop_back();
        std::swap(pi[info.player], tmp);

        for(int i = 0; i < NUM_PLAYER; i++) util[i] /= cur[a];
    }
    else
    {
        double cv_util[NUM_ACTION];
        double regret[NUM_ACTION];
        double next_util[NUM_PLAYER];

        std::fill(util, util + NUM_PLAYER, 0);
        for(int a = 0; a < NUM_ACTION; a++)
        {
            game.act(info, a, next_info);
            double tmp = pi[info.player] * cur[a];

            history.push_back(std::make_pair(info.player, a));
            std::swap(pi[info.player], tmp);
            cfr(next_info, game, history, pi, next_util, dep+1);

            history.pop_back();
            std::swap(pi[info.player], tmp);

            cv_util[a] = next_util[game.pov];
            for(int i = 0; i < NUM_PLAYER; i++) util[i] += next_util[i] * cur[a];
        }

        double fac = 1; // pi_{-i}^sigma
        for(int i = 0; i < NUM_PLAYER; i++) if(i != game.pov) fac *= pi[i];
        for(int a = 0; a < NUM_ACTION; a++) regret[a] = fac * (cv_util[a] - util[a]);
        // _oracle->update(game, node, regret);
    }
}

void CFR::train(int iter)
{
    Game game;
    vector<pair<int, int>> history;
    srand(SEED);
    double pi[NUM_PLAYER], util[NUM_PLAYER];

    for(int i = 0; i < iter; i++)
    {
        int start = rand()%NUM_PLAYER, pov = rand()%NUM_PLAYER;
        game.generate(start, pov);
        InfoSet info(game);

        std::fill(pi, pi+NUM_PLAYER, 1);
        cfr(info, game, history, pi, util, 0);
    }
}