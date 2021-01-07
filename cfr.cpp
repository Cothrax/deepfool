//
// Created by cothrax on 1/5/21.
//

#include "cfr.h"
#include <cstdlib>
#include <cmath>
#include <cassert>

extern long long debug_counter;
extern int debug_max_dep;


bool CFR::cfr(InfoSet &info, Game &game, vector<pair<int, int>> &history, double *pi, double *util, int dep)
{
    debug_max_dep = std::max(debug_max_dep, dep);
    debug_counter++;
    if(debug_counter == 50000000L)
    {
        for(auto i: history)
        {
            printf("%d %s\n", i.first,
                   i.second == FOLD ? "FOLD" :
                   i.second == CHECK ? "CHECK" :
                   i.second == RAISE_3BB ? "3BB" :
                   i.second == RAISE_HALF_POT ? "HALF_POT":
                   i.second == RAISE_POT ? "POT" :
                   i.second == RAISE_2POT ? "2POT" :
                   "ALL_IN");
        }
        return false;
    }

    if(debug_counter % 1000000 == 0)
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
        return true;
    }
    Strategy cur;
    ull node = _oracle->single_query(game, info, cur);

    InfoSet next_info;
    if(info.player != game.pov)
    {
        // External Sampling
        int a;
        if(sampling_actions.empty())
        {
            double cum = 0;
            double r = (rand()&32767)/32768.0;
            for(a = 0; a < NUM_ACTION; a++)
            {
                cum += cur[a];
                if(r <= cum) break;
            }
        }
        else
        {
            a = sampling_actions.front();
            sampling_actions.pop();
        }

        game.act(info, a, next_info);

        double tmp = pi[info.player] * cur[a];

        std::swap(pi[info.player], tmp);
        history.push_back(std::make_pair(info.player, a));
        bool status = cfr(next_info, game, history, pi, util, dep+1);
        history.pop_back();
        std::swap(pi[info.player], tmp);
        if(!status) return false;

        //for(int i = 0; i < NUM_PLAYER; i++) util[i] /= cur[a];
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
            bool status = cfr(next_info, game, history, pi, next_util, dep+1);
            history.pop_back();
            std::swap(pi[info.player], tmp);

            if(!status) return false;

            cv_util[a] = next_util[game.pov];
            for(int i = 0; i < NUM_PLAYER; i++) util[i] += next_util[i] * cur[a];
        }

        double s = 0;
        for(int i = 0; i < NUM_PLAYER; i++) s += util[i];
        assert(fabs(s) < 1e-2);

        double fac = 1; // pi_{-i}^sigma
        for(int i = 0; i < NUM_PLAYER; i++) if(i != game.pov) fac *= pi[i];
        for(int a = 0; a < NUM_ACTION; a++) regret[a] = fac * (cv_util[a] - util[a]);
         _oracle->update(game, node, regret);
    }
    return true;
}

void CFR::train(int iter)
{
    Game game;
    vector<pair<int, int>> history;
    srand(SEED);
    double pi[NUM_PLAYER], util[NUM_PLAYER];

    int cnt = 0;

    double acc_util[NUM_PLAYER];
    for(int i = 0; i < iter; i++)
    {
        debug_counter = 0;
        debug_max_dep = 0;

        int start = rand()%NUM_PLAYER, pov = rand()%NUM_PLAYER;
        game.generate(start, pov);
        InfoSet info(game);

        std::fill(pi, pi+NUM_PLAYER, 1);
        int succ = cfr(info, game, history, pi, util, 0);
        if(!succ)
        {
            cnt++;
            printf("failed iter\n");
        }

//        printf("iter %d: ", i);
//        for(int j = 0; j < NUM_PLAYER; j++) printf("%.3lf ", util[j]);
//        printf("\ncnt = %lld, max_dep = %d\n", debug_counter, debug_max_dep);
        for(int j = 0; j < NUM_PLAYER; j++) acc_util[j] += util[j];
        if(i && i % 1000 == 0)
        {
            printf("%d/%d:: ", i, iter);
            for(int j = 0; j < NUM_PLAYER; j++) printf("%.3lf ", acc_util[j] / i);
            printf("\ncnt = %lld, max_dep = %d\n", debug_counter, debug_max_dep);
        }
    }
    printf("FAILED ITER: %d / %d\n", cnt, iter);
}

void CFR::dfs(InfoSet &info, Game &game, vector<pair<int, int>> &history)
{
    _oracle->submit(info, game, history);
    if(game.change_state(info) == GAME_OVER) return;
    Strategy cur;
    ull node = _oracle->single_query(game, info, cur);

    InfoSet next_info;
    if(info.player != game.pov)
    {
        // External Sampling
        int a = rand()%NUM_ACTION;
        sampling_actions.push(a);

        game.act(info, a, next_info);
        history.push_back(std::make_pair(info.player, a));
        dfs(next_info, game, history);
        history.pop_back();
        //for(int i = 0; i < NUM_PLAYER; i++) util[i] /= cur[a];
    }
    else
    {
        for(int a = 0; a < NUM_ACTION; a++)
        {
            game.act(info, a, next_info);
            history.push_back(std::make_pair(info.player, a));
            dfs(next_info, game, history);
            history.pop_back();
        }
    }
}

void CFR::parallel_cfr(Game &game, double *util)
{
    while(!sampling_actions.empty()) sampling_actions.pop();

    InfoSet info(game);
    vector<pair<int, int>> history;
    dfs(info, game, history);

    _oracle->commit();
    double pi[NUM_ACTION];
    for(int i = 0; i < NUM_ACTION; i++) pi[i] = 1;

    cfr(info, game, history, pi, util, 0);
}

void CFR::parallel_train(int iter, int batch_size)
{
    Game game;
    double cum_util[NUM_PLAYER], util[NUM_PLAYER];
    for(int i = 0; i < iter; i++)
    {
        for(int k = 0 ; k < batch_size; k++)
        {
            int start = rand()%NUM_PLAYER, pov = rand()%NUM_PLAYER;
            game.generate(start, pov);
            parallel_cfr(game, util);
            for(int j = 0; j < NUM_PLAYER; j++) cum_util[j] += util[j];
        }

        printf("%d/%d:: ", i, iter);
        for(int j = 0; j < NUM_PLAYER; j++) printf("%.3lf ", cum_util[j] / i);
        printf("\ncnt = %lld, max_dep = %d\n", debug_counter, debug_max_dep);
    }
}

CFR::CFR()
{
    _oracle = new NaiveOracleV2();
}
