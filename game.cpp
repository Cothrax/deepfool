//
// Created by cothrax on 1/5/21.
//

#include "game.h"

InfoSet::InfoSet(Game &game)
{
    step = 0;
    memcpy(chips, game.chips, sizeof(_chips));
    memset(folds, 0, sizeof(folds));
    memset(bets, 0, sizeof(bets));
    pot = SMALL_BLIND_CHIP + BIG_BLIND_CHIP;

    bets[game.start] = SMALL_BLIND_CHIP;
    bets[game.big_blind] = BIG_BLIND_CHIP;
    player = (game.start+2) % NUM_PLAYER;
}

void generate(int start, int pov)
{
    // TODO
}

int Game::change_state(InfoSet &info)
{
    int i, lst_bet = -1, cnt = 0;
    for(i = 0; i < NUM_PLAYER; i++)
        if(!info.folds[i])
        {
            if(lst_bet == -1 || lst_bet == info.bets[i]) lst_bet = info.bets[i], cnt++;
            else return NO_CHANGE;
        }

    info.player = big_blind;
    for(int i = 0; i < NUM_PLAYER; i++) info.pot += info.bets[i];
    memset(info.bets, 0, sizeof(info.bets));
    return cnt > 1 ? ++info.step : GAME_OVER;
}

void Game::calc_result(InfoSet &info, double *util)
{
    int maxv = -1, cnt = 0;
    for(int i = 0; i < NUM_PLAYER; i++)
        if(!info.folds[i]) maxv = max(maxv, power[i]);
    for(int i = 0; i < NUM_PLAYER; i++)
        cnt += !info.folds[i] && power[i] == maxv;

    for(int i = 0; i < NUM_PLAYER; i++)
        if(val[i] == maxv) util[i] = info.pot / cnt;
        else util[i] = chips[i] - info.chips[i];
}

int InfoSet::bet(int more){
    more = min(more, chips[player]);
    chips[info.player] -= more;
    bets[info.player] += more;
}

void Game::act(InfoSet &info, int action, InfoSet &next_info)
{
    memcpy(&next_info, &info, sizeof(InfoSet));
    int prev_player = info.player, next_player = info.player;

    do prev_player = prev_player ? prev_player - 1 : NUM_PLAYER - 1;
    while(info.folds[prev_player]);
    do next_player = next_player == NUM_PLAYER - 1 ? 0 : next_player + 1;
    while(info.folds[next_player]);

    int det = info.bets[prev_player] - info.bets[info.player];
    switch(action) {
        case FOLD: next_info.folds[info.player] = true; break;
        case CHECK: next_info.bet(det); break;
        case RAISE_3BB: next_info.bet(det + 3 * BIG_BLIND_CHIP); break;
        case RAISE_HALF_POT: next_info.bet(det + next_info.pot / 2); break;
        case RAISE_POT: next_info.bet(det + next_info.pot); break;
        case RAISE_2POT: next_info.bet(det + next_info.pot * 2); break;
        case ALL_IN: next_info.bet(INF32);
    }

    next_info.player = next_player;
}