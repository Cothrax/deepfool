//
// Created by cothrax on 1/5/21.
//

#include "calculator.h"
#include "game.h"
#include <algorithm>
#include <cmath>
#include <random>

InfoSet::InfoSet(Game &game)
{
    step = 0;
    memcpy(chips, game.chips, sizeof(chips));
    memset(folds, 0, sizeof(folds));
    memset(bets, 0, sizeof(bets));
    pot = SMALL_BLIND_CHIP + BIG_BLIND_CHIP;

    bets[game.start] = SMALL_BLIND_CHIP;
    bets[game.big_blind] = BIG_BLIND_CHIP;
    chips[game.start] -= SMALL_BLIND_CHIP;
    chips[game.big_blind] -= BIG_BLIND_CHIP;

    player = (game.start+2) % NUM_PLAYER;
    num = NUM_PLAYER;
}

void InfoSet::bet(int more){
    more = std::min(more, chips[player]);
    chips[player] -= more;
    bets[player] += more;
}

void Game::generate(int start_, int pov_)
{
    std::shuffle(cards, cards+52, std::mt19937(std::random_device()()));
    memcpy(pubs, cards, sizeof(pubs));
    memcpy(holes, cards + 5, sizeof(holes));

    pov = pov_;
    start = start_;
    big_blind = (start_ + 1) % NUM_PLAYER;
    std::fill(chips, chips+NUM_PLAYER, INIT_CHIP);

    for(int i = 0; i < NUM_PLAYER; i++)
        power[i] = calculator.power(holes[i], pubs);

    printf("gen game: start = %d, pov = %d\n\tholes: ", start_, pov_);
    for(int i = 0; i < NUM_PLAYER; i++) printf("(%d, %d), ", holes[i][0], holes[i][1]);
    printf("\n\tpubs:(%d, %d, %d)\n", pubs[0], pubs[1], pubs[2]);
}

int Game::change_state(InfoSet &info)
{
    if(info.num == 1) return GAME_OVER;
    int i, lst_bet = -1;
    for(i = 0; i < NUM_PLAYER; i++)
        if(!info.folds[i])
        {
            if(lst_bet == -1 || lst_bet == info.bets[i]) lst_bet = info.bets[i];
            else return NO_CHANGE;
        }

    info.player = big_blind;
    for(int bet : info.bets) info.pot += bet;
    memset(info.bets, 0, sizeof(info.bets));
    return ++info.step;
}

void Game::calc_result(InfoSet &info, double *util)
{
    int maxv = -1, cnt = 0;
    for(int i = 0; i < NUM_PLAYER; i++)
        if(!info.folds[i]) maxv = std::max(maxv, power[i]);
    for(int i = 0; i < NUM_PLAYER; i++)
        cnt += !info.folds[i] && power[i] == maxv;

    for(int i = 0; i < NUM_PLAYER; i++)
        if(!info.folds[i] && power[i] == maxv) util[i] = info.pot / cnt;
        else util[i] = chips[i] - info.chips[i];
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
    int pot = next_info.pot;
    for(int i = 0; i < NUM_PLAYER; i++) pot += info.bets[i];

    switch(action) {
        case FOLD: next_info.folds[info.player] = true; next_info.num--; break;
        case CHECK: next_info.bet(det); break;
        case RAISE_3BB: next_info.bet(det + 3 * BIG_BLIND_CHIP); break;
        case RAISE_HALF_POT: next_info.bet(det + pot / 2); break;
        case RAISE_POT: next_info.bet(det + pot); break;
        case RAISE_2POT: next_info.bet(det + pot * 2); break;
        default: next_info.bet(INF32); break;
    }

    next_info.player = next_player;
}