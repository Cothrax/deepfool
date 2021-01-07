//
// Created by cothrax on 1/5/21.
//

#ifndef DEEPFOOL_PARAM_H
#define DEEPFOOL_PARAM_H

#include <string.h>


#define NUM_PLAYER 6

#define FOLD 0
#define CHECK 1
#define RAISE_3BB 6
#define RAISE_HALF_POT 2
#define RAISE_POT 4
#define RAISE_2POT 3
#define ALL_IN 5
#define NUM_ACTION 7


#define INIT_CHIP 20000
#define SMALL_BLIND_CHIP 50
#define BIG_BLIND_CHIP 100

#define INF32 0x3f3f3f3f

typedef double Strategy[NUM_ACTION];

#define decomp(x) ((x)/13+'a'),(x%13)


struct Game;
struct InfoSet
{
    int player;
    int num;
    int pot;
    int step;
    int cur_bet;
    int chips[NUM_PLAYER];
    int bets[NUM_PLAYER];
    bool folds[NUM_PLAYER];

    InfoSet() {}
    InfoSet(Game &game);
    void bet(int more);
};

struct Game
{
    int cards[52];
    int holes[NUM_PLAYER][2];
    int pubs[5];
    int pov;
    int start;
    int big_blind;
    int chips[NUM_PLAYER];
    int power[NUM_PLAYER];

    Game() { for(int i = 0; i < 52; i++) cards[i] = i; }
    void generate(int start, int pov);
    int change_state(InfoSet &info);
    void calc_result(InfoSet &info, double *util);
    void act(InfoSet &info, int action, InfoSet &next_info);
};


#define NO_CHANGE 0
#define PRE_FLOP 0
#define FLOP 1
#define TURN 2
#define RIVER 3
#define GAME_OVER 4
#define NUM_STATE 5

#endif //DEEPFOOL_PARAM_H