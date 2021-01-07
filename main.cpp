#include <iostream>
#include<algorithm>
#include "cfr.h"
#include "oracle.h"
#include "calculator.h"

long long debug_counter;
int debug_max_dep;
NaiveOracleV2 oracle;
//CFR cfr(&oracle);
CFR cfr;



void test_calculator()
{
    int cards[52];
    for(int i = 0; i < 52; i++) cards[i] = i;

//    while(1)
//    {
//        std::random_shuffle(cards, cards+52);
//        for(int i = 0; i < 7; i++) printf("(%d, %d)\n", cards[i]/13, cards[i]%13);
//        printf("power = %d", calculator.power(cards, cards+2));
//        getchar();
//    }
//    while(1)
//    {
//        for(int i = 0; i < 7; i++)
//        {
//            int p, q;
//            scanf("%d%d", &p, &q);
//            cards[i] = p * 13 + q;
//        }
//        printf("power = %d\n", calculator.power(cards, cards+2));
//    }

    while(1)
    {
        int step; scanf("%d", &step);
        int num = (step == 0 ? 0 : 2 + step) + 2;
        for(int i = 0; i < num; i++)
        {
            int p, q;
            scanf("%d%d", &p, &q);
            cards[i] = p * 13 + q;
        }

        for(int i = 0; i < 100; i++)
        {
            printf("my:%d, opp:%d\n",
                   calculator.potential_power(cards, cards+2, step),
                   calculator.opp_potential_power(cards, cards+2, step));
        }
    }
}

void play_with_oracle()
{
    oracle.load("naive_oracle_v2_10000000.dat");
    for(int _ = 0; _ < 100; _++)
    {
        Game game;
        game.generate(0, 0);
        InfoSet info(game);

        while(game.change_state(info) != GAME_OVER)
        {
            printf("step = %d, player = %d, pot = %d\n"
                   "\tchip = (%d, %d, %d, %d, %d, %d)\n"
                   "\tbet = (%d, %d, %d, %d ,%d ,%d)\n"
                   "\tfold = (%d, %d, %d, %d, %d, %d)\n",
                   info.step, info.player, info.pot,
                   info.chips[0], info.chips[1], info.chips[2], info.chips[3], info.chips[4], info.chips[5],
                   info.bets[0], info.bets[1], info.bets[2], info.bets[3], info.bets[4], info.bets[5],
                   info.folds[0], info.folds[1], info.folds[2], info.folds[3], info.folds[4], info.folds[5]);

            int a;
            if(info.player == 0) scanf("%d", &a);
            else
            {
                Strategy strategy;
                oracle.single_query(game, info, strategy);

                int i; double r = (rand()&32767)/32768.0;
                double cum = 0;
                for(i = 0; i < NUM_ACTION; i++)
                {
                    cum += strategy[i];
                    if(r < cum) break;
                }
                a = i;
                printf("action: %d\n", a);
            }

            game.act(info, a, info);
        }

        double util[NUM_PLAYER];
        game.calc_result(info, util);
        for(int j = 0; j < NUM_PLAYER; j++) printf("(%d): %.3lf\n", j, util[j]);
        printf("\n\n");
    }
}

int main() {
//    test_calculator();
//
//    test_oracle();
//    return 0;

//    play_with_oracle();
//    return 0;

    time_t now = time(0);

    cfr.train(10000000);
    std::cout << "COST: " << time(0) - now << " sec" << std::endl;
    oracle.dump("naive_oracle_v2_10000000.dat");

    return 0;
}