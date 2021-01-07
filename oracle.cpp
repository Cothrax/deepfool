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
    init_prob[FOLD]           = 1./6;
    init_prob[CHECK]          = 1./6;
    init_prob[RAISE_3BB]      = 1./6;
//    init_prob[RAISE_QUARTER_POT]= 1./4;
    init_prob[RAISE_HALF_POT] = 1./6;
    init_prob[RAISE_POT]      = 1./6;
    init_prob[RAISE_2POT]     = 1./6;
    init_prob[ALL_IN]         = 0;
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

inline bool comp(int a, int b) { return a%13 < b%13; }

ull NaiveOracleV2::encode(Game &game, InfoSet &info)
{
    int cards[7];
    int num = info.step == 0 ? 0 : info.step + 2;

    memcpy(cards, game.holes[info.player], sizeof(int) * 2);
    memcpy(cards + 2, game.holes[info.player], sizeof(int) * num);
    std::sort(cards, cards+2, comp);
    std::sort(cards+2, cards+2+num, comp);

//    for(int i = 0; i < num+2; i++) printf("<%c:%d>, ", decomp(cards[i]));

    int type[4] = {-1, -1, -1, -1};
    int cnt = 0;
    for(int i = 0; i < num+2; i++)
        if(type[cards[i]/13] == -1) type[cards[i]/13] = cnt++;
    for(int i = 0; i < num+2; i++)
        cards[i] = type[cards[i]/13]*13 + cards[i]%13;

//    std::sort(cards, cards+2);
//    std::sort(cards+2, cards+2+num);

//    printf(" => ");
//    for(int i = 0; i < num+2; i++) printf("<%c:%d>, ", decomp(cards[i]));
//    printf("\n");

    ull key = 0;
    for(int i = 0; i < num+2; i++) key = key * 52 + cards[i];

    int bet_num = 0;
    for(int i = 0; i < NUM_PLAYER; i++) bet_num += info.bets[i] != 0;

    key = ((key * NUM_PLAYER + bet_num) * NUM_PLAYER + info.num) * 4 + info.step;

    return key;
}

void NaiveOracleV2::decode(ull key, int *res)
{
    int step = key%4; key /= 4;
    int play_num = key%NUM_PLAYER; key /= NUM_PLAYER;
    int bet_num = key%NUM_PLAYER; key /= NUM_PLAYER;


    res[0] = step;
    res[1] = play_num;
    res[2] = bet_num;
    int num = step == 0 ? 0 : step + 2;
    for(int i = 0; i < num + 2; i++)
    {
        res[3+i] = key%52;
        key /= 52;
    }
}

void NaiveOracleV2::load(const char *fn)
{
    std::ifstream fin(fn, std::ios_base::in | std::ios_base::binary);
    int s; fin.read((char*)&s, sizeof(int));
    printf("node map size = %d\n", s);

    ull key; Node node;

    int res[20];
    Strategy strategy;
    for(int _ = 0; _ < s; _++) {
        fin.read((char *) &key, sizeof(key));
        fin.read((char *) &node, sizeof(node));
        node_map[key] = node;

        decode(key, res);
        int num = res[0] == 0 ? 0 : res[0] + 2;

        node.get_strategy(strategy, init_prob);

        if (res[0] == 0) {
            printf("(");
            for (int i = 0; i < 3; i++) printf("%d, ", res[i]);
            for (int i = 0; i < num + 2; i++) printf("<%c:%d>, ", decomp(res[3 + i]));
            printf(") => (");
            for (int i = 0; i < NUM_ACTION; i++) printf("%.3lf, ", strategy[i]);
            printf(")\n");
        }
    }
}

void DeepOracle::submit(InfoSet &info, Game &game, vector<pair<int, int>> &history)
{
    Wrap wrap;
    wrap.player = info.player;
    memcpy(wrap.holes, game.holes[info.player], sizeof(int) * 2);
    memcpy(wrap.pubs, game.pubs, sizeof(int) * 5);
    int num = info.step == 0 ? 0 : info.step + 2;
    for(int i = num; i < 5; i++) wrap.pubs[i] = -1;

    int ofs = std::max(0, ((int)history.size())-MAX_HIST_SIZE);
    for(int i = ofs; i < history.size(); i++)
    {
        wrap.history[i-ofs][0] = history[i].first;
        wrap.history[i-ofs][1] = history[i].second;
    }
    for(int i = history.size(); i < MAX_HIST_SIZE; i++)
    {
        wrap.history[i][0] = -1;
        wrap.history[i][1] = -1;
    }
    cache.push_back(wrap);
}

ull DeepOracle::single_query(Game &game, InfoSet &info, Strategy &strategy)
{

}

void DeepOracle::commit()
{
    commit_time = cache.size();
}

void DeepOracle::update(Game &game, ull node, double *regret)
{
    memcpy(regret, cache[node].regret, sizeof(double) * NUM_ACTION);
}

void DeepOracle::learn()
{

}

