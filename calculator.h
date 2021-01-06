//
// Created by cothrax on 1/5/21.
//

#ifndef DEEPFOOL_CALCULATOR_H
#define DEEPFOOL_CALCULATOR_H

#include <unordered_map>
using std::unordered_map;
typedef long long ll;
typedef unsigned long long ull;
#define POWER_RANGE  1000 // TODO
#define MAX_POWER    9175040
#define MC_ITER      200

// 花色
// 牌的编码为 (花色*13 + 数字)，范围[0, 51]
// A 的编码为12而不是0 (可以改)
#define SPADE 0
#define DIAMOND 1
#define CLUB 2
#define HEART 3

class Calculator {
    bool mask[52];
    int cards[52];

private:
    // 你也许需要初始化一些东西？
    void init();
    unordered_map<ull, int> pp_cache;
    unordered_map<ull, int> opp_cache;

public:
    Calculator() { init(); }

    // 牌力
    // 底牌 holes, 5张公共牌 pubs
    // done
    int power(int *holes, int *pubs);

    // 玩家的先验期望牌力
    // 阶段 step (game.h)
    int potential_power(int *holes, int *pubs, int step);

    // 任一对手的先验期望牌力
    // 阶段 step (game.h)
    int opp_potential_power(int *holes, int *pubs, int step);

};

extern Calculator calculator;

#endif //DEEPFOOL_CALCULATOR_H
