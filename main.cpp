#include <iostream>
#include "cfr.h"
#include "oracle.h"

long long debug_counter;
int debug_max_dep;
NaiveOracle oracle;
CFR cfr(&oracle);

int main() {
    std::cout << "Hello, World!" << std::endl;

    cfr.train(1);
    printf("counter = %lld, dep = %d\n", debug_counter, debug_max_dep);

    return 0;
}