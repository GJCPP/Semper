#include "include/header"

#include <vector>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <chrono>

#include "timer.h"
#include "test.h"
#include "VCG16.h"

int main() {
    // run_test();

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    VCG16 vcg16("/home/gaojc/Desktop/zkCNN/training_trace/", 0, 1 << 14, 1 << 24, 2);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cout << "Loading VCG16 took " << duration.count() / 1000000.0 << " s" << std::endl;

    // vcg16.check();
    vcg16.prove(32);

    return 0;
}
