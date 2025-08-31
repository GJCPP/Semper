#include "include/header"

#include <vector>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <chrono>

#include "timer.h"
#include "test.h"
#include "VGG16.h"
#include "VGG11.h"
#include "counter.h"

void bench_VGG16() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    VGG16 VGG16("/home/gaojc/Desktop/zkCNN/training_trace/", 0, 1 << 14, 1 << 24, 2);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cout << "Loading VGG16 took " << duration.count() / 1000000.0 << " s" << std::endl;

    // VGG16.check();
    VGG16.prove(32);
}

void bench_VGG11() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    VGG11 VGG11("/home/gaojc/Desktop/zkCNN/training_trace/", 0, 1 << 14, 1 << 24, 2);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cout << "Loading VGG11 took " << duration.count() / 1000000.0 << " s" << std::endl;

    // VGG16.check();
    start_proof("VGG11");
    VGG11.prove(32);
    end_proof("VGG11");
    print_all_proof_size(Counter::MB);
    clear_proof();
}

int main() {
    // run_test();

    bench_VGG11();
    // {
    //     std::vector<Goldilocks2::Element> vec(1ull << 20, Goldilocks2::one());
    //     std::vector<Goldilocks2::Element> cha(20);

    //     auto pcs = ligero_commit_base(vec, 2);
    //     clear_proof();

    //     startCounter counter("open");
    //     pcs.open(cha, 32);
    //     print_all_proof_size(Counter::MB);
    // }
    return 0;
}
