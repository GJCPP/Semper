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
    start_proof("VGG16");
    VGG16.prove(32);
    end_proof("VGG16");
    std::cout << "-----------------------------" << std::endl;
    print_all_proof_size(Counter::MB);
    std::cout << "-----------------------------" << std::endl;
    clear_proof();
}

void bench_VGG11() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    VGG11 VGG11("/home/gaojc/Desktop/zkCNN/training_trace/", 0, 1 << 14, 1 << 24, 2);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cout << "Loading VGG11 took " << duration.count() / 1000000.0 << " s" << std::endl;

    print_all_proof_size(Counter::MB);
    // VGG16.check();
    start_proof("VGG11");
    VGG11.prove(32);
    end_proof("VGG11");
    print_all_timers();
    clear_all_timers();
    std::cout << "-----------------------------" << std::endl;
    print_all_proof_size(Counter::MB);
    std::cout << "-----------------------------" << std::endl;
    clear_proof();
}

int main() {
    run_test();

    // find_parameter();
    // bench_VGG11();
    // bench_VGG16();
    // std::vector<Goldilocks2::Element> z = random_vec_ext(1 << 24);
    // MLE mle = z;
    // auto pcs = ligero_commit_ext(mle, 2);
    // auto cha = random_vec_ext(24);
    // set_timer("pcs_open");
    // pcs.open(cha, 32);
    // pause_timer("pcs_open");
    // set_timer("mle_open");
    // mle.open(cha, 32);
    // pause_timer("mle_open");
    // print_all_timers();
    // for (auto& [key, value] : lazy_pcs_open_cnt) {
    //     std::cout << "lazy_pcs index " << key << " opened " << value << " times" << std::endl;
    // }
    return 0;
}
