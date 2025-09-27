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
    VGG16 VGG16("/home/gaojc/Desktop/zkCNN/training_trace/", 0, 1 << 14, 1 << 24, 2, 32);
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
    // VGG11 VGG11("/home/gaojc/Desktop/zkCNN/training_trace", 0, 1 << 14, 1 << 24, 2, 32);
    VGG11 VGG11("../training_trace", 0, 1 << 14, 1 << 27, 2, 32);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cout << "Loading VGG11 took " << duration.count() / 1000000.0 << " s" << std::endl;
    
    // VGG11.check(100);
    // return;
    // print_all_proof_size(Counter::MB);
    clear_proof();
    start_proof("VGG11");
    VGG11.prove(32);
    end_proof("VGG11");
    print_all_timers();
    clear_all_timers();
    std::cout << "-----------------------------" << std::endl;
    print_all_proof_size(Counter::MB);
    std::cout << "-----------------------------" << std::endl;
    print_all_open_cnt();
    clear_proof();
}

void bench_transpose() {
    size_t num_rows(1ull << 13), num_cols(1ull << 19);

    // Plain vector<vector<>>
    std::vector<std::vector<Goldilocks2::Element>> all(num_rows, std::vector<Goldilocks2::Element>(num_cols));
    set_timer("plain");
    std::vector<std::vector<Goldilocks2::Element>> transposed(num_cols, std::vector<Goldilocks2::Element>(num_rows));
    #pragma omp parallel for
    for (size_t i = 0; i != num_rows; ++i) {
        for (size_t j = 0; j != num_cols; ++j) {
            transposed[j][i] = all[i][j];
        }
    }
    pause_timer("plain");
    std::cout << "plain done." << std::endl;

    all.clear();
    transposed.clear();

    // Flat transpose
    std::vector<Goldilocks2::Element> flat_all(num_rows * num_cols);
    std::vector<Goldilocks2::Element> flat_transposed(num_rows * num_cols);

    const size_t BLOCK = 128; // tune for cache

    set_timer("flat_blocked");

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i0 = 0; i0 < num_rows; i0 += BLOCK) {
        for (size_t j0 = 0; j0 < num_cols; j0 += BLOCK) {
            size_t i_max = std::min(i0 + BLOCK, num_rows);
            size_t j_max = std::min(j0 + BLOCK, num_cols);
            for (size_t i = i0; i < i_max; ++i) {
                const size_t row_offset = i * num_cols;
                for (size_t j = j0; j < j_max; ++j) {
                    flat_transposed[j * num_rows + i] = flat_all[row_offset + j];
                }
            }
        }
    }

    pause_timer("flat_blocked");
    std::cout << "blocked flat done." << std::endl;

    print_all_timers();
}


void bench_commit() {
    clear_all_timers();
    for (int i = 32; i <= 32; ++i) {
        std::cout << "i = " << i << std::endl;
        set_timer("allocation");
        std::vector<Goldilocks2::Element> vec(1ull << i);
        pause_timer("allocation");
        set_timer("copy");
        MLE mle(vec);
        pause_timer("copy");
        
        print_all_timers();
        clear_all_timers();
        std::cout << std::endl;

        start_proof("commit/open");
        set_timer("commit/open");
        auto pcs = ligero_commit_base(mle, 2);
        pcs.open(random_vec_ext(i), 32);
        end_proof("commit/open");
        pause_timer("commit/open");
        print_all_proof_size(Counter::B);
        print_all_timers();
        std::cout << std::endl;
        clear_all_timers();
        clear_proof();
    }
}

int main() {
    // if (!run_test()) return 0;
    // find_parameter();
    
    // bench_VGG11();
    bench_commit();
    // bench_transpose();
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
// int main() {
//     // if (!run_test()) return 0;
//     // find_parameter();
//     bench_VGG11();
//     // bench_commit();
//     // bench_VGG16();
//     // std::vector<Goldilocks2::Element> z = random_vec_ext(1 << 24);
//     // MLE mle = z;
//     // auto pcs = ligero_commit_ext(mle, 2);
//     // auto cha = random_vec_ext(24);
//     // set_timer("pcs_open");
//     // pcs.open(cha, 32);
//     // pause_timer("pcs_open");
//     // set_timer("mle_open");
//     // mle.open(cha, 32);
//     // pause_timer("mle_open");
//     // print_all_timers();
//     // for (auto& [key, value] : lazy_pcs_open_cnt) {
//     //     std::cout << "lazy_pcs index " << key << " opened " << value << " times" << std::endl;
//     // }
//     return 0;
// }
