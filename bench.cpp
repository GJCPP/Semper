#include "include/header"

#include <vector>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <chrono>

#include <omp.h>

#include "mle_sumcheck.h"
#include "timer.h"
#include "test.h"
#include "VGG16.h"
#include "VGG11.h"
#include "AlexNet.h"
#include "LeNet.h"
#include "counter.h"
#include "product2_sumcheck.h"

const std::string DATA_DIR = "./training_trace/";

void bench_VGG16() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    VGG16 VGG16(DATA_DIR, 0, 1 << 14, 1 << 24, 2, 32);
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
    VGG11 VGG11(DATA_DIR, 0, 1 << 14, 1 << 27, 2, 32);
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

void bench_AlexNet() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    AlexNet AlexNet(DATA_DIR, 0, 1 << 14, 1 << 27, 2, 32);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cout << "Loading AlexNet took " << duration.count() / 1000000.0 << " s" << std::endl;

    // VGG11.check(100);
    // return;
    // print_all_proof_size(Counter::MB);
    clear_proof();
    start_proof("AlexNet");
    AlexNet.prove(32);
    end_proof("AlexNet");
    print_all_timers();
    clear_all_timers();
    std::cout << "-----------------------------" << std::endl;
    print_all_proof_size(Counter::MB);
    std::cout << "-----------------------------" << std::endl;
    print_all_open_cnt();
    clear_proof();
}

void bench_LeNet() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    LeNet LeNet(DATA_DIR, 0, 1 << 14, 1 << 27, 2, 32);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cout << "Loading LeNet took " << duration.count() / 1000000.0 << " s" << std::endl;

    // VGG11.check(100);
    // return;
    // print_all_proof_size(Counter::MB);
    clear_proof();
    start_proof("LeNet");
    LeNet.prove(32);
    end_proof("LeNet");
    print_all_timers();
    clear_all_timers();
    std::cout << "-----------------------------" << std::endl;
    print_all_proof_size(Counter::MB);
    std::cout << "-----------------------------" << std::endl;
    print_all_open_cnt();
    clear_proof();
}

void bench_transpose() {
    size_t num_rows(1ull << 12), num_cols(1ull << 18);

    // Flat transpose
    set_timer("allocate");
    std::vector<Goldilocks2::Element> flat_all(num_rows * num_cols);
    std::vector<Goldilocks2::Element> flat_transposed(num_rows * num_cols);
    pause_timer("allocate");

    const size_t BLOCK = 16; // tune for cache

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
    for (int i = 31; i <= 31; ++i) {
        std::cout << "i = " << i << std::endl;
        set_timer("allocation");
        std::vector<Goldilocks2::Element> vec = random_vec_ext(1ull << i);
        pause_timer("allocation");
        set_timer("copy");
        MLE mle(vec);
        pause_timer("copy");
        
        print_all_timers();
        clear_all_timers();
        std::cout << std::endl;

        set_timer("commit");
        auto pcs = ligero_commit_base(mle, 2);
        pause_timer("commit");

        print_all_timers();
        clear_all_timers();

        set_timer("open");
        pcs.open(random_vec_ext(i), 32);
        pause_timer("open");
        
        print_all_timers();
        clear_all_timers();
    }
}

void bench_p2_sumcheck() {
    size_t num_vars = 30;
    size_t n = 1ull << num_vars;
    set_timer("allocate");
    std::vector<Goldilocks2::Element> v1(n), v2(n);
    auto p1 = std::make_unique<MultilinearPolynomial>(v1);
    auto p2 = std::make_unique<MultilinearPolynomial>(v2);

    pause_timer("allocate");
    print_all_timers();
    clear_all_timers();

    // set_timer("commit");
    // auto oracle1 = std::make_shared<ligeropcs_base>(ligero_commit_base(v1, 2));
    // auto oracle2 = std::make_shared<ligeropcs_base>(ligero_commit_base(v2, 2));
    // pause_timer("commit");

    MLE mle1 = v1, mle2 = v2;

    // print_all_timers();
    // clear_all_timers();

    p2Prover prover(std::move(p1), std::move(p2));

    size_t sec_param = 32;
    
    set_timer("execute partial sumcheck");
    auto result = p2Verifier::partial_sumcheck(prover, sec_param);
    // auto result = p2Verifier::execute_sumcheck(prover, {&mle1, &mle2}, sec_param);
    pause_timer("execute partial sumcheck");

    print_all_timers();
    clear_all_timers();

    if (result) {
        if (result->claim == mle1.open(result->challenges, 32) * mle2.open(result->challenges, 32)) {
            std::cout << "partial_sumcheck succeeded." << std::endl;
            return;
        }
    }
    std::cout << "partial_sumcheck failed." << std::endl;
}

void bench_lazypcs() {
    const int n = 2;
    std::array<MLE, n> mles;
    std::array<lazy_pcs, n> pcss;
    auto pcs_pool = lazy_pcs_pool::create(32, false);
    for (int i = 0; i != n; ++i) {
        std::vector<Goldilocks2::Element> vec(1 << (rand() % 10 + 4));
        random_vec_u64(vec.data(), vec.size());
        mles[i] = vec;
        pcss[i] = commit_lazy_pcs(mles[i], pcs_pool);
    }
    auto uni_pcs = pcs_pool->commit(2);
    for (int i = 0; i != 100; ++i) {
        int ind = rand() % n;
        int num_vars = mles[ind].get_num_vars();
        pcss[ind].open(random_vec_ext(num_vars), 32);
    }
    pcs_pool->prove_open(uni_pcs, random_ext());
    
}

#define NUM_THREADS 8

int main(int argc, char** argv) {
    // if (!run_test()) return 0;
    // find_parameter();
    // std::string model_name = "AlexNet";
    // if (argc >= 2) {
    //     omp_set_num_threads(std::atoi(argv[1]));
    //     model_name = (argc >= 3) ? std::string(argv[2]) : model_name;
    // } else {
    //     omp_set_num_threads(NUM_THREADS);
    // }
    // if (model_name == "VGG16") {
    //     bench_VGG16();
    // } else if (model_name == "VGG11") {
    //     bench_VGG11();
    // } else if (model_name == "AlexNet") {
    //     bench_AlexNet();
    // } else if (model_name == "LeNet") {
    //     bench_LeNet();
    // } else {
    //     std::cerr << "Unknown model name: " << model_name << std::endl;
    //     return 1;
    // }
    // bench_AlexNet();
    // bench_VGG11();
    // bench_VGG16();

    // bench_p2_sumcheck();
    // bench_lazypcs();
    bench_commit();
    // bench_transpose();
    return 0;
}

