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
    VGG11 VGG11("/home/gaojc/Desktop/zkCNN/training_trace/", 0, 1 << 14, 1 << 24, 2, 32);
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

void bench_commit() {
    int tot_n = 27, bat = 3;
    set_timer("small commit");
    for (int i = 0; i < (1 << bat); i++) {
        auto pcs = ligero_commit_base(random_vec_ext(1ull << (tot_n - bat)), 2);
        auto cha = random_vec_ext(tot_n - bat);
        pcs.open(cha, 32);
    }
    pause_timer("small commit");
    set_timer("large commit");
    auto pcs = ligero_commit_base(random_vec_ext(1ull << tot_n), 2);
    auto cha = random_vec_ext(tot_n);
    pcs.open(cha, 32);
    pause_timer("large commit");
    print_all_timers();
    clear_all_timers();
}
#include <cstdio>
#include <chrono>
#include "Orion/include/linear_code/linear_code_encode.h"
#include "Orion/include/VPD/linearPC.h"
#define timer_mark(x) auto x = std::chrono::high_resolution_clock::now()
#define time_diff(start, end) (std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count())
int main()
{
    int N, lg_N = 20;
    N = 1 << lg_N;

    for (int i = 0; i != 10; ++i) {
        expander_init(N / column_size);

        prime_field::field_element *coefs = new prime_field::field_element[N];
        for(int i = 0; i < N; ++i)
            coefs[i] = prime_field::random();

        timer_mark(commit_t0);
        auto h = commit(coefs, N);
        timer_mark(commit_t1);

        auto r = new prime_field::field_element[lg_N];
        for (int i = 0; i < lg_N; ++i)
        {
            r[i] = prime_field::random();
        }

        timer_mark(open_t0);
        auto result = open_and_verify(r, lg_N, N, h);
        timer_mark(open_t1);
        printf("Commit time %lf\n", time_diff(commit_t0, commit_t1));
        printf("Open time %lf\n", time_diff(open_t0, open_t1));
        printf("%s\n\n\n", result.second ? "succ" : "fail");
        delete [] coefs;
        delete [] r;
    }
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
