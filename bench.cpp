#include "include/header"

#include <vector>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <string>

#include <omp.h>

#include "mle_sumcheck.h"
#include "timer.h"
#include "test.h"
#include "VGG16.h"
#include "VGG11.h"
#include "AlexNet.h"
#include "LeNet.h"
#include "counter.h"
#include "lazy_pcs.h"
#include "product2_sumcheck.h"
#include "orion.h"

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

int default_orion_loga_for_logm(int logn) {
    return orion_default_loga(logn);
}

uint64_t bench_splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

std::vector<uint64_t> deterministic_vec_uint(size_t n) {
    std::vector<uint64_t> vec(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        vec[i] = bench_splitmix64(i) % RAND_MAX;
    }
    return vec;
}

void bench_pcs_compare(int logn = 18, bool base_only = false, int orion_loga = -1) {
    constexpr size_t sec_param = 32;
    constexpr uint64_t rho_inv = 2;
    constexpr size_t orion_queries = 100;
    size_t n = 1ull << logn;

    std::cout << (base_only ? "PCS compare base-only, logn=" : "PCS compare, logn=")
              << logn << ", n=" << n << ", orion_loga=" << orion_loga << std::endl;
    srand(79);

    struct metrics {
        double commit_s = 0;
        double open_s = 0;
        double pcs_size_B = 0;
        double open_proof_B = 0;
        size_t rows = 0;
        size_t cols = 0;
        size_t code_len = 0;
        size_t queries = 0;
        size_t v_prime_B = 0;
        size_t column_B = 0;
        size_t path_B = 0;
        size_t codeswitch_B = 0;
        bool ok = false;
    };

    auto calculate_ligero_t = [](size_t security, uint64_t rate_inv, size_t codeword_len) {
        double residual = static_cast<double>(codeword_len) / std::pow(2.0, FIELD_BITS);
        double pr = std::pow(2.0, -static_cast<int>(security));
        double numerator = std::log2(pr - residual) - 1;
        double denominator = std::log2((1.0 + 1.0 / static_cast<double>(rate_inv)) * 0.5);
        size_t t = static_cast<size_t>(std::ceil(numerator / denominator));
        return std::min(t, codeword_len);
    };

    auto path_depth = [](size_t code_len) {
        return static_cast<size_t>(std::ceil(std::log2(static_cast<double>(code_len))));
    };

    auto fill_ligero_breakdown = [&](metrics& m, size_t rows, size_t cols, size_t code_len, size_t field_element_size) {
        m.rows = rows;
        m.cols = cols;
        m.code_len = code_len;
        m.queries = calculate_ligero_t(sec_param, rho_inv, code_len);
        m.v_prime_B = cols * sizeof(Goldilocks2::Element);
        m.column_B = rows * m.queries * field_element_size;
        m.path_B = path_depth(code_len) * m.queries * sizeof(MerkleDef::Digest);
    };

    auto fill_orion_breakdown = [&](metrics& m, size_t rows, size_t cols, size_t code_len, size_t field_element_size) {
        m.rows = rows;
        m.cols = cols;
        m.code_len = code_len;
        m.queries = std::min(orion_queries, code_len);
        m.v_prime_B = cols * sizeof(Goldilocks2::Element);
        m.column_B = rows * m.queries * field_element_size;
        m.path_B = path_depth(code_len) * m.queries * sizeof(MerkleDef::Digest);
    };

    auto print_metrics = [](const std::string& field, const std::string& scheme, const metrics& m) {
        std::cout << field << ","
                  << scheme << ","
                  << m.commit_s << ","
                  << m.open_s << ","
                  << m.pcs_size_B << ","
                  << m.open_proof_B << ","
                  << m.rows << ","
                  << m.cols << ","
                  << m.code_len << ","
                  << m.queries << ","
                  << m.v_prime_B << ","
                  << m.column_B << ","
                  << m.path_B << ","
                  << m.codeswitch_B << ","
                  << (m.ok ? "true" : "false") << std::endl;
    };

    auto fill_remainder = [](metrics& m) {
        size_t accounted = m.v_prime_B + m.column_B + m.path_B;
        m.codeswitch_B = m.open_proof_B > accounted ? static_cast<size_t>(m.open_proof_B) - accounted : 0;
    };

    auto run_ligero_ext = [&](const MLE& mle, const std::vector<Goldilocks2::Element>& challenge, Goldilocks2::Element expected) {
        metrics m;
        clear_proof();
        start_proof("ligero_commit_cmp");
        auto begin = std::chrono::steady_clock::now();
        auto pcs = ligero_commit_ext(mle, rho_inv);
        auto end = std::chrono::steady_clock::now();
        end_proof("ligero_commit_cmp");
        m.commit_s = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
        m.pcs_size_B = get_proof_size("ligero_commit_cmp", Counter::B);
        fill_ligero_breakdown(m, pcs.num_rows, pcs.num_cols, pcs.prover->codelen, sizeof(Goldilocks2::Element));

        clear_proof();
        start_proof("ligero_open_cmp");
        begin = std::chrono::steady_clock::now();
        Goldilocks2::Element val = pcs.open(challenge, sec_param);
        end = std::chrono::steady_clock::now();
        end_proof("ligero_open_cmp");
        m.open_s = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
        m.open_proof_B = get_proof_size("ligero_open_cmp", Counter::B);
        fill_remainder(m);
        m.ok = (val == expected);
        return m;
    };

    auto run_orion_ext = [&](const MLE& mle, const std::vector<Goldilocks2::Element>& challenge, Goldilocks2::Element expected) {
        metrics m;
        clear_proof();
        start_proof("orion_commit_cmp");
        auto begin = std::chrono::steady_clock::now();
        auto pcs = orion_commit_ext(mle, rho_inv, orion_loga);
        auto end = std::chrono::steady_clock::now();
        end_proof("orion_commit_cmp");
        m.commit_s = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
        m.pcs_size_B = get_proof_size("orion_commit_cmp", Counter::B);
        fill_orion_breakdown(m, pcs.num_rows, pcs.num_cols, pcs.prover->get_code_len(), sizeof(Goldilocks2::Element));

        clear_proof();
        start_proof("orion_open_cmp");
        begin = std::chrono::steady_clock::now();
        Goldilocks2::Element val = pcs.open(challenge, sec_param);
        end = std::chrono::steady_clock::now();
        end_proof("orion_open_cmp");
        m.open_s = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
        m.open_proof_B = get_proof_size("orion_open_cmp", Counter::B);
        m.v_prime_B = 0;
        fill_remainder(m);
        m.ok = (val == expected);
        return m;
    };

    auto run_ligero_base = [&](const MLE& mle, const std::vector<Goldilocks2::Element>& challenge, Goldilocks2::Element expected) {
        metrics m;
        clear_proof();
        start_proof("ligero_commit_cmp");
        auto begin = std::chrono::steady_clock::now();
        auto pcs = ligero_commit_base(mle, rho_inv);
        auto end = std::chrono::steady_clock::now();
        end_proof("ligero_commit_cmp");
        m.commit_s = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
        m.pcs_size_B = get_proof_size("ligero_commit_cmp", Counter::B);
        fill_ligero_breakdown(m, pcs.num_rows, pcs.num_cols, pcs.prover->codelen, sizeof(Goldilocks::Element));

        clear_proof();
        start_proof("ligero_open_cmp");
        begin = std::chrono::steady_clock::now();
        Goldilocks2::Element val = pcs.open(challenge, sec_param);
        end = std::chrono::steady_clock::now();
        end_proof("ligero_open_cmp");
        m.open_s = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
        m.open_proof_B = get_proof_size("ligero_open_cmp", Counter::B);
        fill_remainder(m);
        m.ok = (val == expected);
        return m;
    };

    auto run_orion_base = [&](const MLE& mle, const std::vector<Goldilocks2::Element>& challenge, Goldilocks2::Element expected) {
        metrics m;
        clear_proof();
        start_proof("orion_commit_cmp");
        auto begin = std::chrono::steady_clock::now();
        auto pcs = orion_commit_base(mle, rho_inv, orion_loga);
        auto end = std::chrono::steady_clock::now();
        end_proof("orion_commit_cmp");
        m.commit_s = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
        m.pcs_size_B = get_proof_size("orion_commit_cmp", Counter::B);
        fill_orion_breakdown(m, pcs.num_rows, pcs.num_cols, pcs.prover->get_code_len(), sizeof(Goldilocks::Element));

        clear_proof();
        start_proof("orion_open_cmp");
        begin = std::chrono::steady_clock::now();
        Goldilocks2::Element val = pcs.open(challenge, sec_param);
        end = std::chrono::steady_clock::now();
        end_proof("orion_open_cmp");
        m.open_s = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
        m.open_proof_B = get_proof_size("orion_open_cmp", Counter::B);
        m.v_prime_B = 0;
        fill_remainder(m);
        m.ok = (val == expected);
        return m;
    };

    clear_all_timers();

    metrics ligero_ext, orion_ext;
    if (!base_only) {
        std::vector<Goldilocks2::Element> ext_vec = random_vec_ext(n);
        MLE ext_mle(ext_vec);
        std::vector<Goldilocks2::Element> ext_challenge = random_vec_ext(logn);
        Goldilocks2::Element ext_expected = ext_mle.open(ext_challenge, sec_param);
        ligero_ext = run_ligero_ext(ext_mle, ext_challenge, ext_expected);
        orion_ext = run_orion_ext(ext_mle, ext_challenge, ext_expected);
    }

    std::vector<uint64_t> base_vec = random_vec_uint(n);
    MLE base_mle(base_vec);
    std::vector<Goldilocks2::Element> base_challenge = random_vec_ext(logn);
    Goldilocks2::Element base_expected = base_mle.open(base_challenge, sec_param);
    auto ligero_base = run_ligero_base(base_mle, base_challenge, base_expected);
    auto orion_base = run_orion_base(base_mle, base_challenge, base_expected);

    std::cout << "field,scheme,commit_s,open_s,pcs_size_B,open_proof_B,rows,cols,code_len,queries,direct_v_prime_B,column_B,path_B,codeswitch_B,open_ok" << std::endl;
    if (!base_only) {
        print_metrics("ext", "Ligero", ligero_ext);
        print_metrics("ext", "Orion", orion_ext);
    }
    print_metrics("base", "Ligero", ligero_base);
    print_metrics("base", "Orion", orion_base);

    print_all_timers();
    clear_all_timers();
    clear_proof();
}

void bench_orion_loga_sweep_base(int logn_min = 20,
                                 int logn_max = 30,
                                 int below_default = 3,
                                 int above_default = 1) {
    constexpr size_t sec_param = 32;
    constexpr uint64_t rho_inv = 2;
    constexpr size_t orion_queries = 100;
    srand(79);

    auto path_depth = [](size_t code_len) {
        return static_cast<size_t>(std::ceil(std::log2(static_cast<double>(code_len))));
    };

    struct metrics {
        double commit_s = 0;
        double open_s = 0;
        double pcs_size_B = 0;
        double open_proof_B = 0;
        size_t rows = 0;
        size_t cols = 0;
        size_t code_len = 0;
        size_t queries = 0;
        size_t column_B = 0;
        size_t path_B = 0;
        size_t codeswitch_B = 0;
        bool ok = false;
    };

    auto fill_orion_breakdown = [&](metrics& m, size_t rows, size_t cols, size_t code_len) {
        m.rows = rows;
        m.cols = cols;
        m.code_len = code_len;
        m.queries = std::min(orion_queries, code_len);
        m.column_B = rows * m.queries * sizeof(Goldilocks::Element);
        m.path_B = path_depth(code_len) * m.queries * sizeof(MerkleDef::Digest);
        size_t accounted = m.column_B + m.path_B;
        m.codeswitch_B = m.open_proof_B > accounted ? static_cast<size_t>(m.open_proof_B) - accounted : 0;
    };

    std::cout << "logn,loga,default_loga,commit_s,open_s,prover_s,pcs_size_B,open_proof_B,rows,cols,code_len,queries,column_B,path_B,codeswitch_B,open_ok" << std::endl;
    for (int logn = logn_min; logn <= logn_max; ++logn) {
        size_t n = 1ull << logn;
        int default_loga = default_orion_loga_for_logm(logn);
        int loga_min = std::max(1, default_loga - below_default);
        int loga_max = std::min(logn, default_loga + above_default);

        std::vector<uint64_t> base_vec = deterministic_vec_uint(n);
        std::vector<Goldilocks2::Element> challenge = random_vec_ext(logn);
        Goldilocks2::Element expected = Goldilocks2::zero();
        bool has_expected = false;

        for (int loga = loga_min; loga <= loga_max; ++loga) {
            clear_all_timers();
            clear_proof();
            metrics m;

            start_proof("orion_commit_sweep");
            auto begin = std::chrono::steady_clock::now();
            auto pcs = orion_commit_base(base_vec, rho_inv, loga);
            auto end = std::chrono::steady_clock::now();
            end_proof("orion_commit_sweep");
            m.commit_s = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
            m.pcs_size_B = get_proof_size("orion_commit_sweep", Counter::B);

            clear_proof();
            start_proof("orion_open_sweep");
            begin = std::chrono::steady_clock::now();
            Goldilocks2::Element val = pcs.open(challenge, sec_param);
            end = std::chrono::steady_clock::now();
            end_proof("orion_open_sweep");
            m.open_s = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
            m.open_proof_B = get_proof_size("orion_open_sweep", Counter::B);
            if (!has_expected) {
                expected = val;
                has_expected = true;
            }
            m.ok = (val == expected);
            fill_orion_breakdown(m, pcs.num_rows, pcs.num_cols, pcs.prover->get_code_len());

            std::cout << logn << ","
                      << loga << ","
                      << default_loga << ","
                      << m.commit_s << ","
                      << m.open_s << ","
                      << (m.commit_s + m.open_s) << ","
                      << m.pcs_size_B << ","
                      << m.open_proof_B << ","
                      << m.rows << ","
                      << m.cols << ","
                      << m.code_len << ","
                      << m.queries << ","
                      << m.column_B << ","
                      << m.path_B << ","
                      << m.codeswitch_B << ","
                      << (m.ok ? "true" : "false") << std::endl;
            std::cout.flush();
        }
    }
    clear_all_timers();
    clear_proof();
}

#define NUM_THREADS 8

bool is_decimal_integer(const std::string& value) {
    if (value.empty()) return false;
    size_t start = value[0] == '-' ? 1 : 0;
    if (start == value.size()) return false;
    for (size_t i = start; i < value.size(); ++i) {
        if (value[i] < '0' || value[i] > '9') return false;
    }
    return true;
}

bool run_model_benchmark(const std::string& model_name) {
    if (model_name == "VGG16") {
        bench_VGG16();
    } else if (model_name == "VGG11") {
        bench_VGG11();
    } else if (model_name == "AlexNet") {
        bench_AlexNet();
    } else if (model_name == "LeNet") {
        bench_LeNet();
    } else {
        std::cerr << "Unknown model name: " << model_name << std::endl;
        return false;
    }
    return true;
}

void print_bench_usage(const char* argv0) {
    std::cout << "Usage:\n"
              << "  " << argv0 << " <threads> <LeNet|AlexNet|VGG11|VGG16>\n"
              << "  " << argv0 << " lazy_pcs_smoke [threads]\n"
              << "  " << argv0 << " pcs_compare <logn> <threads> [orion_loga]\n"
              << "  " << argv0 << " pcs_compare_base <logn> <threads> [orion_loga]\n"
              << "  " << argv0 << " pcs_orion_loga_sweep_base <logn_min> <logn_max> <threads> <below_default> <above_default>\n";
}

int main(int argc, char** argv) {
    if (argc >= 2 && (std::string(argv[1]) == "pcs_compare" || std::string(argv[1]) == "pcs_compare_base")) {
        int logn = (argc >= 3) ? std::atoi(argv[2]) : 18;
        int threads = (argc >= 4) ? std::atoi(argv[3]) : NUM_THREADS;
        int orion_loga = (argc >= 5) ? std::atoi(argv[4]) : -1;
        omp_set_num_threads(threads);
        bench_pcs_compare(logn, std::string(argv[1]) == "pcs_compare_base", orion_loga);
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "pcs_orion_loga_sweep_base") {
        int logn_min = (argc >= 3) ? std::atoi(argv[2]) : 20;
        int logn_max = (argc >= 4) ? std::atoi(argv[3]) : 30;
        int threads = (argc >= 5) ? std::atoi(argv[4]) : NUM_THREADS;
        int below_default = (argc >= 6) ? std::atoi(argv[5]) : 3;
        int above_default = (argc >= 7) ? std::atoi(argv[6]) : 1;
        omp_set_num_threads(threads);
        bench_orion_loga_sweep_base(logn_min, logn_max, below_default, above_default);
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "lazy_pcs_smoke") {
        int threads = (argc >= 3) ? std::atoi(argv[2]) : NUM_THREADS;
        omp_set_num_threads(threads);
        std::cout << "PCS backend: " << pcs_backend_name(default_pcs_backend()) << std::endl;
        bench_lazypcs();
        std::cout << "lazy_pcs_smoke ok" << std::endl;
        return 0;
    }
    if (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        print_bench_usage(argv[0]);
        return 0;
    }
    // if (!run_test()) return 0;
    // find_parameter();
    if (argc >= 2) {
        int threads = NUM_THREADS;
        std::string model_name;
        if (is_decimal_integer(argv[1])) {
            threads = std::atoi(argv[1]);
            model_name = (argc >= 3) ? std::string(argv[2]) : "AlexNet";
        } else {
            model_name = std::string(argv[1]);
        }
        omp_set_num_threads(threads);
        std::cout << "PCS backend: " << pcs_backend_name(default_pcs_backend()) << std::endl;
        return run_model_benchmark(model_name) ? 0 : 1;
    }
    print_bench_usage(argv[0]);
    return 0;
    // bench_AlexNet();
    // bench_VGG11();
    // bench_VGG16();

    // bench_p2_sumcheck();
    // bench_lazypcs();
    // bench_commit();
    // bench_transpose();
    return 0;
}
