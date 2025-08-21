#include "test.h"
#include "util.h"
#include "product2_sumcheck.h"
#include "product3_sumcheck.h"
#include "logup.h"
#include "mle_pow.h"
#include "conv_check.h"
#include "array_view.h"
#include "pad_check.h"
#include "mat_check.h"
#include "div_check.h"
#include "sign_check.h"
#include "prod_check.h"
#include "ltn_check.h"
#include "e_pow_check.h"
#include "perm_check.h"

#define CNT_TEST 100

bool test_arithmetic() {
    typedef Goldilocks2::Element Element;
    for (int cnt(0); cnt != CNT_TEST; ++cnt) {
        Element a = random_ext(), b = random_ext();
        Element res = a + b;
        Element res2;
        Goldilocks2::add(res2, a, b);
        if (res != res2) {
            return false;
        }
        res = a * b;
        Goldilocks2::mul(res2, a, b);
        if (res != res2) {
            return false;
        }
    }
    return true;
}

bool test_eval_power_mle() {
    typedef Goldilocks2::Element Element;

    for (int cnt(0); cnt != CNT_TEST; ++cnt) {
        int l = (rand() % 10 + 1);
        int u = (rand() % (1 << l));
        std::vector<Element> r = random_vec_ext(l);
        Element beta = random_ext(), beta_pow = Goldilocks2::one();

        // beta^0, beta^1, ..., beta^(u), 0, ..., 0
        std::vector<Element> beta_pow_n;
        for (int i = 0; i <= u; i++) {
            beta_pow_n.push_back(beta_pow);
            beta_pow = beta_pow * beta;
        }
        for (int i = u + 1; i < (1 << l); ++i) {
            beta_pow_n.push_back(Goldilocks2::zero());
        }

        Element res = eval_power_mle(beta, r, u, l);
        MultilinearPolynomial mle(beta_pow_n);
        if (mle.evaluate(r) != res) {
            return false;
        }
    }
    return true;
}

bool test_product3_sumcheck() {
    typedef Goldilocks2::Element Element;
    int inv_rho = 2, sec_param = 32;
    for (int cnt(0); cnt != CNT_TEST; ++cnt) {
        int l = (rand() % 10 + 1);
        // Generate random polynomials and their commitments
        std::vector<Element> poly1_vec = random_vec_ext(1 << l);
        std::vector<Element> poly2_vec = random_vec_ext(1 << l);
        std::vector<Element> poly3_vec = random_vec_ext(1 << l);
        MultilinearPolynomial poly1(poly1_vec);
        MultilinearPolynomial poly2(poly2_vec);
        MultilinearPolynomial poly3(poly3_vec);
        ligeropcs_ext pcs1 = ligeroProver_ext(poly1, inv_rho).commit();
        ligeropcs_ext pcs2 = ligeroProver_ext(poly2, inv_rho).commit();
        ligeropcs_ext pcs3 = ligeroProver_ext(poly3, inv_rho).commit();

        // Run product3 sumcheck
        p3Prover prover(poly1, poly2, poly3);
        p3Verifier verifier;
        std::array<const oracle*, 3> oracle = { &pcs1, &pcs2, &pcs3 };
        if (!verifier.execute_sumcheck(prover, oracle, sec_param)) {
            return false;
        }
    }
    return true;
}

bool test_product2_sumcheck() {
    typedef Goldilocks2::Element Element;
    int inv_rho = 2, sec_param = 32;
    for (int cnt(0); cnt != CNT_TEST; ++cnt) {
        int l = (rand() % 10 + 1);
        // Generate random polynomials and their commitments
        std::vector<Element> poly1_vec = random_vec_ext(1 << l);
        std::vector<Element> poly2_vec = random_vec_ext(1 << l);
        MultilinearPolynomial poly1(poly1_vec);
        MultilinearPolynomial poly2(poly2_vec);
        ligeropcs_ext pcs1 = ligeroProver_ext(poly1, inv_rho).commit();
        ligeropcs_ext pcs2 = ligeroProver_ext(poly2, inv_rho).commit();

        // Run product3 sumcheck
        p2Prover prover(std::make_unique<MultilinearPolynomial>(poly1), std::make_unique<MultilinearPolynomial>(poly2));
        p2Verifier verifier;
        std::array<const oracle*, 2> oracle = { &pcs1, &pcs2 };
        if (!verifier.execute_sumcheck(prover, oracle, sec_param)) {
            return false;
        }
    }
    for (int cnt(0); cnt != CNT_TEST; ++cnt) {
        int l = (rand() % 10 + 1);
        int u = (rand() % (1 << l));
        // Generate random polynomials and their commitments
        std::vector<Element> poly1_vec = random_vec_ext(1 << l);
        std::vector<Element> poly2_vec = random_vec_ext(1 << l);
        MultilinearPolynomial poly1(poly1_vec);
        MLE_Pow poly2(random_ext(), l, u, true);
        ligeropcs_ext pcs1 = ligeroProver_ext(poly1, inv_rho).commit();

        // Run product3 sumcheck
        p2Prover prover(std::make_unique<MultilinearPolynomial>(poly1), std::make_unique<MLE_Pow>(poly2));
        p2Verifier verifier;
        std::array<const oracle*, 2> oracle = { &pcs1, &poly2 };
        if (!verifier.execute_sumcheck(prover, oracle, sec_param)) {
            return false;
        }
    }
    return true;
}

bool test_partial_sumcheck_product2() {
    typedef Goldilocks2::Element Element;
    int inv_rho = 2, sec_param = 32;
    for (int cnt(0); cnt != CNT_TEST; ++cnt) {
        int l = (rand() % 7 + 1);
        // Generate random polynomials and their commitments
        std::vector<Element> f_vec = random_vec_ext(1 << (2 * l));
        std::vector<Element> g_vec = random_vec_ext(1 << l);
        MultilinearPolynomial f(f_vec);
        MultilinearPolynomial fp(f.sum_over_lowbits(l));
        MultilinearPolynomial g(g_vec);

        // Create commitments
        ligeropcs_ext pcs = ligeroProver_ext(f_vec, inv_rho).commit();
        ligeropcs_ext pcs2 = ligeroProver_ext(g_vec, inv_rho).commit();

        // Run partial sumcheck
        p2Prover prover(std::make_unique<MultilinearPolynomial>(fp), std::make_unique<MultilinearPolynomial>(g));
        p2Verifier verifier;
        auto result = verifier.partial_sumcheck(prover, sec_param);
        if (!result.has_value()) {
            return false;
        }
        // check the claim
        // Obtain val2
        Goldilocks2::Element val2 = pcs2.open(result->challenges, sec_param);
        Goldilocks2::Element claimed_val1 = fp.evaluate(result->challenges); // Prover sends this to verifier
        if (claimed_val1 * val2 != result->claim) {
            return false;
        }
        result->claim = claimed_val1;
        // check the claimed_val1
        f.fix(0, result->challenges);
        sProver prover2(f);
        sVerifier verifier2;
        if (!verifier2.partial_sumcheck(prover2, result->challenges, result->claim, sec_param)) {
            return false;
        }
        if (result->claim != pcs.open(result->challenges, sec_param)) {
            return false;
        }
    }
    return true;
}


bool test_conv_check() {
    for (int cnt(0); cnt != CNT_TEST; ++cnt) {
        size_t C = rand() % 10 + 1, N = rand() % 10 + 1, D = rand() % 10 + 1, K = rand() % 10 + 1;
        
        // X: C x N, W: C x D x K, Y: D x (N + K - 1)
        std::vector<std::vector<Goldilocks2::Element>> X(C, std::vector<Goldilocks2::Element>(N));
        std::vector<std::vector<std::vector<Goldilocks2::Element>>> W(C, std::vector<std::vector<Goldilocks2::Element>>(D, std::vector<Goldilocks2::Element>(K)));
        std::vector<std::vector<Goldilocks2::Element>> Y(D, std::vector<Goldilocks2::Element>(N + K - 1));

        for (size_t i = 0; i < C; ++i) {
            for (size_t j = 0; j < N; ++j) {
                X[i][j] = random_ext();
            }
        }
        for (size_t i = 0; i < C; ++i) {
            for (size_t j = 0; j < D; ++j) {
                for (size_t k = 0; k < K; ++k) {
                    W[i][j][k] = random_ext();
                }
            }
        }
        for (size_t d = 0; d < D; ++d) {
            for (size_t n = 0; n < N; ++n) {
                for (size_t c = 0; c < C; ++c) {
                    for (size_t k = 0; k < K; ++k) {
                        Y[d][n + k] += X[c][n] * W[c][d][k];
                    }
                }
            }
        }


        convTriple triple(X, W, Y);
        convProver prover(triple);
        std::array<ligeropcs_ext, 3> pcs = triple.commit(2);
        std::array<const oracle*, 3> oracle = { &pcs[0], &pcs[1], &pcs[2] };

        if (!triple.check()) {
            return false;
        }

        if (!convVerifier::execute_convcheck_1d(prover, oracle, 32)) {
            return false;
        }
    }
    return true;
}

void random_conv2(
    size_t C, size_t D, size_t n, size_t m,
    std::vector<std::vector<std::vector<Goldilocks2::Element>>>& X, // C x n x n
    std::vector<std::vector<std::vector<std::vector<Goldilocks2::Element>>>>& W
    ) {

    X.resize(C, std::vector<std::vector<Goldilocks2::Element>>(n, std::vector<Goldilocks2::Element>(n)));
    W.resize(C, std::vector<std::vector<std::vector<Goldilocks2::Element>>>(D, std::vector<std::vector<Goldilocks2::Element>>(m, std::vector<Goldilocks2::Element>(m))));

    for (size_t c = 0; c < C; ++c) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                X[c][i][j] = random_ext();
            }
        }
    }
    for (size_t c = 0; c < C; ++c) {
        for (size_t d = 0; d < D; ++d) {
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    W[c][d][i][j] = random_ext();
                }
            }
        }
    }
}

void random_conv2_padding(
    size_t C, size_t D, size_t n, size_t m, size_t padding,
    array_view<Goldilocks2::Element>& X,
    array_view<Goldilocks2::Element>& W,
    array_view<Goldilocks2::Element>& Y
    ) {
    // size_t in = n + 2 * padding;
    size_t on = n + 2 * padding - m + 1;

    assert(X.shape(0) == C);
    assert(X.shape(1) == n);
    assert(X.shape(2) == n);
    assert(W.shape(0) == D); // D
    assert(W.shape(1) == C);
    assert(W.shape(2) == m);
    assert(W.shape(3) == m);
    assert(Y.shape(0) == D);
    assert(Y.shape(1) == on);
    assert(Y.shape(2) == on);
    
    random_vec_u64(X.get_data(), X.size());
    random_vec_u64(W.get_data(), W.size());

    // Compute 2D convolution with padding
    #pragma omp parallel for
    for (int64_t d = 0; d < int64_t(D); ++d) {
        for (int64_t i = 0; i < int64_t(on); ++i) {
            for (int64_t j = 0; j < int64_t(on); ++j) {
                Goldilocks2::Element sum = Goldilocks2::zero();
                for (int64_t c = 0; c < int64_t(C); ++c) {
                    for (int64_t ki = 0; ki < int64_t(m); ++ki) {
                        for (int64_t kj = 0; kj < int64_t(m); ++kj) {
                            int64_t x_i = i + ki - padding;
                            int64_t x_j = j + kj - padding;
                            if (x_i >= 0 && x_i < int64_t(n) && x_j >= 0 && x_j < int64_t(n)) {
                                sum = sum + X(c, x_i, x_j) * W(d, c, ki, kj);
                            }
                        }
                    }
                }
                Y(d, i, j) = sum;
            }
        }
    }
}

bool test_conv2_check() {
    for (int cnt(0); cnt != CNT_TEST; ++cnt) {
        srand(cnt);
        size_t padding = rand() % 4;
        size_t C = rand() % 5 + 1, D = rand() % 5 + 1, n = 1 << (rand() % 3 + 1), m = 3;
        size_t on = n + 2 * padding - m + 1;
        // X: C x n x n, W: C x D x m x m, Y: D x on x on
        std::vector<Goldilocks2::Element> X(C * n * n);
        std::vector<Goldilocks2::Element> W(D * C * m * m);
        std::vector<Goldilocks2::Element> Y(D * on * on);
        array_view<Goldilocks2::Element> X_view(X.data(), {C, n, n});
        array_view<Goldilocks2::Element> W_view(W.data(), {D, C, m, m});
        array_view<Goldilocks2::Element> Y_view(Y.data(), {D, on, on});


        size_t up_D = (1ull << find_ceiling_log2(D)), up_on = (1ull << find_ceiling_log2(on));

        
        random_conv2_padding(C, D, n, m, padding, X_view, W_view, Y_view);
        
        MultilinearPolynomial p1(X_view);
        MultilinearPolynomial p3(Y_view);

        array<Goldilocks2::Element> W_pad, Y_pad;
        std::vector<size_t> mapfrom, mapto(Y.size());
        for (size_t i = 0; i < Y.size(); ++i) mapto[i] = i;
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < on; ++j) {
                for (size_t k = 0; k < on; ++k) {
                    mapfrom.push_back(i * up_on * up_on + j * up_on + k);
                }
            }
        }

        pad_weights(C, D, n, m, padding, X_view, W_view, Y_view, W_pad, Y_pad, m, padding, mapto, true);

        // check mapfrom & mapto with p3 & Y_pad
        for (size_t i = 0; i != mapfrom.size(); ++i) {
            if (p3.get_eval_table()[mapfrom[i]] != Y_pad.data[mapto[i]]) {
                std::cout << "Map check 1 fail" << std::endl;
                std::cout << "Map from " << mapfrom[i] << " to " << mapto[i] << " is incorrect." << std::endl;
                throw;
            }
        }
        
        
        MultilinearPolynomial p_w(W_pad);

        ligeropcs_ext pcs_w = ligero_commit_ext(p_w, 2); // commit as D C m m

        W_pad.view.swap_dim(0, 1); // C D m m
        convProver prover(make_conv2_prover(C, D, n, m, padding, X_view, W_pad, Y_pad, mapto));
        prover.init_ori_Y(Y_view);

        // check mapfrom & mapto with p3 & prover.triple.Y
        for (size_t i = 0; i != mapfrom.size(); ++i) {
            if (p3.get_eval_table()[mapfrom[i]] != prover.triple.Y->get_eval_table()[mapto[i]]) {
                std::cout << "Map check 2 fail" << std::endl;
                std::cout << "Map from " << mapfrom[i] << " to " << mapto[i] << " is incorrect." << std::endl;
                throw;
            }
        }

        // std::array<ligeropcs_ext, 3> pcs = { ligero_commit_ext(p1, 2), pcs_w, ligero_commit_ext(p3, 2) };
        // std::array<const oracle*, 3> oracle = { &pcs[0], &pcs[1], &pcs[2] };

        auto pcs_x = ligero_commit_ext(p1, 2);
        auto pcs_y = ligero_commit_ext(p3, 2);
        open_param op_x(X_view, &pcs_x);
        open_param op_w(W_pad.view, &pcs_w);
        open_param op_y(Y_view, &pcs_y);
        
        op_w.rev[2] = op_w.rev[2] ^ true;
        op_w.rev[3] = op_w.rev[3] ^ true;

        if (!prover.triple.check()) {
            std::cout << "convTriple check failed" << std::endl;
            return false;
        }

        if (!convVerifier::execute_convcheck_2d(prover, op_x, op_w, op_y, mapfrom, mapto, 2, 32, true)) {
            return false;
        }
    }
    return true;
}

bool test_pad_check() {
    for (int cnt(0); cnt != CNT_TEST; ++cnt) {
        size_t n = rand() % 20 + 2, m = rand() % 20 + 2;
        std::vector<Goldilocks2::Element> r = random_vec_ext(n * m);
        array_view<Goldilocks2::Element> r_view(r.data(), {n, m});
        MultilinearPolynomial X(r_view);
        ligeropcs_ext pcs = ligero_commit_ext(X, 2);

        size_t pad_row = 1ull << find_ceiling_log2(m + rand() % 100 + 1);
        std::vector<Goldilocks2::Element> r_pad(n * pad_row, Goldilocks2::zero());
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                r_pad[i * pad_row + j] = r_view(i, j);
            }
        }
        array_view<Goldilocks2::Element> r_pad_view(r_pad.data(), {n, pad_row});
        MultilinearPolynomial X_pad(r_pad_view);
        int begin = find_ceiling_log2(n), end = find_ceiling_log2(n * pad_row) - find_ceiling_log2(m);
        auto challenge = random_vec_ext(X_pad.get_num_vars());
        Goldilocks2::Element claimed_Xr = X_pad.evaluate(challenge);
        auto cha_claim = execute_pad_check(claimed_Xr, begin, end, challenge, 32);
        if (cha_claim.claim != pcs.open(cha_claim.challenges, 32)) {
            return false;
        }
    }
    return true;
}


bool test_pad_weights() {
    for (int cnt(0); cnt != CNT_TEST; ++cnt) {
        srand(cnt);
        size_t padding = rand() % 4;
        size_t C = rand() % 5 + 1, D = rand() % 5 + 1, n = 1 << (rand() % 3 + 3), m = rand() % 3 + 2;
        size_t on = n + 2 * padding - m + 1;
        // X: C x n x n, W: C x D x m x m, Y: D x on x on
        array<Goldilocks2::Element> X({C, n, n});
        array<Goldilocks2::Element> W({D, C, m, m});
        array<Goldilocks2::Element> Y({D, on, on});


        random_conv2_padding(C, D, n, m, padding, X.view, W.view, Y.view);
        
        ligeropcs_ext pcs_w;

        bool pad_right_bottom = rand() % 2 == 0;
        if (!pad_right_bottom) {
            W.view.reverse(2);
            W.view.reverse(3);
            pcs_w = ligero_commit_ext(W.view, 2);
            W.rearrange();
            W.view.reverse(2);
            W.view.reverse(3);
        } else {
            pcs_w = ligero_commit_ext(W.view, 2);
        }

        MultilinearPolynomial p3 = Y.view;

        array<Goldilocks2::Element> W_pad, Y_pad;
        std::vector<size_t> mapfrom, mapto(Y.data.size());
        for (size_t i = 0; i != mapto.size(); ++i) mapto[i] = i;
        size_t up_on = (1ull << find_ceiling_log2(on));
        mapfrom.reserve(D * on * on);
        for (size_t i = 0; i != D; ++i) {
            for (size_t j = 0; j != on; ++j) {
                for (size_t k = 0; k != on; ++k) {
                    mapfrom.push_back(i * up_on * up_on + j * up_on + k);
                }
            }
        }
        pad_weights(C, D, n, m, padding, X.view, W.view, Y.view, W_pad, Y_pad, m, padding, mapto, pad_right_bottom);

        // check mapfrom & mapto with p3 & Y_pad
        for (size_t i = 0; i != mapfrom.size(); ++i) {
            if (p3.get_eval_table()[mapfrom[i]] != Y_pad.data[mapto[i]]) {
                std::cerr << "test_pad_weights: mapfrom & mapto fail at 1." << std::endl;
                std::cout << "Map from " << mapfrom[i] << " to " << mapto[i] << " is incorrect." << std::endl;
                throw;
            }
        }

        // MultilinearPolynomial p_w(W_pad.view);
        // ligeropcs_ext pcs_w = ligero_commit_ext(p_w, 2); // commit as D C pad_m pad_m

        W_pad.view.swap_dim(0, 1); // C D m m
        convProver prover(make_conv2_prover(C, D, n, m, padding, X.view, W_pad.view, Y_pad.view, mapto));
        prover.init_ori_Y(Y.view);
        if (!prover.triple.check()) {
            std::cout << "convTriple check failed" << std::endl;
            return false;
        }

        // std::array<ligeropcs_ext, 3> pcs = prover.triple.commit(2);
        // std::array<const oracle*, 3> oracle = { &pcs[0], &pcs[1], &pcs[2] };
        MultilinearPolynomial p1(X);
        MLE_Convker p2 = *dynamic_cast<MLE_Convker*>(prover.triple.W.get());

        ligeropcs_ext pcs_x = ligero_commit_ext(p1, 2);
        ligeropcs_ext pcs_y = ligero_commit_ext(p3, 2);

        open_param op_x(X.view, &pcs_x);
        open_param op_w(W_pad.view, &pcs_w);
        open_param op_y(Y.view, &pcs_y);

        op_w.rev[2] = op_w.rev[2] ^ true;
        op_w.rev[3] = op_w.rev[3] ^ true;

        // if (!prover.triple.check()) {
        //     std::cout << "convTriple check failed" << std::endl;
        //     return false;
        // }

        if (!convVerifier::execute_convcheck_2d(prover, op_x, op_w, op_w, mapfrom, mapto, 2, 32, true)) {
            return false;
        }
    }
    return true;
}

bool test_mat_mult() {
    for (int cnt(0); cnt != CNT_TEST; ++cnt) {
        srand(cnt);
        size_t n = (rand() % 100 + 3), m = (rand() % 100 + 3), l = (rand() % 100 + 3);
        // X: n x m, W: m x l, Y: n x l
        array<Goldilocks2::Element> X({n, m});
        array<Goldilocks2::Element> W({m, l});
        array<Goldilocks2::Element> Y({n, l});

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                X(i, j) = random_ext();
            }
        }
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < l; ++j) {
                W(i, j) = random_ext();
            }
        }
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < l; ++j) {
                Goldilocks2::Element sum = Goldilocks2::zero();
                for (size_t k = 0; k < m; ++k) {
                    sum = sum + X(i, k) * W(k, j);
                }
                Y(i, j) = sum;
            }
        }
        
        MultilinearPolynomial p_x(X);
        MultilinearPolynomial p_w(W);
        MultilinearPolynomial p_y(Y);
        ligeropcs_ext pcs_x = ligero_commit_ext(p_x, 2);
        ligeropcs_ext pcs_w = ligero_commit_ext(p_w, 2);
        ligeropcs_ext pcs_y = ligero_commit_ext(p_y, 2);
        mat_mult_prover prover(n, m, l, p_x, p_w, p_y);
        if (!mat_mult_verifier::execute_mat_mult_check(
            prover,
            open_param(X.view, &pcs_x),
            open_param(W.view, &pcs_w),
            open_param(Y.view, &pcs_y),
            32)) {
            return false;
        }
    }
    return true;
}

bool test_logup() {
    for (int cnt(0); cnt != CNT_TEST; ++cnt) {
        srand(cnt);
        size_t fsize = 4 << (rand() % 10), tsize = 4 << (rand() % 10);
        std::vector<std::pair<uint64_t, uint64_t>> t(tsize);
        std::vector<bool> vis(1 << 18);
        for (size_t i = 0; i < tsize; ++i) {
            size_t new_t1 = rand() % (1ull << 18);
            while (vis[new_t1]) {
                new_t1 = rand() % (1ull << 18);
            }
            vis[new_t1] = true;
            t[i].first = new_t1;
            t[i].second = rand() % (1ull << 5);
        }
        std::sort(t.begin(), t.end());
        std::vector<uint64_t> t1(tsize), t2(tsize);
        for (size_t i = 0; i < tsize; ++i) {
            t1[i] = t[i].first;
            t2[i] = t[i].second;
        }

        std::vector<uint64_t> f1(fsize);
        std::vector<uint64_t> f2(f1.size());

        for (size_t i = 0; i < f1.size(); ++i) {
            size_t r = rand() % t1.size();
            f1[i] = t1[r];
            f2[i] = t2[r];
        }

        LogupProver lpr(f1, f2, t1, t2);
        auto pcs_f1 = ligero_commit_base(f1, 2),
             pcs_f2 = ligero_commit_base(f2, 2),
             pcs_t1 = ligero_commit_base(t1, 2),
             pcs_t2 = ligero_commit_base(t2, 2);
        if (!LogupVerifier::execute_logup(lpr, pcs_f1, pcs_f2, pcs_t1, pcs_t2, 2, 32)) {
            return false;
        }
    }
    return true;
}

bool test_div_check() {
    for (int cnt = 0; cnt < CNT_TEST; ++cnt) {
        srand(cnt);
        bool allow_neg_sign = rand() % 2 == 0;
        int64_t denom = (1ull << (rand() % 10 + 1));
        size_t n = (1 << (rand() % 10 + 1)), max_val = (1ull << (rand() % 20 + 1));
        std::vector<int64_t> num_val(n);
        for (size_t i = 0; i < n; ++i) {
            num_val[i] = (rand() % max_val) * (rand() % 2 == 0 ? 1 : -1);
        }
        std::vector<Goldilocks2::Element> num(n);
        for (size_t i = 0; i < n; ++i) {
            num[i] = Goldilocks2::fromS64(num_val[i]);
        }
        auto [quo, rem] = get_quo_rem(num, denom, allow_neg_sign);
        divProver prover(num, quo, rem, denom, allow_neg_sign, 2);
        ligeropcs_base pcs_num = ligero_commit_base(num, 2);
        ligeropcs_base pcs_quo = ligero_commit_base(quo, 2);
        ligeropcs_base pcs_rem = ligero_commit_base(rem, 2);
        if (!divVerifier::execute_div_check(
            prover,
            pcs_num,
            pcs_quo,
            pcs_rem,
            32)) {
            return false;
        }
    }
    return true;
}

bool test_sign_check() {
    for (int cnt = 0; cnt < CNT_TEST; ++cnt) {
        srand(cnt);
        bool strict = rand() % 2 == 0;
        int64_t denom = (1ull << (rand() % 10 + 1));
        int64_t n = (1 << (rand() % 10 + 1)), max_val = (1ull << (rand() % 20 + 1));
        std::vector<int64_t> num_val(n), sign_val(n);
        for (size_t i = 0; i < n; ++i) {
            num_val[i] = (rand() % max_val) * (rand() % 2 == 0 ? 1 : -1);
            if (!strict) sign_val[i] = (num_val[i] < 0) ? 0 : 1;
            else sign_val[i] = (num_val[i] <= 0) ? 0 : 1;
        }
        std::vector<Goldilocks2::Element> num(n), sign(n);
        for (size_t i = 0; i < n; ++i) {
            num[i] = Goldilocks2::fromS64(num_val[i]);
            sign[i] = Goldilocks2::fromS64(sign_val[i]);
        }
        signProver prover(num, sign, denom, max_val, strict, 2);
        ligeropcs_base pcs_num = ligero_commit_base(num, 2);
        ligeropcs_base pcs_sign = ligero_commit_base(sign, 2);
        if (!signVerifier::execute_sign_check(
            prover,
            pcs_num,
            pcs_sign,
            32)) {
            return false;
        }
    }
    return true;
}

bool test_prod_check() {
    for (int cnt = 0; cnt < CNT_TEST; ++cnt) {
        srand(cnt);
        int logn = rand() % 10 + 2;
        size_t n = 1 << logn;
        size_t suf = (rand() % (logn - 1)) + 1;
        std::vector<Goldilocks2::Element> raw(n);
        for (size_t i = 0; i < n; ++i) {
            raw[i] = Goldilocks2::fromS64(rand() % 1000);
        }
        std::vector<Goldilocks2::Element> pro(1 << (logn - suf));
        for (int x = 0; x < (1 << (logn - suf)); ++x) {
            pro[x] = Goldilocks2::one();
            for (int y = 0; y < (1 << suf); ++y) {
                pro[x] *= raw[(x << suf) | y];
            }
        }
        auto pcs_raw = ligero_commit_base(raw, 2), pcs_pro = ligero_commit_base(pro, 2);
        array_view<Goldilocks2::Element> view_raw(raw.data(), {size_t(1 << logn)});
        array_view<Goldilocks2::Element> view_pro(pro.data(), {size_t(1 << (logn - suf))});
        open_param open_raw(view_raw, &pcs_raw);
        open_param open_pro(view_pro, &pcs_pro);
        prodProver prover(raw, pro, suf, 2);
        if (!prodVerifier::execute_prod_check(
            prover,
            open_raw,
            open_pro,
            32)) {
            return false;
        }
    }
    return true;
}

bool test_ltn_check() {
    for (int cnt = 0; cnt < CNT_TEST; ++cnt) {
        srand(cnt);
        int logn = rand() % 10 + 1;
        size_t n = (1ull << logn);
        std::vector<Goldilocks2::Element> a(n);
        Goldilocks2::Element b = Goldilocks2::fromS64(rand() % 1000);
        for (size_t i = 0; i < n; ++i) {
            a[i] = Goldilocks2::fromS64((rand() % 1999) - 999);
        }
        bool strict = rand() % 2 == 0;
        ltnProver prover(a, b, 1<<14, 1000, strict, 2);
        auto ltn = prover.get_ltn();
        for (size_t i = 0; i < n; ++i) {
            if (strict) {
                if ((Goldilocks2::toS64(a[i]) < Goldilocks2::toS64(b)) ^ (ltn[i] == Goldilocks2::one())) {
                    std::cerr << "Strict LTN manual check failed at index " << i << std::endl;
                    return false;
                }
            } else {
                if ((Goldilocks2::toS64(a[i]) <= Goldilocks2::toS64(b)) ^ (ltn[i] == Goldilocks2::one())) {
                    std::cerr << "Non-strict LTN manual check failed at index " << i << std::endl;
                    return false;
                }
            }
        }
        auto pcs_a = ligero_commit_base(a, 2);
        auto pcs_ltn = prover.get_pcs_ltn();
        if (!ltnVerifier::execute_ltn_check(
            prover,
            pcs_a,
            pcs_ltn,
            b,
            1000, strict, 32)) {
            return false;
        }
    }
    return true;
}

bool test_e_pow_check() {
    for (int cnt = 0; cnt < CNT_TEST; ++cnt) {
        srand(cnt);
        int logn = rand() % 10 + 1;
        int logscale = rand() % 10 + 4;
        size_t n = (1ull << logn), scale = (1ull << logscale);
        size_t max_val = (1ull << (rand() % 10 + logscale));
        size_t rho_inv = 2;
        std::vector<uint64_t> a(n), b(n);
        for (size_t i = 0; i < n; ++i) {
            uint64_t v = rand() % max_val;
            a[i] = v;
            b[i] = std::round(std::exp(-double(v) / double(scale)) * scale);
        }
        eProver prover(a, b, scale, max_val, rho_inv);
        auto pcs_a = ligero_commit_base(a, 2);
        auto pcs_b = ligero_commit_base(b, 2);
        if (!eVerifier::execute_check(
            prover,
            pcs_a,
            pcs_b,
            32)) {
            return false;
        }
    }
    return true;
}

bool test_pd_check() {
    for (int cnt = 0; cnt < CNT_TEST; ++cnt) {
        srand(cnt);
        int logn = rand() % 10 + 2;
        size_t n = (1ull << logn);
        auto f1 = random_vec_ext(n), f2 = random_vec_ext(n);
        Goldilocks2::Element pd = Goldilocks2::one();
        for (size_t i = 0; i != n; ++i) {
            pd *= f1[i] / f2[i];
        }
        MultilinearPolynomial mle_f1(f1), mle_f2(f2);
        ligeropcs_ext pcs_f1(ligero_commit_ext(mle_f1, 2)), pcs_f2(ligero_commit_ext(mle_f2, 2));
        pdProver prover(mle_f1, mle_f2);
        if (!pdVerifier::execute_check(prover, &pcs_f1, &pcs_f2, pd, 2, 32)) {
            return false;
        }
    }
    return true;
}

bool test_set_check() {
    for (int cnt = 0; cnt < CNT_TEST; ++cnt) {
        srand(cnt);
        int n = rand() % 10 + 1;
        int log_num_vars = rand() % 10 + 1;
        size_t num_vars = (1ull << log_num_vars);
        std::vector<std::vector<Goldilocks2::Element>> set1, set2;
        std::vector<size_t> ind(num_vars);
        for (int i = 0; i < n; ++i) {
            set1.push_back(random_vec_ext(num_vars));
            set2.push_back(set1.back());
        }
        for (size_t i = 0; i < num_vars; ++i) {
            ind[i] = i;
        }
        std::shuffle(ind.begin(), ind.end(), std::mt19937{std::random_device{}()});
        for (int i = 0; i < n; ++i) {
            for (size_t j = 0; j < num_vars; ++j) {
                set2[i][j] = set1[i][ind[j]];
            }
        }
        std::vector<MLE> mle_set1, mle_set2;
        std::vector<const MLE*> ptr_set1, ptr_set2;
        std::vector<ligeropcs_ext> pcs_set1, pcs_set2;
        std::vector<const oracle*> ptr_pcs1, ptr_pcs2;
        mle_set1.reserve(n);
        mle_set2.reserve(n);
        for (int i = 0; i < n; ++i) {
            mle_set1.emplace_back(set1[i]);
            mle_set2.emplace_back(set2[i]);
            pcs_set1.push_back(ligero_commit_ext(mle_set1.back(), 2));
            pcs_set2.push_back(ligero_commit_ext(mle_set2.back(), 2));
        }
        for (int i = 0; i < n; ++i) {
            ptr_set1.push_back(&mle_set1[i]);
            ptr_set2.push_back(&mle_set2[i]);
            ptr_pcs1.push_back(&pcs_set1[i]);
            ptr_pcs2.push_back(&pcs_set2[i]);
        }
        setProver prover(ptr_set1, ptr_set2);
        if (!setVerifier::execute_check(prover, ptr_pcs1, ptr_pcs2, 2, 32)) {
            return false;
        }
    }
    return true;
}

bool test_perm_check() {
    for (int cnt = 0; cnt < CNT_TEST; ++cnt) {
        srand(cnt);
        int n = rand() % 10 + 1;
        int logn1(rand() % 10 + 1), logn2(rand() % 10 + 2);
        if (logn1 > logn2) std::swap(logn1, logn2);
        int n1 = (1 << logn1), n2 = (1 << logn2);
        std::vector<std::vector<Goldilocks2::Element>> f1(n), f2(n);
        std::vector<bool> vis(1 << logn2);
        std::vector<size_t> perm(1 << logn1);
        for (int i = 0; i < n; ++i) {
            f1[i].resize(n1);
            f2[i].resize(n2);
        }
        for (size_t i = 0; i < n1; ++i) {
            size_t mapto = rand() % n2;
            while (vis[mapto]) {
                mapto = rand() % n2;
            }
            vis[mapto] = true;
            perm[i] = mapto;
            for (int j = 0; j < n; ++j) {
                f2[j][mapto] = f1[j][i] = Goldilocks2::fromU64(rand() % (1 << 20) + 1);
            }
        }
        for (size_t i = 0; i < n2; ++i) {
            if (!vis[i]) {
                for (int j = 0; j != n; ++j) {
                    f2[j][i] = Goldilocks2::fromU64(rand() % (1 << 20) + 1);
                }
            }
        }
        std::vector<MLE> mle_f1, mle_f2;
        std::vector<ligeropcs_base> pcs_f1, pcs_f2;
        permProver prover(perm, false);
        permVerifier verifier;
        for (int i = 0; i != n; ++i) {
            mle_f1.emplace_back(f1[i]);
            mle_f2.emplace_back(f2[i]);
            pcs_f1.push_back(ligero_commit_base(mle_f1.back(), 2));
            pcs_f2.push_back(ligero_commit_base(mle_f2.back(), 2));
        }
        for (int i = 0; i != n; ++i) {
            prover.add_mle(&mle_f1[i], &mle_f2[i]);
            verifier.add_pcs(&pcs_f1[i], &pcs_f2[i]);
        }
        if (!verifier.execute_check(prover, 2, 32)) {
            return false;
        }
    }
    return true;
}

bool test_map_check() {
    for (int cnt = 0; cnt < CNT_TEST; ++cnt) {
        srand(cnt);
        int len = rand() % 10 + 1;
        int logn1(rand() % 10 + 1), logn2(rand() % 10 + 2);
        if (logn1 > logn2) std::swap(logn1, logn2);
        int n1 = (1 << logn1), n2 = (1 << logn2);
        int perm_sz = (rand() % n1 + 1);
        std::vector<std::vector<Goldilocks2::Element>> f1(len), f2(len);
        std::vector<bool> vis_from(1 << logn1), vis_to(1 << logn2);
        std::vector<size_t> perm_from(perm_sz);
        std::vector<size_t> perm_to(perm_sz);
        std::map<size_t, size_t> map;
        for (int i = 0; i < len; ++i) {
            f1[i].resize(n1, Goldilocks2::fromU64(rand() % (1 << 20)));
            f2[i].resize(n2, Goldilocks2::fromU64(rand() % (1 << 20)));
        }
        for (size_t i = 0; i < perm_sz; ++i) {
            size_t v;
            do {
                v = rand() % n1;
            } while (vis_from[v]);
            vis_from[v] = true;
            perm_from[i] = v;
        }
        std::sort(perm_from.begin(), perm_from.end());
        for (size_t i = 0; i < perm_sz; ++i) {
            size_t v;
            do { v = rand() % n2; } while (vis_to[v]);
            vis_to[v] = true;
            perm_to[i] = v;
            map[perm_from[i]] = perm_to[i];
        }
        for (size_t i = 0; i != len; ++i) {
            for (const auto& m : map) {
                f2[i][m.second] = f1[i][m.first];
            }
        }

        std::vector<MLE> mle_f1, mle_f2;
        std::vector<ligeropcs_base> pcs_f1, pcs_f2;
        mapProver prover(perm_from, perm_to, false);
        mapVerifier verifier;
        for (int i = 0; i != len; ++i) {
            mle_f1.emplace_back(f1[i]);
            mle_f2.emplace_back(f2[i]);
            pcs_f1.push_back(ligero_commit_base(mle_f1.back(), 2));
            pcs_f2.push_back(ligero_commit_base(mle_f2.back(), 2));
        }
        for (int i = 0; i != len; ++i) {
            prover.add_mle(&mle_f1[i], &mle_f2[i]);
            verifier.add_pcs(&pcs_f1[i], &pcs_f2[i]);
        }
        if (!verifier.execute_check(prover, 2, 32)) {
            return false;
        }
    }
    return true;
}

bool run_test() {
    srand(79);
    if (!test_arithmetic()) {
        std::cout << "test_arithmetic failed" << std::endl;
        return false;
    }
    if (!test_eval_power_mle()) {
        std::cout << "test_eval_power_mle failed" << std::endl;
        return false;
    }
    if (!test_product3_sumcheck()) {
        std::cout << "test_product3_sumcheck failed" << std::endl;
        return false;
    }
    if (!test_product2_sumcheck()) {
        std::cout << "test_product2_sumcheck failed" << std::endl;
        return false;
    }
    if (!test_partial_sumcheck_product2()) {
        std::cout << "test_partial_sumcheck_product2 failed" << std::endl;
        return false;
    }
    if (!test_logup()) {
        std::cout << "test_logup failed" << std::endl;
        return false;
    }
    
    if (!test_pd_check()) {
        std::cout << "test_pd_check failed" << std::endl;
        return false;
    }
    if (!test_set_check()) {
        std::cout << "test_set_check failed" << std::endl;
        return false;
    }
    if (!test_perm_check()) {
        std::cout << "test_perm_check failed" << std::endl;
        return false;
    }
    if (!test_map_check()) {
        std::cout << "test_map_check failed" << std::endl;
        return false;
    }
    if (!test_conv_check()) {
        std::cout << "test_conv_check failed" << std::endl;
        return false;
    }
    if (!test_conv2_check()) {
        std::cout << "test_conv2_check failed" << std::endl;
        return false;
    }
    if (!test_pad_check()) {
        std::cout << "test_pad_check failed" << std::endl;
        return false;
    }
    if (!test_pad_weights()) {
        std::cout << "test_pad_weights failed" << std::endl;
        return false;
    }
    if (!test_mat_mult()) {
        std::cout << "test_mat_mult failed" << std::endl;
        return false;
    }
    if (!test_div_check()) {
        std::cout << "test_div_check failed" << std::endl;
        return false;
    }
    if (!test_sign_check()) {
        std::cout << "test_sign_check failed" << std::endl;
        return false;
    }
    if (!test_prod_check()) {
        std::cout << "test_prod_check failed" << std::endl;
        return false;
    }
    if (!test_ltn_check()) {
        std::cout << "test_ltn_check failed" << std::endl;
        return false;
    }
    if (!test_e_pow_check()) {
        std::cout << "test_e_pow_check failed" << std::endl;
        return false;
    }
    std::cout << "All tests passed" << std::endl;
    return true;
}
