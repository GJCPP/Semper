#include "conv_check.h"
#include "mle_pow.h"

convProver::convProver(const convTriple& triple)
    : triple(triple) {
}

convProver::convProver(convTriple&& triple)
    : triple(std::move(triple)) {
}


bool convTriple::check() const {
    for (size_t d = 0; d < D; ++d) {
        std::vector<Goldilocks2::Element> res(N + K - 1);
        for (size_t c = 0; c < C; ++c) {
            for (size_t n = 0; n < N; ++n) {
                for (size_t k = 0; k < K; ++k) {
                    res[n + k] += X->eval_hypercube((c << logN) | n) *
                        W->eval_hypercube((c << (logD + logK)) | (d << logK) | k);
                }
            }
        }
        for (size_t n = 0; n < N + K - 1; ++n) {
            if (res[n] != Y->eval_hypercube((d << logNK1) | n)) {
                return false;
            }
        }
    }
    return true;
}

p2Prover convProver::fix_beta_r_D(
    const Goldilocks2::Element& beta,
    const std::vector<Goldilocks2::Element>& r_D) {

    this->beta = beta;
    assert(r_D.size() == triple.logD);
    assert(triple.Y->get_num_vars() == triple.logD + triple.logNK1); // fix_r_D should be called only once
    assert(triple.W->get_num_vars() == triple.logC + triple.logD + triple.logK);

    // Fix Y and W
    triple.Y->fix(0, r_D); // Y'(nk) = Y(r_d, nk)
    triple.W->fix(triple.logC, r_D); // W'(c, d, k) = W(c, r_d, k)

    // Create p2_prover for RHS
    std::unique_ptr<MLE_Pow> rhs_beta = std::make_unique<MLE_Pow>(beta, triple.logNK1, triple.N + triple.K - 2, true);
    return p2Prover(triple.Y->clone(), std::move(rhs_beta));
}

p2Prover convProver::shrink_XW() {
    return p2Prover(
        std::make_unique<MultilinearPolynomial>(triple.W->sum_over_lowbits_with_power(triple.logK, beta)),
        std::make_unique<MultilinearPolynomial>(triple.X->sum_over_lowbits_with_power(triple.logN, beta))
    );
}

std::array<p2Prover, 2> convProver::fix_r_C(const std::vector<Goldilocks2::Element>& r_C) {
    triple.X->fix(0, r_C);
    triple.W->fix(0, r_C);
    std::unique_ptr<MLE_Pow> X_beta = std::make_unique<MLE_Pow>(beta, triple.logN, triple.N - 1, true);
    std::unique_ptr<MLE_Pow> W_beta = std::make_unique<MLE_Pow>(beta, triple.logK, triple.K - 1, true);
    return {
        p2Prover(triple.X->clone(), std::move(X_beta)),
        p2Prover(triple.W->clone(), std::move(W_beta))
    };
}

convProver make_conv_prover(
    const std::vector<std::vector<Goldilocks2::Element>>& X, // in_channels x N
    const std::vector<std::vector<std::vector<Goldilocks2::Element>>>& W, // in_channels x out_channels x kernel_size
    const std::vector<std::vector<Goldilocks2::Element>>& Y) { // out_channels x (N + kernel_size - 1)

    return convProver(convTriple(X, W, Y));
}

convProver make_conv2_prover(
    size_t C, size_t D, size_t n, size_t m,
    const std::vector<std::vector<std::vector<Goldilocks2::Element>>>& X, // in_channels x n x n
    const std::vector<std::vector<std::vector<std::vector<Goldilocks2::Element>>>>& W) { // in_channels x out_channels x m x m

    assert(X.size() == C);
    assert(X[0].size() == n);
    assert(X[0][0].size() == n);
    assert(W.size() == C);
    assert(W[0].size() == D);
    assert(W[0][0].size() == m);
    assert(W[0][0][0].size() == m);

    std::vector<std::vector<Goldilocks2::Element>> X_1d(C);
    std::vector<std::vector<std::vector<Goldilocks2::Element>>> W_1d(C);
    std::vector<std::vector<Goldilocks2::Element>> Y_1d(D);

    for (size_t c = 0; c < C; ++c) {
        X_1d[c] = flatten(X[c]);
        W_1d[c].resize(D);
        for (size_t d = 0; d < D; ++d) {
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    W_1d[c][d].push_back(W[c][d][m - i - 1][m - j - 1]);
                }
                W_1d[c][d].resize(n * (i + 1));
            }
        }
    }
    for (size_t d = 0; d < D; ++d) {
        Y_1d[d].resize(n * n + n * m - 1);
        for (size_t c = 0; c < C; ++c) {
            auto temp = conv(X_1d[c], W_1d[c][d]);
            assert(temp.size() == Y_1d[d].size());
            for (size_t i = 0; i < temp.size(); ++i) {
                Y_1d[d][i] += temp[i];
            }
        }
    }

    size_t N = n * n;
    size_t K = n * m;

    return convProver(convTriple(C, N, D, K,
        std::make_unique<MultilinearPolynomial>(X_1d),
        std::make_unique<MLE_Convker>(W, C, D, n, m),
        std::make_unique<MultilinearPolynomial>(Y_1d)));
}

bool convVerifier::execute_convcheck(
    convProver& prover,
    const std::array<const oracle_ext*, 3>& oracle,
    const size_t& sec_param) {

    // Step 1: Verifier samples beta and r_D
    Goldilocks2::Element beta = random_ext();
    std::vector<Goldilocks2::Element> r_D(random_vec_ext(prover.triple.logD));
    p2Prover rhs_prover = prover.fix_beta_r_D(beta, r_D);

    // Step 2: Prover proves the RHS
    Goldilocks2::Element rhs = rhs_prover.get_sum();
    MLE_Pow rhs_beta(beta, prover.triple.logNK1, prover.triple.N + prover.triple.K - 2);
    std::optional<challenge_claim> claim = p2Verifier::partial_sumcheck(rhs_prover, rhs, sec_param);
    if (!claim) return false;
    auto query_Y = combine_challenges(r_D, claim->challenges);
    if (claim->claim != oracle[2]->open(query_Y, sec_param) * rhs_beta.evaluate(claim->challenges)) {
        return false;
    }

    // Step 3: Prover proves the LHS = RHS
    // Step 3.1: Prove \sum_c X'(c) * W'(c), end with r_C
    auto lhs_prover = prover.shrink_XW();
    auto claim_xw = p2Verifier::partial_sumcheck(lhs_prover, rhs, sec_param);
    if (!claim_xw) return false;
    auto r_C = std::move(claim_xw->challenges);
    
    // Step 3.2: Prover claims x and w, proves separately X'(r_C) = \sum_n X(r_C, n) \beta^n
    // End with r_N for X and r_K for W
    auto [X_prover, W_prover] = prover.fix_r_C(r_C);
    Goldilocks2::Element x_val = X_prover.get_sum(), w_val = W_prover.get_sum();
    if (x_val * w_val != claim_xw->claim) {
        return false;
    }
    auto claim_x = p2Verifier::partial_sumcheck(X_prover, x_val, sec_param);
    if (!claim_x) return false;
    auto& r_N = claim_x->challenges;
    auto claim_w = p2Verifier::partial_sumcheck(W_prover, w_val, sec_param);
    if (!claim_w) return false;
    auto& r_K = claim_w->challenges;

    // Step 4: Query the oracle
    // Query X(r_C, r_N) and W(r_C, r_D, r_K)
    auto query_X = combine_challenges(r_C, r_N);
    auto query_W = combine_challenges(r_C, r_D, r_K);

    MLE_Pow beta_X(beta, prover.triple.logN, prover.triple.N - 1);
    MLE_Pow beta_W(beta, prover.triple.logK, prover.triple.K - 1); // beta^r_K

    if (oracle[0]->open(query_X, sec_param) * beta_X.evaluate(r_N) != claim_x->claim ||
        oracle[1]->open(query_W, sec_param) * beta_W.evaluate(r_K) != claim_w->claim) {
        return false;
    }
    return true;
}




convTriple::convTriple(const convTriple& other)
    : C(other.C), N(other.N), D(other.D), K(other.K),
      logC(other.logC), logN(other.logN), logD(other.logD), logK(other.logK), logNK1(other.logNK1),
      X(std::make_unique<MultilinearPolynomial>(*other.X)),
      W(std::make_unique<MultilinearPolynomial>(*other.W)),
      Y(std::make_unique<MultilinearPolynomial>(*other.Y)) {
}

convTriple::convTriple(
    const std::vector<std::vector<Goldilocks2::Element>>& X,
    const std::vector<std::vector<std::vector<Goldilocks2::Element>>>& W,
    const std::vector<std::vector<Goldilocks2::Element>>& Y) {

    assert(X.size() == W.size());
    assert(W[0].size() == Y.size());
    assert(X[0].size() + W[0][0].size() == Y[0].size() + 1);

    C = X.size();
    N = X[0].size();
    D = W[0].size();
    K = W[0][0].size();
    logC = find_ceiling_log2(C);
    logN = find_ceiling_log2(N);
    logD = find_ceiling_log2(D);
    logK = find_ceiling_log2(K);
    logNK1 = find_ceiling_log2(N + K - 1);
    std::vector<Goldilocks2::Element> X_vec(1ull << (logC + logN));
    std::vector<Goldilocks2::Element> W_vec(1ull << (logC + logD + logK));
    std::vector<Goldilocks2::Element> Y_vec(1ull << (logD + logNK1));
    for (size_t i = 0; i < C; ++i) {
        for (size_t j = 0; j < N; ++j) {
            X_vec[(i << logN) | j] = X[i][j];
        }
    }
    for (size_t c = 0; c < C; ++c) {
        for (size_t d = 0; d < D; ++d) {
            for (size_t k = 0; k < K; ++k) {
                W_vec[(c << (logD + logK)) | (d << logK) | k] = W[c][d][k];
            }
        }
    }
    for (size_t d = 0; d < D; ++d) {
        for (size_t n = 0; n < N + K - 1; ++n) {
            Y_vec[(d << logNK1) | n] = Y[d][n];
        }
    }
    this->X = std::make_unique<MultilinearPolynomial>(X_vec);
    this->W = std::make_unique<MultilinearPolynomial>(W_vec);
    this->Y = std::make_unique<MultilinearPolynomial>(Y_vec);
}

convTriple::convTriple(
    size_t C, size_t N, size_t D, size_t K,
    std::unique_ptr<MultilinearPolynomial> _X,
    std::unique_ptr<MultilinearPolynomial> _W,
    std::unique_ptr<MultilinearPolynomial> _Y)
    : C(C), N(N), D(D), K(K), 
      X(std::move(_X)), 
      W(std::move(_W)), 
      Y(std::move(_Y)) {

    logC = find_ceiling_log2(C);
    logN = find_ceiling_log2(N);
    logD = find_ceiling_log2(D);
    logK = find_ceiling_log2(K);
    logNK1 = find_ceiling_log2(N + K - 1);

    assert(X->get_num_vars() == logC + logN);
    assert(W->get_num_vars() == logC + logD + logK);
    assert(Y->get_num_vars() == logD + logNK1);
}

std::array<ligeropcs_ext, 3> convTriple::commit(size_t rho_inv) const {
    return {
        ligero_commit_ext(*X, rho_inv),
        ligero_commit_ext(*W, rho_inv),
        ligero_commit_ext(*Y, rho_inv)
    };
}
