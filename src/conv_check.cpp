#include "conv_check.h"
#include "mle_pow.h"
#include "pad_check.h"

convProver::convProver(const convTriple& triple)
    : triple(triple) {
}

convProver::convProver(convTriple&& triple)
    : triple(std::move(triple)) {
}


bool convTriple::check() const {
    bool ret = true;
#pragma omp parallel for
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
                ret = false;
            }
        }
    }
    return ret;
}

p2Prover convProver::fix_beta_r_D(
    const Goldilocks2::Element& beta,
    const std::vector<Goldilocks2::Element>& r_D) {

    this->beta = beta;
    assert(r_D.size() == size_t(triple.logD));
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
    size_t C,
    size_t D,
    size_t n,
    size_t m,
    size_t padding,
    const array_view<Goldilocks2::Element>& X,
    const array_view<Goldilocks2::Element>& W,
    const array_view<Goldilocks2::Element>& Y) {

    size_t in = n + 2 * padding;
    size_t on = in - m + 1;
    // size_t Y_len = in * in + m * in - 1;

    assert(is_power_of_2(n));

    assert(X.shape(0) == C);
    assert(X.shape(1) == n);
    assert(X.shape(2) == n);
    assert(W.shape(0) == C);
    assert(W.shape(1) == D);
    assert(W.shape(2) == m);
    assert(W.shape(3) == m);
    assert(Y.shape(0) == D);
    assert(Y.shape(1) == on);
    assert(Y.shape(2) == on);

    size_t new_in = 1ull << find_ceiling_log2(in);
    size_t new_padding = ((new_in - n) >> 1);
    // size_t new_on = new_in - m + 1;
    size_t new_Y_len = new_in * new_in + m * new_in - 1;

    assert(is_power_of_2(new_padding) || new_padding == 0);

    std::vector<std::vector<Goldilocks2::Element>> X_1d(C);
    std::vector<std::vector<bool>> Y_1d_visited(D);
    std::vector<std::vector<Goldilocks2::Element>> Y_1d(D); // D x Y_len

#pragma omp parallel for
    for (size_t c = 0; c < C; ++c) {
        X_1d[c] = flatten_2d(X[c], new_padding);
        assert(X_1d[c].size() == new_in * new_in);
    }

#pragma omp parallel for
    for (size_t d = 0; d < D; ++d) {
        Y_1d_visited[d].resize(new_Y_len);
        Y_1d[d].resize(new_Y_len);
    }

    // Fill Y_1d with Y
#pragma omp parallel for
    for (size_t d = 0; d < D; ++d) {
        for (size_t i = 0; i < on; ++i) {
            for (size_t j = 0; j < on; ++j) {
                size_t deg = i * new_in + j + (new_in + 1) * (m + new_padding - padding - 1);
                assert(deg < new_Y_len);
                Y_1d[d][deg] = Y(d, i, j);
                Y_1d_visited[d][deg] = true;
            }
        }
        for (size_t i = 0; i < new_Y_len; ++i) {
            if (Y_1d_visited[d][i]) continue;

            // Check if there will be intersection
            size_t start = i + 1 + new_in - m;
            size_t x = start / new_in, y = start % new_in;
            if (x < new_padding || x > new_padding + n + m ||
                (y + m < new_padding && (y - m + new_in) % new_in > new_padding + n) ||
                (y > new_padding + n && (y + m) % new_in < new_padding)) {
                continue;
            }

            // Compute and fill Y_1d[d][i]
            for (size_t c = 0; c < C; ++c) {
                for (size_t k = 0; k < m; ++k) {
                    for (size_t l = 0; l < m; ++l) {

                        size_t delta = k * new_in + l;
                        if (i - delta < X_1d[c].size()) {
                            Y_1d[d][i] += X_1d[c][i - delta] * W(c, d, m - k - 1, m - l - 1);
                        }
                    }
                }
            }
        }
    }

    return convProver(convTriple(C, new_in * new_in, D, new_in * m,
        std::make_unique<MultilinearPolynomial>(std::move(X_1d)),
        std::make_unique<MLE_Convker>(W, C, D, new_in, m),
        std::make_unique<MultilinearPolynomial>(std::move(Y_1d)),
        new_padding != 0, find_ceiling_log2(new_in), find_ceiling_log2(m)));
}

void pad_weights(
    size_t C, size_t D, size_t n, size_t m, size_t padding_X,
    const array_view<Goldilocks2::Element>& X,
    const array_view<Goldilocks2::Element>& W,
    const array_view<Goldilocks2::Element>& Y,
    array<Goldilocks2::Element>& W_pad,
    array<Goldilocks2::Element>& Y_pad,
    size_t& new_m,
    size_t& new_padding_X,
    bool pad_right_bottom) {

    assert(X.shape(0) == C);
    assert(X.shape(1) == n);
    assert(X.shape(2) == n);
    assert(W.shape(0) == D);
    assert(W.shape(1) == C);
    assert(W.shape(2) == m);
    assert(W.shape(3) == m);
    assert(Y.shape(0) == D);
    assert(Y.shape(1) == n + 2 * padding_X - m + 1);
    assert(Y.shape(2) == n + 2 * padding_X - m + 1);

    if (is_power_of_2(m)) {
        W_pad = W;
        Y_pad = Y;
        new_m = m;
        new_padding_X = padding_X;
        return;
    }

    new_m = 1ull << find_ceiling_log2(m);
    size_t pad_m = new_m - m;
    new_padding_X = padding_X + pad_m;
    W_pad.init({ D, C, new_m, new_m });
    Y_pad.init({ D, n + 2 * new_padding_X - new_m + 1, n + 2 * new_padding_X - new_m + 1 });

    W_pad.view.mimic(W, { D, C, new_m, new_m });
    Y_pad.view.mimic(Y, { D, n + 2 * new_padding_X - new_m + 1, n + 2 * new_padding_X - new_m + 1 });

    for (size_t d = 0; d < D; ++d) {
        auto W_view_d = W[d];
        auto W_pad_view_d = W_pad.view[d];

        for (size_t c = 0; c < C; ++c) {
            auto W_view_c = W_view_d[c];
            auto W_pad_view_c = W_pad_view_d[c];

            for (size_t i = 0; i < m; ++i) {
                auto W_view_i = W_view_c[i];
                auto W_pad_view_i = pad_right_bottom ? W_pad_view_c[i] : W_pad_view_c[i + pad_m];

                for (size_t j = 0; j < m; ++j) {
                    if (pad_right_bottom) {
                        W_pad_view_i(j) = W_view_i(j);
                    }
                    else {
                        W_pad_view_i(j + pad_m) = W_view_i(j);
                    }
                }
            }
        }
    }


    size_t y_sz = Y.shape(1);

    for (size_t d = 0; d < D; ++d) {
        auto Y_view_d = Y[d];
        auto Y_pad_view_d = Y_pad.view[d];

        for (size_t i = 0; i < y_sz; ++i) {
            auto Y_view_i = Y_view_d[i];
            auto Y_pad_view_i = pad_right_bottom ? Y_pad_view_d[i + pad_m] : Y_pad_view_d[i];

            for (size_t j = 0; j < y_sz; ++j) {
                if (pad_right_bottom) {
                    Y_pad_view_i(j + pad_m) = Y_view_i(j);
                }
                else {
                    Y_pad_view_i(j) = Y_view_i(j);
                }
            }
        }
    }

    size_t on = Y_pad.view.shape(1); // Output size

    for (int64_t i = 0; i < int64_t(on); ++i) {
        int64_t x = i - new_padding_X;

        for (int64_t j = 0; j < int64_t(on); ++j) {
            int64_t y = j - new_padding_X;

            bool needs_compute = pad_right_bottom
                ? (x < -int64_t(padding_X) || y < -int64_t(padding_X))
                : (x + new_m > n + padding_X || y + new_m > n + padding_X);

            if (!needs_compute) continue;

            for (int64_t d = 0; d < int64_t(D); ++d) {
                Goldilocks2::Element acc = Goldilocks2::zero();

                for (int64_t c = 0; c < int64_t(C); ++c) {
                    for (int64_t k = 0; k < int64_t(new_m); ++k) {
                        int64_t xk = x + k;
                        if (xk < 0 || xk >= int64_t(n)) continue;

                        for (int64_t l = 0; l < int64_t(new_m); ++l) {
                            int64_t yl = y + l;
                            if (yl < 0 || yl >= int64_t(n)) continue;

                            acc += X(c, xk, yl) * W_pad(d, c, k, l);
                        }
                    }
                }

                Y_pad.view(d, i, j) = acc; // directly write result
            }
        }
    }
}

bool convVerifier::execute_convcheck_1d(convProver& prover, const std::array<const oracle*, 3>& oracle, size_t sec_param) {
    auto claim = execute_convcheck(prover, oracle, sec_param);
    if (!claim) return false;
    return claim->at(0).claim == oracle[0]->open(claim->at(0).challenges, sec_param) &&
        claim->at(1).claim == oracle[1]->open(claim->at(1).challenges, sec_param);
}

bool convVerifier::execute_convcheck_2d(
    convProver& prover,
    open_param X,
    open_param W,
    open_param Y,
    size_t rho_inv,
    size_t sec_param,
    bool base_com) {

    std::unique_ptr<oracle> pcs_flat_Y;
    if (base_com) {
        pcs_flat_Y = std::make_unique<ligeropcs_base>(ligero_commit_base(*prover.triple.Y, rho_inv));
    }
    else {
        pcs_flat_Y = std::make_unique<ligeropcs_ext>(ligero_commit_ext(*prover.triple.Y, rho_inv));
    }

    std::array<const oracle*, 3> ora_1d = {
        X.pcs, W.pcs, pcs_flat_Y.get()
    };
    auto claim = execute_convcheck(prover, ora_1d, sec_param);
    if (!claim) return false;
    // pad check W  
    auto aux_info = prover.triple.W->process_challenges(claim->at(1).challenges);
    if (W.parse_all(aux_info.r).open(sec_param) * aux_info.comp != claim->at(1).claim) {
        return false;
    }
    // check pad X
    if (prover.triple.padding == false) { // No padding
        auto factor = Goldilocks2::one();
        std::vector<Goldilocks2::Element> cha;
        if (prover.triple.C == 1) {
            factor -= claim->at(0).challenges[0];
            cha = { claim->at(0).challenges.begin() + 1, claim->at(0).challenges.end() };
        }
        else {
            cha = claim->at(0).challenges;
        }
        return X.parse_all(cha).open(sec_param) * factor == claim->at(0).claim;
    }

    const Goldilocks2::Element zero = Goldilocks2::zero(), one = Goldilocks2::one();
    size_t logC = prover.triple.logC;
    size_t logn = prover.triple.log_n;
    std::vector<Goldilocks2::Element>& r = claim->at(0).challenges;
    std::vector<Goldilocks2::Element> prefix(r.begin(), r.begin() + logC);
    if (prover.triple.C == 1)
        prefix.clear();
    size_t beg_x = logC + 2, end_x = logC + logn;
    size_t beg_y = end_x + 2, end_y = end_x + logn;
    Goldilocks2::Element x_val[2] = { r[beg_x - 2], r[beg_x - 1] };
    Goldilocks2::Element y_val[2] = { r[beg_y - 2], r[beg_y - 1] };
    std::vector<Goldilocks2::Element> x(r.begin() + beg_x - 1, r.begin() + end_x);
    std::vector<Goldilocks2::Element> y(r.begin() + beg_y - 1, r.begin() + end_y);
    Goldilocks2::Element A[2][2], res = Goldilocks2::zero();
    X = X(prefix);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            x[0] = i ? one : zero;
            y[0] = j ? one : zero;
            A[i][j] = X(x)(y).open(sec_param);
            res += A[i][j] *
                (i ? x_val[0] * (one - x_val[1]) : (one - x_val[0]) * x_val[1]) *
                (j ? y_val[0] * (one - y_val[1]) : (one - y_val[0]) * y_val[1]);
            // 0 -> 01, 1 -> 10
        }
    }
    if (prover.triple.C == 1) {
        res *= one - r[0];
    }
    if (res != claim->at(0).claim) {
        return false;
    }
    // TODO copy constraint between Y and Y_flatten
    return true;
}

std::optional<std::array<challenge_claim, 2>> convVerifier::execute_convcheck(
    convProver& prover,
    const std::array<const oracle*, 3>& oracle,
    const size_t& sec_param) {

    // Step 1: Verifier samples beta and r_D
    Goldilocks2::Element beta = random_ext();
    std::vector<Goldilocks2::Element> r_D(random_vec_ext(prover.triple.logD));
    p2Prover rhs_prover = prover.fix_beta_r_D(beta, r_D);

    // Step 2: Prover proves the RHS
    Goldilocks2::Element rhs = rhs_prover.get_sum();
    MLE_Pow rhs_beta(beta, prover.triple.logNK1, prover.triple.N + prover.triple.K - 2);
    std::optional<challenge_claim> claim = p2Verifier::partial_sumcheck(rhs_prover, rhs, sec_param);
    if (!claim) return std::nullopt;
    auto query_Y = combine_challenges(r_D, claim->challenges);

    if (claim->claim != oracle[2]->open(query_Y, sec_param) * rhs_beta.evaluate(claim->challenges)) {
        return std::nullopt;
    }

    // Step 3: Prover proves the LHS = RHS
    // Step 3.1: Prove \sum_c X'(c) * W'(c), end with r_C
    auto lhs_prover = prover.shrink_XW();
    auto claim_xw = p2Verifier::partial_sumcheck(lhs_prover, rhs, sec_param);
    if (!claim_xw) return std::nullopt;
    auto r_C = std::move(claim_xw->challenges);

    // Step 3.2: Prover claims x and w, proves separately X'(r_C) = \sum_n X(r_C, n) \beta^n
    // End with r_N for X and r_K for W
    auto [X_prover, W_prover] = prover.fix_r_C(r_C);
    Goldilocks2::Element x_val = X_prover.get_sum(), w_val = W_prover.get_sum();
    if (x_val * w_val != claim_xw->claim) {
        return std::nullopt;
    }
    auto claim_x = p2Verifier::partial_sumcheck(X_prover, x_val, sec_param);
    if (!claim_x) return std::nullopt;
    auto& r_N = claim_x->challenges;
    auto claim_w = p2Verifier::partial_sumcheck(W_prover, w_val, sec_param);
    if (!claim_w) return std::nullopt;
    auto& r_K = claim_w->challenges;

    // Step 4: Query the oracle
    // Query X(r_C, r_N) and W(r_C, r_D, r_K)
    auto query_X = combine_challenges(r_C, r_N);
    auto query_W = combine_challenges(r_C, r_D, r_K);

    MLE_Pow beta_X(beta, prover.triple.logN, prover.triple.N - 1);
    MLE_Pow beta_W(beta, prover.triple.logK, prover.triple.K - 1); // beta^r_K

    std::array<challenge_claim, 2> ret;
    ret[0] = { query_X, claim_x->claim / beta_X.evaluate(r_N) };
    ret[1] = { query_W, claim_w->claim / beta_W.evaluate(r_K) };
    return ret;
}




convTriple::convTriple(const convTriple& other)
    : X(std::make_unique<MultilinearPolynomial>(*other.X)),
    W(std::make_unique<MultilinearPolynomial>(*other.W)),
    Y(std::make_unique<MultilinearPolynomial>(*other.Y)),
    C(other.C), N(other.N), D(other.D), K(other.K),
    logC(other.logC), logN(other.logN), logD(other.logD), logK(other.logK), logNK1(other.logNK1)
{
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
    std::unique_ptr<MultilinearPolynomial> _Y,
    bool padding, int log_n, int log_m)
    : X(std::move(_X)),
    W(std::move(_W)),
    Y(std::move(_Y)),
    C(C), N(N), D(D), K(K), padding(padding), log_n(log_n), log_m(log_m)
{

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
