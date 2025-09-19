#include "mat_check.h"

template <typename field>
mat_mult_prover<field>::mat_mult_prover(
    size_t n, size_t m, size_t l,
    const MultilinearPolynomial<field>& X,
    const MultilinearPolynomial<field>& W,
    const MultilinearPolynomial<field>& Y)
    : n(n), m(m), l(l), X(X.clone()), W(W.clone()), Y(Y.clone()) {

    log_n = find_ceiling_log2(n);
    log_m = find_ceiling_log2(m);
    log_l = find_ceiling_log2(l);

    assert(X.get_num_vars() == log_n + log_m);
    assert(W.get_num_vars() == log_m + log_l);
    assert(Y.get_num_vars() == log_n + log_l);
}

template <typename field>
p2Prover<field> mat_mult_prover<field>::fix_alpha_beta(const std::vector<field>& alpha, const std::vector<field>& beta) {
    assert(alpha.size() == log_n);
    assert(beta.size() == log_l);

    // Fix alpha and beta
    X->fix(0, alpha); // [n, m] -> [0, m]
    W->fix(log_m, beta); // [m, l] -> [m, 0]

    // Create p2Prover for the product of X and W
    return p2Prover(std::move(X), std::move(W));
}

template <typename field>
std::array<size_t, 3> mat_mult_prover<field>::get_shape() const {
    return { n, m, l };
}

template <typename field>
std::array<int, 3> mat_mult_prover<field>::get_log_shape() const {
    return { log_n, log_m, log_l };
}

template <typename field>
bool mat_mult_verifier<field>::execute_mat_mult_check(
    mat_mult_prover<field>& prover,
    open_param<field> oX, open_param<field> oW, open_param<field> oY,
    size_t sec_param) {

    auto [log_n, log_m, log_l] = prover.get_log_shape();
    auto alpha = random_vec_ext(log_n), beta = random_vec_ext(log_l); // Challenges

    auto p2_prover = prover.fix_alpha_beta(alpha, beta);

    field claimed = oY(alpha)(beta).open(sec_param);
    auto cha = p2Verifier::partial_sumcheck(p2_prover, claimed, sec_param);

    if (!cha) return false;

    if (cha->claim != oX(alpha)(cha->challenges).open(sec_param) * oW(cha->challenges)(beta).open(sec_param)) {
        return false;
    }

    return true;
}
