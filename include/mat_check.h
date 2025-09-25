#pragma once

#include "mle.h"
#include "mle_open.h"
#include "mle_sumcheck.h"
#include "product2_sumcheck.h"

// Prove Y = X * W
class mat_mult_prover {
public:
    mat_mult_prover(
            size_t n, size_t m, size_t l,
            const MultilinearPolynomial& X, // [n, m]
            const MultilinearPolynomial& W, // [m, l]
            const MultilinearPolynomial& Y); // [n, l];.

    p2Prover fix_alpha_beta(const std::vector<Goldilocks2::Element>& alpha, const std::vector<Goldilocks2::Element>& beta);

    std::array<size_t, 3> get_shape() const;
    std::array<int, 3> get_log_shape() const;
protected:
    size_t n, m, l;
    int log_n, log_m, log_l;
    std::unique_ptr<MultilinearPolynomial> X, W, Y;
};

class mat_mult_verifier {
public:
    static bool execute_mat_mult_check(
        mat_mult_prover& prover,
        open_param oX,
        open_param oW,
        open_param oY,
        size_t sec_param);
};