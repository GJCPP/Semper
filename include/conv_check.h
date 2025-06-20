#pragma once

#include "ligero.h"
#include "util.h"
#include "mle.h"
#include "mle_pow.h"
#include "mle_sumcheck.h"
#include "product2_sumcheck.h"

class convProver;
class convVerifier;

// X: C x N, W: C x D x K, Y: D x (N + K - 1)
class convTriple {
    friend class convProver;
    friend class convVerifier;
public:
    convTriple(const std::vector<std::vector<Goldilocks2::Element>>& X,
        const std::vector<std::vector<std::vector<Goldilocks2::Element>>>& W,
        const std::vector<std::vector<Goldilocks2::Element>>& Y);

    convTriple(size_t C, size_t N, size_t D, size_t K,
        const MultilinearPolynomial& X,
        const MultilinearPolynomial& W,
        const MultilinearPolynomial& Y);

    

    // For debugging only.
    bool check() const;


    std::array<ligeropcs_ext, 3> commit(size_t rho_inv) const;

protected:
    size_t C, N, D, K;
    int logC, logN, logD, logK, logNK1;
    MultilinearPolynomial X, W, Y;
};

/*
    This implements the CNN prover in CNN-Verf.
*/
class convProver {
    friend class convVerifier;
public:
    convProver(const convTriple& triple);
    convProver(convTriple&& triple);

    // Fix beta and r_D, return the p2_prover for the RHS
    p2Prover fix_beta_r_D(const Goldilocks2::Element& beta, const std::vector<Goldilocks2::Element>& r_D);

    /* X'(c) = sum_n X(c, n), W'(c) = sum_k W(c, r_D, k)
     * Note that W is already fixed to r_D.
     * return the p2_prover for sum_c X'(c) * W'(c)
     */
    p2Prover shrink_XW();

    /*
     * Fix X''(n) = X(r_C, n) and W''(c) = W(r_C, r_D, k)
     * return sumcheck provers for X' = \sum_n X''(n) x beta^n and W' = \sum_c W''(c) x beta^c
     */
    std::array<p2Prover, 2> fix_r_C(const std::vector<Goldilocks2::Element>& r_C);

protected:
    convTriple triple;
    // std::unique_ptr<MultilinearPolynomial> X_prime, W_prime;
    Goldilocks2::Element beta;
};

class convVerifier {
public:
    static bool execute_convcheck(convProver& prover, const std::array<const ligeropcs_ext, 3>& oracle, const size_t& sec_param);
};

// Flatten the tensor and create a convProver
convProver make_conv_prover(
    const std::vector<std::vector<Goldilocks2::Element>>& X, // in_channels x N
    const std::vector<std::vector<std::vector<Goldilocks2::Element>>>& W, // in_channels x out_channels x kernel_size
    const std::vector<std::vector<Goldilocks2::Element>>& Y); // out_channels x (N + kernel_size - 1)
