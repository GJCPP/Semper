#pragma once

#include "ligero.h"
#include "util.h"
#include "mle.h"
#include "mle_pow.h"
#include "mle_sumcheck.h"
#include "product2_sumcheck.h"
#include "mle_convker.h"
#include "array_view.h"

class convProver;
class convVerifier;

// X: C x N, W: C x D x K, Y: D x (N + K - 1)
class convTriple {
    friend class convProver;
    friend class convVerifier;
public:
    convTriple(const convTriple& other);
    convTriple(convTriple&& other) = default;

    convTriple(const std::vector<std::vector<Goldilocks2::Element>>& X,
        const std::vector<std::vector<std::vector<Goldilocks2::Element>>>& W,
        const std::vector<std::vector<Goldilocks2::Element>>& Y);

    convTriple(size_t C, size_t N, size_t D, size_t K,
        std::unique_ptr<MultilinearPolynomial> X,
        std::unique_ptr<MultilinearPolynomial> W,
        std::unique_ptr<MultilinearPolynomial> Y,
        int log_padding = 0, int log_n = 0, int log_m = 0);

    

    // For debugging only.
    bool check() const;


    std::array<ligeropcs_ext, 3> commit(size_t rho_inv) const;
    std::unique_ptr<MultilinearPolynomial> X, W, Y;

protected:
    size_t C, N, D, K;
    int logC, logN, logD, logK, logNK1;
    // aux info for 2d conv
    int log_padding, log_n, log_m;
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

    convTriple triple;
protected:
    // std::unique_ptr<MultilinearPolynomial> X_prime, W_prime;
    Goldilocks2::Element beta;

};

class convVerifier {
public:
    static bool execute_convcheck_1d(convProver& prover, const std::array<const oracle_ext*, 3>& oracle, const size_t& sec_param);
    static bool execute_convcheck_2d(convProver& prover, const std::array<const oracle_ext*, 3>& oracle, const size_t& sec_param);
protected:
    static std::optional<std::array<challenge_claim, 2>> execute_convcheck(
                                                            convProver& prover,
                                                            const std::array<const oracle_ext*, 3>& oracle,
                                                            const size_t& sec_param);
};

// Flatten the tensor and create a convProver
convProver make_conv_prover(
    const std::vector<std::vector<Goldilocks2::Element>>& X, // in_channels x N
    const std::vector<std::vector<std::vector<Goldilocks2::Element>>>& W, // in_channels x out_channels x kernel_size
    const std::vector<std::vector<Goldilocks2::Element>>& Y); // out_channels x (N + kernel_size - 1)

// Flatten the tensor and create a conv2Prover
convProver make_conv2_prover(
    size_t C, size_t D, size_t n, size_t m,
    const std::vector<std::vector<std::vector<Goldilocks2::Element>>>& X, // in_channels x n x n
    const std::vector<std::vector<std::vector<std::vector<Goldilocks2::Element>>>>& W); // in_channels x out_channels x m x m

// Flatten the tensor and create a conv2Prover
convProver make_conv2_prover(
    size_t C, size_t D, size_t n, size_t m, size_t padding,
    const array_view<Goldilocks2::Element>& X, // in_channels x n x n
    const array_view<Goldilocks2::Element>& W, // in_channels x out_channels x m x m
    const array_view<Goldilocks2::Element>& Y // out_channels x (n + 2 * padding - m + 1) x (n + 2 * padding - m + 1)
);

