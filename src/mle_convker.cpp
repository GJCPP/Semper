#include "mle_convker.h"

MLE_Convker::MLE_Convker(const std::vector<std::vector<std::vector<std::vector<Goldilocks2::Element>>>>& kernel, size_t C, size_t D, size_t n, size_t m)
    : C(C), D(D), n(n), m(m) {

    assert(kernel.size() == C && kernel[0].size() == D && kernel[0][0].size() == m && kernel[0][0][0].size() == m);

    logC = find_ceiling_log2(C);
    logD = find_ceiling_log2(D);
    lognm = find_ceiling_log2(n * m);
    logmm = find_ceiling_log2(m * m);
    num_vars = logC + logD + lognm;
    evaluations.resize(1ull << num_vars, Goldilocks2::zero());

    size_t off_c = (1 << logD + lognm), off_d = (1 << lognm);
    for (size_t c = 0; c < C; ++c) {
        for (size_t d = 0; d < D; ++d) {
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    evaluations[c * off_c + d * off_d + i * m + j] = kernel[c][d][m - i - 1][m - j - 1]; // Reverse the kernel
                }
            }
        }
    }
}

Goldilocks2::Element MLE_Convker::eval_hypercube(size_t mask) const {
    size_t c, d, ij, i, j;
    const size_t off_c = (1 << logD + lognm), off_d = (1 << lognm);
    ij = mask & ((1 << lognm) - 1);
    i = ij / n;
    j = ij % n;

    if (i >= m || j >= m) return Goldilocks2::zero();

    c = (mask >> (logD + lognm)) & ((1 << logC) - 1);
    d = (mask >> lognm) & ((1 << logD) - 1);

    return evaluations[c * off_c + d * off_d + i * m + j];
}

MultilinearPolynomial MLE_Convker::sum_over_lowbits_with_power(size_t len, Goldilocks2::Element beta) const {
    assert(len == lognm);
    std::vector<Goldilocks2::Element> evs(1ull << (num_vars - len));
    Goldilocks2::Element beta_delta = pow(beta, n - m);
    for (size_t i = 0; i < (1ull << (num_vars - len)); ++i) {
        Goldilocks2::Element power = Goldilocks2::one();
        for (size_t x = 0; x < m; ++x) {
            for (size_t y = 0; y < m; ++y) {
                evs[i] = evs[i] + evaluations[(i << len) | (x * m + y)] * power;
                power = power * beta;
            }
            power = power * beta_delta;
        }
    }
    return MultilinearPolynomial(evs);
}

// Goldilocks2::Element MLE_Convker::evaluate(const std::vector<Goldilocks2::Element>& input) const {
//     MLE_Convker copy = *this;
//     copy.fix(0, input.begin(), input.end() - lognm);
//     MLE_Eq eq(input.end() - lognm, input.end());
//     Goldilocks2::Element ret = Goldilocks2::zero();
//     for (size_t i = 0; i < m; ++i) {
//         for (size_t j = 0; j < m; ++j) {
//             ret = ret + copy.evaluations[i * m + j] * eq.eval_hypercube(i * n + j);
//         }
//     }
//     return ret;
// }
