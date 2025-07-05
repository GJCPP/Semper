#include "mle_convker.h"

MLE_Convker::MLE_Convker(const std::vector<std::vector<std::vector<std::vector<Goldilocks2::Element>>>>& kernel, size_t C, size_t D, size_t n, size_t m)
    : C(C), D(D), n(n), m(m), expanded(false) {

    assert(kernel.size() == C && kernel[0].size() == D && kernel[0][0].size() == m && kernel[0][0][0].size() == m);

    logC = find_ceiling_log2(C);
    logD = find_ceiling_log2(D);
    lognm = find_ceiling_log2(n * m);
    logmm = find_ceiling_log2(m * m);
    num_vars = logC + logD + lognm;
    evaluations.resize(1ull << (logC + logD + logmm), Goldilocks2::zero());

    size_t off_c = (1 << logD + logmm), off_d = (1 << logmm);
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

MLE_Convker::MLE_Convker(const array_view<Goldilocks2::Element>& kernel, size_t C, size_t D, size_t n, size_t m)
    : C(C), D(D), n(n), m(m), expanded(false) {

    assert(kernel.shape(0) == C && kernel.shape(1) == D && kernel.shape(2) == m && kernel.shape(3) == m);

    logC = find_ceiling_log2(C);
    logD = find_ceiling_log2(D);
    lognm = find_ceiling_log2(n * m);
    logmm = find_ceiling_log2(m * m);
    num_vars = logC + logD + lognm;
    evaluations.resize(1ull << (logC + logD + logmm), Goldilocks2::zero());

    size_t off_c = (1 << logD + logmm), off_d = (1 << logmm);
    for (size_t c = 0; c < C; ++c) {
        for (size_t d = 0; d < D; ++d) {
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    evaluations[c * off_c + d * off_d + i * m + j] = kernel(c, d, m - i - 1, m - j - 1); // Reverse the kernel
                }
            }
        }
    }
}

Goldilocks2::Element MLE_Convker::eval_hypercube(size_t mask) const {
    if (expanded) {
        return evaluations[mask];
    }
    auto real_index = to_real_index(mask);
    if (real_index) {
        return evaluations[*real_index];
    }
    return Goldilocks2::zero();
}

void MLE_Convker::fix(size_t pos, const Goldilocks2::Element& val) {
    if (!expanded && num_vars - pos <= lognm) { // Need to expand
        expand();
    }
    if (expanded) return MultilinearPolynomial::fix(pos, val);
    size_t real_num_vars = num_vars + logmm - lognm;
    pos = real_num_vars - pos - 1; // reverse the pos, 0 -> MSB
    std::vector<Goldilocks2::Element> new_evs(1ull << (real_num_vars - 1));
    Goldilocks2::Element one_minus_val = Goldilocks2::one() - val;
    for (size_t i = 0; i < (1ull << real_num_vars); ++i) {
        size_t index = (i & ((1ull << pos) - 1)) | ((i >> (pos + 1)) << pos);
        if ((i >> (pos)) & 1) {
            new_evs[index] = new_evs[index] + evaluations[i] * val;
        }
        else {
            new_evs[index] = new_evs[index] + evaluations[i] * one_minus_val;
        }
    }
    evaluations = std::move(new_evs);
    --num_vars;
    if (num_vars == lognm) {
        expand();
    }
}

void MLE_Convker::expand() {
    if (expanded) return;
    std::vector<Goldilocks2::Element> new_evs(1ull << num_vars, Goldilocks2::zero());
    for (size_t i = 0; i < (1ull << num_vars); ++i) {
        new_evs[i] = eval_hypercube(i);
    }
    evaluations = std::move(new_evs);
    expanded = true;
}

std::optional<size_t> MLE_Convker::to_real_index(size_t mask) const {
    if (expanded) {
        return mask;
    }
    size_t c, d, ij, i, j;
    const size_t off_c = (1 << logD + logmm), off_d = (1 << logmm);
    ij = mask & ((1 << lognm) - 1);
    i = ij / n;
    j = ij % n;

    if (i >= m || j >= m) return std::nullopt;

    c = (mask >> (logD + lognm)) & ((1 << logC) - 1);
    d = (mask >> lognm) & ((1 << logD) - 1);

    return c * off_c + d * off_d + i * m + j;
}

size_t MLE_Convker::to_virtual_index(size_t mask) const {
    size_t c, d, ij, i, j;
    c = (mask >> (logD + logmm)) & ((1 << logC) - 1);
    d = (mask >> logmm) & ((1 << logD) - 1);
    ij = mask & ((1 << logmm) - 1);
    i = ij / m;
    j = ij % m;

    return (c << (logD + lognm)) | (d << lognm) | (i * n + j);
}


MultilinearPolynomial MLE_Convker::sum_over_lowbits_with_power(size_t len, Goldilocks2::Element beta) const {
    assert(len == lognm);
    std::vector<Goldilocks2::Element> evs(1ull << (num_vars - len));
    Goldilocks2::Element beta_delta = pow(beta, n - m);
    for (size_t i = 0; i < (1ull << (num_vars - len)); ++i) {
        Goldilocks2::Element power = Goldilocks2::one();
        for (size_t x = 0; x < m; ++x) {
            for (size_t y = 0; y < m; ++y) {
                evs[i] += evaluations[(i << logmm) | (x * m + y)] * power;
                power = power * beta;
            }
            power = power * beta_delta;
        }
    }
    return MultilinearPolynomial(evs);
}

void MLE_Convker::iterate_nonzero(const std::function<void(size_t)> f, size_t offset) const {
    if (expanded) {
        return MultilinearPolynomial::iterate_nonzero(f, offset);
    }
    size_t real_offset = (offset >> lognm);
    for (size_t i = 0; i < real_offset; ++i) {
        for (size_t x = 0; x < m; ++x) {
            for (size_t y = 0; y < m; ++y) {
                f((i << lognm) | (x * n + y));
            }
        }
    }
}

Goldilocks2::Element MLE_Convker::evaluate(const std::vector<Goldilocks2::Element>& input) const {
    assert(!expanded);
    assert(input.size() == num_vars);
    MLE_Convker copy = *this;
    for (auto& e : input) {
        copy.fix(0, e);
    }
    return copy.evaluations[0];
}
