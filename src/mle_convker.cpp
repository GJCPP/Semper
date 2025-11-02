#include "mle_convker.h"
#include "ligero.h"
#include "pad_check.h"


MLE_Convker::MLE_Convker(const array_view<Goldilocks2::Element>& kernel, size_t C, size_t D, size_t n, size_t m)
    : C(C), D(D), n(n), m(m), expanded(false) {

    assert(kernel.shape(0) == C && kernel.shape(1) == D && kernel.shape(2) == m && kernel.shape(3) == m);

    if (!is_power_of_2(n)) {
        throw std::invalid_argument("MLE_Convker: n must be a power of 2");
    }

    logC = find_ceiling_log2(C);
    logD = find_ceiling_log2(D);
    logn = find_ceiling_log2(n);
    logm = find_ceiling_log2(m);
    num_vars = logC + logD + logn + logm;
    evaluations.resize(1ull << (logC + logD + logm + logm), Goldilocks2::zero());

    size_t off_c = (1 << logD + logm + logm), off_d = (1 << logm + logm);
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t c = 0; c < C; ++c) {
        for (size_t d = 0; d < D; ++d) {
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    evaluations[c * off_c + d * off_d + i * (1 << logm) + j] = kernel(c, d, m - i - 1, m - j - 1); // Reverse the kernel
                }
            }
        }
    }

    // evaluations_view = kernel;
    // evaluations_view.reverse(2);
    // evaluations_view.reverse(3);
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
    if (!expanded && num_vars - pos <= logn + logm) { // Need to expand
        expand();
    }
    if (expanded) return MultilinearPolynomial::fix(pos, val);
    size_t real_num_vars = num_vars + logm - logn;
    pos = real_num_vars - pos - 1; // reverse the pos, 0 -> MSB
    std::vector<Goldilocks2::Element> new_evs(1ull << (real_num_vars - 1));
    Goldilocks2::Element one_minus_val = Goldilocks2::one() - val;
    #pragma omp parallel for schedule(static)
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
    if (num_vars == logn + logm) {
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
    const size_t off_c = (1 << (logD + logm + logm)), off_d = (1 << (logm + logm));
    ij = mask & ((1 << (logn + logm)) - 1);
    i = (ij >> logn);
    j = (ij & ((1 << logn) - 1));

    if (i >= m || j >= m) return std::nullopt;

    c = (mask >> (logD + logn + logm)) & ((1 << logC) - 1);
    d = (mask >> (logn + logm)) & ((1 << logD) - 1);

    return c * off_c + d * off_d + (i << logm) + j;
}

size_t MLE_Convker::to_virtual_index(size_t mask) const {
    size_t c, d, ij, i, j;
    c = (mask >> (logD + logm + logm)) & ((1 << logC) - 1);
    d = (mask >> (logm + logm)) & ((1 << logD) - 1);
    ij = mask & ((1 << (logm + logm)) - 1);
    i = (ij >> logm);
    j = (ij & ((1 << logm) - 1));

    return (c << (logD + logn + logm)) | (d << (logn + logm)) | (i << logn) | j;
}


MultilinearPolynomial MLE_Convker::sum_over_lowbits_with_power(size_t len, Goldilocks2::Element beta) const {
    assert(len == logn + logm);
    std::vector<Goldilocks2::Element> evs(1ull << (num_vars - len));
    Goldilocks2::Element beta_delta = pow(beta, n - m);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < (1ull << (num_vars - len)); ++i) {
        Goldilocks2::Element power = Goldilocks2::one();
        for (size_t x = 0; x < m; ++x) {
            for (size_t y = 0; y < m; ++y) {
                evs[i] += evaluations[(i << (logm + logm)) | ((x << logm) | y)] * power;
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
    size_t real_offset = (offset >> (logn + logm));
    for (size_t i = 0; i < real_offset; ++i) {
        for (size_t x = 0; x < m; ++x) {
            for (size_t y = 0; y < m; ++y) {
                f((i << (logn + logm)) | (x * (1 << logn)) | y);
            }
        }
    }
}

void MLE_Convker::get_pad_range(int& begin, int& end) const {
    begin = logC + logD + logm;
    end = begin + logn - logm;
}

mle_aux_info MLE_Convker::process_challenges(
    const std::vector<Goldilocks2::Element>& challenges) const {
    mle_aux_info aux;
    
    // handle padding
    int begin = logC + logD + logm;
    int end = begin + logn - logm;
    
    aux.r.reserve(challenges.size() - (end - begin));
    
    Goldilocks2::Element one = Goldilocks2::one();
    Goldilocks2::Element factor = one;
    for (int i = 0; i < static_cast<int>(challenges.size()); ++i) {
        if (i < begin || i >= end) {
            aux.r.push_back(challenges[i]);
        } else {
            factor = factor * (one - challenges[i]);
        }
    }
    mle_aux_info tem = MultilinearPolynomial::process_challenges(aux.r);
    aux.comp = factor * tem.comp;
    aux.r = tem.r;
    return aux;
}

Goldilocks2::Element MLE_Convker::evaluate(const std::vector<Goldilocks2::Element>& input) const {
    assert(!expanded);
    assert(input.size() == static_cast<size_t>(num_vars));
    MLE_Convker copy = *this;
    for (auto& e : input) {
        copy.fix(0, e);
    }
    return copy.evaluations[0];
}


