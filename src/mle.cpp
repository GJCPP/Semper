#include "mle.h"
#include "util.h"
#include "ligero.h"
#include <string>
#include <cassert>

MultilinearPolynomial::MultilinearPolynomial(size_t num_vars, bool init)
    : num_vars(num_vars) {
    if (init) {
        evaluations.resize(1ull << num_vars, Goldilocks2::zero());
    }
}

MultilinearPolynomial::MultilinearPolynomial(const std::vector<Goldilocks2::Element>& evaluations)
    : evaluations(evaluations) {

    size_t r = find_ceiling_log2(evaluations.size());
    num_vars = r;
    this->evaluations.resize(1ull << num_vars);
}

MultilinearPolynomial::MultilinearPolynomial(const std::vector<std::vector<Goldilocks2::Element>>& x) {
    int log1 = find_ceiling_log2(x.size()), log2 = find_ceiling_log2(x[0].size());
    num_vars = log1 + log2;
    evaluations.resize(1ull << num_vars);
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < x[i].size(); ++j) {
            evaluations[(i << log2) + j] = x[i][j];
        }
    }
}

MultilinearPolynomial::MultilinearPolynomial(const std::vector<std::vector<std::vector<Goldilocks2::Element>>>& x) {
    int log1 = find_ceiling_log2(x.size()), log2 = find_ceiling_log2(x[0].size()), log3 = find_ceiling_log2(x[0][0].size());
    num_vars = log1 + log2 + log3;
    evaluations.resize(1ull << num_vars);
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < x[i].size(); ++j) {
            for (size_t k = 0; k < x[i][j].size(); ++k) {
                evaluations[(i << (log2 + log3)) + (j << log3) + k] = x[i][j][k];
            }
        }
    }
}

MultilinearPolynomial::MultilinearPolynomial(const std::vector<uint64_t>& val_table) {
    size_t r = find_ceiling_log2(val_table.size());
    num_vars = r;
    evaluations.resize(1ull << num_vars);
    for (size_t i = 0;i < val_table.size(); ++i) {
        evaluations[i] = Goldilocks2::fromU64(val_table[i]);
    }
}

MultilinearPolynomial::MultilinearPolynomial(const array_view<Goldilocks2::Element>& val_table)
    : num_vars(0) {
    int n = val_table.get_dims();
    std::vector<size_t> dims(n + 1), logdims(n + 1), pow_dims(n + 1);
    for (size_t i = 1; i <= n; ++i) {
        dims[i] = val_table.shape(i - 1);
        logdims[i] = find_ceiling_log2(dims[i]);
        num_vars += logdims[i];
        pow_dims[i] = 1ull << logdims[i];
    }
    evaluations.resize(1ull << num_vars);
    std::vector<size_t> offset(n + 1), ind(n + 1);
    size_t sz = evaluations.size(), last_sz = dims.back();
    
    assert(val_table.offset(n - 1) == 1);
    size_t i = 0;
    while (true) {
        assert(offset.back() < val_table.size());
        assert(i < sz);
        evaluations[i] = val_table.get(offset.back());
        ++ind.back();
        int high = n;
        bool out_flag = false;
        do {
            high = n;
            out_flag = false;
            while (high > 0 && ind[high] == pow_dims[high]) {
                ind[high] = 0;
                --high;
                ++ind[high];
            }
            ++i;
            if (high && ind[high] >= dims[high]) {
                out_flag = true;
                ++ind.back();
            }
        } while (out_flag);

        if (high == 0) {
            break;
        }

        for (int j = high; j <= n; ++j) {
            offset[j] = offset[j - 1] + ind[j] * val_table.offset(j - 1);
        }
    }
    evaluations_view = array_view<Goldilocks2::Element>(evaluations.data(), std::vector<size_t>(pow_dims.begin() + 1, pow_dims.end()));
}

void MultilinearPolynomial::set_value(const std::string& mask, const Goldilocks2::Element& c) {
    evaluations[convert_mask_to_u64(mask, num_vars)] = c;
}

void MultilinearPolynomial::set_value(const std::string& mask, const uint64_t& c) {
    evaluations[convert_mask_to_u64(mask, num_vars)] = Goldilocks2::fromU64(c);
}

Goldilocks2::Element MultilinearPolynomial::eval_hypercube(uint64_t mask) const {
    return evaluations[mask];
}

// point[0]: Most Significant Bit
Goldilocks2::Element MultilinearPolynomial::evaluate(const std::vector<Goldilocks2::Element>& point) const {
    // for denote purpose
    const std::vector<Goldilocks2::Element>& r = point;
    std::vector<Goldilocks2::Element> one_minus_r(num_vars);
    for (size_t i = 0; i < num_vars; ++i) {
        Goldilocks2::sub(one_minus_r[i], Goldilocks2::one(), r[i]);
    }

    // construct lagrage bases
    std::vector<Goldilocks2::Element> lag_basis;
    lag_basis.resize(1ull << num_vars, Goldilocks2::one());
    // every round add a new bit to the highest digit, so the reversed ri should be utilized
    for (uint64_t i = 0;i < num_vars; ++i) {
        for (uint64_t j = 0;j < (1ull << i); ++j) {
            Goldilocks2::mul(lag_basis[j + (1ull << i)], lag_basis[j], r[num_vars - i - 1]);
            Goldilocks2::mul(lag_basis[j], lag_basis[j], one_minus_r[num_vars - i - 1]);
        }
    }

    Goldilocks2::Element result = Goldilocks2::zero();
    // directly uses lagrange bases rather than creates a new tmp var
    for (size_t i = 0; i < lag_basis.size(); ++i) {
        // Goldilocks2::Element tmp;
        Goldilocks2::mul(lag_basis[i], lag_basis[i], evaluations[i]);
        Goldilocks2::add(result, result, lag_basis[i]);
    }
    return result;
}

Goldilocks2::Element MultilinearPolynomial::open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const {
    return evaluate(z);
}

MultilinearPolynomial MultilinearPolynomial::operator+(const MultilinearPolynomial& g) const {
    assert(num_vars == g.get_num_vars());
    std::vector<Goldilocks2::Element> evs(evaluations.size()), evalg = g.evaluations;
    for (size_t i = 0;i < evaluations.size(); ++i) {
        evs[i] = evaluations[i] + evalg[i];
    }
    return MultilinearPolynomial(evs);
}


MultilinearPolynomial MultilinearPolynomial::operator-(const MultilinearPolynomial& g) const {
    assert(num_vars == g.get_num_vars());
    std::vector<Goldilocks2::Element> evs(evaluations.size()), evalg = g.evaluations;
    for (size_t i = 0;i < evaluations.size(); ++i) {
        evs[i] = evaluations[i] - evalg[i];
    }
    return MultilinearPolynomial(evs);
}

Goldilocks2::Element MultilinearPolynomial::get_sum() const {
    Goldilocks2::Element sum = Goldilocks2::zero();
    for (const Goldilocks2::Element& v : evaluations) {
        Goldilocks2::add(sum, sum, v);
    }
    return sum;
}

void MultilinearPolynomial::iterate_nonzero(const std::function<void(size_t)> f, size_t offset) const {
    for (size_t i = 0; i < offset; ++i) {
        f(i);
    }
}

void MultilinearPolynomial::fix(size_t pos, const Goldilocks2::Element& val) {
    assert(pos >= 0 && pos < num_vars);
    pos = num_vars - pos - 1; // reverse the pos, 0 -> MSB
    std::vector<Goldilocks2::Element> new_evs(1ull << (num_vars - 1));
    Goldilocks2::Element one_minus_val = Goldilocks2::one() - val;
    for (size_t i = 0; i < (1ull << num_vars); ++i) {
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
}

void MultilinearPolynomial::fix(size_t pos, const std::vector<Goldilocks2::Element>& val) {
    assert(pos >= 0 && pos + val.size() <= num_vars);
    for (const Goldilocks2::Element& v : val) {
        fix(pos, v);
    }
}

void MultilinearPolynomial::fix(size_t pos, const std::vector<Goldilocks2::Element>::const_iterator begin, const std::vector<Goldilocks2::Element>::const_iterator end) {
    for (auto it = begin; it != end; ++it) {
        fix(pos, *it);
    }
}

MultilinearPolynomial MultilinearPolynomial::sum_over_lowbits(size_t len) const {
    std::vector<Goldilocks2::Element> evs(1ull << (num_vars - len));
    for (size_t i = 0; i < (1ull << (num_vars - len)); ++i) {
        for (size_t j = 0; j < (1ull << len); ++j) {
            evs[i] = evs[i] + evaluations[(i << len) | j];
        }
    }
    return MultilinearPolynomial(evs);
}

MultilinearPolynomial MultilinearPolynomial::sum_over_lowbits_with_power(size_t len, Goldilocks2::Element beta) const {
    std::vector<Goldilocks2::Element> evs(1ull << (num_vars - len));
    for (size_t i = 0; i < (1ull << (num_vars - len)); ++i) {
        Goldilocks2::Element power = Goldilocks2::one();
        for (size_t j = 0; j < (1ull << len); ++j) {
            evs[i] += evaluations[(i << len) | j] * power;
            power = power * beta;
        }
    }
    return MultilinearPolynomial(evs);
}

const std::vector<Goldilocks2::Element>& MultilinearPolynomial::get_eval_table() const {
    return evaluations;
}

bool MultilinearPolynomial::check_open(
    const ligeropcs_base *pcs,
    const std::vector<Goldilocks2::Element>& challenges,
    const Goldilocks2::Element& claim,
    const size_t& sec_param) const {
 
    return claim == pcs->open(get_open_r(challenges), sec_param);
}


bool MultilinearPolynomial::check_open(
    const ligeropcs_ext *pcs,
    const std::vector<Goldilocks2::Element>& challenges,
    const Goldilocks2::Element& claim,
    const size_t& sec_param) const {
 
    return claim == pcs->open(get_open_r(challenges), sec_param);
}

std::vector<Goldilocks2::Element> MultilinearPolynomial::get_open_r(const std::vector<Goldilocks2::Element>& challenges) const {

    int dims = evaluations_view.get_dims();
    std::vector<Goldilocks2::Element> r;
    std::vector<int> start(dims + 1);
    for (int i = 0; i < dims; ++i) {
        start[evaluations_view.get_order(i) + 1] = find_ceiling_log2(evaluations_view.shape(i));
    }
    for (int i = 1; i <= dims; ++i) {
        start[i] += start[i - 1];
    }
    Goldilocks2::Element one = Goldilocks2::one();
    for (int i = 0; i < dims; ++i) {
        int ind = evaluations_view.get_order(i);
        int begin = start[ind], end = start[ind + 1];
        bool reversed = evaluations_view.is_reversed(i);
        for (int j = begin; j < end; ++j) {
            if (reversed) {
                r.push_back(one - challenges[j]);
            } else {
                r.push_back(challenges[j]);
            }
        }
    }
    return r;
}
