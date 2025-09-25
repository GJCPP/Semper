#include <cassert>

#include <string>
#include <chrono>

#include "mle.h"
#include "util.h"
#include "ligero.h"

MultilinearPolynomial::MultilinearPolynomial(int num_vars, bool init)
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
    auto start = std::chrono::high_resolution_clock::now();

    const int dim = val_table.get_dims();
    std::vector<size_t> dims(dim), log_upper_dims(dim);

    size_t total_input_size = 1;
    size_t total_output_size = 1;

    // Compute dims, upper_dims, and num_vars
    for (int i = 0; i < dim; ++i) {
        dims[i] = val_table.shape(i);
        int bit = find_ceiling_log2(dims[i]);
        log_upper_dims[i] = bit;
        num_vars += bit;
        total_input_size *= dims[i];
        total_output_size *= (1ull << bit);
    }

    evaluations.resize(total_output_size);

    // const Goldilocks2::Element* input_data = val_table.get_data();
    Goldilocks2::Element* output_data = evaluations.data();

    // Index vector for iteration
    std::vector<size_t> ind(dim, 0);

    for (size_t i = 0; i < total_output_size; ++i) {
        bool in_bounds = true;
        // size_t input_offset = 0;

        // Map flat index to multi-dimensional index (ind)
        size_t tmp = i;
        for (int d = dim - 1; d >= 0; --d) {
            ind[d] = (tmp & ((1ull << log_upper_dims[d]) - 1));
            if (ind[d] >= dims[d]) {
                in_bounds = false;
                break;
            }
            tmp >>= log_upper_dims[d];
        }

        // Copy or pad with zero
        if (in_bounds) {
            output_data[i] = val_table(ind);
        } else {
            output_data[i] = Goldilocks2::zero();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // std::cout << "MLE for shape ";
    // for (auto d : dims) std::cout << d << " ";
    // std::cout << "costs " << elapsed.count() << " seconds." << std::endl;

    // evaluations_view = val_table;
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
Goldilocks2::Element MultilinearPolynomial::evaluate(const std::vector<Goldilocks2::Element>& z) const {
    int loga = (num_vars >> 1), logb = num_vars - loga;
    size_t a = (1ull << loga), b = (1ull << logb);
    std::vector<Goldilocks2::Element> zh(z.begin(), z.begin() + loga);
    // low bits of z
    std::vector<Goldilocks2::Element> zl(z.begin() + loga, z.end());

    std::vector<Goldilocks2::Element> L = eq_table(logb, zl);
    std::vector<Goldilocks2::Element> R = eq_table(loga, zh);
    std::vector<Goldilocks2::Element> v(b, Goldilocks2::zero());
    for (size_t j = 0; j < a; ++j) {
        Goldilocks2::Element tmp;
        size_t offset = j * b;
        for (size_t i = 0; i < b; ++i) {
            v[i] += evaluations[offset + i] * R[j];
        }
    }
    return dot_product(v, L);
}

Goldilocks2::Element MultilinearPolynomial::open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const {
    return evaluate(z);
}

MultilinearPolynomial MultilinearPolynomial::operator+(const MultilinearPolynomial& g) const {
    assert(num_vars == g.get_num_vars());
    std::vector<Goldilocks2::Element> evs(evaluations.size());
    #pragma omp parallel for
    for (size_t i = 0;i < evaluations.size(); ++i) {
        evs[i] = evaluations[i] + g.evaluations[i];
    }
    return MultilinearPolynomial(evs);
}


MultilinearPolynomial MultilinearPolynomial::operator-(const MultilinearPolynomial& g) const {
    assert(num_vars == g.get_num_vars());
    std::vector<Goldilocks2::Element> evs(evaluations.size());
    #pragma omp parallel for
    for (size_t i = 0;i < evaluations.size(); ++i) {
        evs[i] = evaluations[i] - g.evaluations[i];
    }
    return MultilinearPolynomial(evs);
}

MultilinearPolynomial MultilinearPolynomial::operator*(const MultilinearPolynomial& g) const {
    if (num_vars != g.get_num_vars()) {
        throw std::invalid_argument("MultilinearPolynomial::operation *: polynomials must have same number of variables");
    }
    std::vector<Goldilocks2::Element> evs(evaluations.size());
    #pragma omp parallel for
    for (size_t i = 0;i < evaluations.size(); ++i) {
        evs[i] = evaluations[i] * g.evaluations[i];
    }
    return MultilinearPolynomial(evs);
}

MultilinearPolynomial MultilinearPolynomial::operator*(size_t scale) const {
    std::vector<Goldilocks2::Element> evs(evaluations.size());
    #pragma omp parallel for
    for (size_t i = 0;i < evaluations.size(); ++i) {
        evs[i] = evaluations[i] * Goldilocks2::fromU64(scale);
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
    assert(pos >= 0 && pos + val.size() <= size_t(num_vars));
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
    #pragma omp parallel for
    for (size_t i = 0; i < (1ull << (num_vars - len)); ++i) {
        for (size_t j = 0; j < (1ull << len); ++j) {
            evs[i] = evs[i] + evaluations[(i << len) | j];
        }
    }
    return MultilinearPolynomial(evs);
}

MultilinearPolynomial MultilinearPolynomial::prod_over_lowbits(size_t len) const {
    std::vector<Goldilocks2::Element> evs(1ull << (num_vars - len));
    #pragma omp parallel for
    for (size_t i = 0; i < (1ull << (num_vars - len)); ++i) {
        evs[i] = Goldilocks2::one();
        for (size_t j = 0; j < (1ull << len); ++j) {
            evs[i] = evs[i] * evaluations[(i << len) | j];
        }
    }
    return MultilinearPolynomial(evs);
}

MultilinearPolynomial MultilinearPolynomial::sum_over_lowbits_with_power(size_t len, Goldilocks2::Element beta) const {
    std::vector<Goldilocks2::Element> evs(1ull << (num_vars - len));
    #pragma omp parallel for
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

mle_aux_info MultilinearPolynomial::process_challenges(
    const std::vector<Goldilocks2::Element>& challenges) const {
    
    return mle_aux_info{.r = challenges, .comp = Goldilocks2::one()};
}

// mle_aux_info process_challenges(
//     int n,
//     const std::vector<size_t>& shape,
//     const std::vector<int>& order,
//     const std::vector<bool>& reversed,
//     const std::vector<Goldilocks2::Element>& challenges) {
    
    
//     assert(shape.size() == size_t(n));
//     assert(order.size() == size_t(n));
//     assert(reversed.size() == size_t(n));
//     mle_aux_info aux;

//     const Goldilocks2::Element one = Goldilocks2::one();
//     std::vector<size_t> start(n + 1);
//     for (int i = 0; i < n; ++i) {
//         start[i + 1] = find_ceiling_log2(shape[i]);
//     }
//     for (int i = 1; i <= n; ++i) {
//         start[i] += start[i - 1];
//     }
//     for (int i = 0; i < n; ++i) {
//         size_t ord = order[i];
//         size_t begin = start[ord], end = start[ord + 1];
//         bool rev = reversed[ord];
//         for (size_t j = begin; j < end; ++j) {
//             if (rev) {
//                 aux.r.push_back(one - challenges[j]);
//             } else {
//                 aux.r.push_back(challenges[j]);
//             }
//         }
//     }

//     aux.comp = Goldilocks2::one();
//     return aux;
// }
