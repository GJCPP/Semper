#include <cassert>

#include <string>
#include <chrono>

#include "mle.h"
#include "util.h"
#include "ligero.h"

template <typename field>
MultilinearPolynomial<field>::MultilinearPolynomial(int num_vars, bool init)
    : num_vars(num_vars) {
    if (init) {
        evaluations.resize(1ull << num_vars);
    }
}

template <typename field>
MultilinearPolynomial<field>::MultilinearPolynomial(const std::vector<field>& evaluations)
    : evaluations(evaluations) {

    size_t r = find_ceiling_log2(evaluations.size());
    num_vars = r;
    this->evaluations.resize(1ull << num_vars);
}

template <typename field>
MultilinearPolynomial<field>::MultilinearPolynomial(const std::vector<std::vector<field>>& x) {
    int log1 = find_ceiling_log2(x.size()), log2 = find_ceiling_log2(x[0].size());
    num_vars = log1 + log2;
    evaluations.resize(1ull << num_vars);
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < x[i].size(); ++j) {
            evaluations[(i << log2) + j] = x[i][j];
        }
    }
}

template <typename field>
MultilinearPolynomial<field>::MultilinearPolynomial(const std::vector<std::vector<std::vector<field>>>& x) {
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

template <typename field>
MultilinearPolynomial<field>::MultilinearPolynomial(const std::vector<uint64_t>& val_table) {
    size_t r = find_ceiling_log2(val_table.size());
    num_vars = r;
    evaluations.resize(1ull << num_vars);
    for (size_t i = 0;i < val_table.size(); ++i) {
        evaluations[i] = Goldilocks2::fromU64(val_table[i]);
    }
}

template <typename field>
MultilinearPolynomial<field>::MultilinearPolynomial(const array_view<field>& val_table)
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

    const field* input_data = val_table.get_data();
    field* output_data = evaluations.data();

    // Index vector for iteration
    std::vector<size_t> ind(dim, 0);

    for (size_t i = 0; i < total_output_size; ++i) {
        bool in_bounds = true;
        size_t input_offset = 0;

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


template <typename field>
void MultilinearPolynomial<field>::set_value(const std::string& mask, const field& c) {
    evaluations[convert_mask_to_u64(mask, num_vars)] = c;
}

template <typename field>
field MultilinearPolynomial<field>::eval_hypercube(uint64_t mask) const {
    return evaluations[mask];
}

// point[0]: Most Significant Bit
template <typename field>
field MultilinearPolynomial<field>::evaluate(const std::vector<field>& z) const {
    int loga = (num_vars >> 1), logb = num_vars - loga;
    size_t a = (1ull << loga), b = (1ull << logb);
    std::vector<field> zh(z.begin(), z.begin() + loga);
    // low bits of z
    std::vector<field> zl(z.begin() + loga, z.end());

    std::vector<field> L = eq_table(logb, zl);
    std::vector<field> R = eq_table(loga, zh);
    std::vector<field> v(b);
    for (size_t j = 0; j < a; ++j) {
        field tmp;
        size_t offset = j * b;
        for (size_t i = 0; i < b; ++i) {
            v[i] += evaluations[offset + i] * R[j];
        }
    }
    return dot_product(v, L);
}

template <typename field>
field MultilinearPolynomial<field>::open(const std::vector<field>& z, const size_t& sec_param) const {
    return evaluate(z);
}

template <typename field>
MultilinearPolynomial<field> MultilinearPolynomial<field>::operator+(const MultilinearPolynomial<field>& g) const {
    assert(num_vars == g.get_num_vars());
    std::vector<field> evs(evaluations.size());
    #pragma omp parallel for
    for (size_t i = 0;i < evaluations.size(); ++i) {
        evs[i] = evaluations[i] + g.evaluations[i];
    }
    return MultilinearPolynomial<field>(evs);
}


template <typename field>
MultilinearPolynomial<field> MultilinearPolynomial<field>::operator-(const MultilinearPolynomial<field>& g) const {
    assert(num_vars == g.get_num_vars());
    std::vector<field> evs(evaluations.size());
    #pragma omp parallel for
    for (size_t i = 0;i < evaluations.size(); ++i) {
        evs[i] = evaluations[i] - g.evaluations[i];
    }
    return MultilinearPolynomial<field>(evs);
}

template <typename field>
MultilinearPolynomial<field> MultilinearPolynomial<field>::operator*(const MultilinearPolynomial<field>& g) const {
    if (num_vars != g.get_num_vars()) {
        throw std::invalid_argument("MultilinearPolynomial::operation *: polynomials must have same number of variables");
    }
    std::vector<field> evs(evaluations.size());
    #pragma omp parallel for
    for (size_t i = 0;i < evaluations.size(); ++i) {
        evs[i] = evaluations[i] * g.evaluations[i];
    }
    return MultilinearPolynomial(evs);
}

template <typename field>
MultilinearPolynomial<field> MultilinearPolynomial<field>::operator*(size_t scale) const {
    std::vector<field> evs(evaluations.size());
    #pragma omp parallel for
    for (size_t i = 0;i < evaluations.size(); ++i) {
        evs[i] = evaluations[i] * scale;
    }
    return MultilinearPolynomial(evs);
}

template <typename field>
field MultilinearPolynomial<field>::get_sum() const {
    field sum = {};
    for (const field& v : evaluations) {
        sum += v;
    }
    return sum;
}

template <typename field>
void MultilinearPolynomial<field>::iterate_nonzero(const std::function<void(size_t)> f, size_t offset) const {
    for (size_t i = 0; i < offset; ++i) {
        f(i);
    }
}

template <typename field>
void MultilinearPolynomial<field>::fix(size_t pos, const field& val) {
    assert(pos >= 0 && pos < num_vars);
    pos = num_vars - pos - 1; // reverse the pos, 0 -> MSB
    std::vector<field> new_evs(1ull << (num_vars - 1));
    field one_minus_val = Goldilocks2::one() - val;
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

template <typename field>
void MultilinearPolynomial<field>::fix(size_t pos, const std::vector<field>& val) {
    assert(pos >= 0 && pos + val.size() <= size_t(num_vars));
    for (const field& v : val) {
        fix(pos, v);
    }
}

template <typename field>
void MultilinearPolynomial<field>::fix(size_t pos, const std::vector<field>::const_iterator begin, const std::vector<field>::const_iterator end) {
    for (auto it = begin; it != end; ++it) {
        fix(pos, *it);
    }
}

template <typename field>
MultilinearPolynomial<field> MultilinearPolynomial<field>::sum_over_lowbits(size_t len) const {
    std::vector<field> evs(1ull << (num_vars - len));
    #pragma omp parallel for
    for (size_t i = 0; i < (1ull << (num_vars - len)); ++i) {
        for (size_t j = 0; j < (1ull << len); ++j) {
            evs[i] = evs[i] + evaluations[(i << len) | j];
        }
    }
    return MultilinearPolynomial(evs);
}

template <typename field>
MultilinearPolynomial<field> MultilinearPolynomial<field>::prod_over_lowbits(size_t len) const {
    std::vector<field> evs(1ull << (num_vars - len));
    #pragma omp parallel for
    for (size_t i = 0; i < (1ull << (num_vars - len)); ++i) {
        evs[i] = Goldilocks2::one();
        for (size_t j = 0; j < (1ull << len); ++j) {
            evs[i] = evs[i] * evaluations[(i << len) | j];
        }
    }
    return MultilinearPolynomial(evs);
}

template <typename field>
MultilinearPolynomial<field> MultilinearPolynomial<field>::sum_over_lowbits_with_power(size_t len, field beta) const {
    std::vector<field> evs(1ull << (num_vars - len));
    #pragma omp parallel for
    for (size_t i = 0; i < (1ull << (num_vars - len)); ++i) {
        field power = Goldilocks2::one();
        for (size_t j = 0; j < (1ull << len); ++j) {
            evs[i] += evaluations[(i << len) | j] * power;
            power = power * beta;
        }
    }
    return MultilinearPolynomial(evs);
}

template <typename field>
const std::vector<field>& MultilinearPolynomial<field>::get_eval_table() const {
    return evaluations;
}

template <typename field>
mle_aux_info<field> MultilinearPolynomial<field>::process_challenges(
    const std::vector<field>& challenges) const {
    
    return mle_aux_info{.r = challenges, .comp = field::one()};
}

// template <typename field>
// mle_aux_info<field> process_challenges(
//     int n,
//     const std::vector<size_t>& shape,
//     const std::vector<int>& order,
//     const std::vector<bool>& reversed,
//     const std::vector<field>& challenges) {
    
    
//     assert(shape.size() == size_t(n));
//     assert(order.size() == size_t(n));
//     assert(reversed.size() == size_t(n));
//     mle_aux_info aux;

//     const field one = Goldilocks2::one();
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
