#pragma once

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include "goldilocks_quadratic_ext.h"
#include "mle.h"
#include "array_view.h"

const Goldilocks::Element ROOTS[33] = {
    Goldilocks::fromU64(0x1),
    Goldilocks::fromU64(18446744069414584320ULL),
    Goldilocks::fromU64(281474976710656ULL),
    Goldilocks::fromU64(16777216ULL),
    Goldilocks::fromU64(4096ULL),
    Goldilocks::fromU64(64ULL),
    Goldilocks::fromU64(8ULL),
    Goldilocks::fromU64(2198989700608ULL),
    Goldilocks::fromU64(4404853092538523347ULL),
    Goldilocks::fromU64(6434636298004421797ULL),
    Goldilocks::fromU64(4255134452441852017ULL),
    Goldilocks::fromU64(9113133275150391358ULL),
    Goldilocks::fromU64(4355325209153869931ULL),
    Goldilocks::fromU64(4308460244895131701ULL),
    Goldilocks::fromU64(7126024226993609386ULL),
    Goldilocks::fromU64(1873558160482552414ULL),
    Goldilocks::fromU64(8167150655112846419ULL),
    Goldilocks::fromU64(5718075921287398682ULL),
    Goldilocks::fromU64(3411401055030829696ULL),
    Goldilocks::fromU64(8982441859486529725ULL),
    Goldilocks::fromU64(1971462654193939361ULL),
    Goldilocks::fromU64(6553637399136210105ULL),
    Goldilocks::fromU64(8124823329697072476ULL),
    Goldilocks::fromU64(5936499541590631774ULL),
    Goldilocks::fromU64(2709866199236980323ULL),
    Goldilocks::fromU64(8877499657461974390ULL),
    Goldilocks::fromU64(3757607247483852735ULL),
    Goldilocks::fromU64(4969973714567017225ULL),
    Goldilocks::fromU64(2147253751702802259ULL),
    Goldilocks::fromU64(2530564950562219707ULL),
    Goldilocks::fromU64(1905180297017055339ULL),
    Goldilocks::fromU64(3524815499551269279ULL),
    Goldilocks::fromU64(7277203076849721926ULL)};

uint64_t convert_mask_to_u64(const std::string& mask, const size_t &nvar);

int find_ceiling_log2(const uint64_t& n);

bool is_power_of_2(const uint64_t& n);

// pad a vector to length of power of 2
void pad(std::vector<Goldilocks2::Element>& table, const Goldilocks2::Element dummy = Goldilocks2::zero());
void pad(std::vector<Goldilocks::Element>& table, const Goldilocks::Element dummy = Goldilocks::zero());
void pad(std::vector<uint64_t>& table, const uint64_t dummy = 0ull);

// evaluate \tilde{eq}(r, x) = \prod_{i=0}^{n-1} (1 - r_i x_i) in O(2^l) linear time
std::vector<Goldilocks2::Element> eq_table(const size_t& num_var, const std::vector<Goldilocks2::Element>& r);

// calculate the inverse of all elements in arr with calculating only one inverse
void batch_inverse(std::vector<Goldilocks2::Element>& inv, const std::vector<Goldilocks2::Element>& arr);

Goldilocks::Element random_base();

std::vector<Goldilocks::Element> random_vec_base(const size_t& n);

std::vector<uint64_t> random_vec_uint(const size_t& n);

Goldilocks2::Element random_ext();

std::vector<Goldilocks2::Element> random_vec_ext(const size_t& n);

void random_vec_ext(Goldilocks2::Element* arr, const size_t& n);

void random_vec_u64(Goldilocks2::Element* arr, const size_t& n);

std::vector<uint64_t> trange(const uint64_t& lbound, const uint64_t& ubound);

void alert(const std::string& mes);

void print_table(const std::vector<Goldilocks2::Element>& table);

void print_table(const std::vector<Goldilocks::Element>& table);

void print_table(const std::vector<size_t>& table);

void print_hash(const std::array<uint8_t, 32>& hash);

void print_bytes(const std::array<uint8_t, 16>& hash);

size_t bisearch(const std::vector<uint64_t>& arr, const uint64_t& val);

// evaluate a polynomial with its coefficients known as coefs at point x with Horners method
Goldilocks2::Element Horner(const std::vector<Goldilocks2::Element> &coefs, const Goldilocks2::Element& x);

std::vector<Goldilocks::Element> eval_with_ntt(std::vector<Goldilocks::Element> f, const size_t& N);

std::vector<Goldilocks2::Element> eval_with_ntt(std::vector<Goldilocks2::Element> f, const size_t& N);

// returns vec on base field
std::vector<Goldilocks::Element> eval_with_ntt_base(std::vector<Goldilocks2::Element> f, const size_t& N);

// returns vec on ext field
std::vector<Goldilocks2::Element> eval_with_ntt_ext(std::vector<Goldilocks2::Element> f, const size_t& N);

/*
  B[bin(i)] = i < u ? \beta^i : 0
  Compute MLE_B(r)
  See Page 7 of CNN-Verf
*/
Goldilocks2::Element eval_power_mle(const Goldilocks2::Element& beta, 
    const std::vector<Goldilocks2::Element>& r, const size_t& u, int l);

Goldilocks2::Element pow(Goldilocks2::Element beta, size_t u);

std::vector<Goldilocks2::Element> combine_challenges(const std::vector<Goldilocks2::Element>& c1, const std::vector<Goldilocks2::Element>& c2);
std::vector<Goldilocks2::Element> combine_challenges(const std::vector<Goldilocks2::Element>& c1, const std::vector<Goldilocks2::Element>& c2, const std::vector<Goldilocks2::Element>& c3);

// Return Y = X * W, where |Y| = |X| + |W| - 1
std::vector<Goldilocks2::Element> conv(const std::vector<Goldilocks2::Element>& X, const std::vector<Goldilocks2::Element>& W);

// Flatten the tensor
std::vector<Goldilocks2::Element> flatten(const std::vector<std::vector<Goldilocks2::Element>>& vec);
std::vector<Goldilocks2::Element> flatten(const std::vector<std::vector<std::vector<Goldilocks2::Element>>>& vec);
std::vector<Goldilocks2::Element> flatten(const std::vector<std::vector<std::vector<std::vector<Goldilocks2::Element>>>>& vec);

template <typename T>
std::vector<T> flatten_2d(const array_view<T>& vec, size_t padding, T pad_val = T()) {
    if (vec.get_dims() != 2) {
        throw std::invalid_argument("vec must be a 2D array");
    }
    size_t n1 = vec.shape(0), n2 = vec.shape(1);
    size_t sz = (n1 + 2 * padding) * (n2 + 2 * padding);
    size_t offset = n2 + 2 * padding;
    std::vector<T> res(sz, pad_val);
    for (size_t i = 0; i < n1; ++i) {
        for (size_t j = 0; j < n2; ++j) {
            res[(i + padding) * offset + (j + padding)] = vec(i, j);
        }
    }
    return res;
}

