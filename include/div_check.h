#pragma once

#include <map>

#include "mle.h"
#include "logup.h"

std::array<std::vector<Goldilocks2::Element>, 2> get_quo_rem(
    const std::vector<Goldilocks2::Element>& num,
    uint64_t den,
    bool allow_neg_rem);


std::array<std::vector<uint64_t>, 2> get_quo_rem(
    const std::vector<uint64_t>& num,
    uint64_t den,
    bool allow_neg_rem);

std::vector<Goldilocks2::Element> get_rem(
    const std::vector<Goldilocks2::Element>& num,
    uint64_t den,
    const std::vector<Goldilocks2::Element>& quo,
    bool allow_neg_rem);

class divVerifier;

class divProver {
    friend class divVerifier;
public:
    divProver() = default;
    
    divProver(
        const std::vector<Goldilocks2::Element>& num,
        const std::vector<Goldilocks2::Element>& quo,
        const std::vector<Goldilocks2::Element>& rem,
        uint64_t denominator, bool allow_neg_rem, uint64_t rho_inv);

    divProver(
        const std::vector<uint64_t>& num,
        const std::vector<uint64_t>& quo,
        const std::vector<uint64_t>& rem,
        uint64_t denominator, bool allow_neg_rem, uint64_t rho_inv);

    inline int get_num_vars() const { return num_vars; }

    void init_range(uint64_t denominator, bool allow_neg_rem, uint64_t rho_inv);

    void init_zeros(size_t vec_len);

    uint64_t get_denom() const;

    LogupProver get_logup_prover() const;

    inline MLE get_mle_range() const {
        return mle_range[{denom, allow_neg_rem, rho_inv}];
    }

    inline MLE get_mle_valid() const {
        return mle_valid[{denom, allow_neg_rem, rho_inv}];
    }

    inline MLE get_mle_zeros() const {
        return mle_zeros[{rem_u64.size(), rho_inv}];
    }


protected:
    bool allow_neg_rem;
    uint64_t denom;
    uint64_t rho_inv;
    int num_vars;
    std::vector<uint64_t> num_u64, quo_u64, rem_u64;
    static std::map<std::array<uint64_t, 2>, std::vector<Goldilocks2::Element>> range; // (-denom, denom) or (0, denom) if allow_neg_rem is false
    static std::map<std::array<uint64_t, 2>, std::vector<Goldilocks2::Element>> valid; // = 0 if valid, = 1 if not
    static std::map<uint64_t, std::vector<Goldilocks2::Element>> zeros; // Assume all valid

    static std::map<std::array<uint64_t, 2>, std::vector<uint64_t>> range_u64, valid_u64;
    static std::map<uint64_t, std::vector<uint64_t>> zeros_u64;
    
    static std::map<std::array<uint64_t, 3>, MLE> mle_range, mle_valid;
    static std::map<std::array<uint64_t, 2>, MLE> mle_zeros;
};

class divVerifier {
public:
    static bool execute_div_check(
        const divProver& prover,
        const oracle& pcs_x,
        const oracle& pcs_quo,
        const oracle& pcs_rem,
        size_t sec_param
    );
};
