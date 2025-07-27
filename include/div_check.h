#pragma once

#include <map>

#include "mle.h"
#include "logup.h"

std::array<std::vector<Goldilocks2::Element>, 2> get_quo_rem(
    const std::vector<Goldilocks2::Element>& num,
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
    divProver(
        const std::vector<Goldilocks2::Element>& num,
        const std::vector<Goldilocks2::Element>& quo,
        const std::vector<Goldilocks2::Element>& rem,
        uint64_t denominator, bool allow_neg_rem, uint64_t rho_inv);

    inline int get_num_vars() const { return num_vars; }

    void init_range(uint64_t denominator, bool allow_neg_rem, uint64_t rho_inv);

    void init_zeros(size_t vec_len);

    uint64_t get_denom() const;

    LogupProver get_logup_prover() const;

    inline ligeropcs_base get_pcs_range() const {
        return pcs_range[{denom, allow_neg_rem, rho_inv}];
    }

    inline ligeropcs_base get_pcs_valid() const {
        return pcs_valid[{denom, allow_neg_rem, rho_inv}];
    }

    inline ligeropcs_base get_pcs_zeros() const {
        return pcs_zeros[{rem_u64.size(), rho_inv}];
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
    
    static std::map<std::array<uint64_t, 3>, ligeropcs_base> pcs_range, pcs_valid;
    static std::map<std::array<uint64_t, 2>, ligeropcs_base> pcs_zeros;
};

class divVerifier {
public:
    static bool execute_div_check(
        const divProver& prover,
        ligeropcs_base pcs_x,
        ligeropcs_base pcs_quo,
        ligeropcs_base pcs_rem,
        size_t sec_param
    );
};
