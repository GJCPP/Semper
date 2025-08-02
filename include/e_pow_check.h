#pragma once

#include <map>

#include "ligero.h"
#include "mle.h"
#include "logup.h"
#include "ltn_check.h"
class eProver {
public:
    eProver(const std::vector<uint64_t>& from, const std::vector<uint64_t>& to, size_t scale, size_t max_val, size_t rho_inv);
    eProver(const std::vector<Goldilocks2::Element>& from, const std::vector<Goldilocks2::Element>& to, size_t scale, size_t max_val, size_t rho_inv);

    inline size_t get_scale() const { return scale; }
    inline size_t get_max_val() const { return max_val; }
    inline size_t get_rho_inv() const { return rho_inv; }
    inline const std::vector<uint64_t>& get_from() const { return from; }
    inline size_t get_num() const { return num; }

    LogupProver get_logup_prover(const std::vector<uint64_t>& e_from, const std::vector<uint64_t>& e_to);

    ltnProver prove_ltn(uint64_t bar);

    // Map element [>= bar] to 0
    ligeropcs_base commit_filtered_from();
    inline const std::vector<uint64_t>& get_filtered_from() const { return filtered_from; }

    // Map element [>= bar] to max_val - 1
    ligeropcs_base commit_masked_from();

    LogupProver get_masked_logup_prover(const std::vector<uint64_t>& e_from, const std::vector<uint64_t>& e_to);
protected:
    std::vector<uint64_t> from, to, filtered_from, masked_from;
    size_t num, scale, max_val, rho_inv;
    size_t bar;

    ligeropcs_base pcs_filtered_from, pcs_masked_from;
};

class eVerifier {
public:
    friend class eProver;
    static void init_e_table(size_t scale, size_t rho_inv);
    static bool execute_check(eProver& prover, ligeropcs_base pcs_from, ligeropcs_base pcs_to, size_t sec_param);
    static std::vector<size_t> get_exp_inv(const std::vector<size_t>& from, size_t scale, size_t rho_inv);
protected:
    static std::map<size_t, std::vector<uint64_t>> e_pow_from, e_pow_to;    
    static std::map<std::array<size_t, 2>, ligeropcs_base> pcs_e_pow_from, pcs_e_pow_to;
};
