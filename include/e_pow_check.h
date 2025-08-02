#pragma once

#include <map>

#include "ligero.h"
#include "mle.h"
#include "logup.h"
#include "div_check.h"
#include "sign_check.h"
class eProver {
public:
    eProver(const std::vector<uint64_t>& from, const std::vector<uint64_t>& to, size_t scale, size_t max_val, size_t rho_inv);

    inline size_t get_scale() { return scale; }
    inline size_t get_max_val() { return max_val; }
    inline size_t get_rho_inv() { return rho_inv; }

    LogupProver get_logup_prover(const std::vector<uint64_t>& e_from, const std::vector<uint64_t>& e_to);

    // Prove from = quo * n + rem
    divProver prove_div_n(size_t n, ligeropcs_base& pcs_quo, ligeropcs_base& pcs_rem);
    
    // Prove sign = [rem > 0]
    signProver prove_sign(ligeropcs_base& pcs_sign, ligeropcs_base& pcs_rev_sign);
protected:
    std::vector<uint64_t> from, to, quo, rem, sign, rev_sign;
    size_t num, scale, max_val, rho_inv;
};

class eVerifier {
public:
    friend class eProver;
    static void init_e_table(size_t scale, size_t rho_inv);
    static bool execute_check(eProver& prover, ligeropcs_base pcs_from, ligeropcs_base pcs_to, size_t sec_param);
protected:
    static std::map<size_t, std::vector<uint64_t>> e_pow_from, e_pow_to;    
    static std::map<std::array<size_t, 2>, ligeropcs_base> pcs_e_pow_from, pcs_e_pow_to;
};
