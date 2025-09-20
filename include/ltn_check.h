#pragma once

#include <vector>

#include "goldilocks_quadratic_ext.h"
#include "mle.h"
#include "ligero.h"
#include "sign_check.h"
#include "lazy_logup.h"

template <typename field>
class ltnProver {
public:
    ltnProver() = default;

    ltnProver(const std::vector<field>& vec,
              field bar,
              uint64_t scale, uint64_t max_val, bool strict, 
              uint64_t rho_inv,
              lazyLogupProver* lazy_logup_prover);

    ltnProver(const std::vector<uint64_t>& vec,
              field bar,
              uint64_t scale, uint64_t max_val, bool strict, uint64_t rho_inv,
              lazyLogupProver* lazy_logup_prover);

    inline const std::vector<field>& get_ltn() const { return ltn; }

    inline ligeropcs_base get_pcs_ltn() const { return pcs_ltn; }

    inline size_t get_num() const { return num; }

    inline ligeropcs_base get_pcs_rev_ltn() const { 
        if (pcs_rev_ltn.empty()) throw std::runtime_error("ltnProver::get_pcs_rev_ltn: pcs_rev_ltn not initialized");
        return pcs_rev_ltn;
    }

    inline bool use_lazy_logup() const {
        return lazy_logup_prover != nullptr;
    }

    ligeropcs_base commit_sub();

    ligeropcs_base commit_rev_ltn();

    signProver prove_rev_ltn(bool strict);

protected:
    std::vector<field> vec, ltn, rev_ltn, sub; // The vector to check
    field bar; // The bar value for the check
    uint64_t scale, max_val;
    bool strict;
    uint64_t rho_inv, num;

    ligeropcs_base pcs_ltn, pcs_rev_ltn, pcs_sub;

    lazyLogupProver* lazy_logup_prover = nullptr;

    void init_ltn();
};

template <typename field>
class ltnVerifier {
public:
    static bool execute_ltn_check(
        ltnProver<field>& prover,
        std::shared_ptr<oracle<field>> pcs_vec,
        std::shared_ptr<oracle<field>> pcs_ltn,
        field bar,
        uint64_t max_val,
        bool strict,
        size_t sec_param,
        lazyLogupVerifier* lazy_logup_verifier);
};

