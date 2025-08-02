#pragma once

#include <vector>

#include "goldilocks_quadratic_ext.h"
#include "mle.h"
#include "ligero.h"
#include "sign_check.h"

class ltnProver {
public:
    ltnProver() = default;

    ltnProver(const std::vector<Goldilocks2::Element>& vec,
              Goldilocks2::Element bar,
              uint64_t scale, uint64_t max_val, bool strict, uint64_t rho_inv);

    ltnProver(const std::vector<uint64_t>& vec,
              Goldilocks2::Element bar,
              uint64_t scale, uint64_t max_val, bool strict, uint64_t rho_inv);

    inline const std::vector<Goldilocks2::Element>& get_ltn() const { return ltn; }

    inline ligeropcs_base get_pcs_ltn() const { return pcs_ltn; }

    inline size_t get_num() const { return num; }

    inline ligeropcs_base get_pcs_rev_ltn() const { 
        if (pcs_rev_ltn.empty()) throw std::runtime_error("ltnProver::get_pcs_rev_ltn: pcs_rev_ltn not initialized");
        return pcs_rev_ltn;
    }

    ligeropcs_base commit_sub();

    ligeropcs_base commit_rev_ltn();

    signProver prove_rev_ltn(bool strict);

protected:
    std::vector<Goldilocks2::Element> vec, ltn, rev_ltn, sub; // The vector to check
    Goldilocks2::Element bar; // The bar value for the check
    uint64_t scale, max_val;
    bool strict;
    uint64_t rho_inv, num;

    ligeropcs_base pcs_ltn, pcs_rev_ltn, pcs_sub;

    void init_ltn();
};

class ltnVerifier {
public:
    static bool execute_ltn_check(
        ltnProver& prover,
        ligeropcs_base pcs_vec,
        ligeropcs_base pcs_ltn,
        Goldilocks2::Element bar,
        uint64_t max_val,
        bool strict,
        size_t sec_param);
};

