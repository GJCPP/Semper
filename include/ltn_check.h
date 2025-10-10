#pragma once

#include <vector>

#include "goldilocks_quadratic_ext.h"
#include "mle.h"
#include "ligero.h"
#include "sign_check.h"
#include "lazy_logup.h"

class ltnProver {
public:
    ltnProver() = default;

    ltnProver(const std::vector<Goldilocks2::Element>& vec,
              Goldilocks2::Element bar,
              uint64_t scale, uint64_t max_val, bool strict, 
              uint64_t rho_inv,
              lazyLogupProver* lazy_logup_prover);

    ltnProver(const std::vector<uint64_t>& vec,
              Goldilocks2::Element bar,
              uint64_t scale, uint64_t max_val, bool strict, uint64_t rho_inv,
              lazyLogupProver* lazy_logup_prover);

    void init_sub();
    
    void init_ltn();

    inline const std::vector<Goldilocks2::Element>& get_ltn() const { return ltn; }

    inline size_t get_num() const { return num; }

    inline bool use_lazy_logup() const {
        return lazy_logup_prover != nullptr;
    }

    signProver prove_rev_ltn(bool strict);

protected:
    std::vector<Goldilocks2::Element> vec, ltn, rev_ltn, sub; 
    Goldilocks2::Element bar; // The bar value for the check
    uint64_t scale, max_val;
    bool strict;
    uint64_t rho_inv, num;

    lazyLogupProver* lazy_logup_prover = nullptr;
};

class ltnVerifier {
public:
    static bool execute_ltn_check(
        ltnProver& prover,
        std::shared_ptr<oracle> pcs_vec,
        std::shared_ptr<oracle> pcs_ltn,
        Goldilocks2::Element bar,
        uint64_t max_val,
        bool strict,
        size_t sec_param,
        lazyLogupVerifier* lazy_logup_verifier);

    class resource {
    public:
        signVerifier::resource sign_res;
    };

    
    static resource pre_execute_ltn_check(
        ltnProver& prover,
        Goldilocks2::Element bar,
        uint64_t max_val,
        bool strict,
        lazyLogupVerifier* lazy_logup_verifier,
        std::shared_ptr<lazy_pcs_pool> pool);

        
    static bool execute_ltn_check(
        ltnProver& prover,
        std::shared_ptr<oracle> pcs_vec,
        std::shared_ptr<oracle> pcs_ltn,
        Goldilocks2::Element bar,
        uint64_t max_val,
        bool strict,
        size_t sec_param,
        lazyLogupVerifier* lazy_logup_verifier,
        resource& prev_resource);
};