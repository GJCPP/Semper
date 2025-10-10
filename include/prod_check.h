#pragma once

#include "mle.h"
#include "mle_open.h"
#include "product3_sumcheck.h"
#include "lazy_pcs.h"

class prodVerifier;
/*
    Given MLE raw(x, y), where y is of length suffix_len.
    Prove that pro(x) = \prod_y raw(x, y).
*/
class prodProver {
public:
    friend class prodVerifier;
    prodProver(const MultilinearPolynomial& raw, const MultilinearPolynomial& pro, int suffix_len, size_t rho_inv);

    p3Prover prove_next(const MLE_Eq& cha, ligeropcs_base& pcs_new_raw);
    p3Prover prove_next(const MLE_Eq& cha);

    lazy_pcs pre_prove_next(std::shared_ptr<lazy_pcs_pool> pool);

protected:
    MultilinearPolynomial raw, pro;
    int num_vars, suffix_len;
    size_t rho_inv;
};

class prodVerifier {
public:
    static bool execute_prod_check(
        prodProver& prover,
        open_param raw,
        open_param pro,
        size_t sec_param
    );

    struct resource {
        std::vector<lazy_pcs> pcs_raw;
    };

    static resource pre_execute_prod_check(prodProver& prover, std::shared_ptr<lazy_pcs_pool> pool);

    
    static bool execute_prod_check(
        prodProver& prover,
        open_param raw,
        open_param pro,
        size_t sec_param,
        resource& res
    );
};
