#pragma once

#include "goldilocks_quadratic_ext.h"
#include "mle_sumcheck.h"
#include "product3_sumcheck.h"
#include "mle.h"
#include "mle_eq.h"
#include "ligero.h"
#include "lazy_pcs.h"
#include <vector>
#include <array>
#include <optional>
#include <random>

namespace LogupDef {
    typedef ligeropcs_base pcs_base;
    typedef ligeropcs_ext pcs_ext;
}

class LogupProver {
public:
    // using table_base = std::vector<Goldilocks::Element>;
    using table_base = std::vector<uint64_t>;
    using table_ext = std::vector<Goldilocks2::Element>;
    // should be replaced with a pcs
private:
    table_base f1, f2, t1, t2, c; // c : multiplicity
    table_ext g, h;
    MultilinearPolynomial polyg, polyh;
    // intermediate tables used for last 2 sumchecks
    table_ext denomg, denomh;
public:

    // Prove (f1[i], f2[i]) in { (t1[i], t2[i]) }
    LogupProver() = default;
    LogupProver(const table_base& f1, const table_base& f2, const table_base& t1, const table_base& t2);
    LogupProver(const table_ext& f1, const table_ext& f2, const table_ext& t1, const table_ext& t2);
    void calculate_multiplicities();
    void calculate_gh(const Goldilocks2::Element& gamma, const Goldilocks2::Element& lambda);

    LogupDef::pcs_base commit_c(const uint64_t& rho_inv);
    lazy_pcs commit_c(lazy_pcs_pool* pool_c);

    std::array<LogupDef::pcs_base, 4> commit_ft(const uint64_t& rho_inv);

    std::array<LogupDef::pcs_ext, 2> commit_gh(const uint64_t& rho_inv);
    std::array<lazy_pcs, 2> commit_gh(lazy_pcs_pool* pool);

    std::array<sProver, 2> firstProvers();
    std::array<p3Prover, 2> secondProvers(const std::vector<Goldilocks2::Element>& rg, const std::vector<Goldilocks2::Element>& rh);
};

class LogupVerifier {
public:
    static bool execute_logup(LogupProver& lpr, const uint64_t& rho_inv, const size_t& sec_param);
    static bool execute_logup(LogupProver& lpr, 
        const oracle& f1, const oracle& f2,
        const oracle& t1, const oracle& t2,
        const uint64_t& rho_inv, const size_t& sec_param);

    bool execute_logup_first_part(LogupProver& lpr, 
        const oracle& f1, const oracle& f2,
        const oracle& t1, const oracle& t2,
        const uint64_t& rho_inv, const size_t& sec_param,
        lazy_pcs_pool *pool_c);

    bool execute_logup_second_part(LogupProver& lpr, 
        const oracle& f1, const oracle& f2,
        const oracle& t1, const oracle& t2,
        const uint64_t& rho_inv, const size_t& sec_param,
        lazy_pcs_pool *pool_gh);
        
    bool execute_logup_third_part(LogupProver& lpr,
        const oracle& f1, const oracle& f2, 
        const oracle& t1, const oracle& t2, 
        const uint64_t& rho_inv, const size_t& sec_param);
private:
    static std::mt19937_64 gen;
    static std::uniform_int_distribution<uint64_t> dist;
    static Goldilocks2::Element randnum();
    static std::vector<Goldilocks2::Element> randvec(const uint64_t& n);

    Goldilocks2::Element gamma, lambda;
    lazy_pcs lazy_pcs_c, lazy_pcs_g, lazy_pcs_h;
};