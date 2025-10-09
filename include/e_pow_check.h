#pragma once

#include <map>

#include "ligero.h"
#include "mle.h"
#include "logup.h"
#include "lazy_logup.h"
#include "ltn_check.h"
class eProver {
public:
    eProver(const std::vector<uint64_t>& from, const std::vector<uint64_t>& to, size_t scale, size_t max_val, size_t rho_inv, lazyLogupProver *lazy_logup_prover = nullptr);
    eProver(const std::vector<Goldilocks2::Element>& from, const std::vector<Goldilocks2::Element>& to, size_t scale, size_t max_val, size_t rho_inv, lazyLogupProver *lazy_logup_prover = nullptr);

    void init();

    inline size_t get_scale() const { return scale; }
    inline size_t get_max_val() const { return max_val; }
    inline size_t get_rho_inv() const { return rho_inv; }
    inline const std::vector<uint64_t>& get_from() const { return from; }
    inline size_t get_num() const { return num; }

    LogupProver get_logup_prover(
        const std::vector<uint64_t>& e_from, 
        const std::vector<uint64_t>& e_to);

    std::array<size_t, 2> get_lazy_logup_prover(
        const std::vector<uint64_t>& e_from, 
        const std::vector<uint64_t>& e_to);

    ltnProver prove_ltn(uint64_t bar);

    ligeropcs_base commit_ltn();
    lazy_pcs pre_commit_ltn(lazy_pcs_pool *pool);

    // Map element [>= bar] to 0
    ligeropcs_base commit_filtered_from();
    lazy_pcs pre_commit_filtered_from(lazy_pcs_pool *pool);
    inline const std::vector<uint64_t>& get_filtered_from() const { return filtered_from; }

    // Map element [>= bar] to max_val - 1
    ligeropcs_base commit_masked_from();
    lazy_pcs pre_commit_masked_from(lazy_pcs_pool *pool);

    LogupProver get_masked_logup_prover(
        const std::vector<uint64_t>& e_from,
        const std::vector<uint64_t>& e_to);

    std::array<size_t, 2> get_lazy_masked_logup_prover(
        const std::vector<uint64_t>& e_from, 
        const std::vector<uint64_t>& e_to);

    bool use_lazy_logup() const { return lazy_logup_prover != nullptr; }
protected:
    std::vector<uint64_t> from, to, filtered_from, masked_from;
    size_t num, scale, max_val, rho_inv;
    size_t bar;
    ligeropcs_base pcs_filtered_from, pcs_masked_from;
    lazyLogupProver* lazy_logup_prover;
    ltnProver ltn_prover;
};

class eVerifier {
public:
    friend class eProver;
    static void init_e_table(size_t scale, size_t rho_inv);
    static bool execute_check(eProver& prover, std::shared_ptr<oracle> pcs_from, std::shared_ptr<oracle> pcs_to, size_t sec_param, lazyLogupVerifier* lazy_logup_verifier = nullptr);
    static std::vector<size_t> get_exp_inv(const std::vector<size_t>& from, size_t scale, size_t rho_inv);
    
    class resource {
    public:
    ltnVerifier::resource ltn_res;
        lazy_pcs pcs_ltn, pcs_filtered_from, pcs_masked_from;
    };

    static resource pre_execute_check(
        eProver& prover,
        lazyLogupVerifier* lazy_logup_verifier,
        lazy_pcs_pool *pool);

    static bool execute_check(eProver& prover, std::shared_ptr<oracle> pcs_from, std::shared_ptr<oracle> pcs_to, size_t sec_param, lazyLogupVerifier* lazy_logup_verifier, resource& res);

protected:
    static std::map<size_t, std::vector<uint64_t>> e_pow_from, e_pow_to;
    static std::map<size_t, MLE> mle_e_pow_from, mle_e_pow_to;
};
