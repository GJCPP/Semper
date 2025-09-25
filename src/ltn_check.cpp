#include "ltn_check.h"
#include "util.h"

ltnProver::ltnProver(
    const std::vector<Goldilocks2::Element>& vec,
    Goldilocks2::Element bar,
    uint64_t scale, uint64_t max_val, bool strict, uint64_t rho_inv,
    lazyLogupProver* lazy_logup_prover)
    : vec(vec), bar(bar), scale(scale), max_val(max_val), strict(strict), rho_inv(rho_inv), lazy_logup_prover(lazy_logup_prover) {
    num = vec.size();
    init_ltn();
}

ltnProver::ltnProver(
    const std::vector<uint64_t>& vec_u,
    Goldilocks2::Element bar,
    uint64_t scale, uint64_t max_val, bool strict, uint64_t rho_inv,
    lazyLogupProver* lazy_logup_prover)
    : vec(vec_u.size()), bar(bar), scale(scale), max_val(max_val), strict(strict), rho_inv(rho_inv), lazy_logup_prover(lazy_logup_prover) {
    num = vec.size();
    ltn.resize(num);
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = Goldilocks2::fromU64(vec_u[i]);
    }
    init_sub();
    init_ltn();
}

void ltnProver::init_sub() {
    sub.resize(num);
    for (size_t i = 0; i < num; ++i) {
        sub[i] = vec[i] - bar;
    }
}

ligeropcs_base ltnProver::commit_sub() {
    return ligero_commit_base(sub, rho_inv);
}

lazy_pcs ltnProver::pre_commit_sub(lazy_pcs_pool* pool) {
    return commit_lazy_pcs(sub, pool);
}

signProver ltnProver::prove_rev_ltn(bool _strict) {
    if (strict != _strict) {
        throw std::invalid_argument("ltnProver: Proving with different strictness than initialized");
    }
    return signProver(sub, rev_ltn, scale, 2 * max_val, !strict, rho_inv, lazy_logup_prover);
}

void ltnProver::init_ltn() {
    ltn.resize(num);
    rev_ltn.resize(num);
    for (size_t i = 0; i < num; ++i) {
        if (strict) {
            if (Goldilocks2::toS64(vec[i]) < Goldilocks2::toS64(bar)) {
                ltn[i] = Goldilocks2::one(); // (< bar) -> 1
            } else {
                ltn[i] = Goldilocks2::zero(); // (>= bar) -> 0
            }
        } else {
            if (Goldilocks2::toS64(vec[i]) <= Goldilocks2::toS64(bar)) {
                ltn[i] = Goldilocks2::one(); // (<= bar) -> 1
            } else {
                ltn[i] = Goldilocks2::zero(); // (> bar) -> 0
            }
        }
        rev_ltn[i] = Goldilocks2::one() - ltn[i]; // rev_ltn = 1 - ltn
    }
}

bool ltnVerifier::execute_ltn_check(
    ltnProver& prover,
    std::shared_ptr<oracle> pcs_vec,
    std::shared_ptr<oracle> pcs_ltn,
    Goldilocks2::Element bar,
    uint64_t max_val,
    bool strict,
    size_t sec_param,
    lazyLogupVerifier* lazy_logup_verifier) {

    size_t num = prover.get_num();
    int lognum = find_ceiling_log2(num);

    bool use_lazy_logup = (lazy_logup_verifier != nullptr);
    if (use_lazy_logup != prover.use_lazy_logup()) {
        throw std::invalid_argument("ltnVerifier: disagree in lazy_logup.");
    }

    // Step 1. Commit/Prove sub
    auto pcs_sub = prover.commit_sub();
    oracle_sum pcs_rev_ltn;
    pcs_rev_ltn.add(pcs_ltn, Goldilocks2::negone());
    pcs_rev_ltn.add_const(Goldilocks2::one());
    auto sub_cha = random_vec_ext(lognum);
    if (pcs_vec->open(sub_cha, sec_param) != pcs_sub.open(sub_cha, sec_param) + bar) {
        std::cerr << "❌ LTN check failed: pcs_vec != pcs_sub + bar" << std::endl;
        return false;
    }

    // Step 2. Prove rev_ltn = strict ? [sub >= 0] : [sub > 0]
    auto sign_prover = prover.prove_rev_ltn(strict);
    
    if (!signVerifier::execute_sign_check(
        sign_prover, 
        std::make_shared<ligeropcs_base>(pcs_sub), 
        std::make_shared<oracle_sum>(pcs_rev_ltn), 
        sec_param,
        lazy_logup_verifier)) {
        std::cerr << "❌ LTN check failed: Sign check failed" << std::endl;
        return false;
    }
    return true;
}

ltnVerifier::resource ltnVerifier::pre_execute_ltn_check(
    ltnProver& prover,
    Goldilocks2::Element bar,
    uint64_t max_val,
    bool strict,
    lazyLogupVerifier* lazy_logup_verifier,
    lazy_pcs_pool *pool) {

    resource ret;
    size_t num = prover.get_num();
    int lognum = find_ceiling_log2(num);

    // Step 1. Commit/Prove sub
    ret.pcs_sub = prover.pre_commit_sub(pool);
    
    auto sign_prover = prover.prove_rev_ltn(strict);
    
    ret.sign_res = signVerifier::pre_execute_sign_check(sign_prover, pool, lazy_logup_verifier);
    return ret;
}

bool ltnVerifier::execute_ltn_check(
    ltnProver& prover,
    std::shared_ptr<oracle> pcs_vec,
    std::shared_ptr<oracle> pcs_ltn,
    Goldilocks2::Element bar, uint64_t max_val, bool strict,
    size_t sec_param,
    lazyLogupVerifier* lazy_logup_verifier,
    resource& res) {

    
    size_t num = prover.get_num();
    int lognum = find_ceiling_log2(num);

    bool use_lazy_logup = (lazy_logup_verifier != nullptr);
    if (use_lazy_logup != prover.use_lazy_logup()) {
        throw std::invalid_argument("ltnVerifier: disagree in lazy_logup.");
    }

    // Step 1. Commit/Prove sub
    auto pcs_sub = std::make_shared<lazy_pcs>(res.pcs_sub);
    oracle_sum pcs_rev_ltn;
    pcs_rev_ltn.add(pcs_ltn, Goldilocks2::negone());
    pcs_rev_ltn.add_const(Goldilocks2::one());
    auto sub_cha = random_vec_ext(lognum);
    if (pcs_vec->open(sub_cha, sec_param) != pcs_sub->open(sub_cha, sec_param) + bar) {
        std::cerr << "❌ LTN check failed: pcs_vec != pcs_sub + bar" << std::endl;
        return false;
    }

    // Step 2. Prove rev_ltn = strict ? [sub >= 0] : [sub > 0]
    auto sign_prover = prover.prove_rev_ltn(strict);
    
    if (!signVerifier::execute_sign_check(
        sign_prover, 
        pcs_sub, 
        std::make_shared<oracle_sum>(pcs_rev_ltn), 
        sec_param,
        lazy_logup_verifier,
        res.sign_res)) {
        std::cerr << "❌ LTN check failed: Sign check failed" << std::endl;
        return false;
    }
    return true;
}