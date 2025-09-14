#include "ltn_check.h"
#include "util.h"

ltnProver::ltnProver(
    const std::vector<Goldilocks2::Element>& vec,
    Goldilocks2::Element bar,
    uint64_t scale, uint64_t max_val, bool strict, uint64_t rho_inv)
    : vec(vec), bar(bar), scale(scale), max_val(max_val), strict(strict), rho_inv(rho_inv) {
    num = vec.size();
    init_ltn();
}

ltnProver::ltnProver(
    const std::vector<uint64_t>& vec_u,
    Goldilocks2::Element bar,
    uint64_t scale, uint64_t max_val, bool strict, uint64_t rho_inv)
    : vec(vec_u.size()), bar(bar), scale(scale), max_val(max_val), strict(strict), rho_inv(rho_inv) {
    num = vec.size();
    ltn.resize(num);
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = Goldilocks2::fromU64(vec_u[i]);
    }
    init_ltn();
}

ligeropcs_base ltnProver::commit_sub() {
    sub.resize(num);
    for (size_t i = 0; i < num; ++i) {
        sub[i] = vec[i] - bar;
    }
    pcs_sub = ligero_commit_base(sub, rho_inv);
    return pcs_sub;
}

ligeropcs_base ltnProver::commit_rev_ltn() {
    rev_ltn.resize(num);
    for (size_t i = 0; i < num; ++i) {
        rev_ltn[i] = Goldilocks2::one() - ltn[i];
    }
    pcs_rev_ltn = ligero_commit_base(rev_ltn, rho_inv);
    return pcs_rev_ltn;
}

signProver ltnProver::prove_rev_ltn(bool _strict) {
    if (strict != _strict) {
        throw std::invalid_argument("ltnProver: Proving with different strictness than initialized");
    }
    return signProver(sub, rev_ltn, scale, 2 * max_val, !strict, rho_inv);
}

void ltnProver::init_ltn() {
    ltn.resize(num);
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
    }
    pcs_ltn = ligero_commit_base(ltn, rho_inv);
}

bool ltnVerifier::execute_ltn_check(
    ltnProver& prover,
    ligeropcs_base pcs_vec, ligeropcs_base pcs_ltn,
    Goldilocks2::Element bar,
    uint64_t max_val,
    bool strict,
    size_t sec_param) {

    size_t num = prover.get_num();
    int lognum = find_ceiling_log2(num);

    // Step 1. Commit/Prove sub, rev_ltn
    auto pcs_sub = prover.commit_sub(), pcs_rev_ltn = prover.commit_rev_ltn();
    auto sub_cha = random_vec_ext(lognum);
    if (pcs_vec.open(sub_cha, sec_param) != pcs_sub.open(sub_cha, sec_param) + bar) {
        std::cerr << "❌ LTN check failed: pcs_vec != pcs_sub + bar" << std::endl;
        return false;
    }
    if (pcs_rev_ltn.open(sub_cha, sec_param) != Goldilocks2::one() - pcs_ltn.open(sub_cha, sec_param)) {
        std::cerr << "❌ LTN check failed: pcs_rev_ltn != 1 - pcs_ltn" << std::endl;
        return false;
    }

    // Step 2. Prove rev_ltn = strict ? [sub >= 0] : [sub > 0]
    auto sign_prover = prover.prove_rev_ltn(strict);
    if (!signVerifier::execute_sign_check(
        sign_prover, 
        std::make_shared<ligeropcs_base>(pcs_sub), 
        std::make_shared<ligeropcs_base>(pcs_rev_ltn), 
        sec_param)) {
        std::cerr << "❌ LTN check failed: Sign check failed" << std::endl;
        return false;
    }
    return true;
}
