#include "sign_check.h"
#include "util.h"

signProver::signProver(
    const std::vector<Goldilocks2::Element>& vec,
    const std::vector<Goldilocks2::Element>& sign,
    uint64_t scale, uint64_t max_val, uint64_t rho_inv)
    : scale(scale), max_val(max_val), rho_inv(rho_inv), num_vars(find_ceiling_log2(vec.size())), vec(vec), sign(sign) {
    if (!is_power_of_2(scale)) {
        throw std::invalid_argument("signProver: Scale must be a power of 2");
    }
    if (vec.size() != sign.size()) {
        throw std::invalid_argument("signProver: Vector and sign sizes must match");
    }
}

divProver signProver::next_prover(ligeropcs_base& pcs_quo, ligeropcs_base& pcs_rem, signProver& next_prover) const {
    auto [quo, rem] = get_quo_rem(vec, scale, false);
    pcs_quo = ligero_commit_base(quo, rho_inv);
    pcs_rem = ligero_commit_base(rem, rho_inv);
    next_prover = signProver(quo, sign, scale, max_val / scale, rho_inv);

    return divProver(vec, quo, rem, scale, false, rho_inv);
}

std::vector<Goldilocks2::Element> get_sign(const std::vector<Goldilocks2::Element>& vec) {
    size_t sz = vec.size();
    std::vector<Goldilocks2::Element> sign(sz);
    for (size_t i = 0; i < sz; ++i) {
        if (Goldilocks2::isNeg(vec[i])) {
            sign[i] = Goldilocks2::zero(); // (< 0) -> 0
        } else {
            sign[i] = Goldilocks2::one(); // (>= 0) -> 1
        }
    }
    return sign;
}

bool signVerifier::execute_sign_check(const signProver& prover,ligeropcs_base pcs_x, ligeropcs_base pcs_sign, uint64_t sec_param) {
    if (prover.final_round()) {
        // Final round, check if pcs_sign = pcs_x + 1
        auto alpha = random_vec_ext(prover.get_num_vars());
        if (pcs_x.open(alpha, sec_param) + Goldilocks2::one() != pcs_sign.open(alpha, sec_param)) {
            std::cerr << "Sign check failed: pcs_sign != pcs_x + 1" << std::endl;
            return false;
        }
    } else {
        ligeropcs_base pcs_quo, pcs_rem;
        signProver next_prover;
        divProver div_prover = prover.next_prover(pcs_quo, pcs_rem, next_prover);
        // 1. Check pcs_x = pcs_quo * scale + pcs_rem
        if (!divVerifier::execute_div_check(
            div_prover, pcs_x, pcs_quo, pcs_rem, sec_param)) {
            std::cerr << "Sign check failed: div check failed" << std::endl;
            return false;
        }
        // 2. Recursively check the next round
        if (!execute_sign_check(next_prover, pcs_quo, pcs_sign, sec_param)) {
            std::cerr << "Sign check failed: next round check failed" << std::endl;
            return false;
        }
    }
    return true;
}
