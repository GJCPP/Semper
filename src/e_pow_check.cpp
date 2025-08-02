#include "util.h"
#include "logup.h"
#include "e_pow_check.h"

eProver::eProver(const std::vector<size_t>& from, const std::vector<size_t>& to, size_t scale, size_t max_val, size_t rho_inv)
    : from(from), to(to), num(from.size()), scale(scale), max_val(max_val), rho_inv(rho_inv) {
    
    if (from.size() != to.size()) {
        throw std::invalid_argument("eProver: Size of 'from' and 'to' vectors must be the same.");
    }
    if (!is_power_of_2(from.size())) {
        throw std::invalid_argument("eProver: Size of 'from' vector must be a power of 2.");
    }
    for (const auto& v : from) {
        if (v >= max_val) {
            throw std::invalid_argument("eProver: Elements in 'from' vector must be less than max_val.");
        }
    }
    eVerifier::init_e_table(scale, rho_inv);
}

LogupProver eProver::get_logup_prover(const std::vector<uint64_t>& e_from, const std::vector<uint64_t>& e_to) {
    return LogupProver(from, to, e_from, e_to);
}

divProver eProver::prove_div_n(size_t n, ligeropcs_base& pcs_quo, ligeropcs_base& pcs_rem) {
    auto [_quo, _rem] = get_quo_rem(from, n, false);
    quo = std::move(_quo);
    rem = std::move(_rem);
    pcs_quo = ligero_commit_base(quo, rho_inv);
    pcs_rem = ligero_commit_base(rem, rho_inv);
    return divProver(from, quo, rem, n, false, rho_inv);
}

signProver eProver::prove_sign(ligeropcs_base& pcs_sign, ligeropcs_base& pcs_rev_sign) {
    sign.resize(num);
    rev_sign.resize(num);
    for (size_t i = 0; i < num; ++i) {
        sign[i] = (rem[i] > 0) ? 1 : 0;
        rev_sign[i] = 1 - sign[i];
    }
    pcs_sign = ligero_commit_base(sign, rho_inv);
    pcs_rev_sign = ligero_commit_base(rev_sign, rho_inv);
    return signProver(sign, rev_sign, scale, max_val, true, rho_inv);
}

void eVerifier::init_e_table(size_t scale, size_t rho_inv) {
    if (e_pow_from.find(scale) != e_pow_from.end()) {
        return; // Already initialized
    }
    auto& from = e_pow_from[scale] = {};
    auto& to = e_pow_to[scale] = {};
    int end = -1;
    for (int i = 0; i != end; ++i) {
        from.push_back(i);
        to.push_back(Goldilocks2::toU64(Goldilocks2::fromS64(std::round(std::exp(static_cast<double>(-i) / scale) * scale)))[0]);
        if (end == -1 && to.back() == 0) {
            end = (1 << find_ceiling_log2(i));
            if (end == i) break;
        }
    }
    pcs_e_pow_from[{scale, rho_inv}] = ligero_commit_base(from, rho_inv);
    pcs_e_pow_to[{scale, rho_inv}] = ligero_commit_base(to, rho_inv);
}

bool eVerifier::execute_check(eProver& prover, ligeropcs_base pcs_from, ligeropcs_base pcs_to, size_t sec_param) {
    const size_t scale = prover.get_scale();
    const size_t max_val = prover.get_max_val();
    const size_t rho_inv = prover.get_rho_inv();
    if (pcs_e_pow_from.find({scale, rho_inv}) == pcs_e_pow_from.end()) {
        throw std::invalid_argument("eVerifier: e_pow table not initialized for the given scale and rho_inv.");
    }
    const auto& e_pow_from = eVerifier::e_pow_from[scale];
    const auto& e_pow_to = eVerifier::e_pow_to[scale];
    const auto& pcs_e_pow_from = eVerifier::pcs_e_pow_from[{scale, rho_inv}];
    const auto& pcs_e_pow_to = eVerifier::pcs_e_pow_to[{scale, rho_inv}];
    const size_t n = e_pow_from.size();
    if (max_val < e_pow_from.size()) {
        // Just look up the table.
        auto logup_prover = prover.get_logup_prover(e_pow_from, e_pow_to);
        if (!LogupVerifier::execute_logup(logup_prover, pcs_from, pcs_to, pcs_e_pow_from, pcs_e_pow_to, rho_inv, sec_param)) {
            return false;
        }
    } else {
        // Step 1. Filter out [>= n] elements
        ligeropcs_base pcs_quo, pcs_rem;
        divProver div_prover = prover.prove_div_n(n, pcs_quo, pcs_rem);
        if (!divVerifier::execute_div_check(div_prover, pcs_from, pcs_quo, pcs_rem, sec_param)) {
            std::cerr << "❌ eVerifier: Division check failed." << std::endl;
            return false;
        }
    }
    return false;
}


std::map<size_t, std::vector<uint64_t>> eVerifier::e_pow_from, eVerifier::e_pow_to;    
std::map<std::array<size_t, 2>, ligeropcs_base> eVerifier::pcs_e_pow_from, eVerifier::pcs_e_pow_to;
