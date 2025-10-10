#include "prod_check.h"
#include "product3_sumcheck.h"

#include "counter.h"

prodProver::prodProver(const MultilinearPolynomial& raw, const MultilinearPolynomial& pro, int suffix_len, size_t rho_inv)
    : raw(raw), pro(pro), num_vars(raw.get_num_vars()), suffix_len(suffix_len), rho_inv(rho_inv) {
    if (raw.get_num_vars() != pro.get_num_vars() + suffix_len) {
        throw std::invalid_argument("prodProver: raw and pro must have compatible variable counts.");
    }
}

p3Prover prodProver::prove_next(const MLE_Eq& cha, ligeropcs_base& pcs_new_raw) {
    if (cha.get_num_vars() != raw.get_num_vars() - 1) {
        throw std::runtime_error("prodProver::prove_next: Challenge size does not match raw polynomial variable count.");
    }
    MultilinearPolynomial new_raw = raw.prod_over_lowbits(1);
    MultilinearPolynomial raw_0 = raw, raw_1 = raw;
    raw_0.fix(num_vars - 1, Goldilocks2::zero());
    raw_1.fix(num_vars - 1, Goldilocks2::one());
    pcs_new_raw = ligero_commit_base(new_raw, rho_inv);
    
    raw = new_raw;
    --num_vars;
    --suffix_len;

    return p3Prover(cha, raw_0, raw_1);
}

p3Prover prodProver::prove_next(const MLE_Eq& cha) {
    if (cha.get_num_vars() != raw.get_num_vars() - 1) {
        throw std::runtime_error("prodProver::prove_next: Challenge size does not match raw polynomial variable count.");
    }
    MultilinearPolynomial new_raw = raw.prod_over_lowbits(1);
    MultilinearPolynomial raw_0 = raw, raw_1 = raw;
    raw_0.fix(num_vars - 1, Goldilocks2::zero());
    raw_1.fix(num_vars - 1, Goldilocks2::one());
    
    raw = new_raw;
    --num_vars;
    --suffix_len;

    return p3Prover(cha, raw_0, raw_1);
}

lazy_pcs prodProver::pre_prove_next(std::shared_ptr<lazy_pcs_pool> pool) {
    MultilinearPolynomial new_raw = raw.prod_over_lowbits(1);
    MultilinearPolynomial raw_0 = raw, raw_1 = raw;
    raw_0.fix(num_vars - 1, Goldilocks2::zero());
    raw_1.fix(num_vars - 1, Goldilocks2::one());
    auto pcs_new_raw = commit_lazy_pcs(new_raw, pool);
    
    raw = new_raw;
    --num_vars;
    --suffix_len;

    return pcs_new_raw;
}

bool prodVerifier::execute_prod_check(prodProver& prover, open_param raw, open_param pro, size_t sec_param) {
    startCounter counter("prod_proof");
    ligeropcs_base pcs_raw, pcs_new_raw;
    bool first = true;
    while (prover.suffix_len) {
        std::vector<Goldilocks2::Element> cha = random_vec_ext(prover.num_vars - 1);
        MLE_Eq eq_cha(cha);
        p3Prover pr = prover.prove_next(eq_cha, pcs_new_raw);
        auto claim = p3Verifier::partial_sumcheck(pr, pcs_new_raw.open(cha, sec_param), sec_param);
        if (!claim) return false;
        auto cha_0 = combine_challenges(claim->challenges, {Goldilocks2::zero()});
        auto cha_1 = combine_challenges(claim->challenges, {Goldilocks2::one()});
        if (first) {
            if (claim->claim != eq_cha.open(claim->challenges, sec_param) *
                raw.parse_all(cha_0).open(sec_param) *
                raw.parse_all(cha_1).open(sec_param)) return false;
            first = false;
        } else {
            if (claim->claim != eq_cha.open(claim->challenges, sec_param) *
                pcs_raw.open(cha_0, sec_param) *
                pcs_raw.open(cha_1, sec_param)) return false;
        }
        pcs_raw = pcs_new_raw;
    }
    auto cha = random_vec_ext(prover.num_vars);
    return pcs_raw.open(cha, sec_param) == pro.parse_all(cha).open(sec_param);
}

prodVerifier::resource prodVerifier::pre_execute_prod_check(prodProver& prover, std::shared_ptr<lazy_pcs_pool> pool) {
    resource ret;
    ret.pcs_raw.resize(prover.suffix_len);
    for (int rnd = prover.suffix_len - 1; rnd >= 0; --rnd) {
        std::vector<Goldilocks2::Element> cha = random_vec_ext(prover.num_vars - 1);
        MLE_Eq eq_cha(cha);
        ret.pcs_raw[rnd] = prover.pre_prove_next(pool);
    }
    return ret;
}

bool prodVerifier::execute_prod_check(prodProver& prover, open_param raw, open_param pro, size_t sec_param, prodVerifier::resource& res) {
    startCounter counter("prod_proof");
    lazy_pcs pcs_raw, pcs_new_raw;
    bool first = true;
    if (res.pcs_raw.size() != static_cast<size_t>(prover.suffix_len)) {
        throw std::runtime_error("prodVerifier::execute_prod_check: pcs_raw size does not match prover's suffix_len");
    }
    for (int rnd = prover.suffix_len - 1; rnd >= 0; --rnd) {
        std::vector<Goldilocks2::Element> cha = random_vec_ext(prover.num_vars - 1);
        MLE_Eq eq_cha(cha);
        p3Prover pr = prover.prove_next(eq_cha);
        pcs_new_raw = res.pcs_raw[rnd];
        auto claim = p3Verifier::partial_sumcheck(pr, pcs_new_raw.open(cha, sec_param), sec_param);
        if (!claim) return false;
        auto cha_0 = combine_challenges(claim->challenges, {Goldilocks2::zero()});
        auto cha_1 = combine_challenges(claim->challenges, {Goldilocks2::one()});
        if (first) {
            if (claim->claim != eq_cha.open(claim->challenges, sec_param) *
                raw.parse_all(cha_0).open(sec_param) *
                raw.parse_all(cha_1).open(sec_param)) {
                return false;
            }
            first = false;
        } else {
            if (claim->claim != eq_cha.open(claim->challenges, sec_param) *
                pcs_raw.open(cha_0, sec_param) *
                pcs_raw.open(cha_1, sec_param)) {
                return false;
            }
        }
        pcs_raw = pcs_new_raw;
    }
    auto cha = random_vec_ext(prover.num_vars);
    return pcs_raw.open(cha, sec_param) == pro.parse_all(cha).open(sec_param);
}
