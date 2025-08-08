#include "perm_check.h"
#include "product3_sumcheck.h"
#include "mle_open.h"

bool pdVerifier::execute_check(pdProver& prover, const oracle* pcsf1, const oracle* pcsf2, Goldilocks2::Element claim, uint64_t rho_inv, uint64_t sec_param) {

    int num_vars = prover.get_num_vars();
    ligeropcs_ext pcs_v = prover.commit_v(rho_inv);
    open_param ov_front({1, num_vars}, &pcs_v), ov_back({num_vars, 1}, &pcs_v);

    // Check product
    std::vector<Goldilocks2::Element> cha_110(num_vars + 1, Goldilocks2::one());
    cha_110.back() = Goldilocks2::zero();
    if (pcs_v.open(cha_110, sec_param) != claim) {
        std::cerr << "prodVerifier::execute_check : Product claim does not match." << std::endl;
        return false;
    }

    // Prove transition.
    auto cha = random_vec_ext(num_vars);
    MLE_Eq eq_cha(cha);
    p3Prover trans_prover = prover.prove_trans(cha);
    auto claim_trans = p3Verifier::partial_sumcheck(trans_prover, 
        ov_front(Goldilocks2::one())(cha).open(sec_param),
        sec_param);
    if (!claim_trans) {
        std::cerr << "prodVerifier::execute_check : Transition claim empty." << std::endl;
        return false;
    }
    if (claim_trans->claim != eq_cha.open(claim_trans->challenges, sec_param) *
        ov_back(claim_trans->challenges)(Goldilocks2::zero()).open(sec_param) *
        ov_back(claim_trans->challenges)(Goldilocks2::one()).open(sec_param)) {
        std::cerr << "prodVerifier::execute_check : Transition claim does not match." << std::endl;
        return false;
    }

    // Prove initialization.
    p3Prover init_prover = prover.prove_init(cha);
    auto claim_init = p3Verifier::partial_sumcheck(init_prover, 
        pcsf1->open(cha, sec_param),
        sec_param);
    if (!claim_init) {
        std::cerr << "prodVerifier::execute_check : Initialization claim empty." << std::endl;
        return false;
    }
    if (claim_init->claim != eq_cha.open(claim_init->challenges, sec_param) *
        pcsf2->open(claim_init->challenges, sec_param) *
        ov_front(Goldilocks2::zero())(claim_init->challenges).open(sec_param)) {
        std::cerr << "prodVerifier::execute_check : Initialization claim does not match." << std::endl;
        return false;
    }
    return true;
}

ligeropcs_ext pdProver::commit_v(uint64_t rho_inv) {
    std::vector<Goldilocks2::Element> evaluations(1 << (1 + num_vars));
    std::vector<Goldilocks2::Element> inv_f2(1 << num_vars);
    batch_inverse(inv_f2, f2->get_eval_table());
    size_t end = 1 << num_vars;
    for (size_t i = 0; i != end; ++i) { // first half
        evaluations[i] = f1->eval_hypercube(i) * inv_f2[i];
    }
    for (size_t i = 0; i != end; ++i) { // second half
        evaluations[end + i] = evaluations[i << 1] * evaluations[(i << 1) | 1];
    }
    v = MultilinearPolynomial(evaluations);
    return ligero_commit_ext(v, rho_inv);
}

p3Prover pdProver::prove_trans(const std::vector<Goldilocks2::Element>& cha) {
    auto v_0 = v, v_1 = v;
    v_0.fix(num_vars, Goldilocks2::zero());
    v_1.fix(num_vars, Goldilocks2::one());
    return p3Prover(MLE_Eq(cha), std::move(v_0), std::move(v_1));
}

    // prove f1(x) = v(0, x) f2(x)
p3Prover pdProver::prove_init(const std::vector<Goldilocks2::Element>& cha) {
    auto v_0 = v;
    v_0.fix(0, Goldilocks2::zero());
    return p3Prover(MLE_Eq(cha), std::move(v_0), *f2);
}
