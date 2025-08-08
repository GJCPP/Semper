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

setProver::setProver(const std::vector<MultilinearPolynomial*> set1, const std::vector<MultilinearPolynomial*> set2) {
    if (set1.size() != set2.size()) {
        throw std::invalid_argument("setProver::setProver: f1 and f2 must have same length");
    }
    if (set1.size() == 0) {
        throw std::invalid_argument("setProver::setProver: f1 and f2 must not be empty");
    }
    n = static_cast<int>(set1.size());
    num_vars = set1[0]->get_num_vars();
    f1.resize(n);
    f2.resize(n);
    for (int i = 0; i < n; ++i) {
        f1[i] = set1[i]->clone();
        f2[i] = set2[i]->clone();
        if (f1[i]->get_num_vars() != num_vars || f2[i]->get_num_vars() != num_vars) {
            throw std::invalid_argument("setProver::setProver: f1 and f2 must have same number of variables");
        }
    }
}

std::array<ligeropcs_ext, 2> setProver::combine(const std::vector<Goldilocks2::Element>& cha, uint64_t rho_inv) {
    if (cha.size() != n) {
        throw std::invalid_argument("setProver::combine: challenge must have same length as f1 and f2");
    }
    std::vector<Goldilocks2::Element> eval1 = f1[0]->get_eval_table(), eval2 = f2[0]->get_eval_table();
    size_t sz = (1ull << num_vars);
    for (int i = 1; i < n; ++i) {
        const auto& t1 = f1[i]->get_eval_table(), &t2 = f2[i]->get_eval_table();
        for (size_t j = 0; j != sz; ++j) {
            eval1[j] += t1[j] * cha[i];
            eval2[j] += t2[j] * cha[i];
        }
    }
    for (size_t j = 0; j != sz; ++j) {
        eval1[j] += cha[0];
        eval2[j] += cha[0];
    }
    set1 = eval1;
    set2 = eval2;
    return { ligero_commit_ext(set1, rho_inv), ligero_commit_ext(set2, rho_inv) };
}

pdProver setProver::get_pd_prover() {
    return pdProver(set1, set2);
}

bool setVerifier::execute_check(
    setProver& prover,
    const std::vector<const oracle*>& pcs_f1,
    const std::vector<const oracle*>& pcs_f2,
    uint64_t rho_inv, uint64_t sec_param) {
    
    const int n = prover.get_n(), num_vars = prover.get_num_vars();
    auto alpha = random_vec_ext(n);
    auto [pcs_set1, pcs_set2] = prover.combine(alpha, rho_inv);

    // Check set[b] = alpha[0] + fb[0] \sum_i alpha[i] x fb[i]
    auto cha = random_vec_ext(num_vars);
    auto expect1 = alpha[0] + pcs_f1[0]->open(cha, sec_param);
    auto expect2 = alpha[0] + pcs_f2[0]->open(cha, sec_param);
    for (int i = 1; i < n; ++i) {
        expect1 += alpha[i] * pcs_f1[i]->open(cha, sec_param);
        expect2 += alpha[i] * pcs_f2[i]->open(cha, sec_param);
    }
    if (pcs_set1.open(cha, sec_param) != expect1 || pcs_set2.open(cha, sec_param) != expect2) {
        std::cerr << "setVerifier::execute_check : combination check fails" << std::endl;
        return false;
    }

    // Check \prod pcs_set1/pcs_set2 = 1
    pdProver pd_prover = prover.get_pd_prover();
    if (!pdVerifier::execute_check(pd_prover, &pcs_set1, &pcs_set2, Goldilocks2::one(), rho_inv, sec_param)) {
        std::cerr << "setVerifier::execute_check : product check fails" << std::endl;
        return false;
    }
    return true;
}
