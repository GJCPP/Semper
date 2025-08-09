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

setProver::setProver(const std::vector<const MLE*>& set1, const std::vector<const MLE*>& set2)
    : f1(set1), f2(set2) {
    if (set1.size() != set2.size()) {
        throw std::invalid_argument("setProver::setProver: f1 and f2 must have same length");
    }
    if (set1.size() == 0) {
        throw std::invalid_argument("setProver::setProver: f1 and f2 must not be empty");
    }
    n = static_cast<int>(set1.size());
    num_vars = set1[0]->get_num_vars();
    for (int i = 0; i < n; ++i) {
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


// prove f2(pi(x)) = f1(x)
permProverBase::permProverBase(const std::vector<size_t>& _perm)
    : perm(_perm) {
}

void permProverBase::add_mle(const MLE& new_f1, const MLE& new_f2) {
    f1.push_back(&new_f1);
    f2.push_back(&new_f2);
    const auto& evs1(new_f1.get_eval_table()), &evs2(new_f2.get_eval_table());
    if (f1.size() == 1) {
        sz = evs2.size();
        if (evs1.size() > evs2.size()) {
            throw std::invalid_argument("permProver::permProver : f1 must have size no larger than f2");
        }
        if (perm.size() != evs1.size()) {
            throw std::invalid_argument("permProver::permProver : permutation vector must have same size as f1");
        }
    } else {
        if (new_f1.get_num_vars() != f1[0]->get_num_vars() ||
            new_f2.get_num_vars() != f2[0]->get_num_vars()) {
            throw std::invalid_argument("permProver::permProver : all MLEs must have the same number of variables");
        }
    }
#ifdef DEBUG
    for (size_t i = 0; i < evs1.size(); ++i) {
        if (evs1[i] != evs2[perm[i]]) {
            throw std::invalid_argument("permProver::permProver : f2(pi(x)) != f1(x)");
        }
    }
#endif
}

std::vector<ligeropcs_base> permProverBase::commit_pad_f1(uint64_t rho_inv) {
    const size_t n1 = (1 << f1[0]->get_num_vars()), n2 = (1 << f2[0]->get_num_vars());
    const size_t len = f1.size();
    if (n1 == n2) {
        // No padding needed
        pad_f1.reserve(len);
        for (auto& ptr : f1) {
            pad_f1.push_back(*ptr);
        }
        return {};
    }

    std::vector<const std::vector<Goldilocks2::Element> *> evs1, evs2;
    std::vector<bool> vis(n2);
    for (size_t i = 0; i < len; ++i) {
        evs1.push_back(&f1[i]->get_eval_table());
        evs2.push_back(&f2[i]->get_eval_table());
    }
    std::vector<std::vector<Goldilocks2::Element>> new_evs(len, std::vector<Goldilocks2::Element>(n2, Goldilocks2::zero()));
    for (size_t i = 0; i < n1; ++i) {
        vis[perm[i]] = true;
    }
    for (size_t i = 0; i < len; ++i) {
        memcpy(new_evs[i].data(), evs1[i]->data(), sizeof(Goldilocks2::Element) * n1);
    }
    perm.resize(n2);
    size_t next = 0;
    for (size_t i = n1; i < n2; ++i) {
        while (vis[next]) ++next;
        for (size_t j = 0; j < len; ++j) {
            new_evs[j][i] = (*evs2[j])[next];
        }
        perm[i] = next;
        ++next;
    }
    std::vector<ligeropcs_base> ret;
    pad_f1.reserve(len);
    ret.reserve(len);
    for (size_t i = 0; i < len; ++i) {
        pad_f1.emplace_back(new_evs[i]);
        ret.push_back(ligero_commit_base(pad_f1.back(), rho_inv));
    }
    return ret;
}

setProver permProverBase::get_set_prover(const MLE& id_perm, const MLE& perm) {
    std::vector<const MLE*> set1, set2;
    const size_t len = f1.size();
    set1.reserve(len + 1);
    set2.reserve(len + 1);
    set1.push_back(&perm);
    set2.push_back(&id_perm);
    for (size_t i = 0; i != len; ++i) {
        set1.push_back(&pad_f1[i]);
        set2.push_back(f2[i]);
    }
    return setProver(set1, set2);
}

bool permVerifierBase::execute_check(
    permProverBase& prover,
    uint64_t rho_inv, uint64_t sec_param) {

    auto _pcs_pad_f1 = prover.commit_pad_f1(rho_inv);
    std::vector<const oracle *> pcs_pad_f1;
    if (_pcs_pad_f1.empty()) {
        pcs_pad_f1 = pcs_f1;
    } else {
        pcs_pad_f1.reserve(pcs_f1.size());
        for (auto& p : _pcs_pad_f1) {
            pcs_pad_f1.push_back(&p);
        }
    }
    
    // Compute MLE for permutation & identity permutation
    size_t sz = prover.get_size();
    std::vector<size_t> id_perm(sz);
    for (size_t i = 0; i != sz; ++i) id_perm[i] = i;
    MLE mle_id_perm(id_perm);
    MLE mle_perm(prover.get_perm());

    setProver set_prover = prover.get_set_prover(mle_id_perm, mle_perm);
    std::vector<const oracle*> pcs_vec1, pcs_vec2;
    pcs_vec1.reserve(pcs_f1.size() + 1);
    pcs_vec2.reserve(pcs_f2.size() + 1);
    pcs_vec1.push_back(&mle_perm);
    pcs_vec2.push_back(&mle_id_perm);
    pcs_vec1.insert(pcs_vec1.end(), pcs_pad_f1.begin(), pcs_pad_f1.end());
    pcs_vec2.insert(pcs_vec2.end(), pcs_f2.begin(), pcs_f2.end());
    if (!setVerifier::execute_check(set_prover, pcs_vec1, pcs_vec2, rho_inv, sec_param)) {
        std::cerr << "permVerifierBase::execute_check : set check fails" << std::endl;
        return false;
    }
    return true;
}

void permVerifierBase::add_pcs(const oracle* new_pcs_f1, const oracle* new_pcs_f2) {
    pcs_f1.push_back(new_pcs_f1);
    pcs_f2.push_back(new_pcs_f2);
}
