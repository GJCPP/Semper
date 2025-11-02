#include <format>

#include "perm_check.h"
#include "product3_sumcheck.h"
#include "mle_open.h"
#include "counter.h"

bool pdVerifier::execute_check(pdProver& prover, const oracle* pcsf, Goldilocks2::Element claim, uint64_t sec_param) {

    startCounter counter("pd_execute_proof");
    int num_vars = prover.get_num_vars();
    int round = 1;

    std::array<Goldilocks2::Element, 2> msg = prover.first_msg(); 
    if (msg[0] * msg[1] != claim) {
        std::cerr << __FUNCTION__ << ": failed at round 0." << std::endl;
        return false;
    }
    std::vector<Goldilocks2::Element> cha;
    while (round < num_vars) {
        auto r = random_ext();
        claim = msg[0] + (msg[1] - msg[0]) * r;
        cha.push_back(r);
        // sumcheck g_round(cha) = \sum_i eq(cha, i) x g_i+1(b, 0) x g_i+1(b, 1)
        MLE_Eq eq(cha);
        p3Prover p3_prover = prover.get_next_sumcheck(eq);
        auto sum_claim = p3Verifier::partial_sumcheck(p3_prover, claim, sec_param);
        if (!sum_claim) {
            std::cerr << __FUNCTION__ << std::format(": p3sumcheck failed at round {}/{}", round, num_vars) << std::endl;
            return false;
        }
        cha = sum_claim->challenges;
        msg = prover.next_msg(cha);
        ++round;
    }
    auto r = random_ext();
    claim = msg[0] + (msg[1] - msg[0]) * r;
    cha.push_back(r);

    if (pcsf->open(cha, sec_param) != claim) {
        std::cerr << __FUNCTION__ << ": failed at last round " << round << "." << std::endl;
        return false;
    }
    return true;
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

void setProver::combine(const std::vector<Goldilocks2::Element>& cha, uint64_t rho_inv) {
    if (cha.size() != static_cast<size_t>(n)) {
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
    // return { ligero_commit_ext(set1, rho_inv), ligero_commit_ext(set2, rho_inv) };
}

std::array<pdProver, 2> setProver::get_pd_prover() {
    return {pdProver(set1), pdProver(set2)};
}

bool setVerifier::execute_check(
    setProver& prover,
    const std::vector<std::shared_ptr<oracle>>& pcs_f1,
    const std::vector<std::shared_ptr<oracle>>& pcs_f2,
    uint64_t rho_inv, uint64_t sec_param) {

    startCounter counter("set_proof");
    
    const int n = prover.get_n();
    // const int num_vars = prover.get_num_vars();
    auto alpha = random_vec_ext(n);
    prover.combine(alpha, rho_inv);
    oracle_sum pcs_set1, pcs_set2;
    pcs_set1.add_const(alpha[0]);
    pcs_set2.add_const(alpha[0]);
    pcs_set1.add(pcs_f1[0], Goldilocks2::one());
    pcs_set2.add(pcs_f2[0], Goldilocks2::one());
    for (int i = 1; i < n; ++i) {
        pcs_set1.add(pcs_f1[i], alpha[i]);
        pcs_set2.add(pcs_f2[i], alpha[i]);
    }

    // Check set[b] = alpha[0] + fb[0] \sum_i alpha[i] x fb[i]
    // {
    //     startCounter counter("set_relation_proof");
    //     auto cha = random_vec_ext(num_vars);
    //     auto expect1 = alpha[0] + pcs_f1[0]->open(cha, sec_param);
    //     auto expect2 = alpha[0] + pcs_f2[0]->open(cha, sec_param);
    //     for (int i = 1; i < n; ++i) {
    //         expect1 += alpha[i] * pcs_f1[i]->open(cha, sec_param);
    //         expect2 += alpha[i] * pcs_f2[i]->open(cha, sec_param);
    //     }
    //     if (pcs_set1.open(cha, sec_param) != expect1 || pcs_set2.open(cha, sec_param) != expect2) {
    //         std::cerr << "setVerifier::execute_check : combination check fails" << std::endl;
    //         return false;
    //     }
    // }

    // Check \prod pcs_set1/pcs_set2 = 1
    auto pd_prover = prover.get_pd_prover();
    auto pd1 = pd_prover[0].get_prod(), pd2 = pd_prover[1].get_prod();
    if (pd1 != pd2) {
        std::cerr << "setVerifier::execute_check : products mismatch" << std::endl;
        return false;
    }
    if (!pdVerifier::execute_check(pd_prover[0], &pcs_set1, pd1, sec_param) ||
        !pdVerifier::execute_check(pd_prover[1], &pcs_set2, pd2, sec_param)) {
        std::cerr << "setVerifier::execute_check : product check fails" << std::endl;
        return false;
    }
    return true;
}


// prove f2(pi(x)) = f1(x)
permProver::permProver(const std::vector<size_t>& _perm, bool _ext)
    : ext(_ext), perm(_perm) {
}

void permProver::add_mle(const MLE *new_f1, const MLE *new_f2) {
    f1.push_back(new_f1);
    f2.push_back(new_f2);
    const auto& evs1(new_f1->get_eval_table()), &evs2(new_f2->get_eval_table());
    if (f1.size() == 1) {
        sz = evs2.size();
        if (evs1.size() > evs2.size()) {
            throw std::invalid_argument("permProver::permProver : f1 must have size no larger than f2");
        }
        if (perm.size() != evs1.size()) {
            throw std::invalid_argument("permProver::permProver : permutation vector must have same size as f1");
        }
    } else {
        if (new_f1->get_num_vars() != f1[0]->get_num_vars() ||
            new_f2->get_num_vars() != f2[0]->get_num_vars()) {
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

std::vector<std::shared_ptr<oracle>> permProver::commit_pad_f1(uint64_t rho_inv) {
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

    if (!pcs_pad_f1.empty()) return pcs_pad_f1;

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
    std::vector<std::shared_ptr<oracle>> ret;
    pad_f1.reserve(len);
    ret.reserve(len);
    for (size_t i = 0; i < len; ++i) {
        pad_f1.emplace_back(new_evs[i]);
        if (ext) {
            ret.push_back(std::make_unique<ligeropcs_ext>(ligero_commit_ext(pad_f1.back(), rho_inv)));
        } else {
            ret.push_back(std::make_unique<ligeropcs_base>(ligero_commit_base(pad_f1.back(), rho_inv)));
        }
    }
    return ret;
}


void permProver::forward_commit_pad_f1(std::shared_ptr<lazy_pcs_pool> pool) {
    const size_t n1 = (1 << f1[0]->get_num_vars()), n2 = (1 << f2[0]->get_num_vars());
    const size_t len = f1.size();
    if (n1 == n2) {
        return;
    }

    if (pool->is_ext() != ext) {
        throw std::invalid_argument("permProver::forward_commit_pad_f1 : ext mismatch");
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
    #pragma omp parallel for
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
    std::vector<std::shared_ptr<oracle>> ret(len);
    pad_f1.resize(len);
    #pragma omp parallel for
    for (size_t i = 0; i < len; ++i) {
        pad_f1[i] = std::move(new_evs[i]);
        ret[i] = std::make_shared<lazy_pcs>(commit_lazy_pcs(pad_f1[i], pool));
    }
    pcs_pad_f1 = ret;
}

setProver permProver::get_set_prover(const MLE& id_perm, const MLE& perm) {
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

bool permVerifier::execute_check(
    permProver& prover,
    uint64_t rho_inv, uint64_t sec_param) {

    // std::cout << "Skipping permVerifier::execute_check" << std::endl;
    // return true;
    startCounter counter("perm_proof");

    auto _pcs_pad_f1 = prover.commit_pad_f1(rho_inv);
    std::vector<std::shared_ptr<oracle>> pcs_pad_f1;
    if (_pcs_pad_f1.empty()) {
        pcs_pad_f1 = pcs_f1;
    } else {
        pcs_pad_f1.reserve(pcs_f1.size());
        for (auto& p : _pcs_pad_f1) {
            pcs_pad_f1.push_back(p);
        }
    }
    
    // Compute MLE for permutation & identity permutation
    size_t sz = prover.get_size();
    std::vector<size_t> id_perm(sz);
    for (size_t i = 0; i != sz; ++i) id_perm[i] = i;
    MLE mle_id_perm(id_perm);
    MLE mle_perm(prover.get_perm());

    setProver set_prover = prover.get_set_prover(mle_id_perm, mle_perm);
    std::vector<std::shared_ptr<oracle>> pcs_vec1, pcs_vec2;
    pcs_vec1.reserve(pcs_f1.size() + 1);
    pcs_vec2.reserve(pcs_f2.size() + 1);
    pcs_vec1.push_back(std::make_shared<MLE>(mle_perm));
    pcs_vec2.push_back(std::make_shared<MLE>(mle_id_perm));
    pcs_vec1.insert(pcs_vec1.end(), pcs_pad_f1.begin(), pcs_pad_f1.end());
    pcs_vec2.insert(pcs_vec2.end(), pcs_f2.begin(), pcs_f2.end());
    if (!setVerifier::execute_check(set_prover, pcs_vec1, pcs_vec2, rho_inv, sec_param)) {
        std::cerr << "permVerifierBase::execute_check : set check fails" << std::endl;
        return false;
    }
    return true;
}

void permVerifier::add_pcs(std::shared_ptr<oracle> new_pcs_f1, std::shared_ptr<oracle> new_pcs_f2) {
    pcs_f1.push_back(new_pcs_f1);
    pcs_f2.push_back(new_pcs_f2);
}

mapProver::mapProver(const std::vector<size_t>& _map_from, const std::vector<size_t>& _map_to, bool _ext) 
    : ext(_ext), map_from(_map_from), map_to(_map_to) {
    if (!std::is_sorted(map_from.begin(), map_from.end())) {
        throw std::invalid_argument("mapProver::mapProver : map_from must be sorted");
    }
}

mapProver::mapProver(std::vector<size_t>&& _map_from, std::vector<size_t>&& _map_to, bool _ext)
    : ext(_ext), map_from(std::move(_map_from)), map_to(std::move(_map_to)) {
    if (!std::is_sorted(map_from.begin(), map_from.end())) {
        throw std::invalid_argument("mapProver::mapProver : map_from must be sorted");
    }
}

void mapProver::add_mle(const MLE* mle_from, const MLE* mle_to) {
    if (mapto.empty()) {
        init_map(mle_from, mle_to);
    }
    if (mle_from->get_num_vars() != left_num_vars) {
        throw std::invalid_argument("mapProver::add_mle : mle_from has incorrect number of variables");
    }
    auto& evs_left = mle_from->get_eval_table();
    auto evs_right = mle_to->get_eval_table();
    size_t ori_right_size = (1 << mle_to->get_num_vars());
    evs_right.resize(right_size);
    
    for (size_t i = 0; i != left_size; ++i) {
        if (mapto[i] < ori_right_size && evs_left[i] != evs_right[mapto[i]]) {
            throw std::invalid_argument("mapProver::add_mle : evaluation tables do not match");
        }
        evs_right[mapto[i]] = evs_left[i];
    }

    left.push_back(mle_from);
    right.emplace_back(evs_right);
}

std::vector<std::shared_ptr<oracle>> mapProver::commit_right(uint64_t rho_inv) const {
    std::vector<std::shared_ptr<oracle>> right_pcs;
    right_pcs.reserve(right.size());
    for (const auto& mle : right) {
        if (ext) {
            right_pcs.push_back(std::make_unique<ligeropcs_ext>(ligero_commit_ext(mle, rho_inv)));
        } else {
            right_pcs.push_back(std::make_unique<ligeropcs_base>(ligero_commit_base(mle, rho_inv)));
        }
    }
    return right_pcs;
}

std::vector<std::shared_ptr<oracle>> mapProver::commit_right(std::shared_ptr<lazy_pcs_pool> pool) const {
    if (ext != pool->is_ext()) {
        throw std::invalid_argument("mapProver::commit_right : ext mismatch.");
    }
    std::vector<std::shared_ptr<oracle>> right_pcs;
    right_pcs.reserve(right.size());
    for (const auto& mle : right) {
        right_pcs.push_back(std::make_shared<lazy_pcs>(commit_lazy_pcs(mle, pool)));
    }
    return right_pcs;
}


permProver mapProver::get_perm_prover() {
    permProver prover(mapto, ext);
    size_t len = left.size();
    for (size_t i = 0; i != len; ++i) {
        prover.add_mle(left[i], &right[i]);
    }
    return prover;
}

void mapProver::init_map(const MLE* mle_from, const MLE* mle_to) {
    left_num_vars = mle_from->get_num_vars();
    right_num_vars = std::max(left_num_vars, mle_to->get_num_vars()) + 1;
    pad_num_vars = right_num_vars - mle_to->get_num_vars();
    left_size = (1 << mle_from->get_num_vars());
    right_size = (1 << right_num_vars);

    size_t map_to_sz = map_to.size();
    mapto.resize(left_size);
    size_t next = 0, next_new = (1 << mle_to->get_num_vars());
    for (size_t i = 0; i != left_size; ++i) {
        if (next != map_to_sz && i == map_from[next]) {
            mapto[i] = map_to[next];
            ++next;
        } else {
            mapto[i] = next_new;
            ++next_new;
        }
    }
    // std::cout << next_new << " " << left_size << std::endl;
}


void mapVerifier::add_pcs(std::shared_ptr<oracle> left, std::shared_ptr<oracle> right) {
    pcs_left.push_back(left);
    pcs_right.push_back(right);
}

bool mapVerifier::execute_check(mapProver& prover, uint64_t rho_inv, uint64_t sec_param) {
    // std::cout << "Skipping mapVerifier::execute_check" << std::endl;
    // return true;
    startCounter counter("map_proof");
    std::shared_ptr<lazy_pcs_pool> pool = lazy_pcs_pool::create(sec_param, prover.use_ext());
    // 1. Commit/Check padded right
    size_t len = pcs_left.size();
    int right_num_vars = prover.get_right_num_vars(), pad_num_vars = prover.get_pad_num_vars();
    auto pcs_pad_right = prover.commit_right(pool);

    permProver perm_prover = prover.get_perm_prover();
    permVerifier perm_verifier;
    perm_prover.forward_commit_pad_f1(pool);

    auto pcs_pool = pool->commit(rho_inv);

    auto cha = random_vec_ext(right_num_vars);
    std::vector<Goldilocks2::Element> small_cha(cha.begin() + pad_num_vars, cha.end());
    for (int i = 0; i != pad_num_vars; ++i) cha[i] = Goldilocks2::zero();

    start_proof("map_proof_check_pad");
    {
        bool ret = true;
        for (size_t i = 0; i != len; ++i) {
            if (pcs_right[i]->open(small_cha, sec_param) != pcs_pad_right[i]->open(cha, sec_param)) {
                std::cerr << "mapVerifier::execute_check : pcs_right and pcs_pad_right do not match" << std::endl;
                ret = false;
            }
        }
        if (!ret) return false;
    }
    end_proof("map_proof_check_pad");
    // 2. Perform perm check.
    for (size_t i = 0; i != len; ++i) {
        perm_verifier.add_pcs(pcs_left[i], pcs_pad_right[i]);
    }
    if (!perm_verifier.execute_check(perm_prover, rho_inv, sec_param)) {
        std::cerr << "mapVerifier::execute_check : perm check fails" << std::endl;
        return false;
    }
    pool->prove_open(pcs_pool, random_ext());
    return true;
}
