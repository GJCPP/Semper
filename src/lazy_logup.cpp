#include "util.h"
#include "logup.h"
#include "lazy_logup.h"
#include "counter.h"



void lazyLogupProver::add(
    const std::vector<uint64_t>& f1,
    const std::vector<uint64_t>& f2,
    const std::vector<uint64_t>& t1,
    const std::vector<uint64_t>& t2) {

    if (t1.size() != t2.size()) {
        throw std::runtime_error("lazyLogupProver: t1 and t2 must have the same size");
    }
    if (f1.size() != f2.size()) {
        throw std::runtime_error("lazyLogupProver: f1 and f2 must have the same size");
    }
    if (!is_power_of_2(t1.size())) {
        throw std::runtime_error("lazyLogupProver: size of t1 & t2 must be a power of 2");
    }
    if (!is_power_of_2(f1.size())) {
        throw std::runtime_error("lazyLogupProver: size of f1 & f2 must be a power of 2");
    }
    logupTable table = {t1, t2};
    uint64_t h = table.hash();
    auto it = index.find(h);
    bool found = false;
    size_t ind;
    if (it != index.end()) {
        for (const auto& i : it->second) {
            if (tables_all[i] == table) {
                found = true;
                ind = i;
                break;
            }
        }
    }
    if (found) {
        instances_all[ind].push_back({f1, f2});
        return;
    } else {
        ind = instances_all.size();
        tables_all.push_back(table);
        instances_all.push_back(std::vector<logupInstance>({{f1, f2}}));
        if (index.find(h) == index.end()) {
            index[h] = std::vector<size_t>();
        }
        index[h].push_back(ind);
    }
}

void lazyLogupProver::start_prove() {
    f1_all.resize(tables_all.size());
    f2_all.resize(tables_all.size());
}

std::pair<lazy_pcs, lazy_pcs> lazyLogupProver::commit_sort_f(size_t id, const std::vector<size_t>& order, lazy_pcs_pool* pool) {
    if (order.size() != instances_all[id].size()) {
        throw std::runtime_error("lazyLogupProver: order size must match the number of instances");
    }
    std::vector<logupInstance> tmp(instances_all[id].size());
    for (size_t i = 0; i < instances_all[id].size(); i++) {
        tmp[i] = instances_all[id][order[i]];
    }
    instances_all[id] = tmp;
    // check if sorted
    for (size_t i = 1; i < instances_all[id].size(); i++) {
        if (instances_all[id][i-1].f1.size() < instances_all[id][i].f1.size()) {
            throw std::runtime_error("lazyLogupProver: f is not sorted");
        }
    }
    std::vector<uint64_t> &f1 = f1_all[id], &f2 = f2_all[id];
    for (size_t i = 0; i < instances_all[id].size(); ++i) {
        f1.insert(f1.end(), instances_all[id][i].f1.begin(), instances_all[id][i].f1.end());
        f2.insert(f2.end(), instances_all[id][i].f2.begin(), instances_all[id][i].f2.end());
    }
    int num_vars = find_ceiling_log2(f1.size());
    if ((1ull << num_vars) != f1.size()) {
        // pad to power of 2
        size_t remainder = (1ull << num_vars) - f1.size();
        f1.insert(f1.end(), remainder, tables_all[id].t1[0]);
        f2.insert(f2.end(), remainder, tables_all[id].t2[0]);
    }
    return { commit_lazy_pcs(f1, pool), commit_lazy_pcs(f2, pool) };
}

LogupProver lazyLogupProver::get_logup_prover(size_t id) {
    return LogupProver(f1_all[id], f2_all[id],
                 tables_all[id].t1, tables_all[id].t2);
}

void lazyLogupVerifier::add(
    std::shared_ptr<oracle> pcs_f1,
    std::shared_ptr<oracle> pcs_f2,
    const std::vector<uint64_t>& t1,
    const std::vector<uint64_t>& t2) {

    if (t1.size() != t2.size()) {
        throw std::runtime_error("lazyLogupProver: t1 and t2 must have the same size");
    }
    if (pcs_f1->get_num_vars() != pcs_f2->get_num_vars()) {
        throw std::runtime_error("lazyLogupProver: f1 and f2 must have the same size");
    }
    if (!is_power_of_2(t1.size())) {
        throw std::runtime_error("lazyLogupProver: size of t1 & t2 must be a power of 2");
    }
    logupTable table = {t1, t2};
    uint64_t h = table.hash();
    auto it = index.find(h);
    bool found = false;
    size_t ind;
    if (it != index.end()) {
        for (const auto& i : it->second) {
            if (tables_all[i] == table) {
                found = true;
                ind = i;
                break;
            }
        }
    }
    if (found) {
        instances_all[ind].push_back({pcs_f1, pcs_f2});
        return;
    } else {
        ind = instances_all.size();
        tables_all.push_back(table);
        instances_all.push_back(std::vector<logupInstance>({{pcs_f1, pcs_f2}}));
        if (index.find(h) == index.end()) {
            index[h] = std::vector<size_t>();
        }
        index[h].push_back(ind);
    }
}

bool lazyLogupVerifier::prove_all(lazyLogupProver& prover, uint64_t rho_inv, uint64_t sec_param) {
    lazy_pcs_pool pool(sec_param);
    std::vector<lazy_pcs> pcs_f1_all, pcs_f2_all;
    for (size_t id = 0; id < tables_all.size(); id++) { // table id
        // 0. preprocess
        size_t sz = 0;
        int num_vars;
        for (const auto& inst : instances_all[id]) {
            sz += (1ull << inst.pcs_f1->get_num_vars());
        }
        num_vars = find_ceiling_log2(sz);

        prover.start_prove();

        // 1. Sort and commit all f
        auto &t1 = tables_all[id].t1, &t2 = tables_all[id].t2;
        auto order = sort_f(instances_all[id]);
        auto [pcs_f1, pcs_f2] = prover.commit_sort_f(id, order, &pool);
        if (num_vars != pcs_f1.get_num_vars() || num_vars != pcs_f2.get_num_vars()) {
            throw std::runtime_error("lazyLogupVerifier: committed f does not match the number of variables");
        }
        pcs_f1_all.push_back(pcs_f1);
        pcs_f2_all.push_back(pcs_f2);
        // std::cout << "logup id = " << id << ": f_size = " << (1ull << pcs_f1.get_num_vars()) << ", table size = " << t1.size() << std::endl;
    }
    auto pcs_pool = pool.commit(rho_inv);
    for (size_t id = 0; id < tables_all.size(); id++) { // table id
        // 0. preprocess
        size_t sz = 0;
        int num_vars;
        for (const auto& inst : instances_all[id]) {
            sz += (1ull << inst.pcs_f1->get_num_vars());
        }
        num_vars = find_ceiling_log2(sz);
        auto &pcs_f1 = pcs_f1_all[id], &pcs_f2 = pcs_f2_all[id];
        auto &t1 = tables_all[id].t1, &t2 = tables_all[id].t2;

        // 2. Check committed f
        std::vector<Goldilocks2::Element> cha = random_vec_ext(num_vars);
        Goldilocks2::Element claim_f1_cha = pcs_f1.open(cha, sec_param);
        Goldilocks2::Element claim_f2_cha = pcs_f2.open(cha, sec_param);
        Goldilocks2::Element verifier_f1_cha = Goldilocks2::zero();
        Goldilocks2::Element verifier_f2_cha = Goldilocks2::zero();
        size_t pos = 0;
        for (const auto& inst : instances_all[id]) {
            int len = inst.pcs_f1->get_num_vars();
            size_t pre = (pos >> len);
            std::vector<Goldilocks2::Element> prefix(num_vars - len);
            std::vector<Goldilocks2::Element> r0(cha.begin(), cha.begin() + num_vars - len);
            std::vector<Goldilocks2::Element> r1(cha.begin() + num_vars - len, cha.end());
            for (int i = 0; i < num_vars - len; ++i) {
                prefix[num_vars - len - i - 1] = ((pre >> i) & 1) ? Goldilocks2::one() : Goldilocks2::zero();
            }
            pos += (1ull << len);

            auto coeff = compute_eq(r0, prefix);
            verifier_f1_cha = verifier_f1_cha + inst.pcs_f1->open(r1, sec_param) * coeff;
            verifier_f2_cha = verifier_f2_cha + inst.pcs_f2->open(r1, sec_param) * coeff;
        }
        while (pos != (1ull << num_vars)) {
            size_t pre_len = 0;
            for (int i = 0; i != num_vars; ++i) { // find last 1
                if (pos & (1ull << (num_vars - i - 1))) {
                    pre_len = i + 1;
                }
            }
            std::vector<Goldilocks2::Element> prefix(pre_len);
            std::vector<Goldilocks2::Element> r0(cha.begin(), cha.begin() + pre_len);
            for (int i = 0; i < pre_len; ++i) {
                prefix[i] = ((pos >> (num_vars - i - 1)) & 1) ? Goldilocks2::one() : Goldilocks2::zero();
            }
            pos += (1ull << (num_vars - pre_len));

            auto coeff = compute_eq(r0, prefix);
            verifier_f1_cha += Goldilocks2::fromU64(t1[0]) * coeff;
            verifier_f2_cha += Goldilocks2::fromU64(t2[0]) * coeff;
        }
        if (claim_f1_cha != verifier_f1_cha || claim_f2_cha != verifier_f2_cha) {
            std::cout << "lazyLogupVerifier: f1/f2 commitment check failed." << std::endl;
            return false;
        }

        // 3. Prove logup
        LogupProver logup_prover = prover.get_logup_prover(id);
        MLE mle_t1 = t1, mle_t2 = t2;

        // // Debug: check pcs_f1 == prover.f1 & pcs_f2 == prover.f2
        // for (size_t i = 0; i < prover.f1_all[id].size(); i++) {
        //     std::vector<Goldilocks2::Element> z(num_vars);
        //     for (int j = 0; j != num_vars; ++j) {
        //         z[j] = ((i >> (num_vars - j - 1)) & 1) ? Goldilocks2::one() : Goldilocks2::zero();
        //     }
        //     if (pcs_f1.open(z, sec_param)[0].fe != logup_prover.f1[i]) {
        //         throw std::runtime_error("lazyLogupVerifier: pcs_f1 does not match prover.f1");
        //     }
        //     if (pcs_f2.open(z, sec_param)[0].fe != logup_prover.f2[i]) {
        //         throw std::runtime_error("lazyLogupVerifier: pcs_f2 does not match prover.f2");
        //     }
        // }
        // // Debug: check mle_t1 == prover.t1 & mle_t2 == prover.t2
        // for (size_t i = 0; i < t1.size(); i++) {
        //     if (mle_t1.get_eval_table()[i][0].fe != logup_prover.t1[i]) {
        //         throw std::runtime_error("lazyLogupVerifier: t1 does not match prover.t1");
        //     }
        //     if (mle_t2.get_eval_table()[i][0].fe != logup_prover.t2[i]) {
        //         throw std::runtime_error("lazyLogupVerifier: t2 does not match prover.t2");
        //     }
        // }
        start_proof("lazylogup_logup_proof");
        if (!LogupVerifier::execute_logup(logup_prover, pcs_f1, pcs_f2, mle_t1, mle_t2, rho_inv, sec_param)) {
            std::cout << "lazyLogupVerifier: logup proof failed." << std::endl;
            return false;
        }
        end_proof("lazylogup_logup_proof");
    }
    // 4. Prove lazy_pcs
    start_proof("lazylogup_lazy_pcs_proof");
    pool.prove_open(pcs_pool, random_ext());
    end_proof("lazylogup_lazy_pcs_proof");
    return true;
}

std::vector<size_t> lazyLogupVerifier::sort_f(std::vector<logupInstance>& instances) {
    std::vector<std::pair<logupInstance, size_t>> vec;
    for (size_t i = 0; i < instances.size(); i++) {
        vec.push_back({instances[i], i});
    }
    std::sort(vec.begin(), vec.end(),
                [](const auto& a, const auto& b) {
                    return a.first.pcs_f1->get_num_vars() > b.first.pcs_f1->get_num_vars();
                }); // sort in descending order
    std::vector<size_t> order(instances.size());
    for (size_t i = 0; i < vec.size(); i++) {
        instances[i] = vec[i].first;
        order[i] = vec[i].second;
    }
    return order;
}