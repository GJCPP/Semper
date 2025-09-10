#include "util.h"
#include "lazy_logup.h"





lazyLogupProver::lazyLogupProver(
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

lazyLogupVerifier::lazyLogupVerifier(
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
    for (size_t id = 0; id < tables_all.size(); id++) { // table id
        // 0. preprocess
        size_t sz = 0;
        int num_vars;
        for (const auto& inst : instances_all[id]) {
            sz += (1ull << inst.pcs_f1->get_num_vars());
        }
        num_vars = find_ceiling_log2(sz);

        // 1. Sort and commit all f
        auto t1 = tables_all[id].t1, t2 = tables_all[id].t2;
        auto order = sort_f(instances_all[id]);
        auto [pcs_f1, pcs_f2] = prover.commit_sort_f(id, order, rho_inv);
        if (num_vars != pcs_f1.get_num_vars() || num_vars != pcs_f2.get_num_vars()) {
            throw std::runtime_error("lazyLogupVerifier: committed f does not match the number of variables");
        }

        // 2. Check committed f
        std::vector<Goldilocks2::Element> cha = random_vec_ext(num_vars);
        size_t pos = 0;
        for (const auto& inst : instances_all[id]) {
            int len = inst.pcs_f1->get_num_vars();
        }
    }
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