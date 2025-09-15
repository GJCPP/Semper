#include "ligero.h"
#include "lazy_map_check.h"

void lazyMapProver::add(
    const std::vector<size_t>& map_from, 
    const std::vector<size_t>& map_to, 
    const MLE& mle_left, 
    const MLE& mle_right) {

    if (map_from.size() != map_to.size()) {
        throw std::invalid_argument("lazyPermProver::add: map_from and map_to must have the same size");
    }
    ins_left.push_back({map_from, mle_left, ins_right.size()});
    ins_right.push_back({map_to, mle_right, ins_right.size()});
}

void lazyMapProver::add(
    const std::vector<size_t>& map_from, const std::vector<size_t>& map_to, 
    const std::vector<MLE>& mle_left, const std::vector<MLE>& mle_right, 
    Goldilocks2::Element alpha) {

    if (mle_left.size() != mle_right.size() || mle_left.size() == 0) {
        throw std::invalid_argument("lazyPermProver::add: mle_left and mle_right must have the same non-zero size");
    }
    if (map_from.size() != map_to.size()) {
        throw std::invalid_argument("lazyPermProver::add: map_from and map_to must have the same size");
    }
    if (mle_left.size() > 1 && !use_ext) {
        throw std::invalid_argument("lazyPermProver::add: multiple maps only supported in ext field");
    }

    std::vector<Goldilocks2::Element> sum_left(mle_left[0].get_eval_table()), sum_right(mle_right[0].get_eval_table());
    Goldilocks2::Element alpha_pow = alpha;
    // std::cout << "============Warning: lazyPermProver::add checks map." << std::endl;
    // for (size_t i = 0; i != mle_left.size(); ++i) {
    //     const auto& evs_left = mle_left[i].get_eval_table();
    //     const auto& evs_right = mle_right[i].get_eval_table();
    //     for (size_t j = 0; j != map_from.size(); ++j) {
    //         if (evs_left[map_from[j]] != evs_right[map_to[j]]) {
    //             throw std::invalid_argument("lazyPermProver::add: evaluation tables do not match");
    //         }
    //     }
    // }
    for (size_t i = 1; i != mle_left.size(); ++i) {
        const auto& evs_left = mle_left[i].get_eval_table();
        const auto& evs_right = mle_right[i].get_eval_table();
        if (evs_left.size() != sum_left.size() || evs_right.size() != sum_right.size()) {
            throw std::invalid_argument("lazyPermProver::add: all MLEs must have the same size");
        }
        for (size_t j = 0; j != sum_left.size(); ++j) {
            sum_left[j] += alpha_pow * evs_left[j];
        }
        for (size_t j = 0; j != sum_right.size(); ++j) {
            sum_right[j] += alpha_pow * evs_right[j];
        }
        alpha_pow *= alpha;
    }
    return add(map_from, map_to, MLE(sum_left), MLE(sum_right));
}

mapProver lazyMapProver::commit_left_right(
    lazy_pcs_pool& pool_left, lazy_pcs_pool& pool_right,
    std::shared_ptr<oracle>& pcs_left, std::shared_ptr<oracle>& pcs_right,
    uint64_t rho_inv) {
    for (auto& ins : ins_left) {
        commit_lazy_pcs(ins.mle, &pool_left);
    }
    for (auto& ins : ins_right) {
        commit_lazy_pcs(ins.mle, &pool_right);
    }
    pcs_left = pool_left.commit(rho_inv);
    pcs_right = pool_right.commit(rho_inv);

    auto& perm_left = pool_left.get_perm(), &perm_right = pool_right.get_perm();
    auto& order_left = pool_left.get_order(), &order_right = pool_right.get_order();
    std::vector<size_t> pos_left(ins_left.size()), pos_right(ins_right.size());
    pos_left[0] = 0, pos_right[0] = 0;
    for (size_t i = 1; i != ins_left.size(); ++i) {
        pos_left[i] = pos_left[i - 1] + (1ull << ins_left[perm_left[i - 1]].mle.get_num_vars());
        pos_right[i] = pos_right[i - 1] + (1ull << ins_right[perm_right[i - 1]].mle.get_num_vars());
    }
    std::vector<std::pair<size_t, size_t>> map_index;
    for (size_t i = 0; i != ins_left.size(); ++i) {
        size_t left = order_left[i], right = order_right[i];
        for (size_t j = 0; j != ins_left[i].map_index.size(); ++j) {
            map_index.push_back({
                pos_left[left] + ins_left[i].map_index[j], 
                pos_right[right] + ins_right[i].map_index[j]
            });
        }
    }
    std::sort(map_index.begin(), map_index.end());
    const MLE& mle_left = pool_left.get_uni_mle(), &mle_right = pool_right.get_uni_mle();
    // std::cout << "============Warning: lazyPermProver::commit_left_right checks map." << std::endl;
    // for (size_t i = 0; i < map_index.size(); ++i) {
    //     if (mle_left.eval_hypercube(map_index[i].first) != mle_right.eval_hypercube(map_index[i].second)) {
    //         throw std::invalid_argument("lazyPermProver::commit_left_right: evaluation tables do not match");
    //     }
    // }
    std::vector<size_t> map_from, map_to;
    map_from.reserve(map_index.size());
    map_to.reserve(map_index.size());
    for (const auto& p : map_index) {
        map_from.push_back(p.first);
        map_to.push_back(p.second);
    }
    mapProver prover(map_from, map_to, use_ext);
    prover.add_mle(&mle_left, &mle_right);
    return prover;
}

void lazyMapVerifier::add(
    const std::vector<size_t>& map_from, 
    const std::vector<size_t>& map_to, 
    std::shared_ptr<oracle> pcs_left, 
    std::shared_ptr<oracle> pcs_right) {
    if (map_from.size() != map_to.size()) {
        throw std::invalid_argument("lazyPermVerifier::add: map_from and map_to must have the same size");
    }
    ins_left.push_back({map_from, pcs_left, ins_right.size()});
    ins_right.push_back({map_to, pcs_right, ins_right.size()});
}

void lazyMapVerifier::add(
    const std::vector<size_t>& map_from, const std::vector<size_t>& map_to, 
    std::vector<std::shared_ptr<oracle>> pcs_left, std::vector<std::shared_ptr<oracle>> pcs_right, 
    Goldilocks2::Element alpha) {

    if (map_from.size() != map_to.size()) {
        throw std::invalid_argument("lazyPermVerifier::add: map_from and map_to must have the same size");
    }
    if (pcs_left.size() != pcs_right.size()) {
        throw std::invalid_argument("lazyPermVerifier::add: pcs_left and pcs_right must have the same size");
    }
    int num_vars = pcs_left[0]->get_num_vars();
    for (const auto& pcs : pcs_left) {
        if (pcs->get_num_vars() != num_vars) {
            throw std::invalid_argument("lazyPermVerifier::add: pcs_left must have the same num_vars");
        }
    }
    oracle_sum sum_left, sum_right;
    Goldilocks2::Element alpha_pow = Goldilocks2::one();
    for (size_t i = 0; i != pcs_left.size(); ++i) {
        sum_left.add(pcs_left[i], alpha_pow);
        sum_right.add(pcs_right[i], alpha_pow);
        alpha_pow *= alpha;
    }
    return add(map_from, map_to, 
        std::make_shared<oracle_sum>(std::move(sum_left)), 
        std::make_shared<oracle_sum>(std::move(sum_right)));
}

bool lazyMapVerifier::prove_all(lazyMapProver& prover, uint64_t rho_inv, uint64_t sec_param) {
    lazy_pcs_pool pool_left(sec_param, use_ext), pool_right(sec_param, use_ext);
    std::shared_ptr<oracle> pcs_left, pcs_right;
    mapProver map_prover = prover.commit_left_right(pool_left, pool_right, pcs_left, pcs_right, rho_inv);
    mapVerifier map_verifier;
    map_verifier.add_pcs(pcs_left.get(), pcs_right.get());
    if (!map_verifier.execute_check(map_prover, rho_inv, sec_param)) {
        std::cerr << "lazyMapVerifier::prove_all: map check failed." << std::endl;
        return false;
    }
    return true;
}
