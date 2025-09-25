#include <map>

#include "lazy_pcs.h"
#include "timer.h"
#include "util.h"
#include "ligero.h"
#include "counter.h"
std::shared_ptr<oracle> lazy_pcs_pool::commit(uint64_t rho_inv) {
    // std::cout << "====== warning: lazy_pcs_pool::commit skipped." << std::endl;
    // committed = true;
    // return {};

    if (committed) {
        throw std::runtime_error("lazy_pcs_pool::commit: already committed");
    }
    committed = true;


    struct sortIns {
        size_t ind;
        int num_vars;
    };
    std::vector<sortIns> elements;
    for (size_t i = 0; i < mles.size(); i++) {
        elements.push_back({i, mles[i].first.get_num_vars()});
    }
    std::sort(elements.begin(), elements.end(), [](const auto& a, const auto& b) {
        return a.num_vars > b.num_vars;
    }); // sort in descending order
    std::vector<std::pair<MLE, size_t>> sorted_mles;
    order.resize(mles.size());
    perm.resize(mles.size());
    for (size_t i = 0; i != elements.size(); ++i) {
        sorted_mles.push_back(mles[elements[i].ind]);
        order[elements[i].ind] = i;
        perm[i] = elements[i].ind;
    }
    mles = std::move(sorted_mles);

    size_t ind = 0;
    size_t total = 0;
    for (auto& mle : mles) {
        total += (1ull << mle.first.get_num_vars());
    }
    num_vars = find_ceiling_log2(total);
    total = (1ull << num_vars);
    prefix.resize(mles.size());
    std::vector<Goldilocks2::Element> all_vals;
    all_vals.reserve(total);
    for (auto& mle : mles) {
        std::vector<Goldilocks2::Element> pre;
        size_t len = (1ull << mle.first.get_num_vars());
        size_t left = 0, right = total;
        while (right - left != len) {
            size_t mid = (left + right) / 2;
            if (ind < mid) {
                right = mid;
                pre.push_back(Goldilocks2::zero());
            } else {
                left = mid;
                pre.push_back(Goldilocks2::one());
            }
        }
        prefix[mle.second] = pre;
        all_vals.insert(all_vals.end(), mle.first.get_eval_table().begin(), mle.first.get_eval_table().end());
        ind += mle.first.get_eval_table().size();
    }
    all_vals.resize(total);
    uni_mle = all_vals;
    
    // std::cout << "================Warning: skip lazy_pcs commit" << std::endl;
    // return std::make_shared<MLE>(uni_mle);
    // std::cout << "Committing lazy pcs with num_vars = " << num_vars << "...\n";
    if (use_ext) {
        return std::make_shared<ligeropcs_ext>(ligero_commit_ext(uni_mle, rho_inv));
    } else {
        return std::make_shared<ligeropcs_base>(ligero_commit_base(uni_mle, rho_inv));
    }
}

void lazy_pcs_pool::record_open(size_t ind, const std::vector<Goldilocks2::Element>& z, Goldilocks2::Element val, size_t sec) {
    // std::cout << "====== warning: lazy_pcs_pool::record_open skipped." << std::endl;
    // return ;
    
    if (!committed) {
        throw std::runtime_error("lazy_pcs_pool::record_open: not committed");
    }
    if (finalized) {
        throw std::runtime_error("lazy_pcs_pool::record_open: already finalized");
    }
    if (z.size() + prefix[ind].size() != static_cast<size_t>(num_vars)) {
        throw std::runtime_error("lazy_pcs_pool::record_open: number of variables mismatch");
    }
    // std::chrono::high_resolution_clock clock;
    // auto start = clock.now();
    if (sec_param < sec) sec_param = sec;
    std::vector<Goldilocks2::Element> zpad(num_vars);
    size_t pre_sz = prefix[ind].size();
    for (size_t i = 0; i < pre_sz; ++i) {
        zpad[i] = prefix[ind][i];
    }
    for (size_t i = 0; i < z.size(); ++i) {
        zpad[i + pre_sz] = z[i];
    }
    open_val.push_back(val);
    open_eqs.push_back(zpad);
    open_ind.push_back(ind);
}

bool lazy_pcs_pool::prove_open(std::shared_ptr<oracle> pcs, Goldilocks2::Element lambda) {
    // std::cout << "====== warning: lazy_pcs_pool::prove_open skipped." << std::endl;
    // return true;

    if (!committed) {
        throw std::runtime_error("lazy_pcs_pool::prove_open: not committed");
    }
    if (finalized) {
        throw std::runtime_error("lazy_pcs_pool::prove_open: already finalized");
    }
    finalized = true;

    std::vector<Goldilocks2::Element> table(1ull << num_vars);
    Goldilocks2::Element claim = Goldilocks2::zero();
    Goldilocks2::Element base = Goldilocks2::one();
    oracle_sum eq_sum_oracle;
    for (size_t i = 0; i != open_eqs.size(); ++i) {
        int pre_len = prefix[open_ind[i]].size();
        size_t offset = 0;
        for (int j = 0; j != pre_len; ++j) {
            offset = open_eqs[i][j] == Goldilocks2::one() ? ((offset << 1) | 1) : (offset << 1);
        }
        offset <<= (num_vars - pre_len);
        MLE_Eq eq(open_eqs[i].begin() + pre_len, open_eqs[i].end());
        auto& eqt = eq.get_eval_table();
        for (size_t k = 0; k != (1ull << (num_vars - pre_len)); ++k) {
            table[k + offset] += eqt[k] * base;
        }
        eq_sum_oracle.add(std::make_shared<MLE_Eq_Oracle>(open_eqs[i]), base);
        claim += open_val[i] * base;
        base *= lambda;
    }
    MLE mle_sum(std::move(table));
    p2Prover prover(mle_sum.clone(), uni_mle.clone());
    if (!p2Verifier::execute_sumcheck(prover, {&eq_sum_oracle, pcs.get()}, claim, sec_param)) {
        std::cerr << __LINE__ << ": lazy_pcs_pool::prove_open: sumcheck failed" << std::endl;
        return false;
    }
    return true;
}

lazy_pcs commit_lazy_pcs(const MLE& mle, lazy_pcs_pool* pool) {
    startCounter counter("lazy_pcs_commit");
    lazy_pcs res(mle, pool);
    res.index = pool->add_mle(mle);
    return res;
}

std::map<int, int> lazy_pcs_open_cnt;

Goldilocks2::Element lazy_pcs::open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const {
    startCounter counter("lazy_pcs_open");
    auto val = mle->open(z, sec_param);
    pool->record_open(index, z, val, sec_param);
    return val;
}
