#include <map>

#include "lazy_pcs.h"
#include "timer.h"
#include "util.h"
#include "ligero.h"
#include "counter.h"
ligeropcs_base lazy_pcs_pool::commit(uint64_t rho_inv) {
    // std::cout << "====== warning: lazy_pcs_pool::commit skipped." << std::endl;
    // committed = true;
    // return {};

    if (committed) {
        throw std::runtime_error("lazy_pcs_pool::commit: already committed");
    }
    committed = true;

    std::sort(mles.begin(), mles.end(), [](const auto& a, const auto& b) {
        return a.first.get_num_vars() > b.first.get_num_vars(); // big mle first
    });

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
    return ligero_commit_base(uni_mle, rho_inv);
}

void lazy_pcs_pool::record_open(size_t ind, const std::vector<Goldilocks2::Element>& z, Goldilocks2::Element val, size_t sec) {
    // std::cout << "====== warning: lazy_pcs_pool::record_open skipped." << std::endl;
    // return ;
    if (!committed) {
        throw std::runtime_error("lazy_pcs_pool::record_open: not committed");
    }
    if (z.size() + prefix[ind].size() != num_vars) {
        throw std::runtime_error("lazy_pcs_pool::record_open: number of variables mismatch");
    }
    std::chrono::high_resolution_clock clock;
    auto start = clock.now();
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
}

bool lazy_pcs_pool::prove_open(ligeropcs_base pcs, Goldilocks2::Element lambda) const {
    // std::cout << "====== warning: lazy_pcs_pool::prove_open skipped." << std::endl;
    // return true;

    std::vector<Goldilocks2::Element> table(1ull << num_vars);
    Goldilocks2::Element claim = Goldilocks2::zero();
    Goldilocks2::Element base = Goldilocks2::one();
    for (size_t i = 0; i != open_eqs.size(); ++i) {
        int j = 0;
        size_t offset = 0;
        while (open_eqs[i][j] == Goldilocks2::one() || open_eqs[i][j] == Goldilocks2::zero()) {
            offset = open_eqs[i][j] == Goldilocks2::one() ? ((offset << 1) | 1) : (offset << 1);
            ++j;
        }
        offset <<= (num_vars - j);
        MLE_Eq eq(open_eqs[i].begin() + j, open_eqs[i].end());
        auto& eqt = eq.get_eval_table();
        for (size_t k = 0; k != (1ull << (num_vars - j)); ++k) {
            table[k + offset] += eqt[k] * base;
        }
        claim += open_val[i] * base;
        base *= lambda;
    }
    MLE mle_sum(std::move(table));
    p2Prover prover(mle_sum.clone(), uni_mle.clone());
    auto cha = p2Verifier::partial_sumcheck(prover, claim, sec_param);
    if (!cha) {
        std::cerr << __LINE__ << ": lazy_pcs_pool::prove_open: partial_sumcheck failed" << std::endl;
        return false;
    }
    if (cha->claim != pcs.open(cha->challenges, sec_param) * mle_sum.open(cha->challenges, sec_param)) {
        std::cerr << __LINE__ << ": lazy_pcs_pool::prove_open: open value mismatch" << std::endl;
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
