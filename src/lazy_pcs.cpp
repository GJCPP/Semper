#include <map>

#include "lazy_pcs.h"
#include "mle_sum.h"
#include "timer.h"
#include "util.h"
#include "ligero.h"
#include "counter.h"
ligeropcs_base lazy_pcs_pool::commit(uint64_t rho_inv) {
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
    if (!committed) {
        throw std::runtime_error("lazy_pcs_pool::record_open: not committed");
    }
    if (z.size() + prefix[ind].size() != num_vars) {
        throw std::runtime_error("lazy_pcs_pool::record_open: number of variables mismatch");
    }
    if (sec_param < sec) sec_param = sec;
    std::vector<Goldilocks2::Element> zpad(num_vars);
    memcpy(zpad.data(), prefix[ind].data(), prefix[ind].size() * sizeof(Goldilocks2::Element));
    memcpy(zpad.data() + prefix[ind].size(), z.data(), z.size() * sizeof(Goldilocks2::Element));
    open_val.push_back(val);
    open_eqs.emplace_back(zpad);
}

bool lazy_pcs_pool::prove_open(ligeropcs_base pcs, Goldilocks2::Element lambda) const {
    MLE_Sum sum;
    Goldilocks2::Element c = Goldilocks2::one(), claim = Goldilocks2::zero();
    for (size_t i = 0; i != open_eqs.size(); ++i) {
        sum.add(std::make_unique<MLE_Eq>(open_eqs[i]), c);
        claim += c * open_val[i];
        c *= lambda;
    }
    sum.init_eval();
    p2Prover prover(sum.clone(), uni_mle.clone());
    auto cha = p2Verifier::partial_sumcheck(prover, claim, sec_param);
    if (!cha) {
        std::cerr << __LINE__ << ": lazy_pcs_pool::prove_open: partial_sumcheck failed" << std::endl;
        return false;
    }
    if (cha->claim != pcs.open(cha->challenges, sec_param) * sum.open(cha->challenges, sec_param)) {
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
    return mle->open(z, sec_param);
}
