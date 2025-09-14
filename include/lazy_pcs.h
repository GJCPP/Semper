#pragma once

#include <vector>

#include "goldilocks_quadratic_ext.h"
#include "ligero.h"
#include "mle.h"
#include "mle_eq.h"
#include "product2_sumcheck.h"
#include "util.h"

class lazy_pcs;
class lazy_pcs_pool;

lazy_pcs commit_lazy_pcs(const MLE& mle, lazy_pcs_pool *pool);

class lazy_pcs : public oracle {
public:
    lazy_pcs() = default;
    friend lazy_pcs commit_lazy_pcs(const MLE& mle, lazy_pcs_pool *pool);

    int get_num_vars() const override {
        return mle->get_num_vars();
    }

    Goldilocks2::Element open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const override;
    
protected:
    lazy_pcs(const MLE& mle, lazy_pcs_pool *pool) : mle(std::make_shared<MLE>(mle)), pool(pool) {}

    std::shared_ptr<MLE> mle;
    size_t index;
    lazy_pcs_pool *pool;
};

class lazy_pcs_pool {
public:
    friend class lazy_pcs;
    
    lazy_pcs_pool(size_t sec_param) : sec_param(sec_param) {}
    size_t add_mle(const MLE& mle) {
        mles.push_back({mle, mles.size()});
        return mles.size() - 1;
    }

    ligeropcs_base commit(uint64_t rho_inv);

    bool prove_open(ligeropcs_base pcs, Goldilocks2::Element lambda) const;

protected:
    bool committed = false;
    int num_vars = 0;
    size_t sec_param = 32;
    std::vector<std::pair<MLE, size_t>> mles; // (mle, add order)
    std::vector<std::vector<Goldilocks2::Element>> prefix;
    std::vector<std::vector<Goldilocks2::Element>> open_eqs;
    std::vector<size_t> open_ind;
    std::vector<Goldilocks2::Element> open_val;
    MLE uni_mle;

    void record_open(size_t ind, const std::vector<Goldilocks2::Element>& z, Goldilocks2::Element val, size_t sec_param);
};


extern std::map<int, int> lazy_pcs_open_cnt;