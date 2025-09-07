#pragma once

#include "mle.h"
#include "div_check.h"
#include "lazy_pcs.h"


std::vector<Goldilocks2::Element> get_sign(const std::vector<Goldilocks2::Element>& vec, bool strict_positive);

class signVerifier;

class signProver {
    friend class signVerifier;
public:
    signProver() = default;

    signProver(const std::vector<Goldilocks2::Element>& vec,
            const std::vector<Goldilocks2::Element>& sign,
            uint64_t scale, uint64_t max_val, bool strict_positive, uint64_t rho_inv);
            
    signProver(const std::vector<uint64_t>& vec,
            const std::vector<uint64_t>& sign,
            uint64_t scale, uint64_t max_val, bool strict_positive, uint64_t rho_inv);

    divProver next_prover(ligeropcs_base& pcs_quo, ligeropcs_base& pcs_rem, signProver& next_prover) const;

    divProver pre_next_prover(lazy_pcs &pcs_quo, lazy_pcs &pcs_rem, signProver& next_prover, lazy_pcs_pool *pool) const;

    inline int get_round() const {
        uint64_t v = max_val;
        int ret = 0;
        while (v) {
            ++ret;
            v /= scale;
        }
        return ret;
    }

    inline bool final_round() const { return max_val == 0; }

    inline int get_num_vars() const { return num_vars; }

    inline ligeropcs_base get_pcs_bias_x() const {
        if (!strict) {
            throw std::runtime_error("signProver: get_pcs_bias_x called when strict is false");
        }
        return ligero_commit_base(vec, rho_inv);
    }

    inline lazy_pcs pre_get_pcs_bias_x(lazy_pcs_pool *pool) const {
        if (!strict) {
            throw std::runtime_error("signProver: get_pcs_bias_x called when strict is false");
        }
        return commit_lazy_pcs(vec, pool);
    }

protected:
    uint64_t scale, max_val;
    bool strict;
    uint64_t rho_inv, num_vars;
    std::vector<Goldilocks2::Element> vec, sign; 

};

class signVerifier {
public:
    class resource {
    public:
        std::vector<std::map<std::string, lazy_pcs>> pcs;
        std::vector<divProver> div_provers;
        std::vector<signProver> sign_provers;
        // std::vector<divVerifier::resource> div_resources;
        void pop_back() {
            pcs.pop_back();
            div_provers.pop_back();
            sign_provers.pop_back();
            // div_resources.pop_back();
        }
        void push_back(const std::map<std::string, lazy_pcs>& r, const divProver& d, const signProver& s) {
            pcs.push_back(r);
            div_provers.push_back(d);
            sign_provers.push_back(s);
        }
    };
    

    static resource pre_execute_sign_check(
        const signProver& prover,
        const oracle& pcs_x,
        const oracle& pcs_sign,
        lazy_pcs_pool *pool
    );

    static bool execute_sign_check(
        const signProver& prover,
        const oracle *pcs_x,
        const oracle *pcs_sign,
        uint64_t sec_param,
        resource& prev_resource
    );

    static bool execute_sign_check(
        const signProver& prover,
        const oracle *pcs_x,
        const oracle *pcs_sign,
        uint64_t sec_param
    );
};
