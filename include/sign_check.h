#pragma once

#include "mle.h"
#include "div_check.h"


// (< 0) -> 0, (>= 0) -> 1
std::vector<Goldilocks2::Element> get_sign(const std::vector<Goldilocks2::Element>& vec);

class signVerifier;

class signProver {
    friend class signVerifier;
public:
    signProver() = default;

    signProver(const std::vector<Goldilocks2::Element>& vec,
            const std::vector<Goldilocks2::Element>& sign,
            uint64_t scale, uint64_t max_val, uint64_t rho_inv);

    divProver next_prover(ligeropcs_base& pcs_quo, ligeropcs_base& pcs_rem, signProver& next_prover) const;

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

protected:
    uint64_t scale, max_val, rho_inv, num_vars;
    std::vector<Goldilocks2::Element> vec, sign; 

};

class signVerifier {
public:
    static bool execute_sign_check(
        const signProver& prover,
        ligeropcs_base pcs_x,
        ligeropcs_base pcs_sign,
        uint64_t sec_param
    );
};
