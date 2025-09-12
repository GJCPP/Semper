#pragma once

#include "goldilocks_quadratic_ext.h"
#include <vector>


/*
 * Base class for oracle for quadratic extension field
 * To be inherited by the mle (if it's a known polynomial)
*/
class oracle {
public:
    virtual Goldilocks2::Element open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const = 0;
    virtual int get_num_vars() const = 0;
};

class oracle_sum : public oracle {
public:
    oracle_sum() = default;

    void add(std::shared_ptr<oracle> o, Goldilocks2::Element coeff = Goldilocks2::one()) {
        if (!oracles.empty() && o->get_num_vars() != oracles[0]->get_num_vars()) {
            throw std::runtime_error("oracle_sum::add: number of variables does not match.");
        }
        oracles.push_back(o);
        coeffs.push_back(coeff);
    }

    Goldilocks2::Element open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const override {
        Goldilocks2::Element result = Goldilocks2::zero();
        for (size_t i = 0; i < oracles.size(); ++i) {
            result += coeffs[i] * oracles[i]->open(z, sec_param);
        }
        return result;
    }

    int get_num_vars() const override {
        if (oracles.empty()) {
            throw std::runtime_error("oracle_sum::get_num_vars: no oracle added.");
        }
        return oracles[0]->get_num_vars();
    }

    std::vector<std::shared_ptr<oracle>> oracles;
    std::vector<Goldilocks2::Element> coeffs; // coefficients for each oracle
};

class challenge_claim {
public:
    std::vector<Goldilocks2::Element> challenges;
    Goldilocks2::Element claim;
};

// Auxiliary information returned by MLE challenge processing
struct mle_aux_info {
    std::vector<Goldilocks2::Element> r;
    // Additional auxiliary data can be added here as needed
    Goldilocks2::Element comp; // compensation x open_val = claim
};
