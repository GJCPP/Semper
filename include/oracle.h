#pragma once

#include "goldilocks_quadratic_ext.h"
#include "Orion/include/linear_gkr/prime_field.h"
#include <vector>


/*
 * Base class for oracle for quadratic extension field
 * To be inherited by the mle (if it's a known polynomial)
*/

typedef prime_field::field_element Orionfield;


template <typename field>
class oracle {
public:
    virtual field open(const std::vector<field>& z, const size_t& sec_param) const = 0;
    virtual int get_num_vars() const = 0;
};

template <typename field>
class oracle_sum : public oracle<field> {
public:
    oracle_sum() = default;
    oracle_sum(const oracle_sum& other) 
        : oracles(other.oracles), coeffs(other.coeffs), constant(other.constant) {}

    void add(std::shared_ptr<oracle<field>> o, field coeff) {
        if (!oracles.empty() && o->get_num_vars() != oracles[0]->get_num_vars()) {
            throw std::runtime_error("oracle_sum::add: number of variables does not match.");
        }
        oracles.push_back(o);
        coeffs.push_back(coeff);
    }

    void add_const(field c) {
        constant += c;
    }

    field open(const std::vector<field>& z, const size_t& sec_param) const override {
        field result = {};
        for (size_t i = 0; i < oracles.size(); ++i) {
            result += coeffs[i] * oracles[i]->open(z, sec_param);
        }
        ++*open_counter;
        if (*open_counter >= 2) {
            std::cout << "Warning: oracle_sum::open called multiple times (" << open_counter << ")." << std::endl;
        }
        return result + constant;
    }

    int get_num_vars() const override {
        if (oracles.empty()) {
            throw std::runtime_error("oracle_sum::get_num_vars: no oracle added.");
        }
        return oracles[0]->get_num_vars();
    }
protected:
    std::unique_ptr<int> open_counter = std::make_unique<int>(0);
    std::vector<std::shared_ptr<oracle<field>>> oracles;
    std::vector<field> coeffs; // coefficients for each oracle
    field constant = {}; // constant term
};

template <typename field>
class challenge_claim {
public:
    std::vector<field> challenges;
    Goldilocks2::Element claim;
};

// Auxiliary information returned by MLE challenge processing
template <typename field>
struct mle_aux_info {
    std::vector<field> r;
    // Additional auxiliary data can be added here as needed
    field comp; // compensation x open_val = claim
};
