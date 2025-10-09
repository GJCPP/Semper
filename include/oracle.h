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
    oracle_sum(const oracle_sum& other) 
        : oracles(other.oracles), coeffs(other.coeffs), use_coeff(other.use_coeff), constant(other.constant) {}

    void add(std::shared_ptr<oracle> o, Goldilocks2::Element coeff = Goldilocks2::one()) {
        if (!oracles.empty() && o->get_num_vars() != oracles[0]->get_num_vars()) {
            throw std::runtime_error("oracle_sum::add: number of variables does not match.");
        }
        oracles.push_back(o);
        coeffs.push_back(coeff);
        use_coeff.push_back(true);
    }

    void add(std::shared_ptr<oracle> o1, std::shared_ptr<oracle> o2) {
        oracles.push_back(o1);
        mul.push_back(o2);
        use_coeff.push_back(false);
    }

    void add_const(Goldilocks2::Element c) {
        constant += c;
    }

    Goldilocks2::Element open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const override {
        Goldilocks2::Element result = Goldilocks2::zero();
        for (size_t i = 0; i < oracles.size(); ++i) {
            if (use_coeff[i]) {
                result += coeffs[i] * oracles[i]->open(z, sec_param);
            } else {
                result += mul[i]->open(z, sec_param) * oracles[i]->open(z, sec_param);
            }
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
    std::vector<std::shared_ptr<oracle>> oracles;
    std::vector<std::shared_ptr<oracle>> mul;
    std::vector<Goldilocks2::Element> coeffs; // coefficients for each oracle
    std::vector<bool> use_coeff;
    Goldilocks2::Element constant = Goldilocks2::zero(); // constant term
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
