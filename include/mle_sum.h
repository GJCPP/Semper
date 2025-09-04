#pragma once

#include "mle.h"

class MLE_Sum : public MLE {
public:
    MLE_Sum() = default;

    void add(std::unique_ptr<MLE> term, const Goldilocks2::Element& c = Goldilocks2::one()) {
        if (terms.empty()) {
            num_vars = term->get_num_vars();
        } else if (term->get_num_vars() != num_vars) {
            throw std::runtime_error("MLE_Sum::add: number of variables mismatch");
        }
        terms.push_back(std::move(term));
        coeff.push_back(c);
    }

    Goldilocks2::Element evaluate(const std::vector<Goldilocks2::Element>& point) const override {
        Goldilocks2::Element result = Goldilocks2::zero();
        for (size_t i = 0; i < terms.size(); ++i) {
            result += coeff[i] * terms[i]->evaluate(point);
        }
        return result;
    }

    Goldilocks2::Element open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const override {
        Goldilocks2::Element result = Goldilocks2::zero();
        for (size_t i = 0; i < terms.size(); ++i) {
            result += coeff[i] * terms[i]->open(z, sec_param);
        }
        return result;
    }

    void init_eval() {
        evaluations.resize(1ull << num_vars, Goldilocks2::zero());
        #pragma omp parallel for
        for (size_t i = 0; i < evaluations.size(); ++i) {
            for (size_t j = 0; j < terms.size(); ++j) {
                evaluations[i] += coeff[j] * (*terms[j])[i];
            }
        }
    }    

    
protected:
    std::vector<Goldilocks2::Element> coeff;
    std::vector<std::unique_ptr<MLE>> terms;
};
