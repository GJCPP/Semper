#pragma once
#include "mle.h"
#include "util.h"

/*
  The vector to be extended is of form
  [1, beta, beta^2, ..., beta^(degree-1)]
*/
class MLE_Pow : public MultilinearPolynomial {
public:
    MLE_Pow(Goldilocks2::Element beta, size_t num_vars, size_t degree, bool init=false);
    
    std::unique_ptr<MultilinearPolynomial> clone() const override {
        return std::make_unique<MLE_Pow>(*this);
    }

    // Cost O(degree^2)
    Goldilocks2::Element evaluate(const std::vector<Goldilocks2::Element>& point) const override;

    inline int get_num_vars() const override { return num_vars; }
    
protected:
    Goldilocks2::Element beta;
    size_t degree;
};
