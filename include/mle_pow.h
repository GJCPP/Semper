#pragma once
#include "mle.h"
#include "util.h"

/*
  The vector to be extended is of form
  [1, beta, beta^2, ..., beta^(degree-1)]
*/
template <typename field>
class MLE_Pow : public MultilinearPolynomial<field> {
public:
    MLE_Pow(field beta, size_t num_vars, size_t degree, bool init=false);
    
    std::unique_ptr<MultilinearPolynomial<field>> clone() const override {
        return std::make_unique<MLE_Pow>(*this);
    }

    // Cost O(degree^2)
    field evaluate(const std::vector<field>& point) const override;

    inline int get_num_vars() const override { return num_vars; }
    
protected:
    field beta;
    size_t degree;
};
