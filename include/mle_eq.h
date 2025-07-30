#pragma once
#include <cassert>

#include "mle.h"
#include "util.h"

/*
 * eq_r(x) = \prod_{i=0}^{num_var-1} ( r_i*x_i + (1-r_i)*(1-x_i) )
*/
class MLE_Eq : public MultilinearPolynomial {
public:
    MLE_Eq(const std::vector<Goldilocks2::Element>& r);
    MLE_Eq(const std::vector<Goldilocks2::Element>::const_iterator begin, const std::vector<Goldilocks2::Element>::const_iterator end);

    std::unique_ptr<MultilinearPolynomial> clone() const override {
        return std::make_unique<MLE_Eq>(*this);
    }

    Goldilocks2::Element evaluate(const std::vector<Goldilocks2::Element>& point) const override;

    Goldilocks2::Element open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const override;
private:
    std::vector<Goldilocks2::Element> r, one_minus_r;
};



