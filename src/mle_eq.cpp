#include "mle_eq.h"

MLE_Eq::MLE_Eq(const std::vector<Goldilocks2::Element>& r)
    : MultilinearPolynomial(r.size()), one_minus_r(r.size()) {

    this->r = r;
    Goldilocks2::Element one = Goldilocks2::one();
    for (size_t i = 0; i < num_vars; ++i) {
        one_minus_r[i] = one - r[i];
    }
    evaluations = eq_table(num_vars, r);
}

MLE_Eq::MLE_Eq(const std::vector<Goldilocks2::Element>::const_iterator begin, const std::vector<Goldilocks2::Element>::const_iterator end)
    : MultilinearPolynomial(end - begin), one_minus_r(end - begin) {

    this->r = std::vector<Goldilocks2::Element>(begin, end);
    Goldilocks2::Element one = Goldilocks2::one();
    for (size_t i = 0; i < num_vars; ++i) {
        one_minus_r[i] = one - r[i];
    }
    evaluations = eq_table(num_vars, r);
}

Goldilocks2::Element MLE_Eq::evaluate(const std::vector<Goldilocks2::Element>& point) const {
    // eq_r(x) = \prod_{i=0}^{num_var-1} ( r_i*x_i + (1-r_i)*(1-x_i) )
    Goldilocks2::Element one = Goldilocks2::one();
    Goldilocks2::Element result = one;
    for (size_t i = 0; i < num_vars; ++i) {
        result = result * (r[i] * point[i] + one_minus_r[i] * (one - point[i]));
    }
    return result;
}
