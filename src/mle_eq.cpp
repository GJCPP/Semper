#include "mle_eq.h"

MLE_Eq::MLE_Eq(const size_t& num_var, const std::vector<Goldilocks2::Element>& r)
    : MultilinearPolynomial(num_var), one_minus_r(r.size()) {
    assert(r.size() == num_var);
    this->r = r;
    Goldilocks2::Element one = Goldilocks2::one();
    for (size_t i = 0; i < num_var; ++i) {
        one_minus_r[i] = one - r[i];
    }
    evaluations = eq_table(num_var, r);
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
