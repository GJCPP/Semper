#include "mle_eq.h"

template <typename field>
MLE_Eq<field>::MLE_Eq(const std::vector<field>& r)
    : MultilinearPolynomial<field>(r.size()), one_minus_r(r.size()) {

    this->r = r;
    field one = field::one();
    for (int i = 0; i < num_vars; ++i) {
        one_minus_r[i] = one - r[i];
    }
    evaluations = eq_table(num_vars, r);
}

template <typename field>
MLE_Eq<field>::MLE_Eq(const std::vector<field>::const_iterator begin, const std::vector<field>::const_iterator end)
    : MultilinearPolynomial<field>(end - begin), one_minus_r(end - begin) {

    this->r = std::vector<field>(begin, end);
    field one = field::one();
    for (int i = 0; i < num_vars; ++i) {
        one_minus_r[i] = one - r[i];
    }
    evaluations = eq_table(num_vars, r);
}

template <typename field>
field MLE_Eq<field>::evaluate(const std::vector<field>& point) const {
    // eq_r(x) = \prod_{i=0}^{num_var-1} ( r_i*x_i + (1-r_i)*(1-x_i) )
    field one = field::one();
    field result = one;
    for (int i = 0; i < num_vars; ++i) {
        result = result * (r[i] * point[i] + one_minus_r[i] * (one - point[i]));
    }
    return result;
}

template <typename field>
field MLE_Eq<field>::open(const std::vector<field>& z, const size_t& sec_param) const {
    if (z.size() != size_t(num_vars)) {
        throw std::runtime_error("MLE_Eq::open: z size does not match number of variables.");
    }
    field result = field::one();
    for (int i = 0; i < num_vars; ++i) {
        result = result * (r[i] * z[i] + one_minus_r[i] * (field::one() - z[i]));
    }
    return result;
}
