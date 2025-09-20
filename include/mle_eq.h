#pragma once
#include <cassert>

#include "mle.h"
#include "util.h"

/*
 * eq_r(x) = \prod_{i=0}^{num_var-1} ( r_i*x_i + (1-r_i)*(1-x_i) )
*/
template <typename field>
class MLE_Eq : public MultilinearPolynomial<field> {
public:
    MLE_Eq(const std::vector<field>& r);
    MLE_Eq(const std::vector<field>::const_iterator begin, const std::vector<field>::const_iterator end);

    std::unique_ptr<MultilinearPolynomial<field>> clone() const override {
        return std::make_unique<MLE_Eq<field>>(*this);
    }

    field evaluate(const std::vector<field>& point) const override;

    field open(const std::vector<field>& z, const size_t& sec_param) const override;

    inline int get_num_vars() const override { return static_cast<int>(r.size()); }
private:
    std::vector<field> r, one_minus_r;
};

// one-time computation of eq_r(x)
template <typename field>
inline field compute_eq(const std::vector<field>& r, const std::vector<field>& x) {
    if (r.size() != x.size()) {
        throw std::runtime_error("compute_eq: r and x size do not match.");
    }
    field result = field::one();
    field one = field::one();
    for (size_t i = 0; i < r.size(); ++i) {
        result = result * (r[i] * x[i] + (one - r[i]) * (one - x[i]));
    }
    return result;
}

template <typename field>
class MLE_Eq_Oracle : public oracle<field> {
public:
    MLE_Eq_Oracle(const std::vector<field>& r) : r(r) {}
    MLE_Eq_Oracle(const std::vector<field>::const_iterator begin, const std::vector<field>::const_iterator end)
        : r(begin, end) {}

    field open(const std::vector<field>& z, const size_t& sec_param) const override {
        return compute_eq(r, z);
    }

    inline int get_num_vars() const override { return static_cast<int>(r.size()); }
protected:
    std::vector<field> r;
};

