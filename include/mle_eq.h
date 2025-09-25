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

    inline int get_num_vars() const override { return static_cast<int>(r.size()); }
private:
    std::vector<Goldilocks2::Element> r, one_minus_r;
};

// one-time computation of eq_r(x)
inline Goldilocks2::Element compute_eq(const std::vector<Goldilocks2::Element>& r, const std::vector<Goldilocks2::Element>& x) {
    if (r.size() != x.size()) {
        throw std::runtime_error("compute_eq: r and x size do not match.");
    }
    Goldilocks2::Element result = Goldilocks2::one();
    Goldilocks2::Element one = Goldilocks2::one();
    for (size_t i = 0; i < r.size(); ++i) {
        result = result * (r[i] * x[i] + (one - r[i]) * (one - x[i]));
    }
    return result;
}

class MLE_Eq_Oracle : public oracle {
public:
    MLE_Eq_Oracle(const std::vector<Goldilocks2::Element>& r) : r(r) {}
    MLE_Eq_Oracle(const std::vector<Goldilocks2::Element>::const_iterator begin, const std::vector<Goldilocks2::Element>::const_iterator end)
        : r(begin, end) {}

    Goldilocks2::Element open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const override {
        return compute_eq(r, z);
    }

    inline int get_num_vars() const override { return static_cast<int>(r.size()); }
protected:
    std::vector<Goldilocks2::Element> r;
};