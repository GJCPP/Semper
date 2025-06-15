#include <cassert>

#include "mle.h"
#include "util.h"

/*
 * eq_r(x) = \prod_{i=0}^{num_var-1} ( r_i*x_i + (1-r_i)*(1-x_i) )
*/
class MLE_Eq : public MultilinearPolynomial {
public:
    MLE_Eq(const size_t& num_var, const std::vector<Goldilocks2::Element>& r);

    Goldilocks2::Element evaluate(const std::vector<Goldilocks2::Element>& point) const override;
private:
    std::vector<Goldilocks2::Element> r, one_minus_r;
};



