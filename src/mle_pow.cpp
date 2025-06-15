#include "mle_pow.h"

MLE_Pow::MLE_Pow(Goldilocks2::Element beta, size_t num_vars, size_t degree)
    : MultilinearPolynomial(num_vars), beta(beta), degree(degree)
{
    assert(degree < (1 << num_vars) && degree >= 0);

    // evaluations is already initialized in MultilinearPolynomial constructor
    evaluations[0] = Goldilocks2::one();
    for (size_t i = 0; i < degree; ++i) {
        evaluations[i + 1] = evaluations[i] * beta;
    }
}

Goldilocks2::Element MLE_Pow::evaluate(const std::vector<Goldilocks2::Element>& point) const {
    return eval_power_mle(beta, point, degree, num_vars);
}


