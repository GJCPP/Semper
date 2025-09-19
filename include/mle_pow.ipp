#include "mle_pow.h"

template <typename field>
MLE_Pow<field>::MLE_Pow(field beta, size_t num_vars, size_t degree, bool init)
    : MultilinearPolynomial(num_vars, init), beta(beta), degree(degree)
{
    assert(degree < (1 << num_vars) && degree >= 0);

    // evaluations is already initialized in MultilinearPolynomial constructor
    if (init) {
        evaluations[0] = Goldilocks2::one();
        for (size_t i = 0; i < degree; ++i) {
            evaluations[i + 1] = evaluations[i] * beta;
        }
    } else {
        assert(evaluations.size() == 0);
    }
}

template <typename field>
field MLE_Pow<field>::evaluate(const std::vector<field>& point) const {
    return eval_power_mle(beta, point, degree, num_vars);
}

