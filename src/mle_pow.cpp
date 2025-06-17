#include "mle_pow.h"

MLE_Pow::MLE_Pow(Goldilocks2::Element beta, size_t num_vars, size_t degree, bool init)
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

Goldilocks2::Element MLE_Pow::evaluate(const std::vector<Goldilocks2::Element>& point) const {
    return eval_power_mle(beta, point, degree, num_vars);
}

std::vector<Goldilocks2::Element> MLE_Pow::get_eval_table() const {
    if (evaluations.size() == 0) {
        throw std::runtime_error("MLE_Pow: evaluation table is not initialized");
        return {};
    }
    return evaluations;
}


