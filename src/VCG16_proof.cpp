#include <cassert>

#include "VCG16_proof.h"
#include "conv_check.h"
#include "utils.h"

bool prove_conv(
    size_t C, size_t D, size_t n, size_t m, size_t padding,
    const oracle* pcs_input, const oracle* pcs_weight, const oracle* pcs_output,
    const array_view<Goldilocks2::Element>& X, // [C, n, n]
    const array_view<Goldilocks2::Element>& W, // [D, C, m, m]
    const array_view<Goldilocks2::Element>& Y) { // [D, n, n]

    assert(X.shape(0) == C);
    assert(X.shape(1) == n);
    assert(X.shape(2) == n);
    assert(W.shape(0) == D);
    assert(W.shape(1) == C);
    assert(W.shape(2) == m);
    assert(W.shape(3) == m);
    assert(Y.shape(0) == D);
    assert(Y.shape(1) == n);
    assert(Y.shape(2) == n);

    assert(is_power_of_two(m));

    std::array<const oracle*, 3> oracle = { pcs_input, pcs_weight, pcs_output };
    auto prover = make_conv2_prover(C, D, n, m, padding, X, W, Y);

    // prover.triple.W

    if (!convVerifier::execute_convcheck_2d(prover, oracle, { &p1, &p2, &p3 }, 32)) {
        return false;
    }
    return false;
}