#include "VCG16.h"
#include "conv_check.h"



bool prove_conv(
    const array_view<Goldilocks2::Element>& input, // [C, n, n]
    const array_view<Goldilocks2::Element>& weights, // [D, C, 3, 3]
    const array_view<Goldilocks2::Element>& expected, // [D, n, n]
    const ligeropcs_base *pcs_input,
    const ligeropcs_base *pcs_weights,
    const ligeropcs_base *pcs_expected,
    size_t pad, // = 1
    int sec_param
) {
    
    size_t C = input.shape(0), D = weights.shape(0), n = input.shape(1), m = weights.shape(2);
    auto prover = make_conv2_prover(C, D, n, m, pad, input, weights, expected);
    return convVerifier::execute_convcheck_2d(prover, {pcs_input, pcs_weights, pcs_expected}, sec_param);

}