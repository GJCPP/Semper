#include "VCG16.h"
#include "ligero.h"

bool prove_conv(
    const array_view<Goldilocks2::Element>& input, // [C, D, n, n]
    const array_view<Goldilocks2::Element>& weights, // [D, C, 3, 3]
    const array_view<Goldilocks2::Element>& expected, // [N, D, n, n]
    const ligeropcs_base *pcs_input,
    const ligeropcs_base *pcs_weights,
    const ligeropcs_base *pcs_expected,
    size_t pad, // = 1
    int sec_param
);
