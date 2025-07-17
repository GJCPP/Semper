#pragma once

#include "VCG16.h"

bool prove_conv(
    size_t C, size_t D, size_t n, size_t m, size_t padding,
    const oracle* pcs_input, const oracle* pcs_weight, const oracle* pcs_output,
    const array_view<Goldilocks2::Element>& X, // [C, n, n]
    const array_view<Goldilocks2::Element>& W, // [D, C, m, m]
    const array_view<Goldilocks2::Element>& Y,
    size_t rho_inv, size_t sec_param); // [D, n, n]

bool prove_conv_layer(VCG16::layer_info layer, size_t rho_inv, size_t sec_param);
