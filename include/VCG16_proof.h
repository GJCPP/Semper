#pragma once

#include "VCG16.h"


bool prove_conv_layer(const VCG16::layer_info& layer, size_t rho_inv, size_t sec_param);

bool prove_relu_layer(const VCG16::layer_info& layer,
    const std::vector<size_t>& sign_from, ligeropcs_base pcs_sign_from,
    const std::vector<size_t>& sign_to, ligeropcs_base pcs_sign_to,
    const std::vector<size_t>& relu_from, ligeropcs_base pcs_relu_from,
    const std::vector<size_t>& relu_to, ligeropcs_base pcs_relu_to,
    const std::vector<size_t>& scale_from, ligeropcs_base pcs_scale_from,
    const std::vector<size_t>& scale_to, ligeropcs_base pcs_scale_to,
    size_t scale,
    size_t rho_inv, size_t sec_param);

bool prove_full_layer(const VCG16::layer_info& layer, size_t rho_inv, size_t sec_param);
