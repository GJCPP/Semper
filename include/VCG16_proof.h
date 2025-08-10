#pragma once

#include "VCG16.h"


bool prove_conv_layer(const VCG16::layer_info& layer, size_t rho_inv, size_t sec_param);

bool prove_relu_layer(const VCG16::layer_info& layer,
    size_t scale, size_t max_val, size_t sqr_val,
    size_t rho_inv, size_t sec_param);

bool prove_pool_layer(const VCG16::layer_info& layer, size_t scale, size_t max_val, size_t rho_inv, size_t sec_param);

bool prove_softmax(const VCG16::layer_info& layer, size_t scale, size_t max_val, size_t rho_inv, size_t sec_param);

bool prove_full_layer(const VCG16::layer_info& layer, size_t rho_inv, size_t sec_param);

bool prove_flat_layer(const VCG16::layer_info& layer, size_t rho_inv, size_t sec_param);

