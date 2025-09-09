#pragma once

#include "CNN.h"


CNN::layer_res pre_prove_conv_layer(const CNN::layer_info& layer, CNN::conv_wit wit, lazy_pcs_pool *pool);

bool prove_conv_layer(const CNN::layer_info& layer, CNN::conv_wit wit, size_t rho_inv, size_t sec_param);

CNN::layer_res pre_prove_relu_layer(const CNN::layer_info& layer, size_t scale, size_t max_val, size_t sqr_val, size_t rho_inv, lazy_pcs_pool *pool);

bool prove_relu_layer(const CNN::layer_info& layer,
    size_t scale, size_t max_val, size_t sqr_val,
    size_t rho_inv, size_t sec_param, const CNN::layer_res& wit);

CNN::layer_res pre_prove_pool_layer(const CNN::layer_info& layer, size_t scale, size_t max_val, size_t rho_inv, lazy_pcs_pool* pool);

bool prove_pool_layer(const CNN::layer_info& layer, size_t scale, size_t max_val, size_t rho_inv, size_t sec_param, const CNN::layer_res& wit);

bool prove_softmax(const CNN::layer_info& layer, size_t scale, size_t max_val, size_t rho_inv, size_t sec_param);

bool prove_full_layer(const CNN::layer_info& layer, size_t rho_inv, size_t sec_param);

bool prove_flat_layer(const CNN::layer_info& layer, size_t rho_inv, size_t sec_param);

