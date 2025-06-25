#pragma once
#include <string>
#include <vector>
#include <format>
#include <random>
#include <cnpy.h>

#include "highdim_array.h"

cnpy::npz_t loadEpochData(const std::string& dir, int epoch);

void play_with_data_loader();


bool check_conv_relu(
    const array_view<int64_t>& input, // [N, C, H, W]
    const array_view<int64_t>& weights, // [OC, C, 3, 3]
    const array_view<int64_t>& expected, // [N, OC, H, W]
    int64_t scale
);

bool random_check_conv_relu(
    const array_view<int64_t>& input, // [N, C, H, W]
    const array_view<int64_t>& weights, // [OC, C, 3, 3]
    const array_view<int64_t>& expected, // [N, OC, H, W]
    int64_t scale,
    size_t n_samples
);