#pragma once
#include <string>
#include <vector>
#include <cnpy.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>

cnpy::npz_t loadEpochData(const std::string& dir, int epoch);

void play_with_data_loader();


// Check Conv2D + ReLU using xtensor
template <typename T1, typename T2, typename T3>
bool verify_conv_relu(
    const T1& input, // [N, C, H+2, W+2]
    const T2& weights, // [OC, C, 3, 3]
    const T3& expected, // [N, OC, H, W]
    int64_t scale
) {
    size_t N = input.shape(0);
    size_t C = input.shape(1);
    size_t H = expected.shape(2);
    size_t W = expected.shape(3);
    size_t OC = weights.shape(0);
    assert(input.shape(0) == N);
    assert(input.shape(1) == C);
    assert(input.shape(2) == H + 2);
    assert(input.shape(3) == W + 2);
    assert(weights.shape(0) == OC);
    assert(weights.shape(1) == C);
    assert(weights.shape(2) == 3);
    assert(weights.shape(3) == 3);
    assert(expected.shape(0) == N);
    assert(expected.shape(1) == OC);
    assert(expected.shape(2) == H);
    assert(expected.shape(3) == W);

    for (size_t n = 0; n < N; ++n) {
        for (size_t oc = 0; oc < OC; ++oc) {
            for (size_t i = 0; i < H; ++i) {
                for (size_t j = 0; j < W; ++j) {
                    int64_t acc = 0;
                    for (size_t c = 0; c < C; ++c) {
                        auto in_patch = xt::view(input, n, c, xt::range(i, i + 3), xt::range(j, j + 3));
                        auto w_patch = xt::view(weights, oc, c, xt::all(), xt::all());
                        acc += xt::sum(in_patch * w_patch)();
                    }
                    acc = acc / scale;
                    if (acc < 0) acc = 0;

                    int64_t actual = expected(n, oc, i, j);
                    if (std::abs(actual - acc) > 1) {
                        std::cout << "❌ Mismatch at (n=" << n << ", oc=" << oc << ", i=" << i << ", j=" << j << "): manual=" << acc << ", expected=" << actual << std::endl;
                        std::cout << "scale = " << scale << std::endl;
                        return false;
                    }
                }
            }
        }
    }
    return true;
}