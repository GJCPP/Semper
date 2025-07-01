#pragma once
#include <format>
#include <random>
#include "data_loader.h"

class VCG16 {
public:
    enum layer_type {
        conv, pool, full, relu, softmax, flat
    };
    class layer_info {
    public:
        layer_info() = default;
        layer_type type;
        std::string name;
        array_view<int64_t> input, d_input;
        array_view<int64_t> weight, d_weight;
        array_view<int64_t> output, d_output; // Must be the input of the next layer
    };
    VCG16(std::string data_dir, int epoch, int64_t scale);

    bool check(size_t n_samples = 0) const;

    void add_layer(layer_type type,
                    const std::string& name,
                    const std::string& input,
                    const std::string& output,
                    const std::string& weight,
                    const std::string& d_input,
                    const std::string& d_weight,
                    const std::string& d_output);

protected:
    int64_t scale;
    int epoch, minibatch, img_per_batch;
    cnpy::npz_t filedata; // responsible for releasing data
    std::map<std::string, int64_t*> data;
    std::map<std::string, std::vector<size_t>> data_shape;
    std::vector<layer_info> layers;

    // bool check_conv_relu(int n_samples = 0) const;
};

bool check_conv(
    const array_view<int64_t>& input, // [N, C, H, W]
    const array_view<int64_t>& weights, // [OC, C, 3, 3]
    const array_view<int64_t>& expected, // [N, OC, H, W]
    size_t pad,
    size_t n_samples = 0
);

bool random_check_conv(
    const array_view<int64_t>& input, // [N, C, H, W]
    const array_view<int64_t>& weights, // [OC, C, 3, 3]
    const array_view<int64_t>& expected, // [N, OC, H, W]
    size_t pad,
    size_t n_samples
);

void add_conv(
    const array_view<int64_t>& input, // [H, W]
    const array_view<int64_t>& weights, // [K, K]
    array_view<int64_t>& output, // [H + 2 * pad - K + 1, W + 2 * pad - K + 1]
    size_t pad
);

bool check_relu(
    const array_view<int64_t>& sign,
    const array_view<int64_t>& input,
    const array_view<int64_t>& output,
    int64_t scale,
    size_t n_samples = 0
);

bool random_check_relu(
    const array_view<int64_t>& sign,
    const array_view<int64_t>& input,
    const array_view<int64_t>& output,
    int64_t scale,
    size_t n_samples
);