#pragma once
#include <format>
#include "data_loader.h"

class VCG16 {
public:
    enum layer_type {
        conv_relu, pool, full, relu, softmax, flat
    };
    class layer_info {
    public:
        layer_info();
        layer_type type;
        std::string name;
        array_view<int64_t> input, d_input;
        array_view<int64_t> weight, d_weight;
        array_view<int64_t> output, d_output; // Must be the input of the next layer
    };
    VCG16(std::string data_dir, int epoch, int64_t scale);

    bool check(int n_samples = 0) const;

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

    bool check_conv_relu(int n_samples = 0) const;
};

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