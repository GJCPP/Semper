
#include "VCG16.h"

VCG16::VCG16(std::string data_dir, int epoch, int64_t scale)
    : epoch(epoch), scale(scale) {
    filedata = loadEpochData(data_dir, epoch);
    for (auto& [key, value] : filedata) {
        data[key] = value.data<int64_t>();
        data_shape[key] = value.shape;
    }
    minibatch = data_shape["z_q0"][0];
    img_per_batch = data_shape["z_q0"][1];

    std::vector<std::vector<int>> conv_layers = {{1, 2}, {3, 4}, {5, 6, 7}, {8, 9, 10}, {11, 12, 13}};
    int ind_pool = 1;
    for (auto& layer : conv_layers) {
        for (auto& lid : layer) {
            add_layer(layer_type::conv_relu,
                std::format("conv_{}", lid),
                std::format("z_q{}", lid - 1),
                std::format("z_q{}", lid),
                std::format("W_conv_q{}", lid),
                std::format("grad_z_q{}", lid - 1),
                std::format("grad_z_q{}", lid),
                std::format("dW_conv_q{}", lid));
        }
        add_layer(layer_type::pool,
            std::format("pool_{}", ind_pool),
            std::format("z_q{}", layer.back()),
            std::format("pool_q{}", ind_pool),
            {},
            std::format("grad_z_q{}", layer.back()),
            std::format("grad_pool_q{}", ind_pool),
            {});
        ++ind_pool;
    }
    add_layer(layer_type::flat,
        "flat",
        std::format("pool_q{}", ind_pool - 1),
        "flat_q",
        {},
        std::format("grad_pool_q{}", ind_pool),
        std::format("grad_flat_q"),
        {});
}

bool VCG16::check(int n_samples) const {
    return check_conv_relu(n_samples);
}

void VCG16::add_layer(layer_type type,
                      const std::string& name,
                      const std::string& input,
                      const std::string& output,
                      const std::string& weight,
                      const std::string& d_input,
                      const std::string& d_output,
                      const std::string& d_weight) {
    layer_info info;
    info.type = type;
    info.name = name;
    info.input = array_view<int64_t>(data[input], data_shape[input]);
    info.output = array_view<int64_t>(data[output], data_shape[output]);
    info.weight = array_view<int64_t>(data[weight], data_shape[weight]);
    info.d_input = array_view<int64_t>(data[d_input], data_shape[d_input]);
    info.d_output = array_view<int64_t>(data[d_output], data_shape[d_output]);
    info.d_weight = array_view<int64_t>(data[d_weight], data_shape[d_weight]);
    layers.push_back(info);
}

bool VCG16::check_conv_relu(int n_samples) const {
    std::vector<std::vector<int>> layers = {{1, 2}, {3, 4}, {5, 6, 7}, {8, 9, 10}, {11, 12, 13}};
    int ind_pool = 1;
    for (auto& layer : layers) {
        for (auto& lid : layer) {
            std::cout << "Checking layer " << lid << std::endl;
            std::string input_key, weights_key, expected_out_key;

            if (lid == 1) {
                input_key = "input_q";
            }
            else {
                if (lid == layer.front()) {
                    input_key = std::format("pool_q{}", ind_pool);
                    ind_pool++;
                }
                else {
                    input_key = std::format("z_q{}", lid - 1);
                }
            }
            weights_key = std::format("W_conv_q{}", lid);
            expected_out_key = std::format("z_q{}", lid);
            array_view<int64_t> input = array_view<int64_t>(data.at(input_key), data_shape.at(input_key));
            array_view<int64_t> weights = array_view<int64_t>(data.at(weights_key), data_shape.at(weights_key));
            array_view<int64_t> expected_out = array_view<int64_t>(data.at(expected_out_key), data_shape.at(expected_out_key));
            
            for (size_t mini_batch = 0; mini_batch < input.shape(0); ++mini_batch) {
                std::cout << "\tChecking mini_batch " << mini_batch << std::endl;
                auto view_input = input[mini_batch];
                auto view_weights = weights[mini_batch];
                auto view_expected_out = expected_out[mini_batch];

                if (n_samples > 0 &&
                    !random_check_conv_relu(
                        view_input,
                        view_weights,
                        view_expected_out,
                        scale,
                        n_samples) ||
                    n_samples == 0 &&
                    !::check_conv_relu(
                        view_input,
                        view_weights,
                        view_expected_out,
                        scale)
                    ) {
                    std::cout << "Conv1 + ReLU output verification failed at layer " << lid << ".\n";
                    return false;
                }
            }
        }
    }
    return true;
}

bool check_conv_relu(
    const array_view<int64_t>& input, // [N, C, H, W]
    const array_view<int64_t>& weights, // [OC, C, 3, 3]
    const array_view<int64_t>& expected, // [N, OC, H, W]
    int64_t scale
) {
    size_t N = input.shape(0);
    size_t C = input.shape(1);
    size_t H = expected.shape(2);
    size_t W = expected.shape(3);
    size_t OC = weights.shape(0);
    assert(input.shape(0) == N);
    assert(input.shape(1) == C);
    assert(input.shape(2) == H);
    assert(input.shape(3) == W);
    assert(weights.shape(0) == OC);
    assert(weights.shape(1) == C);
    assert(weights.shape(2) == 3);
    assert(weights.shape(3) == 3);
    assert(expected.shape(0) == N);
    assert(expected.shape(1) == OC);
    assert(expected.shape(2) == H);
    assert(expected.shape(3) == W);

    bool ret = true;

#pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {

        auto view_input_n = input[n];
        auto view_expected_n = expected[n];

        for (size_t oc = 0; oc < OC && ret; ++oc) {

            auto view_weights_oc = weights[oc];
            auto view_expected_n_oc = view_expected_n[oc];

            for (size_t i = 0; i < H && ret; ++i) {

                auto view_expected_n_oc_h = view_expected_n_oc[i];

                for (size_t j = 0; j < W && ret; ++j) {

                    int64_t acc = 0;
                    for (size_t c = 0; c < C && ret; ++c) {

                        auto view_weights_oc_c = view_weights_oc[c];
                        auto view_input_n_c = view_input_n[c];

                        for (size_t ki = 0; ki < 3 && ret; ++ki) {
                            for (size_t kj = 0; kj < 3 && ret; ++kj) {
                                acc += view_input_n_c(i + ki - 1, j + kj - 1) * view_weights_oc_c(ki, kj);
                            }
                        }
                    }
                    acc = acc / scale;
                    if (acc < 0) acc = 0;

                    int64_t actual = view_expected_n_oc_h(j);
                    if (std::abs(actual - acc) > 1) {
                        ret = false;
                        break;
                    }
                }
            }
        }
    }
    return ret;
}


bool random_check_conv_relu(
    const array_view<int64_t>& input, // [N, C, H, W]
    const array_view<int64_t>& weights, // [OC, C, 3, 3]
    const array_view<int64_t>& expected, // [N, OC, H, W]
    int64_t scale,
    size_t n_samples
) {
    size_t N = input.shape(0);
    size_t C = input.shape(1);
    size_t H = expected.shape(2);
    size_t W = expected.shape(3);
    size_t OC = weights.shape(0);
    assert(input.shape(0) == N);
    assert(input.shape(1) == C);
    assert(input.shape(2) == H);
    assert(input.shape(3) == W);
    assert(weights.shape(0) == OC);
    assert(weights.shape(1) == C);
    assert(weights.shape(2) == 3);
    assert(weights.shape(3) == 3);
    assert(expected.shape(0) == N);
    assert(expected.shape(1) == OC);
    assert(expected.shape(2) == H);
    assert(expected.shape(3) == W);

    bool ret = true;

#pragma omp parallel for
    for (size_t cnt = 0; cnt < n_samples; ++cnt) {
        size_t n = rand() % N;
        size_t oc = rand() % OC;
        size_t i = rand() % H;
        size_t j = rand() % W;
        int64_t acc = 0;
        for (size_t c = 0; c < C; ++c) {
            for (size_t ki = 0; ki < 3; ++ki) {
                for (size_t kj = 0; kj < 3; ++kj) {
                    acc += input(n, c, i + ki - 1, j + kj - 1) * weights(oc, c, ki, kj);
                }
            }
        }
        acc = acc / scale;
        if (acc < 0) acc = 0;

        int64_t actual = expected(n, oc, i, j);
        if (std::abs(actual - acc) > 1) {
            std::cout << "❌ Mismatch at (n=" << n << ", oc=" << oc << ", i=" << i << ", j=" << j << "): manual=" << acc << ", expected=" << actual << std::endl;
            std::cout << "scale = " << scale << std::endl;
            ret = false;
        }
    }
    return ret;
}