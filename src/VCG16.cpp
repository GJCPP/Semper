
#include "VCG16.h"

VCG16::VCG16(std::string data_dir, int epoch, int64_t scale)
    : epoch(epoch), scale(scale) {
    filedata = loadEpochData(data_dir, epoch);
    for (auto& [key, value] : filedata) {
        data[key] = value.data<int64_t>();
        data_shape[key] = value.shape;
    }
    data[{}] = nullptr;
    data_shape[{}] = {};
    minibatch = data_shape["a_q0"][0];
    img_per_batch = data_shape["a_q0"][1];

    std::vector<std::vector<int>> conv_layers = {{1, 2}, {3, 4}, {5, 6, 7}, {8, 9, 10}, {11, 12, 13}};
    int ind_pool = 1;
    for (auto& layer : conv_layers) {
        for (auto& lid : layer) {
            std::string input_name;
            if (lid > 1 && lid == layer.front()) {
                input_name = std::format("pool_q{}", ind_pool - 1);
            }
            else {
                input_name = std::format("a_q{}", lid - 1);
            }
            add_layer(layer_type::conv,
                std::format("conv_{}", lid),
                input_name,
                std::format("z_q{}", lid),
                std::format("W_conv_q{}", lid),
                std::format("grad_{}", input_name),
                std::format("grad_z_q{}", lid),
                std::format("dW_conv_q{}", lid));
            add_layer(layer_type::relu,
                std::format("relu_{}", lid),
                std::format("z_q{}", lid),
                std::format("a_q{}", lid),
                {},
                std::format("grad_z_q{}", lid),
                std::format("grad_a_q{}", lid),
                {});
        }
        add_layer(layer_type::pool,
            std::format("pool_{}", ind_pool),
            std::format("a_q{}", layer.back()),
            std::format("pool_q{}", ind_pool),
            {},
            std::format("grad_a_q{}", layer.back()),
            std::format("grad_pool_q{}", ind_pool),
            {},
            std::format("pool_idx_q{}", ind_pool));
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
    for (int layer = 1; layer <= 3; ++layer) {
        std::string input_name = layer == 1 ? "flat_q" : std::format("a{}_q", layer - 1);
        add_layer(layer_type::full,
            std::format("full_{}", layer),
            input_name,
            std::format("z{}_q", layer),
            std::format("W_fc{}_q", layer),
            std::format("grad_{}", input_name),
            std::format("grad_z{}_q", layer),
            std::format("dW_fc{}_q", layer));
        if (layer < 3) {
            add_layer(layer_type::relu,
                std::format("fc_relu_{}", layer),
                std::format("z{}_q", layer),
                std::format("a{}_q", layer),
                {},
                std::format("grad_z{}_q", layer),
                std::format("grad_a{}_q", layer),
                {});
        }
    }
    add_layer(layer_type::softmax,
        "softmax",
        "z3_q",
        "probs_q",
        {},
        "probs_q",
        {},
        {});
}

bool VCG16::check(size_t n_samples) const {
    for (auto& layer : layers) {
        bool pass = true;
        std::cout << "Checking layer " << layer.name << std::endl;
        switch (layer.type) {
            case layer_type::conv:
                // Check forward pass
                for (size_t i = 0; i < layer.input.shape(0) && pass; ++i) { // mini-batch
                    std::cout << "Checking layer " << layer.name << " (forward) for mini-batch " << i << std::endl;
                    pass &= check_conv(layer.input[i], layer.weight[i], layer.output[i], 1, n_samples);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (forward)" << std::endl;
                    break;
                }

                // Check backward pass, check d_weight
                for (size_t i = 0; i < layer.input.shape(0) && pass; ++i) { // mini-batch
                    std::cout << "Checking layer " << layer.name << " (backward, d_weight) for mini-batch " << i << std::endl;
                    array_view<int64_t> input_i(layer.input[i]);
                    array_view<int64_t> d_output_i(layer.d_output[i]);
                    array_view<int64_t> d_weight_i(layer.d_weight[i]);
                    input_i.swap_dim(0, 1);
                    d_output_i.swap_dim(0, 1);
                    d_weight_i.swap_dim(0, 1);
                    pass &= check_conv(input_i, d_output_i, d_weight_i, 1, n_samples);
                    // if (!pass) break;
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (backward, d_weight)" << std::endl;
                    break;
                }

                // Check backward pass, check d_input
                for (size_t i = 0; i < layer.input.shape(0) && pass; ++i) { // mini-batch
                    std::cout << "Checking layer " << layer.name << " (backward, d_input) for mini-batch " << i << std::endl;
                    array_view<int64_t> d_input_i(layer.d_input[i]);
                    array_view<int64_t> d_output_i(layer.d_output[i]);
                    array_view<int64_t> weight_i(layer.weight[i]);
                    weight_i.reverse(2);
                    weight_i.reverse(3);
                    weight_i.swap_dim(0, 1);
                    pass &= check_conv(d_output_i, weight_i, d_input_i, 1, n_samples);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (backward, d_input)" << std::endl;
                    break;
                }
                break;

            case layer_type::relu:
                std::cout << "Checking layer " << layer.name << " (forward)" << std::endl;
                pass &= check_relu(layer.input, layer.input, layer.output, scale, n_samples);
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (forward)" << std::endl;
                    break;
                }

                std::cout << "Checking layer " << layer.name << " (backward)" << std::endl;
                pass &= check_relu(layer.input, layer.d_output, layer.d_input, scale, n_samples);
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (backward)" << std::endl;
                    break;
                }
                break;

            case layer_type::pool:
                std::cout << "Checking layer " << layer.name << " (forward)" << std::endl;
                for (int i = 0; i < layer.input.shape(0); ++i) {
                    std::cout << "Checking layer " << layer.name << " (forward) for mini-batch " << i << std::endl;
                    pass &= check_pool(layer.input[i], layer.output[i], layer.aux[i], 2, 2, false, n_samples);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (forward)" << std::endl;
                    break;
                }
                std::cout << "Checking layer " << layer.name << " (backward)" << std::endl;
                for (int i = 0; i < layer.input.shape(0); ++i) {
                    std::cout << "Checking layer " << layer.name << " (backward) for mini-batch " << i << std::endl;
                    pass &= check_pool(layer.d_input[i], layer.d_output[i], layer.aux[i], 2, 2, true, n_samples);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (backward)" << std::endl;
                    break;
                }
                break;

            default:
                break;
        }
        if (!pass) {
            return false;
        }
    }
    std::cout << "✅ All layers passed." << std::endl;
    return true;
}

void VCG16::add_layer(layer_type type,
                      const std::string& name,
                      const std::string& input,
                      const std::string& output,
                      const std::string& weight,
                      const std::string& d_input,
                      const std::string& d_output,
                      const std::string& d_weight,
                      const std::string& aux) {
    layer_info info;
    info.type = type;
    info.name = name;
    info.input = array_view<int64_t>(data[input], data_shape[input]);
    info.output = array_view<int64_t>(data[output], data_shape[output]);
    info.weight = array_view<int64_t>(data[weight], data_shape[weight]);
    info.d_input = array_view<int64_t>(data[d_input], data_shape[d_input]);
    info.d_output = array_view<int64_t>(data[d_output], data_shape[d_output]);
    info.d_weight = array_view<int64_t>(data[d_weight], data_shape[d_weight]);
    info.aux = array_view<int64_t>(data[aux], data_shape[aux]);
    layers.push_back(info);
}


bool check_conv(
    const array_view<int64_t>& input, // [N, C, H, W]
    const array_view<int64_t>& weights, // [OC, C, K, K]
    const array_view<int64_t>& expected, // [N, OC, H + 2 * pad - K + 1, W + 2 * pad - K + 1]
    size_t pad,
    size_t n_samples
) {
    if (n_samples > 0) {
        return random_check_conv(input, weights, expected, pad, n_samples);
    }
    size_t N = input.shape(0);
    size_t C = input.shape(1);
    size_t H = input.shape(2);
    size_t W = input.shape(3);
    size_t OC = weights.shape(0);
    size_t K = weights.shape(2);
    size_t OH = H + 2 * pad - K + 1;
    size_t OW = W + 2 * pad - K + 1;
    assert(input.get_dims() == 4);
    assert(weights.get_dims() == 4);
    assert(expected.get_dims() == 4);
    assert(input.shape(0) == N);
    assert(input.shape(1) == C);
    assert(input.shape(2) == H);
    assert(input.shape(3) == W);
    assert(weights.shape(0) == OC);
    assert(weights.shape(1) == C);
    assert(weights.shape(2) == K);
    assert(weights.shape(3) == K);
    assert(expected.shape(0) == N);
    assert(expected.shape(1) == OC);
    assert(expected.shape(2) == OH);
    assert(expected.shape(3) == OW);

    bool ret = true;

#pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {

        auto view_input_n = input[n];
        auto view_expected_n = expected[n];

        for (size_t oc = 0; oc < OC && ret; ++oc) {

            auto view_weights_oc = weights[oc];
            auto view_expected_n_oc = view_expected_n[oc];

            for (size_t i = 0; i < OH && ret; ++i) {

                auto view_expected_n_oc_h = view_expected_n_oc[i];

                for (size_t j = 0; j < OW && ret; ++j) {

                    int64_t acc = 0;
                    for (size_t c = 0; c < C && ret; ++c) {

                        auto view_weights_oc_c = view_weights_oc[c];
                        auto view_input_n_c = view_input_n[c];

                        for (size_t ki = 0; ki < K && ret; ++ki) {
                            for (size_t kj = 0; kj < K && ret; ++kj) {
                                if (i + ki >= pad && i + ki < H + pad && j + kj >= pad && j + kj < W + pad) {
                                    acc += view_input_n_c(i + ki - pad, j + kj - pad) * view_weights_oc_c(ki, kj);
                                }
                            }
                        }
                    }
                    // acc = acc / scale; // downscale
                    // if (acc < 0) acc = 0; // relu

                    int64_t actual = view_expected_n_oc_h(j);
                    if (actual != acc) {
                        ret = false;
                        std::cout << "❌ Mismatch at (n=" << n << ", oc=" << oc << ", i=" << i << ", j=" << j << "): manual=" << acc << ", expected=" << actual << std::endl;
                    }
                }
            }
        }
    }
    return ret;
}

bool random_check_conv(
    const array_view<int64_t>& input, // [N, C, H, W]
    const array_view<int64_t>& weights, // [OC, C, K, K]
    const array_view<int64_t>& expected, // [N, OC, H + 2 * pad - K + 1, W + 2 * pad - K + 1]
    size_t pad,
    size_t n_samples
) {
    size_t N = input.shape(0);
    size_t C = input.shape(1);
    size_t H = input.shape(2);
    size_t W = input.shape(3);
    size_t K = weights.shape(2);
    size_t OC = weights.shape(0);
    size_t OH = H + 2 * pad - K + 1;
    size_t OW = W + 2 * pad - K + 1;
    assert(input.shape(0) == N);
    assert(input.shape(1) == C);
    assert(input.shape(2) == H);
    assert(input.shape(3) == W);
    assert(weights.shape(0) == OC);
    assert(weights.shape(1) == C);
    assert(weights.shape(2) == K);
    assert(weights.shape(3) == K);
    assert(expected.shape(0) == N);
    assert(expected.shape(1) == OC);
    assert(expected.shape(2) == OH);
    assert(expected.shape(3) == OW);

    bool ret = true;

#pragma omp parallel for
    for (size_t cnt = 0; cnt < n_samples; ++cnt) {
        size_t n = rand() % N;
        size_t oc = rand() % OC;
        size_t i = rand() % OH;
        size_t j = rand() % OW;
        int64_t acc = 0;
        for (size_t c = 0; c < C; ++c) {
            for (size_t ki = 0; ki < K; ++ki) {
                for (size_t kj = 0; kj < K; ++kj) {
                    if (i + ki >= pad && i + ki < H + pad && j + kj >= pad && j + kj < W + pad) {
                        acc += input(n, c, i + ki - pad, j + kj - pad) * weights(oc, c, ki, kj);
                    }
                }
            }
        }
        // acc = acc / scale; // downscale
        // if (acc < 0) acc = 0; // relu

        int64_t actual = expected(n, oc, i, j);
        if (actual != acc) {
            std::cout << "❌ Mismatch at (n=" << n << ", oc=" << oc << ", i=" << i << ", j=" << j << "): manual=" << acc << ", expected=" << actual << std::endl;
            ret = false;
        }
    }
    return ret;
}


void add_conv(const array_view<int64_t>& input, const array_view<int64_t>& weights, array_view<int64_t>& output, size_t pad) {
    
    size_t H = input.shape(0);
    size_t W = input.shape(1);
    size_t K = weights.shape(0);
    size_t OH = H + 2 * pad - K + 1;
    size_t OW = W + 2 * pad - K + 1;
    assert(input.get_dims() == 2);
    assert(weights.get_dims() == 2);
    assert(output.get_dims() == 2);
    assert(input.shape(0) == H);
    assert(input.shape(1) == W);
    assert(weights.shape(0) == K);
    assert(weights.shape(1) == K);
    assert(output.shape(0) == OH);
    assert(output.shape(1) == OW);

// #pragma omp parallel for
    for (size_t i = 0; i < OH; ++i) {
        for (size_t j = 0; j < OW; ++j) {
            int64_t acc = 0;
            for (size_t ki = 0; ki < K; ++ki) {
                for (size_t kj = 0; kj < K; ++kj) {
                    if (i + ki >= pad && i + ki < H + pad && j + kj >= pad && j + kj < W + pad) {
                        acc += input(i + ki - pad, j + kj - pad) * weights(ki, kj);
                    }
                }
            }
            output(i, j) += acc;
        }
    }
}

bool check_relu(
    const array_view<int64_t>& sign,
    const array_view<int64_t>& input, // [N, C, H, W]
    const array_view<int64_t>& output, // [N, C, H, W]
    int64_t scale,
    size_t n_samples
) {
    if (n_samples > 0) {
        return random_check_relu(sign, input, output, scale, n_samples);
    }
    assert(input.size() == output.size());

    bool ret = true;
    size_t sz = input.size();
    
    #pragma omp parallel for
    for (size_t i = 0; i < sz; ++i) {
        int64_t s = sign.get(i);
        int64_t actual = input.get(i);
        int64_t expected = output.get(i);
        if (s < 0) actual = 0;
        actual = actual / scale;
        if (std::abs(actual - expected) > 1) {
            ret = false;
            #pragma omp critical
            {
                std::cout << "❌ Mismatch at (i=" << i << "): manual=" << actual << ", expected=" << expected << std::endl;
            }
        }
    }
    return ret;
}

bool random_check_relu(
    const array_view<int64_t>& sign,
    const array_view<int64_t>& input, // [N, C, H, W]
    const array_view<int64_t>& output, // [N, C, H, W]
    int64_t scale,
    size_t n_samples
) {
    bool ret = true;
    size_t sz = input.size();
    static std::mt19937 rng(std::random_device{}());

    #pragma omp parallel for
    for (size_t cnt = 0; cnt < n_samples; ++cnt) {
        size_t i = std::uniform_int_distribution<size_t>(0, sz - 1)(rng);
        int64_t s = sign.get(i);
        int64_t actual = input.get(i);
        int64_t expected = output.get(i);
        if (s < 0) actual = 0;
        actual = actual / scale;
        if (std::abs(actual - expected) > 1) {
            ret = false;
            #pragma omp critical
            {
                std::cout << "❌ Mismatch at (i=" << i << "): manual=" << actual << ", expected=" << expected << std::endl;
            }
        }
    }
    return ret;
}

bool check_pool(
    const array_view<int64_t>& input,
    const array_view<int64_t>& output,
    const array_view<int64_t>& idx,
    size_t kernel_size,
    size_t stride,
    bool backward,
    size_t n_samples
) {
    size_t N = input.shape(0);
    size_t C = input.shape(1);
    size_t H = input.shape(2);
    size_t W = input.shape(3);
    size_t OH = H / stride;
    size_t OW = W / stride;
    assert(input.get_dims() == 4);
    assert(output.get_dims() == 4);
    assert(output.shape(0) == N);
    assert(output.shape(1) == C);
    assert(output.shape(2) == OH);
    assert(output.shape(3) == OW);
    assert(idx.get_dims() == 4);
    assert(idx.shape(0) == N);
    assert(idx.shape(1) == C);
    assert(idx.shape(2) == OH);
    assert(idx.shape(3) == OW);
    assert(H % stride == 0);
    assert(W % stride == 0);

    if (n_samples > 0) {
        return random_check_pool(input, output, idx, kernel_size, stride, backward, n_samples);
    }

    bool ret = true;
    #pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {

        auto view_input_n = input[n];
        auto view_output_n = output[n];
        auto view_idx_n = idx[n];

        for (size_t c = 0; c < C; ++c) {
            
            auto view_input_n_c = view_input_n[c];
            auto view_output_n_c = view_output_n[c];
            auto view_idx_n_c = view_idx_n[c];

            for (size_t i = 0; i < OH; ++i) {
                for (size_t j = 0; j < OW; ++j) {
                    size_t index = view_idx_n_c(i, j);
                    int64_t max_val = view_input_n_c.get(index);
                    for (size_t ki = 0; ki < kernel_size; ++ki) {
                        for (size_t kj = 0; kj < kernel_size; ++kj) {
                            size_t x = i * stride + ki;
                            size_t y = j * stride + kj;
                            if (backward) {
                                if (x * W + y != index && view_input_n_c(x, y) != 0) {
                                    ret = false;
                                    #pragma omp critical
                                    {
                                        std::cout << "❌ Incorrect at (n=" << n << ", c=" << c << ", i=" << i << ", j=" << j << "): expect 0-grad,"
                                                << ", but find " << view_input_n_c(x, y) << std::endl;
                                    }
                                }
                            } else {
                                if (view_input_n_c(x, y) > max_val) {
                                    ret = false;
                                    #pragma omp critical
                                    {
                                        std::cout << "❌ Incorrect at (n=" << n << ", c=" << c << ", i=" << i << ", j=" << j << "): expect max_val = " << max_val
                                                << ", but find larger value " << view_input_n_c(x, y) << std::endl;
                                    }
                                }
                            }
                        }
                    }
                    if (max_val != view_output_n_c(i, j)) {
                        ret = false;
                        #pragma omp critical
                        {
                            std::cout << "❌ Mismatch at (n=" << n << ", c=" << c << ", i=" << i << ", j=" << j << "): manual=" << max_val << ", expected=" << view_output_n_c(i, j) << std::endl;
                        }
                    }
                }
            }
        }
    }
    return ret;
}

bool random_check_pool(
    const array_view<int64_t>& input,
    const array_view<int64_t>& output,
    const array_view<int64_t>& idx,
    size_t kernel_size,
    size_t stride,
    bool backward,
    size_t n_samples
) {
    size_t N = input.shape(0);
    size_t C = input.shape(1);
    size_t H = input.shape(2);
    size_t W = input.shape(3);
    size_t OH = H / stride;
    size_t OW = W / stride;
    bool ret = true;

    #pragma omp parallel for
    for (size_t cnt = 0; cnt < n_samples; ++cnt) {
        size_t n = rand() % N;
        size_t c = rand() % C;
        size_t i = rand() % OH;
        size_t j = rand() % OW;
        size_t index = idx(n, c, i, j);
        int64_t max_val = input[n][c].get(index);
        if (max_val != output(n, c, i, j)) {
            ret = false;
            #pragma omp critical
            {
                std::cout << "❌ Mismatch at (n=" << n << ", c=" << c << ", i=" << i << ", j=" << j << "): manual=" << max_val << ", expected=" << output(n, c, i, j) << std::endl;
            }
        }
        for (size_t ki = 0; ki < kernel_size; ++ki) {
            for (size_t kj = 0; kj < kernel_size; ++kj) {
                size_t x = i * stride + ki;
                size_t y = j * stride + kj;
        
                if (backward) {
                    if (x * W + y != index && input(n, c, x, y) != 0) {
                        ret = false;
                        #pragma omp critical
                        {
                            std::cout << "❌ Incorrect at (n=" << n << ", c=" << c << ", i=" << i << ", j=" << j << "): expect 0-grad,"
                                    << ", but find " << input(n, c, x, y) << std::endl;
                        }
                    }
                } else {
                    if (input(n, c, x, y) > max_val) {
                        ret = false;
                        #pragma omp critical
                        {
                            std::cout << "❌ Incorrect at (n=" << n << ", c=" << c << ", i=" << i << ", j=" << j << "): expect max_val = " << max_val
                                    << ", but find larger value " << input(n, c, x, y) << std::endl;
                        }
                    }
                }
            }
        }
    }
    return ret;
}
