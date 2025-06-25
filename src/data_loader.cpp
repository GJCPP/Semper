#include "data_loader.h"


cnpy::npz_t loadEpochData(const std::string& dir, int epoch) {
    std::string data_file = dir + "/epoch_" + std::to_string(epoch) + ".npz";
    cnpy::npz_t npz_data = cnpy::npz_load(data_file);

    cnpy::npz_t data_map;
    for (auto& pair : npz_data) {
        data_map[pair.first] = pair.second;
    }

    return data_map;
}

void play_with_data_loader() {
    cnpy::npz_t data = loadEpochData("/home/gaojc/Desktop/logup-main/training_trace", 0);
    for (auto& [key, value] : data) {
        std::cout << key << " of shape : "<< std::endl;
        for (auto& shape : value.shape) {
            std::cout << shape << " ";
        }
        std::cout << std::endl;
    }
    std::vector<std::vector<int>> layers = {{1, 2}, {3, 4}, {5, 6, 7}, {8, 9, 10}, {11, 12, 13}};
    int ind_pool = 1;
    const size_t scale = 1 << 14;
    const int n_samples = 100;
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
            array_view<int64_t> input = array_view<int64_t>(data[input_key].data<int64_t>(), data[input_key].shape);
            array_view<int64_t> weights = array_view<int64_t>(data[weights_key].data<int64_t>(), data[weights_key].shape);
            array_view<int64_t> expected_out = array_view<int64_t>(data[expected_out_key].data<int64_t>(), data[expected_out_key].shape);
            
            for (size_t mini_batch = 0; mini_batch < input.shape(0); ++mini_batch) {
                std::cout << "\tChecking mini_batch " << mini_batch << std::endl;
                auto view_input = input[mini_batch];
                auto view_weights = weights[mini_batch];
                auto view_expected_out = expected_out[mini_batch];

                if (!check_conv_relu(
                    view_input,
                    view_weights,
                    view_expected_out,
                    scale)) {
                    std::cout << "Conv1 + ReLU output verification failed at layer " << lid << ".\n";
                    return;
                }
            }
        }
    }
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

    for (size_t n = 0; n < N; ++n) {

        auto view_input_n = input[n];
        auto view_expected_n = expected[n];

        for (size_t oc = 0; oc < OC; ++oc) {

            auto view_weights_oc = weights[oc];
            auto view_expected_n_oc = view_expected_n[oc];

            for (size_t i = 0; i < H; ++i) {

                auto view_expected_n_oc_h = view_expected_n_oc[i];

                for (size_t j = 0; j < W; ++j) {

                    int64_t acc = 0;
                    for (size_t c = 0; c < C; ++c) {

                        auto view_weights_oc_c = view_weights_oc[c];
                        auto view_input_n_c = view_input_n[c];

                        for (size_t ki = 0; ki < 3; ++ki) {
                            for (size_t kj = 0; kj < 3; ++kj) {
                                acc += view_input_n_c(i + ki - 1, j + kj - 1) * view_weights_oc_c(ki, kj);
                            }
                        }
                    }
                    acc = acc / scale;
                    if (acc < 0) acc = 0;

                    int64_t actual = view_expected_n_oc_h(j);
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
            return false;
        }
        
    }
    return true;
}

