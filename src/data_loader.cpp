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
    xt::xarray<int64_t> input = xt::adapt(data["000_input_q"].data<int64_t>(), data["000_input_q"].shape);
    xt::xarray<int64_t> input_padded = xt::pad(input, {{0,0}, {0,0}, {0,0}, {1,1}, {1,1}}, xt::pad_mode::constant, 0);
    xt::xarray<int64_t> weights = xt::adapt(data["001_W_conv_q1"].data<int64_t>(), data["001_W_conv_q1"].shape);
    xt::xarray<int64_t> expected_out = xt::adapt(data["002_z_q1"].data<int64_t>(), data["002_z_q1"].shape);
    auto view_input = xt::view(input_padded, 0);
    auto view_weights = xt::view(weights, 0);
    auto view_expected_out = xt::view(expected_out, 0);

    const size_t scale = 1 << 14;
    if (!verify_conv_relu(
        view_input,
        view_weights,
        view_expected_out,
        scale)) {
        std::cout << "Conv1 + ReLU output verification failed ❌\n";
    }
}

