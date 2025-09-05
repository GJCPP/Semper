#include <chrono>

#include "VGG16.h"
#include "CNN_check.h"
#include "CNN_proof.h"
#include "perm_check.h"
#include "timer.h"
// static std::unordered_map<size_t, size_t> relu_map = {};

VGG16::VGG16(std::string _data_dir, int epoch, int64_t scale, int64_t max_value, uint64_t rho_inv)
    : CNN("VGG16", _data_dir + "/VGG16", epoch, scale, max_value, rho_inv) {

    set_timer("VGG16 load & commit");

    dataset = loadDataset(data_dir);
    filedata = loadEpochData(data_dir, epoch);
    std::vector<std::string> keys;
    std::vector<cnpy::NpyArray*> values;
    for (auto& [key, value] : filedata) {
        keys.push_back(key);
        values.push_back(&value);
    }
    for (auto& [key, value] : dataset) {
        keys.push_back(key);
        values.push_back(&value);
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        std::string key = keys[i];

        cnpy::NpyArray *value = values[i];
        size_t num_vals = value->num_vals;
        data[key] = std::make_unique<Goldilocks2::Element[]>(num_vals);
        data_shape[key] = {};
        data_view[key] = {};
    }
    #pragma omp parallel for
    for (size_t i = 0; i < keys.size(); ++i) {
        std::string key = keys[i];
        cnpy::NpyArray *value = values[i];
        size_t num_vals = value->num_vals;
        auto ptr = data[key].get();
        for (size_t i = 0; i < num_vals; ++i) {
            ptr[i] = Goldilocks2::fromS64(value->data<int64_t>()[i]);
        }
        data_shape[key] = value->shape;
        data_view[key] = array_view<Goldilocks2::Element>(ptr, value->shape);
    }
    data[{}] = nullptr;
    data_shape[{}] = {};
    data_view[{}] = {};
    mle[{}] = {};
    pcs[{}] = {};

    minibatch = data_shape["a_q0"][0];
    img_per_batch = data_shape["a_q0"][1];

    dataset_input = array_view<Goldilocks2::Element>(data["dataset_inputs"].get(), data_shape["dataset_inputs"]);
    dataset_label = array_view<Goldilocks2::Element>(data["dataset_labels"].get(), data_shape["dataset_labels"]);
    input = array_view<Goldilocks2::Element>(data["input"].get(), data_shape["input"]);
    label = array_view<Goldilocks2::Element>(data["label"].get(), data_shape["label"]);
    input_index = array_view<Goldilocks2::Element>(data["index"].get(), data_shape["index"]);

    mle_dataset_input = dataset_input;
    mle_dataset_label = dataset_label;
    mle_input = input;
    mle_label = label;
    mle_index = input_index;

    pcs_dataset_input = commit_lazy_pcs(dataset_input, &pcs_pool);
    pcs_dataset_label = commit_lazy_pcs(dataset_label, &pcs_pool);
    pcs_input = commit_lazy_pcs(input, &pcs_pool);
    pcs_label = commit_lazy_pcs(label, &pcs_pool);
    pcs_index = commit_lazy_pcs(input_index, &pcs_pool);

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
            add_layer(layer_type::conv, lid,
                std::format("conv_{}", lid),
                input_name,
                std::format("z_q{}", lid),
                std::format("W_conv_q{}", lid),
                std::format("grad_{}", input_name),
                std::format("grad_z_q{}", lid),
                std::format("dW_conv_q{}", lid));
            add_layer(layer_type::relu, lid,
                std::format("relu_{}", lid),
                std::format("z_q{}", lid),
                std::format("a_q{}", lid),
                {},
                std::format("grad_z_q{}", lid),
                std::format("grad_a_q{}", lid),
                {});
        }
        add_layer(layer_type::pool, ind_pool,
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
    add_layer(layer_type::flat, ind_pool - 1,
        "flat",
        std::format("pool_q{}", ind_pool - 1),
        "flat_q",
        {},
        std::format("grad_pool_q{}", ind_pool - 1),
        "grad_flat_q",
        {});
    for (int layer = 1; layer <= 3; ++layer) {
        std::string input_name = layer == 1 ? "flat_q" : std::format("a{}_q", layer - 1);
        add_layer(layer_type::full, layer,
            std::format("full_{}", layer),
            input_name,
            std::format("z{}_q", layer),
            std::format("W_fc{}_q", layer),
            std::format("grad_{}", input_name),
            std::format("grad_z{}_q", layer),
            std::format("dW_fc{}_q", layer));
        if (layer < 3) {
            add_layer(layer_type::relu, layer,
                std::format("fc_relu_{}", layer),
                std::format("z{}_q", layer),
                std::format("a{}_q", layer),
                {},
                std::format("grad_z{}_q", layer),
                std::format("grad_a{}_q", layer),
                {});
        }
    }
    add_layer(layer_type::softmax, -1,
        "softmax",
        "z3_q", // input
        "probs_q", // output
        {}, // weight = {}
        "probs_q", // d_input = output
        {}, // d_output = {}
        {}, // d_weight = {}
        "a_q0_label");

    finish_add_layer();
    

    pause_timer("VGG16 load & commit");
}
