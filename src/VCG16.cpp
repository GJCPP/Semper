#include <chrono>

#include "header"

#include "VCG16.h"
#include "VCG16_check.h"
#include "VCG16_prove.h"

VCG16::VCG16(std::string data_dir, int epoch, int64_t scale, int64_t max_value)
    : epoch(epoch), scale(scale), max_val(max_value), sqr_val(max_value * max_val) {
    filedata = loadEpochData(data_dir, epoch);
    std::vector<std::string> keys;
    std::vector<cnpy::NpyArray*> values;
    for (auto& [key, value] : filedata) {
        keys.push_back(key);
        values.push_back(&value);
    }

    minibatch = filedata["a_q0"].shape[0];
    img_per_batch = filedata["a_q0"].shape[1];

    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string& key = keys[i];
        mle[key] = {};
        pcs[key] = {};
        data[key] = nullptr;
        data_shape[key] = {};
    }

    #pragma omp parallel for
    for (size_t i = 0; i < keys.size(); ++i) {
        std::string key = keys[i];

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        cnpy::NpyArray *value = values[i];
        size_t num_vals = value->num_vals;
        data[key] = std::make_unique<Goldilocks2::Element[]>(num_vals);
        auto ptr = data[key].get();
        for (size_t i = 0; i < num_vals; ++i) {
            ptr[i] = Goldilocks2::fromS64(value->data<int64_t>()[i]);
        }
        data_shape[key] = value->shape;

        mle[key] = {};
        pcs[key] = {};
    }
    std::cout << "Done processing keys." << std::endl;
    data[{}] = nullptr;
    data_shape[{}] = {};
    mle[{}] = {};
    pcs[{}] = {};

    input_data = array_view<Goldilocks2::Element>(data["a_q0"].get(), data_shape["a_q0"]);
    input_label = array_view<Goldilocks2::Element>(data["a_q0_label"].get(), data_shape["a_q0_label"]);

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
            add_layer(layer_type::conv, minibatch,
                std::format("conv_{}", lid),
                input_name,
                std::format("z_q{}", lid),
                std::format("W_conv_q{}", lid),
                std::format("grad_{}", input_name),
                std::format("grad_z_q{}", lid),
                std::format("dW_conv_q{}", lid));
            add_layer(layer_type::relu, minibatch,
                std::format("relu_{}", lid),
                std::format("z_q{}", lid),
                std::format("a_q{}", lid),
                {},
                std::format("grad_z_q{}", lid),
                std::format("grad_a_q{}", lid),
                {});
        }
        add_layer(layer_type::pool, minibatch,
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
    add_layer(layer_type::flat, minibatch,
        "flat",
        std::format("pool_q{}", ind_pool - 1),
        "flat_q",
        {},
        std::format("grad_pool_q{}", ind_pool - 1),
        "grad_flat_q",
        {});
    for (int layer = 1; layer <= 3; ++layer) {
        std::string input_name = layer == 1 ? "flat_q" : std::format("a{}_q", layer - 1);
        add_layer(layer_type::full, minibatch,
            std::format("full_{}", layer),
            input_name,
            std::format("z{}_q", layer),
            std::format("W_fc{}_q", layer),
            std::format("grad_{}", input_name),
            std::format("grad_z{}_q", layer),
            std::format("dW_fc{}_q", layer));
        if (layer < 3) {
            add_layer(layer_type::relu, minibatch,
                std::format("fc_relu_{}", layer),
                std::format("z{}_q", layer),
                std::format("a{}_q", layer),
                {},
                std::format("grad_z{}_q", layer),
                std::format("grad_a{}_q", layer),
                {});
        }
    }
    add_layer(layer_type::softmax, minibatch,
        "softmax",
        "z3_q", // input
        "probs_q", // output
        {}, // weight = {}
        "probs_q", // d_input = output
        {}, // d_output = {}
        {}, // d_weight = {}
        "a_q0_label");


    init_e_pow();
}

void VCG16::init_e_pow() {
    e_pow_inv.resize(max_val);
    for (int64_t i = 0; i < max_val; ++i) {
        e_pow_inv[i] = Goldilocks2::fromS64(std::round(std::exp(static_cast<double>(-i) / scale) * scale));
        if (e_pow_inv[i] == Goldilocks2::zero()) {
            // std::cout << "e_pow_inv[" << i << "] is 0, scale=" << scale << std::endl;
            return;
        }
    }
}

bool VCG16::check(size_t n_samples) const {
    for (auto& layer : layers) {
        bool pass = true;
        std::cout << "Checking layer " << layer.name << std::endl;
        switch (layer.type) {
            case layer_type::conv:
                // Check range
                std::cout << "Checking range." << std::endl;
                for (int i = 0; i < minibatch; ++i) {
                    pass &= check_range(layer.input[i], max_val);
                    pass &= check_range(layer.weight[i], max_val);
                    pass &= check_range(layer.output[i], sqr_val);
                    pass &= check_range(layer.d_input[i], sqr_val);
                    pass &= check_range(layer.d_output[i], max_val);
                    pass &= check_range(layer.d_weight[i], max_val);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (range)" << std::endl;
                    break;
                }

                // Check forward pass
                for (size_t i = 0; i < minibatch && pass; ++i) { // mini-batch
                    std::cout << "Checking layer " << layer.name << " (forward) for mini-batch " << i << std::endl;
                    pass &= check_conv(layer.input[i], layer.weight[i], layer.output[i], 1, n_samples);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (forward)" << std::endl;
                    break;
                }

                // Check backward pass, check d_weight
                for (size_t i = 0; i < minibatch && pass; ++i) { // mini-batch
                    std::cout << "Checking layer " << layer.name << " (backward, d_weight) for mini-batch " << i << std::endl;
                    array_view<Goldilocks2::Element> input_i(layer.input[i]);
                    array_view<Goldilocks2::Element> d_output_i(layer.d_output[i]);
                    array_view<Goldilocks2::Element> d_weight_i(layer.d_weight[i]);
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
                for (size_t i = 0; i < minibatch && pass; ++i) { // mini-batch
                    std::cout << "Checking layer " << layer.name << " (backward, d_input) for mini-batch " << i << std::endl;
                    array_view<Goldilocks2::Element> d_input_i(layer.d_input[i]);
                    array_view<Goldilocks2::Element> d_output_i(layer.d_output[i]);
                    array_view<Goldilocks2::Element> weight_i(layer.weight[i]);
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
                //Check range
                std::cout << "Checking range." << std::endl;
                for (int i = 0; i < minibatch; ++i) {
                    pass &= check_range(layer.input[i], sqr_val);
                    pass &= check_range(layer.output[i], max_val);
                    pass &= check_range(layer.d_input[i], max_val);
                    pass &= check_range(layer.d_output[i], sqr_val);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (range)" << std::endl;
                    break;
                }

                //Check forward pass
                std::cout << "Checking layer " << layer.name << " (forward)" << std::endl;
                for (int i = 0; i < minibatch && pass; ++i) {
                    pass &= check_relu(layer.input[i], layer.input[i], layer.output[i], scale, n_samples);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (forward)" << std::endl;
                    break;
                }

                //Check backward pass
                std::cout << "Checking layer " << layer.name << " (backward)" << std::endl;
                for (int i = 0; i < minibatch && pass; ++i) {
                    pass &= check_relu(layer.input[i], layer.d_output[i], layer.d_input[i], scale, n_samples);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (backward)" << std::endl;
                    break;
                }
                break;

            case layer_type::pool:
                //Check range
                std::cout << "Checking range." << std::endl;
                for (int i = 0; i < minibatch; ++i) {
                    pass &= check_range(layer.input[i], max_val);
                    pass &= check_range(layer.output[i], max_val);
                    pass &= check_range(layer.d_input[i], sqr_val);
                    pass &= check_range(layer.d_output[i], sqr_val);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (range)" << std::endl;
                    break;
                }

                //Check forward pass
                std::cout << "Checking layer " << layer.name << " (forward)" << std::endl;
                for (int i = 0; i < minibatch && pass; ++i) {
                    std::cout << "Checking layer " << layer.name << " (forward) for mini-batch " << i << std::endl;
                    pass &= check_pool(layer.input[i], layer.output[i], layer.aux[i], 2, 2, false, n_samples);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (forward)" << std::endl;
                    break;
                }

                //Check backward pass
                
                //Check backward pass
                std::cout << "Checking layer " << layer.name << " (backward)" << std::endl;
                for (int i = 0; i < minibatch && pass; ++i) {
                    std::cout << "Checking layer " << layer.name << " (backward) for mini-batch " << i << std::endl;
                    pass &= check_pool(layer.d_input[i], layer.d_output[i], layer.aux[i], 2, 2, true, n_samples);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (backward)" << std::endl;
                    break;
                }
                break;

            case layer_type::flat:
                //Check range
                std::cout << "Checking range." << std::endl;
                for (int i = 0; i < minibatch; ++i) {
                    pass &= check_range(layer.input[i], max_val);
                    pass &= check_range(layer.output[i], max_val);
                    pass &= check_range(layer.d_input[i], sqr_val);
                    pass &= check_range(layer.d_output[i], sqr_val);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (range)" << std::endl;
                    break;
                }

                //Check forward pass
                std::cout << "Checking layer " << layer.name << " (forward)" << std::endl;
                for (int i = 0; i < minibatch && pass; ++i) {
                    std::cout << "Checking layer " << layer.name << " (forward) for mini-batch " << i << std::endl;
                    pass &= check_flat(layer.input[i], layer.output[i], n_samples);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (forward)" << std::endl;
                    break;
                }
                
                //Check backward pass
                std::cout << "Checking layer " << layer.name << " (backward)" << std::endl;
                for (int i = 0; i < minibatch && pass; ++i) {
                    std::cout << "Checking layer " << layer.name << " (backward) for mini-batch " << i << std::endl;
                    pass &= check_flat(layer.d_input[i], layer.d_output[i], n_samples);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (backward)" << std::endl;
                    break;
                }
                break;

            case layer_type::full:
                //Check range
                std::cout << "Checking range." << std::endl;
                for (int i = 0; i < minibatch; ++i) {
                    pass &= check_range(layer.input[i], max_val);
                    pass &= check_range(layer.output[i], sqr_val);
                    pass &= check_range(layer.d_input[i], sqr_val);
                    pass &= check_range(layer.d_output[i], max_val);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (range)" << std::endl;
                    break;
                }

                //Check forward pass
                std::cout << "Checking layer " << layer.name << " (forward)" << std::endl;
                for (int i = 0; i < minibatch && pass; ++i) {
                    std::cout << "Checking layer " << layer.name << " (forward) for mini-batch " << i << std::endl;
                    pass &= check_full(layer.input[i], layer.weight[i], layer.output[i], n_samples);
                }
                
                //Check backward pass, check d_weight
                std::cout << "Checking layer " << layer.name << " (backward, d_weight)" << std::endl;
                for (int i = 0; i < minibatch && pass; ++i) {
                    std::cout << "Checking layer " << layer.name << " (backward, d_weight) for mini-batch " << i << std::endl;
                    array_view<Goldilocks2::Element> input_i(layer.input[i]);
                    input_i.swap_dim(0, 1); // Transpose input to [C, N]
                    pass &= check_full(input_i, layer.d_output[i], layer.d_weight[i], n_samples);
                }

                //Check backward pass, check d_input
                std::cout << "Checking layer " << layer.name << " (backward, d_input)" << std::endl;
                for (int i = 0; i < minibatch && pass; ++i) {
                    std::cout << "Checking layer " << layer.name << " (backward, d_input) for mini-batch " << i << std::endl;
                    array_view<Goldilocks2::Element> weight_i(layer.weight[i]);
                    weight_i.swap_dim(0, 1); // Transpose weight to [OC, C]
                    pass &= check_full(layer.d_output[i], weight_i, layer.d_input[i], n_samples);
                }
                break;

            case layer_type::softmax:
                //Check range
                std::cout << "Checking range." << std::endl;
                for (int i = 0; i < minibatch; ++i) {
                    pass &= check_range(layer.input[i], sqr_val);
                    pass &= check_range(layer.output[i], max_val);
                    pass &= check_range(layer.d_input[i], max_val);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (range)" << std::endl;
                    break;
                }

                //Check softmax
                std::cout << "Checking layer " << layer.name << std::endl;
                for (int i = 0; i < minibatch && pass; ++i) {
                    std::cout << "Checking layer " << layer.name << " (forward) for mini-batch " << i << std::endl;
                    pass &= check_softmax(layer.input[i], layer.output[i], layer.aux[i], e_pow_inv, scale);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (forward)" << std::endl;
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

bool VCG16::prove(int sec_param) {
    bool ret = true;
    for (auto layer : layers) {
        init_layer_mle_pcs(layer);
        bool pass = true;
        std::cout << "Proving layer " << layer.name << std::endl;
        switch (layer.type) {
        case layer_type::conv:
            std::cout << "Proving layer " << layer.name << " (forward)" << std::endl;
            for (int i = 0; i < minibatch; ++i) {
                for (int j = 0; j < img_per_batch; ++j) {
                    #ifndef DEBUG

                    ret &= prove_conv(layer.input[i][j], layer.weight[i][j], layer.output[i][j], layer.pcs_input->at(i)[j].get(), layer.pcs_weight->at(i)[j].get(), layer.pcs_output->at(i)[j].get(), 1, sec_param);

                    #else

                    ret &= prove_conv(
                        layer.input[i][j],
                        layer.weight[i],
                        layer.output[i][j],
                        layer.get_pcs_input(i, j),
                        layer.get_pcs_weight(i, 0),
                        layer.get_pcs_output(i, j),
                        1,
                        sec_param);
                        
                    #endif
                }
                if (!ret) {
                    std::cout << "❌ Layer " << layer.name << " failed. (forward)" << std::endl;
                    break;
                }
            }
            if (!ret) break;
            break;
        }
        if (!ret) break;
    }
    if (ret) {
        std::cout << "✅ All layers proved successfully." << std::endl;
    } else {
        std::cout << "❌ Some layers failed to prove." << std::endl;
    }
    return ret;
}

void VCG16::add_layer(layer_type type, int batches,
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

    std::vector<array_view<Goldilocks2::Element>> *arrs[] = {
        &info.input,
        &info.output,
        &info.weight,
        &info.d_input,
        &info.d_output,
        &info.d_weight,
        &info.aux
    };
    std::string *store_keys[7] = {&info.key_input, &info.key_output, &info.key_weight, &info.key_d_input, &info.key_d_output, &info.key_d_weight, &info.key_aux};
    bool val[7];
    const std::string *keys[7] = {&input, &output, &weight, &d_input, &d_output, &d_weight, &aux};
    for (int i = 0; i < 7; ++i) {
        val[i] = !keys[i]->empty();
        *store_keys[i] = *keys[i];
        if (val[i]) {
            auto arr = array_view<Goldilocks2::Element>(data[*keys[i]].get(), data_shape[*keys[i]]);
            for (int j = 0; j < batches; ++j) {
                arrs[i]->push_back(arr[j]);
            }
        }
    }
    
    std::vector<std::vector<std::shared_ptr<MultilinearPolynomial>>> **mles[] = {
        &info.mle_input,
        &info.mle_output,
        &info.mle_weight,
        &info.mle_d_input,
        &info.mle_d_output,
        &info.mle_d_weight
    };
    std::vector<std::vector<std::shared_ptr<ligeropcs_base>>> **pcses[] = {
        &info.pcs_input,
        &info.pcs_output,
        &info.pcs_weight,
        &info.pcs_d_input,
        &info.pcs_d_output,
        &info.pcs_d_weight
    };
    for (int i = 0; i < 6; ++i) {
        if (val[i]) {
            *mles[i] = &mle[*keys[i]];
            *pcses[i] = &pcs[*keys[i]];
        }
    }

    layers.push_back(info);
}

void VCG16::init_layer_mle_pcs(layer_info& info) {
    if (!info.key_input.empty()) init_mle_pcs_input(info.key_input);
    if (!info.key_weight.empty()) init_mle_pcs_weight(info.key_weight);
    if (!info.key_output.empty()) init_mle_pcs_input(info.key_output);
    if (!info.key_d_input.empty()) init_mle_pcs_input(info.key_d_input);
    if (!info.key_d_output.empty()) init_mle_pcs_input(info.key_d_output);
    if (!info.key_d_weight.empty()) init_mle_pcs_weight(info.key_d_weight);
}

void VCG16::init_mle_pcs_input(const std::string& key) {
    array_view<Goldilocks2::Element> arr(data[key].get(), data_shape[key]);
    assert(arr.shape(0) == minibatch);
    assert(arr.shape(1) == img_per_batch);
    auto& _mle = mle[key];
    auto& _pcs = pcs[key];
    if (!_mle.empty()) return;
    for (int i = 0; i < minibatch; ++i) {
        _mle.push_back({});
        _pcs.push_back({});
        for (int j = 0; j < img_per_batch; ++j) {
            _mle.back().push_back(std::make_shared<MultilinearPolynomial>(arr[i][j]));
            _pcs.back().push_back(std::make_shared<ligeropcs_base>(ligero_commit_base(*_mle.back().back(), 2)));
        }
    }
}

void VCG16::init_mle_pcs_weight(const std::string& key) {
    array_view<Goldilocks2::Element> arr(data[key].get(), data_shape[key]);
    assert(arr.shape(0) == minibatch);
    auto& _mle = mle[key];
    auto& _pcs = pcs[key];
    if (!_mle.empty()) return;
    for (int i = 0; i < minibatch; ++i) {
        _mle.push_back({});
        _pcs.push_back({});
        _mle.back().push_back(std::make_shared<MultilinearPolynomial>(arr[i]));
        _pcs.back().push_back(std::make_shared<ligeropcs_base>(ligero_commit_base(*_mle.back().back(), 2)));
    }
}

ligeropcs_base *VCG16::layer_info::get_pcs_input(size_t batch, size_t img) {
    if (pcs_input->empty()) {
        for (int i = 0; i < mle_input->size(); ++i) {
            pcs_input->push_back({});
            for (int j = 0; j < mle_input->at(i).size(); ++j) {
                pcs_input->back().push_back(std::make_shared<ligeropcs_base>(ligero_commit_base(*mle_input->at(i)[j], 2)));
            }
        }
    }
    return pcs_input->at(batch)[img].get();
}

ligeropcs_base *VCG16::layer_info::get_pcs_output(size_t batch, size_t img) {
    if (pcs_output->empty()) {
        for (int i = 0; i < mle_output->size(); ++i) {
            pcs_output->push_back({});
            for (int j = 0; j < mle_output->at(i).size(); ++j) {
                pcs_output->back().push_back(std::make_shared<ligeropcs_base>(ligero_commit_base(*mle_output->at(i)[j], 2)));
            }
        }
    }
    return pcs_output->at(batch)[img].get();
}

ligeropcs_base *VCG16::layer_info::get_pcs_weight(size_t batch, size_t img) {
    if (pcs_weight->empty()) {
        for (int i = 0; i < mle_weight->size(); ++i) {
            pcs_weight->push_back({});
            pcs_weight->back().push_back(std::make_shared<ligeropcs_base>(ligero_commit_base(*mle_weight->at(i)[0], 2)));
        }
    }
    return pcs_weight->at(batch)[0].get();
}

ligeropcs_base *VCG16::layer_info::get_pcs_d_weight(size_t batch, size_t img) {
    if (pcs_d_weight->empty()) {
        for (int i = 0; i < mle_d_weight->size(); ++i) {
            pcs_d_weight->push_back({});
            pcs_d_weight->back().push_back(std::make_shared<ligeropcs_base>(ligero_commit_base(*mle_d_weight->at(i)[0], 2)));
        }
    }
    return pcs_d_weight->at(batch)[0].get();
}

ligeropcs_base *VCG16::layer_info::get_pcs_d_input(size_t batch, size_t img) {
    if (pcs_d_input->empty()) {
        for (int i = 0; i < mle_d_input->size(); ++i) {
            pcs_d_input->push_back({});
            for (int j = 0; j < mle_d_input->at(i).size(); ++j) {
                pcs_d_input->back().push_back(std::make_shared<ligeropcs_base>(ligero_commit_base(*mle_d_input->at(i)[j], 2)));
            }
        }
    }
    return pcs_d_input->at(batch)[img].get();
}

ligeropcs_base *VCG16::layer_info::get_pcs_d_output(size_t batch, size_t img) {
    if (pcs_d_output->empty()) {
        for (int i = 0; i < mle_d_output->size(); ++i) {
            pcs_d_output->push_back({});
            for (int j = 0; j < mle_d_output->at(i).size(); ++j) {
                pcs_d_output->back().push_back(std::make_shared<ligeropcs_base>(ligero_commit_base(*mle_d_output->at(i)[j], 2)));
            }
        }
    }
    return pcs_d_output->at(batch)[img].get();
}

