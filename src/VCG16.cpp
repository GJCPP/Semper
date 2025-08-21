#include <chrono>

#include "VCG16.h"
#include "VCG16_check.h"
#include "VCG16_proof.h"
#include "perm_check.h"

// static std::unordered_map<size_t, size_t> relu_map = {};

VCG16::VCG16(std::string data_dir, int epoch, int64_t scale, int64_t max_value, uint64_t rho_inv)
    : epoch(epoch), scale(scale), max_val(max_value), sqr_val(max_value * scale), rho_inv(rho_inv) {
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

    pcs_dataset_input = ligero_commit_base(dataset_input, rho_inv);
    pcs_dataset_label = ligero_commit_base(dataset_label, rho_inv);
    pcs_input = ligero_commit_base(input, rho_inv);
    pcs_label = ligero_commit_base(label, rho_inv);
    pcs_index = ligero_commit_base(input_index, rho_inv);

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
        std::format("grad_pool_q{}", ind_pool - 1),
        "grad_flat_q",
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
        "z3_q", // input
        "probs_q", // output
        {}, // weight = {}
        "probs_q", // d_input = output
        {}, // d_output = {}
        {}, // d_weight = {}
        "a_q0_label");

}


bool VCG16::check(size_t n_samples) const {
    for (auto& layer : layers) {
        bool pass = true;
        std::cout << "Checking layer " << layer.name << std::endl;
        switch (layer.type) {
            case layer_type::conv:
                // Check range
                std::cout << "Checking range." << std::endl;
                pass &= check_range(layer.input, max_val);
                pass &= check_range(layer.weight, max_val);
                pass &= check_range(layer.output, sqr_val);
                pass &= check_range(layer.d_input, sqr_val);
                pass &= check_range(layer.d_output, max_val);
                pass &= check_range(layer.d_weight, max_val);
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (range)" << std::endl;
                    break;
                }

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
                for (size_t i = 0; i < layer.input.shape(0) && pass; ++i) { // mini-batch
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
                pass &= check_range(layer.input, sqr_val);
                pass &= check_range(layer.output, max_val);
                pass &= check_range(layer.d_input, max_val);
                pass &= check_range(layer.d_output, sqr_val);
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (range)" << std::endl;
                    break;
                }

                //Check forward pass
                std::cout << "Checking layer " << layer.name << " (forward)" << std::endl;
                pass &= check_relu(layer.input, layer.input, layer.output, scale, n_samples);
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (forward)" << std::endl;
                    break;
                }

                //Check backward pass
                std::cout << "Checking layer " << layer.name << " (backward)" << std::endl;
                pass &= check_relu(layer.output, layer.d_output, layer.d_input, scale, n_samples);
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (backward)" << std::endl;
                    break;
                }
                break;

            case layer_type::pool:
                //Check range
                std::cout << "Checking range." << std::endl;
                pass &= check_range(layer.input, max_val);
                pass &= check_range(layer.output, max_val);
                pass &= check_range(layer.d_input, sqr_val);
                pass &= check_range(layer.d_output, sqr_val);
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (range)" << std::endl;
                    break;
                }

                //Check forward pass
                std::cout << "Checking layer " << layer.name << " (forward)" << std::endl;
                for (size_t i = 0; i < layer.input.shape(0) && pass; ++i) {
                    std::cout << "Checking layer " << layer.name << " (forward) for mini-batch " << i << std::endl;
                    pass &= check_pool(layer.input[i], layer.output[i], layer.aux[i], 2, 2, false, n_samples);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (forward)" << std::endl;
                    break;
                }
                
                //Check backward pass
                std::cout << "Checking layer " << layer.name << " (backward)" << std::endl;
                for (size_t i = 0; i < layer.input.shape(0) && pass; ++i) {
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
                pass &= check_range(layer.input, max_val);
                pass &= check_range(layer.output, max_val);
                pass &= check_range(layer.d_input, sqr_val);
                pass &= check_range(layer.d_output, sqr_val);
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (range)" << std::endl;
                    break;
                }

                //Check forward pass
                std::cout << "Checking layer " << layer.name << " (forward)" << std::endl;
                for (size_t i = 0; i < layer.input.shape(0) && pass; ++i) {
                    std::cout << "Checking layer " << layer.name << " (forward) for mini-batch " << i << std::endl;
                    pass &= check_flat(layer.input[i], layer.output[i], n_samples);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (forward)" << std::endl;
                    break;
                }
                
                //Check backward pass
                std::cout << "Checking layer " << layer.name << " (backward)" << std::endl;
                for (size_t i = 0; i < layer.input.shape(0) && pass; ++i) {
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
                pass &= check_range(layer.input, max_val);
                pass &= check_range(layer.output, sqr_val);
                pass &= check_range(layer.d_input, sqr_val);
                pass &= check_range(layer.d_output, max_val);
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (range)" << std::endl;
                    break;
                }

                //Check forward pass
                std::cout << "Checking layer " << layer.name << " (forward)" << std::endl;
                for (size_t i = 0; i < layer.input.shape(0) && pass; ++i) {
                    std::cout << "Checking layer " << layer.name << " (forward) for mini-batch " << i << std::endl;
                    pass &= check_full(layer.input[i], layer.weight[i], layer.output[i], n_samples);
                }

                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (forward)" << std::endl;
                    break;
                }
                
                //Check backward pass, check d_weight
                std::cout << "Checking layer " << layer.name << " (backward, d_weight)" << std::endl;
                for (size_t i = 0; i < layer.input.shape(0) && pass; ++i) {
                    std::cout << "Checking layer " << layer.name << " (backward, d_weight) for mini-batch " << i << std::endl;
                    array_view<Goldilocks2::Element> input_i(layer.input[i]);
                    input_i.swap_dim(0, 1); // Transpose input to [C, N]
                    pass &= check_full(input_i, layer.d_output[i], layer.d_weight[i], n_samples);
                }

                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (backward, d_weight)" << std::endl;
                    break;
                }

                //Check backward pass, check d_input
                std::cout << "Checking layer " << layer.name << " (backward, d_input)" << std::endl;
                for (size_t i = 0; i < layer.input.shape(0) && pass; ++i) {
                    std::cout << "Checking layer " << layer.name << " (backward, d_input) for mini-batch " << i << std::endl;
                    array_view<Goldilocks2::Element> weight_i(layer.weight[i]);
                    weight_i.swap_dim(0, 1); // Transpose weight to [OC, C]
                    pass &= check_full(layer.d_output[i], weight_i, layer.d_input[i], n_samples);
                }
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (backward, d_input)" << std::endl;
                    break;
                }
                break;

            case layer_type::softmax:
                //Check range
                std::cout << "Checking range." << std::endl;
                pass &= check_range(layer.input, sqr_val);
                pass &= check_range(layer.output, max_val);
                pass &= check_range(layer.d_input, max_val);
                if (!pass) {
                    std::cout << "❌ Layer " << layer.name << " failed. (range)" << std::endl;
                    break;
                }

                //Check softmax
                std::cout << "Checking layer " << layer.name << std::endl;
                for (size_t i = 0; i < layer.input.shape(0) && pass; ++i) {
                    std::cout << "Checking layer " << layer.name << " (forward) for mini-batch " << i << std::endl;
                    pass &= check_softmax(layer.input[i], layer.output[i], layer.aux[i], scale);
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

bool VCG16::prove(size_t sec_param) {
    // auto start_prove_input = std::chrono::high_resolution_clock::now();
    // if (!prove_input(sec_param)) {
    //     std::cout << "❌ Input layer failed." << std::endl;
    //     return false;
    // }
    // std::cout << "✅ Input proved in " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_prove_input).count() << " seconds." << std::endl;
    for (auto& layer : layers) {
        std::cout << "Proving layer " << layer.name << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        switch (layer.type) {
            case layer_type::conv:
                if (!prove_conv_layer(layer, rho_inv, sec_param)) {
                    std::cout << "❌ Layer " << layer.name << " failed." << std::endl;
                    return false;
                }
                break;

            case layer_type::full:
                if (!prove_full_layer(layer, rho_inv, sec_param)) {
                    std::cout << "❌ Layer " << layer.name << " failed." << std::endl;
                    return false;
                }
                break;

            case layer_type::relu:
                if (!prove_relu_layer(layer, scale, max_val, sqr_val, rho_inv, sec_param)) {
                    std::cout << "❌ Layer " << layer.name << " failed." << std::endl;
                    return false;
                }
                break;

            case layer_type::pool:
                if (!prove_pool_layer(layer, scale, max_val, rho_inv, sec_param)) {
                    std::cout << "❌ Layer " << layer.name << " failed." << std::endl;
                    return false;
                }
                break;

            case layer_type::softmax:
                if (!prove_softmax(layer, scale, max_val, rho_inv, sec_param)) {
                    std::cout << "❌ Layer " << layer.name << " failed." << std::endl;
                    return false;
                }
                break;

            case layer_type::flat:
                if (!prove_flat_layer(layer, rho_inv, sec_param)) {
                    std::cout << "❌ Layer " << layer.name << " failed." << std::endl;
                    return false;
                }
                break;

            default:
                break;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "✅ Layer " << layer.name << " proved in " << elapsed.count() << " seconds." << std::endl;

    }
    return false;
}

bool VCG16::prove_input(size_t sec_param) {
    // input_data : [batch, img, channel, wide, height]
    size_t batch = input.shape(0),
            img = input.shape(1),
            channel = input.shape(2),
            wide = input.shape(3),
            height = input.shape(4);
    int log_batch = find_ceiling_log2(batch),
        log_img = find_ceiling_log2(img),
        log_channel = find_ceiling_log2(channel),
        log_wide = find_ceiling_log2(wide),
        log_height = find_ceiling_log2(height);
    size_t up_batch = (1ull << log_batch),
            up_img = (1ull << log_img),
            up_channel = (1ull << log_channel),
            up_wide = (1ull << log_wide),
            up_height = (1ull << log_height);
    std::vector<size_t> from, to;
    from.reserve(img * channel * wide * height);
    to.reserve(img * channel * wide * height);
    for (size_t b = 0; b != batch; ++b) {
        size_t ind_b = b * up_img * up_channel * up_wide * up_height;

        for (size_t i = 0; i != img; ++i) {
            size_t ind_i = ind_b + i * up_channel * up_wide * up_height;
            size_t ind_to_i = input_index(b, i)[0].fe * up_channel * up_wide * up_height;

            for (size_t c = 0; c != channel; ++c) {
                size_t ind_c = ind_i + c * up_wide * up_height;
                size_t ind_to_c = ind_to_i + c * up_wide * up_height;

                for (size_t w = 0; w != wide; ++w) {
                    size_t ind_w = ind_c + w * up_height;
                    size_t ind_to_w = ind_to_c + w * up_height;

                    for (size_t h = 0; h != height; ++h) {
                        from.push_back(ind_w + h);
                        to.push_back(ind_to_w + h);
                    }
                }
            }
        }
    }
    // Check that input is subset of dataset_input
    mapProver prover(from, to, false);
    prover.add_mle(&mle_input, &mle_dataset_input);
    mapVerifier verifier;
    verifier.add_pcs(&pcs_input, &pcs_dataset_input);
    if (!verifier.execute_check(prover, rho_inv, sec_param)) {
        std::cout << "❌ Input permutation check failed." << std::endl;
        return false;
    }

    std::vector<size_t> label_from, label_to;
    label_from.reserve(batch * img);
    label_to.reserve(batch * img);
    for (size_t b = 0; b != batch; ++b) {
        for (size_t i = 0; i != img; ++i) {
            label_from.push_back(b * up_img + i);
            label_to.push_back(input_index(b, i)[0].fe);
        }
    }
    mapProver label_prover(label_from, label_to, false);
    label_prover.add_mle(&mle_label, &mle_dataset_label);
    mapVerifier label_verifier;
    label_verifier.add_pcs(&pcs_label, &pcs_dataset_label);
    if (!label_verifier.execute_check(label_prover, rho_inv, sec_param)) {
        std::cout << "❌ Label permutation check failed." << std::endl;
        return false;
    }

    // Check that input is used as the input of a0 layer
    auto cha = random_vec_ext(mle_input.get_num_vars() - log_batch);
    for (size_t b = 0; b != batch; ++b) {
        std::vector<Goldilocks2::Element> prefix(log_batch);
        for (size_t i = 0; i != log_batch; ++i) {
            prefix[i] = ((b >> (log_batch - i - 1)) & 1) ? Goldilocks2::one() : Goldilocks2::zero();
        }
        prefix = combine_challenges(prefix, cha);
        auto pcs_layer0_input = layers[0].get_pcs_input(b);
        if (pcs_layer0_input->open(cha, sec_param) != pcs_input.open(prefix, sec_param)) {
            std::cout << "❌ Input is not used as the input of a0 layer." << std::endl;
            return false;
        }
        auto pcs_output_label = layers.back().get_pcs_aux(b);
        if (pcs_output_label->open(cha, sec_param) != pcs_output_label->open(prefix, sec_param)) {
            std::cout << "❌ Output label is not used as the output of the last layer." << std::endl;
            return false;
        }
    }

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
    info.input = array_view<Goldilocks2::Element>(data[input].get(), data_shape[input]);
    info.output = array_view<Goldilocks2::Element>(data[output].get(), data_shape[output]);
    info.weight = array_view<Goldilocks2::Element>(data[weight].get(), data_shape[weight]);
    info.d_input = array_view<Goldilocks2::Element>(data[d_input].get(), data_shape[d_input]);
    info.d_output = array_view<Goldilocks2::Element>(data[d_output].get(), data_shape[d_output]);
    info.d_weight = array_view<Goldilocks2::Element>(data[d_weight].get(), data_shape[d_weight]);
    info.aux = array_view<Goldilocks2::Element>(data[aux].get(), data_shape[aux]);

    
    auto init_mle = [&](const std::string& key) {
        auto& mle = this->mle[key];
        mle.resize(minibatch);
        for (int i = 0; i < minibatch; ++i) {
            mle[i] = std::make_shared<MultilinearPolynomial>(data_view[key][i]);
        }
    };
    if (mle.find(input) == mle.end()) init_mle(input);
    if (mle.find(output) == mle.end()) init_mle(output);
    if (mle.find(weight) == mle.end()) init_mle(weight);
    if (mle.find(d_input) == mle.end()) init_mle(d_input);
    if (mle.find(d_output) == mle.end()) init_mle(d_output);
    if (mle.find(d_weight) == mle.end()) init_mle(d_weight);
    if (mle.find(aux) == mle.end()) init_mle(aux);

    auto init_pcs = [&](const std::string& key) {
        auto& mle = this->mle[key];
        auto& pcs = this->pcs[key];
        pcs.resize(minibatch);
#ifndef DEBUG
        for (int i = 0; i < minibatch; ++i) {
            pcs[i] = std::make_shared<ligeropcs_base>(mle[i], rho_inv);
        }
#else
        for (int i = 0; i < minibatch; ++i) {
            pcs[i] = std::make_shared<ligeropcs_base>();
        }
#endif
    };
    if (pcs.find(input) == pcs.end()) init_pcs(input);
    if (pcs.find(output) == pcs.end()) init_pcs(output);
    if (pcs.find(weight) == pcs.end()) init_pcs(weight);
    if (pcs.find(d_input) == pcs.end()) init_pcs(d_input);
    if (pcs.find(d_output) == pcs.end()) init_pcs(d_output);
    if (pcs.find(d_weight) == pcs.end()) init_pcs(d_weight);
    if (pcs.find(aux) == pcs.end()) init_pcs(aux);


    info.mle_input = mle[input];
    info.mle_output = mle[output];
    info.mle_weight = mle[weight];
    info.mle_d_input = mle[d_input];
    info.mle_d_output = mle[d_output];
    info.mle_d_weight = mle[d_weight];
    info.mle_aux = mle[aux];
    
    info.pcs_input = pcs[input];
    info.pcs_output = pcs[output];
    info.pcs_weight = pcs[weight];
    info.pcs_d_input = pcs[d_input];
    info.pcs_d_output = pcs[d_output];
    info.pcs_d_weight = pcs[d_weight];
    info.pcs_aux = pcs[aux];

    layers.push_back(info);
}

#ifdef DEBUG
std::shared_ptr<ligeropcs_base> VCG16::layer_info::get_pcs_input(int bat) const {
    auto& pcs = pcs_input[bat];
    if (pcs->empty()) {
        *pcs = ligero_commit_base(*mle_input[bat], 2);
    }
    return pcs;
}
std::shared_ptr<ligeropcs_base> VCG16::layer_info::get_pcs_output(int bat) const {
    auto& pcs = pcs_output[bat];
    if (pcs->empty()) {
        *pcs = ligero_commit_base(*mle_output[bat], 2);
    }
    return pcs;
}

std::shared_ptr<ligeropcs_base> VCG16::layer_info::get_pcs_weight(int bat) const {
    auto& pcs = pcs_weight[bat];
    if (pcs->empty()) {
        *pcs = ligero_commit_base(*mle_weight[bat], 2);
    }
    return pcs;
}

std::shared_ptr<ligeropcs_base> VCG16::layer_info::get_pcs_d_input(int bat) const {
    auto& pcs = pcs_d_input[bat];
    if (pcs->empty()) {
        *pcs = ligero_commit_base(*mle_d_input[bat], 2);
    }
    return pcs;
}

std::shared_ptr<ligeropcs_base> VCG16::layer_info::get_pcs_d_output(int bat) const {
    auto& pcs = pcs_d_output[bat];
    if (pcs->empty()) {
        *pcs = ligero_commit_base(*mle_d_output[bat], 2);
    }
    return pcs;
}

std::shared_ptr<ligeropcs_base> VCG16::layer_info::get_pcs_d_weight(int bat) const {
    auto& pcs = pcs_d_weight[bat];
    if (pcs->empty()) {
        *pcs = ligero_commit_base(*mle_d_weight[bat], 2);
    }
    return pcs;
}


std::shared_ptr<ligeropcs_base> VCG16::layer_info::get_pcs_aux(int bat) const {
    auto& pcs = pcs_aux[bat];
    if (pcs->empty()) {
        *pcs = ligero_commit_base(*mle_aux[bat], 2);
    }
    return pcs;
}

#endif

