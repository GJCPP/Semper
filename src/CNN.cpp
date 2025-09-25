#include <chrono>

#include "CNN.h"
#include "CNN_check.h"
#include "CNN_proof.h"
#include "perm_check.h"
#include "timer.h"

#include "counter.h"


CNN::CNN(std::string _model_name, std::string _data_dir, int epoch, int64_t scale, int64_t max_value, uint64_t rho_inv, uint64_t sec_param)
    : model_name(_model_name), data_dir(_data_dir), epoch(epoch), scale(scale), max_val(max_value), sqr_val(max_value * scale), rho_inv(rho_inv), sec_param(sec_param), pcs_pool(sec_param), lazy_map_prover(true), lazy_map_verifier(true) {
    ;
}


bool CNN::check(size_t n_samples) const {
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

void CNN::pre_prove(size_t sec_param) {
    // set_timer(std::format("prove {} total", model_name));
    // std::cout << "model_name = " << model_name << std::endl;
    // std::cout << "Checking input..." << std::endl;
    // set_timer("check input");

    // if (!prove_input(sec_param)) {
    //     std::cout << "❌ Input layer failed." << std::endl;
    //     return false;
    // }
    // pause_timer("check input");
    for (auto& layer : layers) {
        // print_all_proof_size(Counter::MB);
        // if (layer.type != layer_type::conv) {
        //     std::cout << "Skipping layer " << layer.name << " (not conv)" << std::endl;
        //     continue;
        // }
        
        // if (layer.type != layer_type::softmax) {
        //     std::cout << "=================Skipping layer " << layer.name << " (not softmax)" << std::endl;
        //     continue;
        // }

        std::cout << "Pre-proving layer " << layer.name << "..." << std::endl;
         
        switch (layer.type) {
            case layer_type::conv:
                set_timer("preprove conv");
                layer.wit = pre_prove_conv_layer(layer, conv_wit(data_dir, epoch, layer.id), &pcs_pool);
                pause_timer("preprove conv");
                break;

            // case layer_type::full:
            //     set_timer("prove full");
            //     if (!prove_full_layer(layer, rho_inv, sec_param)) {
            //         std::cout << "❌ Layer " << layer.name << " failed." << std::endl;
            //         return false;
            //     }
            //     pause_timer("prove full");
            //     break;

            case layer_type::relu:
                set_timer("preprove relu");
                layer.wit = pre_prove_relu_layer(layer, scale, max_val, sqr_val, rho_inv, &lazy_logup_prover, &lazy_logup_verifier, &pcs_pool);
                pause_timer("preprove relu");
                break;

            case layer_type::pool:
                set_timer("preprove pool");
                layer.wit = pre_prove_pool_layer(layer, scale, max_val, rho_inv, &lazy_logup_prover, &lazy_logup_verifier, &pcs_pool);
                pause_timer("preprove pool");
                break;

            case layer_type::softmax:
                set_timer("preprove softmax");
                layer.wit = pre_prove_softmax(layer, scale, max_val, rho_inv, &lazy_logup_prover, &lazy_logup_verifier, &pcs_pool);
                pause_timer("preprove softmax");
                break;

            // case layer_type::flat:
            //     set_timer("prove flat");
            //     if (!prove_flat_layer(layer, rho_inv, sec_param)) {
            //         std::cout << "❌ Layer " << layer.name << " failed." << std::endl;
            //         return false;
            //     }
            //     pause_timer("prove flat");
            //     break;

            default:
                break;
        }
    }

    // start_proof("final open");
    // prove_final_open(random_ext());
    // end_proof("final open");

    // pause_timer(std::format("prove {} total", model_name));
    // print_all_timers();
    // clear_all_timers();
    // return false;
    std::cout << "Finishing preprove..." << std::endl;
    finish_pre_prove();
}

bool CNN::prove(size_t sec_param) {

    std::cout << "model_name = " << model_name << std::endl;
    std::cout << "batch_sz = " << img_per_batch << std::endl;
    std::cout << "batch_num = " << minibatch << std::endl;

    std::cout << "Pre-proving..." << std::endl;
    set_timer("pre_prove");
    pre_prove(sec_param);
    pause_timer("pre_prove");

    set_timer(std::format("prove {} total", model_name));
    
    std::cout << "===================Warning: skip proving input." << std::endl;
    // std::cout << "Checking input..." << std::endl;
    // set_timer("check input");
    // if (!prove_input(sec_param)) {
    //     std::cout << "❌ Input layer failed." << std::endl;
    //     return false;
    // }
    pause_timer("check input");
    for (auto& layer : layers) {
        // print_all_proof_size(Counter::MB);
        // if (layer.type != layer_type::softmax) {
        //     std::cout << "=================Skipping layer " << layer.name << " (not softmax)" << std::endl;
        //     continue;
        // }
        std::cout << "Proving layer " << layer.name << "..." << std::endl;
        switch (layer.type) {
            case layer_type::conv:
                set_timer("prove conv");
                if (!prove_conv_layer(layer, conv_wit(data_dir, epoch, layer.id), rho_inv, sec_param, &lazy_map_prover, &lazy_map_verifier)) {
                    std::cout << "❌ Layer " << layer.name << " failed." << std::endl;
                    return false;
                }
                pause_timer("prove conv");
                break;

            case layer_type::full:
                set_timer("prove full");
                if (!prove_full_layer(layer, rho_inv, sec_param)) {
                    std::cout << "❌ Layer " << layer.name << " failed." << std::endl;
                    return false;
                }
                pause_timer("prove full");
                break;

            case layer_type::relu:
                set_timer("prove relu");
                if (!prove_relu_layer(layer, scale, max_val, sqr_val, rho_inv, sec_param, layer.wit, &lazy_logup_prover, &lazy_logup_verifier)) {
                    std::cout << "❌ Layer " << layer.name << " failed." << std::endl;
                    return false;
                }
                pause_timer("prove relu");
                break;

            case layer_type::pool:
                set_timer("prove pool");
                if (!prove_pool_layer(layer, scale, max_val, rho_inv, sec_param, &lazy_logup_prover, &lazy_logup_verifier, layer.wit)) {
                    std::cout << "❌ Layer " << layer.name << " failed." << std::endl;
                    return false;
                }
                pause_timer("prove pool");
                break;

            case layer_type::softmax:
                set_timer("prove softmax");
                if (!prove_softmax(layer, scale, max_val, rho_inv, sec_param, layer.wit, &lazy_logup_prover, &lazy_logup_verifier)) {
                    std::cout << "❌ Layer " << layer.name << " failed." << std::endl;
                    return false;
                }
                pause_timer("prove softmax");
                break;

            case layer_type::flat:
                set_timer("prove flat");
                if (!prove_flat_layer(layer, rho_inv, sec_param)) {
                    std::cout << "❌ Layer " << layer.name << " failed." << std::endl;
                    return false;
                }
                pause_timer("prove flat");
                break;

            default:
                break;
        }
    }

    std::cout << "Proving final logup..." << std::endl;
    set_timer("final logup");
    start_proof("final logup");
    if (!lazy_logup_verifier.prove_all(lazy_logup_prover, rho_inv, sec_param)) {
        std::cout << "❌ Final lazy logup proof failed." << std::endl;
        return false;
    }
    pause_timer("final logup");
    end_proof("final logup");

    std::cout << "Proving final map..." << std::endl;
    set_timer("final map");
    start_proof("final map");
    if (!lazy_map_verifier.prove_all(lazy_map_prover, rho_inv, sec_param)) {
        std::cout << "❌ Final lazy map proof failed." << std::endl;
        return false;
    }
    pause_timer("final map");
    end_proof("final map");

    // std::cout << "===================Warning: skip final opening." << std::endl;
    std::cout << "Proving final opening..." << std::endl;
    start_proof("final open");
    if (!prove_final_open(random_ext())) {
        std::cout << "❌ Final opening proof failed." << std::endl;
        return false;
    }
    end_proof("final open");

    pause_timer(std::format("prove {} total", model_name));
    print_all_timers();
    clear_all_timers();
    return false;
}

bool CNN::prove_input(size_t sec_param) {
    startCounter counter("input_proof");
    // input_data : [batch, img, channel, wide, height]
    size_t batch = input.shape(0),
            img = input.shape(1),
            channel = input.shape(2),
            wide = input.shape(3),
            height = input.shape(4),
            num_class = label.shape(2);
    int log_batch = find_ceiling_log2(batch),
        log_img = find_ceiling_log2(img),
        log_channel = find_ceiling_log2(channel),
        log_wide = find_ceiling_log2(wide),
        log_height = find_ceiling_log2(height),
        log_num_class = find_ceiling_log2(num_class);
    size_t up_img = (1ull << log_img),
            up_channel = (1ull << log_channel),
            up_wide = (1ull << log_wide),
            up_height = (1ull << log_height),
            up_num_class = (1ull << log_num_class);
    std::vector<size_t> from, to;
    from.resize(batch * img * channel * wide * height);
    to.resize(batch * img * channel * wide * height);

    #pragma omp parallel for
    for (size_t b = 0; b != batch; ++b) {
        size_t ind_b = b * up_img * up_channel * up_wide * up_height;
        size_t out_ind_b = b * img * channel * wide * height;

        for (size_t i = 0; i != img; ++i) {
            size_t ind_i = ind_b + i * up_channel * up_wide * up_height;
            size_t ind_to_i = input_index(b, i)[0].fe * up_channel * up_wide * up_height;
            size_t out_ind_i = out_ind_b + i * channel * wide * height;

            for (size_t c = 0; c != channel; ++c) {
                size_t ind_c = ind_i + c * up_wide * up_height;
                size_t ind_to_c = ind_to_i + c * up_wide * up_height;
                size_t out_ind_c = out_ind_i + c * wide * height;

                for (size_t w = 0; w != wide; ++w) {
                    size_t ind_w = ind_c + w * up_height;
                    size_t ind_to_w = ind_to_c + w * up_height;
                    size_t out_ind_w = out_ind_c + w * height;

                    for (size_t h = 0; h != height; ++h) {
                        from[out_ind_w + h] = ind_w + h;
                        to[out_ind_w + h] = ind_to_w + h;
                    }
                }
            }
        }
    }

    // Check that input is subset of dataset_input
    lazy_map_prover.add(from, to, mle_input, mle_dataset_input);
    lazy_map_verifier.add(from, to, std::make_shared<lazy_pcs>(pcs_input), std::make_shared<lazy_pcs>(pcs_dataset_input));


    // Check that label is subset of dataset_label
    std::vector<size_t> label_from, label_to;
    label_from.reserve(batch * img * num_class);
    label_to.reserve(batch * img * num_class);
    for (size_t b = 0; b != batch; ++b) {
        for (size_t i = 0; i != img; ++i) {
            for (size_t j = 0; j != num_class; ++j) {
                label_from.push_back(b * up_img * up_num_class + i * up_num_class + j);
                label_to.push_back(input_index(b, i)[0].fe * up_num_class + j);
            }
        }
    }
    lazy_map_prover.add(label_from, label_to, mle_label, mle_dataset_label);
    lazy_map_verifier.add(label_from, label_to, std::make_shared<lazy_pcs>(pcs_label), std::make_shared<lazy_pcs>(pcs_dataset_label));


    // Check that input is used as the input of a0 layer
    start_proof("check_input_layer");
    auto cha = random_vec_ext(mle_input.get_num_vars() - log_batch);
    auto label_cha = random_vec_ext(mle_label.get_num_vars() - log_batch);
    bool succ = true;
    #pragma omp parallel for
    for (size_t b = 0; b != batch; ++b) {
        std::vector<Goldilocks2::Element> pre(log_batch);
        for (int i = 0; i != log_batch; ++i) {
            pre[i] = ((b >> (log_batch - i - 1)) & 1) ? Goldilocks2::one() : Goldilocks2::zero();
        }
        auto ext_cha = combine_challenges(pre, cha);
        auto ext_label_cha = combine_challenges(pre, label_cha);
        auto pcs_layer0_input = layers[0].get_pcs_input(b);
        if (pcs_layer0_input.open(cha, sec_param) != pcs_input.open(ext_cha, sec_param)) {
            std::cout << "❌ Input is not used as the input of a0 layer." << std::endl;
            succ = false;
        }
        auto pcs_output_label = layers.back().get_pcs_aux(b);
        if (pcs_output_label.open(label_cha, sec_param) != pcs_label.open(ext_label_cha, sec_param)) {
            std::cout << "❌ Output label is not used as the output of the last layer." << std::endl;
            succ = false;
        }
    }
    end_proof("check_input_layer");

    return succ;
}

void CNN::add_layer(layer_type type, int id,
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
    info.id = id;
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
        #pragma omp parallel for
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
        for (int i = 0; i < minibatch; ++i) {
            pcs[i] = commit_lazy_pcs(*mle[i], &pcs_pool);
        }
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

lazy_pcs CNN::layer_info::get_pcs_input(int bat) const {
    auto& pcs = pcs_input[bat];
    return pcs;
}
lazy_pcs CNN::layer_info::get_pcs_output(int bat) const {
    auto& pcs = pcs_output[bat];
    return pcs;
}

lazy_pcs CNN::layer_info::get_pcs_weight(int bat) const {
    auto& pcs = pcs_weight[bat];
    return pcs;
}

lazy_pcs CNN::layer_info::get_pcs_d_input(int bat) const {
    auto& pcs = pcs_d_input[bat];
    return pcs;
}

lazy_pcs CNN::layer_info::get_pcs_d_output(int bat) const {
    auto& pcs = pcs_d_output[bat];
    return pcs;
}

lazy_pcs CNN::layer_info::get_pcs_d_weight(int bat) const {
    auto& pcs = pcs_d_weight[bat];
    return pcs;
}


lazy_pcs CNN::layer_info::get_pcs_aux(int bat) const {
    auto& pcs = pcs_aux[bat];
    return pcs;
}

