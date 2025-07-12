#pragma once

#include <format>
#include <random>

#include "ligero.h"
#include "goldilocks_quadratic_ext.h"
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
        array_view<Goldilocks2::Element> input, d_input;
        array_view<Goldilocks2::Element> weight, d_weight;
        array_view<Goldilocks2::Element> output, d_output; // Must be the input of the next layer
        array_view<Goldilocks2::Element> aux; // auxiliary data for checking

        std::shared_ptr<MultilinearPolynomial> mle_input, mle_output,
                                            mle_weight, mle_d_weight,
                                            mle_d_input, mle_d_output;
                                            
        std::shared_ptr<ligeropcs_base> pcs_input, pcs_output,
                                        pcs_weight, pcs_d_weight,
                                        pcs_d_input, pcs_d_output;
    };
    VCG16(std::string data_dir, int epoch, int64_t scale, int64_t max_value);

    bool check(size_t n_samples = 0) const;

    void add_layer(layer_type type,
                    const std::string& name,
                    const std::string& input,
                    const std::string& output,
                    const std::string& weight,
                    const std::string& d_input,
                    const std::string& d_weight,
                    const std::string& d_output,
                    const std::string& aux = "");

protected:
    void init_e_pow();

    int64_t scale, max_val, sqr_val;
    int epoch, minibatch, img_per_batch;
    
    cnpy::npz_t filedata; // responsible for releasing data
    
    std::map<std::string, std::unique_ptr<Goldilocks2::Element[]>> data;
    std::map<std::string, std::vector<size_t>> data_shape;
    std::map<std::string, std::shared_ptr<MultilinearPolynomial>> mle;
    std::map<std::string, std::shared_ptr<ligeropcs_base>> pcs;

    array_view<Goldilocks2::Element> input_data, input_label;

    std::vector<layer_info> layers;
    
    std::vector<Goldilocks2::Element> e_pow_inv;

    // bool check_conv_relu(int n_samples = 0) const;
};


