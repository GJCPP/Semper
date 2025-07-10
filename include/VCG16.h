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
        std::string key_input, key_output, key_weight, key_d_input, key_d_output, key_d_weight, key_aux;

        std::vector<array_view<Goldilocks2::Element>> input, d_input,
                                                      weight, d_weight,
                                                      output, d_output, // Must also be the input of the next layer
                                                      aux; // auxiliary data for checking

        std::vector<std::vector<std::shared_ptr<MultilinearPolynomial>>> *mle_input, *mle_output,
                                                                      *mle_weight, *mle_d_weight,
                                                                      *mle_d_input, *mle_d_output;

        std::vector<std::vector<std::shared_ptr<ligeropcs_base>>> *pcs_input, *pcs_output, // [batch][img]
                                                                  *pcs_weight, *pcs_d_weight, // [batch][0]
                                                                  *pcs_d_input, *pcs_d_output; // [batch][img]

        // For debug only.
        ligeropcs_base *get_pcs_input(size_t batch, size_t img);
        ligeropcs_base *get_pcs_output(size_t batch, size_t img);
        ligeropcs_base *get_pcs_weight(size_t batch, size_t img);
        ligeropcs_base *get_pcs_d_weight(size_t batch, size_t img);
        ligeropcs_base *get_pcs_d_input(size_t batch, size_t img);
        ligeropcs_base *get_pcs_d_output(size_t batch, size_t img);
    };
    VCG16(std::string data_dir, int epoch, int64_t scale, int64_t max_value);

    bool check(size_t n_samples = 0) const;

    bool prove(int sec_param);

    void add_layer(layer_type type, int batches,
                    const std::string& name,
                    const std::string& input,
                    const std::string& output,
                    const std::string& weight,
                    const std::string& d_input,
                    const std::string& d_weight,
                    const std::string& d_output,
                    const std::string& aux = "");

protected:
    void init_layer_mle_pcs(layer_info& info);
    void init_mle_pcs_input(const std::string& key);
    void init_mle_pcs_weight(const std::string& key);

    void init_e_pow();

    int64_t scale, max_val, sqr_val;
    int epoch, minibatch, img_per_batch;
    
    cnpy::npz_t filedata; // responsible for releasing data
    
    std::map<std::string, std::unique_ptr<Goldilocks2::Element[]>> data;
    std::map<std::string, std::vector<size_t>> data_shape;
    std::map<std::string, std::vector<std::vector<std::shared_ptr<MultilinearPolynomial>>>> mle; // [key][batch][img]
    std::map<std::string, std::vector<std::vector<std::shared_ptr<ligeropcs_base>>>> pcs;

    array_view<Goldilocks2::Element> input_data, input_label;

    std::vector<layer_info> layers;
    
    std::vector<Goldilocks2::Element> e_pow_inv;

    // bool check_conv_relu(int n_samples = 0) const;
};

