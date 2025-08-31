#pragma once

#include <format>
#include <random>
#include <unordered_map>

#include "header"

#include "ligero.h"
#include "goldilocks_quadratic_ext.h"
#include "data_loader.h"

class CNN {
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

        std::vector<std::shared_ptr<MultilinearPolynomial>> mle_input, mle_output,
                                            mle_weight, mle_d_weight,
                                            mle_d_input, mle_d_output,
                                            mle_aux;
                                            
        std::vector<std::shared_ptr<ligeropcs_base>> pcs_input, pcs_output,
                                        pcs_weight, pcs_d_weight,
                                        pcs_d_input, pcs_d_output,
                                        pcs_aux;

        int id;

        std::shared_ptr<ligeropcs_base> get_pcs_input(int bat) const;
        std::shared_ptr<ligeropcs_base> get_pcs_output(int bat) const;
        std::shared_ptr<ligeropcs_base> get_pcs_weight(int bat) const ;
        std::shared_ptr<ligeropcs_base> get_pcs_d_weight(int bat) const;
        std::shared_ptr<ligeropcs_base> get_pcs_d_input(int bat) const;
        std::shared_ptr<ligeropcs_base> get_pcs_d_output(int bat) const;
        std::shared_ptr<ligeropcs_base> get_pcs_aux(int bat) const;
    };
    class conv_wit {
    public:
        conv_wit() = default;
        conv_wit(const std::string& data_dir, int epoch, int conv_id)
             : data_dir(data_dir), epoch(epoch), batch(-1), conv_id(conv_id) {}

        bool empty() const { return data_dir.empty(); }

        std::map<std::string, array<Goldilocks2::Element>> get_conv_wit(const std::vector<Goldilocks2::Element>& alpha) const {
            std::map<std::string, array<Goldilocks2::Element>> res;
            if (batch == -1) {
                throw std::invalid_argument("CNN::conv_wit::get_conv_wit Batch not set");
            }
            switch (state) {
            case 0:
                res = to_field(loadConvWitForward(data_dir, epoch, batch, conv_id));
                break;
            case 1:
                res = to_field(loadConvWitDW(data_dir, epoch, batch, conv_id));
                break;
            case 2:
                res = to_field(loadConvWitDX(data_dir, epoch, batch, conv_id));
                break;
            }
            
            auto& X = res["X"], &W = res["W"], &Y = res["Y"];
            int C = X.view.shape(1), in = X.view.shape(2),
                D = Y.view.shape(1), on = Y.view.shape(2);
            auto batch_sz = X.view.shape(0);
            if (Y.view.shape(0) != batch_sz) {
                throw std::runtime_error("conv_wit::get_conv_wit: Batch size mismatch");
            }
            if (alpha.size() != batch_sz) {
                throw std::runtime_error("conv_wit::get_conv_wit: Alpha size mismatch");
            }

            // Combine X and Y
            array<Goldilocks2::Element> rX, rY;
            rX.init({size_t(C), size_t(in)});
            rY.init({size_t(D), size_t(on)});
            for (int i = 0; i != batch_sz; ++i) {
                for (int j = 0; j != C; ++j) {
                    for (int k = 0; k != in; ++k) {
                        rX.view(j, k) += alpha[i] * X.view(i, j, k);
                    }
                }
                for (int j = 0; j != D; ++j) {
                    for (int k = 0; k != on; ++k) {
                        rY.view(j, k) += alpha[i] * Y.view(i, j, k);
                    }
                }
            }
            std::map<std::string, array<Goldilocks2::Element>> ret;
            ret["X"] = std::move(rX);
            ret["W"] = std::move(W);
            ret["Y"] = std::move(rY);
            return ret;
        }

        void set_forward() { state = 0; }
        void set_dW() { state = 1; }
        void set_dX() { state = 2; }

        void set_batch(int bat) { batch = bat; }
    protected:
        std::string data_dir;
        int epoch, batch, conv_id;
        int state; // 0, 1, 2 for forward, dW, dX

        static std::map<std::string, array<Goldilocks2::Element>> to_field(const cnpy::npz_t& data) {
            std::map<std::string, array<Goldilocks2::Element>> res;
            for (auto& [key, val] : data) {
                array<Goldilocks2::Element> arr(val.shape);
                auto ptr = arr.data.data();
                for (size_t i = 0; i != val.num_vals; ++i) {
                    ptr[i] = Goldilocks2::fromS64(val.data<int64_t>()[i]);
                }
                res[key] = std::move(arr);
            }
            return res;
        }
    };
    CNN(std::string model_name, std::string data_dir, int epoch, int64_t scale, int64_t max_value, uint64_t rho_inv);

    // This is for checking data integrity, and is not to be executed in real proof.
    bool check(size_t n_samples = 0) const;

    bool prove(size_t sec_param);

    bool prove_input(size_t sec_param);

    void add_layer(layer_type type, int id,
                    const std::string& name,
                    const std::string& input,
                    const std::string& output,
                    const std::string& weight,
                    const std::string& d_input,
                    const std::string& d_weight,
                    const std::string& d_output,
                    const std::string& aux = "");

protected:
    std::string model_name, data_dir;

    int epoch, minibatch, img_per_batch;
    int64_t scale, max_val, sqr_val;
    uint64_t rho_inv;
    
    cnpy::npz_t dataset, filedata; // responsible for releasing data
    
    std::map<std::string, std::unique_ptr<Goldilocks2::Element[]>> data;
    std::map<std::string, std::vector<size_t>> data_shape;
    std::map<std::string, array_view<Goldilocks2::Element>> data_view;
    std::map<std::string, std::vector<std::shared_ptr<MultilinearPolynomial>>> mle;
    std::map<std::string, std::vector<std::shared_ptr<ligeropcs_base>>> pcs;

    MLE mle_dataset_input, mle_dataset_label, mle_input, mle_label, mle_index;
    ligeropcs_base pcs_dataset_input, pcs_dataset_label, pcs_input, pcs_label, pcs_index;
    array_view<Goldilocks2::Element> dataset_input, dataset_label, input, label, input_index;

    std::vector<layer_info> layers;
};

