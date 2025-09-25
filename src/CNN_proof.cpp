#include <cassert>

#include <chrono>

#include "CNN.h"
#include "CNN_check.h"
#include "CNN_proof.h"
#include "conv_check.h"
#include "mat_check.h"
#include "div_check.h"
#include "sign_check.h"
#include "prod_check.h"
#include "e_pow_check.h"
#include "timer.h"
#include "counter.h"

bool prove_conv(
    size_t C, size_t D, size_t n, size_t m, size_t padding,
    open_param oX, // [C, n, n]
    open_param oW, // [C, D, m, m]
    open_param oY, // [D, on, on]
    open_param oflat_Y,
    const std::map<std::string, array<Goldilocks2::Element>>& wit,
    const array_view<Goldilocks2::Element>& X, // [C, n, n]
    const array_view<Goldilocks2::Element>& W, // [C, D, m, m]
    const array_view<Goldilocks2::Element>& Y, // [D, on, on]
    const array_view<Goldilocks2::Element>& ori_Y,
    std::vector<size_t>& mapfrom,
    std::vector<size_t>& mapto,
    size_t rho_inv, size_t sec_param,
    lazyMapProver *lazy_map_prover, lazyMapVerifier *lazy_map_verifier) {

    // auto _mapto = mapto;
    auto prover = wit.empty() ?
             make_conv2_prover(C, D, n, m, padding, X, W, Y, mapto) :
             make_conv2_prover_wit(C, D, n, m, padding, wit.at("X"), wit.at("W"), wit.at("Y"), mapto);

    prover.init_ori_Y(ori_Y);

    oW.rev[2] = oW.rev[2] ^ true;
    oW.rev[3] = oW.rev[3] ^ true;

    
    set_timer("execute convcheck");
    if (!convVerifier::execute_convcheck_2d(prover, oX, oW, oY, oflat_Y, mapfrom, mapto, rho_inv, sec_param, true, lazy_map_prover, lazy_map_verifier)) {
        return false;
    }
    pause_timer("execute convcheck");
    return true;
}

void pre_prove_conv(
    CNN::conv_wit wit,
    int padding,
    const array_view<Goldilocks2::Element>& X, // [N, IC, in, in]
    const array_view<Goldilocks2::Element>& W, // [OC, IC, m, m]
    const array_view<Goldilocks2::Element>& Y, // [N, OC, on, on]
    bool pad_right_bottom,
    CNN::layer_res& res, lazy_pcs_pool *pool, const std::string& prename) {

    std::map<std::string, array<Goldilocks2::Element>> witness;
    if (wit.empty()) {
        std::cerr << "Error: No witness provided for convolution proof." << std::endl;
        throw std::invalid_argument("CNN::pre_prove_conv: No witness provided");
    }
    witness = wit.get_conv_wit();

    res.pcs[prename + "Y_flat"] = commit_lazy_pcs(MLE(witness["Y"]), pool);
    // std::cout << "Y_flat shape = ";
    // for (size_t s : witness["Y"].view.get_shape()) {
    //     std::cout << s << " ";
    // }
    // std::cout << std::endl;
}
/*
bool prove_conv(
    CNN::conv_wit wit,
    int padding,
    const oracle* pcs_input, const oracle* pcs_weight, const oracle* pcs_output,
    const array_view<Goldilocks2::Element>& X, // [N, IC, in, in]
    const array_view<Goldilocks2::Element>& W, // [OC, IC, m, m]
    const array_view<Goldilocks2::Element>& Y, // [N, OC, on, on]
    bool pad_right_bottom,
    size_t rho_inv, size_t sec_param,
    lazyMapProver *lazy_map_prover, lazyMapVerifier *lazy_map_verifier) {

    const int N = X.shape(0), IC = X.shape(1), in = X.shape(2), OC = Y.shape(1), on = Y.shape(2);
    const int m = W.shape(2);
    const int logN = find_ceiling_log2(N);

    assert(X.shape(0) == static_cast<size_t>(N));
    assert(X.shape(1) == static_cast<size_t>(IC));
    assert(X.shape(2) == static_cast<size_t>(in));
    assert(X.shape(3) == static_cast<size_t>(in));
    assert(W.shape(0) == static_cast<size_t>(OC));
    assert(W.shape(1) == static_cast<size_t>(IC));
    assert(W.shape(2) == static_cast<size_t>(m));
    assert(W.shape(3) == static_cast<size_t>(m));
    assert(Y.shape(0) == static_cast<size_t>(N));
    assert(Y.shape(1) == static_cast<size_t>(OC));
    assert(Y.shape(2) == static_cast<size_t>(on));
    assert(Y.shape(3) == static_cast<size_t>(on));
    assert(on == in + 2 * padding - m + 1);

    // Step 1. Combine conv with same kernel
    
    array<Goldilocks2::Element> cX, cY;
    cX.init({size_t(IC), size_t(in), size_t(in)});
    cY.init({size_t(OC), size_t(on), size_t(on)});
    auto c_cha = random_vec_ext(logN);
    MLE_Eq c_cha_eq(c_cha);
    auto c_cha_table = c_cha_eq.get_eval_table();
    c_cha_table.resize(N);
    for (int i = 0; i < N; ++i) {
        for (int a = 0; a != IC; ++a) {
            for (int b = 0; b != in; ++b) {
                for (int c = 0; c != in; ++c) {
                    cX(a, b, c) += c_cha_table[i] * X(i, a, b, c);
                }
            }
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int a = 0; a != OC; ++a) {
            for (int b = 0; b != on; ++b) {
                for (int c = 0; c != on; ++c) {
                    cY(a, b, c) += c_cha_table[i] * Y(i, a, b, c);
                }
            }
        }
    }
    
    std::map<std::string, array<Goldilocks2::Element>> witness;
    if (!wit.empty()) {
        set_timer("load conv wit");
        witness = wit.get_conv_wit(c_cha_table);
        pause_timer("load conv wit");
    }

    // Step 2. Padding
    auto& iW = W;

    auto iX = cX.view; // [C, n, n]
    auto iY = cY.view; // [OC, on, on]
    auto& pX = iX;
    array<Goldilocks2::Element> pW; // [OC, IC, m, m]
    array<Goldilocks2::Element> pY; // [OC, on, on]
    size_t new_m, new_padding;

    size_t up_on = (1ull << find_ceiling_log2(on));
    std::vector<size_t> mapto(OC * on * on);
    std::vector<size_t> mapfrom;
    for (size_t i = 0; i != mapto.size(); ++i) mapto[i] = i;
    mapfrom.reserve(OC * on * on);
    for (int i = 0; i != OC; ++i) {
        for (int j = 0; j != on; ++j) {
            for (int k = 0; k != on; ++k) {
                mapfrom.push_back(i * up_on * up_on + j * up_on + k);
            }
        }
    }
    if (witness.empty()) {
        std::cout << "Warning: No witness provided for convolution proof." << std::endl;
        pad_weights(IC, OC, in, m, padding,
            iX,
            iW,
            iY,
            pW, pY, new_m, new_padding, pad_right_bottom);
        pW.view.swap_dim(0, 1); // [IC, OC, m, m]
    }
    pad_weights_map(IC, OC, in, m, padding, new_m, new_padding, mapto, pad_right_bottom);

    auto cW = W;
    cW.swap_dim(0, 1);
    open_param open_cX(X, pcs_input);
    open_param oW(cW, pcs_weight);
    open_param open_cY(Y, pcs_output);
    auto pcs_flat_Y = ligero_commit_ext(MLE(witness.at("Y")), rho_inv);
    open_param open_flat_Y(witness.at("Y").view, &pcs_flat_Y);
    open_cX = open_cX(c_cha);
    open_cY = open_cY(c_cha);

    if (!prove_conv(IC, OC, in, new_m, new_padding,
        open_cX, oW, open_cY, open_flat_Y,
        witness,
        pX, pW, pY, 
        iY,
        mapfrom, mapto,
        rho_inv, sec_param,
        lazy_map_prover, lazy_map_verifier)) throw;

    return true;
}
*/

bool prove_conv(
    CNN::conv_wit wit,
    int padding,
    const oracle* pcs_input, const oracle* pcs_weight, const oracle* pcs_output,
    const array_view<Goldilocks2::Element>& X, // [N, IC, in, in]
    const array_view<Goldilocks2::Element>& W, // [OC, IC, m, m]
    const array_view<Goldilocks2::Element>& Y, // [N, OC, on, on]
    bool pad_right_bottom,
    size_t rho_inv, size_t sec_param,
    const CNN::layer_res& res, const std::string& prename,
    lazyMapProver *lazy_map_prover, lazyMapVerifier *lazy_map_verifier) {

    const int N = X.shape(0), IC = X.shape(1), in = X.shape(2), OC = Y.shape(1), on = Y.shape(2);
    const int m = W.shape(2);
    const int logN = find_ceiling_log2(N);

    assert(X.shape(0) == static_cast<size_t>(N));
    assert(X.shape(1) == static_cast<size_t>(IC));
    assert(X.shape(2) == static_cast<size_t>(in));
    assert(X.shape(3) == static_cast<size_t>(in));
    assert(W.shape(0) == static_cast<size_t>(OC));
    assert(W.shape(1) == static_cast<size_t>(IC));
    assert(W.shape(2) == static_cast<size_t>(m));
    assert(W.shape(3) == static_cast<size_t>(m));
    assert(Y.shape(0) == static_cast<size_t>(N));
    assert(Y.shape(1) == static_cast<size_t>(OC));
    assert(Y.shape(2) == static_cast<size_t>(on));
    assert(Y.shape(3) == static_cast<size_t>(on));
    assert(on == in + 2 * padding - m + 1);

    // Step 1. Combine conv with same kernel
    
    array<Goldilocks2::Element> cX, cY;
    cX.init({size_t(IC), size_t(in), size_t(in)});
    cY.init({size_t(OC), size_t(on), size_t(on)});
    auto c_cha = random_vec_ext(logN);
    MLE_Eq c_cha_eq(c_cha);
    auto c_cha_table = c_cha_eq.get_eval_table();
    c_cha_table.resize(N);
    for (int i = 0; i < N; ++i) {
        for (int a = 0; a != IC; ++a) {
            for (int b = 0; b != in; ++b) {
                for (int c = 0; c != in; ++c) {
                    cX(a, b, c) += c_cha_table[i] * X(i, a, b, c);
                }
            }
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int a = 0; a != OC; ++a) {
            for (int b = 0; b != on; ++b) {
                for (int c = 0; c != on; ++c) {
                    cY(a, b, c) += c_cha_table[i] * Y(i, a, b, c);
                }
            }
        }
    }
    
    std::map<std::string, array<Goldilocks2::Element>> witness;
    if (!wit.empty()) {
        set_timer("load conv wit");
        witness = wit.get_conv_wit(c_cha_table);
        pause_timer("load conv wit");
    }

    // Step 2. Padding

    auto iX = cX.view; // [C, n, n]
    auto iY = cY.view; // [OC, on, on]
    auto& pX = iX;
    array<Goldilocks2::Element> pW; // [OC, IC, m, m]
    array<Goldilocks2::Element> pY; // [OC, on, on]
    size_t new_m, new_padding;

    size_t up_on = (1ull << find_ceiling_log2(on));
    std::vector<size_t> mapto(OC * on * on);
    std::vector<size_t> mapfrom;
    for (size_t i = 0; i != mapto.size(); ++i) mapto[i] = i;
    mapfrom.reserve(OC * on * on);
    for (int i = 0; i != OC; ++i) {
        for (int j = 0; j != on; ++j) {
            for (int k = 0; k != on; ++k) {
                mapfrom.push_back(i * up_on * up_on + j * up_on + k);
            }
        }
    }
    if (witness.empty()) {
        std::cerr << __LINE__ << ") Error: No witness provided for convolution proof." << std::endl;
        throw std::invalid_argument("CNN::prove_conv: No witness provided");
    }
    pad_weights_map(IC, OC, in, m, padding, new_m, new_padding, mapto, pad_right_bottom);

    auto cW = W;
    cW.swap_dim(0, 1);
    open_param open_cX(X, pcs_input);
    open_param oW(cW, pcs_weight);
    open_param open_cY(Y, pcs_output);
    open_param open_flat_Y(witness.at("raw_Y").view, &res.pcs.at(prename + "Y_flat"));

    open_cX = open_cX(c_cha);
    open_cY = open_cY(c_cha);
    open_flat_Y = open_flat_Y(c_cha);

    if (!prove_conv(IC, OC, in, new_m, new_padding,
        open_cX, oW, open_cY, open_flat_Y,
        witness,
        pX, pW, pY, 
        iY,
        mapfrom, mapto,
        rho_inv, sec_param,
        lazy_map_prover, lazy_map_verifier)) throw;

    return true;
}

bool pre_prove_conv_forward(const CNN::layer_info& layer, CNN::conv_wit wit, int padding, CNN::layer_res& res, lazy_pcs_pool* pool) {
    int bat = int(layer.input.shape(0));

    // #pragma omp parallel for
    for (int i = 0; i < bat; ++i) {
        auto wit_copy = wit;
        wit_copy.set_forward();
        wit_copy.set_batch(i);

        pre_prove_conv(wit_copy, padding,
            layer.input[i], layer.weight[i], layer.output[i],
            true,
            res, pool, "forward_" + std::to_string(i));
    }
    return true;
}

bool prove_conv_forward(
    const CNN::layer_info& layer, 
    CNN::conv_wit wit, 
    int padding, size_t rho_inv, size_t sec_param,
    lazyMapProver *lazy_map_prover, lazyMapVerifier *lazy_map_verifier) {

    int bat = int(layer.input.shape(0));

    #pragma omp parallel for
    for (int i = 0; i < bat; ++i) {
        auto wit_copy = wit;
        wit_copy.set_forward();
        auto pcs_input = layer.get_pcs_input(i);
        auto pcs_weight = layer.get_pcs_weight(i);
        auto pcs_output = layer.get_pcs_output(i);

        wit_copy.set_batch(i);

        if (!prove_conv(wit_copy, padding,
            &pcs_input, &pcs_weight, &pcs_output,
            layer.input[i], layer.weight[i], layer.output[i],
            true,
            rho_inv, sec_param,
            layer.wit, "forward_" + std::to_string(i),
            lazy_map_prover, lazy_map_verifier)) throw;
    }
    return true;
}


bool pre_prove_conv_backward_dW(const CNN::layer_info& layer, CNN::conv_wit _wit, int padding, CNN::layer_res& res, lazy_pcs_pool* pool) {
    int bat = int(layer.input.shape(0));

    // #pragma omp parallel for
    for (int i = 0; i < bat; ++i) {
        auto wit = _wit;
        wit.set_dW();
        wit.set_batch(i);

        pre_prove_conv(wit, padding,
            layer.input[i], layer.d_output[i], layer.d_weight[i],
            true,
            res, pool, "dW_" + std::to_string(i));
    }
    return true;
}

bool prove_conv_backward_dW(
    const CNN::layer_info& layer, 
    CNN::conv_wit _wit, 
    int padding, size_t rho_inv, size_t sec_param,
    lazyMapProver *lazy_map_prover, lazyMapVerifier *lazy_map_verifier) {

    int bat = int(layer.input.shape(0));

    #pragma omp parallel for
    for (int i = 0; i < bat; ++i) {
        auto pcs_input = layer.get_pcs_input(i);
        auto pcs_d_output = layer.get_pcs_d_output(i);
        auto pcs_d_weight = layer.get_pcs_d_weight(i);
        auto X = layer.input[i]; // [N, C, n, n]
        auto dY = layer.d_output[i]; // [N, D, on, on]
        auto dW = layer.d_weight[i]; // [D, C, m, m]
        
        auto wit = _wit;
        wit.set_dW();
        wit.set_batch(i);

        X.swap_dim(0, 1); // [C, N, n, n]
        dY.swap_dim(0, 1); // [D, N, on, on]
        dW.swap_dim(0, 1); // [C, D, m, m]

        if (!prove_conv(wit, padding,
            &pcs_input, &pcs_d_output, &pcs_d_weight,
            X, dY, dW,
            true,
            rho_inv, sec_param,
            layer.wit, "dW_" + std::to_string(i),
            lazy_map_prover, lazy_map_verifier)) throw;
    }
    return true;
}

bool pre_prove_conv_backward_dX(const CNN::layer_info& layer, CNN::conv_wit _wit, int padding, CNN::layer_res& res, lazy_pcs_pool* pool) {
    int bat = int(layer.input.shape(0));

    // #pragma omp parallel for
    for (int i = 0; i < bat; ++i) {
        auto wit = _wit;
        wit.set_dX();
        wit.set_batch(i);

        pre_prove_conv(wit, padding,
            layer.d_output[i], layer.weight[i], layer.d_input[i],
            false,
            res, pool, "dX_" + std::to_string(i));
    }
    return true;
}

bool prove_conv_backward_dX(
    const CNN::layer_info& layer, 
    CNN::conv_wit _wit, 
    int padding, size_t rho_inv, size_t sec_param,
    lazyMapProver *lazy_map_prover, lazyMapVerifier *lazy_map_verifier) {

    int bat = int(layer.input.shape(0));

    #pragma omp parallel for
    for (int i = 0; i < bat; ++i) {
        auto pcs_d_output = layer.get_pcs_d_output(i);
        auto pcs_weight = layer.get_pcs_weight(i);
        auto pcs_d_input = layer.get_pcs_d_input(i);
        auto dY = layer.d_output[i]; // [N, D, on, on]
        auto W = layer.weight[i]; // [D, C, m, m]
        auto dX = layer.d_input[i]; // [N, C, n, n]
        
        auto wit = _wit;
        wit.set_dX();
        wit.set_batch(i);

        W.swap_dim(0, 1); // [C, D, m, m]
        W.reverse(2);
        W.reverse(3); // [C, D, m, m]

        if (!prove_conv(wit, padding,
            &pcs_d_output, &pcs_weight, &pcs_d_input,
            dY, W, dX,
            false,
            rho_inv, sec_param,
            layer.wit, "dX_" + std::to_string(i),
            lazy_map_prover, lazy_map_verifier)) throw;
    }
    return true;
}

CNN::layer_res pre_prove_conv_layer(const CNN::layer_info& layer, CNN::conv_wit wit, lazy_pcs_pool* pool) {
    const int padding = 1;

    assert(layer.type == CNN::layer_type::conv);
    assert(layer.input.get_dims() == 5); // bat x img x C x n x n
    assert(layer.weight.get_dims() == 5); // bat x D x C x m x m
    assert(layer.output.get_dims() == 5); // bat x img x D x n x n



    CNN::layer_res res;

    set_timer("preprove conv forward");
    pre_prove_conv_forward(layer, wit, padding, res, pool);
    pause_timer("preprove conv forward");

    set_timer("prove conv dW");
    pre_prove_conv_backward_dW(layer, wit, padding, res, pool);
    pause_timer("prove conv dW");

    set_timer("prove conv dX");
    pre_prove_conv_backward_dX(layer, wit, padding, res, pool);
    pause_timer("prove conv dX");


    return res;
}

bool prove_conv_layer(
    const CNN::layer_info& layer, 
    CNN::conv_wit wit, 
    size_t rho_inv, size_t sec_param,
    lazyMapProver *lazy_map_prover, lazyMapVerifier *lazy_map_verifier) {

    startCounter counter("conv_proof");
    const int padding = 1;

    assert(layer.type == CNN::layer_type::conv);
    assert(layer.input.get_dims() == 5); // bat x img x C x n x n
    assert(layer.weight.get_dims() == 5); // bat x D x C x m x m
    assert(layer.output.get_dims() == 5); // bat x img x D x n x n





    set_timer("prove conv forward");
    if (!prove_conv_forward(layer, wit, padding, rho_inv, sec_param, lazy_map_prover, lazy_map_verifier)) {
        std::cout << "❌ Proving forward pass failed." << std::endl;
        throw;
    }
    pause_timer("prove conv forward");

    set_timer("prove conv dW");
    if (!prove_conv_backward_dW(layer, wit, padding, rho_inv, sec_param, lazy_map_prover, lazy_map_verifier)) {
        std::cout << "❌ Proving backward dW failed." << std::endl;
        throw;
    }
    pause_timer("prove conv dW");

    set_timer("prove conv dX");
    if (!prove_conv_backward_dX(layer, wit, padding, rho_inv, sec_param, lazy_map_prover, lazy_map_verifier)) {
        std::cout << "❌ Proving backward dX failed." << std::endl;
        throw;
    }
    pause_timer("prove conv dX");


    return true;
}

bool prove_full(
    open_param oX, // [N, n]
    open_param oW, // [n, m]
    open_param oY, // [N, m]
    const array_view<Goldilocks2::Element>& X, // [N, n]
    const array_view<Goldilocks2::Element>& W, // [n, m]
    const array_view<Goldilocks2::Element>& Y, // [N, m]
    size_t sec_param) {

    const size_t N = X.shape(0), n = X.shape(1), m = Y.shape(1);
    assert(X.shape(0) == N && X.shape(1) == n);
    assert(W.shape(0) == n && W.shape(1) == m);
    assert(Y.shape(0) == N && Y.shape(1) == m);

    mat_mult_prover prover(N, n, m, X, W, Y);
    return mat_mult_verifier::execute_mat_mult_check(prover, oX, oW, oY, sec_param);
}

bool prove_full_layer(const CNN::layer_info& layer, size_t rho_inv, size_t sec_param) {
    startCounter counter("full_proof");
    const int batch = int(layer.input.shape(0));
    // Prove forward
    #pragma omp parallel for
    for (int i = 0; i < batch; ++i) {
        auto pcs_input = layer.get_pcs_input(i);
        auto pcs_weight = layer.get_pcs_weight(i);
        auto pcs_output = layer.get_pcs_output(i);

        open_param oX(layer.input[i], &pcs_input);
        open_param oW(layer.weight[i], &pcs_weight);
        open_param oY(layer.output[i], &pcs_output);

        if (!prove_full(oX, oW, oY,
            layer.input[i], layer.weight[i], layer.output[i],
            sec_param)) {
            std::cout << "❌ Proving full forward failed." << std::endl;
            throw;
        }
    }

    // Prove backward dW = X^T * dY
    #pragma omp parallel for
    for (int i = 0; i < batch; ++i) {
        auto pcs_input = layer.get_pcs_input(i);
        auto pcs_d_weight = layer.get_pcs_d_weight(i);
        auto pcs_d_output = layer.get_pcs_d_output(i);
        auto X = layer.input[i]; // [N, n]
        auto dW = layer.d_weight[i]; // [n, m]
        auto dY = layer.d_output[i]; // [N, m]

        X.swap_dim(0, 1); // [n, N]

        auto oX = open_param(X, &pcs_input);
        auto odW = open_param(dW, &pcs_d_weight);
        auto odY = open_param(dY, &pcs_d_output);

        if (!prove_full(oX, odY, odW,
            X, dY, dW,
            sec_param)) {
            std::cout << "❌ Proving full backward dW failed." << std::endl;
            throw;
        }
    }

    // Prove backward dX = dY * W^T
    #pragma omp parallel for
    for (int i = 0; i < batch; ++i) {
        auto pcs_d_output = layer.get_pcs_d_output(i);
        auto pcs_weight = layer.get_pcs_weight(i);
        auto pcs_d_input = layer.get_pcs_d_input(i);
        auto dY = layer.d_output[i]; // [N, m]
        auto W = layer.weight[i];    // [n, m]
        auto dX = layer.d_input[i];  // [N, n]

        W.swap_dim(0, 1); // [m, n]

        auto odY = open_param(dY, &pcs_d_output);
        auto oW = open_param(W, &pcs_weight);
        auto odX = open_param(dX, &pcs_d_input);

        if (!prove_full(odY, oW, odX,
            dY, W, dX,
            sec_param)) {
            std::cout << "❌ Proving full backward dX failed." << std::endl;
            throw;
        }
    }
    return true;
}


CNN::layer_res pre_prove_relu_layer(
    const CNN::layer_info& layer,
    size_t scale, size_t max_val, size_t sqr_val,
    size_t rho_inv,
    lazyLogupProver* lazy_logup_prover,
    lazyLogupVerifier* lazy_logup_verifier,
    lazy_pcs_pool* pool) {

    CNN::layer_res ret;


    const int batch = int(layer.input.shape(0));

    // #pragma omp parallel for
    for (int i = 0; i != batch; ++i) {
        std::string _i = "_" + std::to_string(i);
        // Prove forward
        auto pcs_X = layer.get_pcs_input(i);
        auto pcs_Y = layer.get_pcs_output(i);
        auto X = layer.input[i];
        auto Y = layer.output[i];

        size_t n = X.size();

        std::vector<Goldilocks2::Element> X_copy(n), Y_copy(n);

        X.copy_to(X_copy.data());
        Y.copy_to(Y_copy.data());

        // 1. Prove scale_X
        std::vector<Goldilocks2::Element> X_quo(n);
        for (size_t i = 0; i < n; ++i) {
            if (Goldilocks2::toS64(X_copy[i]) > 0) {
                X_quo[i] = Y_copy[i]; // ceil division
            }
            else {
                X_quo[i] = Goldilocks2::fromS64(Goldilocks2::toS64(X_copy[i]) / int64_t(scale));
            }
        }
        auto X_rem = get_rem(X_copy, scale, X_quo, true);

        // ret.vec["X_quo" + _i] = X_quo;
        // ret.vec["X_rem" + _i] = X_rem;
        lazy_pcs pcs_X_quo = commit_lazy_pcs(X_quo, pool);
        lazy_pcs pcs_X_rem = commit_lazy_pcs(X_rem, pool);
        ret.pcs["pcs_X_quo" + _i] = pcs_X_quo;
        ret.pcs["pcs_X_rem" + _i] = pcs_X_rem;

        // 2. Prove sign of scaled X
        auto X_scale_sign = get_sign(X_quo, true);
        lazy_pcs pcs_X_scale_sign = commit_lazy_pcs(X_scale_sign, pool);
        // ret.vec["X_scale_sign" + _i] = X_scale_sign;
        ret.pcs["pcs_X_scale_sign" + _i] = pcs_X_scale_sign;
        signProver sign_prover_X(X_quo, X_scale_sign, scale, max_val, true, rho_inv, lazy_logup_prover);
        signVerifier::resource sign_res = signVerifier::pre_execute_sign_check(sign_prover_X, pool, lazy_logup_verifier);
        ret.res["sign_res" + _i] = std::make_shared<signVerifier::resource>(sign_res);
        // 3. Prove Y = X_scale_sign * X_quo

        // Prove backward
        auto dX = layer.d_input[i];
        auto dY = layer.d_output[i];
        std::vector<Goldilocks2::Element> dX_copy(n), dY_copy(n);
        dX.copy_to(dX_copy.data());
        dY.copy_to(dY_copy.data());
        // 1. Prove scaled dY_filtered = dY * X_scale_sign
        std::vector<Goldilocks2::Element> dY_filtered(n);
        for (size_t j = 0; j < n; ++j) {
            if (X_scale_sign[j] == Goldilocks2::one()) {
                dY_filtered[j] = dY_copy[j];
            }
        }
        // ret.vec["dY_filtered" + _i] = dY_filtered;
        ret.pcs["pcs_dY_filtered" + _i] = commit_lazy_pcs(dY_filtered, pool);

        // 2. Prove dX = dY_filtered / scale
        auto dX_rem = get_rem(dY_filtered, scale, dX_copy, true);
        // ret.vec["dX_rem" + _i] = dX_rem;
        ret.pcs ["pcs_dX_rem" + _i] = commit_lazy_pcs(dX_rem, pool);
    }
    return ret;
}

bool prove_relu_layer(const CNN::layer_info& layer,
    size_t scale, size_t max_val, size_t sqr_val,
    size_t rho_inv, size_t sec_param, const CNN::layer_res& wit,
    lazyLogupProver* lazy_logup_prover, lazyLogupVerifier* lazy_logup_verifier) {

    startCounter counter("relu_proof");

    const int batch = int(layer.input.shape(0));

    #pragma omp parallel for
    for (int i = 0; i != batch; ++i) {
        std::string _i = "_" + std::to_string(i);
        // Prove forward
        auto pcs_X = layer.get_pcs_input(i);
        auto pcs_Y = layer.get_pcs_output(i);
        auto X = layer.input[i];
        auto Y = layer.output[i];

        size_t n = X.size();
        int logn = find_ceiling_log2(n);

        std::vector<Goldilocks2::Element> X_copy(n), Y_copy(n);

        X.copy_to(X_copy.data());
        Y.copy_to(Y_copy.data());

        // 1. Prove scale_X        
        std::vector<Goldilocks2::Element> X_quo(n);
        for (size_t i = 0; i < n; ++i) {
            if (Goldilocks2::toS64(X_copy[i]) > 0) {
                X_quo[i] = Y_copy[i]; // ceil division
            }
            else {
                X_quo[i] = Goldilocks2::fromS64(Goldilocks2::toS64(X_copy[i]) / int64_t(scale));
            }
        }
        auto X_rem = get_rem(X_copy, scale, X_quo, true);
        const auto& pcs_X_quo = wit.pcs.at("pcs_X_quo" + _i);
        const auto& pcs_X_rem = wit.pcs.at("pcs_X_rem" + _i);
        start_proof("relu_div_proof");
        divProver div_prover_X(X_copy, X_quo, X_rem, scale, true, rho_inv, lazy_logup_prover);
        if (!divVerifier::execute_div_check(div_prover_X,
            std::make_shared<lazy_pcs>(pcs_X),
            std::make_shared<lazy_pcs>(pcs_X_quo),
            std::make_shared<lazy_pcs>(pcs_X_rem),
            sec_param, lazy_logup_verifier)) {
            std::cout << "❌ Proving relu forward scale_X failed." << std::endl;
            throw;
        }
        end_proof("relu_div_proof");

        // 2. Prove sign of scaled X
        auto X_scale_sign = get_sign(X_quo, true);
        const auto& pcs_X_scale_sign = wit.pcs.at("pcs_X_scale_sign" + _i);
        start_proof("relu_sign_proof");
        signProver sign_prover_X(X_quo, X_scale_sign, scale, max_val, true, rho_inv, lazy_logup_prover);
        auto sign_res = reinterpret_cast<signVerifier::resource*>(wit.res.at("sign_res" + _i).get());
        if (!signVerifier::execute_sign_check(
            sign_prover_X, 
            std::make_shared<lazy_pcs>(pcs_X_quo), 
            std::make_shared<lazy_pcs>(pcs_X_scale_sign), 
            sec_param, 
            lazy_logup_verifier,
            *sign_res)) {

            std::cout << "❌ Proving relu forward X_scale_sign failed." << std::endl;
            throw;
        }
        end_proof("relu_sign_proof");

        // 3. Prove Y = X_scale_sign * X_quo
        auto Y_challenge = random_vec_ext(logn); // draw new challenges
        auto mle_challenge = MLE_Eq(Y_challenge);
        auto claim_Y_cha = pcs_Y.open(Y_challenge, sec_param);
        p3Prover p3_prover_Y(X_scale_sign, X_quo, mle_challenge);
        if (!p3Verifier::execute_sumcheck(p3_prover_Y, claim_Y_cha, { &pcs_X_scale_sign, &pcs_X_quo, &mle_challenge }, sec_param)) {
            std::cout << "❌ Proving relu forward Y failed." << std::endl;
            throw;
        }

        // Prove backward
        auto pcs_dX = layer.get_pcs_d_input(i);
        auto pcs_dY = layer.get_pcs_d_output(i);
        auto dX = layer.d_input[i];
        auto dY = layer.d_output[i];
        std::vector<Goldilocks2::Element> dX_copy(n), dY_copy(n);
        dX.copy_to(dX_copy.data());
        dY.copy_to(dY_copy.data());

        // 1. Prove scaled dY_filtered = dY * X_scale_sign
        std::vector<Goldilocks2::Element> dY_filtered(n);
        for (size_t j = 0; j < n; ++j) {
            if (X_scale_sign[j] == Goldilocks2::one()) {
                dY_filtered[j] = dY_copy[j];
            }
        }
        const auto& pcs_dY_filtered = wit.pcs.at("pcs_dY_filtered" + _i);

        auto dY_challenge = random_vec_ext(logn); // draw new challenges
        auto mle_challenge_dY_filtered = MLE_Eq(dY_challenge);
        p3Prover p3_prover_dY_filtered(X_scale_sign, dY_copy, mle_challenge_dY_filtered);
        auto claim_dY_filtered = pcs_dY_filtered.open(dY_challenge, sec_param);
        if (!p3Verifier::execute_sumcheck(p3_prover_dY_filtered, claim_dY_filtered, { &pcs_X_scale_sign, &pcs_dY, &mle_challenge_dY_filtered }, sec_param)) {
            std::cout << "❌ Proving relu backward dY_filtered failed." << std::endl;
            throw;
        }
        // 2. Prove dX = dY_filtered / scale
        auto dX_rem = get_rem(dY_filtered, scale, dX_copy, true);
        auto pcs_dX_rem = wit.pcs.at("pcs_dX_rem" + _i);
        start_proof("relu_div_proof");
        divProver div_prover_dX(dY_filtered, dX_copy, dX_rem, scale, true, rho_inv, lazy_logup_prover);
        if (!divVerifier::execute_div_check(div_prover_dX,
            std::make_shared<lazy_pcs>(pcs_dY_filtered),
            std::make_shared<lazy_pcs>(pcs_dX),
            std::make_shared<lazy_pcs>(pcs_dX_rem),
            sec_param, 
            lazy_logup_verifier)) {
            std::cout << "❌ Proving relu backward dX failed." << std::endl;
            throw;
        }
        end_proof("relu_div_proof");
    }
    return true;
}

bool prove_pool_shrink(int logimg, int logC, int logN, const oracle& before, const oracle& after, size_t sec_param) {

    auto before_cha = random_vec_ext(logimg + logC + 2 * logN);
    int p0 = logimg + logC + logN - 1, p1 = p0 + logN;
    Goldilocks2::Element sum = Goldilocks2::zero();
    for (int i = 0; i != 2; ++i) {
        for (int j = 0; j != 2; ++j) {
            before_cha[p0] = i ? Goldilocks2::one() : Goldilocks2::zero();
            before_cha[p1] = j ? Goldilocks2::one() : Goldilocks2::zero();
            sum += before.open(before_cha, sec_param);
        }
    }
    std::vector<Goldilocks2::Element> after_cha;
    if (logN == 1) {
        after_cha = before_cha;
        after_cha[p0] = Goldilocks2::zero();
        after_cha[p1] = Goldilocks2::zero();
    } else {
        after_cha.reserve(logimg + logC + 2 * logN - 2);
        for (int i = 0; i < logimg + logC + 2 * logN; ++i) {
            if (i != p0 && i != p1) {
                after_cha.push_back(before_cha[i]);
            }
        }
    }
    return sum == after.open(after_cha, sec_param);
}

CNN::layer_res pre_prove_pool_layer(const CNN::layer_info& layer,
    size_t scale, size_t max_val,
    size_t rho_inv, 
    lazyLogupProver* lazy_logup_prover,
    lazyLogupVerifier* lazy_logup_verifier,
    lazy_pcs_pool* pool)  {
    
    CNN::layer_res ret;
    startCounter counter("pool_proof");

    const int batch = int(layer.input.shape(0));
    const int img = int(layer.input.shape(1));
    const int C = int(layer.input.shape(2));
    const int N = layer.input.shape(3);
    // stride = kernel size = 2
    // layer.input : [batch, img, C, N, N]
    // layer.output: [batch, img, C, N/2, N/2]

    // #pragma omp parallel for
    for (int i = 0; i < batch; ++i) {
        std::string _i = "_" + std::to_string(i);
        auto input = layer.input[i]; // [img, C, N, N]
        auto output = layer.output[i]; // [img, C, N/2, N/2]
        auto d_input = layer.d_input[i]; // [img, C, N, N]
        auto d_output = layer.d_output[i]; // [img, C, N/2, N/2]
        auto aux = layer.aux[i];

        // I. Prepare one-hot selector
        // I.1 Commit selector and reversed selector
        array<Goldilocks2::Element> sel(input.get_shape()), rev_sel(input.get_shape());
        for (int j = 0; j < img; ++j) {
            for (int k = 0; k < C; ++k) {
                auto sel_j_k = sel.view[j][k], rev_sel_j_k = rev_sel.view[j][k];
                for (int x = 0; x < N / 2; ++x) {
                    for (int y = 0; y < N / 2; ++y) {
                        sel_j_k.get(aux(j, k, x, y)[0].fe) = Goldilocks2::one();
                    }
                }
                for (int x = 0; x < N; ++x) {
                    for (int y = 0; y < N; ++y) {
                        rev_sel_j_k(x, y) = Goldilocks2::one() - sel_j_k(x, y);
                    }
                }
            }
        }
        MultilinearPolynomial mle_sel(sel.view), mle_rev_sel(rev_sel.view);
        auto pcs_sel = commit_lazy_pcs(mle_sel, pool);
        auto pcs_rev_sel = commit_lazy_pcs(mle_rev_sel, pool);
        ret.pcs["pcs_sel" + _i] = pcs_sel;
        ret.pcs["pcs_rev_sel" + _i] = pcs_rev_sel;
        // TODO 1. Check pcs_sel = pcs_rev_sel

        // I.2 Prove sel is boolean

        // I.3 Prove sel is one-hot



        // II. Forward.
        // II.1 Prover commits diff
        array<Goldilocks2::Element> diff(input.get_shape());
        for (int j = 0; j < img; ++j) {
            for (int k = 0; k < C; ++k) {
                auto diff_j_k = diff.view[j][k];
                for (int x = 0; x < N; ++x) {
                    for (int y = 0; y < N; ++y) {
                        diff(j, k, x, y) = output(j, k, x / 2, y / 2) - input(j, k, x, y);
                    }
                }
            }
        }
        auto pcs_diff = commit_lazy_pcs(diff.view, pool);
        ret.pcs["pcs_diff" + _i] = pcs_diff;
        // II.2. Prover proves diff = expand(output) - input

        // II.3 Prover proves diff is non-negative
        std::vector<Goldilocks2::Element> input_ones(input.size(), Goldilocks2::one());
        MLE mle_input_ones = MLE(input_ones);
        // TODO 3. Check pcs_input_ones = 1
        signProver sign_prover_diff(diff.data, input_ones, scale, max_val * 2, false, rho_inv, lazy_logup_prover);
        auto sign_res = signVerifier::pre_execute_sign_check(sign_prover_diff, pool, lazy_logup_verifier);
        ret.res["sign_res_diff" + _i] = std::make_shared<signVerifier::resource>(sign_res);

        // II.4 Prover commits and proves sel_input = input * sel
        array<Goldilocks2::Element> sel_input(input.get_shape());
        for (int j = 0; j < img; ++j) {
            for (int k = 0; k < C; ++k) {
                for (int x = 0; x < N; ++x) {
                    for (int y = 0; y < N; ++y) {
                        sel_input(j, k, x, y) = input(j, k, x, y) * sel(j, k, x, y);
                    }
                }
            }
        }
        MultilinearPolynomial mle_sel_input(sel_input.view);
        auto pcs_sel_input = commit_lazy_pcs(mle_sel_input, pool);
        ret.pcs["pcs_sel_input" + _i] = pcs_sel_input;
        
        // II.5 Prover proves output = shrink(sel_input)
        

        // III. Backward.
        // III.1 Prover commits and proves sel_d_input = d_input * sel
        array<Goldilocks2::Element> sel_d_input(input.get_shape());
        for (int j = 0; j < img; ++j) {
            for (int k = 0; k < C; ++k) {
                auto sel_d_input_j_k = sel_d_input.view[j][k];
                for (int x = 0; x < N; ++x) {
                    for (int y = 0; y < N; ++y) {
                        sel_d_input_j_k(x, y) = d_input(j, k, x, y) * sel(j, k, x, y);
                    }
                }
            }
        }
        auto pcs_sel_d_input = commit_lazy_pcs(sel_d_input.view, pool);
        ret.pcs["pcs_sel_d_input" + _i] = pcs_sel_d_input;
        // auto pcs_sel_d_input = ligero_commit_base(sel_d_input.view, rho_inv);

        // III.3 Prover proves 0 = d_input * rev_sel


        // IV. Check all TODO
    }
    return ret;
}
bool prove_pool_layer(const CNN::layer_info& layer,
    size_t scale, size_t max_val,
    size_t rho_inv, size_t sec_param,
    lazyLogupProver* lazy_logup_prover,
    lazyLogupVerifier* lazy_logup_verifier,
    const CNN::layer_res& wit) {

    startCounter counter("pool_proof");

    const int batch = int(layer.input.shape(0));
    const int img = int(layer.input.shape(1));
    const int C = int(layer.input.shape(2));
    const int N = layer.input.shape(3);
    // stride = kernel size = 2
    // layer.input : [batch, img, C, N, N]
    // layer.output: [batch, img, C, N/2, N/2]
    int logimg = find_ceiling_log2(img);
    int logC = find_ceiling_log2(C);
    int logN = find_ceiling_log2(N);

    #pragma omp parallel for
    for (int i = 0; i < batch; ++i) {
        std::string _i = "_" + std::to_string(i);

        auto pcs_input = layer.get_pcs_input(i);
        auto pcs_output = layer.get_pcs_output(i);
        auto pcs_d_input = layer.get_pcs_d_input(i);
        auto pcs_d_output = layer.get_pcs_d_output(i);

        auto input = layer.input[i]; // [img, C, N, N]
        auto output = layer.output[i]; // [img, C, N/2, N/2]
        auto d_input = layer.d_input[i]; // [img, C, N, N]
        auto d_output = layer.d_output[i]; // [img, C, N/2, N/2]
        auto aux = layer.aux[i];

        // const auto& mle_input = *layer.mle_input[i];
        // const auto& mle_output = *layer.mle_output[i];
        const auto& mle_d_input = *layer.mle_d_input[i];
        // const auto& mle_d_output = *layer.mle_d_output[i];

        // I. Prepare one-hot selector
        // I.1 Commit selector and reversed selector
        array<Goldilocks2::Element> sel(input.get_shape()), rev_sel(input.get_shape());
        for (int j = 0; j < img; ++j) {
            for (int k = 0; k < C; ++k) {
                auto sel_j_k = sel.view[j][k], rev_sel_j_k = rev_sel.view[j][k];
                for (int x = 0; x < N / 2; ++x) {
                    for (int y = 0; y < N / 2; ++y) {
                        sel_j_k.get(aux(j, k, x, y)[0].fe) = Goldilocks2::one();
                    }
                }
                for (int x = 0; x < N; ++x) {
                    for (int y = 0; y < N; ++y) {
                        rev_sel_j_k(x, y) = Goldilocks2::one() - sel_j_k(x, y);
                    }
                }
            }
        }
        MultilinearPolynomial mle_sel(sel.view), mle_rev_sel(rev_sel.view);
        auto pcs_sel = wit.pcs.at("pcs_sel" + _i), pcs_rev_sel = wit.pcs.at("pcs_rev_sel" + _i);
        // auto pcs_sel = ligero_commit_base(mle_sel, rho_inv);
        // auto pcs_rev_sel = ligero_commit_base(mle_rev_sel, rho_inv);
        // TODO 1. Check pcs_sel = pcs_rev_sel

        // I.2 Prove sel is boolean
        std::vector<Goldilocks2::Element> input_zeros(input.size());
        MultilinearPolynomial mle_input_zeros(input_zeros);
        if (!prove_mle_product(mle_input_zeros, mle_sel, mle_rev_sel, mle_input_zeros, pcs_sel, pcs_rev_sel, sec_param)) {
            std::cout << "❌ I.2 Proving selector is boolean failed." << std::endl;
            throw;
        }

        // I.3 Prove sel is one-hot
        std::vector<Goldilocks2::Element> sel_cha = random_vec_ext(logimg + logC + 2 * logN);
        int p0 = logimg + logC + logN - 1, p1 = p0 + logN;
        Goldilocks2::Element sum = Goldilocks2::zero();
        for (int i = 0; i != 2; ++i) {
            for (int j = 0; j != 2; ++j) {
                sel_cha[p0] = i ? Goldilocks2::one() : Goldilocks2::zero();
                sel_cha[p1] = j ? Goldilocks2::one() : Goldilocks2::zero();
                sum += pcs_sel.open(sel_cha, sec_param);
            }
        }
        if (sum != Goldilocks2::one()) {
            std::cout << "❌ I.3 Proving selector is one-hot failed." << std::endl;
            throw;
        }

        // II. Forward.
        // II.1 Prover commits diff
        array<Goldilocks2::Element> diff(input.get_shape());
        for (int j = 0; j < img; ++j) {
            for (int k = 0; k < C; ++k) {
                auto diff_j_k = diff.view[j][k];
                for (int x = 0; x < N; ++x) {
                    for (int y = 0; y < N; ++y) {
                        diff(j, k, x, y) = output(j, k, x / 2, y / 2) - input(j, k, x, y);
                    }
                }
            }
        }
        auto pcs_diff = wit.pcs.at("pcs_diff" + _i);
        // II.2. Prover proves diff = expand(output) - input
        std::vector<Goldilocks2::Element> input_cha = random_vec_ext(logimg + logC + 2 * logN);
        std::vector<Goldilocks2::Element> extra_cha = {input_cha[p0], input_cha[p1]};
        std::vector<Goldilocks2::Element> output_cha;
        if (logN == 1) {
            output_cha = input_cha;
            output_cha[p0] = Goldilocks2::zero();
            output_cha[p1] = Goldilocks2::zero();
        } else {
            output_cha.reserve(input_cha.size() - 2);
            for (int j = 0; j < static_cast<int>(input_cha.size()); ++j) {
                if (j != p0 && j != p1) {
                    output_cha.push_back(input_cha[j]);
                }
            }
        }
        MLE_Eq eq_extra(extra_cha);
        Goldilocks2::Element factor = Goldilocks2::zero();
        for (int j = 0; j != 2; ++j)
            for (int k = 0; k != 2; ++k)
                factor += eq_extra.open({j ? Goldilocks2::one() : Goldilocks2::zero(),
                                        k ? Goldilocks2::one() : Goldilocks2::zero()},
                                        sec_param);
        if (pcs_diff.open(input_cha, sec_param) != factor * pcs_output.open(output_cha, sec_param) - pcs_input.open(input_cha, sec_param)) {
            std::cout << "❌ II.2 Proving diff = expand(output) - input failed." << std::endl;
            throw;
        }
        // II.3 Prover proves diff is non-negative
        std::vector<Goldilocks2::Element> input_ones(input.size(), Goldilocks2::one());
        MLE mle_input_ones = MLE(input_ones);
        // TODO 3. Check pcs_input_ones = 1
        signVerifier::resource sign_res_diff = *reinterpret_cast<signVerifier::resource*>(wit.res.at("sign_res_diff" + _i).get());
        signProver sign_prover_diff(diff.data, input_ones, scale, max_val * 2, false, rho_inv, lazy_logup_prover);
        if (!signVerifier::execute_sign_check(
            sign_prover_diff, 
            std::make_shared<lazy_pcs>(pcs_diff), 
            std::make_shared<MLE>(mle_input_ones),
            sec_param, 
            lazy_logup_verifier,
            sign_res_diff)) {
            std::cout << "❌ II.3 Proving diff is non-negative failed: sign proof fails." << std::endl;
            throw;
        }
        // II.4 Prover commits and proves sel_input = input * sel
        array<Goldilocks2::Element> sel_input(input.get_shape());
        for (int j = 0; j < img; ++j) {
            for (int k = 0; k < C; ++k) {
                for (int x = 0; x < N; ++x) {
                    for (int y = 0; y < N; ++y) {
                        sel_input(j, k, x, y) = input(j, k, x, y) * sel(j, k, x, y);
                    }
                }
            }
        }
        MultilinearPolynomial mle_sel_input(sel_input.view);
        auto pcs_sel_input = wit.pcs.at("pcs_sel_input" + _i);
        if (!prove_mle_product(mle_sel_input, mle_sel, input, pcs_sel_input, pcs_sel, pcs_input, sec_param)) {
            std::cout << "❌ II.4 Proving sel_input = input * sel failed." << std::endl;
            throw;
        }
        // II.5 Prover proves output = shrink(sel_input)
        if (!prove_pool_shrink(logimg, logC, logN, pcs_sel_input, pcs_output, sec_param)) {
            std::cout << "❌ II.5 Proving output = shrink(sel_input) failed." << std::endl;
            throw;
        }

        // III. Backward.
        // III.1 Prover commits and proves sel_d_input = d_input * sel
        array<Goldilocks2::Element> sel_d_input(input.get_shape());
        for (int j = 0; j < img; ++j) {
            for (int k = 0; k < C; ++k) {
                auto sel_d_input_j_k = sel_d_input.view[j][k];
                for (int x = 0; x < N; ++x) {
                    for (int y = 0; y < N; ++y) {
                        sel_d_input_j_k(x, y) = d_input(j, k, x, y) * sel(j, k, x, y);
                    }
                }
            }
        }
        auto pcs_sel_d_input = wit.pcs.at("pcs_sel_d_input" + _i);
        // auto pcs_sel_d_input = ligero_commit_base(sel_d_input.view, rho_inv);
        MultilinearPolynomial mle_sel_d_input(sel_d_input.view);
        if (!prove_mle_product(mle_sel_d_input, mle_sel, d_input,
            pcs_sel_d_input, pcs_sel, pcs_d_input, sec_param)) {
            std::cout << "❌ III.1 Proving sel_d_input = d_input * sel failed." << std::endl;
            throw;
        }
        // III.2 Prover proves d_ouput = shrink(sel_d_input)
        if (!prove_pool_shrink(logimg, logC, logN, pcs_sel_d_input, pcs_d_output, sec_param)) {
            std::cout << "❌ III.2 Proving d_output = shrink(sel_d_input) failed." << std::endl;
            throw;
        }
        // III.3 Prover proves 0 = d_input * rev_sel
        if (!prove_mle_product(mle_input_zeros, mle_d_input, mle_rev_sel,
            mle_input_zeros, pcs_d_input, pcs_rev_sel, sec_param)) {
            std::cout << "❌ III.3 Proving 0 = d_input * rev_sel failed." << std::endl;
            throw;
        }

        // IV. Check all TODO
        input_cha = random_vec_ext(logimg + logC + 2 * logN);
        if (pcs_sel.open(input_cha, sec_param) != Goldilocks2::one() - pcs_rev_sel.open(input_cha, sec_param)) {
            std::cout << "❌ IV.1 Proving pcs_sel + pcs_rev_sel = 1 failed." << std::endl;
            throw;
        }
    }
    return true;
}

CNN::layer_res pre_prove_softmax(const CNN::layer_info& layer,
    size_t scale, size_t max_val,
    size_t rho_inv, 
    lazyLogupProver* lazy_logup_prover,
    lazyLogupVerifier *lazy_logup_verifier,
    lazy_pcs_pool* pool)  {
    
    CNN::layer_res ret;
    
    const int batch = int(layer.input.shape(0));
    const int img = int(layer.input.shape(1));
    const int n = int(layer.input.shape(2));
    // const int logimg = find_ceiling_log2(img);
    const int logn = find_ceiling_log2(n);

    if (!is_power_of_2(img)) {
        throw std::invalid_argument("prove_softmax: Image num must be a power of 2");
    }
    
    for (int i = 0; i < batch; ++i) {
        std::string _i = "_" + std::to_string(i);
        auto pcs_input = layer.get_pcs_input(i);
        auto pcs_d_input = layer.get_pcs_d_input(i);
        auto pcs_label = layer.get_pcs_aux(i);

        auto input = layer.input[i]; // [img, n]
        auto d_input = layer.d_input[i]; // [img, n]
        auto label = layer.aux[i];


        auto mle_d_input = *layer.mle_d_input[i];
        auto mle_label = *layer.mle_aux[i];

        // 1. Scale input.
        std::vector<Goldilocks2::Element> input_copy(img * (1ull << logn));
        std::vector<size_t> new_shape = {size_t(img), 1ull << logn};
        input.copy_to(input_copy.data(), new_shape);
        auto [quo, rem] = get_quo_rem(input_copy, scale, true);
        MultilinearPolynomial mle_quo(quo);
        // auto pcs_quo = ligero_commit_base(mle_quo, rho_inv);
        // auto pcs_rem = ligero_commit_base(rem, rho_inv);
        ret.pcs["pcs_quo" + _i] =  commit_lazy_pcs(mle_quo, pool); // warm up
        ret.pcs["pcs_rem" + _i] = commit_lazy_pcs(rem, pool);

        // 2. Commit/Prove mask
        array<Goldilocks2::Element> mask(new_shape);
        for (int j = 0; j < img; ++j) {
            for (int k = 0; k < n; ++k) {
                mask(j, k) = Goldilocks2::one();
            }
        }
        MultilinearPolynomial mle_mask(mask.view);
        // auto pcs_mask = ligero_commit_base(mle_mask, rho_inv);
        ret.pcs["pcs_mask" + _i] = commit_lazy_pcs(mle_mask, pool);

        // 3. Commit onehot selector
        array_view<Goldilocks2::Element> view_quo(quo.data(), new_shape);
        std::vector<Goldilocks2::Element> max(img);
        array<Goldilocks2::Element> sel(new_shape), rev_sel(new_shape);
        for (int j = 0; j < img; ++j) {
            max[j] = view_quo(j, 0);
            int ind = 0;
            for (int k = 1; k < n; ++k) {
                if (Goldilocks2::toS64(view_quo(j, k)) > Goldilocks2::toS64(max[j])) {
                    max[j] = view_quo(j, k);
                    ind = k;
                }
            }
            sel(j, ind) = Goldilocks2::one();
            for (int k = 0; k < (1 << logn); ++k) {
                rev_sel(j, k) = Goldilocks2::one() - sel(j, k);
            }
        }
        MultilinearPolynomial mle_max(max), mle_sel(sel), mle_rev_sel(rev_sel);
        ret.pcs["pcs_max" + _i] = commit_lazy_pcs(mle_max, pool);
        ret.pcs["pcs_sel" + _i] = commit_lazy_pcs(mle_sel, pool);
        ret.pcs["pcs_rev_sel" + _i] = commit_lazy_pcs(mle_rev_sel, pool);
        // 3.1 Check sel is one-hot
        MultilinearPolynomial mle_prod_sel = mle_sel * mle_rev_sel;
        ret.pcs["pcs_prod_sel" + _i] = commit_lazy_pcs(mle_prod_sel, pool);
        // 3.2 Prove max = shrink(sel * quo)
        MultilinearPolynomial mle_sel_quo = mle_sel * mle_quo;
        auto pcs_sel_quo = ligero_commit_base(mle_sel_quo, rho_inv);
        ret.pcs["pcs_sel_quo" + _i] = commit_lazy_pcs(mle_sel_quo, pool);
        // 4. Commit/Prove diff_masked = mask * (max - quo)
        array<Goldilocks2::Element> diff(new_shape), diff_masked(new_shape);
        for (int j = 0; j < img; ++j) {
            for (int k = 0; k < (1 << logn); ++k) {
                diff(j, k) = max[j] - view_quo(j, k);
                diff_masked(j, k) = mask(j, k) * diff(j, k);
            }
        }
        MultilinearPolynomial mle_diff(diff.view), mle_diff_masked(diff_masked.view);
        ret.pcs["pcs_diff" + _i] = commit_lazy_pcs(mle_diff, pool);
        ret.pcs["pcs_diff_masked" + _i] = commit_lazy_pcs(mle_diff_masked, pool);

        // 5. Prove diff_masked >= 0
        std::vector<Goldilocks2::Element> ones(diff_masked.data.size(), Goldilocks2::one());
        MLE mle_ones = MLE(ones);
        // auto pcs_ones = ligero_commit_base(ones, rho_inv);
        // TODO: Check pcs_ones = 1
        signProver sign_prover_diff_masked(diff_masked.data, ones, scale, 2 * max_val, false, rho_inv, lazy_logup_prover);
        auto sign_diff_masked_res = signVerifier::pre_execute_sign_check(sign_prover_diff_masked, pool, lazy_logup_verifier);
        ret.res["sign_diff_masked_res" + _i] = std::make_shared<signVerifier::resource>(sign_diff_masked_res);


        // 6. Prove diff_masked contains zero
        
        // 7. Prove exp_diff = exp(diff_masked)
        std::vector<size_t> exp_diff(img * (1 << logn)), diff_masked_vec(img * (1 << logn));
        for (size_t j = 0; j != exp_diff.size(); ++j) {
            diff_masked_vec[j] = diff_masked.data[j][0].fe;
        }
        exp_diff = eVerifier::get_exp_inv(diff_masked_vec, scale, rho_inv);
        MultilinearPolynomial mle_exp_diff(exp_diff);
        ret.pcs["pcs_exp_diff" + _i] = commit_lazy_pcs(mle_exp_diff, pool);

        eProver e_prover(diff_masked_vec, exp_diff, scale, max_val, rho_inv, lazy_logup_prover);
        ret.res["e_res" + _i] = std::make_shared<eVerifier::resource>(eVerifier::pre_execute_check(e_prover, lazy_logup_verifier, pool));

        // 8. Prove exp_diff_masked = mask * exp_diff
        array<Goldilocks2::Element> exp_diff_masked(new_shape);
        for (int j = 0; j < img; ++j) {
            for (int k = 0; k < (1 << logn); ++k) {
                exp_diff_masked(j, k) = mask(j, k) * Goldilocks2::fromU64(exp_diff[j * (1 << logn) + k]);
            }
        }
        MultilinearPolynomial mle_exp_diff_masked(exp_diff_masked.view);
        ret.pcs["pcs_exp_diff_masked" + _i] = commit_lazy_pcs(mle_exp_diff_masked, pool);

        
        // 8. Commit/Prove sum
        std::vector<Goldilocks2::Element> sum(img);
        for (int j = 0; j != img; ++j) {
            for (int k = 0; k != (1 << logn); ++k) {
                sum[j] += exp_diff_masked(j, k);
            }
        }
        MultilinearPolynomial mle_sum(sum);
        ret.pcs["pcs_sum" + _i] = commit_lazy_pcs(mle_sum, pool);
        
        array<Goldilocks2::Element> expand_sum(new_shape);
        for (int j = 0; j != img; ++j) {
            for (int k = 0; k != (1 << logn); ++k) {
                expand_sum(j, k) = sum[j];
            }
        }
        MultilinearPolynomial mle_expand_sum(expand_sum.view);
        // auto pcs_expand_sum = ligero_commit_base(mle_expand_sum, rho_inv);
        ret.pcs["pcs_expand_sum" + _i] = commit_lazy_pcs(mle_expand_sum, pool);
        // 9. Prove d_quo = exp_diff_masked * scale / expand_sum
        array<Goldilocks2::Element> d_quo(new_shape), d_rem(new_shape);
        for (int j = 0; j < img; ++j) {
            for (int k = 0; k < (1 << logn); ++k) {
                d_quo(j, k) = Goldilocks2::fromS64(Goldilocks2::toS64(exp_diff_masked(j, k)) * int64_t(scale) /
                    Goldilocks2::toS64(expand_sum(j, k)));
                d_rem(j, k) = exp_diff_masked(j, k) * Goldilocks2::fromU64(scale) - d_quo(j, k) * expand_sum(j, k);
            }
        }
        MultilinearPolynomial mle_d_quo(d_quo.view), mle_d_rem(d_rem.view);
        // auto pcs_d_quo = ligero_commit_base(mle_d_quo, rho_inv);
        // auto pcs_d_rem = ligero_commit_base(mle_d_rem, rho_inv);
        ret.pcs["pcs_d_quo" + _i] = commit_lazy_pcs(mle_d_quo, pool);
        ret.pcs["pcs_d_rem" + _i] = commit_lazy_pcs(mle_d_rem, pool);
        MultilinearPolynomial mle_prod = mle_d_quo * mle_expand_sum;
        // auto pcs_prod = ligero_commit_base(mle_prod, rho_inv);
        ret.pcs["pcs_prod" + _i] = commit_lazy_pcs(mle_prod, pool);


        // 9.2 d_rem - sum < 0
        MultilinearPolynomial mle_d_rem_sum = mle_d_rem - mle_expand_sum;
        std::vector<Goldilocks2::Element> input_zeros(img * (1 << logn));
        // auto pcs_d_rem_sum = ligero_commit_base(mle_d_rem_sum, rho_inv);
        ret.pcs["pcs_d_rem_sum" + _i] = commit_lazy_pcs(mle_d_rem_sum, pool);
        
        
        signProver sign_prover_d_rem_sum(mle_d_rem_sum.get_eval_table(), input_zeros, scale, 2 * max_val, false, rho_inv, lazy_logup_prover);
        auto sign_d_rem_sum_res = signVerifier::pre_execute_sign_check(sign_prover_d_rem_sum, pool, lazy_logup_verifier);
        ret.res["sign_d_rem_sum_res" + _i] = std::make_shared<signVerifier::resource>(sign_d_rem_sum_res);
        // 10. Prove d_input = (d_quo - label * scale) / img
        MultilinearPolynomial mle_d_delta = mle_d_quo - mle_label * scale;
        // auto pcs_d_delta = ligero_commit_base(mle_d_delta, rho_inv);
        ret.pcs["pcs_d_delta" + _i] = commit_lazy_pcs(mle_d_delta, pool);
        auto d_input_rem = get_rem(mle_d_delta.get_eval_table(), img, mle_d_input.get_eval_table(), true);
        // auto pcs_d_input_rem = ligero_commit_base(d_input_rem, rho_inv);
        ret.pcs["pcs_d_input_rem" + _i] = commit_lazy_pcs(d_input_rem, pool);
    }
    return ret;
  
}

bool prove_softmax(const CNN::layer_info& layer,
    size_t scale, size_t max_val,
    size_t rho_inv, size_t sec_param,
    CNN::layer_res& wit,
    lazyLogupProver* lazy_logup_prover, lazyLogupVerifier *lazy_logup_verifier) {
    
    startCounter counter("softmax_proof");

    const int batch = int(layer.input.shape(0));
    const int img = int(layer.input.shape(1));
    const int n = int(layer.input.shape(2));
    const int logimg = find_ceiling_log2(img);
    const int logn = find_ceiling_log2(n);

    if (!is_power_of_2(img)) {
        throw std::invalid_argument("prove_softmax: Image num must be a power of 2");
    }
    
    for (int i = 0; i < batch; ++i) {
        std::string _i = "_" + std::to_string(i);
        auto pcs_input = layer.get_pcs_input(i);
        auto pcs_d_input = layer.get_pcs_d_input(i);
        auto pcs_label = layer.get_pcs_aux(i);

        auto input = layer.input[i]; // [img, n]
        auto d_input = layer.d_input[i]; // [img, n]
        auto label = layer.aux[i];


        auto mle_d_input = *layer.mle_d_input[i];
        auto mle_label = *layer.mle_aux[i];

        // 1. Scale input.
        std::vector<Goldilocks2::Element> input_copy(img * (1ull << logn));
        std::vector<size_t> new_shape = {size_t(img), 1ull << logn};
        input.copy_to(input_copy.data(), new_shape);
        auto [quo, rem] = get_quo_rem(input_copy, scale, true);
        MultilinearPolynomial mle_quo(quo);
        // auto pcs_quo = ligero_commit_base(mle_quo, rho_inv);
        // auto pcs_rem = ligero_commit_base(rem, rho_inv);
        auto pcs_quo = wit.pcs.at("pcs_quo" + _i);
        auto pcs_rem = wit.pcs.at("pcs_rem" + _i);
        divProver div_prover(input_copy, quo, rem, scale, true, rho_inv, lazy_logup_prover);
        if (!divVerifier::execute_div_check(div_prover,
            std::make_shared<lazy_pcs>(pcs_input),
            std::make_shared<lazy_pcs>(pcs_quo),
            std::make_shared<lazy_pcs>(pcs_rem),
            sec_param,
            lazy_logup_verifier)) {
            std::cout << "❌ Proving softmax scale input failed." << std::endl;
            return false;
        }
        // 2. Commit/Prove mask
        array<Goldilocks2::Element> mask(new_shape);
        for (int j = 0; j < img; ++j) {
            for (int k = 0; k < n; ++k) {
                mask(j, k) = Goldilocks2::one();
            }
        }
        MultilinearPolynomial mle_mask(mask.view);
        // auto pcs_mask = ligero_commit_base(mle_mask, rho_inv);
        auto pcs_mask = wit.pcs.at("pcs_mask" + _i);
        std::vector<Goldilocks2::Element> mask_cha(logimg + logn);
        std::vector<Goldilocks2::Element> first_cha(mask_cha.begin(), mask_cha.begin() + logimg);
        std::vector<Goldilocks2::Element> second_cha(mask_cha.begin() + logimg, mask_cha.end());
        Goldilocks2::Element factor = Goldilocks2::zero();
        MLE_Eq mle_eq(first_cha);
        factor = mle_eq.get_sum();
        if (pcs_mask.open(mask_cha, sec_param) != factor * MLE_Pow(Goldilocks2::one(), logn, n - 1).evaluate(second_cha)) {
            std::cout << "❌ Proving softmax mask failed." << std::endl;
            return false;
        }
        // 3. Commit onehot selector
        array_view<Goldilocks2::Element> view_quo(quo.data(), new_shape);
        std::vector<Goldilocks2::Element> max(img);
        array<Goldilocks2::Element> sel(new_shape), rev_sel(new_shape);
        for (int j = 0; j < img; ++j) {
            max[j] = view_quo(j, 0);
            int ind = 0;
            for (int k = 1; k < n; ++k) {
                if (Goldilocks2::toS64(view_quo(j, k)) > Goldilocks2::toS64(max[j])) {
                    max[j] = view_quo(j, k);
                    ind = k;
                }
            }
            sel(j, ind) = Goldilocks2::one();
            for (int k = 0; k < (1 << logn); ++k) {
                rev_sel(j, k) = Goldilocks2::one() - sel(j, k);
            }
        }
        MultilinearPolynomial mle_max(max), mle_sel(sel), mle_rev_sel(rev_sel);
        // auto pcs_max = ligero_commit_base(mle_max, rho_inv);
        // auto pcs_sel = ligero_commit_base(mle_sel, rho_inv);
        // auto pcs_rev_sel = ligero_commit_base(mle_rev_sel, rho_inv);
        auto pcs_max = wit.pcs.at("pcs_max" + _i);
        auto pcs_sel = wit.pcs.at("pcs_sel" + _i);
        auto pcs_rev_sel = wit.pcs.at("pcs_rev_sel" + _i);

        // 3.1 Check sel is one-hot
        auto sel_cha = random_vec_ext(logimg + logn);
        MultilinearPolynomial mle_prod_sel = mle_sel * mle_rev_sel;
        // auto pcs_prod_sel = ligero_commit_base(mle_prod_sel, rho_inv);
        auto pcs_prod_sel = wit.pcs.at("pcs_prod_sel" + _i);
        if (pcs_sel.open(sel_cha, sec_param) != Goldilocks2::one() - pcs_rev_sel.open(sel_cha, sec_param)) {
            std::cout << "❌ Proving softmax sel + rev_sel = 1 failed." << std::endl;
            return false;
        }
        if (!prove_mle_product(mle_prod_sel, mle_sel, mle_rev_sel, pcs_prod_sel, pcs_sel, pcs_rev_sel, sec_param)) {
            std::cout << "❌ Proving softmax sel * rev_sel = sel * rev_sel failed." << std::endl;
            return false;
        }
        if (pcs_prod_sel.open(sel_cha, sec_param) != Goldilocks2::zero()) {
            std::cout << "❌ Proving softmax sel boolean failed." << std::endl;
            return false;
        }
        // 3.2 Prove max = shrink(sel * quo)
        MultilinearPolynomial mle_sel_quo = mle_sel * mle_quo;
        // auto pcs_sel_quo = ligero_commit_base(mle_sel_quo, rho_inv);
        auto pcs_sel_quo = wit.pcs.at("pcs_sel_quo" + _i);
        if (!prove_mle_product(mle_sel_quo, mle_sel, mle_quo, pcs_sel_quo, pcs_sel, pcs_quo, sec_param)) {
            std::cout << "❌ Proving softmax sel_quo = sel * quo failed." << std::endl;
            return false;
        }
        auto max_cha = random_vec_ext(logimg);
        auto mle_fix_sel_quo = mle_sel_quo;
        mle_fix_sel_quo.fix(0, max_cha);
        auto max_cha_val = pcs_max.open(max_cha, sec_param);
        sProver spr(mle_fix_sel_quo);
        if (!sVerifier::partial_sumcheck(spr, max_cha, max_cha_val, sec_param) ||
            pcs_sel_quo.open(max_cha, sec_param) != max_cha_val) {
            std::cout << "❌ Proving softmax max = shrink(sel * quo) failed." << std::endl;
            return false;
        }
        // 4. Commit/Prove diff_masked = mask * (max - quo)
        array<Goldilocks2::Element> diff(new_shape), diff_masked(new_shape);
        for (int j = 0; j < img; ++j) {
            for (int k = 0; k < (1 << logn); ++k) {
                diff(j, k) = max[j] - view_quo(j, k);
                diff_masked(j, k) = mask(j, k) * diff(j, k);
            }
        }
        MultilinearPolynomial mle_diff(diff.view), mle_diff_masked(diff_masked.view);
        // auto pcs_diff = ligero_commit_base(mle_diff, rho_inv);
        // auto pcs_diff_masked = ligero_commit_base(mle_diff_masked, rho_inv);
        auto pcs_diff = wit.pcs.at("pcs_diff" + _i);
        auto pcs_diff_masked = std::make_shared<lazy_pcs>(wit.pcs.at("pcs_diff_masked" + _i));
        auto diff_cha = random_vec_ext(logimg + logn);
        std::vector<Goldilocks2::Element> diff_first_cha(diff_cha.begin(), diff_cha.begin() + logimg),
                diff_second_cha(diff_cha.begin() + logimg, diff_cha.end());

        if (pcs_diff.open(diff_cha, sec_param) != pcs_max.open(diff_first_cha, sec_param) * MLE_Eq(diff_second_cha).get_sum()
                 - pcs_quo.open(diff_cha, sec_param)) {
            std::cout << "❌ Proving softmax diff = max - quo failed." << std::endl;
            return false;
        }
        if (!prove_mle_product(mle_diff_masked, mle_mask, mle_diff, *pcs_diff_masked, pcs_mask, pcs_diff, sec_param)) {
            std::cout << "❌ Proving softmax diff_masked = mask * diff failed." << std::endl;
            return false;
        }
        // 5. Prove diff_masked >= 0
        std::vector<Goldilocks2::Element> ones(diff_masked.data.size(), Goldilocks2::one());
        MLE mle_ones = MLE(ones);
        // auto pcs_ones = ligero_commit_base(ones, rho_inv);
        // TODO: Check pcs_ones = 1
        auto sign_res = *reinterpret_cast<signVerifier::resource*>(wit.res.at("sign_diff_masked_res" + _i).get());
        signProver sign_prover_diff_masked(diff_masked.data, ones, scale, 2 * max_val, false, rho_inv, lazy_logup_prover);
        if (!signVerifier::execute_sign_check(
            sign_prover_diff_masked, 
            pcs_diff_masked, 
            std::make_shared<MLE>(mle_ones), 
            sec_param,
            lazy_logup_verifier,
            sign_res)) {

            std::cout << "❌ Proving softmax diff_masked >= 0 failed: sign proof fails." << std::endl;
            return false;
        }
        // 6. Prove diff_masked contains zero
        std::vector<Goldilocks2::Element> zeros(img, Goldilocks2::zero());
        std::shared_ptr<MLE> mle_zeros = std::make_shared<MLE>(zeros);
        // auto pcs_zeros = ligero_commit_base(mle_zeros, rho_inv);
        // TODO: Check pcs_zeros = 0
        prodProver prod_prover(mle_diff_masked, *mle_zeros, logn, rho_inv);
        if (!prodVerifier::execute_prod_check(prod_prover, {logimg + logn, pcs_diff_masked.get()}, {logimg, mle_zeros.get()}, sec_param)) {
            std::cout << "❌ Proving softmax diff_masked contains zero failed." << std::endl;
            return false;
        }
        // 7. Prove exp_diff = exp(diff_masked)
        std::vector<size_t> exp_diff(img * (1 << logn)), diff_masked_vec(img * (1 << logn));
        for (size_t j = 0; j != exp_diff.size(); ++j) {
            diff_masked_vec[j] = diff_masked.data[j][0].fe;
        }
        exp_diff = eVerifier::get_exp_inv(diff_masked_vec, scale, rho_inv);
        MultilinearPolynomial mle_exp_diff(exp_diff);
        // auto pcs_exp_diff = ligero_commit_base(mle_exp_diff, rho_inv);
        auto pcs_exp_diff = std::make_shared<lazy_pcs>(wit.pcs.at("pcs_exp_diff" + _i));
        eProver e_prover(diff_masked_vec, exp_diff, scale, max_val, rho_inv, lazy_logup_prover);
        eVerifier::resource e_res = *reinterpret_cast<eVerifier::resource*>(wit.res.at("e_res" + _i).get());
        if (!eVerifier::execute_check(e_prover, pcs_diff_masked, pcs_exp_diff, sec_param, lazy_logup_verifier, e_res)) {
            std::cout << "❌ Proving softmax exp(diff_masked) * scale failed." << std::endl;
            return false;
        }
        // 8. Prove exp_diff_masked = mask * exp_diff
        array<Goldilocks2::Element> exp_diff_masked(new_shape);
        for (int j = 0; j < img; ++j) {
            for (int k = 0; k < (1 << logn); ++k) {
                exp_diff_masked(j, k) = mask(j, k) * Goldilocks2::fromU64(exp_diff[j * (1 << logn) + k]);
            }
        }
        MultilinearPolynomial mle_exp_diff_masked(exp_diff_masked.view);
        // auto pcs_exp_diff_masked = ligero_commit_base(mle_exp_diff_masked, rho_inv);
        auto pcs_exp_diff_masked = wit.pcs.at("pcs_exp_diff_masked" + _i);
        if (!prove_mle_product(mle_exp_diff_masked, mle_mask, mle_exp_diff,
            pcs_exp_diff_masked, pcs_mask, *pcs_exp_diff, sec_param)) {
            std::cout << "❌ Proving softmax exp_diff_masked = mask * exp_diff failed." << std::endl;
            return false;
        }
        // 8. Commit/Prove sum
        std::vector<Goldilocks2::Element> sum(img);
        for (int j = 0; j != img; ++j) {
            for (int k = 0; k != (1 << logn); ++k) {
                sum[j] += exp_diff_masked(j, k);
            }
        }
        MultilinearPolynomial mle_sum(sum);
        // auto pcs_sum = ligero_commit_base(mle_sum, rho_inv);
        auto pcs_sum = wit.pcs.at("pcs_sum" + _i);
        std::vector<Goldilocks2::Element> sum_cha = random_vec_ext(logimg);
        auto mle_exp_diff_masked_fixed = mle_exp_diff_masked;
        mle_exp_diff_masked_fixed.fix(0, sum_cha);
        sProver sum_prover(mle_exp_diff_masked_fixed);
        Goldilocks2::Element sum_cha_val = pcs_sum.open(sum_cha, sec_param);
        if (!sVerifier::partial_sumcheck(sum_prover, sum_cha, sum_cha_val, sec_param)
            || pcs_exp_diff_masked.open(sum_cha, sec_param) != sum_cha_val) {
            std::cout << "❌ Proving softmax sum failed." << std::endl;
            return false;
        }
        array<Goldilocks2::Element> expand_sum(new_shape);
        for (int j = 0; j != img; ++j) {
            for (int k = 0; k != (1 << logn); ++k) {
                expand_sum(j, k) = sum[j];
            }
        }
        MultilinearPolynomial mle_expand_sum(expand_sum.view);
        // auto pcs_expand_sum = ligero_commit_base(mle_expand_sum, rho_inv);
        auto pcs_expand_sum = wit.pcs.at("pcs_expand_sum" + _i);
        sum_cha = random_vec_ext(logimg);
        auto extra_cha = random_vec_ext(logn);
        if (pcs_expand_sum.open(combine_challenges(sum_cha, extra_cha), sec_param) != pcs_sum.open(sum_cha, sec_param)
                * MLE_Eq(extra_cha).get_sum()) {
            std::cout << "❌ Proving softmax expand_sum = expand(sum) failed." << std::endl;
            return false;
        }
        // 9. Prove d_quo = exp_diff_masked * scale / expand_sum
        array<Goldilocks2::Element> d_quo(new_shape), d_rem(new_shape);
        for (int j = 0; j < img; ++j) {
            for (int k = 0; k < (1 << logn); ++k) {
                d_quo(j, k) = Goldilocks2::fromS64(Goldilocks2::toS64(exp_diff_masked(j, k)) * int64_t(scale) /
                    Goldilocks2::toS64(expand_sum(j, k)));
                d_rem(j, k) = exp_diff_masked(j, k) * Goldilocks2::fromU64(scale) - d_quo(j, k) * expand_sum(j, k);
            }
        }
        MultilinearPolynomial mle_d_quo(d_quo.view), mle_d_rem(d_rem.view);
        // auto pcs_d_quo = ligero_commit_base(mle_d_quo, rho_inv);
        // auto pcs_d_rem = ligero_commit_base(mle_d_rem, rho_inv);
        auto pcs_d_quo = wit.pcs.at("pcs_d_quo" + _i);
        auto pcs_d_rem = wit.pcs.at("pcs_d_rem" + _i);
        MultilinearPolynomial mle_prod = mle_d_quo * mle_expand_sum;
        // auto pcs_prod = ligero_commit_base(mle_prod, rho_inv);
        auto pcs_prod = wit.pcs.at("pcs_prod" + _i);
        if (!prove_mle_product(mle_prod, mle_d_quo, mle_expand_sum,
            pcs_prod, pcs_d_quo, pcs_expand_sum, sec_param)) {
            std::cout << "❌ Proving softmax mle_prod = mle_d_quo * mle_expand_sum failed." << std::endl;
            return false;
        }
        auto d_div_cha = random_vec_ext(logimg + logn);
        if (pcs_exp_diff_masked.open(d_div_cha, sec_param) * Goldilocks2::fromU64(scale) != pcs_prod.open(d_div_cha, sec_param)
            + pcs_d_rem.open(d_div_cha, sec_param)) {
            std::cout << "❌ Proving softmax d_div = d_quo / expand_sum failed." << std::endl;
            return false;
        }
        // 9.2 d_rem - sum < 0
        MultilinearPolynomial mle_d_rem_sum = mle_d_rem - mle_expand_sum;
        // auto pcs_d_rem_sum = ligero_commit_base(mle_d_rem_sum, rho_inv);
        auto pcs_d_rem_sum = wit.pcs.at("pcs_d_rem_sum" + _i);
        auto d_rem_sum_cha = random_vec_ext(logimg + logn);
        if (pcs_d_rem_sum.open(d_rem_sum_cha, sec_param) != pcs_d_rem.open(d_rem_sum_cha, sec_param) - pcs_expand_sum.open(d_rem_sum_cha, sec_param)) {
            std::cout << "❌ Proving softmax d_rem_sum_cha = d_rem - sum failed." << std::endl;
            return false;
        }
        std::vector<Goldilocks2::Element> input_zeros(img * (1 << logn));
        std::shared_ptr<MultilinearPolynomial> mle_input_zeros = std::make_shared<MLE>(input_zeros);
        signProver sign_prover_d_rem_sum(mle_d_rem_sum.get_eval_table(), input_zeros, scale, 2 * max_val, false, rho_inv, lazy_logup_prover);
        auto sign_d_rem_sum_res = *reinterpret_cast<signVerifier::resource*>(wit.res.at("sign_d_rem_sum_res" + _i).get());
        if (!signVerifier::execute_sign_check(
            sign_prover_d_rem_sum, 
            std::make_shared<lazy_pcs>(pcs_d_rem_sum), 
            mle_input_zeros, 
            sec_param,
            lazy_logup_verifier)) {
            std::cout << "❌ Proving softmax d_rem - sum < 0 failed: sign proof fails." << std::endl;
            return false;
        }
        // 10. Prove d_input = (d_quo - label * scale) / img
        MultilinearPolynomial mle_d_delta = mle_d_quo - mle_label * scale;
        // auto pcs_d_delta = ligero_commit_base(mle_d_delta, rho_inv);
        auto pcs_d_delta = wit.pcs.at("pcs_d_delta" + _i);
        auto delta_cha = random_vec_ext(logimg + logn);
        if (pcs_d_delta.open(delta_cha, sec_param) != pcs_d_quo.open(delta_cha, sec_param) - Goldilocks2::fromU64(scale) * pcs_label.open(delta_cha, sec_param)) {
            std::cout << "❌ Proving softmax d_delta = (d_quo - label * scale) failed." << std::endl;
            return false;
        }
        auto d_input_rem = get_rem(mle_d_delta.get_eval_table(), img, mle_d_input.get_eval_table(), true);
        // auto pcs_d_input_rem = ligero_commit_base(d_input_rem, rho_inv);
        auto pcs_d_input_rem = wit.pcs.at("pcs_d_input_rem" + _i);
        divProver div_prover_d_input(mle_d_delta.get_eval_table(), mle_d_input.get_eval_table(), d_input_rem, img, true, rho_inv, lazy_logup_prover);
        if (!divVerifier::execute_div_check(div_prover_d_input, 
            std::make_shared<lazy_pcs>(pcs_d_delta), 
            std::make_shared<lazy_pcs>(pcs_d_input), 
            std::make_shared<lazy_pcs>(pcs_d_input_rem), 
            sec_param,
            lazy_logup_verifier)) {
            std::cout << "❌ Proving softmax d_input = (d_quo - sel * scale) / n failed." << std::endl;
            return false;
        }
    }
    return true;
}

bool prove_flat_layer(const CNN::layer_info& layer, size_t rho_inv, size_t sec_param) {

    startCounter counter("flat_proof");

    const int batch = layer.input.shape(0);
    const int img = layer.input.shape(1);
    const int C = layer.input.shape(2);
    const int logimg = find_ceiling_log2(img);
    const int logC = find_ceiling_log2(C);
    if (layer.input.shape(3) != 1 || layer.input.shape(4) != 1) {
        throw std::invalid_argument("prove_flat_layer: Input shape must be [batch, img, C, 1, 1]");
    }
    bool ret = true;
    #pragma omp parallel for
    for (int i = 0; i != batch; ++i) {
        auto pcs_input = layer.get_pcs_input(i);
        auto pcs_output = layer.get_pcs_output(i);
        auto pcs_d_input = layer.get_pcs_d_input(i);
        auto pcs_d_output = layer.get_pcs_d_output(i);

        auto output_cha = random_vec_ext(logimg + logC);
        auto input_cha = output_cha;
        input_cha.push_back(Goldilocks2::zero());
        input_cha.push_back(Goldilocks2::zero());
        // Check value
        if (pcs_input.open(input_cha, sec_param) != pcs_output.open(output_cha, sec_param)) {
            std::cout << "❌ Proving input = output failed." << std::endl;
            ret = false;
        }
        if (pcs_d_input.open(input_cha, sec_param) != pcs_d_output.open(output_cha, sec_param)) {
            std::cout << "❌ Proving d_input = d_output failed." << std::endl;
            ret = false;
        }
        // Check zero
        for (int j = 1; j != 4; ++j) {
            input_cha[logimg + logC] = (j & 1) ? Goldilocks2::one() : Goldilocks2::zero();
            input_cha[logimg + logC + 1] = (j >> 1) ? Goldilocks2::one() : Goldilocks2::zero();
            if (pcs_input.open(input_cha, sec_param) != Goldilocks2::zero()) {
                std::cout << "❌ Proving output is zero failed." << std::endl;
                ret = false;
            }
            if (pcs_d_input.open(input_cha, sec_param) != Goldilocks2::zero()) {
                std::cout << "❌ Proving d_output is zero failed." << std::endl;
                ret = false;
            }
        }
    }
    return ret;
}
