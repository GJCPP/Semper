#include <cassert>

#include <chrono>

#include "VCG16.h"
#include "VCG16_check.h"
#include "VCG16_proof.h"
#include "conv_check.h"
#include "mat_check.h"
#include "util.h"

bool prove_conv(
    int N, int img,
    size_t C, size_t D, size_t n, size_t m, size_t padding,
    open_param oX, // [N, C, n, n]
    open_param oW, // [C, D, m, m]
    open_param oY, // [N, D, on, on]
    const array_view<Goldilocks2::Element>& X, // [C, n, n]
    const array_view<Goldilocks2::Element>& W, // [C, D, m, m]
    const array_view<Goldilocks2::Element>& Y, // [D, on, on]
    size_t rho_inv, size_t sec_param) { 

    auto prover = make_conv2_prover(C, D, n, m, padding, X, W, Y);

    // if (!prover.triple.check()) {
    //     std::cout << "❌ Conv triple check failed." << std::endl;
    //     return false;
    // }

    int l = find_ceiling_log2(N);
    std::vector<Goldilocks2::Element> pre(l);
    for (int i = 0; i < l; ++i) {
        pre[l - i - 1] = ((img >> i) & 1) ? Goldilocks2::one() : Goldilocks2::zero();
    }

    oX = oX.fix(pre);
    oY = oY.fix(pre);
    oW.rev[2] = oW.rev[2] ^ true;
    oW.rev[3] = oW.rev[3] ^ true;

    if (!convVerifier::execute_convcheck_2d(prover, oX, oW, oY, rho_inv, sec_param, true)) {
        return false;
    }
    return true;
}

bool prove_conv(
    int padding,
    const oracle* pcs_input, const oracle* pcs_weight, const oracle* pcs_output,
    const array_view<Goldilocks2::Element>& X, // [N, IC, in, in]
    const array_view<Goldilocks2::Element>& W, // [OC, IC, m, m]
    const array_view<Goldilocks2::Element>& Y, // [N, OC, on, on]
    bool pad_right_bottom,
    size_t rho_inv, size_t sec_param) {

    const int N = X.shape(0), IC = X.shape(1), in = X.shape(2), OC = Y.shape(1), on = Y.shape(2);
    const int m = W.shape(2);

    assert(X.shape(0) == N);
    assert(X.shape(1) == IC);
    assert(X.shape(2) == in);
    assert(X.shape(3) == in);
    assert(W.shape(0) == OC);
    assert(W.shape(1) == IC);
    assert(W.shape(2) == m);
    assert(W.shape(3) == m);
    assert(Y.shape(0) == N);
    assert(Y.shape(1) == OC);
    assert(Y.shape(2) == on);
    assert(Y.shape(3) == on);
    assert(on == in + 2 * padding - m + 1);


    auto& iW = W;
    double pad_time = 0, prove_time = 0;
    for (int j = 0; j < N; ++j) {
        std::cout << j << " ";
        std::cout.flush();

        auto iX = X[j]; // [C, n, n]
        auto iY = Y[j]; // [OC, on, on]
        auto& pX = iX;
        array<Goldilocks2::Element> pW; // [OC, IC, m, m]
        array<Goldilocks2::Element> pY; // [OC, on, on]
        size_t new_m, new_padding;

        auto start_pad = std::chrono::high_resolution_clock::now();

        pad_weights(IC, OC, in, m, padding,
            iX,
            iW,
            iY,
            pW, pY, new_m, new_padding, pad_right_bottom);

        // if (!random_check_conv(X, W, Y, padding, 1000)) {
        //     std::cout << "❌ Random check failed." << std::endl;
        //     return false;
        // } else {
        //     std::cout << "✅ Random check passed." << std::endl;
        // }

        // if (!random_check_single_conv(pX, pW, pY, new_padding, 1000)) {
        //     std::cout << "❌ Random single check failed." << std::endl;
        //     return false;
        // } else {
        //     std::cout << "✅ Random single check passed." << std::endl;
        // }



        std::chrono::duration<double> elapsed_pad = std::chrono::high_resolution_clock::now() - start_pad;

        pad_time += elapsed_pad.count();
        // std::cout << "pad_weights time: " << elapsed_pad.count() << " seconds" << std::endl;
        start_pad = std::chrono::high_resolution_clock::now();

        pW.view.swap_dim(0, 1); // [IC, OC, m, m]
        
        open_param oX(X, pcs_input);
        open_param oW(pW, pcs_weight);
        open_param oY(Y, pcs_output);

        if (!prove_conv(N, j, IC, OC, in, new_m, new_padding,
            oX, oW, oY,
            pX, pW, pY, rho_inv, sec_param)) return false;
        std::chrono::duration<double> elapsed_prove = std::chrono::high_resolution_clock::now() - start_pad;
        // std::cout << "prove_conv time: " << elapsed_prove.count() << " seconds" << std::endl;
        prove_time += elapsed_prove.count();
    }
    std::cout << std::endl << "pad time = " << pad_time << "s, prove time = " << prove_time << "s" << std::endl;

    return true;
}

bool prove_conv_forward(const VCG16::layer_info& layer, int padding, size_t rho_inv, size_t sec_param) {
    int bat = int(layer.input.shape(0));
    int img = int(layer.input.shape(1));
    size_t C = layer.input.shape(2);
    size_t D = layer.output.shape(2);
    size_t n = layer.input.shape(3);
    size_t m = layer.weight.shape(3);

    for (int i = 0; i < bat; ++i) {
        double pad_time = 0, prove_time = 0;
        auto pcs_input = layer.get_pcs_input(i);
        auto pcs_weight = layer.get_pcs_weight(i);
        auto pcs_output = layer.get_pcs_output(i);

        if (!prove_conv(padding,
            pcs_input.get(), pcs_weight.get(), pcs_output.get(),
            layer.input[i], layer.weight[i], layer.output[i],
            true,
            rho_inv, sec_param)) return false;
    }
    return true;
}

bool prove_conv_backward_dW(const VCG16::layer_info& layer, int padding, size_t rho_inv, size_t sec_param) {
    int bat = int(layer.input.shape(0));
    int img = int(layer.input.shape(1));
    size_t C = layer.input.shape(2);
    size_t D = layer.output.shape(2);
    size_t n = layer.input.shape(3);
    size_t m = layer.weight.shape(3);

    for (int i = 0; i < bat; ++i) {
        double pad_time = 0, prove_time = 0;
        auto pcs_input = layer.get_pcs_input(i);
        auto pcs_d_output = layer.get_pcs_d_output(i);
        auto pcs_d_weight = layer.get_pcs_d_weight(i);
        auto X = layer.input[i]; // [N, C, n, n]
        auto dY = layer.d_output[i]; // [N, D, on, on]
        auto dW = layer.d_weight[i]; // [D, C, m, m]

        X.swap_dim(0, 1); // [C, N, n, n]
        dY.swap_dim(0, 1); // [D, N, on, on]
        dW.swap_dim(0, 1); // [C, D, m, m]

        if (!prove_conv(padding,
            pcs_input.get(), pcs_d_output.get(), pcs_d_weight.get(),
            X, dY, dW,
            true,
            rho_inv, sec_param)) return false;
    }
    return true;
}

bool prove_conv_backward_dX(const VCG16::layer_info& layer, int padding, size_t rho_inv, size_t sec_param) {
    int bat = int(layer.input.shape(0));
    int img = int(layer.input.shape(1));
    size_t C = layer.input.shape(2);
    size_t D = layer.output.shape(2);
    size_t n = layer.input.shape(3);
    size_t m = layer.weight.shape(3);

    for (int i = 0; i < bat; ++i) {
        auto pcs_d_output = layer.get_pcs_d_output(i);
        auto pcs_weight = layer.get_pcs_weight(i);
        auto pcs_d_input = layer.get_pcs_d_input(i);
        auto dY = layer.d_output[i]; // [N, D, on, on]
        auto W = layer.weight[i]; // [D, C, m, m]
        auto dX = layer.d_input[i]; // [N, C, n, n]

        W.swap_dim(0, 1); // [C, D, m, m]
        W.reverse(2);
        W.reverse(3); // [C, D, m, m]

        if (!prove_conv(padding,
            pcs_d_output.get(), pcs_weight.get(), pcs_d_input.get(),
            dY, W, dX,
            false,
            rho_inv, sec_param)) return false;
    }
    return true;
}

bool prove_conv_layer(const VCG16::layer_info& layer, size_t rho_inv, size_t sec_param) {

    const int padding = 1;

    assert(layer.type == VCG16::layer_type::conv);
    assert(layer.input.get_dims() == 5); // bat x img x C x n x n
    assert(layer.weight.get_dims() == 5); // bat x D x C x m x m
    assert(layer.output.get_dims() == 5); // bat x img x D x n x n





    std::cout << "Proving forward:";
    if (!prove_conv_forward(layer, padding, rho_inv, sec_param)) {
        std::cout << "❌ Proving forward pass failed." << std::endl;
        return false;
    }

    std::cout << "Proving backward dW:";
    if (!prove_conv_backward_dW(layer, padding, rho_inv, sec_param)) {
        std::cout << "❌ Proving backward dW failed." << std::endl;
        return false;
    }
    
    std::cout << "Proving backward dX:";
    if (!prove_conv_backward_dX(layer, padding, rho_inv, sec_param)) {
        std::cout << "❌ Proving backward dX failed." << std::endl;
        return false;
    }


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

bool prove_full_layer(const VCG16::layer_info& layer, uint64_t rho_inv, size_t sec_param) {
    const int batch = int(layer.input.shape(0));
    // Prove forward
    for (int i = 0; i < batch; ++i) {
        auto pcs_input = layer.get_pcs_input(i);
        auto pcs_weight = layer.get_pcs_weight(i);
        auto pcs_output = layer.get_pcs_output(i);

        open_param oX(layer.input[i], pcs_input.get());
        open_param oW(layer.weight[i], pcs_weight.get());
        open_param oY(layer.output[i], pcs_output.get());

        if (!prove_full(oX, oW, oY,
            layer.input[i], layer.weight[i], layer.output[i],
            sec_param)) {
            std::cout << "❌ Proving full forward failed." << std::endl;
            return false;
        }
    }
    
    // Prove backward dW = X^T * dY
    for (int i = 0; i < batch; ++i) {
        auto pcs_input = layer.get_pcs_input(i);
        auto pcs_d_weight = layer.get_pcs_d_weight(i);
        auto pcs_d_output = layer.get_pcs_d_output(i);
        auto X = layer.input[i]; // [N, n]
        auto dW = layer.d_weight[i]; // [n, m]
        auto dY = layer.d_output[i]; // [N, m]

        X.swap_dim(0, 1); // [n, N]

        auto oX = open_param(X, pcs_input.get());
        auto odW = open_param(dW, pcs_d_weight.get());
        auto odY = open_param(dY, pcs_d_output.get());

        if (!prove_full(oX, odY, odW, 
            X, dY, dW,
            sec_param)) {
            std::cout << "❌ Proving full backward dW failed." << std::endl;
            return false;
        }
    }

    // Prove backward dX = dY * W^T
    for (int i = 0; i < batch; ++i) {
        auto pcs_d_output = layer.get_pcs_d_output(i);
        auto pcs_weight = layer.get_pcs_weight(i);
        auto pcs_d_input = layer.get_pcs_d_input(i);
        auto dY = layer.d_output[i]; // [N, m]
        auto W = layer.weight[i];    // [n, m]
        auto dX = layer.d_input[i];  // [N, n]

        W.swap_dim(0, 1); // [m, n]

        auto odY = open_param(dY, pcs_d_output.get());
        auto oW = open_param(W, pcs_weight.get());
        auto odX = open_param(dX, pcs_d_input.get());

        if (!prove_full(odY, oW, odX,
            dY, W, dX,
            sec_param)) {
            std::cout << "❌ Proving full backward dX failed." << std::endl;
            return false;
        }
    }
    return true;
}