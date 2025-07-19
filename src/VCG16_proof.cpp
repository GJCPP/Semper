#include <cassert>

#include <chrono>

#include "VCG16.h"
#include "VCG16_proof.h"
#include "conv_check.h"
#include "util.h"

bool prove_conv(
    int N, int img,
    size_t C, size_t D, size_t n, size_t m, size_t padding,
    const oracle* pcs_input, const oracle* pcs_weight, const oracle* pcs_output,
    const array_view<Goldilocks2::Element>& X, // [C, n, n]
    const array_view<Goldilocks2::Element>& W, // [C, D, m, m]
    const array_view<Goldilocks2::Element>& Y,
    size_t rho_inv, size_t sec_param) { // [D, n, n]

    std::array<const oracle*, 3> ora = { pcs_input, pcs_weight, pcs_output };
    auto prover = make_conv2_prover(C, D, n, m, padding, X, W, Y);

    int l = find_ceiling_log2(N);
    std::vector<Goldilocks2::Element> pre(l);
    for (int i = 0; i < l; ++i) {
        pre[l - i - 1] = ((img >> i) & 1) ? Goldilocks2::one() : Goldilocks2::zero();
    }

    if (!convVerifier::execute_convcheck_2d(prover, ora, rho_inv, sec_param, pre, true)) {
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
    size_t rho_inv, size_t sec_param) {

    const int N = X.shape(0), IC = X.shape(1), in = X.shape(2), OC = Y.shape(1), on = Y.shape(2);
    const int m = W.shape(2);;

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
            pW, pY, new_m, new_padding, true);

        

        std::chrono::duration<double> elapsed_pad = std::chrono::high_resolution_clock::now() - start_pad;

        pad_time += elapsed_pad.count();
        // std::cout << "pad_weights time: " << elapsed_pad.count() << " seconds" << std::endl;
        start_pad = std::chrono::high_resolution_clock::now();

        pW.view.swap_dim(0, 1); // [C, D, m, m]

        if (!prove_conv(N, j, IC, OC, in, new_m, new_padding,
            pcs_input, pcs_weight, pcs_output,
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
        auto pcs_weight = layer.get_pcs_weight(i);
        auto pcs_output = layer.get_pcs_output(i);
        auto X = layer.input[i]; // [N, C, n, n]
        auto dW = layer.d_weight[i]; // [D, C, m, m]
        auto dY = layer.d_output[i]; // [N, D, on, on]

        X.swap_dim(0, 1); // [C, N, n, n]
        dY.swap_dim(0, 1); // [D, N, on, on]
        dW.swap_dim(0, 1); // [C, D, m, m]

        if (!prove_conv(padding,
            pcs_input.get(), pcs_output.get(), pcs_weight.get(),
            X, dY, dW,
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



    std::cout << "Proving backward dW:";
    if (!prove_conv_backward_dW(layer, padding, rho_inv, sec_param)) {
        std::cout << "❌ Proving backward dW failed." << std::endl;
        return false;
    }
    
    std::cout << "Proving forward:";
    if (!prove_conv_forward(layer, padding, rho_inv, sec_param)) {
        std::cout << "❌ Proving forward pass failed." << std::endl;
        return false;
    }


    return true;
}

