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

    assert(X.shape(0) == C);
    assert(X.shape(1) == n);
    assert(X.shape(2) == n);
    assert(W.shape(0) == C);
    assert(W.shape(1) == D);
    assert(W.shape(2) == m);
    assert(W.shape(3) == m);
    assert(Y.shape(0) == D);
    assert(Y.shape(1) == n + 2 * padding - m + 1);
    assert(Y.shape(2) == n + 2 * padding - m + 1);

    assert(is_power_of_2(m));

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

bool prove_conv_layer(VCG16::layer_info layer, size_t rho_inv, size_t sec_param) {

    assert(layer.type == VCG16::layer_type::conv);
    assert(layer.input.get_dims() == 5); // bat x img x C x n x n
    assert(layer.weight.get_dims() == 5); // bat x D x C x m x m
    assert(layer.output.get_dims() == 5); // bat x img x D x n x n

    const size_t padding = 1;

    // Prove forward pass
    int bat = int(layer.input.shape(0));
    int img = int(layer.input.shape(1));
    size_t C = layer.input.shape(2);
    size_t D = layer.output.shape(2);
    size_t n = layer.input.shape(3);
    size_t m = layer.weight.shape(3);

    for (int i = 0; i < bat; ++i) {
        double pad_time = 0, prove_time = 0;
            std::cout << "Proving img ";
        for (int j = 0; j < img; ++j) {
            std::cout << j << " ";
            std::cout.flush();
            auto pcs_input = layer.get_pcs_input(i);
            auto pcs_weight = layer.get_pcs_weight(i);
            auto pcs_output = layer.get_pcs_output(i);

            auto X = layer.input[i][j]; // [C, n, n]
            array<Goldilocks2::Element> W; // [D, C, m, m]
            array<Goldilocks2::Element> Y; // [D, n, n]
            size_t new_m, new_padding;

            auto start_pad = std::chrono::high_resolution_clock::now();

            pad_weights(C, D, n, m, padding,
                layer.input[i][j],
                layer.weight[i],
                layer.output[i][j],
                W, Y, new_m, new_padding, true);

            std::chrono::duration<double> elapsed_pad = std::chrono::high_resolution_clock::now() - start_pad;

            pad_time += elapsed_pad.count();
            // std::cout << "pad_weights time: " << elapsed_pad.count() << " seconds" << std::endl;
            start_pad = std::chrono::high_resolution_clock::now();

            pcs_weight = std::make_shared<ligeropcs_base>(ligero_commit_base(W.view, 2));
            W.view.swap_dim(0, 1); // [C, D, m, m]

            prove_conv(img, i, C, D, n, new_m, new_padding,
                       pcs_input.get(), pcs_weight.get(), pcs_output.get(),
                       X, W, Y, rho_inv, sec_param);
            std::chrono::duration<double> elapsed_prove = std::chrono::high_resolution_clock::now() - start_pad;
            // std::cout << "prove_conv time: " << elapsed_prove.count() << " seconds" << std::endl;
            prove_time += elapsed_prove.count();
        }
        std::cout << std::endl << "pad time = " << pad_time << "s, prove time = " << prove_time << "s" << std::endl;
    }
    return true;
}

