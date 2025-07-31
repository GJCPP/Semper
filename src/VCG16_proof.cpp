#include <cassert>

#include <chrono>

#include "VCG16.h"
#include "VCG16_check.h"
#include "VCG16_proof.h"
#include "conv_check.h"
#include "mat_check.h"
#include "div_check.h"
#include "sign_check.h"

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

bool prove_relu_layer(const VCG16::layer_info& layer,
    size_t scale, size_t max_val, size_t sqr_val,
    size_t rho_inv, size_t sec_param) {

    const int batch = int(layer.input.shape(0));

    for (int i = 0; i != batch; ++i) {
        // Prove forward
        auto pcs_X = *layer.get_pcs_input(i);
        auto pcs_Y = *layer.get_pcs_output(i);
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
            } else {
                X_quo[i] = Goldilocks2::fromS64(Goldilocks2::toS64(X_copy[i]) / int64_t(scale));
            }
        }
        auto X_rem = get_rem(X_copy, scale, X_quo, true);
        auto pcs_X_quo = ligero_commit_base(X_quo, rho_inv);
        auto pcs_X_rem = ligero_commit_base(X_rem, rho_inv);
        divProver div_prover_X(X_copy, X_quo, X_rem, scale, true, rho_inv);
        if (!divVerifier::execute_div_check(div_prover_X, pcs_X, pcs_X_quo, pcs_X_rem, sec_param)) {
            std::cout << "❌ Proving relu forward scale_X failed." << std::endl;
            return false;
        }
        // 2. Prove sign of scaled X
        auto X_scale_sign = get_sign(X_quo, true);
        auto pcs_X_scale_sign = ligero_commit_base(X_scale_sign, rho_inv);
        signProver sign_prover_X(X_quo, X_scale_sign, scale, max_val, true, rho_inv);
        if (!signVerifier::execute_sign_check(sign_prover_X, pcs_X_quo, pcs_X_scale_sign, sec_param)) {
            std::cout << "❌ Proving relu forward X_scale_sign failed." << std::endl;
            return false;
        }
        // 3. Prove Y = X_scale_sign * X_quo
        auto Y_challenge = random_vec_ext(logn); // draw new challenges
        auto mle_challenge = MLE_Eq(Y_challenge);
        auto claim_Y_cha = pcs_Y.open(Y_challenge, sec_param);
        p3Prover p3_prover_Y(X_scale_sign, X_quo, mle_challenge);
        if (!p3Verifier::execute_sumcheck(p3_prover_Y, claim_Y_cha, {&pcs_X_scale_sign, &pcs_X_quo, &mle_challenge}, sec_param)) {
            std::cout << "❌ Proving relu forward Y failed." << std::endl;
            return false;
        }
        
        // Prove backward
        auto pcs_dX = *layer.get_pcs_d_input(i);
        auto pcs_dY = *layer.get_pcs_d_output(i);
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
        auto pcs_dY_filtered = ligero_commit_base(dY_filtered, rho_inv);
        auto dY_challenge = random_vec_ext(logn); // draw new challenges
        auto mle_challenge_dY_filtered = MLE_Eq(dY_challenge);
        p3Prover p3_prover_dY_filtered(X_scale_sign, dY_copy, mle_challenge_dY_filtered);
        auto claim_dY_filtered = pcs_dY_filtered.open(dY_challenge, sec_param);
        if (!p3Verifier::execute_sumcheck(p3_prover_dY_filtered, claim_dY_filtered, {&pcs_X_scale_sign, &pcs_dY, &mle_challenge_dY_filtered}, sec_param)) {
            std::cout << "❌ Proving relu backward dY_filtered failed." << std::endl;
            return false;
        }
        // 2. Prove dX = dY_filtered / scale
        auto dX_rem = get_rem(dY_filtered, scale, dX_copy, true);
        auto pcs_dX_rem = ligero_commit_base(dX_rem, rho_inv);
        divProver div_prover_dX(dY_filtered, dX_copy, dX_rem, scale, true, rho_inv);
        if (!divVerifier::execute_div_check(div_prover_dX, pcs_dY_filtered, pcs_dX, pcs_dX_rem, sec_param)) {
            std::cout << "❌ Proving relu backward dX failed." << std::endl;
            return false;
        }
    }
    return true;
}

bool prove_pool_layer(const VCG16::layer_info& layer, size_t rho_inv, size_t sec_param) {
    const int batch = int(layer.input.shape(0));
    const int img = int(layer.input.shape(1));
    // stride = kernel size = 2
    // layer.input : [batch, img, C, N, N]
    // layer.output: [batch, img, C, N/2, N/2]
    int logimg = find_ceiling_log2(img);
    int logC = find_ceiling_log2(layer.input.shape(2));
    int logN = find_ceiling_log2(layer.input.shape(3));
    // Prove forward
    for (int i = 0; i < batch; ++i) {
        auto pcs_input = layer.get_pcs_input(i);
        auto pcs_output = layer.get_pcs_output(i);

        auto input = layer.input[i]; // [img, C, N, N]
        auto output = layer.output[i]; // [img, C, N/2, N/2]

        // Step 1. Commit expand(output) - input
        array<Goldilocks2::Element> diff(input.get_shape());
        for (int j = 0; j < img; ++j) {
            for (int c = 0; c < input.shape(1); ++c) {
                for (int n = 0; n < input.shape(2); ++n) {
                    for (int m = 0; m < input.shape(3); ++m) {
                        diff(j, c, n, m) = output(j, c, n / 2, m / 2) - input(j, c, n, m);
                    }
                }
            }
        }
        auto pcs_diff = ligero_commit_base(diff.view, rho_inv);
        // Step 2. Prove diff = expand(output) - input
        std::vector<Goldilocks2::Element> output_cha = random_vec_ext(logimg + logC + 2 * logN - 2);
        std::vector<Goldilocks2::Element> extra_cha = random_vec_ext(2);
        std::vector<Goldilocks2::Element> expand_cha; // [logimg | logC | logN - 1 | 1 | logN - 1 | 1]
        expand_cha.reserve(logimg + logC + 2 * logN);
        int next_output_cha = 0, next_extra_cha = 0;
        for (int j = 0; j < logimg + logC + 2 * logN; ++j) {
            if (j != logimg + logC + logN - 1 && j != logimg + logC + 2 * logN - 1) {
                expand_cha.push_back(output_cha[next_output_cha++]);
            } else {
                expand_cha.push_back(extra_cha[next_extra_cha++]);
            }
        }
        if (logN == 1) {
            std::vector<Goldilocks2::Element> new_output_cha = expand_cha;
            new_output_cha[logimg + logC] = Goldilocks2::zero();
            new_output_cha[logimg + logC + 1] = Goldilocks2::zero();
            output_cha = new_output_cha;
        }
        MLE_Eq eq_extra(extra_cha);
        Goldilocks2::Element factor = Goldilocks2::zero();
        for (int i = 0; i != 2; ++i)
            for (int j = 0; j != 2; ++j)
                factor += eq_extra.open({i ? Goldilocks2::one() : Goldilocks2::zero(), j ? Goldilocks2::one() : Goldilocks2::zero()}, sec_param);
        if (factor * pcs_output->open(output_cha, sec_param) - pcs_input->open(expand_cha, sec_param)
                != pcs_diff.open(expand_cha, sec_param)) {
            std::cout << "❌ Proving pool forward failed." << std::endl;
            return false;
        }
    }
    return true;
}