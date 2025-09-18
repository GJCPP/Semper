#include "VGG16.h"
#include "CNN_check.h"

#include <unordered_map>
#include <iostream>
#include <random>

bool check_range(const array_view<Goldilocks2::Element>& input, int64_t max_value) {
    bool ret = true;
    size_t sz = input.size();
    #pragma omp parallel for
    for (size_t i = 0; i < sz; ++i) {
        int64_t actual = Goldilocks2::toS64(input.get(i));
        if (actual > max_value || -actual > max_value) {
            ret = false;
            #pragma omp critical
            {
                std::cout << "❌ Out of range at (i=" << i << "): value=" << actual << ", max_value=" << max_value << std::endl;
            }
        }
    }
    return ret;
}

bool check_conv(
    const array_view<Goldilocks2::Element>& input, // [N, C, H, W]
    const array_view<Goldilocks2::Element>& weights, // [OC, C, K, K]
    const array_view<Goldilocks2::Element>& expected, // [N, OC, H + 2 * pad - K + 1, W + 2 * pad - K + 1]
    size_t pad,
    size_t n_samples
) {
    if (n_samples > 0) {
        return random_check_conv(input, weights, expected, pad, n_samples);
    }
    size_t N = input.shape(0);
    size_t C = input.shape(1);
    size_t H = input.shape(2);
    size_t W = input.shape(3);
    size_t OC = weights.shape(0);
    size_t K = weights.shape(2);
    size_t OH = H + 2 * pad - K + 1;
    size_t OW = W + 2 * pad - K + 1;
    assert(input.get_dims() == 4);
    assert(weights.get_dims() == 4);
    assert(expected.get_dims() == 4);
    assert(input.shape(0) == N);
    assert(input.shape(1) == C);
    assert(input.shape(2) == H);
    assert(input.shape(3) == W);
    assert(weights.shape(0) == OC);
    assert(weights.shape(1) == C);
    assert(weights.shape(2) == K);
    assert(weights.shape(3) == K);
    assert(expected.shape(0) == N);
    assert(expected.shape(1) == OC);
    assert(expected.shape(2) == OH);
    assert(expected.shape(3) == OW);

    bool ret = true;

#pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {

        auto view_input_n = input[n];
        auto view_expected_n = expected[n];

        for (size_t oc = 0; oc < OC && ret; ++oc) {

            auto view_weights_oc = weights[oc];
            auto view_expected_n_oc = view_expected_n[oc];

            for (size_t i = 0; i < OH && ret; ++i) {

                auto view_expected_n_oc_h = view_expected_n_oc[i];

                for (size_t j = 0; j < OW && ret; ++j) {

                    Goldilocks2::Element acc = Goldilocks2::zero();
                    for (size_t c = 0; c < C && ret; ++c) {

                        auto view_weights_oc_c = view_weights_oc[c];
                        auto view_input_n_c = view_input_n[c];

                        for (size_t ki = 0; ki < K && ret; ++ki) {
                            for (size_t kj = 0; kj < K && ret; ++kj) {
                                if (i + ki >= pad && i + ki < H + pad && j + kj >= pad && j + kj < W + pad) {
                                    acc += view_input_n_c(i + ki - pad, j + kj - pad) * view_weights_oc_c(ki, kj);
                                }
                            }
                        }
                    }
                    // acc = acc / scale; // downscale
                    // if (acc < 0) acc = 0; // relu

                    Goldilocks2::Element actual = view_expected_n_oc_h(j);
                    if (actual != acc) {
                        ret = false;
                        std::cout << "❌ Mismatch at (n=" << n << ", oc=" << oc << ", i=" << i << ", j=" << j << "): manual=" << acc << ", expected=" << actual << std::endl;
                    }
                }
            }
        }
    }
    return ret;
}

bool random_check_conv(
    const array_view<Goldilocks2::Element>& input, // [N, C, H, W]
    const array_view<Goldilocks2::Element>& weights, // [OC, C, K, K]
    const array_view<Goldilocks2::Element>& expected, // [N, OC, H + 2 * pad - K + 1, W + 2 * pad - K + 1]
    size_t pad,
    size_t n_samples
) {
    size_t N = input.shape(0);
    size_t C = input.shape(1);
    size_t H = input.shape(2);
    size_t W = input.shape(3);
    size_t K = weights.shape(2);
    size_t OC = weights.shape(0);
    size_t OH = H + 2 * pad - K + 1;
    size_t OW = W + 2 * pad - K + 1;
    assert(input.shape(0) == N);
    assert(input.shape(1) == C);
    assert(input.shape(2) == H);
    assert(input.shape(3) == W);
    assert(weights.shape(0) == OC);
    assert(weights.shape(1) == C);
    assert(weights.shape(2) == K);
    assert(weights.shape(3) == K);
    assert(expected.shape(0) == N);
    assert(expected.shape(1) == OC);
    assert(expected.shape(2) == OH);
    assert(expected.shape(3) == OW);

    bool ret = true;

#pragma omp parallel for
    for (size_t cnt = 0; cnt < n_samples; ++cnt) {
        size_t n = rand() % N;
        size_t oc = rand() % OC;
        size_t i = rand() % OH;
        size_t j = rand() % OW;
        Goldilocks2::Element acc = Goldilocks2::zero();
        for (size_t c = 0; c < C; ++c) {
            for (size_t ki = 0; ki < K; ++ki) {
                for (size_t kj = 0; kj < K; ++kj) {
                    if (i + ki >= pad && i + ki < H + pad && j + kj >= pad && j + kj < W + pad) {
                        acc += input(n, c, i + ki - pad, j + kj - pad) * weights(oc, c, ki, kj);
                    }
                }
            }
        }
        // acc = acc / scale; // downscale
        // if (acc < 0) acc = 0; // relu

        Goldilocks2::Element actual = expected(n, oc, i, j);
        if (actual != acc) {
            std::cout << "❌ Mismatch at (n=" << n << ", oc=" << oc << ", i=" << i << ", j=" << j << "): manual=" << acc << ", expected=" << actual << std::endl;
            ret = false;
        }
    }
    return ret;
}

bool random_check_single_conv(
    const array_view<Goldilocks2::Element>& input, // [C, H, W]
    const array_view<Goldilocks2::Element>& weights, // [OC, C, K, K]
    const array_view<Goldilocks2::Element>& expected, // [OC, H + 2 * pad - K + 1, W + 2 * pad - K + 1]
    size_t pad,
    size_t n_samples
) {
    size_t C = input.shape(0);
    size_t H = input.shape(1);
    size_t W = input.shape(2);
    size_t K = weights.shape(2);
    size_t OC = weights.shape(0);
    size_t OH = H + 2 * pad - K + 1;
    size_t OW = W + 2 * pad - K + 1;
    assert(input.shape(0) == C);
    assert(input.shape(1) == H);
    assert(input.shape(2) == W);
    assert(weights.shape(0) == OC);
    assert(weights.shape(1) == C);
    assert(weights.shape(2) == K);
    assert(weights.shape(3) == K);
    assert(expected.shape(0) == OC);
    assert(expected.shape(1) == OH);
    assert(expected.shape(2) == OW);

    bool ret = true;

#pragma omp parallel for
    for (size_t cnt = 0; cnt < n_samples; ++cnt) {
        size_t oc = rand() % OC;
        size_t i = rand() % OH;
        size_t j = rand() % OW;
        Goldilocks2::Element acc = Goldilocks2::zero();
        for (size_t c = 0; c < C; ++c) {
            for (size_t ki = 0; ki < K; ++ki) {
                for (size_t kj = 0; kj < K; ++kj) {
                    if (i + ki >= pad && i + ki < H + pad && j + kj >= pad && j + kj < W + pad) {
                        acc += input(c, i + ki - pad, j + kj - pad) * weights(oc, c, ki, kj);
                    }
                }
            }
        }
        // acc = acc / scale; // downscale
        // if (acc < 0) acc = 0; // relu

        Goldilocks2::Element actual = expected(oc, i, j);
        if (actual != acc) {
            std::cout << "❌ Mismatch at (oc=" << oc << ", i=" << i << ", j=" << j << "): manual=" << acc << ", expected=" << actual << std::endl;
            ret = false;
        }
    }
    return ret;
}

void add_conv(const array_view<Goldilocks2::Element>& input, const array_view<Goldilocks2::Element>& weights, array_view<Goldilocks2::Element>& output, size_t pad) {
    
    size_t H = input.shape(0);
    size_t W = input.shape(1);
    size_t K = weights.shape(0);
    size_t OH = H + 2 * pad - K + 1;
    size_t OW = W + 2 * pad - K + 1;
    assert(input.get_dims() == 2);
    assert(weights.get_dims() == 2);
    assert(output.get_dims() == 2);
    assert(input.shape(0) == H);
    assert(input.shape(1) == W);
    assert(weights.shape(0) == K);
    assert(weights.shape(1) == K);
    assert(output.shape(0) == OH);
    assert(output.shape(1) == OW);

// #pragma omp parallel for
    for (size_t i = 0; i < OH; ++i) {
        for (size_t j = 0; j < OW; ++j) {
            Goldilocks2::Element acc = Goldilocks2::zero();
            for (size_t ki = 0; ki < K; ++ki) {
                for (size_t kj = 0; kj < K; ++kj) {
                    if (i + ki >= pad && i + ki < H + pad && j + kj >= pad && j + kj < W + pad) {
                        acc += input(i + ki - pad, j + kj - pad) * weights(ki, kj);
                    }
                }
            }
            output(i, j) += acc;
        }
    }
}

bool check_relu(
    const array_view<Goldilocks2::Element>& sign,
    const array_view<Goldilocks2::Element>& input, // [N, C, H, W]
    const array_view<Goldilocks2::Element>& output, // [N, C, H, W]
    int64_t scale,
    size_t n_samples
) {
    if (n_samples > 0) {
        return random_check_relu(sign, input, output, scale, n_samples);
    }
    assert(input.size() == output.size());

    bool ret = true;
    size_t sz = input.size();
    
    #pragma omp parallel for
    for (size_t i = 0; i < sz; ++i) {
        Goldilocks2::Element s = sign.get(i);
        Goldilocks2::Element actual = input.get(i);
        Goldilocks2::Element expected = output.get(i);
        if (Goldilocks2::toS64(s) <= 0) actual = Goldilocks2::zero();
        else {
            actual = Goldilocks2::fromS64(std::round(double(Goldilocks2::toS64(actual)) / double(scale)));
        }
        if (std::abs(Goldilocks2::toS64(actual - expected)) > 1) {
            ret = false;
            #pragma omp critical
            {
                std::cout << "❌ Mismatch at (i=" << i << "): manual=" << actual << ", expected=" << expected << std::endl;
            }
        }
    }
    return ret;
}

bool random_check_relu(
    const array_view<Goldilocks2::Element>& sign,
    const array_view<Goldilocks2::Element>& input, // [N, C, H, W]
    const array_view<Goldilocks2::Element>& output, // [N, C, H, W]
    int64_t scale,
    size_t n_samples
) {
    bool ret = true;
    size_t sz = input.size();
    static thread_local std::mt19937 rng(std::random_device{}());

    // #pragma omp parallel for
    for (size_t cnt = 0; cnt < n_samples; ++cnt) {
        size_t i = std::uniform_int_distribution<size_t>(0, sz - 1)(rng);
        Goldilocks2::Element s = sign.get(i);
        Goldilocks2::Element actual = input.get(i);
        Goldilocks2::Element expected = output.get(i);
        if (Goldilocks2::toS64(s) <= 0) actual = Goldilocks2::zero();
        else {
            actual = Goldilocks2::fromS64(std::round(double(Goldilocks2::toS64(actual)) / double(scale)));
        }
        if (std::abs(Goldilocks2::toS64(actual - expected)) > 1) {
            ret = false;
            // #pragma omp critical
            {
                std::cout << "❌ Mismatch at (i=" << i << "): manual=" << actual << ", expected=" << expected << std::endl;
            }
        }
    }
    return ret;
}

bool check_pool(
    const array_view<Goldilocks2::Element>& input, // [N, C, H, W]
    const array_view<Goldilocks2::Element>& output, // [N, C, H/2, W/2]
    const array_view<Goldilocks2::Element>& idx,
    size_t kernel_size,
    size_t stride,
    bool backward,
    size_t n_samples
) {
    size_t N = input.shape(0);
    size_t C = input.shape(1);
    size_t H = input.shape(2);
    size_t W = input.shape(3);
    size_t OH = H / stride;
    size_t OW = W / stride;
    assert(input.get_dims() == 4);
    assert(output.get_dims() == 4);
    assert(output.shape(0) == N);
    assert(output.shape(1) == C);
    assert(output.shape(2) == OH);
    assert(output.shape(3) == OW);
    assert(idx.get_dims() == 4);
    assert(idx.shape(0) == N);
    assert(idx.shape(1) == C);
    assert(idx.shape(2) == OH);
    assert(idx.shape(3) == OW);
    assert(H % stride == 0);
    assert(W % stride == 0);

    if (n_samples > 0) {
        return random_check_pool(input, output, idx, kernel_size, stride, backward, n_samples);
    }

    bool ret = true;
    #pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {

        auto view_input_n = input[n];
        auto view_output_n = output[n];
        auto view_idx_n = idx[n];

        for (size_t c = 0; c < C; ++c) {
            
            auto view_input_n_c = view_input_n[c];
            auto view_output_n_c = view_output_n[c];
            auto view_idx_n_c = view_idx_n[c];

            for (size_t i = 0; i < OH; ++i) {
                for (size_t j = 0; j < OW; ++j) {
                    size_t index = view_idx_n_c(i, j)[0].fe;
                    Goldilocks2::Element max_val = view_input_n_c.get(index);
                    for (size_t ki = 0; ki < kernel_size; ++ki) {
                        for (size_t kj = 0; kj < kernel_size; ++kj) {
                            size_t x = i * stride + ki;
                            size_t y = j * stride + kj;
                            if (backward) {
                                if (x * W + y != index && view_input_n_c(x, y) != Goldilocks2::zero()) {
                                    ret = false;
                                    #pragma omp critical
                                    {
                                        std::cout << "❌ Incorrect at (n=" << n << ", c=" << c << ", i=" << i << ", j=" << j << "): expect 0-grad,"
                                                << ", but find " << view_input_n_c(x, y) << std::endl;
                                    }
                                }
                            } else {
                                if (view_input_n_c(x, y)[0].fe > max_val[0].fe) {
                                    ret = false;
                                    #pragma omp critical
                                    {
                                        std::cout << "❌ Incorrect at (n=" << n << ", c=" << c << ", i=" << i << ", j=" << j << "): expect max_val = " << max_val
                                                << ", but find larger value " << view_input_n_c(x, y) << std::endl;
                                    }
                                }
                            }
                        }
                    }
                    if (max_val != view_output_n_c(i, j)) {
                        ret = false;
                        #pragma omp critical
                        {
                            std::cout << "❌ Mismatch at (n=" << n << ", c=" << c << ", i=" << i << ", j=" << j << "): manual=" << max_val << ", expected=" << view_output_n_c(i, j) << std::endl;
                        }
                    }
                }
            }
        }
    }
    return ret;
}

bool random_check_pool(
    const array_view<Goldilocks2::Element>& input,
    const array_view<Goldilocks2::Element>& output,
    const array_view<Goldilocks2::Element>& idx,
    size_t kernel_size,
    size_t stride,
    bool backward,
    size_t n_samples
) {
    size_t N = input.shape(0);
    size_t C = input.shape(1);
    size_t H = input.shape(2);
    size_t W = input.shape(3);
    size_t OH = H / stride;
    size_t OW = W / stride;
    bool ret = true;

    #pragma omp parallel for
    for (size_t cnt = 0; cnt < n_samples; ++cnt) {
        size_t n = rand() % N;
        size_t c = rand() % C;
        size_t i = rand() % OH;
        size_t j = rand() % OW;
        size_t index = idx(n, c, i, j)[0].fe;
        Goldilocks2::Element max_val = input[n][c].get(index);
        if (max_val != output(n, c, i, j)) {
            ret = false;
            #pragma omp critical
            {
                std::cout << "❌ Mismatch at (n=" << n << ", c=" << c << ", i=" << i << ", j=" << j << "): manual=" << max_val << ", expected=" << output(n, c, i, j) << std::endl;
            }
        }
        for (size_t ki = 0; ki < kernel_size; ++ki) {
            for (size_t kj = 0; kj < kernel_size; ++kj) {
                size_t x = i * stride + ki;
                size_t y = j * stride + kj;
        
                if (backward) {
                    if (x * W + y != index && input(n, c, x, y) != Goldilocks2::zero()) {
                        ret = false;
                        #pragma omp critical
                        {
                            std::cout << "❌ Incorrect at (n=" << n << ", c=" << c << ", i=" << i << ", j=" << j << "): expect 0-grad,"
                                    << ", but find " << input(n, c, x, y) << std::endl;
                        }
                    }
                } else {
                    if (input(n, c, x, y)[0].fe > max_val[0].fe) {
                        ret = false;
                        #pragma omp critical
                        {
                            std::cout << "❌ Incorrect at (n=" << n << ", c=" << c << ", i=" << i << ", j=" << j << "): expect max_val = " << max_val
                                    << ", but find larger value " << input(n, c, x, y) << std::endl;
                        }
                    }
                }
            }
        }
    }
    return ret;
}

bool check_flat(
    const array_view<Goldilocks2::Element>& input,
    const array_view<Goldilocks2::Element>& output,
    size_t n_samples) {

    size_t N = input.shape(0);
    size_t C = input.shape(1);
    size_t H = input.shape(2);
    size_t W = input.shape(3);
    // size_t OH = C * H * W;
    assert(input.get_dims() == 4);
    assert(output.get_dims() == 2);
    assert(input.shape(0) == N);
    assert(input.shape(1) == C);
    assert(input.shape(2) == H);
    assert(input.shape(3) == W);
    assert(output.shape(0) == N);
    assert(output.shape(1) == OH);

    if (n_samples > 0) {
        return random_check_flat(input, output, n_samples);
    }

    bool ret = true;
    // size_t sz = input.size();
    #pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {

        auto view_input_n = input[n];
        auto view_output_n = output[n];

        for (size_t c = 0; c < C; ++c) {

            auto view_input_n_c = view_input_n[c];

            for (size_t i = 0; i < H; ++i) {

                auto view_input_n_c_i = view_input_n_c[i];

                for (size_t j = 0; j < W; ++j) {
                    size_t index = c * H * W + i * W + j;
                    Goldilocks2::Element actual = view_input_n_c_i.get(j);
                    Goldilocks2::Element expected = view_output_n.get(index);
                    if (actual != expected) {
                        ret = false;
                        #pragma omp critical
                        {
                            std::cout << "❌ Mismatch at (n=" << n << ", c=" << c << ", i=" << i << ", j=" << j << "): manual=" << actual << ", expected=" << expected << std::endl;
                        }
                    }
                }
            }
        }
    }
    return ret;
}

bool random_check_flat(
    const array_view<Goldilocks2::Element>& input,
    const array_view<Goldilocks2::Element>& output,
    size_t n_samples) {

    size_t N = input.shape(0);
    size_t C = input.shape(1);
    size_t H = input.shape(2);
    size_t W = input.shape(3);
    // size_t OH = C * H * W;
    bool ret = true;
    static thread_local std::mt19937 rng(std::random_device{}());

    #pragma omp parallel for
    for (size_t cnt = 0; cnt < n_samples; ++cnt) {
        size_t n = std::uniform_int_distribution<size_t>(0, N - 1)(rng);
        size_t c = std::uniform_int_distribution<size_t>(0, C - 1)(rng);
        size_t i = std::uniform_int_distribution<size_t>(0, H - 1)(rng);
        size_t j = std::uniform_int_distribution<size_t>(0, W - 1)(rng);
        size_t index = c * H * W + i * W + j;

        Goldilocks2::Element actual = input(n, c, i, j);
        Goldilocks2::Element expected = output(n, index);
        if (actual != expected) {
            ret = false;
            #pragma omp critical
            {
                std::cout << "❌ Mismatch at (n=" << n << ", c=" << c << ", i=" << i << ", j=" << j << "): manual=" << actual << ", expected=" << expected << std::endl;
            }
        }
    }
    return ret;
}

bool check_full(
    const array_view<Goldilocks2::Element>& input,
    const array_view<Goldilocks2::Element>& weights,
    const array_view<Goldilocks2::Element>& output,
    size_t n_samples
) {
    size_t N = input.shape(0);
    size_t C = input.shape(1);
    size_t OC = output.shape(1);
    assert(input.get_dims() == 2);
    assert(weights.get_dims() == 2);
    assert(output.get_dims() == 2);
    assert(input.shape(0) == N);
    assert(input.shape(1) == C);
    assert(weights.shape(0) == C);
    assert(weights.shape(1) == OC);
    assert(output.shape(0) == N);
    assert(output.shape(1) == OC);

    if (n_samples > 0) {
        return random_check_full(input, weights, output, n_samples);
    }
    bool ret = true;
    #pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {
        auto view_input_n = input[n];
        // auto view_weights_n = weights[n];
        auto view_output_n = output[n];
        for (size_t oc = 0; oc < OC; ++oc) {
            Goldilocks2::Element actual = Goldilocks2::zero();
            for (size_t c = 0; c < C; ++c) {
                actual += view_input_n(c) * weights(c, oc);
            }
            Goldilocks2::Element expected = view_output_n(oc);
            if (actual != expected) {
                ret = false;
                #pragma omp critical
                {
                    std::cout << "❌ Mismatch at (n=" << n << ", oc=" << oc << "): manual=" << actual << ", expected=" << expected << std::endl;
                }
            }
        }
    }
    return ret;
}

bool random_check_full(
    const array_view<Goldilocks2::Element>& input,
    const array_view<Goldilocks2::Element>& weights,
    const array_view<Goldilocks2::Element>& output,
    size_t n_samples
) {
    size_t N = input.shape(0);
    size_t C = input.shape(1);
    size_t OC = output.shape(1);
    bool ret = true;
    static thread_local std::mt19937 rng(std::random_device{}());

    // #pragma omp parallel for
    for (size_t cnt = 0; cnt < n_samples; ++cnt) {
        size_t n = std::uniform_int_distribution<size_t>(0, N - 1)(rng);
        // size_t c = std::uniform_int_distribution<size_t>(0, C - 1)(rng);
        size_t oc = std::uniform_int_distribution<size_t>(0, OC - 1)(rng);
        Goldilocks2::Element actual = Goldilocks2::zero();
        for (size_t c = 0; c < C; ++c) {
            actual += input(n, c) * weights(c, oc);
        }
        Goldilocks2::Element expected = output(n, oc);
        if (actual != expected) {
            ret = false;
            #pragma omp critical
            {
                std::cout << "❌ Mismatch at (n=" << n << ", oc=" << oc << "): manual=" << actual << ", expected=" << expected << std::endl;
            }
        }
    }
    return ret;
}

bool check_softmax(
    const array_view<Goldilocks2::Element>& input,
    const array_view<Goldilocks2::Element>& output, // output = d_input
    const array_view<Goldilocks2::Element>& label,
    int64_t scale) {

    size_t N = input.shape(0);
    size_t C = input.shape(1);
    assert(input.get_dims() == 2);
    assert(output.get_dims() == 2);
    assert(input.shape(0) == N);
    assert(input.shape(1) == C);
    assert(output.shape(0) == N);
    assert(output.shape(1) == C);

    bool ret = true;
    // #pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {
        auto view_input_n = input[n];
        auto view_output_n = output[n];
        std::vector<Goldilocks2::Element> scale_input(C);
        std::vector<Goldilocks2::Element> exp_input(C);
        Goldilocks2::Element max_input = Goldilocks2::divScalar(view_input_n(0), scale);
        for (size_t c = 0; c < C; ++c) {
            scale_input[c] = Goldilocks2::divScalar(view_input_n(c), scale); // small value
            if (Goldilocks2::toS64(scale_input[c]) > Goldilocks2::toS64(max_input)) {
                max_input = scale_input[c];
            }
        }
        for (size_t c = 0; c < C; ++c) {
            uint64_t diff = max_input[0].fe - scale_input[c][0].fe;
            uint64_t e_pow_inv = std::round(scale * std::exp(double(diff) / double(scale)));
            exp_input[c] = Goldilocks2::fromU64(e_pow_inv);
        }
        Goldilocks2::Element sum = Goldilocks2::zero();
        for (size_t c = 0; c < C; ++c) {
            sum += exp_input[c];
        }
        for (size_t c = 0; c < C; ++c) {
            exp_input[c] = Goldilocks2::divScalar(exp_input[c] * Goldilocks2::fromS64(scale), Goldilocks2::toS64(sum));
        }
        exp_input[Goldilocks2::toS64(label(n))] -= Goldilocks2::fromS64(scale);
        for (size_t c = 0; c < C; ++c) {
            exp_input[c] = Goldilocks2::divScalar(exp_input[c], N);
        }
        for (size_t c = 0; c < C; ++c) {
            if (std::abs(Goldilocks2::toS64(exp_input[c] - view_output_n(c))) > 1) {
                ret = false;
                #pragma omp critical
                {
                    std::cout << "❌ Mismatch at (n=" << n << ", c=" << c << "): manual=" << exp_input[c] << ", expected=" << view_output_n(c) << std::endl;
                }
            }
        }
    }

    return ret;
}
