#include "VCG16.h"

bool check_range(
    const array_view<Goldilocks2::Element>& input,
    int64_t max_value
);

bool check_conv(
    const array_view<Goldilocks2::Element>& input, // [N, C, H, W]
    const array_view<Goldilocks2::Element>& weights, // [OC, C, 3, 3]
    const array_view<Goldilocks2::Element>& expected, // [N, OC, H, W]
    size_t pad,
    size_t n_samples = 0
);

bool random_check_conv(
    const array_view<Goldilocks2::Element>& input, // [N, C, H, W]
    const array_view<Goldilocks2::Element>& weights, // [OC, C, 3, 3]
    const array_view<Goldilocks2::Element>& expected, // [N, OC, H, W]
    size_t pad,
    size_t n_samples
);

void add_conv(
    const array_view<Goldilocks2::Element>& input, // [H, W]
    const array_view<Goldilocks2::Element>& weights, // [K, K]
    array_view<Goldilocks2::Element>& output, // [H + 2 * pad - K + 1, W + 2 * pad - K + 1]
    size_t pad
);

bool check_relu(
    const array_view<Goldilocks2::Element>& sign,
    const array_view<Goldilocks2::Element>& input,
    const array_view<Goldilocks2::Element>& output,
    int64_t scale,
    size_t n_samples = 0
);

bool random_check_relu(
    const array_view<Goldilocks2::Element>& sign,
    const array_view<Goldilocks2::Element>& input,
    const array_view<Goldilocks2::Element>& output,
    int64_t scale,
    size_t n_samples
);

bool check_pool(
    const array_view<Goldilocks2::Element>& input, // [N, C, H, W]
    const array_view<Goldilocks2::Element>& output, // [N, C, H / stride, W / stride]
    const array_view<Goldilocks2::Element>& idx, // [N, C, H / stride, W / stride]
    size_t kernel_size,
    size_t stride,
    bool backward,
    size_t n_samples = 0
);

bool random_check_pool(
    const array_view<Goldilocks2::Element>& input, // [N, C, H, W]
    const array_view<Goldilocks2::Element>& output, // [N, C, H / stride, W / stride]
    const array_view<Goldilocks2::Element>& idx, // [N, C, H / stride, W / stride]
    size_t kernel_size,
    size_t stride,
    bool backward,
    size_t n_samples
);

bool check_flat(
    const array_view<Goldilocks2::Element>& input, // [N, C, H, W]
    const array_view<Goldilocks2::Element>& output, // [N, C * H * W]
    size_t n_samples = 0
);

bool random_check_flat(
    const array_view<Goldilocks2::Element>& input, // [N, C, H, W]
    const array_view<Goldilocks2::Element>& output, // [N, C * H * W]
    size_t n_samples
);

bool check_full(
    const array_view<Goldilocks2::Element>& input, // [N, C]
    const array_view<Goldilocks2::Element>& weights, // [C, OC]
    const array_view<Goldilocks2::Element>& output, // [N, OC]
    size_t n_samples = 0
);

bool random_check_full(
    const array_view<Goldilocks2::Element>& input, // [N, C]
    const array_view<Goldilocks2::Element>& weights, // [C, OC]
    const array_view<Goldilocks2::Element>& output, // [N, OC]
    size_t n_samples
);

bool check_softmax(
    const array_view<Goldilocks2::Element>& input, // [N, C]
    const array_view<Goldilocks2::Element>& output, // [N, C]
    const array_view<Goldilocks2::Element>& label,
    const std::vector<Goldilocks2::Element>& e_pow_inv,
    int64_t scale
);
