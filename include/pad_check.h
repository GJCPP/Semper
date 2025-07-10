#pragma once

#include <cassert>
#include "mle.h"

challenge_claim execute_pad_check(
    Goldilocks2::Element claimed_Xr,
    int begin, int end,
    const std::vector<Goldilocks2::Element>& r,
    const size_t sec_param);

challenge_claim execute_pad_check(
    Goldilocks2::Element claimed_Xr,
    const std::vector<std::array<int, 2>>& ranges,
    const std::vector<Goldilocks2::Element>& r,
    const size_t sec_param);
    