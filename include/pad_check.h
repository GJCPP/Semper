#pragma once

#include <cassert>
#include "mle.h"

bool execute_pad_check(
    Goldilocks2::Element claimed_Xr,
    const oracle_ext *oracle,
    int begin, int end,
    const std::vector<Goldilocks2::Element>& r,
    const size_t sec_param);

bool execute_pad_check(
    Goldilocks2::Element claimed_Xr,
    const oracle_ext *oracle,
    const std::vector<std::array<int, 2>>& ranges,
    const std::vector<Goldilocks2::Element>& r,
    const size_t sec_param);
    