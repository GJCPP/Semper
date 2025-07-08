#include "pad_check.h"

bool execute_pad_check(
    Goldilocks2::Element claimed_Xr,
    const oracle* oracle,
    int begin,
    int end,
    const std::vector<Goldilocks2::Element>& r,
    const size_t sec_param) {


        
    int num_vars = r.size();
    assert(begin >= 0 && end <= num_vars);
    assert(begin <= end);

    std::vector<Goldilocks2::Element> cha;
    cha.reserve(num_vars - (end - begin));
    Goldilocks2::Element factor = Goldilocks2::one();
    for (int i = 0; i < num_vars; ++i) {
        if (i >= begin && i < end) {
            factor = factor * (Goldilocks2::one() - r[i]);
        } else {
            cha.push_back(r[i]);
        }
    }
    factor *= oracle->open(cha, sec_param);
    return claimed_Xr == factor;
}

bool execute_pad_check(
    Goldilocks2::Element claimed_Xr,
    const oracle* oracle,
    const std::vector<std::array<int, 2>>& ranges,
    const std::vector<Goldilocks2::Element>& r,
    const size_t sec_param) {

    int num_vars = r.size();
    auto ra = ranges.begin(), range_end = ranges.end();
    auto [begin, end] = *ra;
    std::vector<Goldilocks2::Element> cha;
    cha.reserve(num_vars);
    Goldilocks2::Element factor = Goldilocks2::one();
    for (int i = 0; i < num_vars; ) {
        if (i >= begin && i < end) {
            factor = factor * (Goldilocks2::one() - r[i]);
            ++i;
        } else if (i >= end && ra + 1 != range_end) {
            begin = ra->at(0);
            end = ra->at(1);
        } else {
            cha.push_back(r[i]);
            ++i;
        }
    }
    factor *= oracle->open(cha, sec_param);
    return claimed_Xr == factor;
}
