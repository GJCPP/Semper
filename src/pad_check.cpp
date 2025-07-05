#include "pad_check.h"

bool execute_pad_check(
    Goldilocks2::Element claimed_Xr,
    const oracle_ext* oracle,
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