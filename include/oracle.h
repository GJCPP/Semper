#pragma once

#include "goldilocks_quadratic_ext.h"
#include <vector>


/*
 * Base class for oracle for quadratic extension field
 * To be inherited by the ligero (if it's a commitment) or the mle (if it's a known polynomial)
*/
class oracle {
public:
    virtual Goldilocks2::Element open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const = 0;
};

class challenge_claim {
public:
    std::vector<Goldilocks2::Element> challenges;
    Goldilocks2::Element claim;
};

class commitment : public oracle {
public:
    virtual bool check_open(
        const std::vector<Goldilocks2::Element>& challenges,
        const Goldilocks2::Element& claim,
        const size_t& sec_param) const = 0;
};
