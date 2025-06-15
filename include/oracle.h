#pragma once

#include "goldilocks_quadratic_ext.h"
#include <vector>

/*
 * Base class for oracle for base field
 * To be inherited by the ligero (if it's a commitment) or the mle (if it's a known polynomial)
*/
class oracle_base {
public:
    virtual Goldilocks2::Element open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const = 0;
};

/*
 * Base class for oracle for quadratic extension field
 * To be inherited by the ligero (if it's a commitment) or the mle (if it's a known polynomial)
*/
class oracle_ext {
public:
    virtual Goldilocks2::Element open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const = 0;
};
