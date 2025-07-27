#pragma once

#include "goldilocks_quadratic_ext.h"
#include <vector>


/*
 * Base class for oracle for quadratic extension field
 * To be inherited by the mle (if it's a known polynomial)
*/
class oracle {
public:
    virtual Goldilocks2::Element open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const = 0;
    // virtual int get_num_vars() const = 0;
};

class challenge_claim {
public:
    std::vector<Goldilocks2::Element> challenges;
    Goldilocks2::Element claim;
};

// Auxiliary information returned by MLE challenge processing
struct mle_aux_info {
    std::vector<Goldilocks2::Element> r;
    // Additional auxiliary data can be added here as needed
    Goldilocks2::Element comp; // compensation x open_val = claim
};
