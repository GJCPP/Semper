#pragma once

#include "ligero.h"

class convProver {
public:
    // Prove c = \sum a * b, where |c| = |a| + |b| - 1
    convProver(const MultilinearPolynomial& c);

private:
    std::vector<MultilinearPolynomial> a, b; // Multi-channel
    MultilinearPolynomial c;
};

