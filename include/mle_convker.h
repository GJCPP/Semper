#pragma once

#include <vector>
#include <optional>

#include "util.h"
#include "mle.h"
#include "mle_eq.h"
// The MLE that encapsulates the 2D conv kernel
class MLE_Convker : public MultilinearPolynomial {
public:
    // Input: n x n, Kernel: m x m
    MLE_Convker(const std::vector<std::vector<std::vector<std::vector<Goldilocks2::Element>>>>& kernel, size_t C, size_t D, size_t n, size_t m);

    MLE_Convker(const array_view<Goldilocks2::Element>& kernel, size_t C, size_t D, size_t n, size_t m);

    std::unique_ptr<MultilinearPolynomial> clone() const override {
        return std::make_unique<MLE_Convker>(*this);
    }

    Goldilocks2::Element eval_hypercube(size_t i) const override;
    // Goldilocks2::Element evaluate(const std::vector<Goldilocks2::Element>& input) const override;

    Goldilocks2::Element evaluate(const std::vector<Goldilocks2::Element>& point) const override;

    void fix(size_t pos, const Goldilocks2::Element& val) override;

    // Sum over the last len bits
    // virtual MultilinearPolynomial sum_over_lowbits(size_t len) const override;

    // Power over the last len bits
    MultilinearPolynomial sum_over_lowbits_with_power(size_t len, Goldilocks2::Element beta) const override;

    // Goldilocks2::Element evaluate(const std::vector<Goldilocks2::Element>& input) const override;

    void iterate_nonzero(const std::function<void(size_t)> f, size_t offset) const override;

protected:
    size_t C, D, n, m;
    int logC, logD, lognm, logmm;
    bool expanded;

    void expand();

    std::optional<size_t> to_real_index(size_t i) const;
    size_t to_virtual_index(size_t i) const;
};