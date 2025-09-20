#pragma once

#include <vector>
#include <optional>

#include "util.h"
#include "mle.h"
#include "mle_eq.h"


// The MLE that encapsulates the 2D conv kernel
template <typename field>
class MLE_Convker : public MLE<field> {
public:
    // Input: n x n, Kernel: m x m
    // MLE_Convker(const std::vector<std::vector<std::vector<std::vector<Goldilocks2::Element>>>>& kernel, size_t C, size_t D, size_t n, size_t m);

    MLE_Convker(const array_view<field>& kernel, size_t C, size_t D, size_t n, size_t m);

    std::unique_ptr<MultilinearPolynomial<field>> clone() const override {
        return std::make_unique<MLE_Convker<field>>(*this);
    }

    field eval_hypercube(size_t i) const override;
    // field evaluate(const std::vector<field>& input) const override;

    field evaluate(const std::vector<field>& point) const override;

    void fix(size_t pos, const field& val) override;

    // Sum over the last len bits
    // virtual MultilinearPolynomial sum_over_lowbits(size_t len) const override;

    // Power over the last len bits
    MultilinearPolynomial<field> sum_over_lowbits_with_power(size_t len, field beta) const override;

    // field evaluate(const std::vector<field>& input) const override;

    void iterate_nonzero(const std::function<void(size_t)> f, size_t offset) const override;

    void get_pad_range(int& begin, int& end) const;

    // Override the new challenge processing method
    mle_aux_info<field> process_challenges(
        const std::vector<field>& challenges) const override;

protected:
    size_t C, D, n, m;
    int logC, logD, logn, logm;
    bool expanded;

    void expand();

    std::optional<size_t> to_real_index(size_t i) const;
    size_t to_virtual_index(size_t i) const;
};
