#pragma once

#include <string>
#include <cstdint>
#include <memory>
#include <functional>
#include "goldilocks_quadratic_ext.h"
#include "oracle.h"

#include "array_view.h"

// store a multilinear polynomial in a vector of evaluations
template <typename field>
class MultilinearPolynomial : public oracle<field> {
public:
    MultilinearPolynomial() : num_vars(0) {}
    MultilinearPolynomial(int num_vars, bool init=true);
    MultilinearPolynomial(const std::vector<field>& evaluations);
    MultilinearPolynomial(const std::vector<std::vector<field>>& evaluations);
    MultilinearPolynomial(const std::vector<std::vector<std::vector<field>>>& evaluations);
    MultilinearPolynomial(const std::vector<uint64_t>& val_table);
    MultilinearPolynomial(const array_view<field>& evaluations);
    inline int get_num_vars() const override { return num_vars; }

    virtual std::unique_ptr<MultilinearPolynomial> clone() const {
        return std::make_unique<MultilinearPolynomial>(*this);
    }

    // only for debug use
    void set_value(const std::string& mask, const field& c);
    void set_value(const std::string& mask, const uint64_t& c);


    virtual field& operator[](size_t idx) { return evaluations[idx]; }
    virtual const field& operator[](size_t idx) const { return evaluations[idx]; }
    
    virtual field eval_hypercube(uint64_t mask) const;

    // Cost O(2^num_vars)
    virtual field evaluate(const std::vector<field>& point) const;

    field open(const std::vector<field>& z, const size_t& sec_param) const override;
    
    virtual MultilinearPolynomial operator+(const MultilinearPolynomial& g) const;
    virtual MultilinearPolynomial operator-(const MultilinearPolynomial& g) const;

    virtual MultilinearPolynomial operator*(const MultilinearPolynomial& g) const;
    virtual MultilinearPolynomial operator*(size_t scale) const;

    // Cost O(2^num_vars)
    virtual field get_sum() const;

    // Iterate through all evaluations with their indices
    virtual void iterate_nonzero(const std::function<void(size_t)> f, size_t offset) const;

    // Fix the pos-th variable to val, and compute new evaluation table
    virtual void fix(size_t pos, const field& val);
    virtual void fix(size_t pos, const std::vector<field>& val);
    virtual void fix(size_t pos, const std::vector<field>::const_iterator begin, const std::vector<field>::const_iterator end);

    // Sum over the last len bits
    virtual MultilinearPolynomial sum_over_lowbits(size_t len) const;

    virtual MultilinearPolynomial prod_over_lowbits(size_t len) const;

    // Power over the last len bits
    virtual MultilinearPolynomial sum_over_lowbits_with_power(size_t len, field beta) const;

    const std::vector<field>& get_eval_table() const;

    // New method to process challenges and return auxiliary information
    virtual mle_aux_info<field> process_challenges(
        const std::vector<field>& challenges) const;

protected:

    // evaliations over the hypercube
    std::vector<field> evaluations;

    // array_view<field> evaluations_view;
    
    // used to assist openning proof. with respect to pcs.
    // array_view<field> evaluations_view;

    int num_vars;
};

template <typename field>
using MLE = MultilinearPolynomial<field>;

template <typename field>
class MLE_Sum : public MLE<field> {
public:
    MLE_Sum() = default;

    void add(const MLE<field>& mle, field coeff = Goldilocks2::one()) {
        if (evaluations.empty()) {
            num_vars = mle.get_num_vars();
            evaluations.resize(1ull << num_vars, Goldilocks2::zero());
        } else if (num_vars != mle.get_num_vars()) {
            throw std::runtime_error("MLE_Sum::add: number of variables does not match.");
        }
        size_t sz = 1ull << num_vars;
        for (size_t i = 0; i < sz; ++i) {
            evaluations[i] += coeff * mle[i];
        }
    }
    
    inline int get_num_vars() const override {
        if (evaluations.empty()) {
            throw std::runtime_error("MLE_Sum::get_num_vars: no MLE added.");
        }
        return num_vars;
    }
};

// mle_aux_info process_challenges(
//     int n,
//     const std::vector<size_t>& shape,
//     const std::vector<int>& order,
//     const std::vector<bool>& reversed,
//     const std::vector<field>& challenges);
    