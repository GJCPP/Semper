#pragma once

#include <string>
#include <cstdint>
#include <memory>
#include <functional>
#include "goldilocks_quadratic_ext.h"
#include "oracle.h"

#include "array_view.h"

class ligeropcs_base;
class ligeropcs_ext;

// store a multilinear polynomial in a vector of evaluations
class MultilinearPolynomial : public oracle {
public:
    MultilinearPolynomial() : num_vars(0) {}
    MultilinearPolynomial(size_t num_vars, bool init=true);
    MultilinearPolynomial(const std::vector<Goldilocks2::Element>& evaluations);
    MultilinearPolynomial(const std::vector<std::vector<Goldilocks2::Element>>& evaluations);
    MultilinearPolynomial(const std::vector<std::vector<std::vector<Goldilocks2::Element>>>& evaluations);
    MultilinearPolynomial(const std::vector<uint64_t>& val_table);
    MultilinearPolynomial(const array_view<Goldilocks2::Element>& evaluations);
    size_t get_num_vars() const { return num_vars; }

    virtual std::unique_ptr<MultilinearPolynomial> clone() const {
        return std::make_unique<MultilinearPolynomial>(*this);
    }

    // only for debug use
    void set_value(const std::string& mask, const Goldilocks2::Element& c);
    void set_value(const std::string& mask, const uint64_t& c);


    virtual Goldilocks2::Element& operator[](size_t idx) { return evaluations[idx]; }
    virtual const Goldilocks2::Element& operator[](size_t idx) const { return evaluations[idx]; }
    
    virtual Goldilocks2::Element eval_hypercube(uint64_t mask) const;

    // Cost O(2^num_vars)
    virtual Goldilocks2::Element evaluate(const std::vector<Goldilocks2::Element>& point) const;

    Goldilocks2::Element open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const override;
    
    virtual MultilinearPolynomial operator+(const MultilinearPolynomial& g) const;
    virtual MultilinearPolynomial operator-(const MultilinearPolynomial& g) const;

    // Cost O(2^num_vars)
    virtual Goldilocks2::Element get_sum() const;

    // Iterate through all evaluations with their indices
    virtual void iterate_nonzero(const std::function<void(size_t)> f, size_t offset) const;

    // Fix the pos-th variable to val, and compute new evaluation table
    virtual void fix(size_t pos, const Goldilocks2::Element& val);
    virtual void fix(size_t pos, const std::vector<Goldilocks2::Element>& val);
    virtual void fix(size_t pos, const std::vector<Goldilocks2::Element>::const_iterator begin, const std::vector<Goldilocks2::Element>::const_iterator end);

    // Sum over the last len bits
    virtual MultilinearPolynomial sum_over_lowbits(size_t len) const;

    // Power over the last len bits
    virtual MultilinearPolynomial sum_over_lowbits_with_power(size_t len, Goldilocks2::Element beta) const;

    const std::vector<Goldilocks2::Element>& get_eval_table() const;

    virtual bool check_open(
        const ligeropcs_base *pcs,
        const std::vector<Goldilocks2::Element>& challenges,
        const Goldilocks2::Element& claim,
        const size_t& sec_param) const;

    virtual bool check_open(
        const ligeropcs_ext *pcs,
        const std::vector<Goldilocks2::Element>& challenges,
        const Goldilocks2::Element& claim,
        const size_t& sec_param) const;

protected:
    virtual std::vector<Goldilocks2::Element> get_open_r(const std::vector<Goldilocks2::Element>& challenges) const;

    // evaliations over the hypercube
    std::vector<Goldilocks2::Element> evaluations;

    array_view<Goldilocks2::Element> evaluations_view;
    
    // used to assist openning proof. with respect to pcs.
    // array_view<Goldilocks2::Element> evaluations_view;

    size_t num_vars;
};
