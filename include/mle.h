#pragma once

#include <string>
#include <cstdint>
#include "goldilocks_quadratic_ext.h"
#include "oracle.h"

// store a multilinear polynomial in a vector of evaluations
class MultilinearPolynomial : public oracle_ext {
public:
    MultilinearPolynomial(size_t num_vars, bool init=true);
    MultilinearPolynomial(const std::vector<Goldilocks2::Element>& evaluations);
    MultilinearPolynomial(const std::vector<uint64_t>& val_table);
    size_t get_num_vars() const { return num_vars; }

    // only for debug use
    void set_value(const std::string& mask, const Goldilocks2::Element& c);
    void set_value(const std::string& mask, const uint64_t& c);


    virtual Goldilocks2::Element& operator[](size_t idx) { return evaluations[idx]; }
    virtual const Goldilocks2::Element& operator[](size_t idx) const { return evaluations[idx]; }
    
    virtual Goldilocks2::Element eval_hypercube(uint64_t mask) const;

    // Cost O(2^num_vars)
    virtual Goldilocks2::Element evaluate(const std::vector<Goldilocks2::Element>& point) const;

    Goldilocks2::Element open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const override;
    
    virtual std::vector<Goldilocks2::Element> get_eval_table() const { return evaluations; }
    virtual MultilinearPolynomial operator+(const MultilinearPolynomial& g) const;
    virtual MultilinearPolynomial operator-(const MultilinearPolynomial& g) const;

    // Cost O(2^num_vars)
    virtual Goldilocks2::Element get_sum() const;

    // Fix the pos-th variable to val, and compute new evaluation table
    void fix(size_t pos, const Goldilocks2::Element& val);
    void fix(size_t pos, const std::vector<Goldilocks2::Element>& val);

    // Sum over the last len bits
    MultilinearPolynomial sum_over_lowbits(size_t len) const;

    // Power over the last len bits
    MultilinearPolynomial sum_over_lowbits_with_power(size_t len, Goldilocks2::Element beta) const;

protected:
    // evaliations over the hypercube
    std::vector<Goldilocks2::Element> evaluations;
    size_t num_vars;
};
