#pragma once

#include <vector>
#include <array>
#include <optional>

#include "mle.h"
#include "goldilocks_quadratic_ext.h"
#include "ligero.h"
#include "protocol.h"

class sProver{
public:
    sProver(const MultilinearPolynomial& g);
    sProver(MultilinearPolynomial&& g);
    sProver() = default;
    void initialize();
    std::array<Goldilocks2::Element, 2> send_message(const size_t& round,const std::vector<Goldilocks2::Element>& rands);
    Goldilocks2::Element get_sum() const { return sum; }
    size_t get_rounds() const { return nrnd; }
private:
    MultilinearPolynomial g;
    size_t nrnd;
    Goldilocks2::Element sum;
};

class sVerifier{
public:
    static bool execute_sumcheck(sProver& pr, const oracle& oracle, const size_t& sec_param);

    // Return nullopt for failure.
    static bool partial_sumcheck(sProver& pr, std::vector<Goldilocks2::Element>& challenges, Goldilocks2::Element& claim, const size_t& sec_param);
private:
    static Goldilocks2::Element challenge();
};

class sproto : public protocol {
public:
    sproto(sProver&& _prover, std::shared_ptr<oracle> _oracle, uint64_t _sec_param)
        : protocol(_sec_param), prover(std::move(_prover)), oracle_ptr(_oracle) {
        ;
    }
    
    bool execute() override {
        return sVerifier::execute_sumcheck(prover, *oracle_ptr, sec_param);
    }
    
private:
    sProver prover;
    std::shared_ptr<oracle> oracle_ptr;
};

