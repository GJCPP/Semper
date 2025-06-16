#pragma once

#include <vector>
#include <array>
#include <optional>

#include "mle.h"
#include "goldilocks_quadratic_ext.h"
#include "ligero.h"

class sProver{
public:
    sProver(const MultilinearPolynomial& g);
    void initialize();
    std::array<Goldilocks2::Element, 2> send_message(const size_t& round,const std::vector<Goldilocks2::Element>& rands);
    Goldilocks2::Element get_sum() const { return sum; }
    size_t get_rounds() const { return nrnd; }
private:
    MultilinearPolynomial g;
    std::vector<Goldilocks2::Element> keepTable;
    Goldilocks2::Element sum;
    size_t nrnd;
};

class sVerifier{
public:
    static bool execute_sumcheck(sProver& pr, const oracle_base& oracle, const size_t& sec_param);
    static bool execute_sumcheck(sProver& pr, const oracle_ext& oracle, const size_t& sec_param);

    static std::optional<challenge_claim> execute_sumcheck(sProver& pr, Goldilocks2::Element claim, const size_t& sec_param);
private:
    static Goldilocks2::Element challenge();
};