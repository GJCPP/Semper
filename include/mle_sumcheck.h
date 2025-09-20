#pragma once

#include <vector>
#include <array>
#include <optional>

#include "mle.h"
#include "goldilocks_quadratic_ext.h"
#include "ligero.h"

template <typename field>
class sProver{
public:
    sProver(const MLE<field>& g);
    sProver(MLE<field>&& g);
    void initialize();
    std::array<field, 2> send_message(const size_t& round,const std::vector<field>& rands);
    field get_sum() const { return sum; }
    size_t get_rounds() const { return nrnd; }
private:
    MLE<field> g;
    size_t nrnd;
    field sum;
};

template <typename field>
class sVerifier{
public:
    static bool execute_sumcheck(sProver<field>& pr, const oracle<field>& oracle, const size_t& sec_param);

    // Return nullopt for failure.
    static bool partial_sumcheck(sProver<field>& pr, std::vector<field>& challenges, field& claim, const size_t& sec_param);
private:
    static field challenge();
};