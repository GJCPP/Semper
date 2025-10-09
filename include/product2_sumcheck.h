#pragma once

#include <array>
#include <vector>
#include <optional>

#include "goldilocks_quadratic_ext.h"
#include "mle.h"
#include "ligero.h"
#include "protocol.h"

/*
prover for sumcheck of product of two multilinear polynomials in O(2 * 2^l) time
*/
class p2Prover{
public:
    p2Prover(std::unique_ptr<MultilinearPolynomial> p1, std::unique_ptr<MultilinearPolynomial> p2);
    std::array<Goldilocks2::Element, 3> send_message(const size_t& round,const std::vector<Goldilocks2::Element>& rands);
    Goldilocks2::Element get_sum() const { return sum; }
    size_t get_rounds() const { return nrnd; }
    
// private:
    void initialize();

    std::unique_ptr<MultilinearPolynomial> p1;
    std::unique_ptr<MultilinearPolynomial> p2;
    
    inline void shrinkTable(const Goldilocks2::Element& r);
    static inline Goldilocks2::Element lincomb(const Goldilocks2::Element& e1, const  Goldilocks2::Element& e0, const  uint64_t& r);
    Goldilocks2::Element sum;
    size_t nrnd;
};

class p2Verifier {
public:
    static bool execute_sumcheck(p2Prover& pr, const std::array<const oracle*, 2>& oracle, const size_t& sec_param);
    static bool execute_sumcheck(p2Prover& pr, const std::array<const oracle*, 2>& oracle, Goldilocks2::Element claim, const size_t& sec_param);

    // Return nullopt for failure.
    static std::optional<challenge_claim> partial_sumcheck(p2Prover& pr, const size_t& sec_param);
    static std::optional<challenge_claim> partial_sumcheck(p2Prover& pr, Goldilocks2::Element claim, const size_t& sec_param);

private:
    static Goldilocks2::Element challenge();

    static inline void interpolate_2(Goldilocks2::Element& fr, const Goldilocks2::Element& r, const Goldilocks2::Element& f1, const Goldilocks2::Element& f2, const Goldilocks2::Element& f3);
};

class p2proto : public protocol {
public:
    p2proto(p2Prover&& _prover, std::array<std::shared_ptr<oracle>, 2> _oracles, Goldilocks2::Element _claim, uint64_t _sec_param)
        : protocol(_sec_param), prover(std::move(_prover)), oracles(_oracles), claim(_claim), has_claim(true) {
        ;
    }
    p2proto(p2Prover&& _prover, std::array<std::shared_ptr<oracle>, 2> _oracles, uint64_t _sec_param)
        : protocol(_sec_param), prover(std::move(_prover)), oracles(_oracles), has_claim(false) {
        ;
    }
    bool execute() override {
        if (has_claim) return p2Verifier::execute_sumcheck(prover, {oracles[0].get(), oracles[1].get()}, claim, sec_param);
        return p2Verifier::execute_sumcheck(prover, {oracles[0].get(), oracles[1].get()}, sec_param);
    }
private:
    p2Prover prover;
    std::array<std::shared_ptr<oracle>, 2> oracles;
    Goldilocks2::Element claim;
    bool has_claim;
};
