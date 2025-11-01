#pragma once

#include "goldilocks_quadratic_ext.h"
#include "mle.h"
#include "mle_eq.h"
#include "ligero.h"
#include "protocol.h"
#include <array>
#include <vector>
#include <optional>

/*
prover for sumcheck of product of three multilinear polynomials in O(3 * 2^l) time
*/
class p3Prover{
public:
    p3Prover(const MultilinearPolynomial& p1, const MultilinearPolynomial& p2, const MultilinearPolynomial& p3);
    p3Prover(MultilinearPolynomial&& p1, MultilinearPolynomial&& p2, MultilinearPolynomial&& p3);
    p3Prover() = default;
    std::array<Goldilocks2::Element, 4> send_message(const size_t& round,const std::vector<Goldilocks2::Element>& rands);
    Goldilocks2::Element get_sum() const { return sum; }
    size_t get_rounds() const { return nrnd; }
    
private:
    void initialize();

    MultilinearPolynomial p1, p2, p3;
    
    inline void shrinkTable(const Goldilocks2::Element& r);
    static inline Goldilocks2::Element mul(const Goldilocks2::Element& e1, const  Goldilocks2::Element& e2, const  Goldilocks2::Element& e3);
    static inline Goldilocks2::Element lincomb(const Goldilocks2::Element& e1, const  Goldilocks2::Element& e0, const  uint64_t& r);
    size_t nrnd;
    Goldilocks2::Element sum;
};

class p3Verifier{
public:
    // should be replaced with a pcs
    // typedef std::array<ligeropcs, 3> Oracle;
    static bool execute_sumcheck(p3Prover& pr, const std::array<const oracle*, 3>& oracle, const size_t& sec_param);
    static bool execute_sumcheck(p3Prover& pr, Goldilocks2::Element claim, const std::array<const oracle*, 3>& oracle, const size_t& sec_param);

    static std::optional<challenge_claim> partial_sumcheck(p3Prover& pr, const size_t& sec_param);
    static std::optional<challenge_claim> partial_sumcheck(p3Prover& pr, Goldilocks2::Element claim, const size_t& sec_param);
    
    // customized sumcheck for \Sigma eq * frac * (gamma - p1 - lambda * p2)
    static bool execute_logup_sumcheck(
        p3Prover& pr,
        const MLE_Eq_Oracle& eqr,
        const oracle& frac,
        const oracle& p1,
        const oracle& p2,
        const Goldilocks2::Element gamma,
        const Goldilocks2::Element labmda,
        const size_t& sec_param
    );
private:
    static Goldilocks2::Element challenge();
    static inline void interpolate_3(Goldilocks2::Element& fr, const Goldilocks2::Element& r, const Goldilocks2::Element& f1, const Goldilocks2::Element& f2, const Goldilocks2::Element& f3, const Goldilocks2::Element& f4);
};

bool prove_mle_product(const MultilinearPolynomial& prod,
    const MultilinearPolynomial& p1, 
    const MultilinearPolynomial& p2, 
    const oracle& o_prod, 
    const oracle& o_p1, 
    const oracle& o_p2, 
    size_t sec_param);

class p3proto : public protocol {
public:
    p3proto(p3Prover&& _prover, std::array<std::shared_ptr<oracle>, 3> _oracles, Goldilocks2::Element _claim, uint64_t _sec_param)
        : protocol(_sec_param), prover(std::move(_prover)), oracles(_oracles), claim(_claim), has_claim(true) {
        ;
    }
    p3proto(p3Prover&& _prover, std::array<std::shared_ptr<oracle>, 3> _oracles, uint64_t _sec_param)
        : protocol(_sec_param), prover(std::move(_prover)), oracles(_oracles), has_claim(false) {
        ;
    }
    bool execute() override {
        if (has_claim) return p3Verifier::execute_sumcheck(prover, claim, {oracles[0].get(), oracles[1].get(), oracles[2].get()}, sec_param);
        return p3Verifier::execute_sumcheck(prover, {oracles[0].get(), oracles[1].get(), oracles[2].get()}, sec_param);
    }
private:
    p3Prover prover;
    std::array<std::shared_ptr<oracle>, 3> oracles;
    Goldilocks2::Element claim;
    bool has_claim;
};

class logup_sum_proto : public protocol {
public:
    logup_sum_proto(p3Prover&& _prover, 
                const MLE_Eq_Oracle& _eqr,
                std::shared_ptr<oracle> _frac,
                std::shared_ptr<oracle> _p1,
                std::shared_ptr<oracle> _p2,
                Goldilocks2::Element _gamma,
                Goldilocks2::Element _lambda,
                uint64_t _sec_param)
        : protocol(_sec_param), prover(std::move(_prover)), eqr(_eqr), frac(_frac), p1(_p1), p2(_p2), gamma(_gamma), lambda(_lambda) {
        ;
    }
    
    bool execute() override {
        return p3Verifier::execute_logup_sumcheck(prover, eqr, *frac, *p1, *p2, gamma, lambda, sec_param);
    }
    
private:
    p3Prover prover;
    MLE_Eq_Oracle eqr;
    std::shared_ptr<oracle> frac;
    std::shared_ptr<oracle> p1;
    std::shared_ptr<oracle> p2;
    Goldilocks2::Element gamma;
    Goldilocks2::Element lambda;
};



