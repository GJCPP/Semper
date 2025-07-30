#pragma once

#include "goldilocks_quadratic_ext.h"
#include "mle.h"
#include "mle_eq.h"
#include "ligero.h"
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
        const MLE_Eq& eqr,
        const ligeropcs_ext& frac,
        const ligeropcs_base& p1,
        const ligeropcs_base& p2,
        const Goldilocks2::Element gamma,
        const Goldilocks2::Element labmda,
        const size_t& sec_param
    );
private:
    static Goldilocks2::Element challenge();
    static inline void interpolate_3(Goldilocks2::Element& fr, const Goldilocks2::Element& r, const Goldilocks2::Element& f1, const Goldilocks2::Element& f2, const Goldilocks2::Element& f3, const Goldilocks2::Element& f4);
};


