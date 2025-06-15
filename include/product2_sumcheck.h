#pragma once

#include "goldilocks_quadratic_ext.h"
#include "mle.h"
#include "ligero.h"
#include <array>
#include <vector>

/*
prover for sumcheck of product of two multilinear polynomials in O(2 * 2^l) time
*/
class p2Prover{
public:
    p2Prover(const MultilinearPolynomial& p1, const MultilinearPolynomial& p2);
    void initialize();
    std::array<Goldilocks2::Element, 3> send_message(const size_t& round,const std::vector<Goldilocks2::Element>& rands);
    Goldilocks2::Element get_sum() const { return sum; }
    size_t get_rounds() const { return nrnd; }
private:
    MultilinearPolynomial p1;
    MultilinearPolynomial p2;
    
    std::vector<Goldilocks2::Element> keepTablep1;
    std::vector<Goldilocks2::Element> keepTablep2;
    inline void shrinkTable(const Goldilocks2::Element& r, const uint64_t& offset);
    static inline Goldilocks2::Element lincomb(const Goldilocks2::Element& e1, const  Goldilocks2::Element& e0, const  uint64_t& r);
    Goldilocks2::Element sum;
    size_t nrnd;
};

class p2Verifier{
public:
    static bool execute_sumcheck(p2Prover& pr, const std::array<std::shared_ptr<oracle_base>, 2>& oracle, const size_t& sec_param);
    static bool execute_sumcheck(p2Prover& pr, const std::array<std::shared_ptr<oracle_ext>, 2>& oracle, const size_t& sec_param);

private:
    static Goldilocks2::Element challenge();

    static inline void interpolate_2(Goldilocks2::Element& fr, const Goldilocks2::Element& r, const Goldilocks2::Element& f1, const Goldilocks2::Element& f2, const Goldilocks2::Element& f3);
};