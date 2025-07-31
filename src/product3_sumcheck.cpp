#include "product3_sumcheck.h"
#include "goldilocks_quadratic_ext.h"
#include "mle.h"
#include "mle_open.h"
#include <array>
#include <vector>
#include <random>
#include <cassert>

p3Prover::p3Prover(const MultilinearPolynomial& p1, const MultilinearPolynomial& p2, const MultilinearPolynomial& p3)
    : p1(p1), p2(p2), p3(p3), nrnd(p1.get_num_vars()), sum(Goldilocks2::zero()) {
    assert(p1.get_num_vars() == p2.get_num_vars() && p1.get_num_vars() == p3.get_num_vars());
    initialize();
}
p3Prover::p3Prover(MultilinearPolynomial&& p1, MultilinearPolynomial&& p2, MultilinearPolynomial&& p3)
    : p1(std::move(p1)), p2(std::move(p2)), p3(std::move(p3)), nrnd(p1.get_num_vars()), sum(Goldilocks2::zero()) {
    assert(p1.get_num_vars() == p2.get_num_vars() && p1.get_num_vars() == p3.get_num_vars());
    initialize();
}
/*
1. intialize the bookkeeping table A of g in time O(l * 2 ^ l) via zeta transform
2. calculate sum
*/
void p3Prover::initialize() {
    uint64_t tsize = 1ull << p1.get_num_vars();

    sum = Goldilocks2::zero();
    for (uint64_t mask = 0; mask < tsize; ++mask) {
        sum += p1[mask] * p2[mask] * p3[mask];
    }
}

inline void p3Prover::shrinkTable(const Goldilocks2::Element& r) {
    p1.fix(0, r);
    p2.fix(0, r);
    p3.fix(0, r);
}

inline Goldilocks2::Element p3Prover::mul(const Goldilocks2::Element& e1, const  Goldilocks2::Element& e2, const  Goldilocks2::Element& e3) {
    Goldilocks2::Element res;
    Goldilocks2::mul(res, e1, e2);
    Goldilocks2::mul(res, res, e3);
    return res;
}


/*
linear combination
returns r * e1 + (1-r)e0
*/
inline Goldilocks2::Element p3Prover::lincomb(const Goldilocks2::Element& e1, const  Goldilocks2::Element& e0, const uint64_t& r) {
    Goldilocks2::Element res, A, B, oneminusr;
    Goldilocks2::sub(oneminusr, Goldilocks2::one(), r);
    Goldilocks2::mul(A, e1, r);
    Goldilocks2::mul(B, e0, oneminusr);
    Goldilocks2::add(res, A, B);
    return res;
}

std::array<Goldilocks2::Element, 4> p3Prover::send_message(const size_t& round, const std::vector<Goldilocks2::Element>& rands) {
    std::array<Goldilocks2::Element, 4> s = { 0, 0, 0, 0 };

    // notation referce: Libra(https://eprint.iacr.org/2019/317.pdf) Algorithm 1
    uint64_t offset = 1ull << (nrnd - round);

    if (round > 1) {
        // namely r_{i-1}
        shrinkTable(rands.back());
    }

    /*
     * sending f(0), f(1), f(2), f(3), where f = g_1 x g_2 x g_3
     * Note that g_i is linear, so g(2) = 2 * g(1) - g(0), etc.
     */ 
    for (uint64_t b = 0; b < offset; ++b) {
        s[0] += p1[b] * p2[b] * p3[b];
        s[1] += p1[b + offset] * p2[b + offset] * p3[b + offset];
        s[2] += lincomb(p1[b + offset], p1[b], 2) * lincomb(p2[b + offset], p2[b], 2) * lincomb(p3[b + offset], p3[b], 2);
        s[3] += lincomb(p1[b + offset], p1[b], 3) * lincomb(p2[b + offset], p2[b], 3) * lincomb(p3[b + offset], p3[b], 3);
    }
    return s;
}

/*
evaluate f(r) given f(0,1,2,3) when f is cubic
*/
inline void p3Verifier::interpolate_3(Goldilocks2::Element& fr, const Goldilocks2::Element& r, const Goldilocks2::Element& f0, const Goldilocks2::Element& f1, const Goldilocks2::Element& f2, const Goldilocks2::Element& f3) {
    uint64_t u6[] = { 15372286724512153601ull, 0 }, u2[] = { 9223372034707292161ull, 0 };
    Goldilocks2::Element inv6 = Goldilocks2::fromU64(u6), inv2 = Goldilocks2::fromU64(u2), minv6, minv2, x0, x1, x2, x3;
    Goldilocks2::neg(minv6, inv6);
    Goldilocks2::neg(minv2, inv2);

    Goldilocks2::sub(x0, r, 0);
    Goldilocks2::sub(x1, r, 1);
    Goldilocks2::sub(x2, r, 2);
    Goldilocks2::sub(x3, r, 3);
    fr = (x1 * x2 * x3 * minv6 * f0) +
         (x0 * x2 * x3 * inv2 * f1) +
         (x0 * x1 * x3 * minv2 * f2) +
         (x0 * x1 * x2 * inv6 * f3);
}

bool p3Verifier::execute_sumcheck(p3Prover& pr, const std::array<const oracle*, 3>& oracle, const size_t& sec_param) {

    Goldilocks2::Element sum = pr.get_sum();
    return execute_sumcheck(pr, sum, oracle, sec_param);
}

bool p3Verifier::execute_sumcheck(p3Prover& pr, Goldilocks2::Element claim, const std::array<const oracle*, 3>& oracle, const size_t& sec_param) {

    auto cha = partial_sumcheck(pr, claim, sec_param);
    if (!cha || cha->claim != oracle[0]->open(cha->challenges, sec_param) * oracle[1]->open(cha->challenges, sec_param) * oracle[2]->open(cha->challenges, sec_param)) {
        return false;
    }
    return true;
}

std::optional<challenge_claim> p3Verifier::partial_sumcheck(p3Prover& pr, const size_t& sec_param) {

    Goldilocks2::Element sum = pr.get_sum();
    return partial_sumcheck(pr, sum, sec_param);
}

std::optional<challenge_claim> p3Verifier::partial_sumcheck(p3Prover& pr, Goldilocks2::Element claim, const size_t& sec_param) {

    size_t nrnd = pr.get_rounds();
    std::vector<Goldilocks2::Element> challenges;

    // s_{i - 1}
    std::array<Goldilocks2::Element, 4> si1 = { Goldilocks2::zero(), Goldilocks2::zero(), Goldilocks2::zero(), Goldilocks2::zero() };
    for (size_t round = 1; round <= nrnd; ++round) {
        // s_i
        std::array<Goldilocks2::Element, 4> si;
        si = pr.send_message(round, challenges);
        // s(0) + s(1)
        Goldilocks2::Element ss;
        Goldilocks2::add(ss, si[0], si[1]);
        if (round == 1) {
            if (!(ss == claim)) return {};
        }
        else {
            Goldilocks2::Element sr;
            Goldilocks2::Element r = challenges[round - 2];
            interpolate_3(sr, r, si1[0], si1[1], si1[2], si1[3]);
            if (!(sr == ss)) return {};

            // final check
            if (round == nrnd) {
                challenges.push_back(challenge());
                Goldilocks2::Element slrl;
                Goldilocks2::Element rl = challenges[round - 1];
                interpolate_3(slrl, rl, si[0], si[1], si[2], si[3]);
                challenge_claim ret{.challenges = challenges, .claim = slrl};
                return ret;
            }
        }

        challenges.push_back(challenge());
        // goto next round
        si1 = si;
    }

    Goldilocks2::Element slrl;
    Goldilocks2::Element rl = challenges.back();
    interpolate_3(slrl, rl, si1[0], si1[1], si1[2], si1[3]);
    challenge_claim ret{.challenges = challenges, .claim = slrl};
    return ret;
}


bool p3Verifier::execute_logup_sumcheck(
    p3Prover& pr,
    const MLE_Eq& eqr,
    const ligeropcs_ext& frac,
    const ligeropcs_base& p1,
    const ligeropcs_base& p2,
    const Goldilocks2::Element gamma,
    const Goldilocks2::Element labmda,
    const size_t& sec_param) {

    Goldilocks2::Element sum = pr.get_sum();
    size_t nrnd = pr.get_rounds();
    std::vector<Goldilocks2::Element> challenges;

    // s_{i - 1}
    std::array<Goldilocks2::Element, 4> si1;
    for (size_t round = 1; round <= nrnd; ++round) {
        // s_i
        std::array<Goldilocks2::Element, 4> si;
        si = pr.send_message(round, challenges);
        // s(0) + s(1)
        Goldilocks2::Element ss;
        Goldilocks2::add(ss, si[0], si[1]);
        if (round == 1) {
            if (!(ss == sum)) return false;
        }
        else {
            Goldilocks2::Element sr;
            Goldilocks2::Element r = challenges[round - 2];
            interpolate_3(sr, r, si1[0], si1[1], si1[2], si1[3]);
            if (!(sr == ss)) return false;

            // final check, different from general product sumcheck
            if (round == nrnd) {
                challenges.push_back(challenge());

                // f(r) from the oracle and the information hold by the verifier
                Goldilocks2::Element third_term;
                Goldilocks2::Element tmp;
                Goldilocks2::mul(tmp, labmda, p2.open(challenges, sec_param));
                Goldilocks2::sub(third_term, gamma, p1.open(challenges, sec_param));
                Goldilocks2::sub(third_term, third_term, tmp);
                Goldilocks2::Element f_r = eqr.evaluate(challenges) * frac.open(challenges, sec_param) * third_term;


                // f(r) from the previous rounds
                Goldilocks2::Element slrl;
                Goldilocks2::Element rl = challenges[round - 1];
                interpolate_3(slrl, rl, si[0], si[1], si[2], si[3]);
                if (!(slrl == f_r)) return false;
            }
        }

        challenges.push_back(challenge());
        // goto next round
        si1 = si;
    }
    return true;
}

// we use goldilocks 2-extension, so no bother specifying the field
Goldilocks2::Element p3Verifier::challenge() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());

    // constexpr uint64_t MODULUS = Goldilocks2::p;
    // std::uniform_int_distribution<uint64_t> dist(0, MODULUS - 1);

    // uint64_t randn[] = { dist(gen), dist(gen) };
    return random_ext();
}

bool prove_mle_product(
    const MultilinearPolynomial& prod,
    const MultilinearPolynomial& p1,
    const MultilinearPolynomial& p2,
    ligeropcs_base o_prod,
    ligeropcs_base o_p1,
    ligeropcs_base o_p2,
    size_t sec_param) {

    int num_vars = prod.get_num_vars();
    if (num_vars != p1.get_num_vars() || num_vars != p2.get_num_vars()) {
        throw std::invalid_argument("prove_product: prod, p1, and p2 must have the same number of variables.");
    }
    std::vector<Goldilocks2::Element> cha = random_vec_ext(num_vars);
    MLE_Eq eq(cha);
    p3Prover prover(eq, p1, p2);
    auto claim = p3Verifier::partial_sumcheck(prover, o_prod.open(cha, sec_param), sec_param);
    if (!claim || eq.open(claim->challenges, sec_param) * 
        o_p1.open(claim->challenges, sec_param) *
        o_p2.open(claim->challenges, sec_param) != claim->claim) {
        return false;
    }
    return true;
}