#include "product2_sumcheck.h"
#include "goldilocks_quadratic_ext.h"
#include "mle.h"
#include "util.h"
#include <array>
#include <vector>
#include <random>
#include <cassert>

p2Prover::p2Prover(std::unique_ptr<MultilinearPolynomial> _p1, std::unique_ptr<MultilinearPolynomial> _p2)
    : p1(std::move(_p1)), p2(std::move(_p2)), nrnd(p1->get_num_vars()), sum(Goldilocks2::zero()) {

    assert(p1->get_num_vars() == p2->get_num_vars());
    initialize();
}

/*
1. intialize the bookkeeping table A of g in time O(l * 2 ^ l) via zeta transform
2. calculate sum
*/
void p2Prover::initialize() {
    uint64_t tsize = 1ull << p1->get_num_vars();

    sum = Goldilocks2::zero();
    for (uint64_t mask = 0; mask < tsize; ++mask) {
        sum += p1->eval_hypercube(mask) * p2->eval_hypercube(mask);
    }
    // std::cout << Goldilocks2::toString(sum) << '\n';
}

inline void p2Prover::shrinkTable(const Goldilocks2::Element& r) {
    p1->fix(0, r);
    p2->fix(0, r);
}


/*
linear combination
returns r * e1 + (1-r)e0
*/
inline Goldilocks2::Element p2Prover::lincomb(const Goldilocks2::Element& e1, const  Goldilocks2::Element& e0, const uint64_t& r) {
    Goldilocks2::Element re = Goldilocks2::fromU64(r);
    Goldilocks2::Element oneminusr = Goldilocks2::one() - re;
    return e1 * re + e0 * oneminusr;
}

std::array<Goldilocks2::Element, 3> p2Prover::send_message(const size_t& round, const std::vector<Goldilocks2::Element>& rands) {
    std::array<Goldilocks2::Element, 3> s = { 0, 0, 0 };

    uint64_t offset = 1ull << (nrnd - round);

    if (round > 1) {
        // namely r_{i-1}
        shrinkTable(rands.back());
    }

    for (uint64_t b = 0; b < offset; ++b) {
        s[0] += p1->eval_hypercube(b) * p2->eval_hypercube(b);
        s[1] += p1->eval_hypercube(b + offset) * p2->eval_hypercube(b + offset);
        s[2] += lincomb(p1->eval_hypercube(b + offset), p1->eval_hypercube(b), 2) * lincomb(p2->eval_hypercube(b + offset), p2->eval_hypercube(b), 2);
    }
    return s;
}


/*
evaluate f(r) given f(0,1,2) when f(x) = a * x^2 + b * x + c
*/
inline void p2Verifier::interpolate_2(Goldilocks2::Element& fr, const Goldilocks2::Element& r, const Goldilocks2::Element& f0, const Goldilocks2::Element& f1, const Goldilocks2::Element& f2) {
    uint64_t u2[] = { 9223372034707292161ull, 0 };
    Goldilocks2::Element inv2 = Goldilocks2::fromU64(u2);

    Goldilocks2::Element c = f0;
    Goldilocks2::Element a = inv2 * (f2 - f1 - f1 + c);
    Goldilocks2::Element b = f1 - a - c;
    fr = a * r * r + b * r + c;
}

bool p2Verifier::execute_sumcheck(p2Prover& pr, const std::array<const oracle_base*, 2>& oracle, const size_t& sec_param) {

    Goldilocks2::Element sum = pr.get_sum();
    size_t nrnd = pr.get_rounds();
    std::vector<Goldilocks2::Element> challenges;

    // s_{i - 1}
    std::array<Goldilocks2::Element, 3> si1;
    for (size_t round = 1; round <= nrnd; ++round) {
        // s_i
        std::array<Goldilocks2::Element, 3> si;
        si = pr.send_message(round, challenges);
        // s(0) + s(1)
        Goldilocks2::Element ss = si[0] + si[1];
        if (round == 1) {
            if (!(ss == sum)) return false;
        }
        else {
            Goldilocks2::Element sr;
            Goldilocks2::Element r = challenges[round - 2];
            interpolate_2(sr, r, si1[0], si1[1], si1[2]);
            if (!(sr == ss)) return false;

            // final check
            if (round == nrnd) {
                challenges.push_back(challenge());

                Goldilocks2::Element f_r = oracle[0]->open(challenges, sec_param) * oracle[1]->open(challenges, sec_param);
                Goldilocks2::Element slrl;
                Goldilocks2::Element rl = challenges[round - 1];
                interpolate_2(slrl, rl, si[0], si[1], si[2]);
                if (!(slrl == f_r)) return false;
            }
        }

        challenges.push_back(challenge());
        // goto next round
        si1 = si;
    }
    return true;
}

bool p2Verifier::execute_sumcheck(p2Prover& pr, const std::array<const oracle_ext*, 2>& oracle, const size_t& sec_param) {

    Goldilocks2::Element sum = pr.get_sum();
    size_t nrnd = pr.get_rounds();
    std::vector<Goldilocks2::Element> challenges;

    // s_{i - 1}
    std::array<Goldilocks2::Element, 3> si1;
    for (int round = 1; round <= nrnd; ++round) {
        // s_i
        std::array<Goldilocks2::Element, 3> si;
        si = pr.send_message(round, challenges);
        // s(0) + s(1)
        Goldilocks2::Element ss = si[0] + si[1];
        if (round == 1) {
            if (!(ss == sum)) return false;
        }
        else {
            Goldilocks2::Element sr;
            Goldilocks2::Element r = challenges[round - 2];
            interpolate_2(sr, r, si1[0], si1[1], si1[2]);
            if (!(sr == ss)) return false;

            // final check
            if (round == nrnd) {
                challenges.push_back(challenge());

                Goldilocks2::Element f_r = oracle[0]->open(challenges, sec_param) * oracle[1]->open(challenges, sec_param);
                Goldilocks2::Element slrl;
                Goldilocks2::Element rl = challenges[round - 1];
                interpolate_2(slrl, rl, si[0], si[1], si[2]);
                if (!(slrl == f_r)) return false;
            }
        }

        challenges.push_back(challenge());
        // goto next round
        si1 = si;
    }
    return true;
}

std::optional<challenge_claim> p2Verifier::partial_sumcheck(p2Prover& pr, const size_t& sec_param) {
    Goldilocks2::Element claim = pr.get_sum();
    return partial_sumcheck(pr, claim, sec_param);
}

std::optional<challenge_claim> p2Verifier::partial_sumcheck(p2Prover& pr, Goldilocks2::Element claim, const size_t& sec_param) {
    size_t nrnd = pr.get_rounds();
    std::vector<Goldilocks2::Element> challenges;

    // s_{i - 1}
    std::array<Goldilocks2::Element, 3> si1;
    for (int round = 1; round <= nrnd; ++round) {
        // s_i
        std::array<Goldilocks2::Element, 3> si;
        si = pr.send_message(round, challenges);
        // s(0) + s(1)
        Goldilocks2::Element ss = si[0] + si[1];
        if (round == 1) {
            if (!(ss == claim)) return std::nullopt;
        }
        else {
            Goldilocks2::Element sr;
            Goldilocks2::Element r = challenges[round - 2];
            interpolate_2(sr, r, si1[0], si1[1], si1[2]);
            if (!(sr == ss)) {
                return std::nullopt;
            }

            // final check
            if (round == nrnd) {
                challenges.push_back(challenge());

                Goldilocks2::Element slrl;
                Goldilocks2::Element rl = challenges[round - 1];
                interpolate_2(slrl, rl, si[0], si[1], si[2]);
                return challenge_claim{challenges, slrl};
            }
        }

        challenges.push_back(challenge());
        // goto next round
        si1 = si;
    }
    Goldilocks2::Element slrl;
    Goldilocks2::Element rl = challenges[nrnd - 1];
    interpolate_2(slrl, rl, si1[0], si1[1], si1[2]);
    return challenge_claim{challenges, slrl};
}

// we use goldilocks 2-extension, so no bother specifying the field
Goldilocks2::Element p2Verifier::challenge() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());

    constexpr uint64_t MODULUS = Goldilocks2::p;
    // std::uniform_int_distribution<uint64_t> dist(0, MODULUS - 1);

    //uint64_t randn[] = { dist(gen), dist(gen) };
    return random_ext();
}