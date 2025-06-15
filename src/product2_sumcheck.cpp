#include "product2_sumcheck.h"
#include "goldilocks_quadratic_ext.h"
#include "mle.h"
#include <array>
#include <vector>
#include <random>
#include <cassert>

p2Prover::p2Prover(const MultilinearPolynomial& p1, const MultilinearPolynomial& p2) 
    : p1(p1), p2(p2), nrnd(p1.get_num_vars()), sum(Goldilocks2::zero()) {
    assert(p1.get_num_vars() == p2.get_num_vars());
    initialize();
}

/*
1. intialize the bookkeeping table A of g in time O(l * 2 ^ l) via zeta transform
2. calculate sum
*/
void p2Prover::initialize() {
    uint64_t tsize = 1ull << p1.get_num_vars();

    keepTablep1 = p1.get_eval_table();
    keepTablep2 = p2.get_eval_table();
    for (uint64_t mask = 0; mask < tsize; ++mask) {
        sum = sum + keepTablep1[mask] * keepTablep2[mask];
    }
    // std::cout << Goldilocks2::toString(sum) << '\n';
}

inline void p2Prover::shrinkTable(const Goldilocks2::Element& r, const uint64_t& offset) {
    for (uint64_t b = 0; b < offset; ++b) {
        Goldilocks2::Element oneminusr = Goldilocks2::one() - r;
        keepTablep1[b] = keepTablep1[b] * oneminusr + keepTablep1[b + offset] * r;
        keepTablep2[b] = keepTablep2[b] * oneminusr + keepTablep2[b + offset] * r;
    }
    keepTablep1.resize(offset);
    keepTablep2.resize(offset);
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
        uint64_t offset_last = (offset << 1);
        Goldilocks2::Element r = rands[round - 2];
        shrinkTable(r, offset_last);
    }

    for (uint64_t b = 0; b < offset; ++b) {
        s[0] = s[0] + keepTablep1[b] * keepTablep2[b]; // f(0) = g_1(0) * g_2(0)
        s[1] = s[1] + keepTablep1[b + offset] * keepTablep2[b + offset]; // f(1) = g_1(1) * g_2(1)
        s[2] = s[2] + lincomb(keepTablep1[b + offset], keepTablep1[b], 2) * lincomb(keepTablep2[b + offset], keepTablep2[b], 2); // f(2) = (2 g_1(1) - g_1(0)) * (2 g_2(1) - g_2(0))
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

bool p2Verifier::execute_sumcheck(p2Prover& pr, const std::array<std::shared_ptr<oracle_base>, 2>& oracle, const size_t& sec_param) {

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

bool p2Verifier::execute_sumcheck(p2Prover& pr, const std::array<std::shared_ptr<oracle_ext>, 2>& oracle, const size_t& sec_param) {

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

// we use goldilocks 2-extension, so no bother specifying the field
Goldilocks2::Element p2Verifier::challenge() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());

    constexpr uint64_t MODULUS = Goldilocks2::p;
    std::uniform_int_distribution<uint64_t> dist(0, MODULUS - 1);

    uint64_t randn[] = { dist(gen), dist(gen) };
    return Goldilocks2::fromU64(randn);
}