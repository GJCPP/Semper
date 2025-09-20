#include "mle.h"
#include "mle_sumcheck.h"
#include "goldilocks_quadratic_ext.h"
#include "util.h"
// #include <gmpxx.h>
#include <random>

template <typename field>
sProver<field>::sProver(const MLE<field>& g) : g(g), nrnd(g.get_num_vars()), sum(field::zero()) {
    initialize();
}

template <typename field>
sProver<field>::sProver(MLE<field>&& g) : g(std::move(g)), nrnd(g.get_num_vars()), sum(field::zero()) {
    initialize();
}

/*
1. intialize the bookkeeping table A of g in time O(l * 2 ^ l) via zeta transform
2. calculate sum
*/
template <typename field>
void sProver<field>::initialize() {
    sum = g.get_sum();
}

template <typename field>
std::array<field, 2> sProver<field>::send_message(const size_t& round, const std::vector<field>& rands) {
    std::array<field, 2> s = { 0, 0 };

    // notation referce: Libra(https://eprint.iacr.org/2019/317.pdf) Algorithm 1
    uint64_t offset = 1ull << (nrnd - round);

    if (round > 1) {
        // namely r_{i-1}
        g.fix(0, rands.back());
    }
    for (uint64_t b = 0; b < offset; ++b) {
        s[0] += g[b];
        s[1] += g[b + offset];
    }
    return s;
}

// sVerifier::sVerifier(){}

template <typename field>
bool sVerifier<field>::execute_sumcheck(sProver<field>& pr, const oracle<field>& oracle, const size_t& sec_param) {
    // if(!ligeroVerifier::check_commit(oracle, sec_param)) return false;
    field sum = pr.get_sum();
    size_t nrnd = pr.get_rounds();
    std::vector<field> challenges;

    // s_{i - 1}
    std::array<field, 2> si1;
    for (size_t round = 1;round <= nrnd; ++round) {
        // s_i
        std::array<field, 2> si;
        si = pr.send_message(round, challenges);
        // s(0) + s(1)
        field ss = si[0] + si[1];
        if (round == 1) {
            if (!(ss == sum)) return false;
        }
        else {
            // s_{i - 1}(r) = r * s_{i - 1}(1) + (1-r) * s_{i - 1}(0)
            // Goldilocks2::Element sr;
            // Goldilocks2::Element r = challenges[round - 2];
            // Goldilocks2::Element A, B, oneminusr;
            // Goldilocks2::sub(oneminusr, Goldilocks::one(), r);
            // Goldilocks2::mul(A, si1[0], oneminusr);
            // Goldilocks2::mul(B, si1[1], r);
            // Goldilocks2::add(sr, A, B);
            field r = challenges[round - 2];
            field sr = si1[0] * (field::one() - r) + si1[1] * r;
            if (!(sr == ss)) return false;

            // final check
            if (round == nrnd) {
                challenges.push_back(challenge());
                // should be implemented later
                // fr: f(r1, r2, ..., rl)
                Goldilocks2::Element f_r = oracle.open(challenges, sec_param);

                // std::cout << Goldilocks2::toString(f_r) << '\n';
                // s_l(r_l)
                // Goldilocks2::Element slrl;
                // Goldilocks2::Element rl = challenges[round - 1];
                // Goldilocks2::Element C, D, oneminusrl;
                // Goldilocks2::sub(oneminusrl, Goldilocks::one(), rl);
                // Goldilocks2::mul(C, si[0], oneminusrl);
                // Goldilocks2::mul(D, si[1], rl);
                // Goldilocks2::add(slrl, C, D);
                // if (!(slrl == f_r)) return false;
                field rl = challenges[round - 1];
                field slrl = si[0] * (field::one() - rl) + si[1] * rl;
                if (!(slrl == f_r)) return false;
                // std::cout << Goldilocks2::toString(slrl) << '\n';
            }
        }

        challenges.push_back(challenge());
        // goto next round
        si1 = si;
    }
    return true;
}

template <typename field>
bool sVerifier<field>::partial_sumcheck(sProver<field>& pr, std::vector<field>& challenges, field& claim, const size_t& sec_param) {
    // if(!ligeroVerifier::check_commit(oracle, sec_param)) return false;
    size_t nrnd = pr.get_rounds();

    // s_{i - 1}
    std::array<field, 2> si1;
    for (size_t round = 1; round <= nrnd; ++round) {
        // s_i
        std::array<field, 2> si;
        si = pr.send_message(round, challenges);
        // s(0) + s(1)
        field ss = si[0] + si[1];
        if (round == 1) {
            if (!(ss == claim)) return false;
        }
        else {
            // s_{i - 1}(r) = r * s_{i - 1}(1) + (1-r) * s_{i - 1}(0)
            // Goldilocks2::Element sr;
            // Goldilocks2::Element r = challenges.back();
            // Goldilocks2::Element A, B, oneminusr;
            // Goldilocks2::sub(oneminusr, Goldilocks::one(), r);
            // Goldilocks2::mul(A, si1[0], oneminusr);
            // Goldilocks2::mul(B, si1[1], r);
            // Goldilocks2::add(sr, A, B);
            field r = challenges.back();
            field sr = si1[0] * (field::one() - r) + si1[1] * r;
            if (!(sr == ss)) return false;

            // final check
            if (round == nrnd) {
                challenges.push_back(challenge());

                // std::cout << Goldilocks2::toString(f_r) << '\n';
                // s_l(r_l)
                // Goldilocks2::Element newclaim;
                // Goldilocks2::Element rl = challenges.back();
                // Goldilocks2::Element C, D, oneminusrl;
                // Goldilocks2::sub(oneminusrl, Goldilocks::one(), rl);
                // Goldilocks2::mul(C, si[0], oneminusrl);
                // Goldilocks2::mul(D, si[1], rl);
                // Goldilocks2::add(newclaim, C, D);
                field rl = challenges.back();
                field newclaim = si[0] * (field::one() - rl) + si[1] * rl;
                claim = newclaim;
                return true;
            }
        }

        challenges.push_back(challenge());
        // goto next round
        si1 = si;
    }
    // si1[0] = b
    // si1[1] = a + b
    // claim = (1 - r) * b + r * (a + b)
    claim = (Goldilocks2::one() - challenges.back()) * si1[0] + challenges.back() * si1[1];
    return true;
}


// we use goldilocks 2-extension, so no bother specifying the field
template <typename field>
field sVerifier<field>::challenge() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    // constexpr uint64_t MODULUS = Goldilocks2::p;
    // std::uniform_int_distribution<uint64_t> dist(0, MODULUS - 1);

    // uint64_t randn[] = { dist(gen), dist(gen) };
    return random_ext();
}