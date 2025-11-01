#include "mle.h"
#include "mle_sumcheck.h"
#include "goldilocks_quadratic_ext.h"
#include "util.h"
// #include <gmpxx.h>
#include <random>

sProver::sProver(const MultilinearPolynomial& g) : g(g), nrnd(g.get_num_vars()), sum(Goldilocks2::zero()) {
    initialize();
}

sProver::sProver(MultilinearPolynomial&& g) : g(std::move(g)), nrnd(g.get_num_vars()), sum(Goldilocks2::zero()) {
    initialize();
}

/*
1. intialize the bookkeeping table A of g in time O(l * 2 ^ l) via zeta transform
2. calculate sum
*/
void sProver::initialize() {
    sum = g.get_sum();
}

std::array<Goldilocks2::Element, 2> sProver::send_message(const size_t& round, const std::vector<Goldilocks2::Element>& rands) {
    std::array<Goldilocks2::Element, 2> s = { 0, 0 };

    // notation referce: Libra(https://eprint.iacr.org/2019/317.pdf) Algorithm 1

    uint64_t offset = 1ull << (nrnd - round);


    if (round > 1) {
        // namely r_{i-1}
        g.fix(0, rands.back());
    }

    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads(); // or omp_get_num_procs()
#endif

    std::vector<Goldilocks2::Element> s_local(2 * num_threads);

    #pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        Goldilocks2::Element* loc = &s_local[2 * tid];

        // Manual blocking for better cache locality
        const uint64_t block_size = 4096; // tune this (e.g., 1024–16384)
        #pragma omp for schedule(static)
        for (uint64_t base = 0; base < offset; base += block_size) {
            uint64_t end = std::min(base + block_size, offset);

            // Local accumulators stay in registers (faster than s_local writes)
            Goldilocks2::Element acc0 = Goldilocks2::zero();
            Goldilocks2::Element acc1 = Goldilocks2::zero();

            for (uint64_t b = base; b < end; ++b) {
                acc0 += g.eval_hypercube(b);
                acc1 += g.eval_hypercube(b + offset);
            }

            loc[0] += acc0;
            loc[1] += acc1;
        }
    }

    // final reduction
    for (int t = 0; t < num_threads; ++t) {
        s[0] += s_local[2 * t + 0];
        s[1] += s_local[2 * t + 1];
    }

    return s;
}

// sVerifier::sVerifier(){}


bool sVerifier::execute_sumcheck(sProver& pr, const oracle& oracle, const size_t& sec_param) {
    // if(!ligeroVerifier::check_commit(oracle, sec_param)) return false;
    Goldilocks2::Element sum = pr.get_sum();
    size_t nrnd = pr.get_rounds();
    std::vector<Goldilocks2::Element> challenges;

    // s_{i - 1}
    std::array<Goldilocks2::Element, 2> si1;
    for (size_t round = 1;round <= nrnd; ++round) {
        // s_i
        std::array<Goldilocks2::Element, 2> si;
        si = pr.send_message(round, challenges);
        // s(0) + s(1)
        Goldilocks2::Element ss;
        Goldilocks2::add(ss, si[0], si[1]);
        if (round == 1) {
            if (!(ss == sum)) return false;
        }
        else {
            // s_{i - 1}(r) = r * s_{i - 1}(1) + (1-r) * s_{i - 1}(0)
            Goldilocks2::Element sr;
            Goldilocks2::Element r = challenges[round - 2];
            Goldilocks2::Element A, B, oneminusr;
            Goldilocks2::sub(oneminusr, Goldilocks::one(), r);
            Goldilocks2::mul(A, si1[0], oneminusr);
            Goldilocks2::mul(B, si1[1], r);
            Goldilocks2::add(sr, A, B);
            if (!(sr == ss)) return false;

            // final check
            if (round == nrnd) {
                challenges.push_back(challenge());
                // should be implemented later
                // fr: f(r1, r2, ..., rl)
                Goldilocks2::Element f_r = oracle.open(challenges, sec_param);

                // std::cout << Goldilocks2::toString(f_r) << '\n';
                // s_l(r_l)
                Goldilocks2::Element slrl;
                Goldilocks2::Element rl = challenges[round - 1];
                Goldilocks2::Element C, D, oneminusrl;
                Goldilocks2::sub(oneminusrl, Goldilocks::one(), rl);
                Goldilocks2::mul(C, si[0], oneminusrl);
                Goldilocks2::mul(D, si[1], rl);
                Goldilocks2::add(slrl, C, D);
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

bool sVerifier::partial_sumcheck(sProver& pr, std::vector<Goldilocks2::Element>& challenges, Goldilocks2::Element& claim, const size_t& sec_param) {
    // if(!ligeroVerifier::check_commit(oracle, sec_param)) return false;
    size_t nrnd = pr.get_rounds();
    set_timer(VERIFIER_TIMER);
    set_timer("verifier partial_sumcheck");

    // s_{i - 1}
    std::array<Goldilocks2::Element, 2> si1;
    for (size_t round = 1; round <= nrnd; ++round) {
        // s_i
        std::array<Goldilocks2::Element, 2> si;
        pause_timer(VERIFIER_TIMER);
        pause_timer("verifier partial_sumcheck");
        si = pr.send_message(round, challenges);
        set_timer(VERIFIER_TIMER);
        set_timer("verifier partial_sumcheck");
        // s(0) + s(1)
        Goldilocks2::Element ss;
        Goldilocks2::add(ss, si[0], si[1]);
        if (round == 1) {
            if (!(ss == claim)) return false;
        }
        else {
            // s_{i - 1}(r) = r * s_{i - 1}(1) + (1-r) * s_{i - 1}(0)
            Goldilocks2::Element sr;
            Goldilocks2::Element r = challenges.back();
            Goldilocks2::Element A, B, oneminusr;
            Goldilocks2::sub(oneminusr, Goldilocks::one(), r);
            Goldilocks2::mul(A, si1[0], oneminusr);
            Goldilocks2::mul(B, si1[1], r);
            Goldilocks2::add(sr, A, B);
            if (!(sr == ss)) return false;

            // final check
            if (round == nrnd) {
                challenges.push_back(challenge());

                // std::cout << Goldilocks2::toString(f_r) << '\n';
                // s_l(r_l)
                Goldilocks2::Element newclaim;
                Goldilocks2::Element rl = challenges.back();
                Goldilocks2::Element C, D, oneminusrl;
                Goldilocks2::sub(oneminusrl, Goldilocks::one(), rl);
                Goldilocks2::mul(C, si[0], oneminusrl);
                Goldilocks2::mul(D, si[1], rl);
                Goldilocks2::add(newclaim, C, D);
                claim = newclaim;
                pause_timer(VERIFIER_TIMER);
                pause_timer("verifier partial_sumcheck");
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
    pause_timer(VERIFIER_TIMER);
    pause_timer("verifier partial_sumcheck");
    return true;
}


// we use goldilocks 2-extension, so no bother specifying the field
Goldilocks2::Element sVerifier::challenge() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    // constexpr uint64_t MODULUS = Goldilocks2::p;
    // std::uniform_int_distribution<uint64_t> dist(0, MODULUS - 1);

    // uint64_t randn[] = { dist(gen), dist(gen) };
    return random_ext();
}