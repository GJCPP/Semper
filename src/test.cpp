#include "test.h"

bool test_arithmetic() {
    typedef Goldilocks2::Element Element;
    for (int cnt(0); cnt != 100; ++cnt) {
        Element a = random_ext(), b = random_ext();
        Element res = a + b;
        Element res2;
        Goldilocks2::add(res2, a, b);
        if (res != res2) {
            return false;
        }
        res = a * b;
        Goldilocks2::mul(res2, a, b);
        if (res != res2) {
            return false;
        }
    }
    return true;
}

bool test_eval_power_mle() {
    typedef Goldilocks2::Element Element;

    for (int cnt(0); cnt != 100; ++cnt) {
        int l = (rand() % 10 + 1);
        int u = (rand() % (1 << l));
        std::vector<Element> r = random_vec_ext(l);
        Element beta = random_ext(), beta_pow = Goldilocks2::one();

        // beta^0, beta^1, ..., beta^(u), 0, ..., 0
        std::vector<Element> beta_pow_n;
        for (int i = 0; i <= u; i++) {
            beta_pow_n.push_back(beta_pow);
            beta_pow = beta_pow * beta;
        }
        for (int i = u + 1; i < (1 << l); ++i) {
            beta_pow_n.push_back(Goldilocks2::zero());
        }

        Element res = eval_power_mle(beta, r, u, l);
        MultilinearPolynomial mle(beta_pow_n);
        if (mle.evaluate(r) != res) {
            return false;
        }
    }
    return true;
}

bool test_product3_sumcheck() {
    typedef Goldilocks2::Element Element;
    int inv_rho = 2, sec_param = 32;
    for (int cnt(0); cnt != 100; ++cnt) {
        int l = (rand() % 10 + 1);
        // Generate random polynomials and their commitments
        std::vector<Element> poly1_vec = random_vec_ext(1 << l);
        std::vector<Element> poly2_vec = random_vec_ext(1 << l);
        std::vector<Element> poly3_vec = random_vec_ext(1 << l);
        MultilinearPolynomial poly1(poly1_vec);
        MultilinearPolynomial poly2(poly2_vec);
        MultilinearPolynomial poly3(poly3_vec);
        ligeroProver_ext pcs1(poly1, inv_rho);
        ligeroProver_ext pcs2(poly2, inv_rho);
        ligeroProver_ext pcs3(poly3, inv_rho);

        // Run product3 sumcheck
        p3Prover prover(poly1, poly2, poly3);
        p3Verifier verifier;
        std::array<ligeropcs_ext, 3> oracle = { pcs1.commit(), pcs2.commit(), pcs3.commit() };
        if (!verifier.execute_sumcheck(prover, oracle, sec_param)) {
            return false;
        }
    }
    return true;
}

bool test_product2_sumcheck() {
    typedef Goldilocks2::Element Element;
    int inv_rho = 2, sec_param = 32;
    for (int cnt(0); cnt != 100; ++cnt) {
        int l = (rand() % 10 + 1);
        // Generate random polynomials and their commitments
        std::vector<Element> poly1_vec = random_vec_ext(1 << l);
        std::vector<Element> poly2_vec = random_vec_ext(1 << l);
        MultilinearPolynomial poly1(poly1_vec);
        MultilinearPolynomial poly2(poly2_vec);
        ligeroProver_ext pcs1(poly1, inv_rho);
        ligeroProver_ext pcs2(poly2, inv_rho);

        // Run product3 sumcheck
        p2Prover prover(poly1, poly2);
        p2Verifier verifier;
        std::array<std::shared_ptr<oracle_ext>, 2> oracle = { std::make_shared<ligeropcs_ext>(pcs1.commit()), std::make_shared<ligeropcs_ext>(pcs2.commit()) };
        if (!verifier.execute_sumcheck(prover, oracle, sec_param)) {
            return false;
        }
    }
    for (int cnt(0); cnt != 100; ++cnt) {
        int l = (rand() % 10 + 1);
        int u = (rand() % (1 << l));
        // Generate random polynomials and their commitments
        std::vector<Element> poly1_vec = random_vec_ext(1 << l);
        std::vector<Element> poly2_vec = random_vec_ext(1 << l);
        MultilinearPolynomial poly1(poly1_vec);
        MLE_Pow poly2(random_ext(), l, u);
        ligeroProver_ext pcs1(poly1, inv_rho);

        // Run product3 sumcheck
        p2Prover prover(poly1, poly2);
        p2Verifier verifier;
        std::array<std::shared_ptr<oracle_ext>, 2> oracle = { std::make_shared<ligeropcs_ext>(pcs1.commit()), std::make_shared<MLE_Pow>(poly2) };
        if (!verifier.execute_sumcheck(prover, oracle, sec_param)) {
            return false;
        }
    }
    return true;
}

bool test_logup() {
    size_t fsize = 1ull << 16;
    std::vector<uint64_t> t1 = trange(0, (1ull << 16) - 1);
    std::vector<uint64_t> t2(t1.size());
    for (size_t i = 0;i < t1.size(); ++i) {
        t2[i] = t1[i] << 1;
    }

    std::vector<uint64_t> f1(fsize);
    std::vector<uint64_t> f2(f1.size());

    srand(42);
    for (size_t i = 0;i < f1.size(); ++i) {
        size_t r = rand() % t1.size();
        f1[i] = t1[r];
        f2[i] = t2[r];
    }

    LogupProver lpr(f1, f2, t1, t2);
    return LogupVerifier::execute_logup(lpr, 2, 32);
}

bool run_test() {
    srand(time(NULL));
    if (!test_arithmetic()) {
        std::cout << "test_arithmetic failed" << std::endl;
        return false;
    }
    if (!test_eval_power_mle()) {
        std::cout << "test_eval_power_mle failed" << std::endl;
        return false;
    }
    if (!test_product3_sumcheck()) {
        std::cout << "test_product3_sumcheck failed" << std::endl;
        return false;
    }
    if (!test_product2_sumcheck()) {
        std::cout << "test_product2_sumcheck failed" << std::endl;
        return false;
    }
    if (!test_logup()) {
        std::cout << "test_logup failed" << std::endl;
        return false;
    }
    std::cout << "All tests passed" << std::endl;
    return true;
}
