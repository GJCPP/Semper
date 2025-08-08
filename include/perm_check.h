#pragma once

#include "mle.h"
#include "ligero.h"
#include "util.h"
#include "product3_sumcheck.h"

inline bool zero_check(const oracle* pcs, uint64_t sec_param) {
    auto cha = random_vec_ext(pcs->get_num_vars());
    return pcs->open(cha, sec_param) == Goldilocks2::zero();
}

// prove \prod f1/f2 = s.
class pdProver {
public:
    pdProver(const MultilinearPolynomial& num, const MultilinearPolynomial& den)
        : f1(num.clone()), f2(den.clone()), num_vars(num.get_num_vars()) {
        if (f1->get_num_vars() != f2->get_num_vars()) {
            throw std::invalid_argument("prodProver: f1 and f2 must have the same number of variables");
        }
    }

    inline int get_num_vars() const { return num_vars; }

    ligeropcs_ext commit_v(uint64_t rho_inv);

    // prove v(1, x) = v(x, 0) v(x, 1)
    p3Prover prove_trans(const std::vector<Goldilocks2::Element>& cha);
    
    // prove f1(x) = v(0, x) f2(x)
    p3Prover prove_init(const std::vector<Goldilocks2::Element>& cha);
protected:
    std::unique_ptr<MultilinearPolynomial> f1, f2;
    int num_vars;
    MultilinearPolynomial v;
};

class pdVerifier {
public:
    static bool execute_check(pdProver& prover, const oracle *pcs_f1, const oracle *pcs_f2, Goldilocks2::Element claim, uint64_t rho_inv, uint64_t sec_param);
};

class setProver {
public:
    setProver(const std::vector<const MLE*>& f1, const std::vector<const MLE*>& f2);

    

    inline int get_n() const { return n; }
    inline int get_num_vars() const { return num_vars; }
    
    std::array<ligeropcs_ext, 2> combine(const std::vector<Goldilocks2::Element>& cha, uint64_t rho_inv);

    pdProver get_pd_prover();

protected:
    std::vector<std::unique_ptr<MLE>> f1, f2;
    MLE set1, set2;
    int n, num_vars;
};

class setVerifier {
public:
    static bool execute_check(setProver& prover, const std::vector<const oracle *>& pcs_f1, const std::vector<const oracle *>& pcs_f2, uint64_t rho_inv, uint64_t sec_param);
};

class permProverBase {
public:
    
    // prove f2(pi(x)) = f1(x)
    permProverBase(const MLE& _f1, const MLE& _f2, const std::vector<size_t>& _perm);

    const std::vector<size_t>& get_perm() const { return perm; }
    size_t get_size() const { return sz; }

    // return null pcs if padding is not needed
    ligeropcs_base commit_pad_f1(uint64_t rho_inv);

    setProver get_set_prover(const MLE& id_perm, const MLE& perm);

protected:
    MLE f1, f2;
    std::vector<size_t> perm;
    size_t sz;

    MLE pad_f1;
};


class permVerifierBase {
public:
    static bool execute_check(permProverBase& prover, const oracle *pcs_f1, const oracle *pcs_f2, uint64_t rho_inv, uint64_t sec_param);
};
