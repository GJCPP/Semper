#pragma once

#include "mle.h"
#include "ligero.h"
#include "util.h"
#include "product3_sumcheck.h"

inline bool zero_check(const oracle* pcs, uint64_t sec_param) {
    auto cha = random_vec_ext(pcs->get_num_vars());
    return pcs->open(cha, sec_param) == Goldilocks2::zero();
}

// prove \prod f1 = s.
class pdProver {
public:
    pdProver(const MultilinearPolynomial& mle)
        : f(&mle), num_vars(mle.get_num_vars()), round(0) {
        g.resize(num_vars + 1);
        g[num_vars] = f->get_eval_table();
        prod = Goldilocks2::one();
        for (auto& i : g[num_vars]) {
            prod *= i;
        }
        for (int i = num_vars - 1; i >= 0; --i) {
            size_t offset = (1 << i);
            g[i].resize(offset);
            for (size_t j = 0; j != offset; ++j) {
                g[i][j] = g[i + 1][j << 1] * g[i + 1][(j << 1) | 1];
            }
        }
    }

    inline int get_num_vars() const { return num_vars; }

    inline Goldilocks2::Element get_prod() const { return prod; }

    inline std::array<Goldilocks2::Element, 2> first_msg() {
        ++round;
        return {g[1][0], g[1][1]};
    }

    inline p3Prover get_next_sumcheck(const MLE_Eq& eq) {
        if (round >= num_vars) {
            throw std::runtime_error("pdProver::get_next_sumcheck: all rounds have been completed");
        }
        std::vector<Goldilocks2::Element> g0, g1;
        size_t offset = (1ull << round);
        for (size_t i = 0; i != offset; ++i) {
            g0.push_back(g[round + 1][i << 1]);
            g1.push_back(g[round + 1][(i << 1) | 1]);
        }
        mle_g0 = g0;
        mle_g1 = g1;
        return p3Prover(eq, mle_g0, mle_g1);
    }

    inline std::array<Goldilocks2::Element, 2> next_msg(const std::vector<Goldilocks2::Element>& cha) {
        ++round;
        return {mle_g0.evaluate(cha), mle_g1.evaluate(cha)};
    }
protected:
    const MLE *f;
    std::vector<std::vector<Goldilocks2::Element>> g;
    MLE mle_g0, mle_g1;
    Goldilocks2::Element prod;
    int num_vars;
    int round;
};

class pdVerifier {
public:
    static bool execute_check(pdProver& prover, const oracle *pcs_f1, Goldilocks2::Element claim, uint64_t sec_param);
};

class setProver {
public:
    setProver(const std::vector<const MLE*>& f1, const std::vector<const MLE*>& f2);

    

    inline int get_n() const { return n; }
    inline int get_num_vars() const { return num_vars; }
    
    std::array<ligeropcs_ext, 2> combine(const std::vector<Goldilocks2::Element>& cha, uint64_t rho_inv);

    std::array<pdProver, 2> get_pd_prover();

protected:
    std::vector<const MLE *> f1, f2;
    MLE set1, set2;
    int n, num_vars;
};

class setVerifier {
public:
    static bool execute_check(setProver& prover, const std::vector<const oracle *>& pcs_f1, const std::vector<const oracle *>& pcs_f2, uint64_t rho_inv, uint64_t sec_param);
};

class permProver {
public:
    
    // prove f2(pi(x)) = f1(x)
    permProver(const std::vector<size_t>& _perm, bool ext);

    const std::vector<size_t>& get_perm() const { return perm; }
    size_t get_size() const { return sz; }

    void add_mle(const MLE *f1, const MLE *f2);

    // return null pcs if padding is not needed
    std::vector<std::unique_ptr<oracle>> commit_pad_f1(uint64_t rho_inv);

    setProver get_set_prover(const MLE& id_perm, const MLE& perm);

protected:
    bool ext;
    std::vector<const MLE *> f1, f2;
    std::vector<size_t> perm;
    size_t sz;

    std::vector<MLE> pad_f1;
};


class permVerifier {
public:
    void add_pcs(const oracle *pcs_f1, const oracle *pcs_f2);

    bool execute_check(permProver& prover, uint64_t rho_inv, uint64_t sec_param);

protected:
    std::vector<const oracle*> pcs_f1, pcs_f2;
};

class mapProver {
public:
    mapProver(const std::vector<size_t>& from, const std::vector<size_t>& to, bool ext);

    void add_mle(const MLE *mle_from, const MLE *mle_to);
    
    int get_right_num_vars() const { return right_num_vars; }
    int get_pad_num_vars() const { return pad_num_vars; }
    
    std::vector<std::unique_ptr<oracle>> commit_right(uint64_t rho_inv) const;
    
    permProver get_perm_prover();
    
protected:
    bool ext;
    int left_num_vars, right_num_vars, pad_num_vars;
    size_t left_size, right_size;
    std::vector<const MLE*> left;
    std::vector<MLE> right;
    std::vector<size_t> map_from, map_to, mapto;

    void init_map(const MLE *mle_from, const MLE *mle_to);
};

class mapVerifier {
public:
    void add_pcs(const oracle *left, const oracle *right);

    bool execute_check(mapProver& prover, uint64_t rho_inv, uint64_t sec_param);
protected:
    std::vector<const oracle*> pcs_left, pcs_right;
};
