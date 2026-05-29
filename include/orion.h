#pragma once

#include <cstdint>
#include <array>
#include <map>
#include <memory>
#include <vector>

#include "goldilocks_quadratic_ext.h"
#include "merkle.h"
#include "mle.h"
#include "oracle.h"

class orionProver_base;
class orionProver_ext;

class orionpcs_base : public oracle {
public:
    orionpcs_base() = default;
    orionpcs_base(const MerkleDef::Digest& mthash,
                  const std::shared_ptr<orionProver_base>& prover,
                  const size_t& num_rows,
                  const size_t& num_cols);

    bool empty() const;

    Goldilocks2::Element open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const override;
    int get_num_vars() const override;

    MerkleDef::Digest mthash;
    std::shared_ptr<orionProver_base> prover;
    size_t num_rows = 0;
    size_t num_cols = 0; // Original, unencoded column count.
};

class orionpcs_ext : public oracle {
public:
    orionpcs_ext() = default;
    orionpcs_ext(const MerkleDef::Digest& mthash,
                 const std::shared_ptr<orionProver_ext>& prover,
                 const size_t& num_rows,
                 const size_t& num_cols);

    bool empty() const;

    Goldilocks2::Element open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const override;
    int get_num_vars() const override;

    MerkleDef::Digest mthash;
    std::shared_ptr<orionProver_ext> prover;
    size_t num_rows = 0;
    size_t num_cols = 0; // Original, unencoded column count.
};

class orionProver_base {
public:
    orionProver_base(const MultilinearPolynomial& w, const uint64_t& rho_inv, int loga = -1);
    orionProver_base(const std::vector<uint64_t>& w, const uint64_t& rho_inv, int loga = -1);

    orionpcs_base commit() const;
    std::vector<Goldilocks2::Element> lincomb(const std::vector<Goldilocks2::Element>& r) const;
    std::vector<MerkleTree_base::MTPayload> open_cols(const std::vector<size_t>& indexes) const;

    inline int get_num_vars() const { return num_vars; }
    inline size_t get_code_len() const { return code_len; }

    uint64_t rho_inv;
    size_t a = 0;
    size_t b = 0;
    size_t code_len = 0;
    std::shared_ptr<std::vector<Goldilocks::Element>> M;
    int num_vars = 0;
    std::shared_ptr<MerkleTree_base> mt_t;
};

class orionProver_ext {
public:
    orionProver_ext(const MultilinearPolynomial& w, const uint64_t& rho_inv, int loga = -1);
    orionProver_ext(const std::vector<Goldilocks2::Element>& w, const uint64_t& rho_inv, int loga = -1);

    orionpcs_ext commit() const;
    std::vector<Goldilocks2::Element> lincomb(const std::vector<Goldilocks2::Element>& r) const;
    std::vector<MerkleTree_ext::MTPayload> open_cols(const std::vector<size_t>& indexes) const;

    inline int get_num_vars() const { return num_vars; }
    inline size_t get_code_len() const { return code_len; }

    uint64_t rho_inv;
    size_t a = 0;
    size_t b = 0;
    size_t code_len = 0;
    std::shared_ptr<std::vector<Goldilocks2::Element>> M;
    int num_vars = 0;
    MerkleTree_ext mt_t;
};

class orionVerifier {
public:
    static Goldilocks2::Element open(const orionpcs_base& pcs, const std::vector<Goldilocks2::Element>& z, const size_t& sec_param);
    static Goldilocks2::Element open(const orionpcs_ext& pcs, const std::vector<Goldilocks2::Element>& z, const size_t& sec_param);

private:
    static std::vector<size_t> randindexes(const uint64_t& n, const size_t& bound);
    static std::array<std::vector<Goldilocks2::Element>, 2> calculate_lr(const size_t& num_var,
                                                                         const std::vector<Goldilocks2::Element>& z,
                                                                         size_t num_rows);
    static bool check_lincomb(const orionpcs_base& pcs,
                              const std::vector<Goldilocks2::Element>& r,
                              const std::vector<size_t>& indexes,
                              const std::vector<Goldilocks2::Element>& weights,
                              Goldilocks2::Element& claim);
    static bool check_lincomb(const orionpcs_ext& pcs,
                              const std::vector<Goldilocks2::Element>& r,
                              const std::vector<size_t>& indexes,
                              const std::vector<Goldilocks2::Element>& weights,
                              Goldilocks2::Element& claim);
};

std::vector<Goldilocks::Element> orion_encode_base(const std::vector<Goldilocks::Element>& data);
std::vector<Goldilocks2::Element> orion_encode_ext(const std::vector<Goldilocks2::Element>& data);

int orion_default_loga(int num_vars);

orionpcs_base orion_commit_base(const MultilinearPolynomial& w, const uint64_t& rho_inv, int loga = -1);
orionpcs_base orion_commit_base(const std::vector<uint64_t>& w, const uint64_t& rho_inv, int loga = -1);
orionpcs_ext orion_commit_ext(const MultilinearPolynomial& w, const uint64_t& rho_inv, int loga = -1);

void print_all_orion_open_cnt();

extern std::map<int, int> orion_open_cnt;
