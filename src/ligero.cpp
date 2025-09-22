
#include <cmath>
#include <cassert>
#include <openssl/sha.h>

#include "ligero.h"
// #include "mle.h"
#include "goldilocks_quadratic_ext.h"
#include "merkle.h"
#include "util.h"
#include "timer.h"
#include "counter.h"

// opt_loga for sec_param=32
int opt_loga[] = {-1, 1, 1, 2, 3, 4, 5, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

std::map<int, int> ligero_open_cnt;

// reed solomon encode data on base field
std::vector<Goldilocks::Element> rsencode(const std::vector<Goldilocks::Element>& data, const uint64_t& rho_inv) {
    // return eval_with_ntt_ext(data, data.size() * rho_inv);
    return eval_with_ntt(data, data.size() * rho_inv);
}

ligeropcs_base ligero_commit_base(const MultilinearPolynomial& w, const uint64_t& rho_inv, int loga) {
    auto prover = std::make_shared<ligeroProver_base>(w, rho_inv, loga);
    auto ret = prover->commit();
    return ret;
}

ligeropcs_ext ligero_commit_ext(const MultilinearPolynomial& w, const uint64_t& rho_inv) {
    auto prover = std::make_shared<ligeroProver_ext>(w, rho_inv);
    auto ret = prover->commit();
    return ret;
}

// reed solomon encode data on quadratic extension field
// this is simply encode real part and imaginary part respectively and merge the results
std::vector<Goldilocks2::Element> rsencode(const std::vector<Goldilocks2::Element>& data, const uint64_t& rho_inv) {
    return eval_with_ntt(data, data.size() * rho_inv);
}

void get_ab(int num_vars, int loga, size_t& a, size_t& b) {
    if (loga == -1 || loga > num_vars) {
        if (num_vars > 33) {
            std::cerr << "⚠️ Warning: num_vars > 33." << std::endl;
            throw;
        }
        a = (1ull << opt_loga[num_vars]);
        b = (1ull << (num_vars - opt_loga[num_vars]));
    } else {
        a = (1ull << loga);
        b = (1ull << (num_vars - loga));
    }
}

ligeroProver_base::ligeroProver_base(const MultilinearPolynomial& w, const uint64_t& rho_inv, int loga) 
    :rho_inv(rho_inv) {
    const auto& evals = w.get_eval_table();
    num_vars = find_ceiling_log2(evals.size());

    // 2^l = a * b
    get_ab(num_vars, loga, a, b);

    M.resize(1ull << num_vars, Goldilocks::zero());
    codelen = b * rho_inv;
    for (size_t i = 0; i < evals.size(); ++i) {
        M[i] = evals[i][0];
    }
    for (size_t i = 0; i < a; ++i) {
        std::vector<Goldilocks::Element> dataline(b);
        for (size_t j = 0; j < b; ++j) {
            dataline[j] = M[i * b + j];
        }
        codewords.push_back(rsencode(dataline, rho_inv));
    }
    mt_t = MerkleTree_base(codewords);
}


ligeroProver_base::ligeroProver_base(const std::vector<Goldilocks::Element>& w, const uint64_t& rho_inv, int loga) :rho_inv(rho_inv), M(w) {
    // stevals = w.get_eval_table();
    num_vars = find_ceiling_log2(w.size());

    // 2^l = a * b
    get_ab(num_vars, loga, a, b);

    M.resize(1ull << num_vars, Goldilocks::zero());
    codelen = b * rho_inv;
    // for (size_t i = 0; i < w.size(); ++i) {
    //     M[i] = w[i]; // Seems redundant
    // }
    for (size_t i = 0; i < a; ++i) {
        std::vector<Goldilocks::Element> dataline(b);
        for (size_t j = 0; j < b; ++j) {
            dataline[j] = M[i * b + j];
        }
        codewords.push_back(rsencode(dataline, rho_inv));
    }
    mt_t = MerkleTree_base(codewords);
}

ligeroProver_base::ligeroProver_base(const std::vector<uint64_t>& w, const uint64_t& rho_inv, int loga) :rho_inv(rho_inv) {
    // stevals = w.get_eval_table();
    num_vars = find_ceiling_log2(w.size());
    // std::cout << l << '\n';

    // 2^l = a * b
    get_ab(num_vars, loga, a, b);



    // set_timer("make matrix");
    M.resize(1ull << num_vars, Goldilocks::zero());
    codelen = b * rho_inv;
    for (size_t i = 0; i < w.size(); ++i) {
        M[i] = Goldilocks::fromU64(w[i]);
    }
    // set_timer("ntt");
    // pause_timer("ntt");
    // alert("log-size of vector to be ntt-ed: " + std::to_string(b));
    for (size_t i = 0; i < a; ++i) {
        std::vector<Goldilocks::Element> dataline(b);
        for (size_t j = 0; j < b; ++j) {
            dataline[j] = M[i * b + j];
        }
        // resume_timer("ntt");
        codewords.push_back(rsencode(dataline, rho_inv));
        // pause_timer("ntt");
    }
    // end_timer("ntt");
    // end_timer("make matrix");
    // set_timer("build merkle tree");
    mt_t = MerkleTree_base(codewords);
    // end_timer("build merkle tree");
}

std::vector<Goldilocks2::Element> ligeroProver_base::lincomb(const std::vector<Goldilocks2::Element>& r) const {
    assert(r.size() == a);
    // std::cout << r.size() << '\n' << a << '\n';
    std::vector<Goldilocks2::Element> v(b, Goldilocks2::zero());
    for (size_t j = 0; j < a; ++j) {
        Goldilocks2::Element tmp;
        size_t offset = j * b;
        for (size_t i = 0; i < b; ++i) {
            Goldilocks2::mul(tmp, r[j], M[offset + i]);
            Goldilocks2::add(v[i], v[i], tmp);
        }
    }
    return v;
}

std::vector<MerkleTree_base::MTPayload> ligeroProver_base::open_cols(const std::vector<size_t>& indexes) const {
    std::vector<MerkleTree_base::MTPayload> payloads;
    for (size_t e : indexes) {
        payloads.push_back(mt_t.MerkleOpen(e));
    }
    return payloads;
}

ligeropcs_base ligeroProver_base::commit() const {
    auto merk = mt_t.MerkleCommit();
    add_proof_size(sizeof(merk));
    return { merk, std::make_shared<ligeroProver_base>(*this), a, b };
}



ligeroProver_ext::ligeroProver_ext(const MultilinearPolynomial& w, const uint64_t& rho_inv)
    : ligeroProver_ext(w.get_eval_table(), rho_inv) {
}

ligeroProver_ext::ligeroProver_ext(const std::vector<Goldilocks2::Element>& w, const uint64_t& rho_inv) :rho_inv(rho_inv), M(w)  {
    num_vars = find_ceiling_log2(w.size());
    M.resize(1ull << num_vars, Goldilocks2::zero());
    // 2^l = a * b
    get_ab(num_vars, -1, a, b);

    codelen = b * rho_inv;
    for (size_t i = 0; i < w.size(); ++i) {
        M[i] = w[i];
    }
    for (size_t i = 0; i < a; ++i) {
        std::vector<Goldilocks2::Element> dataline(b);
        for (size_t j = 0; j < b; ++j) {
            dataline[j] = M[i * b + j];
        }
        codewords.push_back(rsencode(dataline, rho_inv));
    }
    mt_t = MerkleTree_ext(codewords);
}

std::vector<Goldilocks2::Element> ligeroProver_ext::lincomb(const std::vector<Goldilocks2::Element>& r) const {
    assert(r.size() == a);
    // std::cout << r.size() << '\n' << a << '\n';
    std::vector<Goldilocks2::Element> v(b, Goldilocks2::zero());
    for (size_t j = 0; j < a; ++j) {
        Goldilocks2::Element tmp;
        size_t offset = j * b;
        for (size_t i = 0; i < b; ++i) {
            Goldilocks2::mul(tmp, r[j], M[offset + i]);
            Goldilocks2::add(v[i], v[i], tmp);
        }
    }
    return v;
}

std::vector<MerkleTree_ext::MTPayload> ligeroProver_ext::open_cols(const std::vector<size_t>& indexes) const {
    std::vector<MerkleTree_ext::MTPayload> payloads;
    for (size_t e : indexes) {
        payloads.push_back(mt_t.MerkleOpen(e));
    }
    return payloads;
}



ligeropcs_ext ligeroProver_ext::commit() const {
    return { mt_t.MerkleCommit(), std::make_shared<ligeroProver_ext>(*this), a, b };
}


std::mt19937_64 ligeroVerifier::gen(std::random_device{}());
std::uniform_int_distribution<uint64_t> ligeroVerifier::dist(0, Goldilocks2::p - 1);

Goldilocks2::Element ligeroVerifier::randnum() {
    return { Goldilocks::fromU64(dist(gen)), Goldilocks::fromU64(dist(gen)) };
}

std::vector<Goldilocks2::Element> ligeroVerifier::randvec(const uint64_t& n) {
    std::vector<Goldilocks2::Element> rands;
    rands.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
        rands.push_back(randnum());
    }
    return rands;
}

std::vector<size_t> ligeroVerifier::randindexes(const uint64_t& n, const size_t& bound) {
    static std::mt19937 _gen(std::random_device{}());
    std::uniform_int_distribution<size_t> _dist(0, bound - 1);
    std::vector<size_t> rands;
    rands.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
        rands.push_back(_dist(_gen));
    }
    return rands;
}


bool ligeroVerifier::check_lincomb(const ligeropcs_base& pcs, const std::vector<Goldilocks2::Element>& r, const std::vector<Goldilocks2::Element>& comb, const size_t& t) {
    const auto& prover = *pcs.prover;
    std::vector<size_t> indexes = randindexes(t, std::ceil(pcs.num_cols * prover.rho_inv));
    // std::vector<size_t> indexes = {7, 7, 7, 7, 7};

    auto openings = prover.open_cols(indexes);
    add_proof_size(openings[0].column.size() * openings.size() * sizeof(Goldilocks2::Element));
    for (auto opening : openings) {
        // check if this opening is right
        if (!MerkleTree_base::MerkleVerify(pcs.mthash, opening)) return false;
    }

    for (size_t k = 0; k < indexes.size(); ++k) {
        size_t idx = indexes[k];
        // check if this entry is correctly computed
        Goldilocks2::Element entry = Goldilocks2::zero();
        Goldilocks2::Element tmp;
        for (size_t i = 0; i < pcs.num_rows; ++i) {
            Goldilocks2::mul(tmp, r[i], openings[k].column[i]);
            Goldilocks2::add(entry, entry, tmp);
        }
        if (entry != comb[idx]) return false;
    }

    return true;
}

bool ligeroVerifier::check_lincomb(const ligeropcs_ext& pcs, const std::vector<Goldilocks2::Element>& r, const std::vector<Goldilocks2::Element>& comb, const size_t& t) {
    const auto& prover = *pcs.prover;
    std::vector<size_t> indexes = randindexes(t, pcs.num_cols * prover.rho_inv);
    // std::vector<size_t> indexes = {7, 7, 7, 7, 7};

    auto openings = prover.open_cols(indexes);
    // std::cout << openings[0].column.size() << ", " << openings.size() << ", " << sizeof(Goldilocks2::Element) << std::endl;
    add_proof_size(openings[0].column.size() * openings.size() * sizeof(Goldilocks2::Element));

    for (auto opening : openings) {
        // check if this opening is right
        if (!MerkleTree_ext::MerkleVerify(pcs.mthash, opening)) return false;
    }

    for (size_t k = 0; k < indexes.size(); ++k) {
        size_t idx = indexes[k];
        // check if this entry is correctly computed
        Goldilocks2::Element entry = Goldilocks2::zero();
        Goldilocks2::Element tmp;
        for (size_t i = 0; i < pcs.num_rows; ++i) {
            Goldilocks2::mul(tmp, r[i], openings[k].column[i]);
            Goldilocks2::add(entry, entry, tmp);
        }
        if (entry != comb[idx]) return false;
    }

    return true;
}

bool ligeroVerifier::check_commit(const ligeropcs_base& pcs, const size_t& sec_param) {
    const auto& prover = *pcs.prover;
    std::vector<Goldilocks2::Element> r = randvec(pcs.num_rows);
    std::vector<Goldilocks2::Element> v = prover.lincomb(r);
    std::vector<Goldilocks2::Element> w = rsencode(v, prover.rho_inv);
    size_t t = calculate_t(sec_param, prover.rho_inv, prover.codelen, FIELD_BITS);
    return check_lincomb(pcs, r, w, t);
}

bool ligeroVerifier::check_commit(const ligeropcs_ext& pcs, const size_t& sec_param) {
    const auto& prover = *pcs.prover;
    std::vector<Goldilocks2::Element> r = randvec(pcs.num_rows);
    std::vector<Goldilocks2::Element> v = prover.lincomb(r);
    std::vector<Goldilocks2::Element> w = rsencode(v, prover.rho_inv);
    size_t t = calculate_t(sec_param, prover.rho_inv, prover.codelen, FIELD_BITS);
    return check_lincomb(pcs, r, w, t);
}

// maybe can be moved to util?
// calculate b^T dot u dot a, notation from pazk(https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)
Goldilocks2::Element tensor_product(const std::vector<Goldilocks2::Element>& b, const std::vector<std::vector<Goldilocks2::Element>>& u, const std::vector<Goldilocks2::Element>& a) {
    // assume (1, h) * (h, w) * (w, 1)
    size_t h = b.size();
    size_t w = a.size();
    assert(h == u.size() && u[0].size() == w);
    Goldilocks2::Element res = Goldilocks2::zero();
    for (size_t i = 0; i < w; ++i) {
        Goldilocks2::Element term = Goldilocks2::zero();
        Goldilocks2::Element tmp;
        for (size_t j = 0; j < h; ++j) {
            Goldilocks2::mul(tmp, b[j], u[j][i]);
            Goldilocks2::add(term, term, tmp);
        }
        Goldilocks2::add(res, res, term);
    }
    return res;
}

Goldilocks2::Element dot_product(const std::vector<Goldilocks2::Element>& b, const std::vector<Goldilocks2::Element>& a) {
    size_t n = b.size();
    assert(n == a.size());
    Goldilocks2::Element res = Goldilocks2::zero();
    for (size_t i = 0; i < n; ++i) {
        Goldilocks2::Element tmp;
        Goldilocks2::mul(tmp, b[i], a[i]);
        Goldilocks2::add(res, res, tmp);
    }
    return res;
}

void find_parameter() {
    clear_proof();
    for (int i = 25; i < 30; ++i) {
        double opt_sz;
        int optj = 0;
        for (int j = 10; j < i; ++j) {
            size_t N = (1ull << i);
            size_t rho_inv = 2;
            std::vector<Goldilocks2::Element> e(N);
            auto pcs = ligero_commit_base(e, rho_inv, j);
            start_proof("find_parameter");
            pcs.open(random_vec_ext(i), 32);
            end_proof("find_parameter");
            double sz = get_proof_size("find_parameter", Counter::KB);
            clear_proof();
            if (optj && sz > opt_sz) {
                break;
            }
            if (!optj || sz < opt_sz) {
                opt_sz = sz;
                optj = j;
            }
            std::cout << "logn = " << i << ", a = " << j << ", proof size = " << sz << " KB" << std::endl;
        }
        std::cout << "--------- logn = " << i << ", a = " << optj << ", proof size = " << opt_sz << " KB" << std::endl;
    }
    clear_proof();
}

void print_all_open_cnt() {
    std::cout << "ligero open counts:\n";
    for (const auto& [key, value] : ligero_open_cnt) {
        std::cout << "  size " << key << ": " << value << '\n';
    }
}

Goldilocks2::Element ligeroVerifier::open(const ligeropcs_base& pcs, const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) {
    startCounter counter("ligero_open");
    std::array<std::vector<Goldilocks2::Element>, 2> lr = calculate_lr(z.size(), z, pcs.num_rows);

    std::vector<Goldilocks2::Element> L = lr[0];
    std::vector<Goldilocks2::Element> R = lr[1];

    const auto& prover = *pcs.prover;
    // v_prime for v', idealy we have E(v') = w', w' = R dot uhat
    std::vector<Goldilocks2::Element> v_prime = prover.lincomb(R);
    add_proof_size(sizeof(Goldilocks2::Element) * v_prime.size());

    std::vector<Goldilocks2::Element> w_prime = rsencode(v_prime, prover.rho_inv);
    size_t t = calculate_t(sec_param, prover.rho_inv, prover.codelen, FIELD_BITS);
    if (!check_lincomb(pcs, R, w_prime, t)) {
        throw std::runtime_error("ligeroVerifier::open: lincomb check failed");
    }

    return dot_product(v_prime, L);
}

Goldilocks2::Element ligeroVerifier::open(const ligeropcs_ext& pcs, const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) {
    startCounter counter("ligero_open");
    std::array<std::vector<Goldilocks2::Element>, 2> lr = calculate_lr(z.size(), z, pcs.num_rows);

    std::vector<Goldilocks2::Element> L = lr[0];
    std::vector<Goldilocks2::Element> R = lr[1];

    const auto& prover = *pcs.prover;
    // v_prime for v', idealy we have E(v') = w', w' = R dot uhat
    std::vector<Goldilocks2::Element> v_prime = prover.lincomb(R);
    add_proof_size(sizeof(Goldilocks2::Element) * v_prime.size());

    std::vector<Goldilocks2::Element> w_prime = rsencode(v_prime, prover.rho_inv);
    size_t t = calculate_t(sec_param, prover.rho_inv, prover.codelen, FIELD_BITS);
    if (!check_lincomb(pcs, R, w_prime, t)) {
        throw std::runtime_error("ligeroVerifier::open: lincomb check failed");
    }

    return dot_product(v_prime, L);
}

std::array<std::vector<Goldilocks2::Element>, 2> ligeroVerifier::calculate_lr(const size_t& num_var, const std::vector<Goldilocks2::Element>& z, size_t num_rows) {
    // different from a,b in prover, a, b here are the log of each
    size_t a = find_ceiling_log2(num_rows);
    size_t b = num_var - a;

    // high bit of z
    std::vector<Goldilocks2::Element> zh(z.begin(), z.begin() + a);
    // low bits of z
    std::vector<Goldilocks2::Element> zl(z.begin() + a, z.end());

    std::vector<Goldilocks2::Element> L = eq_table(b, zl);
    std::vector<Goldilocks2::Element> R = eq_table(a, zh);

    return { L, R };
}

size_t ligeroVerifier::calculate_t(
    const size_t& sec_param,
    const uint64_t& rho_inv,
    const size_t& codeword_len,
    const size_t& field_bits
) {
    // residual = n / 2^field_bits
    double residual = static_cast<double>(codeword_len) / std::pow(2.0, field_bits);

    double pr = std::pow(2.0, -static_cast<int>(sec_param));
    // unless target sec level can't be achieved
    assert(pr + 1e-7 > residual);

    double numerator = std::log2(pr - residual) - 1;
    double denominator = std::log2((1.0 + 1.0 / static_cast<double>(rho_inv)) * 0.5);
    size_t t = static_cast<size_t>(std::ceil(numerator / denominator));
    return std::min(t, codeword_len);
}

ligeropcs_base::ligeropcs_base(const MerkleDef::Digest& mthash, const std::shared_ptr<ligeroProver_base>& prover, const size_t& num_rows, const size_t& num_cols)
    : mthash(mthash), prover(prover), num_rows(num_rows), num_cols(num_cols) {
}

bool ligeropcs_base::empty() const {
    return num_rows == 0;
}

bool ligeropcs_ext::empty() const {
    return num_rows == 0;
}

Goldilocks2::Element ligeropcs_base::open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const {
    if (ligero_open_cnt.count(z.size()) == 0) {
        ligero_open_cnt[z.size()] = 0;
    }
    ++ligero_open_cnt[z.size()];
    return ligeroVerifier::open(*this, z, sec_param);
}

int ligeropcs_base::get_num_vars() const {
    return prover->get_num_vars();
}

ligeropcs_ext::ligeropcs_ext(const MerkleDef::Digest& mthash, const std::shared_ptr<ligeroProver_ext>& prover, const size_t& num_rows, const size_t& num_cols)
    : mthash(mthash), prover(prover), num_rows(num_rows), num_cols(num_cols) {
}

Goldilocks2::Element ligeropcs_ext::open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const {
    if (ligero_open_cnt.count(z.size()) == 0) {
        ligero_open_cnt[z.size()] = 0;
    }
    ++ligero_open_cnt[z.size()];
    return ligeroVerifier::open(*this, z, sec_param);
}

int ligeropcs_ext::get_num_vars() const {
    return prover->get_num_vars();
}
