#include "orion.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>

#include "counter.h"
#include "product2_sumcheck.h"
#include "timer.h"
#include "util.h"

namespace {

constexpr double ORION_ALPHA = 0.238;
constexpr double ORION_R = 1.72;
constexpr int ORION_CN = 10;
constexpr int ORION_DN = 20;
constexpr int ORION_DISTANCE_THRESHOLD = 13;
constexpr size_t ORION_DEFAULT_QUERIES = 100;

// Tuned from the single-threaded base-field sweep up to logn=33. The selected
// points favor Pareto knees: smaller opening proofs without the slowest matrix
// shapes in the local window.
constexpr int opt_loga_orion[] = {-1, 1, 1, 2, 3, 4, 5, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                                  6, 6, 6, 7, 5, 8, 7, 8, 8, 9, 9, 10, 10, 11, 11, 11};

struct orionGraph {
    size_t L = 0;
    size_t R = 0;
    int degree = 0;
    std::vector<std::vector<size_t>> neighbor;
    std::vector<std::vector<Goldilocks::Element>> weight;
};

struct orionPlan {
    size_t n = 0;
    std::vector<orionGraph> C;
    std::vector<orionGraph> D;
};

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

Goldilocks::Element deterministic_weight(uint64_t seed) {
    uint64_t raw = splitmix64(seed);
    if (raw == 0) raw = 1;
    return Goldilocks::fromU64(raw);
}

orionGraph generate_graph(size_t L, size_t R, int degree, uint64_t seed) {
    if (R == 0) {
        throw std::runtime_error("orion: graph right side cannot be zero");
    }
    orionGraph ret;
    ret.L = L;
    ret.R = R;
    ret.degree = degree;
    ret.neighbor.resize(L);
    ret.weight.resize(L);
    for (size_t i = 0; i < L; ++i) {
        ret.neighbor[i].resize(degree);
        ret.weight[i].resize(degree);
        for (int d = 0; d < degree; ++d) {
            uint64_t raw = splitmix64(seed ^ (static_cast<uint64_t>(i) << 32) ^ static_cast<uint64_t>(d));
            ret.neighbor[i][d] = raw % R;
            ret.weight[i][d] = deterministic_weight(raw ^ 0xd1b54a32d192ed03ULL);
        }
    }
    return ret;
}

size_t init_plan_rec(orionPlan& plan, size_t n, size_t dep, size_t root_n) {
    if (n <= ORION_DISTANCE_THRESHOLD) {
        return n;
    }

    if (plan.C.size() <= dep) {
        plan.C.resize(dep + 1);
        plan.D.resize(dep + 1);
    }

    size_t c_right = std::max<size_t>(1, static_cast<size_t>(ORION_ALPHA * static_cast<double>(n)));
    uint64_t c_seed = 0x4f52494f4e434f44ULL ^ (root_n << 1) ^ (dep << 17);
    plan.C[dep] = generate_graph(n, c_right, ORION_CN, c_seed);

    size_t middle_len = init_plan_rec(plan, c_right, dep + 1, root_n);
    double d_right_raw = static_cast<double>(n) * (ORION_R - 1.0) - static_cast<double>(middle_len);
    size_t d_right = std::max<size_t>(1, static_cast<size_t>(d_right_raw));
    uint64_t d_seed = 0x4f52494f4e444743ULL ^ (root_n << 1) ^ (dep << 17);
    plan.D[dep] = generate_graph(middle_len, d_right, ORION_DN, d_seed);

    return n + middle_len + d_right;
}

std::shared_ptr<const orionPlan> get_plan(size_t n) {
    static std::mutex mtx;
    static std::map<size_t, std::shared_ptr<const orionPlan>> cache;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = cache.find(n);
    if (it != cache.end()) {
        return it->second;
    }
    auto plan = std::make_shared<orionPlan>();
    plan->n = n;
    size_t actual_len = init_plan_rec(*plan, n, 0, n);
    if (actual_len > 2 * n) {
        throw std::runtime_error("orion: encoded row exceeded 2n buffer");
    }
    cache[n] = plan;
    return plan;
}

void get_ab_orion(int num_vars, int loga, size_t& a, size_t& b) {
    if (loga == -1 || loga > num_vars) {
        int default_loga = orion_default_loga(num_vars);
        a = 1ull << default_loga;
        b = 1ull << (num_vars - default_loga);
    } else {
        a = 1ull << loga;
        b = 1ull << (num_vars - loga);
    }
}

template <typename T>
T zero_of();

template <>
Goldilocks::Element zero_of<Goldilocks::Element>() {
    return Goldilocks::zero();
}

template <>
Goldilocks2::Element zero_of<Goldilocks2::Element>() {
    return Goldilocks2::zero();
}

template <typename T>
T mul_weight(const T& x, const Goldilocks::Element& w) {
    return x * w;
}

template <typename T>
void add_assign(T& dst, const T& src) {
    dst = dst + src;
}

template <typename T>
std::vector<T> encode_inner(const std::vector<T>& src, const orionPlan& plan, size_t dep) {
    size_t n = src.size();
    if (n <= ORION_DISTANCE_THRESHOLD) {
        return src;
    }
    const orionGraph& c_graph = plan.C.at(dep);
    const orionGraph& d_graph = plan.D.at(dep);
    if (c_graph.L != n) {
        throw std::runtime_error("orion: C graph size mismatch");
    }

    std::vector<T> mid(c_graph.R, zero_of<T>());
    for (size_t i = 0; i < n; ++i) {
        for (int d = 0; d < c_graph.degree; ++d) {
            size_t target = c_graph.neighbor[i][d];
            add_assign(mid[target], mul_weight(src[i], c_graph.weight[i][d]));
        }
    }

    std::vector<T> encoded_mid = encode_inner(mid, plan, dep + 1);
    if (d_graph.L != encoded_mid.size()) {
        throw std::runtime_error("orion: D graph size mismatch");
    }

    std::vector<T> right(d_graph.R, zero_of<T>());
    for (size_t i = 0; i < encoded_mid.size(); ++i) {
        for (int d = 0; d < d_graph.degree; ++d) {
            size_t target = d_graph.neighbor[i][d];
            add_assign(right[target], mul_weight(encoded_mid[i], d_graph.weight[i][d]));
        }
    }

    std::vector<T> ret;
    ret.reserve(src.size() + encoded_mid.size() + right.size());
    ret.insert(ret.end(), src.begin(), src.end());
    ret.insert(ret.end(), encoded_mid.begin(), encoded_mid.end());
    ret.insert(ret.end(), right.begin(), right.end());
    return ret;
}

template <typename T>
struct orionEncodeScratch {
    std::vector<std::vector<T>> mids;
};

template <typename T>
size_t encode_inner_into(const T* src,
                         size_t n,
                         const orionPlan& plan,
                         size_t dep,
                         T* out,
                         orionEncodeScratch<T>& scratch) {
    if (n <= ORION_DISTANCE_THRESHOLD) {
        std::copy(src, src + n, out);
        return n;
    }
    const orionGraph& c_graph = plan.C.at(dep);
    const orionGraph& d_graph = plan.D.at(dep);
    if (c_graph.L != n) {
        throw std::runtime_error("orion: C graph size mismatch");
    }

    std::copy(src, src + n, out);

    if (scratch.mids.size() <= dep) {
        scratch.mids.resize(dep + 1);
    }
    std::vector<T>& mid = scratch.mids[dep];
    mid.assign(c_graph.R, zero_of<T>());
    for (size_t i = 0; i < n; ++i) {
        for (int d = 0; d < c_graph.degree; ++d) {
            size_t target = c_graph.neighbor[i][d];
            add_assign(mid[target], mul_weight(src[i], c_graph.weight[i][d]));
        }
    }

    size_t encoded_mid_len = encode_inner_into(mid.data(), c_graph.R, plan, dep + 1, out + n, scratch);
    if (d_graph.L != encoded_mid_len) {
        throw std::runtime_error("orion: D graph size mismatch");
    }

    const T* encoded_mid = out + n;
    T* right = out + n + encoded_mid_len;
    std::fill(right, right + d_graph.R, zero_of<T>());
    for (size_t i = 0; i < encoded_mid_len; ++i) {
        for (int d = 0; d < d_graph.degree; ++d) {
            size_t target = d_graph.neighbor[i][d];
            add_assign(right[target], mul_weight(encoded_mid[i], d_graph.weight[i][d]));
        }
    }

    return n + encoded_mid_len + d_graph.R;
}

template <typename T>
void orion_encode_padded_into(const T* data,
                              size_t n,
                              T* out,
                              orionEncodeScratch<T>& scratch) {
    if (n == 0 || !is_power_of_2(n)) {
        throw std::invalid_argument("orion_encode: data size must be a non-empty power of two");
    }
    auto plan = get_plan(n);
    std::fill(out, out + 2 * n, zero_of<T>());
    size_t encoded_len = encode_inner_into(data, n, *plan, 0, out, scratch);
    if (encoded_len > 2 * n) {
        throw std::runtime_error("orion_encode: encoded row exceeded 2n buffer");
    }
}

template <typename T>
std::vector<T> orion_encode_padded_impl(const std::vector<T>& data) {
    if (data.empty() || !is_power_of_2(data.size())) {
        throw std::invalid_argument("orion_encode: data size must be a non-empty power of two");
    }
    std::vector<T> padded(2 * data.size(), zero_of<T>());
    orionEncodeScratch<T> scratch;
    orion_encode_padded_into(data.data(), data.size(), padded.data(), scratch);
    return padded;
}

template <typename T>
std::vector<T> orion_encode_impl(const std::vector<T>& data) {
    std::vector<T> padded = orion_encode_padded_impl(data);

    std::vector<T> ret;
    ret.reserve(4 * data.size());
    ret.insert(ret.end(), padded.begin(), padded.end());
    ret.insert(ret.end(), padded.begin(), padded.end());
    return ret;
}

Goldilocks2::Element dot_product_ext(const std::vector<Goldilocks2::Element>& a,
                                     const std::vector<Goldilocks2::Element>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("orion dot_product: size mismatch");
    }
    Goldilocks2::Element ret = Goldilocks2::zero();
    for (size_t i = 0; i < a.size(); ++i) {
        ret += a[i] * b[i];
    }
    return ret;
}

size_t orion_query_count(size_t sec_param, size_t code_len) {
    size_t requested = std::max<size_t>(ORION_DEFAULT_QUERIES, (sec_param + 1) / 2);
    return std::min(requested, code_len);
}

size_t encoded_inner_len(const orionPlan& plan, size_t n, size_t dep) {
    if (n <= ORION_DISTANCE_THRESHOLD) {
        return n;
    }
    const orionGraph& c_graph = plan.C.at(dep);
    const orionGraph& d_graph = plan.D.at(dep);
    size_t middle_len = encoded_inner_len(plan, c_graph.R, dep + 1);
    if (d_graph.L != middle_len) {
        throw std::runtime_error("orion dual: D graph size mismatch");
    }
    return n + middle_len + d_graph.R;
}

std::vector<Goldilocks2::Element> encode_inner_dual_ext(const std::vector<Goldilocks2::Element>& adj,
                                                        const orionPlan& plan,
                                                        size_t n,
                                                        size_t dep) {
    if (n <= ORION_DISTANCE_THRESHOLD) {
        if (adj.size() != n) {
            throw std::runtime_error("orion dual: base size mismatch");
        }
        return adj;
    }

    const orionGraph& c_graph = plan.C.at(dep);
    const orionGraph& d_graph = plan.D.at(dep);
    size_t middle_len = encoded_inner_len(plan, c_graph.R, dep + 1);
    size_t expected = n + middle_len + d_graph.R;
    if (adj.size() != expected || c_graph.L != n || d_graph.L != middle_len) {
        throw std::runtime_error("orion dual: graph size mismatch");
    }

    std::vector<Goldilocks2::Element> grad_src(n, Goldilocks2::zero());
    std::copy(adj.begin(), adj.begin() + n, grad_src.begin());

    std::vector<Goldilocks2::Element> grad_encoded_mid(adj.begin() + n, adj.begin() + n + middle_len);
    const Goldilocks2::Element* grad_right = adj.data() + n + middle_len;
    for (size_t i = 0; i < middle_len; ++i) {
        for (int d = 0; d < d_graph.degree; ++d) {
            size_t target = d_graph.neighbor[i][d];
            grad_encoded_mid[i] += grad_right[target] * d_graph.weight[i][d];
        }
    }

    std::vector<Goldilocks2::Element> grad_mid = encode_inner_dual_ext(grad_encoded_mid, plan, c_graph.R, dep + 1);
    for (size_t i = 0; i < n; ++i) {
        for (int d = 0; d < c_graph.degree; ++d) {
            size_t target = c_graph.neighbor[i][d];
            grad_src[i] += grad_mid[target] * c_graph.weight[i][d];
        }
    }
    return grad_src;
}

std::vector<Goldilocks2::Element> orion_encode_dual_ext(size_t message_len,
                                                        const std::vector<size_t>& indexes,
                                                        const std::vector<Goldilocks2::Element>& weights) {
    if (message_len == 0 || !is_power_of_2(message_len)) {
        throw std::invalid_argument("orion dual: message length must be a non-empty power of two");
    }
    if (indexes.size() != weights.size()) {
        throw std::invalid_argument("orion dual: index/weight size mismatch");
    }

    auto plan = get_plan(message_len);
    size_t padded_len = 2 * message_len;
    size_t code_len = 4 * message_len;
    size_t actual_len = encoded_inner_len(*plan, message_len, 0);
    std::vector<Goldilocks2::Element> padded_adj(padded_len, Goldilocks2::zero());
    for (size_t i = 0; i < indexes.size(); ++i) {
        if (indexes[i] >= code_len) {
            throw std::out_of_range("orion dual: queried index exceeds code length");
        }
        size_t inner_idx = indexes[i] % padded_len;
        if (inner_idx < actual_len) {
            padded_adj[inner_idx] += weights[i];
        }
    }
    std::vector<Goldilocks2::Element> adj(padded_adj.begin(), padded_adj.begin() + actual_len);
    return encode_inner_dual_ext(adj, *plan, message_len, 0);
}

} // namespace

std::map<int, int> orion_open_cnt;

int orion_default_loga(int num_vars) {
    if (num_vars < 0 || num_vars >= static_cast<int>(sizeof(opt_loga_orion) / sizeof(opt_loga_orion[0]))) {
        throw std::runtime_error("orion: num_vars too large for default matrix parameters");
    }
    return opt_loga_orion[num_vars];
}

std::vector<Goldilocks::Element> orion_encode_base(const std::vector<Goldilocks::Element>& data) {
    return orion_encode_impl(data);
}

std::vector<Goldilocks2::Element> orion_encode_ext(const std::vector<Goldilocks2::Element>& data) {
    return orion_encode_impl(data);
}

orionProver_base::orionProver_base(const MultilinearPolynomial& w, const uint64_t& _rho_inv, int loga)
    : rho_inv(_rho_inv) {
    startTimer _timer("orion commit");
    const auto& evals = w.get_eval_table();
    num_vars = find_ceiling_log2(evals.size());
    get_ab_orion(num_vars, loga, a, b);

    M = std::make_shared<std::vector<Goldilocks::Element>>(1ull << num_vars, Goldilocks::zero());
    for (size_t i = 0; i < evals.size(); ++i) {
        (*M)[i] = evals[i][0];
    }

    set_timer("orion encode");
    code_len = 4 * b;
    const size_t stored_code_len = 2 * b;
    std::unique_ptr<Goldilocks::Element[]> codewords(new Goldilocks::Element[a * stored_code_len]);
    #pragma omp parallel
    {
        orionEncodeScratch<Goldilocks::Element> scratch;
        #pragma omp for schedule(static)
        for (size_t i = 0; i < a; ++i) {
            orion_encode_padded_into(&(*M)[i * b], b, &codewords[i * stored_code_len], scratch);
        }
    }
    pause_timer("orion encode");

    set_timer("orion merkle");
    if (a * stored_code_len >= (1ull << 33)) {
        mt_t = std::make_shared<MerkleTree_base>(std::move(codewords), a, stored_code_len, code_len);
    } else {
        mt_t = std::make_shared<MerkleTree_base>(codewords.get(), a, stored_code_len, code_len);
    }
    pause_timer("orion merkle");
}

orionProver_base::orionProver_base(const std::vector<uint64_t>& w, const uint64_t& _rho_inv, int loga)
    : rho_inv(_rho_inv) {
    startTimer _timer("orion commit");
    num_vars = find_ceiling_log2(w.size());
    get_ab_orion(num_vars, loga, a, b);

    M = std::make_shared<std::vector<Goldilocks::Element>>(1ull << num_vars, Goldilocks::zero());
    for (size_t i = 0; i < w.size(); ++i) {
        (*M)[i] = Goldilocks::fromU64(w[i]);
    }

    set_timer("orion encode");
    code_len = 4 * b;
    const size_t stored_code_len = 2 * b;
    std::unique_ptr<Goldilocks::Element[]> codewords(new Goldilocks::Element[a * stored_code_len]);
    #pragma omp parallel
    {
        orionEncodeScratch<Goldilocks::Element> scratch;
        #pragma omp for schedule(static)
        for (size_t i = 0; i < a; ++i) {
            orion_encode_padded_into(&(*M)[i * b], b, &codewords[i * stored_code_len], scratch);
        }
    }
    pause_timer("orion encode");

    set_timer("orion merkle");
    if (a * stored_code_len >= (1ull << 33)) {
        mt_t = std::make_shared<MerkleTree_base>(std::move(codewords), a, stored_code_len, code_len);
    } else {
        mt_t = std::make_shared<MerkleTree_base>(codewords.get(), a, stored_code_len, code_len);
    }
    pause_timer("orion merkle");
}

std::vector<Goldilocks2::Element> orionProver_base::lincomb(const std::vector<Goldilocks2::Element>& r) const {
    if (r.size() != a) {
        throw std::invalid_argument("orionProver_base::lincomb: challenge length mismatch");
    }
    std::vector<Goldilocks2::Element> ret(b, Goldilocks2::zero());
    constexpr size_t BLOCK = 512;
    #pragma omp parallel for schedule(static)
    for (size_t j0 = 0; j0 < b; j0 += BLOCK) {
        size_t j1 = std::min(j0 + BLOCK, b);
        for (size_t i = 0; i < a; ++i) {
            const Goldilocks2::Element ri = r[i];
            const Goldilocks::Element* row = M->data() + i * b;
            for (size_t j = j0; j < j1; ++j) {
                ret[j] += ri * row[j];
            }
        }
    }
    return ret;
}

std::vector<MerkleTree_base::MTPayload> orionProver_base::open_cols(const std::vector<size_t>& indexes) const {
    std::vector<MerkleTree_base::MTPayload> payloads;
    payloads.reserve(indexes.size());
    for (size_t idx : indexes) {
        payloads.push_back(mt_t->MerkleOpen(idx));
    }
    return payloads;
}

orionpcs_base orionProver_base::commit() const {
    auto root = mt_t->MerkleCommit();
    add_proof_size(sizeof(root));
    return { root, std::make_shared<orionProver_base>(*this), a, b };
}

orionProver_ext::orionProver_ext(const MultilinearPolynomial& w, const uint64_t& _rho_inv, int loga)
    : orionProver_ext(w.get_eval_table(), _rho_inv, loga) {
}

orionProver_ext::orionProver_ext(const std::vector<Goldilocks2::Element>& w, const uint64_t& _rho_inv, int loga)
    : rho_inv(_rho_inv) {
    startTimer _timer("orion commit");
    num_vars = find_ceiling_log2(w.size());
    get_ab_orion(num_vars, loga, a, b);

    M = std::make_shared<std::vector<Goldilocks2::Element>>(1ull << num_vars, Goldilocks2::zero());
    std::copy(w.begin(), w.end(), M->begin());

    set_timer("orion encode");
    code_len = 4 * b;
    const size_t stored_code_len = 2 * b;
    std::vector<std::vector<Goldilocks2::Element>> codewords(a);
    #pragma omp parallel
    {
        orionEncodeScratch<Goldilocks2::Element> scratch;
        #pragma omp for schedule(static)
        for (size_t i = 0; i < a; ++i) {
            codewords[i].resize(stored_code_len, Goldilocks2::zero());
            orion_encode_padded_into(M->data() + i * b, b, codewords[i].data(), scratch);
        }
    }
    pause_timer("orion encode");

    set_timer("orion merkle");
    mt_t = MerkleTree_ext(codewords, stored_code_len, code_len);
    pause_timer("orion merkle");
}

std::vector<Goldilocks2::Element> orionProver_ext::lincomb(const std::vector<Goldilocks2::Element>& r) const {
    if (r.size() != a) {
        throw std::invalid_argument("orionProver_ext::lincomb: challenge length mismatch");
    }
    std::vector<Goldilocks2::Element> ret(b, Goldilocks2::zero());
    constexpr size_t BLOCK = 512;
    #pragma omp parallel for schedule(static)
    for (size_t j0 = 0; j0 < b; j0 += BLOCK) {
        size_t j1 = std::min(j0 + BLOCK, b);
        for (size_t i = 0; i < a; ++i) {
            const Goldilocks2::Element ri = r[i];
            const Goldilocks2::Element* row = M->data() + i * b;
            for (size_t j = j0; j < j1; ++j) {
                ret[j] += ri * row[j];
            }
        }
    }
    return ret;
}

std::vector<MerkleTree_ext::MTPayload> orionProver_ext::open_cols(const std::vector<size_t>& indexes) const {
    std::vector<MerkleTree_ext::MTPayload> payloads;
    payloads.reserve(indexes.size());
    for (size_t idx : indexes) {
        payloads.push_back(mt_t.MerkleOpen(idx));
    }
    return payloads;
}

orionpcs_ext orionProver_ext::commit() const {
    auto root = mt_t.MerkleCommit();
    add_proof_size(sizeof(root));
    return { root, std::make_shared<orionProver_ext>(*this), a, b };
}

orionpcs_base::orionpcs_base(const MerkleDef::Digest& _mthash,
                             const std::shared_ptr<orionProver_base>& _prover,
                             const size_t& _num_rows,
                             const size_t& _num_cols)
    : mthash(_mthash), prover(_prover), num_rows(_num_rows), num_cols(_num_cols) {
}

bool orionpcs_base::empty() const {
    return num_rows == 0;
}

Goldilocks2::Element orionpcs_base::open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const {
    startTimer _timer("orion open");
    if (orion_open_cnt.count(z.size()) == 0) {
        orion_open_cnt[z.size()] = 0;
    }
    ++orion_open_cnt[z.size()];
    return orionVerifier::open(*this, z, sec_param);
}

int orionpcs_base::get_num_vars() const {
    return prover->get_num_vars();
}

orionpcs_ext::orionpcs_ext(const MerkleDef::Digest& _mthash,
                           const std::shared_ptr<orionProver_ext>& _prover,
                           const size_t& _num_rows,
                           const size_t& _num_cols)
    : mthash(_mthash), prover(_prover), num_rows(_num_rows), num_cols(_num_cols) {
}

bool orionpcs_ext::empty() const {
    return num_rows == 0;
}

Goldilocks2::Element orionpcs_ext::open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const {
    startTimer _timer("orion open");
    if (orion_open_cnt.count(z.size()) == 0) {
        orion_open_cnt[z.size()] = 0;
    }
    ++orion_open_cnt[z.size()];
    return orionVerifier::open(*this, z, sec_param);
}

int orionpcs_ext::get_num_vars() const {
    return prover->get_num_vars();
}

std::vector<size_t> orionVerifier::randindexes(const uint64_t& n, const size_t& bound) {
    static std::mt19937_64 gen(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, bound - 1);
    std::vector<size_t> ret;
    ret.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
        ret.push_back(dist(gen));
    }
    return ret;
}

std::array<std::vector<Goldilocks2::Element>, 2> orionVerifier::calculate_lr(
    const size_t& num_var,
    const std::vector<Goldilocks2::Element>& z,
    size_t num_rows
) {
    size_t row_vars = find_ceiling_log2(num_rows);
    size_t col_vars = num_var - row_vars;
    std::vector<Goldilocks2::Element> zh(z.begin(), z.begin() + row_vars);
    std::vector<Goldilocks2::Element> zl(z.begin() + row_vars, z.end());
    std::vector<Goldilocks2::Element> L = eq_table(col_vars, zl);
    std::vector<Goldilocks2::Element> R = eq_table(row_vars, zh);
    return { L, R };
}

bool orionVerifier::check_lincomb(const orionpcs_base& pcs,
                                  const std::vector<Goldilocks2::Element>& r,
                                  const std::vector<size_t>& indexes,
                                  const std::vector<Goldilocks2::Element>& weights,
                                  Goldilocks2::Element& claim) {
    const auto& prover = *pcs.prover;
    if (indexes.size() != weights.size()) {
        throw std::invalid_argument("orionVerifier::check_lincomb: index/weight size mismatch");
    }
    auto openings = prover.open_cols(indexes);

    set_timer(VERIFIER_TIMER);
    set_timer("verifier orion open");
    if (!openings.empty()) {
        add_proof_size(openings[0].sz_col * openings.size() * sizeof(Goldilocks::Element));
        add_proof_size(openings[0].path.size() * openings.size() * sizeof(MerkleDef::Digest));
    }

    for (auto& opening : openings) {
        if (!MerkleTree_base::MerkleVerify(pcs.mthash, opening)) {
            return false;
        }
    }

    bool ret = true;
    claim = Goldilocks2::zero();
    std::vector<Goldilocks2::Element> entries(openings.size(), Goldilocks2::zero());
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < openings.size(); ++k) {
        Goldilocks2::Element entry = Goldilocks2::zero();
        for (size_t i = 0; i < pcs.num_rows; ++i) {
            entry += r[i] * openings[k].col[i];
        }
        entries[k] = entry;
    }
    for (size_t k = 0; k < entries.size(); ++k) {
        claim += weights[k] * entries[k];
    }
    pause_timer(VERIFIER_TIMER);
    pause_timer("verifier orion open");
    return ret;
}

bool orionVerifier::check_lincomb(const orionpcs_ext& pcs,
                                  const std::vector<Goldilocks2::Element>& r,
                                  const std::vector<size_t>& indexes,
                                  const std::vector<Goldilocks2::Element>& weights,
                                  Goldilocks2::Element& claim) {
    const auto& prover = *pcs.prover;
    if (indexes.size() != weights.size()) {
        throw std::invalid_argument("orionVerifier::check_lincomb: index/weight size mismatch");
    }
    auto openings = prover.open_cols(indexes);

    set_timer(VERIFIER_TIMER);
    set_timer("verifier orion open");
    if (!openings.empty()) {
        add_proof_size(openings[0].column.size() * openings.size() * sizeof(Goldilocks2::Element));
        add_proof_size(openings[0].path.size() * openings.size() * sizeof(MerkleDef::Digest));
    }

    for (auto& opening : openings) {
        if (!MerkleTree_ext::MerkleVerify(pcs.mthash, opening)) {
            return false;
        }
    }

    bool ret = true;
    claim = Goldilocks2::zero();
    std::vector<Goldilocks2::Element> entries(openings.size(), Goldilocks2::zero());
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < openings.size(); ++k) {
        Goldilocks2::Element entry = Goldilocks2::zero();
        for (size_t i = 0; i < pcs.num_rows; ++i) {
            entry += r[i] * openings[k].column[i];
        }
        entries[k] = entry;
    }
    for (size_t k = 0; k < entries.size(); ++k) {
        claim += weights[k] * entries[k];
    }
    pause_timer(VERIFIER_TIMER);
    pause_timer("verifier orion open");
    return ret;
}

Goldilocks2::Element orionVerifier::open(const orionpcs_base& pcs,
                                         const std::vector<Goldilocks2::Element>& z,
                                         const size_t& sec_param) {
    startCounter counter("orion_open");
    auto lr = calculate_lr(z.size(), z, pcs.num_rows);
    std::vector<Goldilocks2::Element>& L = lr[0];
    std::vector<Goldilocks2::Element>& R = lr[1];

    const auto& prover = *pcs.prover;
    set_timer("orion_open_lincomb");
    std::vector<Goldilocks2::Element> v_prime = prover.lincomb(R);
    pause_timer("orion_open_lincomb");
    Goldilocks2::Element value_claim = dot_product_ext(v_prime, L);

    set_timer("orion_open_codeswitch_commit");
    MLE v_mle(v_prime);
    auto v_pcs = ligero_commit_ext(v_mle, 2);
    pause_timer("orion_open_codeswitch_commit");

    size_t t = orion_query_count(sec_param, prover.get_code_len());
    std::vector<size_t> indexes = randindexes(t, prover.get_code_len());
    std::vector<Goldilocks2::Element> query_weights(t, Goldilocks2::one());
    Goldilocks2::Element lambda = random_ext();
    for (size_t i = 1; i < t; ++i) {
        query_weights[i] = query_weights[i - 1] * lambda;
    }

    Goldilocks2::Element codeword_claim = Goldilocks2::zero();
    set_timer("orion_open_check_lincomb");
    if (!check_lincomb(pcs, R, indexes, query_weights, codeword_claim)) {
        throw std::runtime_error("orionVerifier::open: lincomb check failed");
    }
    pause_timer("orion_open_check_lincomb");

    set_timer("orion_open_codeswitch");
    // Code-switching-lite: prove the evaluation claim and sampled encoder
    // consistency against the derived v_prime commitment without sending v_prime.
    std::vector<Goldilocks2::Element> codeword_coeff = orion_encode_dual_ext(v_prime.size(), indexes, query_weights);
    Goldilocks2::Element eta = random_ext();
    std::vector<Goldilocks2::Element> combined_coeff(v_prime.size(), Goldilocks2::zero());
    for (size_t i = 0; i < v_prime.size(); ++i) {
        combined_coeff[i] = L[i] + eta * codeword_coeff[i];
    }
    Goldilocks2::Element combined_claim = value_claim + eta * codeword_claim;

    MLE coeff_mle(std::move(combined_coeff));
    auto coeff_ptr = std::make_unique<MLE>(coeff_mle);
    auto v_ptr = std::make_unique<MLE>(v_mle);
    p2Prover codeswitch_prover(std::move(coeff_ptr), std::move(v_ptr));
    if (!p2Verifier::execute_sumcheck(codeswitch_prover, {&coeff_mle, &v_pcs}, combined_claim, sec_param)) {
        throw std::runtime_error("orionVerifier::open: code-switching sumcheck failed");
    }
    pause_timer("orion_open_codeswitch");

    return value_claim;
}

Goldilocks2::Element orionVerifier::open(const orionpcs_ext& pcs,
                                         const std::vector<Goldilocks2::Element>& z,
                                         const size_t& sec_param) {
    startCounter counter("orion_open");
    auto lr = calculate_lr(z.size(), z, pcs.num_rows);
    std::vector<Goldilocks2::Element>& L = lr[0];
    std::vector<Goldilocks2::Element>& R = lr[1];

    const auto& prover = *pcs.prover;
    set_timer("orion_open_lincomb");
    std::vector<Goldilocks2::Element> v_prime = prover.lincomb(R);
    pause_timer("orion_open_lincomb");
    Goldilocks2::Element value_claim = dot_product_ext(v_prime, L);

    set_timer("orion_open_codeswitch_commit");
    MLE v_mle(v_prime);
    auto v_pcs = ligero_commit_ext(v_mle, 2);
    pause_timer("orion_open_codeswitch_commit");

    size_t t = orion_query_count(sec_param, prover.get_code_len());
    std::vector<size_t> indexes = randindexes(t, prover.get_code_len());
    std::vector<Goldilocks2::Element> query_weights(t, Goldilocks2::one());
    Goldilocks2::Element lambda = random_ext();
    for (size_t i = 1; i < t; ++i) {
        query_weights[i] = query_weights[i - 1] * lambda;
    }

    Goldilocks2::Element codeword_claim = Goldilocks2::zero();
    set_timer("orion_open_check_lincomb");
    if (!check_lincomb(pcs, R, indexes, query_weights, codeword_claim)) {
        throw std::runtime_error("orionVerifier::open: lincomb check failed");
    }
    pause_timer("orion_open_check_lincomb");

    set_timer("orion_open_codeswitch");
    // Code-switching-lite: prove the evaluation claim and sampled encoder
    // consistency against the derived v_prime commitment without sending v_prime.
    std::vector<Goldilocks2::Element> codeword_coeff = orion_encode_dual_ext(v_prime.size(), indexes, query_weights);
    Goldilocks2::Element eta = random_ext();
    std::vector<Goldilocks2::Element> combined_coeff(v_prime.size(), Goldilocks2::zero());
    for (size_t i = 0; i < v_prime.size(); ++i) {
        combined_coeff[i] = L[i] + eta * codeword_coeff[i];
    }
    Goldilocks2::Element combined_claim = value_claim + eta * codeword_claim;

    MLE coeff_mle(std::move(combined_coeff));
    auto coeff_ptr = std::make_unique<MLE>(coeff_mle);
    auto v_ptr = std::make_unique<MLE>(v_mle);
    p2Prover codeswitch_prover(std::move(coeff_ptr), std::move(v_ptr));
    if (!p2Verifier::execute_sumcheck(codeswitch_prover, {&coeff_mle, &v_pcs}, combined_claim, sec_param)) {
        throw std::runtime_error("orionVerifier::open: code-switching sumcheck failed");
    }
    pause_timer("orion_open_codeswitch");

    return value_claim;
}

orionpcs_base orion_commit_base(const MultilinearPolynomial& w, const uint64_t& rho_inv, int loga) {
    auto prover = std::make_shared<orionProver_base>(w, rho_inv, loga);
    return prover->commit();
}

orionpcs_base orion_commit_base(const std::vector<uint64_t>& w, const uint64_t& rho_inv, int loga) {
    auto prover = std::make_shared<orionProver_base>(w, rho_inv, loga);
    return prover->commit();
}

orionpcs_ext orion_commit_ext(const MultilinearPolynomial& w, const uint64_t& rho_inv, int loga) {
    auto prover = std::make_shared<orionProver_ext>(w, rho_inv, loga);
    return prover->commit();
}

void print_all_orion_open_cnt() {
    std::cout << "orion open counts:\n";
    for (const auto& [key, value] : orion_open_cnt) {
        std::cout << key << ": " << value << '\n';
    }
}
