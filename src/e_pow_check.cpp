#include "util.h"
#include "logup.h"
#include "e_pow_check.h"
#include "ltn_check.h"
#include "counter.h"

eProver::eProver(
    const std::vector<size_t>& from, const std::vector<size_t>& to, 
    size_t scale, size_t max_val, size_t rho_inv, 
    lazyLogupProver *lazy_logup_prover)
    : from(from), to(to), num(from.size()), scale(scale), max_val(max_val), rho_inv(rho_inv), lazy_logup_prover(lazy_logup_prover) {
    
    if (from.size() != to.size()) {
        throw std::invalid_argument("eProver: Size of 'from' and 'to' vectors must be the same.");
    }
    if (!is_power_of_2(from.size())) {
        throw std::invalid_argument("eProver: Size of 'from' vector must be a power of 2.");
    }
    for (const auto& v : from) {
        if (v >= max_val) {
            throw std::invalid_argument("eProver: Elements in 'from' vector must be less than max_val.");
        }
    }
    eVerifier::init_e_table(scale, rho_inv);
    bar = eVerifier::e_pow_from[scale].size();
    init();
}

eProver::eProver(
    const std::vector<Goldilocks2::Element>& from_e, 
    const std::vector<Goldilocks2::Element>& to_e, 
    size_t scale, size_t max_val, size_t rho_inv, 
    lazyLogupProver *lazy_logup_prover)
    : num(from_e.size()), scale(scale), max_val(max_val), rho_inv(rho_inv), lazy_logup_prover(lazy_logup_prover) {

    if (from_e.size() != to_e.size()) {
        throw std::invalid_argument("eProver: Size of 'from' and 'to' vectors must be the same.");
    }
    if (!is_power_of_2(from_e.size())) {
        throw std::invalid_argument("eProver: Size of 'from' vector must be a power of 2.");
    }
    from.resize(num);
    to.resize(num);
    for (size_t i = 0; i < num; ++i) {
        from[i] = from_e[i][0].fe;
        to[i] = to_e[i][0].fe;
    }
    for (const auto& v : from) {
        if (v >= max_val) {
            throw std::invalid_argument("eProver: Elements in 'from' vector must be less than max_val.");
        }
    }
    init();
    eVerifier::init_e_table(scale, rho_inv);
}

void eProver::init() {
    
    filtered_from = from;
    for (size_t i = 0; i < num; ++i) {
        if (from[i] >= bar) {
            filtered_from[i] = 0; // Map to 0
        }
    }

    masked_from = from;
    for (size_t i = 0; i < num; ++i) {
        if (from[i] >= bar) {
            masked_from[i] = bar - 1; // Map to max_val - 1
        }
    }
}

LogupProver eProver::get_logup_prover(
    const std::vector<uint64_t>& e_from, 
    const std::vector<uint64_t>& e_to) {
    if (lazy_logup_prover) {
        lazy_logup_prover->add(from, to, e_from, e_to);
        return {};
    }
    return LogupProver(from, to, e_from, e_to);
}

std::array<size_t, 2> eProver::get_lazy_logup_prover(
    const std::vector<uint64_t>& e_from, 
    const std::vector<uint64_t>& e_to) {

    return lazy_logup_prover->add(from, to, e_from, e_to);
}

ltnProver eProver::prove_ltn(uint64_t n) {
    bar = n;
    return ltn_prover = ltnProver(from, Goldilocks2::fromU64(bar), scale, max_val, true, rho_inv, lazy_logup_prover);
}

lazy_pcs eProver::pre_commit_ltn(lazy_pcs_pool* pool) {
    return commit_lazy_pcs(ltn_prover.get_ltn(), pool);
}

ligeropcs_base eProver::commit_ltn() {
    return ligero_commit_base(ltn_prover.get_ltn(), rho_inv);
}

ligeropcs_base eProver::commit_filtered_from() {
    pcs_filtered_from = ligero_commit_base(filtered_from, rho_inv);
    return pcs_filtered_from;
}

lazy_pcs eProver::pre_commit_filtered_from(lazy_pcs_pool* pool) {
    return commit_lazy_pcs(filtered_from, pool);
}

ligeropcs_base eProver::commit_masked_from() {
    pcs_masked_from = ligero_commit_base(masked_from, rho_inv);
    return pcs_masked_from;
}

lazy_pcs eProver::pre_commit_masked_from(lazy_pcs_pool* pool) {
    return commit_lazy_pcs(masked_from, pool);
}

LogupProver eProver::get_masked_logup_prover(
    const std::vector<uint64_t>& e_from,
    const std::vector<uint64_t>& e_to) {

    return LogupProver(masked_from, to, e_from, e_to);
}

std::array<size_t, 2> eProver::get_lazy_masked_logup_prover(
    const std::vector<uint64_t>& e_from,
    const std::vector<uint64_t>& e_to) {

    return lazy_logup_prover->add(masked_from, to, e_from, e_to);
}

void eVerifier::init_e_table(size_t scale, size_t rho_inv) {
    if (e_pow_from.find(scale) != e_pow_from.end()) {
        return; // Already initialized
    }
    auto& from = e_pow_from[scale] = {};
    auto& to = e_pow_to[scale] = {};
    int end = -1;
    for (int i = 0; i != end; ++i) {
        from.push_back(i);
        to.push_back(Goldilocks2::toU64(Goldilocks2::fromS64(std::round(std::exp(static_cast<double>(-i) / scale) * scale)))[0]);
        if (end == -1 && to.back() == 0) {
            end = (1 << find_ceiling_log2(i));
            if (end == i) break;
        }
    }
    mle_e_pow_from[scale] = MLE(from);
    mle_e_pow_to[scale] = MLE(to);
}

bool eVerifier::execute_check(
    eProver& prover, 
    std::shared_ptr<oracle> pcs_from, 
    std::shared_ptr<oracle> pcs_to, 
    size_t sec_param, 
    lazyLogupVerifier* lazy_logup_verifier) {

    startCounter counter("epow_proof");
    const size_t scale = prover.get_scale();
    const size_t max_val = prover.get_max_val();
    const size_t rho_inv = prover.get_rho_inv();
    
    bool use_lazy_logup = (lazy_logup_verifier != nullptr);
    if (use_lazy_logup != prover.use_lazy_logup()) {
        throw std::invalid_argument("eVerifier: disagree in lazy_logup.");
    }

    const auto& e_pow_from = eVerifier::e_pow_from[scale];
    const auto& e_pow_to = eVerifier::e_pow_to[scale];
    const auto& mle_e_pow_from = eVerifier::mle_e_pow_from[scale];
    const auto& mle_e_pow_to = eVerifier::mle_e_pow_to[scale];
    const size_t bar = e_pow_from.size();
    const size_t num = prover.get_num();
    const int lognum = find_ceiling_log2(num);
    if (max_val < e_pow_from.size()) {
        // Just look up the table.
        if (use_lazy_logup) {
            auto ind = prover.get_lazy_logup_prover(e_pow_from, e_pow_to);
            lazy_logup_verifier->add(
                pcs_from,
                pcs_to,
                e_pow_from, e_pow_to, ind);
                return true;
        }
        auto logup_prover = prover.get_logup_prover(e_pow_from, e_pow_to);
        if (!LogupVerifier::execute_logup(logup_prover, *pcs_from, *pcs_to, mle_e_pow_from, mle_e_pow_to, rho_inv, sec_param)) {
            std::cerr << "❌ eVerifier: execute_logup failed (pure lookup)." << std::endl;
            return false;
        }
    } else {
        // Step 1. Filter out [>= n] elements
        ltnProver ltn_prover = prover.prove_ltn(bar);
        auto pcs_ltn = std::make_shared<ligeropcs_base>(prover.commit_ltn());
        if (!ltnVerifier::execute_ltn_check(
            ltn_prover,
            pcs_from,
            pcs_ltn,
            Goldilocks2::fromU64(bar),
            max_val, true, sec_param, lazy_logup_verifier)) {
            std::cerr << "❌ eVerifier: execute_ltn_check failed." << std::endl;
            return false;
        }
        oracle_sum pcs_rev_ltn;
        pcs_rev_ltn.add(pcs_ltn, Goldilocks2::negone());
        pcs_rev_ltn.add_const(Goldilocks2::one());
        // ligeropcs_base pcs_rev_ltn = ltn_prover.get_pcs_rev_ltn();
        ligeropcs_base pcs_filtered_from = prover.commit_filtered_from();
        if (!prove_mle_product(
            prover.get_filtered_from(), ltn_prover.get_ltn(), prover.get_from(),
            pcs_filtered_from, *pcs_ltn, *pcs_from, sec_param)) {
            std::cerr << "❌ eVerifier: prove_mle_product failed." << std::endl;
            return false;
        }

        // Step 2. Prove masked_from = filtered_from + rev_ltn * (bar - 1)
        ligeropcs_base pcs_masked_from = prover.commit_masked_from();
        auto mask_cha = random_vec_ext(lognum);
        if (pcs_masked_from.open(mask_cha, sec_param) != pcs_filtered_from.open(mask_cha, sec_param) + 
            pcs_rev_ltn.open(mask_cha, sec_param) * Goldilocks2::fromU64(bar - 1)) {
            std::cerr << "❌ eVerifier: masked_from != filtered_from + rev_ltn * (max_val - 1)" << std::endl;
            return false;
        }

        // Step 3. Look up with masked_from
        if (use_lazy_logup) {
            auto ind = prover.get_lazy_masked_logup_prover(e_pow_from, e_pow_to);
            lazy_logup_verifier->add(
                std::make_shared<ligeropcs_base>(pcs_masked_from),
                pcs_to,
                e_pow_from, e_pow_to, ind);
                return true;
        }
        LogupProver logup_prover = prover.get_masked_logup_prover(e_pow_from, e_pow_to);
        if (!LogupVerifier::execute_logup(
            logup_prover, pcs_masked_from, *pcs_to, mle_e_pow_from, mle_e_pow_to, rho_inv, sec_param)) {
            std::cerr << "❌ eVerifier: execute_logup failed (masked lookup)." << std::endl;
            return false;
        }
    }
    return true;
}

std::vector<size_t> eVerifier::get_exp_inv(const std::vector<size_t>& from, size_t scale, size_t rho_inv) {
    init_e_table(scale, rho_inv);
    std::vector<size_t> ret(from.size());
    auto& e_to = eVerifier::e_pow_to[scale];
    size_t bar = e_to.size();
    for (size_t i = 0; i < from.size(); ++i) {
        if (from[i] < bar) ret[i] = e_to[from[i]];
        else ret[i] = 0;
    }
    return ret;
}

eVerifier::resource eVerifier::pre_execute_check(eProver& prover, lazyLogupVerifier* lazy_logup_verifier, lazy_pcs_pool *pool) {

    resource ret;

    const size_t scale = prover.get_scale();
    const size_t max_val = prover.get_max_val();
    const size_t rho_inv = prover.get_rho_inv();
    
    bool use_lazy_logup = (lazy_logup_verifier != nullptr);
    if (use_lazy_logup != prover.use_lazy_logup()) {
        throw std::invalid_argument("eVerifier: disagree in lazy_logup.");
    }

    const auto& e_pow_from = eVerifier::e_pow_from[scale];
    const auto& e_pow_to = eVerifier::e_pow_to[scale];
    const auto& mle_e_pow_from = eVerifier::mle_e_pow_from[scale];
    const auto& mle_e_pow_to = eVerifier::mle_e_pow_to[scale];
    const size_t bar = e_pow_from.size();
    const size_t num = prover.get_num();
    const int lognum = find_ceiling_log2(num);
    if (max_val < e_pow_from.size()) {
        // Just look up the table.
        return {};
    } else {
        // Step 1. Filter out [>= n] elements
        ltnProver ltn_prover = prover.prove_ltn(bar);
        ret.pcs_ltn = prover.pre_commit_ltn(pool);
        ret.ltn_res = ltnVerifier::pre_execute_ltn_check(
            ltn_prover,
            Goldilocks2::fromU64(bar),
            max_val, true, lazy_logup_verifier, pool);
            
            
        ret.pcs_filtered_from = prover.pre_commit_filtered_from(pool);

        // Step 2. Prove masked_from = filtered_from + rev_ltn * (bar - 1)
        ret.pcs_masked_from = prover.pre_commit_masked_from(pool);
    }
    return ret;
}

bool eVerifier::execute_check(eProver& prover, std::shared_ptr<oracle> pcs_from, std::shared_ptr<oracle> pcs_to, size_t sec_param, lazyLogupVerifier* lazy_logup_verifier, resource& res) {
    
    startCounter counter("epow_proof");
    const size_t scale = prover.get_scale();
    const size_t max_val = prover.get_max_val();
    const size_t rho_inv = prover.get_rho_inv();
    
    bool use_lazy_logup = (lazy_logup_verifier != nullptr);
    if (use_lazy_logup != prover.use_lazy_logup()) {
        throw std::invalid_argument("eVerifier: disagree in lazy_logup.");
    }

    const auto& e_pow_from = eVerifier::e_pow_from[scale];
    const auto& e_pow_to = eVerifier::e_pow_to[scale];
    const auto& mle_e_pow_from = eVerifier::mle_e_pow_from[scale];
    const auto& mle_e_pow_to = eVerifier::mle_e_pow_to[scale];
    const size_t bar = e_pow_from.size();
    const size_t num = prover.get_num();
    const int lognum = find_ceiling_log2(num);
    if (max_val < e_pow_from.size()) {
        // Just look up the table.
        if (use_lazy_logup) {
            auto ind = prover.get_lazy_logup_prover(e_pow_from, e_pow_to);
            lazy_logup_verifier->add(
                pcs_from,
                pcs_to,
                e_pow_from, e_pow_to, ind);
            return true;
        }
        auto logup_prover = prover.get_logup_prover(e_pow_from, e_pow_to);
        if (!LogupVerifier::execute_logup(logup_prover, *pcs_from, *pcs_to, mle_e_pow_from, mle_e_pow_to, rho_inv, sec_param)) {
            std::cerr << "❌ eVerifier: execute_logup failed (pure lookup)." << std::endl;
            return false;
        }
    } else {
        // Step 1. Filter out [>= n] elements
        ltnProver ltn_prover = prover.prove_ltn(bar);
        auto pcs_ltn = std::make_shared<lazy_pcs>(res.pcs_ltn);
        ltnVerifier::resource ltn_res = res.ltn_res;
        if (!ltnVerifier::execute_ltn_check(
            ltn_prover,
            pcs_from,
            pcs_ltn,
            Goldilocks2::fromU64(bar),
            max_val, true, sec_param, lazy_logup_verifier, ltn_res)) {
            std::cerr << "❌ eVerifier: execute_ltn_check failed." << std::endl;
            return false;
        }
        oracle_sum pcs_rev_ltn;
        pcs_rev_ltn.add(pcs_ltn, Goldilocks2::negone());
        pcs_rev_ltn.add_const(Goldilocks2::one());
        lazy_pcs pcs_filtered_from = res.pcs_filtered_from;
        if (!prove_mle_product(
            prover.get_filtered_from(), ltn_prover.get_ltn(), prover.get_from(),
            pcs_filtered_from, *pcs_ltn, *pcs_from, sec_param)) {
            std::cerr << "❌ eVerifier: prove_mle_product failed." << std::endl;
            return false;
        }

        // Step 2. Prove masked_from = filtered_from + rev_ltn * (bar - 1)
        auto pcs_masked_from = std::make_shared<lazy_pcs>(res.pcs_masked_from);
        auto mask_cha = random_vec_ext(lognum);
        if (pcs_masked_from->open(mask_cha, sec_param) != pcs_filtered_from.open(mask_cha, sec_param) + 
            pcs_rev_ltn.open(mask_cha, sec_param) * Goldilocks2::fromU64(bar - 1)) {
            std::cerr << "❌ eVerifier: masked_from != filtered_from + rev_ltn * (max_val - 1)" << std::endl;
            return false;
        }

        // Step 3. Look up with masked_from
        if (use_lazy_logup) {
            auto ind = prover.get_lazy_masked_logup_prover(e_pow_from, e_pow_to);
            lazy_logup_verifier->add(
                pcs_masked_from,
                pcs_to,
                e_pow_from, e_pow_to, ind);
            return true;
        }
        LogupProver logup_prover = prover.get_masked_logup_prover(e_pow_from, e_pow_to);
        if (!LogupVerifier::execute_logup(
            logup_prover, *pcs_masked_from, *pcs_to, mle_e_pow_from, mle_e_pow_to, rho_inv, sec_param)) {
            std::cerr << "❌ eVerifier: execute_logup failed (masked lookup)." << std::endl;
            return false;
        }
    }
    return true;
}


std::map<size_t, std::vector<uint64_t>> eVerifier::e_pow_from, eVerifier::e_pow_to;    
std::map<size_t, MLE> eVerifier::mle_e_pow_from, eVerifier::mle_e_pow_to;
