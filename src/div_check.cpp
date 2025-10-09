#include "div_check.h"
#include "util.h"
#include "logup.h"
#include "counter.h"

std::map<std::array<uint64_t, 2>, std::vector<Goldilocks2::Element>> divProver::range;
std::map<std::array<uint64_t, 2>, std::vector<Goldilocks2::Element>> divProver::valid;
std::map<uint64_t, std::vector<Goldilocks2::Element>> divProver::zeros;
std::map<std::array<uint64_t, 2>, std::vector<uint64_t>> divProver::range_u64;
std::map<std::array<uint64_t, 2>, std::vector<uint64_t>> divProver::valid_u64;
std::map<uint64_t, std::vector<uint64_t>> divProver::zeros_u64;
std::map<std::array<uint64_t, 3>, MLE> divProver::mle_range;
std::map<std::array<uint64_t, 3>, MLE> divProver::mle_valid;
std::map<std::array<uint64_t, 2>, MLE> divProver::mle_zeros;

divProver::divProver(
    const std::vector<Goldilocks2::Element>& num,
    const std::vector<Goldilocks2::Element>& quo,
    const std::vector<Goldilocks2::Element>& rem,
    uint64_t denominator, bool allow_neg_rem, uint64_t rho_inv,
    lazyLogupProver* _lazy_logup_prover)
    : denom(denominator), allow_neg_rem(allow_neg_rem), rho_inv(rho_inv), lazy_logup_prover(_lazy_logup_prover)
{
    // Step 1. Store num and quo and rem
    if (num.size() != quo.size() || num.size() != rem.size()) {
        throw std::invalid_argument("divProver: Numerator, quotient and remainder must match in sizes");
    }
    num_u64.reserve(num.size());
    quo_u64.reserve(quo.size());
    rem_u64.reserve(rem.size());
    for (const auto& n : num) {
        num_u64.push_back(Goldilocks2::toU64(n)[0]);
    }
    for (const auto& q : quo) {
        quo_u64.push_back(Goldilocks2::toU64(q)[0]);
    }
    for (const auto& r : rem) {
        rem_u64.push_back(Goldilocks2::toU64(r)[0]);
    }

    num_vars = find_ceiling_log2(num.size());

    // Step 2. Init range&valid
    init_range(denominator, allow_neg_rem, rho_inv);

    // Step 3. Init zeros
    init_zeros(num.size());
}


divProver::divProver(
    const std::vector<uint64_t>& num,
    const std::vector<uint64_t>& quo,
    const std::vector<uint64_t>& rem,
    uint64_t denominator, bool allow_neg_rem, uint64_t rho_inv,
    lazyLogupProver* _lazy_logup_prover)
    : denom(denominator), allow_neg_rem(allow_neg_rem), rho_inv(rho_inv), lazy_logup_prover(_lazy_logup_prover)
{
    // Step 1. Store num and quo and rem
    if (num.size() != quo.size() || num.size() != rem.size()) {
        throw std::invalid_argument("divProver: Numerator, quotient and remainder must match in sizes");
    }
    num_u64 = num;
    quo_u64 = quo;
    rem_u64 = rem;

    num_vars = find_ceiling_log2(num.size());

    // Step 2. Init range&valid
    init_range(denominator, allow_neg_rem, rho_inv);

    // Step 3. Init zeros
    init_zeros(num.size());
}

void divProver::init_range(uint64_t denominator, bool allow_neg_rem, uint64_t rho_inv) {
    if (denom == 0) {
        throw std::invalid_argument("divProver: Denominator cannot be zero");
    }
    if (!is_power_of_2(denom)) {
        throw std::invalid_argument("divProver: Denominator must be a power of 2");
    }
    if (range.find({denom, allow_neg_rem}) == range.end()) {
        std::vector<Goldilocks2::Element> new_r, new_v;
        std::vector<uint64_t> new_r_u64, new_v_u64;
        if (allow_neg_rem) new_r.resize(2 * denom);
        else new_r.resize(denom);
        // Positive range
        for (uint64_t i = 0; i < denom; ++i) {
            new_r[i] = Goldilocks2::fromU64(i);
        }
        // Negative range
        if (allow_neg_rem) {
            for (int64_t i = 1; i < denom; ++i) {
                new_r[2 * denom - i - 1] = Goldilocks2::fromS64(-i);
            }
            new_r[2 * denom - 1] = Goldilocks2::fromS64(-static_cast<int64_t>(denom)); // 
        }
        new_v.resize(new_r.size(), Goldilocks2::zero());
        if (allow_neg_rem) new_v.back() = Goldilocks2::one();

        new_r_u64.reserve(new_r.size());
        new_v_u64.reserve(new_v.size());
        for (size_t i = 0; i < new_r.size(); ++i) {
            new_r_u64.push_back(Goldilocks2::toU64(new_r[i])[0]);
        }
        for (size_t i = 0; i < new_v.size(); ++i) {
            new_v_u64.push_back(Goldilocks2::toU64(new_v[i])[0]);
        }
        range[{denom, allow_neg_rem}] = std::move(new_r);
        valid[{denom, allow_neg_rem}] = std::move(new_v);
        range_u64[{denom, allow_neg_rem}] = std::move(new_r_u64);
        valid_u64[{denom, allow_neg_rem}] = std::move(new_v_u64);
    }
    if (mle_range.find({denom, allow_neg_rem, rho_inv}) == mle_range.end()) {
        mle_range[{denom, allow_neg_rem, rho_inv}] = MLE(range[{denom, allow_neg_rem}]);
        mle_valid[{denom, allow_neg_rem, rho_inv}] = MLE(valid[{denom, allow_neg_rem}]);
    }
}

void divProver::init_zeros(size_t vec_len) {
    if (!is_power_of_2(vec_len)) {
        throw std::invalid_argument("divProver: Vector length must be a power of 2");
    }
    if (zeros.find(vec_len) == zeros.end()) {
        zeros[vec_len] = std::vector<Goldilocks2::Element>(vec_len, Goldilocks2::zero());
        zeros_u64[vec_len] = std::vector<uint64_t>(vec_len, 0);
    }
    if (mle_zeros.find({vec_len, rho_inv}) == mle_zeros.end()) {
        mle_zeros[{vec_len, rho_inv}] = MLE(zeros[vec_len]);
    }
}

std::array<std::vector<Goldilocks2::Element>, 2> get_quo_rem(
    const std::vector<Goldilocks2::Element>& num,
    uint64_t den,
    bool allow_neg_rem) {

    std::vector<Goldilocks2::Element> quo(num.size()), rem(num.size());
    for (size_t i = 0; i < num.size(); ++i) {
        int64_t quo_val = Goldilocks2::toS64(num[i]) / int64_t(den);
        int64_t rem_val = Goldilocks2::toS64(num[i]) - quo_val * int64_t(den);
        if (!allow_neg_rem && rem_val < 0) {
            quo_val -= 1;
            rem_val += den;
        }
        quo[i] = Goldilocks2::fromS64(quo_val);
        rem[i] = Goldilocks2::fromS64(rem_val);
    }
    return {quo, rem};
}

std::array<std::vector<uint64_t>, 2> get_quo_rem(
    const std::vector<uint64_t>& num,
    uint64_t den,
    bool allow_neg_rem) {

    std::vector<uint64_t> quo(num.size()), rem(num.size());
    for (size_t i = 0; i < num.size(); ++i) {
        int64_t quo_val = Goldilocks2::toS64(Goldilocks2::fromU64(num[i])) / int64_t(den);
        int64_t rem_val = Goldilocks2::toS64(Goldilocks2::fromU64(num[i])) - quo_val * int64_t(den);
        if (!allow_neg_rem && rem_val < 0) {
            quo_val -= 1;
            rem_val += den;
        }
        quo[i] = Goldilocks2::fromS64(quo_val)[0].fe;
        rem[i] = Goldilocks2::fromS64(rem_val)[0].fe;
    }
    return {quo, rem};
}

std::vector<Goldilocks2::Element> get_rem(
    const std::vector<Goldilocks2::Element>& num,
    uint64_t den,
    const std::vector<Goldilocks2::Element>& quo,
    bool allow_neg_rem) {

    std::vector<Goldilocks2::Element> rem(num.size());
    for (size_t i = 0; i < num.size(); ++i) {
        Goldilocks2::Element quotient = quo[i];
        Goldilocks2::Element remainder = num[i] - quotient * Goldilocks2::fromU64(den);
        int64_t remval = Goldilocks2::toS64(remainder);
        if (allow_neg_rem) {
            if (remval <= -static_cast<int64_t>(den) || remval >= static_cast<int64_t>(den)) {
                throw std::runtime_error("divProver: Remainder out of range");
            }
        }
        else {
            if (remval < 0 || remval >= static_cast<int64_t>(den)) {
                throw std::runtime_error("divProver: Remainder out of range");
            }
        }
        rem[i] = remainder;
    }
    return rem;
}

uint64_t divProver::get_denom() const {
    return denom;
}

LogupProver divProver::get_logup_prover() const {
    
    return LogupProver(rem_u64, zeros_u64[rem_u64.size()], range_u64[{denom, allow_neg_rem}], valid_u64[{denom, allow_neg_rem}]);
}

std::array<size_t, 2> divProver::get_lazy_logup_prover() const {

    return lazy_logup_prover->add(rem_u64, zeros_u64[rem_u64.size()], range_u64[{denom, allow_neg_rem}], valid_u64[{denom, allow_neg_rem}]);
}

bool divVerifier::execute_div_check(
    const divProver& prover,
    std::shared_ptr<oracle> pcs_num,
    std::shared_ptr<oracle> pcs_quo,
    std::shared_ptr<oracle> pcs_rem,
    size_t sec_param,
    lazyLogupVerifier* lazy_logup_verifier) {

    startCounter counter("div_proof");

    bool use_lazy_logup = (lazy_logup_verifier != nullptr);
    if (use_lazy_logup != prover.use_lazy_logup()) {
        throw std::invalid_argument("divVerifier: Lazy logup prover/verifier mismatch");
    }

    // Step 1. Prove pcs_num = pcs_quo * denom + pcs_rem
    std::vector<Goldilocks2::Element> cha = random_vec_ext(prover.get_num_vars());
    if (pcs_num->open(cha, sec_param) != pcs_quo->open(cha, sec_param) * Goldilocks2::fromU64(prover.get_denom()) + pcs_rem->open(cha, sec_param)) {
        std::cerr << "❌ Div check failed: pcs_num != pcs_quo * denom + pcs_rem" << std::endl;
        std::cerr << "pcs_num: " << pcs_num->open(cha, sec_param) << std::endl;
        std::cerr << "pcs_quo: " << pcs_quo->open(cha, sec_param) << std::endl;
        std::cerr << "pcs_rem: " << pcs_rem->open(cha, sec_param) << std::endl;
        std::cerr << "denom: " << prover.get_denom() << std::endl;
        return false;
    }

    // Step 2. Check if pcs_rem is in the valid range
    if (use_lazy_logup) {
        auto ind = prover.get_lazy_logup_prover();
        lazy_logup_verifier->add(
            pcs_rem,
            std::make_shared<MLE>(prover.get_mle_zeros()),
            prover.get_range(),
            prover.get_valid(),
            ind);
        return true;
    }
    auto logup_prover = prover.get_logup_prover();
    if (!LogupVerifier::execute_logup(logup_prover,
        *pcs_rem,
        prover.get_mle_zeros(),
        prover.get_mle_range(),
        prover.get_mle_valid(),
        prover.rho_inv, sec_param)) {
        std::cerr << "❌ Div check failed: Logup verification failed" << std::endl;
        return false;
    }

    return true;
}
