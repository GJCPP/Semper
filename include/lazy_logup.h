#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <mutex>

#include "logup.h"
#include "lazy_pcs.h"

class logupTable {
public:
    std::vector<uint64_t> t1, t2;

    uint64_t hash() const {
        std::hash<uint64_t> vec_hash;
        const uint64_t prime = 0x9e3779b97f4a7c15;
        uint64_t seed = t1.size() + t2.size();
        for (const auto& v : t1) {
            seed ^= vec_hash(v) + prime + (seed << 6) + (seed >> 2);
        }
        for (const auto& v : t2) {
            seed ^= vec_hash(v) + prime + (seed << 6) + (seed >> 2);
        }
        return seed;
    }

    bool operator==(const logupTable& other) const {
        return t1 == other.t1 && t2 == other.t2;
    }
};

class lazyLogupProver {
public:
    
    class logupInstance {
    public:
        std::vector<uint64_t> f1, f2;
    };
    lazyLogupProver() = default;

    std::array<size_t, 2> add(const std::vector<uint64_t>& f1,
                   const std::vector<uint64_t>& f2,
                   const std::vector<uint64_t>& t1,
                   const std::vector<uint64_t>& t2);

    void start_prove();

    std::pair<lazy_pcs, lazy_pcs> commit_sort_f(size_t id, const std::vector<size_t>& order, std::shared_ptr<lazy_pcs_pool> pool);

    LogupProver get_logup_prover(size_t id);

    // void lock() { mut.lock(); }
    // void release() { mut.unlock(); }

protected:
    std::vector<std::vector<logupInstance>> instances_all;
    std::vector<logupTable> tables_all;
    std::vector<std::vector<uint64_t>> f1_all, f2_all;
    std::map<uint64_t, std::vector<size_t>> index;
    std::mutex mut;
};

class lazyLogupVerifier {
public:
    class logupInstance {
    public:
        std::shared_ptr<oracle> pcs_f1, pcs_f2;
    };
    lazyLogupVerifier() = default;

    void add(std::shared_ptr<oracle> pcs_f1,
            std::shared_ptr<oracle> pcs_f2,
            const std::vector<uint64_t>& t1,
            const std::vector<uint64_t>& t2,
            std::array<size_t, 2> index);

    bool prove_all(lazyLogupProver& prover, protoque& que, uint64_t rho_inv, uint64_t sec_param);

    std::vector<size_t> sort_f(std::vector<logupInstance>& instances);
protected:
    std::vector<std::vector<logupInstance>> instances_all;
    std::vector<logupTable> tables_all;
    std::map<uint64_t, std::vector<size_t>> index;

    std::mutex mut;
};
