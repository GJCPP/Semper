#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <map>

#include "oracle.h"
#include "ligero.h"

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

    lazyLogupProver(const std::vector<uint64_t>& f1,
                   const std::vector<uint64_t>& f2,
                   const std::vector<uint64_t>& t1,
                   const std::vector<uint64_t>& t2);

    std::pair<ligeropcs_base, ligeropcs_base> commit_sort_f(size_t id, const std::vector<size_t>& order, uint64_t rho_inv) {
        if (order.size() != instances_all[id].size()) {
            throw std::runtime_error("lazyLogupProver: order size must match the number of instances");
        }
        std::vector<logupInstance> tmp(instances_all[id].size());
        for (size_t i = 0; i < instances_all[id].size(); i++) {
            tmp[i] = instances_all[id][order[i]];
        }
        instances_all[id] = tmp;
        // check if sorted
        for (size_t i = 1; i < instances_all[id].size(); i++) {
            if (instances_all[id][i-1].f1.size() < instances_all[id][i].f1.size()) {
                throw std::runtime_error("lazyLogupProver: f is not sorted");
            }
        }
        std::vector<uint64_t> f1, f2;
        for (size_t i = 0; i < instances_all[id].size(); i++) {
            f1.insert(f1.end(), instances_all[id][i].f1.begin(), instances_all[id][i].f1.end());
            f2.insert(f2.end(), instances_all[id][i].f2.begin(), instances_all[id][i].f2.end());
        }
        return { ligero_commit_base(f1, rho_inv), ligero_commit_base(f2, rho_inv) };
    }

protected:
    std::vector<std::vector<logupInstance>> instances_all;
    std::vector<logupTable> tables_all;
    std::map<uint64_t, std::vector<size_t>> index;
};

class lazyLogupVerifier {
public:
    class logupInstance {
    public:
        std::shared_ptr<oracle> pcs_f1, pcs_f2;
    };
    lazyLogupVerifier() = default;

    lazyLogupVerifier(std::shared_ptr<oracle> pcs_f1,
                     std::shared_ptr<oracle> pcs_f2,
                     const std::vector<uint64_t>& t1,
                     const std::vector<uint64_t>& t2);

    bool prove_all(lazyLogupProver& prover, uint64_t rho_inv, uint64_t sec_param);

    std::vector<size_t> sort_f(std::vector<logupInstance>& instances);
protected:
    std::vector<std::vector<logupInstance>> instances_all;
    std::vector<logupTable> tables_all;
    std::map<uint64_t, std::vector<size_t>> index;
};
