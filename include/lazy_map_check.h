#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <mutex>

#include "lazy_pcs.h"
#include "perm_check.h"

class lazyMapProver {
public:
    struct mapProofInstance {
        std::vector<size_t> map_index;
        MLE mle;
        size_t ins_ind;
    };

    lazyMapProver(bool use_ext = false) : use_ext(use_ext) {}

    void add(
        const std::vector<size_t>& map_from,
        const std::vector<size_t>& map_to,
        const MLE& mle_left,
        const MLE& mle_right);

    void add(
        const std::vector<size_t>& map_from,
        const std::vector<size_t>& map_to,
        const std::vector<MLE>& mle_left,
        const std::vector<MLE>& mle_right,
        Goldilocks2::Element alpha);

    mapProver commit_left_right(
        std::shared_ptr<lazy_pcs_pool> pool_left, std::shared_ptr<lazy_pcs_pool> pool_right, 
        std::shared_ptr<oracle>& pcs_left, std::shared_ptr<oracle>& pcs_right,
        uint64_t rho_inv);

    bool using_ext() const { return use_ext; }
    
    void lock() { mut.lock(); }
    void release() { mut.unlock(); }

protected:
    std::mutex mut;
    bool use_ext;
    std::vector<mapProofInstance> ins_left, ins_right;
};

class lazyMapVerifier {
public:
    lazyMapVerifier(bool use_ext = false) : use_ext(use_ext) {}
    void add(
        const std::vector<size_t>& map_from,
        const std::vector<size_t>& map_to,
        std::shared_ptr<oracle> pcs_left,
        std::shared_ptr<oracle> pcs_right);

    void add(
        const std::vector<size_t>& map_from,
        const std::vector<size_t>& map_to,
        std::vector<std::shared_ptr<oracle>> pcs_left,
        std::vector<std::shared_ptr<oracle>> pcs_right,
        Goldilocks2::Element alpha);

    bool prove_all(lazyMapProver& prover, uint64_t rho_inv, uint64_t sec_param);

protected:
    struct mapProofInstance {
        std::vector<size_t> map_index;
        std::shared_ptr<oracle> pcs;
        size_t ins_ind;
    };
    bool use_ext;
    std::vector<mapProofInstance> ins_left, ins_right;
};
