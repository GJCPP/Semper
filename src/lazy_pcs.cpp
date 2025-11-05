#include <map>

#include "lazy_pcs.h"
#include "timer.h"
#include "util.h"
#include "ligero.h"
#include "counter.h"

// #define OMIT_PCS

std::shared_ptr<oracle> lazy_pcs_pool::commit(uint64_t rho_inv) {

    if (committed) {
        throw std::runtime_error("lazy_pcs_pool::commit: already committed");
    }
    committed = true;


    struct sortIns {
        size_t ind;
        int num_vars;
    };
    std::vector<sortIns> elements;
    elements.reserve(mles.size());
    for (size_t i = 0; i < mles.size(); i++) {
        elements.push_back({i, mles[i].first->get_num_vars()});
    }
    set_timer("lazy_pcs_pool sort");
    std::sort(elements.begin(), elements.end(), [](const auto& a, const auto& b) {
        return a.num_vars > b.num_vars;
    }); // sort in descending order
    pause_timer("lazy_pcs_pool sort");

    
    set_timer("lazy_pcs_pool reorder");
    std::vector<std::pair<std::shared_ptr<MLE>, size_t>> sorted_mles(elements.size());
    order.resize(mles.size());
    perm.resize(mles.size());
    
    // #pragma omp parallel for
    for (size_t i = 0; i != elements.size(); ++i) {
        sorted_mles[i] = std::move(mles[elements[i].ind]);
        order[elements[i].ind] = i;
        perm[i] = elements[i].ind;
    }
    mles = std::move(sorted_mles);
    pause_timer("lazy_pcs_pool reorder");

    set_timer("lazy_pcs_pool mle");
    std::vector<size_t> ind(mles.size() + 1, 0);
    size_t total = 0;
    for (auto& mle : mles) {
        total += (1ull << mle.first->get_num_vars());
    }
    num_vars = find_ceiling_log2(total);
    total = (1ull << num_vars);
    prefix.resize(mles.size());
    std::vector<Goldilocks2::Element> all_vals(total);
    for (size_t i = 0; i < mles.size(); ++i) {
        ind[i + 1] = ind[i] + mles[i].first->get_eval_table().size();

    }
    #pragma omp parallel for
    for (size_t i = 0; i < mles.size(); ++i) {
        std::vector<Goldilocks2::Element> pre;
        size_t len = (1ull << mles[i].first->get_num_vars());
        size_t left = 0, right = total;
        while (right - left != len) {
            size_t mid = (left + right) / 2;
            if (ind[i] < mid) {
                right = mid;
                pre.push_back(Goldilocks2::zero());
            } else {
                left = mid;
                pre.push_back(Goldilocks2::one());
            }
        }
        prefix[mles[i].second] = std::move(pre);
        memcpy(&all_vals[ind[i]], mles[i].first->get_eval_table().data(), mles[i].first->get_eval_table().size() * sizeof(Goldilocks2::Element));
        // all_vals.insert(all_vals.end(), mles[i].first->get_eval_table().begin(), mles[i].first->get_eval_table().end());
    }
    uni_mle = MLE(std::move(all_vals));
    pause_timer("lazy_pcs_pool mle");
    // std::cout << "PCS Size = " << uni_mle.get_num_vars() << ".\n";
    
    // std::cout << "================Warning: skip lazy_pcs commit" << std::endl;
    // return std::make_shared<MLE>(uni_mle);
    // std::cout << "Committing lazy pcs with num_vars = " << num_vars << "...\n";
    // std::cout << "use_ext = " << use_ext << std::endl;

    
#ifdef OMIT_PCS
    std::cout << "====== warning: lazy_pcs_pool::commit skipped." << std::endl;
    return {};
#endif

    set_timer("lazy_pcs_pool commit");
    if (use_ext) {
        auto r = std::make_shared<ligeropcs_ext>(ligero_commit_ext(uni_mle, rho_inv));
        pause_timer("lazy_pcs_pool commit");
        return r;
    } else {
        auto r = std::make_shared<ligeropcs_base>(ligero_commit_base(uni_mle, rho_inv));
        pause_timer("lazy_pcs_pool commit");
        return r;
    }
}

void lazy_pcs_pool::record_open(size_t ind, const std::vector<Goldilocks2::Element>& z, Goldilocks2::Element val, size_t sec) {
    
#ifdef OMIT_PCS
    std::cout << "====== warning: lazy_pcs_pool::record_open skipped." << std::endl;
    return ;
#endif

    std::lock_guard<std::mutex> lock(mtx);
    if (!committed) {
        throw std::runtime_error("lazy_pcs_pool::record_open: not committed");
    }
    if (finalized) {
        throw std::runtime_error("lazy_pcs_pool::record_open: already finalized");
    }
    if (z.size() + prefix[ind].size() != static_cast<size_t>(num_vars)) {
        throw std::runtime_error("lazy_pcs_pool::record_open: number of variables mismatch");
    }
    // std::chrono::high_resolution_clock clock;
    // auto start = clock.now();
    if (sec_param < sec) sec_param = sec;
    std::vector<Goldilocks2::Element> zpad(num_vars);
    size_t pre_sz = prefix[ind].size();
    for (size_t i = 0; i < pre_sz; ++i) {
        zpad[i] = prefix[ind][i];
    }
    for (size_t i = 0; i < z.size(); ++i) {
        zpad[i + pre_sz] = z[i];
    }
    open_val.push_back(val);
    open_eqs.push_back(zpad);
    open_ind.push_back(ind);
}

bool lazy_pcs_pool::prove_open(std::shared_ptr<oracle> pcs, Goldilocks2::Element lambda) {

#ifdef OMIT_PCS
    std::cout << "====== warning: lazy_pcs_pool::prove_open skipped." << std::endl;
    return true;
#endif

    if (!committed) {
        throw std::runtime_error("lazy_pcs_pool::prove_open: not committed");
    }
    if (finalized) {
        throw std::runtime_error("lazy_pcs_pool::prove_open: already finalized");
    }
    finalized = true;

    if (open_eqs.size() == 0) return true;

    std::vector<Goldilocks2::Element> table(1ull << num_vars);
    Goldilocks2::Element claim = Goldilocks2::zero();
    std::vector<Goldilocks2::Element> base(open_eqs.size());
    oracle_sum eq_sum_oracle;
    

    set_timer("lazy pcs open gen");
    base[0] = Goldilocks2::one();
    for (size_t i = 1; i < base.size(); ++i) {
        base[i] = base[i - 1] * lambda;
    }
    #pragma omp parallel
    {
        oracle_sum local_eq_sum_oracle;
        Goldilocks2::Element local_claim = Goldilocks2::zero();

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < open_eqs.size(); ++i) {
            int pre_len = prefix[open_ind[i]].size();
            size_t offset = 0;
            for (int j = 0; j < pre_len; ++j) {
                offset = (offset << 1) | (open_eqs[i][j] == Goldilocks2::one());
            }
            offset <<= (num_vars - pre_len);

            auto eqt = eq_table(num_vars - pre_len, &open_eqs[i][pre_len]);
            for (size_t k = 0; k < (1ull << (num_vars - pre_len)); ++k) {
                eqt[k] *= base[i];
            }

            auto eq_oracle = std::make_shared<MLE_Eq_Oracle>(open_eqs[i]);
            local_eq_sum_oracle.add(eq_oracle, base[i]);
            local_claim += open_val[i] * base[i];

            // Chunked update to shared table
            size_t chunk_size = 64; // tune: larger chunk reduces locking overhead
            for (size_t k = 0; k < (1ull << (num_vars - pre_len)); k += chunk_size) {
                #pragma omp critical(table_update)
                {
                    for (size_t t = 0; t < chunk_size && k + t < (1ull << (num_vars - pre_len)); ++t) {
                        table[k + t + offset] += eqt[k + t];
                    }
                }
            }
        }

        // Merge partial results
        #pragma omp critical(eq_sum_merge)
        {
            eq_sum_oracle.merge(local_eq_sum_oracle);
            claim += local_claim;
        }
    }

    // #pragma omp parallel for
    // for (size_t i = 0; i != open_eqs.size(); ++i) {
    //     int pre_len = prefix[open_ind[i]].size();
    //     size_t offset = 0;
    //     for (int j = 0; j != pre_len; ++j) {
    //         offset = open_eqs[i][j] == Goldilocks2::one() ? ((offset << 1) | 1) : (offset << 1);
    //     }
    //     offset <<= (num_vars - pre_len);
    //     auto eqt = eq_table(num_vars - pre_len, &open_eqs[i][pre_len]);
    //     for (size_t k = 0; k != (1ull << (num_vars - pre_len)); ++k) {
    //         eqt[k] *= base[i];
    //     }
    //     auto eq_oracle = std::make_shared<MLE_Eq_Oracle>(open_eqs[i]);  
    //     #pragma omp critical
    //     {
    //         for (size_t k = 0; k != (1ull << (num_vars - pre_len)); ++k) {
    //             table[k + offset] += eqt[k];
    //         }
    //         eq_sum_oracle.add(eq_oracle, base[i]);
    //         claim += open_val[i] * base[i];
    //     }
    // }
    pause_timer("lazy pcs open gen");
    set_timer("lazy pcs create p2Prover");
    std::unique_ptr<MLE> mle_sum = std::make_unique<MLE>(std::move(table));
    std::unique_ptr<MLE> ptr_uni_mle = std::make_unique<MLE>(std::move(uni_mle));
    p2Prover prover(std::move(mle_sum), std::move(ptr_uni_mle));
    pause_timer("lazy pcs create p2Prover");
    set_timer("lazy pcs p2exe");
    if (!p2Verifier::execute_sumcheck(prover, {&eq_sum_oracle, pcs.get()}, claim, sec_param)) {
        std::cerr << __LINE__ << ": lazy_pcs_pool::prove_open: sumcheck failed" << std::endl;
        return false;
    }
    pause_timer("lazy pcs p2exe");
    return true;
}

lazy_pcs commit_lazy_pcs(const MLE& mle, std::shared_ptr<lazy_pcs_pool> pool) {
    lazy_pcs res(mle, pool);
    res.index = pool->add_mle(res.mle);
    return res;
}

lazy_pcs commit_lazy_pcs(MLE&& mle, std::shared_ptr<lazy_pcs_pool> pool) {
    lazy_pcs res(std::move(mle), pool);
    res.index = pool->add_mle(res.mle);
    return res;
}

std::map<int, int> lazy_pcs_open_cnt;

Goldilocks2::Element lazy_pcs::open(const std::vector<Goldilocks2::Element>& z, const size_t& sec_param) const {
    auto val = mle->open(z, sec_param);
    pool->record_open(index, z, val, sec_param);
    return val;
}
