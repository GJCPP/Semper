#pragma once

#include "goldilocks_quadratic_ext.h"
#include <openssl/sha.h>
#include <openssl/evp.h>
#include <cstdint>
#include <memory>
#include <vector>
#include <array>
#include <utility>

// std::array<uint8_t, 16> to_bytes(const Goldilocks2::Element& e);
// MTtree merkle_hash(const std::vector<std::vector<Goldilocks2::Element>> &data, std::array<uint8_t, SHA256_DIGEST_LENGTH> &hash);

namespace MerkleDef{
    typedef std::array<uint8_t, SHA256_DIGEST_LENGTH> Digest;
    typedef std::vector<Digest> MTtree;
    typedef std::vector<Digest> MTPath;
}

class MerkleTree_base{
public:

    // type for a column
    typedef std::vector<Goldilocks::Element> col_t;
    typedef struct{
        MerkleDef::MTPath path;
        col_t column;
        const Goldilocks::Element *col;
        size_t sz_col;
        // this is index in the tree, not matrix!!
        size_t index;
    } MTPayload;

private:
    MerkleDef::MTtree T;
    MerkleDef::MTtree repeat_T;
    MerkleDef::Digest root{};
    size_t leaf_offset = 0;
    size_t stored_leaf_offset = 0;
    size_t repeat_leaf_offset = 0;
    std::vector<Goldilocks::Element> cols;
    std::unique_ptr<Goldilocks::Element[]> row_major_cols;
    size_t num_rows = 0, num_cols = 0, stored_num_cols = 0;
public:
    MerkleTree_base(){};
    MerkleTree_base(const Goldilocks::Element *data, size_t num_rows, size_t num_cols);
    MerkleTree_base(const Goldilocks::Element *data, size_t num_rows, size_t stored_num_cols, size_t logical_num_cols);
    MerkleTree_base(std::unique_ptr<Goldilocks::Element[]> data, size_t num_rows, size_t stored_num_cols, size_t logical_num_cols);
    // MerkleTree_base(const Goldilocks::Element* data, size_t num_rows, size_t num_cols);
    MTPayload MerkleOpen(const size_t& idx) const;
    MerkleDef::Digest MerkleCommit() const {return root;}
    static bool MerkleVerify(const MerkleDef::Digest& root, const MTPayload& payload);
};

class MerkleTree_ext{
public:
    // type for a column
    typedef std::vector<Goldilocks2::Element> col_t;
    typedef struct {
        MerkleDef::MTPath path;
        col_t column;
        // this is index in the tree, not matrix!!
        size_t index;
    } MTPayload; // Merkle opening statement/proof.

private:
    MerkleDef::MTtree T;
    MerkleDef::MTtree repeat_T;
    MerkleDef::Digest root{};
    size_t leaf_offset = 0;
    size_t stored_leaf_offset = 0;
    size_t repeat_leaf_offset = 0;
    std::vector<col_t> cols;
    size_t num_cols = 0, stored_num_cols = 0;
public:
    MerkleTree_ext(){};
    MerkleTree_ext(const std::vector<std::vector<Goldilocks2::Element>> &data);
    MerkleTree_ext(const std::vector<std::vector<Goldilocks2::Element>> &data, size_t stored_num_cols, size_t logical_num_cols);
    MTPayload MerkleOpen(const size_t& idx) const;
    MerkleDef::Digest MerkleCommit() const {return root;}
    static bool MerkleVerify(const MerkleDef::Digest& root, const MTPayload& payload);
};
