#include "merkle.h"
#include "goldilocks_base_field.hpp"
#include "util.h"
#include "timer.h"

#include <array>
#include <stdexcept>

std::array<uint8_t, 16> to_bytes(const Goldilocks2::Element& e) {
    std::array<uint8_t, 16> bytes;
    uint64_t real = Goldilocks::toU64(e[0]);
    uint64_t imag = Goldilocks::toU64(e[1]);

    for (int i = 0; i < 8; i++) {
        bytes[7 - i] = (real >> (8 * i)) & 0xFF;
        bytes[15 - i] = (imag >> (8 * i)) & 0xFF;
    }

    // std::cout <<"bytes of ";
    // std::cout << Goldilocks2::toString(e) <<':';
    // print_bytes(bytes);
    return bytes;
}

std::array<uint8_t, 8> to_bytes(const Goldilocks::Element& e) {
    std::array<uint8_t, 8> bytes;
    uint64_t literal = Goldilocks::toU64(e);
    for (int i = 0; i < 8; i++) {
        bytes[7 - i] = (literal >> (8 * i)) & 0xFF;
    }

    // std::cout <<"bytes of ";
    // std::cout << Goldilocks2::toString(e) <<':';
    // print_bytes(bytes);
    return bytes;
}

void write_be64(uint8_t* dst, uint64_t value) {
    for (int i = 0; i < 8; i++) {
        dst[7 - i] = (value >> (8 * i)) & 0xFF;
    }
}

MerkleDef::Digest hash_bytes(const uint8_t* bytes, size_t sz) {
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), NULL);
    std::array<uint8_t, SHA256_DIGEST_LENGTH> hash;
    EVP_DigestUpdate(ctx, bytes, sz);
    unsigned int tmp;
    EVP_DigestFinal_ex(ctx, hash.data(), &tmp);
    EVP_MD_CTX_free(ctx);
    return hash;
}

MerkleDef::Digest copy_and_hash_column_from_row_major(const Goldilocks::Element* data,
                                                      size_t num_rows,
                                                      size_t row_width,
                                                      size_t col_idx,
                                                      Goldilocks::Element* dst,
                                                      std::vector<uint8_t>& bytes) {
    bytes.resize(num_rows * 8);
    for (size_t i = 0; i < num_rows; ++i) {
        Goldilocks::Element e = data[i * row_width + col_idx];
        dst[i] = e;
        write_be64(bytes.data() + i * 8, Goldilocks::toU64(e));
    }
    return hash_bytes(bytes.data(), bytes.size());
}

MerkleDef::Digest hash_column_from_row_major(const Goldilocks::Element* data,
                                             size_t num_rows,
                                             size_t row_width,
                                             size_t col_idx,
                                             std::vector<uint8_t>& bytes) {
    bytes.resize(num_rows * 8);
    for (size_t i = 0; i < num_rows; ++i) {
        write_be64(bytes.data() + i * 8, Goldilocks::toU64(data[i * row_width + col_idx]));
    }
    return hash_bytes(bytes.data(), bytes.size());
}

// hash one column
MerkleDef::Digest hash_column(const Goldilocks2::Element col[], size_t sz) {
    // SHA256_CTX ctx;
    // SHA256_Init(&ctx);
    // std::array<uint8_t, SHA256_DIGEST_LENGTH> hash;
    // // 16 for 2 * 64 / 8
    // for(auto e: col) SHA256_Update(&ctx, to_bytes(e).data(), 16);
    // SHA256_Final(hash.data(), &ctx);
    // return hash;

    std::vector<uint8_t> bytes(sz * 16);
    for (size_t i = 0; i < sz; ++i) {
        uint64_t real = Goldilocks::toU64(col[i][0]);
        uint64_t imag = Goldilocks::toU64(col[i][1]);
        write_be64(bytes.data() + i * 16, real);
        write_be64(bytes.data() + i * 16 + 8, imag);
    }
    return hash_bytes(bytes.data(), bytes.size());
}

// hash the column
MerkleDef::Digest hash_column(const Goldilocks::Element col[], size_t sz) {
    // SHA256_CTX ctx;
    // SHA256_Init(&ctx);
    // std::array<uint8_t, SHA256_DIGEST_LENGTH> hash;
    // // 8 for 64 / 8
    // for(auto e: col) SHA256_Update(&ctx, to_bytes(e).data(), 8);
    // SHA256_Final(hash.data(), &ctx);

    std::vector<uint8_t> bytes(sz * 8);
    for (size_t i = 0; i < sz; ++i) {
        write_be64(bytes.data() + i * 8, Goldilocks::toU64(col[i]));
    }
    return hash_bytes(bytes.data(), bytes.size());
}

// hash one column
MerkleDef::Digest hash_column(const MerkleTree_ext::col_t& col) {
    return hash_column(col.data(), col.size());
}

// hash the column
MerkleDef::Digest hash_column(const MerkleTree_base::col_t& col) {
    return hash_column(col.data(), col.size());
}

// hash two child nodes
MerkleDef::Digest hash_node(const std::array<uint8_t, SHA256_DIGEST_LENGTH>& l, const std::array<uint8_t, SHA256_DIGEST_LENGTH>& r) {
    // SHA256_CTX ctx;
    // SHA256_Init(&ctx);
    // std::array<uint8_t, SHA256_DIGEST_LENGTH> hash;
    // SHA256_Update(&ctx, l.data(), SHA256_DIGEST_LENGTH);
    // SHA256_Update(&ctx, r.data(), SHA256_DIGEST_LENGTH);
    // SHA256_Final(hash.data(), &ctx);

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), NULL);
    std::array<uint8_t, SHA256_DIGEST_LENGTH> hash;
    EVP_DigestUpdate(ctx, l.data(), SHA256_DIGEST_LENGTH);
    EVP_DigestUpdate(ctx, r.data(), SHA256_DIGEST_LENGTH);
    unsigned int tmp;
    EVP_DigestFinal_ex(ctx, hash.data(), &tmp);
    EVP_MD_CTX_free(ctx);
    return hash;
}

// construct the merkle hash tree from a matrix
MerkleTree_base::MerkleTree_base(const Goldilocks::Element *data, size_t _num_rows, size_t _num_cols) {
    // set_timer("merkle_commit_col");
    // cols.resize(num_cols);
    // #pragma omp parallel for
    // for (size_t i = 0; i < num_cols; ++i) {
    //     cols[i].resize(data.size());
    //     for (size_t j = 0; j < data.size(); ++j) {
    //         cols[i][j] = data[j][i];
    //     }
    // }
    num_rows = _num_rows;
    num_cols = _num_cols;
    stored_num_cols = _num_cols;
    cols.resize(num_cols * num_rows);

    // for clearer binary tree structure, index starts from 1 (T[0] is not used)
    size_t loopN = find_ceiling_log2(num_cols);
    leaf_offset = 1ul << loopN;
    stored_leaf_offset = leaf_offset;
    repeat_leaf_offset = 1;
    T = MerkleDef::MTtree(leaf_offset << 1);
    // MTtree mt_t(num_cols << 1);

    // set_timer("merkle_commit_hash");
    #pragma omp parallel
    {
        std::vector<uint8_t> bytes;
        #pragma omp for schedule(static)
        for (size_t j = 0; j < num_cols; ++j) {
            T[leaf_offset + j] = copy_and_hash_column_from_row_major(
                data, num_rows, num_cols, j, &cols[j * num_rows], bytes
            );
        }
    }
    #pragma omp parallel for
    for (size_t j = num_cols; j < leaf_offset; ++j) {
        T[leaf_offset + j] = T[leaf_offset + (j % num_cols)];
    }
    // pause_timer("merkle_commit_hash");
    // set_timer("merkle_commit_T");
    for (size_t i = loopN; i > 0; --i) {
        size_t offset = 1ul << (i - 1);
        #pragma omp parallel for
        for (size_t j = 0; j < offset; ++j) {
            size_t idx = offset + j;
            T[idx] = hash_node(T[2 * idx], T[2 * idx + 1]);
        }
    }
    root = T[1];
    // pause_timer("merkle_commit_T");
}

// construct a logical Merkle tree whose leaves repeat a smaller stored matrix
MerkleTree_base::MerkleTree_base(const Goldilocks::Element *data,
                                 size_t _num_rows,
                                 size_t _stored_num_cols,
                                 size_t _logical_num_cols) {
    if (_stored_num_cols == 0 || _logical_num_cols == 0 || _logical_num_cols % _stored_num_cols != 0) {
        throw std::invalid_argument("MerkleTree_base: logical columns must be a non-zero multiple of stored columns");
    }
    num_rows = _num_rows;
    num_cols = _logical_num_cols;
    stored_num_cols = _stored_num_cols;
    const size_t repeat_num_cols = num_cols / stored_num_cols;
    if (!is_power_of_2(stored_num_cols) || !is_power_of_2(repeat_num_cols)) {
        throw std::invalid_argument("MerkleTree_base: repeated logical tree requires power-of-two stored and repeat columns");
    }
    cols.resize(stored_num_cols * num_rows);

    size_t loopN = find_ceiling_log2(stored_num_cols);
    stored_leaf_offset = 1ul << loopN;
    repeat_leaf_offset = repeat_num_cols;
    leaf_offset = stored_leaf_offset * repeat_leaf_offset;
    T = MerkleDef::MTtree(stored_leaf_offset << 1);

    #pragma omp parallel
    {
        std::vector<uint8_t> bytes;
        #pragma omp for schedule(static)
        for (size_t j = 0; j < stored_num_cols; ++j) {
            T[stored_leaf_offset + j] = copy_and_hash_column_from_row_major(
                data, num_rows, stored_num_cols, j, &cols[j * num_rows], bytes
            );
        }
    }
    for (size_t i = loopN; i > 0; --i) {
        size_t offset = 1ul << (i - 1);
        #pragma omp parallel for
        for (size_t j = 0; j < offset; ++j) {
            size_t idx = offset + j;
            T[idx] = hash_node(T[2 * idx], T[2 * idx + 1]);
        }
    }

    size_t repeatLoopN = find_ceiling_log2(repeat_num_cols);
    repeat_T = MerkleDef::MTtree(repeat_leaf_offset << 1);
    #pragma omp parallel for
    for (size_t j = 0; j < repeat_leaf_offset; ++j) {
        repeat_T[repeat_leaf_offset + j] = T[1];
    }
    for (size_t i = repeatLoopN; i > 0; --i) {
        size_t offset = 1ul << (i - 1);
        #pragma omp parallel for
        for (size_t j = 0; j < offset; ++j) {
            size_t idx = offset + j;
            repeat_T[idx] = hash_node(repeat_T[2 * idx], repeat_T[2 * idx + 1]);
        }
    }
    root = repeat_T[1];
}

// construct a logical Merkle tree over row-major data and keep that storage.
MerkleTree_base::MerkleTree_base(std::unique_ptr<Goldilocks::Element[]> data,
                                 size_t _num_rows,
                                 size_t _stored_num_cols,
                                 size_t _logical_num_cols) {
    if (_stored_num_cols == 0 || _logical_num_cols == 0 || _logical_num_cols % _stored_num_cols != 0) {
        throw std::invalid_argument("MerkleTree_base: logical columns must be a non-zero multiple of stored columns");
    }
    num_rows = _num_rows;
    num_cols = _logical_num_cols;
    stored_num_cols = _stored_num_cols;
    row_major_cols = std::move(data);
    const size_t repeat_num_cols = num_cols / stored_num_cols;
    if (!is_power_of_2(stored_num_cols) || !is_power_of_2(repeat_num_cols)) {
        throw std::invalid_argument("MerkleTree_base: repeated logical tree requires power-of-two stored and repeat columns");
    }

    size_t loopN = find_ceiling_log2(stored_num_cols);
    stored_leaf_offset = 1ul << loopN;
    repeat_leaf_offset = repeat_num_cols;
    leaf_offset = stored_leaf_offset * repeat_leaf_offset;
    T = MerkleDef::MTtree(stored_leaf_offset << 1);

    #pragma omp parallel
    {
        std::vector<uint8_t> bytes;
        #pragma omp for schedule(static)
        for (size_t j = 0; j < stored_num_cols; ++j) {
            T[stored_leaf_offset + j] = hash_column_from_row_major(
                row_major_cols.get(), num_rows, stored_num_cols, j, bytes
            );
        }
    }
    for (size_t i = loopN; i > 0; --i) {
        size_t offset = 1ul << (i - 1);
        #pragma omp parallel for
        for (size_t j = 0; j < offset; ++j) {
            size_t idx = offset + j;
            T[idx] = hash_node(T[2 * idx], T[2 * idx + 1]);
        }
    }

    size_t repeatLoopN = find_ceiling_log2(repeat_num_cols);
    repeat_T = MerkleDef::MTtree(repeat_leaf_offset << 1);
    #pragma omp parallel for
    for (size_t j = 0; j < repeat_leaf_offset; ++j) {
        repeat_T[repeat_leaf_offset + j] = T[1];
    }
    for (size_t i = repeatLoopN; i > 0; --i) {
        size_t offset = 1ul << (i - 1);
        #pragma omp parallel for
        for (size_t j = 0; j < offset; ++j) {
            size_t idx = offset + j;
            repeat_T[idx] = hash_node(repeat_T[2 * idx], repeat_T[2 * idx + 1]);
        }
    }
    root = repeat_T[1];
}

MerkleTree_base::MTPayload MerkleTree_base::MerkleOpen(const size_t& idx) const {
    if (idx >= num_cols) {
        throw std::out_of_range("MerkleTree_base::MerkleOpen: index exceeds logical column count");
    }
    size_t stored_idx = idx % stored_num_cols;
    MerkleDef::MTPath path;
    if (repeat_T.empty()) {
        size_t index = leaf_offset + idx;
        while (index != 1) {
            size_t sibling = (index ^ 1);
            path.push_back(T[sibling]);
            index >>= 1;
        }
    } else {
        size_t index = stored_leaf_offset + stored_idx;
        while (index != 1) {
            size_t sibling = (index ^ 1);
            path.push_back(T[sibling]);
            index >>= 1;
        }
        size_t repeat_idx = idx / stored_num_cols;
        index = repeat_leaf_offset + repeat_idx;
        while (index != 1) {
            size_t sibling = (index ^ 1);
            path.push_back(repeat_T[sibling]);
            index >>= 1;
        }
    }
    MTPayload payload;
    payload.path = std::move(path);
    payload.sz_col = num_rows;
    payload.index = idx + leaf_offset;
    if (row_major_cols) {
        payload.column.resize(num_rows);
        for (size_t i = 0; i < num_rows; ++i) {
            payload.column[i] = row_major_cols[i * stored_num_cols + stored_idx];
        }
        payload.col = payload.column.data();
    } else {
        payload.col = &cols[stored_idx * num_rows];
    }
    return payload;
}

bool MerkleTree_base::MerkleVerify(const MerkleDef::Digest& root, const MTPayload& payload) {
    MerkleDef::Digest hash = hash_column(payload.col, payload.sz_col);
    size_t index = payload.index;

    for (const MerkleDef::Digest& sibling : payload.path) {
        if ((index & 1) == 0) {
            // this is left child
            hash = hash_node(hash, sibling);
        }
        else {
            // this is right child
            hash = hash_node(sibling, hash);
        }
        index >>= 1;
    }
    return hash == root;
}


// construct the merkle hash tree from a matrix
MerkleTree_ext::MerkleTree_ext(const std::vector<col_t>& data) {
    num_cols = data[0].size();
    stored_num_cols = num_cols;
    size_t num_rows = data.size();
    cols.resize(num_cols, std::vector<Goldilocks2::Element>(num_rows));
    #pragma omp parallel for
    for (size_t i = 0; i < num_cols; ++i) {
        col_t& col = cols[i];
        for (size_t j = 0; j < data.size(); ++j) {
            col[j] = data[j][i];
        }
    } // cols[i][j] <- data[j][i]

    // for clearer binary tree structure, index starts from 1 (T[0] is not used)
    size_t loopN = find_ceiling_log2(num_cols);
    leaf_offset = 1ul << loopN;
    stored_leaf_offset = leaf_offset;
    repeat_leaf_offset = 1;
    T = MerkleDef::MTtree(leaf_offset << 1);
    // MTtree mt_t(num_cols << 1);
    #pragma omp parallel for
    for (size_t j = 0; j < num_cols; ++j) {
        T[leaf_offset + j] = hash_column(cols[j]);
    }
    #pragma omp parallel for
    for (size_t j = num_cols; j < leaf_offset; ++j) {
        T[leaf_offset + j] = T[leaf_offset + (j % num_cols)];
    }
    for (size_t i = loopN; i > 0; --i) {
        size_t offset = 1ul << (i - 1);
        #pragma omp parallel for
        for (size_t j = 0; j < offset; ++j) {
            size_t idx = offset + j;
            T[idx] = hash_node(T[2 * idx], T[2 * idx + 1]);
        }
    }
    root = T[1];
}

// construct a logical Merkle tree whose leaves repeat a smaller stored matrix
MerkleTree_ext::MerkleTree_ext(const std::vector<col_t>& data,
                               size_t _stored_num_cols,
                               size_t _logical_num_cols) {
    if (_stored_num_cols == 0 || _logical_num_cols == 0 || _logical_num_cols % _stored_num_cols != 0) {
        throw std::invalid_argument("MerkleTree_ext: logical columns must be a non-zero multiple of stored columns");
    }
    if (data.empty() || data[0].size() != _stored_num_cols) {
        throw std::invalid_argument("MerkleTree_ext: row width does not match stored columns");
    }
    num_cols = _logical_num_cols;
    stored_num_cols = _stored_num_cols;
    const size_t repeat_num_cols = num_cols / stored_num_cols;
    if (!is_power_of_2(stored_num_cols) || !is_power_of_2(repeat_num_cols)) {
        throw std::invalid_argument("MerkleTree_ext: repeated logical tree requires power-of-two stored and repeat columns");
    }
    size_t num_rows = data.size();
    cols.resize(stored_num_cols, std::vector<Goldilocks2::Element>(num_rows));
    #pragma omp parallel for
    for (size_t i = 0; i < stored_num_cols; ++i) {
        col_t& col = cols[i];
        for (size_t j = 0; j < data.size(); ++j) {
            col[j] = data[j][i];
        }
    }

    size_t loopN = find_ceiling_log2(stored_num_cols);
    stored_leaf_offset = 1ul << loopN;
    repeat_leaf_offset = repeat_num_cols;
    leaf_offset = stored_leaf_offset * repeat_leaf_offset;
    T = MerkleDef::MTtree(stored_leaf_offset << 1);
    #pragma omp parallel for
    for (size_t j = 0; j < stored_num_cols; ++j) {
        T[stored_leaf_offset + j] = hash_column(cols[j]);
    }
    for (size_t i = loopN; i > 0; --i) {
        size_t offset = 1ul << (i - 1);
        #pragma omp parallel for
        for (size_t j = 0; j < offset; ++j) {
            size_t idx = offset + j;
            T[idx] = hash_node(T[2 * idx], T[2 * idx + 1]);
        }
    }

    size_t repeatLoopN = find_ceiling_log2(repeat_num_cols);
    repeat_T = MerkleDef::MTtree(repeat_leaf_offset << 1);
    #pragma omp parallel for
    for (size_t j = 0; j < repeat_leaf_offset; ++j) {
        repeat_T[repeat_leaf_offset + j] = T[1];
    }
    for (size_t i = repeatLoopN; i > 0; --i) {
        size_t offset = 1ul << (i - 1);
        #pragma omp parallel for
        for (size_t j = 0; j < offset; ++j) {
            size_t idx = offset + j;
            repeat_T[idx] = hash_node(repeat_T[2 * idx], repeat_T[2 * idx + 1]);
        }
    }
    root = repeat_T[1];
}

MerkleTree_ext::MTPayload MerkleTree_ext::MerkleOpen(const size_t& idx) const {
    if (idx >= num_cols) {
        throw std::out_of_range("MerkleTree_ext::MerkleOpen: index exceeds logical column count");
    }
    size_t stored_idx = idx % stored_num_cols;
    MerkleDef::MTPath path;
    if (repeat_T.empty()) {
        size_t index = leaf_offset + idx;
        while (index != 1) {
            size_t sibling = (index ^ 1);
            path.push_back(T[sibling]);
            index >>= 1;
        }
    } else {
        size_t index = stored_leaf_offset + stored_idx;
        while (index != 1) {
            size_t sibling = (index ^ 1);
            path.push_back(T[sibling]);
            index >>= 1;
        }
        size_t repeat_idx = idx / stored_num_cols;
        index = repeat_leaf_offset + repeat_idx;
        while (index != 1) {
            size_t sibling = (index ^ 1);
            path.push_back(repeat_T[sibling]);
            index >>= 1;
        }
    }
    return MTPayload{ path, cols[stored_idx], idx + leaf_offset };
}

bool MerkleTree_ext::MerkleVerify(const MerkleDef::Digest& root, const MTPayload& payload) {
    col_t col = payload.column;
    MerkleDef::Digest hash = hash_column(col);
    size_t index = payload.index;

    for (const MerkleDef::Digest& sibling : payload.path) {
        if ((index & 1) == 0) {
            // this is left child
            hash = hash_node(hash, sibling);
        }
        else {
            // this is right child
            hash = hash_node(sibling, hash);
        }
        index >>= 1;
    }
    return hash == root;
}
