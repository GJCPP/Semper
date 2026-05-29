# Implementing Orion PCS in Semper

## Goal

Implement a Semper-native Orion polynomial commitment backend that aligns with the Orion path used in `kaizen` (`PC_scheme == 1`), while preserving Semper's existing `oracle` abstraction and proof flow.

Do not directly import `kaizen`'s PCS code. Its implementation is coupled to global state, a different field type, recursive proof-of-training logic, and hash aggregation machinery. Instead, reimplement the same core PCS idea inside Semper:

- expander-code encoding
- Merkle commitment to encoded rows/columns
- random query checks for linear-code consistency
- multilinear opening at arbitrary verifier challenge points

## Current Semper PCS Boundary

The main PCS integration point is:

- `include/oracle.h`: every committed object must implement `oracle::open(z, sec_param)`.
- `include/ligero.h` and `src/ligero.cpp`: current Ligero PCS backend.
- `include/lazy_pcs.h` and `src/lazy_pcs.cpp`: batches many MLEs into one large MLE, commits once, records openings, and proves all openings with one final sumcheck.

The first Orion implementation should target `lazy_pcs_pool` first. This keeps CNN, logup, map, sumcheck, and layer-proof logic unchanged.

## Key Design Decisions

1. Keep the existing `oracle` interface.

   Add `orionpcs_base` and `orionpcs_ext`, analogous to `ligeropcs_base` and `ligeropcs_ext`.

2. Keep Semper's field types.

   Use `Goldilocks::Element` for base committed rows and `Goldilocks2::Element` for extension-field rows/opening checks. Do not use `kaizen::fieldElement`.

3. Use deterministic expander graphs.

   `kaizen` has `expander_init(n)` but the current checked-out code does not show a clear call site. Semper's Orion backend should explicitly initialize expander graphs per input row length with a deterministic seed or deterministic PRG so commitments are reproducible.

4. Avoid global scratch state.

   `kaizen` uses global `scratch[2][100]` and `__encode_initialized`. Reimplement the encoder with local/RAII buffers or a reusable encoder object.

5. Preserve Semper's MLE opening shape.

   Semper opens a committed MLE at arbitrary point `z` by viewing the evaluation table as a matrix. The prover sends a row linear combination `v_prime`; the verifier checks that `Encode(v_prime)` matches the same linear combination of committed encoded rows at random queried columns, then returns `dot(v_prime, L)`.

6. Start with Kaizen-compatible query count.

   Kaizen uses `100` queries for Orion. Start with `100`, then later derive it from `sec_param` and the code distance.

## Implementation Steps

### 1. Add Orion files

Create:

- `include/orion.h`
- `src/orion.cpp`

Because `CMakeLists.txt` already uses `file(GLOB SOURCES src/*.cpp)`, adding `src/orion.cpp` should be enough for the `bench` target.

### 2. Define Orion types

Add classes similar to Ligero:

- `orionProver_base`
- `orionProver_ext`
- `orionpcs_base : public oracle`
- `orionpcs_ext : public oracle`
- `orionVerifier`

Minimum API:

```cpp
orionpcs_base orion_commit_base(const MultilinearPolynomial& w, uint64_t rho_inv_or_unused, int loga = -1);
orionpcs_ext orion_commit_ext(const MultilinearPolynomial& w, uint64_t rho_inv_or_unused, int loga = -1);
```

`rho_inv` can initially be ignored or mapped to Orion code-rate parameters, but keep it in the signature so call sites can switch from Ligero with minimal churn.

### 3. Implement expander graph generation

Reimplement the graph structure from `kaizen/src/expanders.h`:

- `distance_threshold = 13`
- `alpha = 0.238`
- `r = 1.72`
- `cn = 10`
- `dn = 20`

Required behavior:

- Given row length `n`, initialize recursive expander graphs.
- Graph generation must be deterministic for the same `n`.
- Cache graphs by `n` to avoid rebuilding on every commit.

Do not use `rand()` without an explicit deterministic seed.

### 4. Implement Orion encoder

Reimplement the recursive expander encoder from `kaizen/src/linear_code_encode.h`.

Needed variants:

- encode base field row: `std::vector<Goldilocks::Element> -> std::vector<Goldilocks::Element>`
- encode extension row: `std::vector<Goldilocks2::Element> -> std::vector<Goldilocks2::Element>`

The encoder must be linear over the base field. For extension elements, use the same graph weights and apply them to `Goldilocks2::Element`.

For initial compatibility with Kaizen's `PC_scheme == 1`, use the same high-level shape:

- allocate an output buffer of size `2 * n`
- run expander encode into that buffer
- concatenate two such encoded buffers, giving codeword length `4 * n`

This mirrors the `poly_commit` branch that inserts the encoded buffer twice.

### 5. Implement commit

For a multilinear polynomial with `2^num_vars` evaluations:

1. Choose matrix shape `(a, b)` using Semper's existing `get_ab` logic or an Orion-specific policy.
2. Store original rows so `open(z)` can compute `v_prime`.
3. Encode each row with Orion.
4. Build a Merkle tree over encoded columns/rows using Semper's existing Merkle code where possible.
5. Return an `orionpcs_base/ext` containing:
   - Merkle root
   - prover state
   - `num_rows`
   - `num_cols`
   - `num_vars`
   - codeword length

For `orionpcs_base`, the committed data can be base-field evaluations. During opening, verifier checks use extension-field linear combinations, so the encoder/check path must support extension-field recomputation of `Encode(v_prime)`.

### 6. Implement open

Follow the current Ligero opening structure:

1. Split challenge `z` into high-row and low-column parts.
2. Compute equality tables `R` and `L`.
3. Prover computes `v_prime = R^T * original_matrix`.
4. Verifier computes `encoded_v_prime = OrionEncode(v_prime)`.
5. Verifier samples query columns.
6. Prover opens those columns from the Merkle tree.
7. Verifier checks:

   ```text
   encoded_v_prime[col] == sum_i R[i] * opened_encoded_matrix[i][col]
   ```

8. Return:

   ```text
   dot(v_prime, L)
   ```

This gives the same external semantics as `oracle::open(z, sec_param)`.

### 7. Add proof-size accounting and timers

Match existing conventions:

- `startCounter("orion_open")` or equivalent.
- `add_proof_size(...)` for Merkle roots, opened columns, and sent `v_prime`.
- timers for:
  - `orion commit`
  - `orion encode`
  - `orion merkle`
  - `orion open`
  - `orion check`

### 8. Add tests before integration

Start with small deterministic tests in `src/test.cpp`:

1. Encoder linearity:

   ```text
   Encode(a*x + b*y) == a*Encode(x) + b*Encode(y)
   ```

2. `orionpcs_base::open`:

   - random base-field MLE
   - random challenge
   - compare `pcs.open(z, 32)` with `mle.open(z, 32)`

3. `orionpcs_ext::open`:

   - random extension-field MLE
   - random challenge
   - compare with direct MLE opening

4. Lazy PCS smoke test:

   - commit several MLEs through `lazy_pcs_pool`
   - record openings
   - run final `prove_open`

### 9. Add backend switch

After standalone tests pass, add a backend switch rather than replacing Ligero everywhere.

Possible minimal option:

```cpp
enum class pcs_backend { ligero, orion };
```

Then let `lazy_pcs_pool::create(...)` or `lazy_pcs_pool` constructor accept a backend. Initially default to Ligero.

First integration target:

- change only `lazy_pcs_pool::commit(...)` to optionally call `orion_commit_base/ext`.

Do not initially replace all direct `ligero_commit_base/ext` calls in `prod_check`, `perm_check`, `sign_check`, `e_pow_check`, `logup`, or tests.

### 10. Benchmark and compare

Use small benchmarks first:

- `bench_commit()` with small `i`
- `test_lazy_pcs`
- one tiny model trace if available

Compare:

- commit time
- open time
- proof size
- memory usage

Only after the lazy PCS backend is stable should we consider replacing direct Ligero call sites.

## Main Risks

1. Expander initialization mismatch with Kaizen.

   Kaizen's current code exposes `expander_init` but does not show a clear call site. Semper must explicitly initialize graphs.

2. Soundness parameter mapping.

   Fixed `100` Orion queries is good for prototype parity with Kaizen, but final experiments should derive queries from `sec_param` and distance assumptions.

3. Base/extension field handling.

   Semper's verifier challenges live in `Goldilocks2`. Even base-field commitments need extension-field linear-combination checks.

4. Merkle layout.

   Existing Semper Merkle code is column-oriented for Ligero. Orion codeword length and matrix layout must be checked carefully so opened columns correspond to verifier queries.

5. Global replacement is broader than lazy PCS.

   Many files directly call `ligero_commit_base/ext`. The first milestone should not try to replace all of them.

## Milestones

1. Standalone Orion encoder passes linearity tests.
2. `orionpcs_base/ext::open` matches direct `MLE::open`.
3. `lazy_pcs_pool` can use Orion for final opening.
4. Existing lazy PCS tests pass with Orion backend.
5. Small benchmark compares Ligero vs Orion.
6. Decide whether to replace direct Ligero call sites.

## Current Implementation Status

Implemented:

- `include/orion.h`
- `src/orion.cpp`
- `orionpcs_base`
- `orionpcs_ext`
- deterministic expander graph generation
- Orion-style recursive expander encoder
- Merkle commitment over encoded rows/columns
- `oracle::open(z, sec_param)` support for base and extension commitments
- Semper-native code-switching-lite opening: the prover no longer sends `v_prime` directly. It commits to `v_prime`, batches the evaluation claim and sampled encoder-consistency claim into one product-2 sumcheck, and opens the derived `v_prime` commitment once.
- `bench pcs_compare <logn> <threads>` benchmark path

Also updated Ligero proof-size accounting so extension commitments count the Merkle root and openings count Merkle authentication paths. This makes the comparison with Orion use the same accounting model. Base-field column openings are counted as base-field elements, not extension-field elements.

Important audit note: this implementation is still not the full Orion Protocol 4/CP-SNARK construction from the paper. The first prototype was only an Orion/expander encoder plugged into Semper's existing Ligero-style MLE opening shape, which meant the prover still sent `v_prime` directly:

```text
|v_prime| + query_count * (opened_column_size + Merkle_path_size)
```

That was the reason the initial Orion proof was larger than Ligero. The current implementation fixes that specific issue by replacing the direct `v_prime` message with a derived commitment plus one batched sumcheck. This is a Semper-native code-switching step, using Ligero as the secondary PCS for the derived vector rather than the paper's full CP-SNARK composition.

Not implemented yet:

- `lazy_pcs_pool` backend switch
- full replacement of direct `ligero_commit_base/ext` call sites
- full Protocol 4/CP-SNARK-style code-switching proof composition
- security-parameter-derived Orion query count
- Fiat-Shamir binding of Orion query positions to transcript/root

Current Orion query count is fixed at 100, matching Kaizen's Orion path.

## Benchmark Results

Command format:

```sh
./build/bench pcs_compare <logn> 8
```

All runs used the same direct-MLE correctness check. `open_ok=true` means the PCS opening matched `MLE::open`.

The breakdown columns are:

- `rows`, `cols`: Semper's MLE matrix shape.
- `code_len`: encoded row length committed by the Merkle tree.
- `queries`: sampled column openings.
- `direct_v_prime_B`: explicit row-linear-combination message. This is zero for Orion after the code-switching-lite fix.
- `column_B`: opened committed column data.
- `path_B`: Merkle authentication paths.
- `codeswitch_B`: derived-vector commitment, product-2 sumcheck, and final derived-vector opening.

### logn = 16

| field | scheme | commit_s | open_s | open_proof_B | rows | cols | code_len | queries | direct_v_prime_B | column_B | path_B | codeswitch_B | open_ok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ext | Ligero | 0.123337 | 0.003696 | 104448 | 32 | 2048 | 4096 | 80 | 32768 | 40960 | 30720 | 0 | true |
| ext | Orion | 0.121339 | 0.023525 | 132272 | 32 | 2048 | 8192 | 100 | 0 | 51200 | 41600 | 39472 | true |
| base | Ligero | 0.077744 | 0.001846 | 83968 | 32 | 2048 | 4096 | 80 | 32768 | 20480 | 30720 | 0 | true |
| base | Orion | 0.058410 | 0.015080 | 106672 | 32 | 2048 | 8192 | 100 | 0 | 25600 | 41600 | 39472 | true |

### logn = 18

| field | scheme | commit_s | open_s | open_proof_B | rows | cols | code_len | queries | direct_v_prime_B | column_B | path_B | codeswitch_B | open_ok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ext | Ligero | 0.115955 | 0.011391 | 180736 | 64 | 4096 | 8192 | 80 | 65536 | 81920 | 33280 | 0 | true |
| ext | Orion | 0.113069 | 0.050494 | 191840 | 64 | 4096 | 16384 | 100 | 0 | 102400 | 44800 | 44640 | true |
| base | Ligero | 0.120213 | 0.011289 | 139776 | 64 | 4096 | 8192 | 80 | 65536 | 40960 | 33280 | 0 | true |
| base | Orion | 0.095584 | 0.011902 | 140640 | 64 | 4096 | 16384 | 100 | 0 | 51200 | 44800 | 44640 | true |

### logn = 20

| field | scheme | commit_s | open_s | open_proof_B | rows | cols | code_len | queries | direct_v_prime_B | column_B | path_B | codeswitch_B | open_ok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ext | Ligero | 0.300770 | 0.023069 | 330752 | 128 | 8192 | 16384 | 80 | 131072 | 163840 | 35840 | 0 | true |
| ext | Orion | 0.401600 | 0.037228 | 308240 | 128 | 8192 | 32768 | 100 | 0 | 204800 | 48000 | 55440 | true |
| base | Ligero | 0.131857 | 0.015325 | 248832 | 128 | 8192 | 16384 | 80 | 131072 | 81920 | 35840 | 0 | true |
| base | Orion | 0.250632 | 0.033143 | 205840 | 128 | 8192 | 32768 | 100 | 0 | 102400 | 48000 | 55440 | true |

Conclusion after the code-switching-lite fix: for small vectors (`logn=16`) the extra derived-vector commitment and sumcheck overhead is larger than the removed `v_prime`; around `logn=18` it is near break-even; by `logn=20` Orion has the expected smaller opening proof size while open time is slower. This matches the qualitative expectation that Orion trades more opening work for less communication at larger sizes.

## Commit-Time Optimization: Duplicate-Half Merkle Storage

Orion's current Kaizen-compatible codeword shape is:

```text
[padded Encode(row), length 2b] || [same padded Encode(row), length 2b]
```

The previous Semper implementation materialized and hashed both halves. That doubled row-codeword storage and repeated the same leaf hashing work. The optimized implementation now keeps the logical code length unchanged at `4b`, but stores only the first padded half of length `2b`.

Implementation details:

- `orion_encode_padded_impl` computes only the first padded half.
- Public `orion_encode_base/ext` still returns the full duplicated codeword, preserving existing verifier/dual-code semantics.
- `MerkleTree_base` and `MerkleTree_ext` now support `stored_num_cols` and `logical_num_cols`.
- Orion base/ext provers commit with `stored_num_cols = 2b` and `logical_num_cols = 4b`.
- `MerkleOpen(idx)` authenticates the logical index `idx`, but returns the stored column `idx % stored_num_cols`.
- The Merkle leaf layer hashes stored columns once and copies the digest into repeated logical leaves.

This does not increase proof size: query indexes, logical code length, Merkle path depth, query count, and opened column sizes are unchanged.

### logn = 20 after duplicate-half optimization

Command:

```sh
./build/bench pcs_compare 20 8
```

| field | scheme | commit_s | open_s | open_proof_B | rows | cols | code_len | queries | direct_v_prime_B | column_B | path_B | codeswitch_B | open_ok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ext | Ligero | 0.292500 | 0.017516 | 330752 | 128 | 8192 | 16384 | 80 | 131072 | 163840 | 35840 | 0 | true |
| ext | Orion | 0.230214 | 0.032931 | 308240 | 128 | 8192 | 32768 | 100 | 0 | 204800 | 48000 | 55440 | true |
| base | Ligero | 0.139620 | 0.010228 | 248832 | 128 | 8192 | 16384 | 80 | 131072 | 81920 | 35840 | 0 | true |
| base | Orion | 0.176390 | 0.025432 | 205840 | 128 | 8192 | 32768 | 100 | 0 | 102400 | 48000 | 55440 | true |

### logn = 24 base field after duplicate-half optimization

Command:

```sh
./build/bench pcs_compare_base 24 8
```

| field | scheme | commit_s | open_s | open_proof_B | rows | cols | code_len | queries | direct_v_prime_B | column_B | path_B | codeswitch_B | open_ok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | Ligero | 1.533070 | 0.151442 | 892928 | 512 | 32768 | 65536 | 80 | 524288 | 327680 | 40960 | 0 | true |
| base | Orion | 1.881790 | 0.156387 | 548720 | 512 | 32768 | 131072 | 100 | 0 | 409600 | 54400 | 84720 | true |

Before this optimization, the same base-field `logn=24` Orion commit time was `2.437920s` with the same `548720B` opening proof. The optimized run reduces Orion commit time by about `22.8%` while preserving proof size.

## Additional Commit-Time Optimization: Half Merkle Tree and Batched Column Hashing

After duplicate-half storage, the main remaining commit bottleneck was Merkle construction. For `logn=24` base-field, `orion merkle` was still about `0.979s` out of `1.881790s`.

Two further optimizations were applied:

- Repeated logical leaves now build only the stored half Merkle tree plus a tiny repeat tree. The committed logical code length remains `4b`; opening still returns a full logical-path proof by appending the repeat-tree sibling path.
- Column hashing now serializes each column into a contiguous byte buffer and calls `EVP_DigestUpdate` once per column, instead of once per field element.

This keeps proof size unchanged. The verifier still sees the same root, same logical index, same path length, and same opened column size.

### logn = 20 after half-tree and batched-hash optimization

Command:

```sh
./build/bench pcs_compare 20 8
```

| field | scheme | commit_s | open_s | open_proof_B | rows | cols | code_len | queries | direct_v_prime_B | column_B | path_B | codeswitch_B | open_ok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ext | Ligero | 0.290456 | 0.017390 | 330752 | 128 | 8192 | 16384 | 80 | 131072 | 163840 | 35840 | 0 | true |
| ext | Orion | 0.215702 | 0.032385 | 308240 | 128 | 8192 | 32768 | 100 | 0 | 204800 | 48000 | 55440 | true |
| base | Ligero | 0.142135 | 0.011662 | 248832 | 128 | 8192 | 16384 | 80 | 131072 | 81920 | 35840 | 0 | true |
| base | Orion | 0.128060 | 0.022838 | 205840 | 128 | 8192 | 32768 | 100 | 0 | 102400 | 48000 | 55440 | true |

### logn = 24 base field after half-tree and batched-hash optimization

Command:

```sh
./build/bench pcs_compare_base 24 8
```

| field | scheme | commit_s | open_s | open_proof_B | rows | cols | code_len | queries | direct_v_prime_B | column_B | path_B | codeswitch_B | open_ok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | Ligero | 1.707520 | 0.143728 | 892928 | 512 | 32768 | 65536 | 80 | 524288 | 327680 | 40960 | 0 | true |
| base | Orion | 1.562320 | 0.157147 | 548720 | 512 | 32768 | 131072 | 100 | 0 | 409600 | 54400 | 84720 | true |

Relative to the previous duplicate-half-only run (`1.881790s`), this reduces `logn=24` base Orion commit time by about `17.0%`. Relative to the pre-optimization Orion run (`2.437920s`), total commit-time reduction is about `35.9%`, still with the same `548720B` opening proof.

## Prover-Time Optimization: Encode-Into and Blocked Lincomb

The next prover-time pass targeted avoidable allocation/copy work in commit and poor cache locality in opening.

Changes:

- Orion commit now uses `orion_encode_padded_into(input_ptr, output_ptr, scratch)` instead of constructing a row vector, returning an encoded vector, and copying it into the committed codeword matrix.
- Each OpenMP worker reuses its own encoder scratch buffers across assigned rows.
- `orionProver_base/ext::lincomb` now computes `R^T M` by column blocks. For each block it scans row memory contiguously, instead of accessing one column across all rows with a large stride.

These changes do not affect proof format, query count, Merkle paths, or proof size.

### logn = 24 base field after encode-into and blocked-lincomb optimization

Command:

```sh
./build/bench pcs_compare_base 24 8
```

| field | scheme | commit_s | open_s | open_proof_B | rows | cols | code_len | queries | direct_v_prime_B | column_B | path_B | codeswitch_B | open_ok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | Ligero | 1.795770 | 0.133930 | 892928 | 512 | 32768 | 65536 | 80 | 524288 | 327680 | 40960 | 0 | true |
| base | Orion | 1.551720 | 0.091712 | 548720 | 512 | 32768 | 131072 | 100 | 0 | 409600 | 54400 | 84720 | true |

For this run, `orion_open_lincomb` dropped to `0.020s`; previously it was about `0.111s-0.115s` at this size.

### logn = 28 base field after encode-into and blocked-lincomb optimization

Command:

```sh
./build/bench pcs_compare_base 28 8
```

| field | scheme | commit_s | open_s | open_proof_B | rows | cols | code_len | queries | direct_v_prime_B | column_B | path_B | codeswitch_B | open_ok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | Ligero | 20.221900 | 2.901470 | 3453952 | 2048 | 131072 | 262144 | 80 | 2097152 | 1310720 | 46080 | 0 | true |
| base | Orion | 25.568400 | 0.608806 | 1839824 | 2048 | 131072 | 524288 | 100 | 0 | 1638400 | 60800 | 140624 | true |

Compared with the previous `logn=28` Orion run (`commit_s=26.813100`, `open_s=3.427420`), commit time improved by about `4.6%`, while open time improved by about `82.2%`. The `orion_open_lincomb` portion dropped from `3.154s` to `0.344s`.

## Commit-Time Optimization: Fused Base Merkle Build and Uninitialized Codeword Buffer

The next commit-time pass focused on avoidable memory traffic in base-field Orion commits.

Changes:

- Base `MerkleTree_base` construction now fuses row-major-to-column-major copy with leaf hashing. Each worker copies one committed column into the stored column buffer and serializes the same values into SHA input bytes in one pass.
- Base Orion commit now allocates its temporary row-major codeword buffer without zero-initializing the whole allocation. The encoder already clears each row output before writing into it, so the previous `vector.resize(..., zero)` was an extra full-buffer write.

These changes do not affect roots, openings, query count, proof size, or verifier behavior.

### logn = 24 base field after fused Merkle and uninitialized-buffer optimization

Command:

```sh
./build/bench pcs_compare_base 24 8
```

| field | scheme | commit_s | open_s | open_proof_B | rows | cols | code_len | queries | direct_v_prime_B | column_B | path_B | codeswitch_B | open_ok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | Ligero | 1.392920 | 0.144822 | 892928 | 512 | 32768 | 65536 | 80 | 524288 | 327680 | 40960 | 0 | true |
| base | Orion | 1.429030 | 0.067124 | 548720 | 512 | 32768 | 131072 | 100 | 0 | 409600 | 54400 | 84720 | true |

### logn = 28 base field after fused Merkle and uninitialized-buffer optimization

Command:

```sh
./build/bench pcs_compare_base 28 8
```

| field | scheme | commit_s | open_s | open_proof_B | rows | cols | code_len | queries | direct_v_prime_B | column_B | path_B | codeswitch_B | open_ok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | Ligero | 22.454200 | 2.868000 | 3453952 | 2048 | 131072 | 262144 | 80 | 2097152 | 1310720 | 46080 | 0 | true |
| base | Orion | 21.971600 | 0.483592 | 1839824 | 2048 | 131072 | 524288 | 100 | 0 | 1638400 | 60800 | 140624 | true |

Compared with the previous `logn=28` Orion run (`commit_s=25.568400`, `open_s=0.608806`), commit time improved by about `14.1%`. Compared with the original post-code-switching `logn=28` Orion run (`commit_s=26.813100`, `open_s=3.427420`), total improvements are about `18.1%` for commit time and `85.9%` for open time, with the same `1839824B` opening proof.

## Orion `loga` Sweep for Proof-Size Tradeoffs

The benchmark CLI now supports:

```sh
./build/bench pcs_compare_base <logn> <threads> <orion_loga>
./build/bench pcs_orion_loga_sweep_base <logn_min> <logn_max> <threads> <below_default> <above_default>
```

The sweep is base-field only. By default it avoids extreme matrix shapes and tests a window around the current Orion default:

```text
loga in [default_loga - 3, default_loga + 1]
```

This captures the useful proof-size/prover-time tradeoff region without spending time on very small `loga` values with huge row lengths or very large `loga` values with excessive opened-column height.

Command used:

```sh
./build/bench pcs_orion_loga_sweep_base 20 30 8 3 1
```

Columns:

- `prover_s = commit_s + open_s`
- `open_proof_B` is the total opening proof size.
- `column_B`, `path_B`, and `codeswitch_B` are the proof-size breakdown.

```csv
logn,loga,default_loga,commit_s,open_s,prover_s,pcs_size_B,open_proof_B,rows,cols,code_len,queries,column_B,path_B,codeswitch_B,open_ok
20,4,7,0.633572,0.07675,0.710322,32,175648,16,65536,262144,100,12800,57600,105248,true
20,5,7,0.321598,0.086182,0.40778,32,164720,32,32768,131072,100,25600,54400,84720,true
20,6,7,0.206141,0.024695,0.230836,32,168128,64,16384,65536,100,51200,51200,65728,true
20,7,7,0.109111,0.020992,0.130103,32,205840,128,8192,32768,100,102400,48000,55440,true
20,8,7,0.075993,0.013721,0.089714,32,294240,256,4096,16384,100,204800,44800,44640,true
21,4,7,1.15903,0.123736,1.28276,32,214224,16,131072,524288,100,12800,60800,140624,true
21,5,7,0.643061,0.051227,0.694288,32,188448,32,65536,262144,100,25600,57600,105248,true
21,6,7,0.342785,0.036102,0.378887,32,190320,64,32768,131072,100,51200,54400,84720,true
21,7,7,0.220896,0.024508,0.245404,32,219328,128,16384,65536,100,102400,51200,65728,true
21,8,7,0.164267,0.018873,0.18314,32,308240,256,8192,32768,100,204800,48000,55440,true
22,5,8,1.33151,0.158175,1.48969,32,227024,32,131072,524288,100,25600,60800,140624,true
22,6,8,0.714893,0.05708,0.771973,32,214048,64,65536,262144,100,51200,57600,105248,true
22,7,8,0.455889,0.040807,0.496696,32,241520,128,32768,131072,100,102400,54400,84720,true
22,8,8,0.320496,0.037746,0.358242,32,321728,256,16384,65536,100,204800,51200,65728,true
22,9,8,0.275594,0.029266,0.30486,32,513040,512,8192,32768,100,409600,48000,55440,true
23,5,8,2.92993,0.242405,3.17234,32,271232,32,262144,1048576,100,25600,64000,181632,true
23,6,8,1.28329,0.106103,1.38939,32,252624,64,131072,524288,100,51200,60800,140624,true
23,7,8,0.848727,0.056464,0.905191,32,265248,128,65536,262144,100,102400,57600,105248,true
23,8,8,0.606805,0.051903,0.658708,32,343920,256,32768,131072,100,204800,54400,84720,true
23,9,8,0.549667,0.026159,0.575826,32,526528,512,16384,65536,100,409600,51200,65728,true
24,6,9,2.98602,0.225358,3.21137,32,296832,64,262144,1048576,100,51200,64000,181632,true
24,7,9,1.74456,0.152883,1.89745,32,303824,128,131072,524288,100,102400,60800,140624,true
24,8,9,1.42806,0.08418,1.51224,32,367648,256,65536,262144,100,204800,57600,105248,true
24,9,9,1.17973,0.06961,1.24934,32,548720,512,32768,131072,100,409600,54400,84720,true
24,10,9,1.1056,0.041615,1.14722,32,936128,1024,16384,65536,100,819200,51200,65728,true
25,6,9,6.64201,0.777554,7.41957,32,368176,64,524288,2097152,100,51200,67200,249776,true
25,7,9,4.37631,0.269661,4.64597,32,348032,128,262144,1048576,100,102400,64000,181632,true
25,8,9,2.65505,0.182806,2.83786,32,406224,256,131072,524288,100,204800,60800,140624,true
25,9,9,2.40229,0.132913,2.5352,32,572448,512,65536,262144,100,409600,57600,105248,true
25,10,9,2.19208,0.115605,2.30768,32,958320,1024,32768,131072,100,819200,54400,84720,true
26,7,10,9.07214,0.657035,9.72917,32,419376,128,524288,2097152,100,102400,67200,249776,true
26,8,10,6.54464,0.467587,7.01223,32,450432,256,262144,1048576,100,204800,64000,181632,true
26,9,10,7.49097,0.212313,7.70329,32,611024,512,131072,524288,100,409600,60800,140624,true
26,10,10,4.4608,0.166706,4.62751,32,982048,1024,65536,262144,100,819200,57600,105248,true
26,11,10,4.30484,0.102744,4.40759,32,1777520,2048,32768,131072,100,1638400,54400,84720,true
27,7,10,21.9341,2.04152,23.9756,32,504544,128,1048576,4194304,100,102400,70400,331744,true
27,8,10,14.4034,1.11413,15.5175,32,521776,256,524288,2097152,100,204800,67200,249776,true
27,9,10,12.7703,0.694236,13.4646,32,655232,512,262144,1048576,100,409600,64000,181632,true
27,10,10,15.2911,0.579043,15.8701,32,1020624,1024,131072,524288,100,819200,60800,140624,true
27,11,10,9.63284,0.327039,9.95988,32,1801250,2048,65536,262144,100,1638400,57600,105248,true
28,8,11,33.0375,2.502,35.5395,32,606944,256,1048576,4194304,100,204800,70400,331744,true
28,9,11,30.8717,1.30124,32.1729,32,726576,512,524288,2097152,100,409600,67200,249776,true
28,10,11,24.6375,0.987219,25.6248,32,1064830,1024,262144,1048576,100,819200,64000,181632,true
28,11,11,26.6605,1.01895,27.6794,32,1839820,2048,131072,524288,100,1638400,60800,140624,true
28,12,11,20.8022,0.60336,21.4056,32,3439650,4096,65536,262144,100,3276800,57600,105248,true
29,8,11,88.9857,3.75245,92.7381,32,743824,256,2097152,8388608,100,204800,73600,465424,true
29,9,11,60.9375,2.4246,63.3621,32,811744,512,1048576,4194304,100,409600,70400,331744,true
29,10,11,52.5031,2.43895,54.9421,32,1136180,1024,524288,2097152,100,819200,67200,249776,true
29,11,11,45.328,1.37324,46.7013,32,1884030,2048,262144,1048576,100,1638400,64000,181632,true
29,12,11,72.5656,1.42316,73.9888,32,3478220,4096,131072,524288,100,3276800,60800,140624,true
30,9,12,151.828,4.87289,156.701,32,948624,512,2097152,8388608,100,409600,73600,465424,true
30,10,12,129.958,4.41599,134.374,32,1221340,1024,1048576,4194304,100,819200,70400,331744,true
30,11,12,107.937,3.86909,111.806,32,1955380,2048,524288,2097152,100,1638400,67200,249776,true
30,12,12,106.134,3.18395,109.318,32,3522430,4096,262144,1048576,100,3276800,64000,181632,true
30,13,12,86.5667,2.48113,89.0478,32,6755020,8192,131072,524288,100,6553600,60800,140624,true
```

Observations:

- Smaller `loga` reduces opening proof size because the opened committed columns are shorter.
- The tradeoff is higher prover time from longer row encodings and larger code-switching vectors.
- For `logn=28`, `loga=10` is a useful communication-saving point: `open_proof_B=1064830` versus default `loga=11` at `1839820`, with slightly lower measured prover time in this run.
- For `logn=29`, default `loga=11` is a better balanced point than the lower-proof `loga=8..10`, which are much slower.
- For `logn=30`, `loga=13` is fastest in this window but has a large proof (`6755020B`); default `loga=12` is more balanced, while `loga=11` halves the proof relative to `loga=12` but has similar prover time in this run.

## Single-Thread Orion Default Sweep

The previous `loga` sweep was useful for finding the shape of the tradeoff, but it used multiple OpenMP threads. The hard-coded Orion table below is based on single-threaded base-field runs.

Command shape:

```sh
./build/bench pcs_orion_loga_sweep_base <logn_min> <logn_max> 1 <below_default> <above_default>
```

For targeted points, `<below_default>` and `<above_default>` were chosen so the command produced one exact `loga`. The sweep keeps at most seven candidates per `logn`; very large `loga` values were treated as proof-size-extreme and skipped for the largest sizes. `default_loga` in the raw CSV is the code default at the time that row was run, not necessarily the final table below. `open_ok=true` means all tested `loga` values in that command produced the same opening value and passed Orion's internal opening checks. For the memory-light sweep path, this is a consistency check across PCS openings rather than an independent `MLE::open` recomputation.

Selection policy:

- Keep the existing small-size table for `logn < 20`; those rows are cheap and noisy.
- For `20 <= logn <= 33`, pick a Pareto knee from the single-thread rows.
- Prefer smaller proofs when the prover-time cost is moderate.
- Prefer faster points when a smaller proof saves little communication but costs substantial wall time.

The resulting table lives in `src/orion.cpp` and is exposed through:

```cpp
int orion_default_loga(int num_vars);
```

```text
logn:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
loga: -1  1  1  2  3  4  5  1  1  1  2  2  3  3  4  4  5  5  6  6  6  7  5  8  7  8  8  9  9 10 10 11 11 11
```

Notable single-thread choices:

- `logn=20`: `7 -> 6`, proof drops from `205840B` to `168128B` with moderate time cost.
- `logn=22`: `8 -> 5`, proof drops from `321728B` to `227024B`; `loga=6` is smaller but much slower.
- `logn=24`: `8 -> 7`, proof drops from `367648B` to `303824B`.
- `logn=26`: `10 -> 8`, proof drops from `982048B` to `450432B` with similar prover time.
- `logn=27`: `8 -> 9`, the `7/8` proof saving costs too much time; `9` is the faster knee.
- `logn=31..33`: all move to `11`; for `logn=33`, `11` dominates the measured `12` result and `13` did not produce a row.

Attempt notes:

- `logn=33, loga=13` was attempted but the output file contained only the CSV header, so it is recorded here as failed/incomplete and was not used.
- `logn=33, loga=11` completed on 2026-05-28 after about 59 minutes, using one OpenMP thread.

Selected single-thread rows:

| logn | selected loga | prover_s | open_proof_B |
| ---: | ---: | ---: | ---: |
| 20 | 6 | 0.337174 | 168128 |
| 21 | 7 | 0.407411 | 219328 |
| 22 | 5 | 1.48448 | 227024 |
| 23 | 8 | 1.54768 | 343920 |
| 24 | 7 | 3.43775 | 303824 |
| 25 | 8 | 6.80112 | 406224 |
| 26 | 8 | 16.3866 | 450432 |
| 27 | 9 | 33.8253 | 655232 |
| 28 | 9 | 74.6246 | 726576 |
| 29 | 10 | 156.396 | 1136180 |
| 30 | 10 | 438.731 | 1221340 |
| 31 | 11 | 822.925 | 2040540 |
| 32 | 11 | 1530.79 | 2177420 |
| 33 | 11 | 3362.88 | 2344510 |

Raw single-thread rows:

```csv
logn,loga,default_loga,commit_s,open_s,prover_s,pcs_size_B,open_proof_B,rows,cols,code_len,queries,column_B,path_B,codeswitch_B,open_ok
1,1,1,0.00135,9.8e-05,0.001448,32,400,2,1,4,4,64,256,80,true
2,1,1,1.6e-05,6.1e-05,7.7e-05,32,976,2,2,8,8,128,768,80,true
2,2,1,1.2e-05,3.4e-05,4.6e-05,32,464,4,1,4,4,128,256,80,true
3,1,2,2.1e-05,0.000146,0.000167,32,2848,2,4,16,16,256,2048,544,true
3,2,2,1.6e-05,5e-05,6.6e-05,32,1104,4,2,8,8,256,768,80,true
3,3,2,1.2e-05,3.5e-05,4.7e-05,32,592,8,1,4,4,256,256,80,true
4,1,3,3.4e-05,0.00022,0.000254,32,6352,2,8,32,32,512,5120,720,true
4,2,3,2.2e-05,0.00012,0.000142,32,3104,4,4,16,16,512,2048,544,true
4,3,3,1.7e-05,5.6e-05,7.3e-05,32,1360,8,2,8,8,512,768,80,true
4,4,3,1.3e-05,3.7e-05,5e-05,32,848,16,1,4,4,512,256,80,true
5,1,4,7.1e-05,0.000431,0.000502,32,14336,2,16,64,64,1024,12288,1024,true
5,2,4,3.5e-05,0.000214,0.000249,32,6864,4,8,32,32,1024,5120,720,true
5,3,4,2.3e-05,0.000122,0.000145,32,3616,8,4,16,16,1024,2048,544,true
5,4,4,1.7e-05,5.4e-05,7.1e-05,32,1872,16,2,8,8,1024,768,80,true
5,5,4,1.3e-05,4e-05,5.3e-05,32,1360,32,1,4,4,1024,256,80,true
6,2,5,5.8e-05,0.000424,0.000482,32,15360,4,16,64,64,2048,12288,1024,true
6,3,5,3.7e-05,0.000221,0.000258,32,7888,8,8,32,32,2048,5120,720,true
6,4,5,2.5e-05,0.000126,0.000151,32,4640,16,4,16,16,2048,2048,544,true
6,5,5,1.9e-05,0.000123,0.000142,32,2896,32,2,8,8,2048,768,80,true
6,6,5,1.6e-05,4.6e-05,6.2e-05,32,2384,64,1,4,4,2048,256,80,true
7,1,1,0.000333,0.001058,0.001391,32,29856,2,64,256,100,1600,25600,2656,true
7,2,1,0.000116,0.00101,0.001126,32,27184,4,32,128,100,3200,22400,1584,true
7,3,1,0.000107,0.000524,0.000631,32,17408,8,16,64,64,4096,12288,1024,true
7,4,1,4.1e-05,0.000229,0.00027,32,9936,16,8,32,32,4096,5120,720,true
8,1,1,0.000463,0.001641,0.002104,32,52272,2,128,512,100,1600,28800,21872,true
8,2,1,0.000207,0.000835,0.001042,32,31456,4,64,256,100,3200,25600,2656,true
8,3,1,0.000123,0.000912,0.001035,32,30384,8,32,128,100,6400,22400,1584,true
8,4,1,0.000114,0.000468,0.000582,32,21504,16,16,64,64,8192,12288,1024,true
9,1,1,0.000968,0.002563,0.003531,32,59104,2,256,1024,100,1600,32000,25504,true
9,2,1,0.000408,0.00159,0.001998,32,53872,4,128,512,100,3200,28800,21872,true
9,3,1,0.000255,0.000869,0.001124,32,34656,8,64,256,100,6400,25600,2656,true
9,4,1,0.000218,0.000935,0.001153,32,36784,16,32,128,100,12800,22400,1584,true
10,1,2,0.001809,0.002739,0.004548,32,66960,2,512,2048,100,1600,35200,30160,true
10,2,2,0.000792,0.001989,0.002781,32,60704,4,256,1024,100,3200,32000,25504,true
10,3,2,0.000477,0.001621,0.002098,32,57072,8,128,512,100,6400,28800,21872,true
10,4,2,0.000301,0.000859,0.00116,32,41056,16,64,256,100,12800,25600,2656,true
10,5,2,0.000193,0.000794,0.000987,32,49584,32,32,128,100,25600,22400,1584,true
11,1,2,0.003528,0.004257,0.007785,32,72768,2,1024,4096,100,1600,38400,32768,true
11,2,2,0.001625,0.002801,0.004426,32,68560,4,512,2048,100,3200,35200,30160,true
11,3,2,0.001219,0.002009,0.003228,32,63904,8,256,1024,100,6400,32000,25504,true
11,4,2,0.000567,0.001626,0.002193,32,63472,16,128,512,100,12800,28800,21872,true
11,5,2,0.00038,0.000945,0.001325,32,53856,32,64,256,100,25600,25600,2656,true
12,1,3,0.007426,0.004928,0.012354,32,82672,2,2048,8192,100,1600,41600,39472,true
12,2,3,0.003013,0.003107,0.00612,32,74368,4,1024,4096,100,3200,38400,32768,true
12,3,3,0.00181,0.002689,0.004499,32,71760,8,512,2048,100,6400,35200,30160,true
12,4,3,0.001467,0.002389,0.003856,32,70304,16,256,1024,100,12800,32000,25504,true
12,5,3,0.001049,0.002226,0.003275,32,76272,32,128,512,100,25600,28800,21872,true
12,6,3,0.00076,0.001455,0.002215,32,79456,64,64,256,100,51200,25600,2656,true
13,1,3,0.015529,0.006426,0.021955,32,91040,2,4096,16384,100,1600,44800,44640,true
13,2,3,0.006044,0.00518,0.011224,32,84272,4,2048,8192,100,3200,41600,39472,true
13,3,3,0.004504,0.003297,0.007801,32,77568,8,1024,4096,100,6400,38400,32768,true
13,4,3,0.002982,0.004444,0.007426,32,78160,16,512,2048,100,12800,35200,30160,true
13,5,3,0.002189,0.003244,0.005433,32,83104,32,256,1024,100,25600,32000,25504,true
13,6,3,0.001692,0.002785,0.004477,32,101872,64,128,512,100,51200,28800,21872,true
14,1,4,0.045423,0.014792,0.060215,32,105040,2,8192,32768,100,1600,48000,55440,true
14,2,4,0.012768,0.006063,0.018831,32,92640,4,4096,16384,100,3200,44800,44640,true
14,3,4,0.00725,0.004641,0.011891,32,87472,8,2048,8192,100,6400,41600,39472,true
14,4,4,0.004821,0.003646,0.008467,32,83968,16,1024,4096,100,12800,38400,32768,true
14,5,4,0.003059,0.00282,0.005879,32,90960,32,512,2048,100,25600,35200,30160,true
14,6,4,0.002386,0.002397,0.004783,32,108704,64,256,1024,100,51200,32000,25504,true
14,7,4,0.001914,0.002015,0.003929,32,153072,128,128,512,100,102400,28800,21872,true
15,1,4,0.058877,0.016829,0.075706,32,118528,2,16384,65536,100,1600,51200,65728,true
15,2,4,0.024984,0.010157,0.035141,32,106640,4,8192,32768,100,3200,48000,55440,true
15,3,4,0.014578,0.005859,0.020437,32,95840,8,4096,16384,100,6400,44800,44640,true
15,4,4,0.008801,0.004728,0.013529,32,93872,16,2048,8192,100,12800,41600,39472,true
15,5,4,0.005962,0.00333,0.009292,32,96768,32,1024,4096,100,25600,38400,32768,true
15,6,4,0.004585,0.002952,0.007537,32,116560,64,512,2048,100,51200,35200,30160,true
15,7,4,0.003901,0.002495,0.006396,32,159904,128,256,1024,100,102400,32000,25504,true
16,2,5,0.050763,0.015135,0.065898,32,120128,4,16384,65536,100,3200,51200,65728,true
16,3,5,0.030794,0.010023,0.040817,32,109840,8,8192,32768,100,6400,48000,55440,true
16,4,5,0.017965,0.006097,0.024062,32,102240,16,4096,16384,100,12800,44800,44640,true
16,5,5,0.012295,0.004959,0.017254,32,106672,32,2048,8192,100,25600,41600,39472,true
16,6,5,0.009437,0.003546,0.012983,32,122368,64,1024,4096,100,51200,38400,32768,true
16,7,5,0.008103,0.003263,0.011366,32,167760,128,512,2048,100,102400,35200,30160,true
16,8,5,0.007432,0.002951,0.010383,32,262304,256,256,1024,100,204800,32000,25504,true
17,2,5,0.127724,0.046866,0.17459,32,142320,4,32768,131072,100,3200,54400,84720,true
17,3,5,0.064454,0.018698,0.083152,32,123328,8,16384,65536,100,6400,51200,65728,true
17,4,5,0.038957,0.012128,0.051085,32,116240,16,8192,32768,100,12800,48000,55440,true
17,5,5,0.024858,0.006774,0.031632,32,115040,32,4096,16384,100,25600,44800,44640,true
17,6,5,0.019707,0.005336,0.025043,32,132272,64,2048,8192,100,51200,41600,39472,true
17,7,5,0.01685,0.004317,0.021167,32,173568,128,1024,4096,100,102400,38400,32768,true
17,8,5,0.015257,0.003903,0.01916,32,270160,256,512,2048,100,204800,35200,30160,true
18,3,6,0.152464,0.033791,0.186255,32,145520,8,32768,131072,100,6400,54400,84720,true
18,4,6,0.168,0.031452,0.199452,32,129728,16,16384,65536,100,12800,51200,65728,true
18,5,6,0.116632,0.024517,0.141149,32,129040,32,8192,32768,100,25600,48000,55440,true
18,6,6,0.067747,0.012204,0.079951,32,140640,64,4096,16384,100,51200,44800,44640,true
18,7,6,0.060949,0.007142,0.068091,32,183472,128,2048,8192,100,102400,41600,39472,true
18,8,6,0.066409,0.010156,0.076565,32,275968,256,1024,4096,100,204800,38400,32768,true
18,9,6,0.051503,0.014222,0.065725,32,474960,512,512,2048,100,409600,35200,30160,true
19,3,6,0.406252,0.108725,0.514977,32,169248,8,65536,262144,100,6400,57600,105248,true
19,4,6,0.356855,0.055349,0.412204,32,151920,16,32768,131072,100,12800,54400,84720,true
19,5,6,0.222597,0.055956,0.278553,32,142528,32,16384,65536,100,25600,51200,65728,true
19,6,6,0.191879,0.023739,0.215618,32,154640,64,8192,32768,100,51200,48000,55440,true
19,7,6,0.141815,0.015066,0.156881,32,191840,128,4096,16384,100,102400,44800,44640,true
19,8,6,0.133795,0.01308,0.146875,32,285872,256,2048,8192,100,204800,41600,39472,true
19,9,6,0.118564,0.012154,0.130718,32,480768,512,1024,4096,100,409600,38400,32768,true
20,4,7,0.701068,0.100714,0.801782,32,175648,16,65536,262144,100,12800,57600,105248,true
20,5,7,0.430621,0.043647,0.474268,32,164720,32,32768,131072,100,25600,54400,84720,true
20,6,7,0.309263,0.027911,0.337174,32,168128,64,16384,65536,100,51200,51200,65728,true
20,7,7,0.270986,0.01924,0.290226,32,205840,128,8192,32768,100,102400,48000,55440,true
20,8,7,0.25297,0.012817,0.265787,32,294240,256,4096,16384,100,204800,44800,44640,true
20,9,7,0.251784,0.015929,0.267713,32,490672,512,2048,8192,100,409600,41600,39472,true
20,10,7,0.232553,0.018602,0.251155,32,890368,1024,1024,4096,100,819200,38400,32768,true
21,4,7,1.05736,0.136538,1.1939,32,214224,16,131072,524288,100,12800,60800,140624,true
21,5,7,0.509303,0.060924,0.570227,32,188448,32,65536,262144,100,25600,57600,105248,true
21,6,7,0.470102,0.050518,0.52062,32,190320,64,32768,131072,100,51200,54400,84720,true
21,7,7,0.38062,0.026791,0.407411,32,219328,128,16384,65536,100,102400,51200,65728,true
21,8,7,0.330967,0.024503,0.35547,32,308240,256,8192,32768,100,204800,48000,55440,true
21,9,7,0.329478,0.018887,0.348365,32,499040,512,4096,16384,100,409600,44800,44640,true
21,10,7,0.328905,0.019038,0.347943,32,900272,1024,2048,8192,100,819200,41600,39472,true
22,5,8,1.33186,0.152622,1.48448,32,227024,32,131072,524288,100,25600,60800,140624,true
22,6,8,1.64936,0.135822,1.78518,32,214048,64,65536,262144,100,51200,57600,105248,true
22,7,8,1.37659,0.080836,1.45742,32,241520,128,32768,131072,100,102400,54400,84720,true
22,8,8,1.19653,0.057368,1.2539,32,321728,256,16384,65536,100,204800,51200,65728,true
22,9,8,0.967583,0.048214,1.0158,32,513040,512,8192,32768,100,409600,48000,55440,true
22,10,8,0.934412,0.038978,0.97339,32,908640,1024,4096,16384,100,819200,44800,44640,true
22,11,8,0.972817,0.047687,1.0205,32,1.71947e+06,2048,2048,8192,100,1638400,41600,39472,true
23,5,8,4.87323,0.593277,5.4665,32,271232,32,262144,1048576,100,25600,64000,181632,true
23,6,8,2.78396,0.228309,3.01227,32,252624,64,131072,524288,100,51200,60800,140624,true
23,7,8,2.13902,0.086661,2.22568,32,265248,128,65536,262144,100,102400,57600,105248,true
23,8,8,1.48667,0.061014,1.54768,32,343920,256,32768,131072,100,204800,54400,84720,true
23,9,8,1.46056,0.051031,1.51159,32,526528,512,16384,65536,100,409600,51200,65728,true
23,10,8,1.45686,0.045693,1.50255,32,922640,1024,8192,32768,100,819200,48000,55440,true
23,11,8,1.60102,0.04589,1.64691,32,1.72784e+06,2048,4096,16384,100,1638400,44800,44640,true
24,5,8,5.79956,0.628996,6.42856,32,342576,32,524288,2097152,100,25600,67200,249776,true
24,6,8,4.52397,0.309197,4.83317,32,296832,64,262144,1048576,100,51200,64000,181632,true
24,7,8,3.2569,0.180853,3.43775,32,303824,128,131072,524288,100,102400,60800,140624,true
24,8,8,3.02583,0.117185,3.14302,32,367648,256,65536,262144,100,204800,57600,105248,true
24,9,8,3.08461,0.100989,3.1856,32,548720,512,32768,131072,100,409600,54400,84720,true
24,10,8,3.48427,0.166598,3.65087,32,936128,1024,16384,65536,100,819200,51200,65728,true
24,11,8,4.25331,0.081729,4.33504,32,1.74184e+06,2048,8192,32768,100,1638400,48000,55440,true
25,5,8,15.9266,1.35255,17.2791,32,427744,32,1048576,4194304,100,25600,70400,331744,true
25,6,8,9.97714,0.659391,10.6365,32,368176,64,524288,2097152,100,51200,67200,249776,true
25,7,8,8.16244,0.349162,8.5116,32,348032,128,262144,1048576,100,102400,64000,181632,true
25,8,8,6.45193,0.349187,6.80112,32,406224,256,131072,524288,100,204800,60800,140624,true
25,9,8,6.90368,0.324695,7.22837,32,572448,512,65536,262144,100,409600,57600,105248,true
25,10,8,9.18524,0.165226,9.35047,32,958320,1024,32768,131072,100,819200,54400,84720,true
25,11,8,6.48843,0.256504,6.74493,32,1.75533e+06,2048,16384,65536,100,1638400,51200,65728,true
26,7,10,19.5678,0.798971,20.3668,32,419376,128,524288,2097152,100,102400,67200,249776,true
26,8,10,15.8716,0.515001,16.3866,32,450432,256,262144,1048576,100,204800,64000,181632,true
26,9,10,15.8995,0.365762,16.2653,32,611024,512,131072,524288,100,409600,60800,140624,true
26,10,10,15.3954,0.328955,15.7244,32,982048,1024,65536,262144,100,819200,57600,105248,true
26,11,10,15.1707,0.395364,15.5661,32,1.77752e+06,2048,32768,131072,100,1638400,54400,84720,true
26,12,10,14.8846,0.510104,15.3947,32,3.39373e+06,4096,16384,65536,100,3276800,51200,65728,true
26,13,10,15.5972,0.486915,16.0841,32,6.65704e+06,8192,8192,32768,100,6553600,48000,55440,true
27,5,8,82.5938,8.84317,91.437,32,731712,32,4194304,16777216,100,25600,76800,629312,true
27,6,8,71.9017,3.89278,75.7944,32,590224,64,2097152,8388608,100,51200,73600,465424,true
27,7,8,46.8004,1.76754,48.5679,32,504544,128,1048576,4194304,100,102400,70400,331744,true
27,8,8,47.1405,1.08149,48.222,32,521776,256,524288,2097152,100,204800,67200,249776,true
27,9,8,33.021,0.804279,33.8253,32,655232,512,262144,1048576,100,409600,64000,181632,true
27,10,8,36.3356,0.891075,37.2267,32,1.02062e+06,1024,131072,524288,100,819200,60800,140624,true
27,11,8,30.5298,0.954284,31.4841,32,1.80125e+06,2048,65536,262144,100,1638400,57600,105248,true
28,6,9,150.979,7.93511,158.914,32,757312,64,4194304,16777216,100,51200,76800,629312,true
28,7,9,105.51,4.29365,109.804,32,641424,128,2097152,8388608,100,102400,73600,465424,true
28,8,9,82.3382,2.18968,84.5278,32,606944,256,1048576,4194304,100,204800,70400,331744,true
28,9,9,72.9805,1.6441,74.6246,32,726576,512,524288,2097152,100,409600,67200,249776,true
28,10,9,61.7538,1.34781,63.1016,32,1.06483e+06,1024,262144,1048576,100,819200,64000,181632,true
28,11,9,50.5536,1.22037,51.774,32,1.83982e+06,2048,131072,524288,100,1638400,60800,140624,true
28,12,9,52.156,1.15334,53.3093,32,3.43965e+06,4096,65536,262144,100,3276800,57600,105248,true
29,7,10,284.524,9.35942,293.883,32,808512,128,4194304,16777216,100,102400,76800,629312,true
29,8,10,198.202,4.93029,203.132,32,743824,256,2097152,8388608,100,204800,73600,465424,true
29,9,10,170.587,3.31766,173.905,32,811744,512,1048576,4194304,100,409600,70400,331744,true
29,10,10,153.758,2.63752,156.396,32,1.13618e+06,1024,524288,2097152,100,819200,67200,249776,true
29,11,10,134.82,2.41148,137.231,32,1.88403e+06,2048,262144,1048576,100,1638400,64000,181632,true
29,12,10,109.287,2.43771,111.725,32,3.47822e+06,4096,131072,524288,100,3276800,60800,140624,true
29,13,10,112.393,3.0858,115.479,32,6.71645e+06,8192,65536,262144,100,6553600,57600,105248,true
30,7,10,609.103,19.9502,629.053,32,1.07646e+06,128,8388608,33554432,100,102400,80000,894064,true
30,8,10,661.224,26.5544,687.779,32,910912,256,4194304,16777216,100,204800,76800,629312,true
30,9,10,498.969,12.2052,511.175,32,948624,512,2097152,8388608,100,409600,73600,465424,true
30,10,10,430.12,8.61154,438.731,32,1.22134e+06,1024,1048576,4194304,100,819200,70400,331744,true
30,11,10,324.523,5.8915,330.414,32,1.95538e+06,2048,524288,2097152,100,1638400,67200,249776,true
30,12,10,367.828,16.0347,383.862,32,3.52243e+06,4096,262144,1048576,100,3276800,64000,181632,true
30,13,10,259.537,4.77778,264.314,32,6.75502e+06,8192,131072,524288,100,6553600,60800,140624,true
31,9,12,1086.02,41.5032,1127.52,32,1.11571e+06,512,4194304,16777216,100,409600,76800,629312,true
31,10,12,1041.34,29.9635,1071.3,32,1.35822e+06,1024,2097152,8388608,100,819200,73600,465424,true
31,11,12,810.923,12.0018,822.925,32,2.04054e+06,2048,1048576,4194304,100,1638400,70400,331744,true
31,12,12,767.306,12.6568,779.962,32,3.59378e+06,4096,524288,2097152,100,3276800,67200,249776,true
31,13,12,797.304,12.0243,809.329,32,6.79923e+06,8192,262144,1048576,100,6553600,64000,181632,true
31,14,12,654.407,29.497,683.904,32,1.33086e+07,16384,131072,524288,100,13107200,60800,140624,true
31,15,12,694.763,18.2625,713.025,32,2.63772e+07,32768,65536,262144,100,26214400,57600,105248,true
32,10,13,1874.2,46.7492,1920.94,32,1.52531e+06,1024,4194304,16777216,100,819200,76800,629312,true
32,11,13,1471.79,59.0058,1530.79,32,2.17742e+06,2048,2097152,8388608,100,1638400,73600,465424,true
32,12,13,1511.64,63.158,1574.79,32,3.67894e+06,4096,1048576,4194304,100,3276800,70400,331744,true
32,13,13,1595.07,61.6231,1656.69,32,6.87058e+06,8192,524288,2097152,100,6553600,67200,249776,true
33,11,13,3288.17,74.706,3362.88,32,2.34451e+06,2048,4194304,16777216,100,1638400,76800,629312,true
33,12,13,3523.65,74.5679,3598.22,32,3.81582e+06,4096,2097152,8388608,100,3276800,73600,465424,true
```

Post-table checks:

```sh
cmake --build build -j
./build/bench pcs_orion_loga_sweep_base 20 22 1 0 0
./build/bench pcs_orion_loga_sweep_base 24 24 1 0 0
./build/bench pcs_orion_loga_sweep_base 27 27 1 0 0
./build/bench pcs_compare_base 20 1
ctest --output-on-failure
```

The default-path checks confirmed `20 -> 6`, `21 -> 7`, `22 -> 5`, `24 -> 7`, and `27 -> 9`, all with `open_ok=true`. The `pcs_compare_base 20 1` run also checked Orion against direct `MLE::open` and returned `open_ok=true`; `ctest` reported no registered tests in this build.

## Lazy PCS Backend Switch

`lazy_pcs_pool` now supports a PCS backend selector:

```cpp
enum class pcs_backend {
    ligero,
    orion
};
```

The default backend is Orion. It can be overridden at runtime with either environment variable:

```sh
ZKCNN_PCS=orion
ZKCNN_PCS=ligero
```

or:

```sh
ZKCNN_PCS_BACKEND=orion
ZKCNN_PCS_BACKEND=ligero
```

All existing `lazy_pcs_pool::create(sec_param, use_ext)` call sites keep working; they now read the default backend. A caller can also pass the backend explicitly through the third `create` argument.

The C++ benchmark entry point again supports the model runner used by `model/benchmark.py`:

```sh
./build/bench <threads> <LeNet|AlexNet|VGG11|VGG16>
```

Small checks after adding the switch:

```sh
cmake --build build -j
./build/bench lazy_pcs_smoke 1
ZKCNN_PCS=ligero ./build/bench lazy_pcs_smoke 1
./build/bench pcs_compare_base 20 1
ctest --output-on-failure
```

The default smoke printed `PCS backend: orion` and completed. The explicit Ligero smoke printed `PCS backend: ligero` and completed. `pcs_compare_base 20 1` still returned `open_ok=true`; `ctest` still has no registered tests.

LeNet smoke:

```sh
python3 -c 'import sys; sys.path.insert(0, "model"); import LeNet, pad_conv; LeNet.train_manual(4, 1); pad_conv.pad_LeNet()'
./build/bench 1 LeNet
```

This ran with the default Orion backend and completed. The trace showed `batch_sz = 4`, `batch_num = 1`, `Final open: num_vars = 23`, `[orion commit]`, and `[orion open]`; proof output ended with `LeNet: 0.76532MB`.

An earlier `batch=1, iters=1` attempt failed before PCS commitment during ReLU pre-processing with `divProver: Remainder out of range`, so it was not useful as a PCS smoke. The script's original smallest batch size is `4`, which is the smoke configuration above.

## Python Benchmark Script Updates

The README benchmark path remains:

```sh
sh ./benchmark.sh
```

`benchmark.sh` now defaults `ZKCNN_PCS=orion`, runs the repeatable CMake build in `setup.sh`, then runs `python3 model/benchmark.py`.

`model/benchmark.py` now:

- passes `ZKCNN_PCS` into the benchmark subprocess;
- defaults to Orion if neither `ZKCNN_PCS` nor `ZKCNN_PCS_BACKEND` is set;
- runs only `threads=1` by default; use `ZKCNN_THREADS=...` to override;
- records results under `./logs/<pcs_backend>` by default;
- parses `PCS backend: ...` from C++ output;
- uses `[orion commit]` and `[orion open]` as top-level PCS timings when backend is Orion;
- keeps `[ligero commit]` and `[ligero open]` as separate columns, since Orion open still uses an internal Ligero commitment for code-switching-lite;
- checks the subprocess return code and treats `terminate called` or `❌` as benchmark failures;
- removes stale traces from `training_trace/<model>` rather than `/training_trace/<model>`.

The default full run is still:

```sh
python3 model/benchmark.py
```

Useful smoke override:

```sh
ZKCNN_MODELS=LeNet ZKCNN_THREADS=1 ZKCNN_BATCH_SIZES=4 ZKCNN_ITERATIONS=1 ZKCNN_BENCH_OUTDIR=./logs/smoke_orion python3 model/benchmark.py
```

That smoke completed with `pcs_backend=orion`, `returncode=0`, `commit_s=2.357`, `open_s=0.197`, and `proof_size_MB=0.76532`.
