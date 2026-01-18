# OmenDB Codebase Review Findings (2026-01-18)

## Overview
A systematic review of the OmenDB codebase was performed to identify stale references to removed features (`RaBitQ`, `FastScan`, `search_asymmetric`), dead code, and logic inconsistencies.

## Stale References & Removed Features

### RaBitQ / Compression
FILE: `src/types.rs:14-17`
TYPE: `stale_ref`
ISSUE: `CompressionTier` enum still contains `RaBitQ4Bit` and `RaBitQ2Bit` variants.
SUGGESTED_FIX: Remove `RaBitQ4Bit` and `RaBitQ2Bit` from `CompressionTier`.

FILE: `src/types.rs:61-68`
TYPE: `stale_ref`
ISSUE: `StorageTier` enum variants L3-L6 mention `RaBitQ` in doc comments and implementation.
SUGGESTED_FIX: Update `StorageTier` to reflect current quantization support (SQ8).

FILE: `src/omen/header.rs:24-26`
TYPE: `stale_ref`
ISSUE: `QuantizationCode` enum contains `RaBitQ` and `RaBitQ4`.
SUGGESTED_FIX: Keep for format compatibility if needed, but ensure they are clearly marked as legacy/unsupported (currently handled in `to_runtime`).

FILE: `README.md:21`
TYPE: `doc_mismatch`
ISSUE: Mentions "4-8x smaller indexes". 8x refers to removed RaBitQ 4-bit.
SUGGESTED_FIX: Change to "4x smaller indexes" to match SQ8.

### FastScan
FILE: `src/vector/hnsw/storage.rs:1371, 1377, 1415`
TYPE: `stale_ref`
ISSUE: Doc comments still reference "FastScan".
SUGGESTED_FIX: Remove mentions of FastScan from comments.

### Search API
FILE: `src/vector/hnsw_index.rs:513`
TYPE: `naming`
ISSUE: `search_asymmetric_ef` is defined but just wraps `search()`. The specific "search_asymmetric" feature was reportedly removed.
SUGGESTED_FIX: Rename to `search_ef` or similar to match the removal of the specific "search_asymmetric" feature name.

FILE: `src/vector/store/mod.rs:1851, 1881`
TYPE: `stale_ref`
ISSUE: Calls `index.search_asymmetric_ef`.
SUGGESTED_FIX: Update to new name once `hnsw_index.rs` is updated.

## Broken / Dead Code

FILE: `benches/adc_bench.rs`
TYPE: `dead_code`
ISSUE: Entire file is broken as it imports and uses removed `RaBitQ` from `omendb::compression`.
SUGGESTED_FIX: Delete this file or update it to use SQ8.

FILE: `benches/fastscan_bench.rs`
TYPE: `dead_code`
ISSUE: Entire file is dedicated to benchmarking the removed "FastScan" feature.
SUGGESTED_FIX: Delete this file.

FILE: `fuzz/fuzz_targets/quantized_ops.rs:5, 185, 193, 270-271`
TYPE: `dead_code`
ISSUE: Fuzz target references and attempts to test removed `RaBitQ` quantization.
SUGGESTED_FIX: Remove RaBitQ-related tests and functions from the fuzz target.

## Logic & Consistency

FILE: `src/vector/store/mod.rs:105-110`
TYPE: `bug`
ISSUE: `quantization_mode_from_id` uses hardcoded u64 `1` for SQ8 and defaults to `None` for everything else, instead of using `QuantizationCode` from `header.rs`.
SUGGESTED_FIX: Use `QuantizationCode` enum for mode mapping.

FILE: `src/omen/wal.rs:52`
TYPE: `bug`
ISSUE: `WalEntryType::from` treats unknown entry types as `Checkpoint`.
SUGGESTED_FIX: Should return an error or skip unknown entries instead of assuming they are checkpoints.

FILE: `src/vector/hnsw/index/search.rs:31-32`
TYPE: `other`
ISSUE: Redundant check in `DistanceContext::new`. `index.supports_l2_decomposition()` already checks the distance function.
SUGGESTED_FIX: Simplify to `let use_l2_decomposition = index.supports_l2_decomposition();`.

FILE: `src/vector/hnsw/types.rs:184`
TYPE: `other`
ISSUE: `HNSWNode` `neighbor_counts` is fixed at 8. If `max_levels` exceeds 8, this will overflow.
SUGGESTED_FIX: Add a check in `HNSWParams::validate` to ensure `max_level <= 8`.

FILE: `src/vector/store/mod.rs:256`
TYPE: `naming`
ISSUE: `hnsw_index` field is public in `VectorStore` struct.
SUGGESTED_FIX: Make `hnsw_index` private.

## Documentation

FILE: `src/vector/hnsw_index.rs:12`
TYPE: `doc_mismatch`
ISSUE: Doc comment mentions "Optional binary quantization (32x memory reduction)" which refers to the removed RaBitQ.
SUGGESTED_FIX: Update to mention SQ8 scalar quantization (4x memory reduction).

FILE: `src/compression/scalar.rs:12`
TYPE: `doc_mismatch`
ISSUE: Mentions ~97% recall, while other files mention ~99% recall for SQ8.
SUGGESTED_FIX: Harmonize recall expectations (99% is the target for the current SIMD SQ8).
