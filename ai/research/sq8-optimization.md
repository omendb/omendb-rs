# SQ8 Optimization Research

**Date:** January 3, 2026
**Status:** Recall fixed, performance analyzed - memory savings confirmed, speed parity with FP32

## Problem Summary

SQ8 had a recall regression from 99.8% to 88.9% on SIFT-50K. Root cause was L2 decomposition causing numerical precision issues (catastrophic cancellation).

## Fix Applied (commit 3478a4f)

Reverted SQ8 from L2 decomposition path back to asymmetric path:

```rust
// storage.rs - supports_l2_decomposition()
// Changed from including ScalarQuantized to excluding it
matches!(self, Self::FullPrecision { .. })  // SQ8 excluded
```

```rust
// search.rs - removed SQ8Accessor from search hot path
// SQ8 now uses distance_cmp_mono → distance_asymmetric_l2
```

## Current Performance (after fix)

| Metric    | L2 (FP32) | SQ8   | Ratio      |
| --------- | --------- | ----- | ---------- |
| Recall@10 | 99.9%     | 99.8% | -          |
| QPS (50K) | ~5400     | ~5400 | ~99%       |
| Memory    | 4x        | 1x    | 4x savings |

This matches 0.0.20 behavior where SQ8 ≈ FP32 speed.

## Benchmark Results (January 3, 2026)

### Raw Distance Computation (768D, 1000 vectors, Apple M3 Max)

| Method                | Time      | vs FP32 SIMD    |
| --------------------- | --------- | --------------- |
| FP32 L2 SIMD          | 57 µs     | baseline        |
| FP32 L2 Decomposed    | 57 µs     | 1.0x            |
| **SQ8 Asymmetric L2** | **93 µs** | **1.6x slower** |
| SQ8 ADC Table         | 496 µs    | 8.7x slower     |

**Key finding:** SQ8 distance is actually 1.6x _slower_ than FP32 due to dequantization overhead.

### Why HNSW Search Shows Parity

HNSW search involves:

- Graph traversal (neighbor list fetching)
- Heap operations (candidate management)
- Visited set updates
- Distance computation (only ~30% of time)

The ~1.6x slower distance is diluted by other overhead, resulting in similar QPS.

## Why SQ8 Is Slower Than FP32

### Operation Count (per 8 dimensions SIMD)

**FP32 L2 SIMD:**

1. Load 8 query f32 (1 op)
2. Load 8 vector f32 (1 op)
3. Subtract (1 op)
4. FMA for diff² (1 op)
   = **4 ops per 8 dims = 0.5 ops/dim**

**SQ8 Asymmetric L2:**

1. Load 8 u8 + convert to i32 (2 ops)
2. Convert i32 to f32 (1 op)
3. Load 8 scales (1 op)
4. Load 8 mins (1 op)
5. FMA for dequant (1 op)
6. Load 8 query (1 op)
7. Subtract (1 op)
8. FMA for diff² (1 op)
   = **9 ops per 8 dims = 1.125 ops/dim**

SQ8 has **2.25x more operations** than FP32. Memory savings (4x less data) only help when data is cache-cold.

## Why Integer SIMD Won't Help (As Designed)

The research papers (Qdrant, Faiss) achieve 2-4x speedup with integer SIMD by using **uniform quantization**:

- Single `scale` and `offset` for entire vector
- Allows: `dot(q_int, v_int)` with integer SIMD (32 ops at once)

Our implementation uses **per-dimension quantization**:

- `scales[d]` and `mins[d]` for each dimension
- Blocks integer SIMD (can't factor out per-dimension params)
- Better recall but no speed benefit

### What Would Be Needed

| Approach          | Speed     | Recall | Changes Required   |
| ----------------- | --------- | ------ | ------------------ |
| Current (per-dim) | 0.6x FP32 | 99%+   | None               |
| Uniform quant     | 2-4x FP32 | ~97%   | New storage format |
| ADC Tables        | 0.1x FP32 | 99%+   | None (too slow)    |

## Conclusion

**SQ8's value proposition is memory, not speed:**

- 4x memory reduction
- 99%+ recall
- Speed parity with FP32 (not faster)

This is still valuable for:

1. Large indices that don't fit in RAM
2. Cold tier storage (S3/GCS)
3. Memory-constrained environments

## Future Options (if speed needed)

1. **Uniform quantization mode**: Add optional single scale/min - enables integer SIMD
2. **RaBitQ (4-bit)**: Already implemented, faster on Apple Silicon
3. **Batched distance**: Process multiple candidates together for better SIMD utilization

## Key Files

- `src/compression/scalar.rs` - ScalarParams, quantization
- `src/vector/hnsw/storage.rs` - VectorStorage, distance_asymmetric_l2
- `src/vector/hnsw/index/search.rs` - search_layer_mono
- `src/distance/ops.rs` - SIMD distance functions
- `benches/sq8_bench.rs` - Benchmark comparing SQ8 vs FP32 distance

## References

- [Qdrant Scalar Quantization](https://qdrant.tech/articles/scalar-quantization)
- [Faiss FastScan](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes)
- [Elastic OSQ](https://www.elastic.co/search-labs/blog/scalar-quantization-optimization)
- Research file: `ai/research/lsm-vec/sq8-performance-research.md`

## Status

**Complete.** Analysis shows SQ8's value is memory reduction (4x), not speed. Current implementation provides:

- 99%+ recall
- Speed parity with FP32 (slightly slower raw distance, but similar HNSW QPS)
- 4x memory savings

Speed optimization would require architectural changes (uniform quantization) which trades recall for speed. Not recommended for current use case.
