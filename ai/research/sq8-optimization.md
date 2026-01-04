# SQ8 Optimization Research

**Date:** January 3, 2026
**Status:** Recall fixed, performance optimization pending

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

## Why SQ8 Isn't Faster (Yet)

### Current Implementation (Naive)

```
Load u8 → Dequantize to f32 → Compute f32 distance
```

Code path: `distance_asymmetric_l2` → `params.asymmetric_l2_squared`

```rust
// For each dimension:
let dequant = f32::from(quantized[i]) * scales[i] + mins[i];
let diff = query[i] - dequant;
sum += diff * diff;
```

**Result:** 4x memory savings, but NO speed benefit (same FLOPs as FP32)

### Optimal Implementation (Compressed Domain)

From Qdrant/Faiss research, SQ8 should be 2-4x faster:

```
Load u8 → Compute int8 dot product (SIMD) → Apply precomputed correction
```

Key insight from Qdrant:

```
dot(q, v) = scale² × int8_dot(q_int, v_int) + [precomputed terms]
```

**Why faster:**

- AVX2: 32 int8 ops vs 8 float32 ops (4x parallelism)
- Precompute per-vector: `sum(int8)`, `sum_squared`, corrections
- Only integer arithmetic in hot path

### Expected Performance (Optimized)

| Implementation       | Memory | Speed vs FP32 | Recall |
| -------------------- | ------ | ------------- | ------ |
| Current (naive)      | 4x ↓   | ~1x           | 99%    |
| Optimized (int SIMD) | 4x ↓   | 2-4x faster   | 99%    |

## Implementation Plan

### Phase 1: Integer SIMD Distance

Implement compressed-domain L2 computation:

```rust
pub struct SQ8Vector {
    data: Vec<i8>,           // Quantized to signed int8
    sum: i32,                // Precomputed: Σ data[i]
    sum_sq: i32,             // Precomputed: Σ data[i]²
}

fn l2_squared_int(query: &[f32], vec: &SQ8Vector, params: &SQ8Params) -> f32 {
    // Hot path: integer dot product with SIMD
    let int_dot = simd_int8_dot(query_quantized, &vec.data);

    // Apply corrections (precomputed terms)
    let scale_sq = params.scale * params.scale;
    scale_sq * (query_sum_sq + vec.sum_sq - 2 * int_dot) as f32
        + correction_terms
}
```

### Phase 2: SIMD Kernels

**AVX2 (x86_64):**

```rust
// Process 32 int8 values at once
let a = _mm256_loadu_si256(query.as_ptr());
let b = _mm256_loadu_si256(vec.as_ptr());
let prod = _mm256_maddubs_epi16(a, b);  // 16-bit products
let sum = _mm256_madd_epi16(prod, ones); // 32-bit accumulate
```

**NEON (aarch64):**

```rust
// Process 16 int8 values at once
let a = vld1q_s8(query.as_ptr());
let b = vld1q_s8(vec.as_ptr());
// Use vdotq_s32 on newer ARM (ARMv8.2+)
// or vmull + vaddl on older
```

### Phase 3: Storage Changes

Add precomputed fields to `ScalarQuantized`:

```rust
Self::ScalarQuantized {
    quantized: Vec<i8>,      // Changed from u8 to i8
    sums: Vec<i32>,          // Precomputed Σ quantized[i]
    sum_sqs: Vec<i32>,       // Precomputed Σ quantized[i]²
    // ... existing fields
}
```

### Phase 4: Benchmark Validation

Target metrics:

- Recall: ≥99% on SIFT-50K
- QPS: ≥2x FP32 baseline
- Memory: 4x reduction maintained

## Key Files

- `src/compression/scalar.rs` - ScalarParams, quantization
- `src/vector/hnsw/storage.rs` - VectorStorage, distance_asymmetric_l2
- `src/vector/hnsw/index/search.rs` - search_layer_mono
- `src/distance/ops.rs` - SIMD distance functions

## References

- [Qdrant Scalar Quantization](https://qdrant.tech/articles/scalar-quantization)
- [Faiss FastScan](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes)
- [Elastic OSQ](https://www.elastic.co/search-labs/blog/scalar-quantization-optimization)
- Research file: `ai/research/lsm-vec/sq8-performance-research.md`

## Next Steps

1. Create bead for SQ8 integer SIMD optimization
2. Implement Phase 1 (integer dot product)
3. Benchmark on SIFT-50K
4. Iterate on SIMD kernels
