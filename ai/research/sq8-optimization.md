# SQ8 Optimization Research

**Date:** January 3, 2026
**Status:** Uniform quantization implemented - 2.4-3.1x faster than FP32

## Problem Summary

SQ8 had a recall regression from 99.8% to 88.9% on SIFT-50K. Root cause was L2 decomposition causing numerical precision issues (catastrophic cancellation).

## Fix Applied (commit 3478a4f)

Reverted SQ8 from L2 decomposition path back to asymmetric path:

```rust
// storage.rs - supports_l2_decomposition()
// Changed from including ScalarQuantized to excluding it
matches!(self, Self::FullPrecision { .. })  // SQ8 excluded
```

## Uniform Quantization Implementation

Added `UniformScalarParams` for high-performance SQ8:

- Single `scale` and `offset` for entire vector (vs per-dimension)
- Enables integer SIMD (u8×u8 dot product)
- 4x loop unrolling with `vpadalq_u16` (pairwise add-accumulate)

### Key Types

```rust
pub struct UniformScalarParams {
    pub scale: f32,      // Global: (max - min) / 255
    pub offset: f32,     // Global minimum
    pub dimensions: usize,
}

pub struct UniformQuantizedVector {
    pub data: Vec<u8>,   // Quantized values
    pub sum: i32,        // Precomputed: Σ data[i]
    pub norm_sq: f32,    // Precomputed: ||v||²
}

pub struct UniformQueryPrep {
    pub quantized: Vec<u8>,  // Quantized query
    pub norm_sq: f32,        // ||q||²
    pub sum: i32,            // Σ q[i]
}
```

### Distance Formula

L2² = ||q||² + ||v||² - 2⟨q,v⟩

Where dot product is reconstructed from integer SIMD:

```
dot(q,v) = scale² × int_dot(q_int, v_int)
         + scale × offset × (Σq_int + Σv_int)
         + offset² × dim
```

## Benchmark Results (January 3, 2026, Apple M3 Max)

### Raw Distance Computation (1000 vectors)

| Method                     | 128D       | 768D        | vs FP32 (768D)  |
| -------------------------- | ---------- | ----------- | --------------- |
| FP32 L2 SIMD               | 10.9 µs    | 57.3 µs     | baseline        |
| SQ8 Asymmetric (per-dim)   | 14.5 µs    | 93.6 µs     | 1.6x slower     |
| **Uniform SQ8 (int SIMD)** | **4.5 µs** | **18.2 µs** | **3.1x faster** |
| SQ8 ADC Table              | 53 µs      | 488 µs      | 8.5x slower     |

### Speedup Summary

| Dimension | Speedup vs FP32 |
| --------- | --------------- |
| 128D      | 2.4x            |
| 768D      | 3.1x            |

### Accuracy

- Distance error: ~2% relative error
- Recall: ~97-98% (vs 99%+ for per-dimension)
- Ordering preserved for nearest neighbor search

## Implementation Details

### NEON Optimization (aarch64)

```rust
// 4x unrolled loop, processes 64 u8 elements per iteration
while i + 64 <= query.len() {
    // Load 16 bytes, widening multiply, pairwise accumulate
    let q = vld1q_u8(query.as_ptr().add(i));
    let v = vld1q_u8(vec.as_ptr().add(i));
    let prod_lo = vmull_u8(vget_low_u8(q), vget_low_u8(v));
    let prod_hi = vmull_high_u8(q, v);
    sum = vpadalq_u16(sum, prod_lo);  // Pairwise add u16→u32
    sum = vpadalq_u16(sum, prod_hi);
    // ... 3 more blocks for ILP
}
```

### AVX2 Optimization (x86_64)

```rust
// Process 32 bytes at a time
while i + 32 <= query.len() {
    let q = _mm256_loadu_si256(...);
    let v = _mm256_loadu_si256(...);
    // Extend to i16 and use madd for pairwise multiply-add
    let q_lo = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(q, 0));
    let v_lo = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v, 0));
    let prod = _mm256_madd_epi16(q_lo, v_lo);
    sum = _mm256_add_epi32(sum, prod);
}
```

## When to Use Each Mode

| Mode        | Speed     | Recall | Use Case                           |
| ----------- | --------- | ------ | ---------------------------------- |
| Uniform SQ8 | 2-3x FP32 | 97%    | High-throughput, large scale       |
| Per-dim SQ8 | 0.6x FP32 | 99%+   | Precision-critical, moderate scale |
| FP32        | 1x        | 100%   | Maximum accuracy                   |

## Key Files

- `src/compression/scalar.rs` - `UniformScalarParams`, SIMD kernels
- `benches/sq8_bench.rs` - Benchmark suite

## Next Steps

1. Integrate `UniformScalarParams` with `VectorStorage` enum
2. Add compression tier option for uniform vs per-dimension
3. Benchmark recall on SIFT-1M and real embedding data

## References

- [Qdrant Scalar Quantization](https://qdrant.tech/articles/scalar-quantization)
- [Faiss FastScan](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes)
- [Elastic OSQ](https://www.elastic.co/search-labs/blog/scalar-quantization-optimization)
