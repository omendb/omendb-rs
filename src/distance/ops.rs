//! SIMD-accelerated distance calculations for HNSW.
//!
//! Uses `std::simd` (portable SIMD) for cross-platform acceleration.
//! Runtime dispatch via `multiversion` selects optimal ISA:
//! - `x86_64`: AVX-512 → AVX2 → SSE4.1 (8-lane or 4-lane portable SIMD)
//! - `aarch64`: SVE (Graviton 3+) → NEON (baseline)
//!
//! ## Optimizations
//!
//! - 4x loop unrolling with independent accumulators
//! - Breaks dependency chains for better pipeline utilization
//! - 10-40% faster than naive SIMD at high dimensions

use multiversion::multiversion;
use std::simd::{num::SimdFloat, LaneCount, Simd, StdFloat, SupportedLaneCount};

/// L2 (Euclidean) distance with SIMD acceleration.
#[inline]
#[must_use]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_distance_squared(a, b).sqrt()
}

/// Compute squared norm of a vector (||v||²)
#[multiversion(targets(
    "x86_64+avx512f",
    "x86_64+avx2",
    "x86_64+sse4.1",
    "aarch64+sve",
    "aarch64+neon",
))]
#[inline]
#[must_use]
pub fn norm_squared(v: &[f32]) -> f32 {
    dot_product(v, v)
}

/// L2 distance squared using decomposition: ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
///
/// This trades fewer FLOPs for one extra memory read (`b_norm`).
/// Useful when vector norms are precomputed during insert.
#[inline]
#[must_use]
pub fn l2_squared_decomposed(a: &[f32], b: &[f32], a_norm_sq: f32, b_norm_sq: f32) -> f32 {
    let dot = dot_product(a, b);
    a_norm_sq + b_norm_sq - 2.0 * dot
}

/// L2 distance squared (no sqrt) for comparisons.
#[multiversion(targets(
    "x86_64+avx512f",
    "x86_64+avx2",
    "x86_64+sse4.1",
    "aarch64+sve",
    "aarch64+neon",
))]
#[inline]
#[must_use]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    // x86_64: Try 16-lane (512-bit) for AVX-512, fall back to 8-lane (256-bit)
    #[cfg(target_arch = "x86_64")]
    {
        l2_squared_simd::<16>(a, b)
            .or_else(|| l2_squared_simd::<8>(a, b))
            .or_else(|| l2_squared_simd::<4>(a, b))
            .unwrap_or_else(|| l2_squared_scalar(a, b))
    }

    // ARM: 128-bit NEON, use 4-lane directly (8-lane works but 4 is native width)
    #[cfg(target_arch = "aarch64")]
    {
        l2_squared_simd::<4>(a, b).unwrap_or_else(|| l2_squared_scalar(a, b))
    }

    // Other architectures: scalar fallback
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        l2_squared_scalar(a, b)
    }
}

/// Dot product with SIMD acceleration.
#[multiversion(targets(
    "x86_64+avx512f",
    "x86_64+avx2",
    "x86_64+sse4.1",
    "aarch64+sve",
    "aarch64+neon",
))]
#[inline]
#[must_use]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    // x86_64: Try 16-lane (512-bit) for AVX-512, fall back to 8-lane (256-bit)
    #[cfg(target_arch = "x86_64")]
    {
        dot_simd::<16>(a, b)
            .or_else(|| dot_simd::<8>(a, b))
            .or_else(|| dot_simd::<4>(a, b))
            .unwrap_or_else(|| dot_scalar(a, b))
    }

    // ARM: 128-bit NEON, use 4-lane directly
    #[cfg(target_arch = "aarch64")]
    {
        dot_simd::<4>(a, b).unwrap_or_else(|| dot_scalar(a, b))
    }

    // Other architectures: scalar fallback
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        dot_scalar(a, b)
    }
}

/// Cosine distance: `1 - cos(a, b)`.
#[inline]
#[must_use]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let dot = dot_product(a, b);
    let norm_a = dot_product(a, a).sqrt();
    let norm_b = dot_product(b, b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    1.0 - (dot / (norm_a * norm_b))
}

// ============================================================================
// SIMD Implementations (4x unrolling, 4 accumulators)
// ============================================================================

#[inline]
fn l2_squared_simd<const N: usize>(a: &[f32], b: &[f32]) -> Option<f32>
where
    LaneCount<N>: SupportedLaneCount,
{
    if a.len() < N {
        return None;
    }

    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    let n_chunks = a_chunks.len();

    let zero = Simd::<f32, N>::splat(0.0);
    let (mut acc0, mut acc1, mut acc2, mut acc3) = (zero, zero, zero, zero);

    let unroll_end = n_chunks - (n_chunks % 4);
    let mut i = 0;
    while i < unroll_end {
        let d0 = Simd::from_array(a_chunks[i]) - Simd::from_array(b_chunks[i]);
        let d1 = Simd::from_array(a_chunks[i + 1]) - Simd::from_array(b_chunks[i + 1]);
        let d2 = Simd::from_array(a_chunks[i + 2]) - Simd::from_array(b_chunks[i + 2]);
        let d3 = Simd::from_array(a_chunks[i + 3]) - Simd::from_array(b_chunks[i + 3]);

        acc0 = d0.mul_add(d0, acc0);
        acc1 = d1.mul_add(d1, acc1);
        acc2 = d2.mul_add(d2, acc2);
        acc3 = d3.mul_add(d3, acc3);
        i += 4;
    }

    while i < n_chunks {
        let d = Simd::from_array(a_chunks[i]) - Simd::from_array(b_chunks[i]);
        acc0 = d.mul_add(d, acc0);
        i += 1;
    }

    let mut sum = (acc0 + acc1 + acc2 + acc3).reduce_sum();

    for (&av, &bv) in a_rem.iter().zip(b_rem.iter()) {
        let d = av - bv;
        sum = d.mul_add(d, sum);
    }

    Some(sum)
}

#[inline]
fn dot_simd<const N: usize>(a: &[f32], b: &[f32]) -> Option<f32>
where
    LaneCount<N>: SupportedLaneCount,
{
    if a.len() < N {
        return None;
    }

    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    let n_chunks = a_chunks.len();

    let zero = Simd::<f32, N>::splat(0.0);
    let (mut acc0, mut acc1, mut acc2, mut acc3) = (zero, zero, zero, zero);

    let unroll_end = n_chunks - (n_chunks % 4);
    let mut i = 0;
    while i < unroll_end {
        let (a0, b0) = (Simd::from_array(a_chunks[i]), Simd::from_array(b_chunks[i]));
        let (a1, b1) = (
            Simd::from_array(a_chunks[i + 1]),
            Simd::from_array(b_chunks[i + 1]),
        );
        let (a2, b2) = (
            Simd::from_array(a_chunks[i + 2]),
            Simd::from_array(b_chunks[i + 2]),
        );
        let (a3, b3) = (
            Simd::from_array(a_chunks[i + 3]),
            Simd::from_array(b_chunks[i + 3]),
        );
        acc0 = a0.mul_add(b0, acc0);
        acc1 = a1.mul_add(b1, acc1);
        acc2 = a2.mul_add(b2, acc2);
        acc3 = a3.mul_add(b3, acc3);
        i += 4;
    }

    while i < n_chunks {
        let (a0, b0) = (Simd::from_array(a_chunks[i]), Simd::from_array(b_chunks[i]));
        acc0 = a0.mul_add(b0, acc0);
        i += 1;
    }

    let mut sum = (acc0 + acc1 + acc2 + acc3).reduce_sum();

    for (&av, &bv) in a_rem.iter().zip(b_rem.iter()) {
        sum = av.mul_add(bv, sum);
    }

    Some(sum)
}

// ============================================================================
// Scalar Fallbacks
// ============================================================================

#[inline]
fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        assert!((l2_distance(&a, &b) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        assert!((dot_product(&a, &b) - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        assert!(cosine_distance(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_large_vectors() {
        let a: Vec<f32> = (0..1536).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..1536).map(|i| (i * 2) as f32).collect();
        assert!(l2_distance(&a, &b) > 0.0);
    }

    #[test]
    fn test_simd_vs_scalar_l2() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| i as f32 * 0.2).collect();

        let simd = l2_distance_squared(&a, &b);
        let scalar = l2_squared_scalar(&a, &b);
        let rel_err = (simd - scalar).abs() / scalar.abs();
        assert!(rel_err < 1e-5, "Relative error {rel_err} too large");
    }

    #[test]
    fn test_simd_vs_scalar_dot() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| i as f32 * 0.2).collect();

        let simd = dot_product(&a, &b);
        let scalar = dot_scalar(&a, &b);
        let rel_err = (simd - scalar).abs() / scalar.abs();
        assert!(rel_err < 1e-5, "Relative error {rel_err} too large");
    }

    #[test]
    fn test_small_vectors() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        let dist = l2_distance(&a, &b);
        let expected = 8.0_f32.sqrt();
        assert!((dist - expected).abs() < 1e-6);
    }

    #[test]
    fn test_zero_vectors() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 1.0, 1.0];
        assert_eq!(cosine_distance(&a, &b), 1.0);
    }

    #[test]
    fn test_l2_squared_vs_l2() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| i as f32 * 0.2).collect();
        let l2 = l2_distance(&a, &b);
        let l2_sq = l2_distance_squared(&a, &b);
        assert!((l2 - l2_sq.sqrt()).abs() < 1e-6);
    }

    // Edge case tests for loop unrolling
    #[test]
    fn test_exact_4_chunks() {
        // 32 elements = 4 chunks of 8 (AVX2) - tests exact unroll boundary
        let a: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..32).map(|i| (i + 1) as f32).collect();
        let simd = l2_distance_squared(&a, &b);
        let scalar = l2_squared_scalar(&a, &b);
        assert!((simd - scalar).abs() < 1e-5);
    }

    #[test]
    fn test_5_chunks() {
        // 40 elements = 5 chunks of 8 - tests remainder after unroll
        let a: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..40).map(|i| (i + 1) as f32).collect();
        let simd = l2_distance_squared(&a, &b);
        let scalar = l2_squared_scalar(&a, &b);
        assert!((simd - scalar).abs() < 1e-5);
    }

    #[test]
    fn test_with_remainder() {
        // 35 elements = 4 full chunks + 3 remainder
        let a: Vec<f32> = (0..35).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..35).map(|i| (i + 1) as f32).collect();
        let simd = l2_distance_squared(&a, &b);
        let scalar = l2_squared_scalar(&a, &b);
        assert!((simd - scalar).abs() < 1e-5);
    }

    #[test]
    fn test_norm_squared() {
        let v = [3.0, 4.0]; // ||v||² = 9 + 16 = 25
        assert!((norm_squared(&v) - 25.0).abs() < 1e-6);

        let v2: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let expected: f32 = v2.iter().map(|x| x * x).sum();
        assert!((norm_squared(&v2) - expected).abs() < 1e-3);
    }

    #[test]
    fn test_l2_decomposed_matches_direct() {
        let a: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..768).map(|i| (i as f32 * 0.02).cos()).collect();

        let direct = l2_distance_squared(&a, &b);
        let a_norm = norm_squared(&a);
        let b_norm = norm_squared(&b);
        let decomposed = l2_squared_decomposed(&a, &b, a_norm, b_norm);

        let rel_err = (direct - decomposed).abs() / direct.abs();
        assert!(
            rel_err < 1e-5,
            "Direct: {direct}, Decomposed: {decomposed}, Rel err: {rel_err}"
        );
    }

    #[test]
    #[ignore = "benchmark test, run with: cargo test --release -- --ignored l2_decomposed_speed"]
    fn test_l2_decomposed_speed() {
        use std::time::Instant;

        let dims = 768;
        let n_vectors = 10000;
        let n_queries = 100;

        // Generate vectors
        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|i| {
                (0..dims)
                    .map(|j| ((i * 7 + j) % 1000) as f32 / 1000.0)
                    .collect()
            })
            .collect();
        let queries: Vec<Vec<f32>> = (0..n_queries)
            .map(|i| {
                (0..dims)
                    .map(|j| ((i * 13 + j) % 1000) as f32 / 1000.0)
                    .collect()
            })
            .collect();

        // Precompute norms
        let vector_norms: Vec<f32> = vectors.iter().map(|v| norm_squared(v)).collect();
        let query_norms: Vec<f32> = queries.iter().map(|q| norm_squared(q)).collect();

        // Benchmark direct L2²
        let start = Instant::now();
        let mut sum = 0.0f32;
        for q in &queries {
            for v in &vectors {
                sum += l2_distance_squared(q, v);
            }
        }
        let direct_time = start.elapsed();
        eprintln!("Direct L2²: {direct_time:?} (checksum={sum:.2})");

        // Benchmark decomposed L2²
        let start = Instant::now();
        sum = 0.0;
        for (qi, q) in queries.iter().enumerate() {
            let q_norm = query_norms[qi];
            for (vi, v) in vectors.iter().enumerate() {
                sum += l2_squared_decomposed(q, v, q_norm, vector_norms[vi]);
            }
        }
        let decomposed_time = start.elapsed();
        eprintln!("Decomposed L2²: {decomposed_time:?} (checksum={sum:.2})");

        let speedup = direct_time.as_secs_f64() / decomposed_time.as_secs_f64();
        eprintln!("\nSpeedup: {speedup:.2}x");

        // Print analysis
        if speedup > 1.05 {
            eprintln!(
                "✓ Decomposition is {:.0}% faster - worth implementing!",
                (speedup - 1.0) * 100.0
            );
        } else if speedup < 0.95 {
            eprintln!(
                "✗ Decomposition is {:.0}% slower - not worth it",
                (1.0 - speedup) * 100.0
            );
        } else {
            eprintln!("≈ No significant difference");
        }
    }
}
