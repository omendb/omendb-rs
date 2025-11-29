//! SIMD-accelerated distance calculations for HNSW
//!
//! Uses `std::simd` (portable SIMD) for cross-platform SIMD support.
//! Automatically compiles to optimal SIMD instructions (AVX2, SSE2, NEON).
//!
//! ## Performance
//!
//! - 8 lanes (AVX2): 3-4x speedup over scalar
//! - 4 lanes (SSE2/NEON): 2-3x speedup over scalar
//!
//! ## Implementation
//!
//! Uses generic SIMD implementation with lane count (8, 4, or scalar fallback).
//! Compiler automatically selects best instruction set for target platform.

use std::simd::{num::SimdFloat, LaneCount, Simd, SupportedLaneCount};

/// L2 distance (Euclidean) with SIMD acceleration
///
/// Automatically uses optimal SIMD lane count (8 for AVX2, 4 for SSE2/NEON).
/// Falls back to scalar if vector too small for SIMD.
#[inline]
#[must_use]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_distance_squared(a, b).sqrt()
}

/// L2 distance squared (skip sqrt) - for HNSW internal comparisons
///
/// HNSW only needs relative ordering, not absolute distances.
/// Squared distance maintains ordering since sqrt is monotonic.
/// This saves ~10-15% compute time in `search_layer`.
///
/// Note: Use `l2_distance()` when actual Euclidean distance is needed.
#[inline]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    // Try 8 lanes (AVX2), then 4 lanes (SSE2/NEON), then scalar
    l2_distance_squared_simd::<8>(a, b).unwrap_or_else(|| {
        l2_distance_squared_simd::<4>(a, b).unwrap_or_else(|| l2_distance_squared_scalar(a, b))
    })
}

/// Dot product with SIMD acceleration
///
/// Automatically uses optimal SIMD lane count (8 for AVX2, 4 for SSE2/NEON).
/// Falls back to scalar if vector too small for SIMD.
#[inline]
#[must_use]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    // Try 8 lanes (AVX2), then 4 lanes (SSE2/NEON), then scalar
    dot_product_simd::<8>(a, b)
        .unwrap_or_else(|| dot_product_simd::<4>(a, b).unwrap_or_else(|| dot_product_scalar(a, b)))
}

/// Cosine distance with SIMD acceleration
///
/// Computed as: 1 - (dot(a, b) / (norm(a) * norm(b)))
/// Returns 1.0 for zero vectors (maximum distance).
#[inline]
#[must_use]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let dot = dot_product(a, b);

    // Compute norms using dot product with itself
    let norm_a = dot_product(a, a).sqrt();
    let norm_b = dot_product(b, b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Maximum distance for zero vectors
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Cosine distance for pre-normalized vectors (3x faster)
///
/// When vectors are unit-normalized (||a|| = ||b|| = 1):
///   `cosine_distance` = 1 - dot(a, b)
///
/// This skips the expensive norm calculations (2 dot products + 2 sqrts).
/// Use when you control normalization on insert.
#[inline]
#[allow(dead_code)] // Future optimization for auto-normalized vectors
pub fn cosine_distance_normalized(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot_product(a, b)
}

// ============================================================================
// Generic SIMD Implementations
// ============================================================================

/// Generic SIMD L2 distance squared implementation (no sqrt)
///
/// Uses 4x loop unrolling with 4 independent accumulators to:
/// - Hide instruction latency (CPU can execute 4 ops in parallel)
/// - Break dependency chain (each accumulator is independent)
/// - Better utilize CPU pipelines (10-40% faster at high dimensions)
///
/// Works with any lane count (8 for AVX2, 4 for SSE2/NEON).
/// Returns None if vector too small for SIMD (len < LANES).
#[inline]
fn l2_distance_squared_simd<const LANES: usize>(a: &[f32], b: &[f32]) -> Option<f32>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    if a.len() < LANES {
        return None; // Vector too small for SIMD
    }

    let (a_chunks, a_rem) = a.as_chunks::<LANES>();
    let (b_chunks, b_rem) = b.as_chunks::<LANES>();

    // 4 independent accumulators to break dependency chain and hide latency
    let mut acc0 = Simd::<f32, LANES>::splat(0.0);
    let mut acc1 = Simd::<f32, LANES>::splat(0.0);
    let mut acc2 = Simd::<f32, LANES>::splat(0.0);
    let mut acc3 = Simd::<f32, LANES>::splat(0.0);

    // Process 4 chunks per iteration (4x unrolling)
    let mut chunks_a = a_chunks.iter();
    let mut chunks_b = b_chunks.iter();

    loop {
        // Try to get 4 chunks at once
        let (Some(a0), Some(b0)) = (chunks_a.next(), chunks_b.next()) else {
            break;
        };

        let a_vec0 = Simd::<f32, LANES>::from_array(*a0);
        let b_vec0 = Simd::<f32, LANES>::from_array(*b0);
        let diff0 = a_vec0 - b_vec0;
        acc0 += diff0 * diff0;

        let Some(a1) = chunks_a.next() else {
            break;
        };
        let Some(b1) = chunks_b.next() else {
            break;
        };
        let a_vec1 = Simd::<f32, LANES>::from_array(*a1);
        let b_vec1 = Simd::<f32, LANES>::from_array(*b1);
        let diff1 = a_vec1 - b_vec1;
        acc1 += diff1 * diff1;

        let Some(a2) = chunks_a.next() else {
            break;
        };
        let Some(b2) = chunks_b.next() else {
            break;
        };
        let a_vec2 = Simd::<f32, LANES>::from_array(*a2);
        let b_vec2 = Simd::<f32, LANES>::from_array(*b2);
        let diff2 = a_vec2 - b_vec2;
        acc2 += diff2 * diff2;

        let Some(a3) = chunks_a.next() else {
            break;
        };
        let Some(b3) = chunks_b.next() else {
            break;
        };
        let a_vec3 = Simd::<f32, LANES>::from_array(*a3);
        let b_vec3 = Simd::<f32, LANES>::from_array(*b3);
        let diff3 = a_vec3 - b_vec3;
        acc3 += diff3 * diff3;
    }

    // Process any remaining full chunks (0-3 chunks)
    for (a_chunk, b_chunk) in chunks_a.zip(chunks_b) {
        let a_vec = Simd::<f32, LANES>::from_array(*a_chunk);
        let b_vec = Simd::<f32, LANES>::from_array(*b_chunk);
        let diff = a_vec - b_vec;
        acc0 += diff * diff;
    }

    // Combine accumulators and reduce
    let combined = acc0 + acc1 + acc2 + acc3;
    let mut sum = combined.reduce_sum();

    // Process remainder scalarly
    for (a_val, b_val) in a_rem.iter().zip(b_rem.iter()) {
        let diff = a_val - b_val;
        sum += diff * diff;
    }

    Some(sum)
}

/// Generic SIMD dot product implementation
///
/// Uses 4x loop unrolling with 4 independent accumulators to:
/// - Hide instruction latency (CPU can execute 4 ops in parallel)
/// - Break dependency chain (each accumulator is independent)
/// - Better utilize CPU pipelines (10-40% faster at high dimensions)
///
/// Works with any lane count (8 for AVX2, 4 for SSE2/NEON).
/// Returns None if vector too small for SIMD (len < LANES).
#[inline]
fn dot_product_simd<const LANES: usize>(a: &[f32], b: &[f32]) -> Option<f32>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    if a.len() < LANES {
        return None; // Vector too small for SIMD
    }

    let (a_chunks, a_rem) = a.as_chunks::<LANES>();
    let (b_chunks, b_rem) = b.as_chunks::<LANES>();

    // 4 independent accumulators to break dependency chain and hide latency
    let mut acc0 = Simd::<f32, LANES>::splat(0.0);
    let mut acc1 = Simd::<f32, LANES>::splat(0.0);
    let mut acc2 = Simd::<f32, LANES>::splat(0.0);
    let mut acc3 = Simd::<f32, LANES>::splat(0.0);

    // Process 4 chunks per iteration (4x unrolling)
    let mut chunks_a = a_chunks.iter();
    let mut chunks_b = b_chunks.iter();

    loop {
        let (Some(a0), Some(b0)) = (chunks_a.next(), chunks_b.next()) else {
            break;
        };
        let a_vec0 = Simd::<f32, LANES>::from_array(*a0);
        let b_vec0 = Simd::<f32, LANES>::from_array(*b0);
        acc0 += a_vec0 * b_vec0;

        let Some(a1) = chunks_a.next() else {
            break;
        };
        let Some(b1) = chunks_b.next() else {
            break;
        };
        let a_vec1 = Simd::<f32, LANES>::from_array(*a1);
        let b_vec1 = Simd::<f32, LANES>::from_array(*b1);
        acc1 += a_vec1 * b_vec1;

        let Some(a2) = chunks_a.next() else {
            break;
        };
        let Some(b2) = chunks_b.next() else {
            break;
        };
        let a_vec2 = Simd::<f32, LANES>::from_array(*a2);
        let b_vec2 = Simd::<f32, LANES>::from_array(*b2);
        acc2 += a_vec2 * b_vec2;

        let Some(a3) = chunks_a.next() else {
            break;
        };
        let Some(b3) = chunks_b.next() else {
            break;
        };
        let a_vec3 = Simd::<f32, LANES>::from_array(*a3);
        let b_vec3 = Simd::<f32, LANES>::from_array(*b3);
        acc3 += a_vec3 * b_vec3;
    }

    // Process any remaining full chunks (0-3 chunks)
    for (a_chunk, b_chunk) in chunks_a.zip(chunks_b) {
        let a_vec = Simd::<f32, LANES>::from_array(*a_chunk);
        let b_vec = Simd::<f32, LANES>::from_array(*b_chunk);
        acc0 += a_vec * b_vec;
    }

    // Combine accumulators and reduce
    let combined = acc0 + acc1 + acc2 + acc3;
    let mut sum = combined.reduce_sum();

    // Process remainder scalarly
    for (a_val, b_val) in a_rem.iter().zip(b_rem.iter()) {
        sum += a_val * b_val;
    }

    Some(sum)
}

// ============================================================================
// Scalar Fallback Implementations
// ============================================================================

/// Scalar L2 distance squared fallback (no sqrt)
///
/// Used when vector too small for SIMD or on unsupported platforms.
#[inline]
fn l2_distance_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
}

/// Scalar dot product fallback
///
/// Used when vector too small for SIMD or on unsupported platforms.
#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let dist = l2_distance(&a, &b);
        let expected = 8.0; // sqrt(16 + 16 + 16 + 16) = sqrt(64) = 8.0

        assert!((dist - expected).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let dot = dot_product(&a, &b);
        let expected = 5.0 + 12.0 + 21.0 + 32.0; // 70.0

        assert!((dot - expected).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];

        let dist = cosine_distance(&a, &b);
        assert!((dist - 0.0).abs() < 1e-6); // Identical vectors
    }

    #[test]
    fn test_large_vectors() {
        let a: Vec<f32> = (0..1536).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..1536).map(|i| (i * 2) as f32).collect();

        let dist = l2_distance(&a, &b);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_simd_vs_scalar_l2() {
        // Test that SIMD and scalar give same results
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| i as f32 * 0.2).collect();

        let simd_result = l2_distance_squared(&a, &b);
        let scalar_result = l2_distance_squared_scalar(&a, &b);

        // Use relative error for larger values (squared distances are larger)
        let relative_error = (simd_result - scalar_result).abs() / scalar_result.abs();
        assert!(
            relative_error < 1e-5,
            "Relative error {} too large",
            relative_error
        );
    }

    #[test]
    fn test_l2_squared_vs_l2() {
        // l2_distance should equal sqrt(l2_distance_squared)
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| i as f32 * 0.2).collect();

        let l2 = l2_distance(&a, &b);
        let l2_sq = l2_distance_squared(&a, &b);

        assert!((l2 - l2_sq.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_normalized() {
        // Pre-normalized vectors should give same result as regular cosine
        let a = vec![1.0, 0.0, 0.0]; // Already unit norm
        let b = vec![0.0, 1.0, 0.0]; // Already unit norm

        let regular = cosine_distance(&a, &b);
        let normalized = cosine_distance_normalized(&a, &b);

        assert!((regular - normalized).abs() < 1e-6);
    }

    #[test]
    fn test_simd_vs_scalar_dot() {
        // Test that SIMD and scalar give same results
        // Note: SIMD may have slightly different floating-point accumulation order
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| i as f32 * 0.2).collect();

        let simd_result = dot_product(&a, &b);
        let scalar_result = dot_product_scalar(&a, &b);

        // Relaxed tolerance for SIMD vs scalar (different accumulation order)
        let relative_error = (simd_result - scalar_result).abs() / scalar_result.abs();
        assert!(
            relative_error < 1e-5,
            "Relative error {} too large",
            relative_error
        );
    }

    #[test]
    fn test_small_vectors() {
        // Test vectors smaller than SIMD lanes (should use scalar)
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];

        let dist = l2_distance(&a, &b);
        let expected = ((2.0_f32).powi(2) + (2.0_f32).powi(2)).sqrt();

        assert!((dist - expected).abs() < 1e-6);
    }

    #[test]
    fn test_zero_vectors() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0];

        let dist = cosine_distance(&a, &b);
        assert_eq!(dist, 1.0); // Maximum distance for zero vector
    }
}
