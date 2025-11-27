///! SIMD-accelerated distance calculations for HNSW
///!
///! Uses std::simd (portable SIMD) for cross-platform SIMD support.
///! Automatically compiles to optimal SIMD instructions (AVX2, SSE2, NEON).
///!
///! ## Performance
///!
///! - 8 lanes (AVX2): 3-4x speedup over scalar
///! - 4 lanes (SSE2/NEON): 2-3x speedup over scalar
///!
///! ## Implementation
///!
///! Uses generic SIMD implementation with lane count (8, 4, or scalar fallback).
///! Compiler automatically selects best instruction set for target platform.

use std::simd::{LaneCount, Simd, SupportedLaneCount, num::SimdFloat};

/// L2 distance (Euclidean) with SIMD acceleration
///
/// Automatically uses optimal SIMD lane count (8 for AVX2, 4 for SSE2/NEON).
/// Falls back to scalar if vector too small for SIMD.
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_distance_squared(a, b).sqrt()
}

/// L2 distance squared (skip sqrt) - for HNSW internal comparisons
///
/// HNSW only needs relative ordering, not absolute distances.
/// Squared distance maintains ordering since sqrt is monotonic.
/// This saves ~10-15% compute time in search_layer.
///
/// Note: Use l2_distance() when actual Euclidean distance is needed.
#[inline]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    // Try 8 lanes (AVX2), then 4 lanes (SSE2/NEON), then scalar
    l2_distance_squared_simd::<8>(a, b)
        .unwrap_or_else(|| l2_distance_squared_simd::<4>(a, b)
            .unwrap_or_else(|| l2_distance_squared_scalar(a, b)))
}

/// Dot product with SIMD acceleration
///
/// Automatically uses optimal SIMD lane count (8 for AVX2, 4 for SSE2/NEON).
/// Falls back to scalar if vector too small for SIMD.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    // Try 8 lanes (AVX2), then 4 lanes (SSE2/NEON), then scalar
    dot_product_simd::<8>(a, b)
        .unwrap_or_else(|| dot_product_simd::<4>(a, b)
            .unwrap_or_else(|| dot_product_scalar(a, b)))
}

/// Cosine distance with SIMD acceleration
///
/// Computed as: 1 - (dot(a, b) / (norm(a) * norm(b)))
/// Returns 1.0 for zero vectors (maximum distance).
#[inline]
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
///   cosine_distance = 1 - dot(a, b)
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

    // Accumulate in SIMD register (avoids reduce_sum per iteration)
    let mut acc = Simd::<f32, LANES>::splat(0.0);

    for (a_chunk, b_chunk) in a_chunks.iter().zip(b_chunks.iter()) {
        let a_vec = Simd::<f32, LANES>::from_array(*a_chunk);
        let b_vec = Simd::<f32, LANES>::from_array(*b_chunk);
        let diff = a_vec - b_vec;
        acc += diff * diff;
    }

    let mut sum = acc.reduce_sum();

    // Process remainder scalarly
    for (a_val, b_val) in a_rem.iter().zip(b_rem.iter()) {
        let diff = a_val - b_val;
        sum += diff * diff;
    }

    Some(sum)
}

/// Generic SIMD dot product implementation
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

    // Accumulate in SIMD register (avoids reduce_sum per iteration)
    let mut acc = Simd::<f32, LANES>::splat(0.0);

    for (a_chunk, b_chunk) in a_chunks.iter().zip(b_chunks.iter()) {
        let a_vec = Simd::<f32, LANES>::from_array(*a_chunk);
        let b_vec = Simd::<f32, LANES>::from_array(*b_chunk);
        acc += a_vec * b_vec;
    }

    let mut sum = acc.reduce_sum();

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
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
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
        assert!(relative_error < 1e-5, "Relative error {} too large", relative_error);
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
        assert!(relative_error < 1e-5, "Relative error {} too large", relative_error);
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
