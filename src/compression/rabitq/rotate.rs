//! Random orthogonal rotation via FFHT + Kac's Walk
//!
//! Implements the rotation step of RaBitQ which spreads information across
//! all dimensions before binary quantization.
//!
//! # Algorithm
//!
//! 4 rounds of: random sign flips → Fast Hadamard Transform → rescale
//!
//! For non-power-of-2 dimensions, Kac's Walk couples the truncated and
//! padded portions to maintain orthogonality.
//!
//! # Reference
//!
//! Based on rotator.hpp from RaBitQ-Library (Apache-2.0)

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Maximum supported dimensions (ceil(dim/64) u64s for sign bits)
const MAX_WORDS: usize = 32; // 32 * 64 = 2048 dimensions

/// Number of rotation rounds (4 rounds for good randomization)
const NUM_ROUNDS: usize = 4;

/// Rotator for random orthogonal transformation
///
/// Stores random sign bits for reproducible rotation.
/// The rotation is orthogonal: R^T R = I
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rotator {
    /// Random sign bits for each round: NUM_ROUNDS x ceil(padded_dim/64) u64s
    bits: [[u64; MAX_WORDS]; NUM_ROUNDS],
    /// Original dimension
    dim: usize,
    /// Padded dimension (next power of 2)
    padded_dim: usize,
    /// Number of u64 words needed
    num_words: usize,
}

impl Rotator {
    /// Create a new rotator with deterministic random bits
    ///
    /// # Arguments
    /// * `dim` - Vector dimension (must be > 0 and <= MAX_WORDS * 64)
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Panics
    /// Panics if dim is 0 or exceeds maximum supported dimensions.
    #[must_use]
    pub fn new(dim: usize, seed: u64) -> Self {
        assert!(dim > 0, "Dimension must be positive");
        assert!(
            dim <= MAX_WORDS * 64,
            "Dimension {} exceeds maximum {}",
            dim,
            MAX_WORDS * 64
        );

        let padded_dim = dim.next_power_of_two();
        let num_words = padded_dim.div_ceil(64);

        let mut rng = StdRng::seed_from_u64(seed);
        let mut bits = [[0u64; MAX_WORDS]; NUM_ROUNDS];

        for round_bits in bits.iter_mut().take(NUM_ROUNDS) {
            for word in round_bits.iter_mut().take(num_words) {
                *word = rng.gen();
            }
        }

        Self {
            bits,
            dim,
            padded_dim,
            num_words,
        }
    }

    /// Get original dimension
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get padded dimension
    #[inline]
    #[must_use]
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Apply random orthogonal rotation in-place
    ///
    /// The input is padded with zeros to the next power of 2, rotated,
    /// and then truncated back to the original dimension.
    ///
    /// # Arguments
    /// * `vector` - Mutable slice of length `dim`
    pub fn rotate(&self, vector: &mut [f32]) {
        assert_eq!(
            vector.len(),
            self.dim,
            "Vector length {} must match rotator dimension {}",
            vector.len(),
            self.dim
        );

        // Fast path: power-of-2 dimensions work in-place (no allocation)
        if self.dim == self.padded_dim {
            self.rotate_inplace(vector);
            return;
        }

        // Slow path: non-power-of-2 needs padding
        let mut padded = vec![0.0f32; self.padded_dim];
        padded[..self.dim].copy_from_slice(vector);

        self.rotate_inplace(&mut padded);

        // Apply Kac's Walk for proper truncation
        kacs_walk(&mut padded, self.dim);

        // Copy back truncated result
        vector.copy_from_slice(&padded[..self.dim]);
    }

    /// Rotate in-place (for power-of-2 or padded vectors)
    #[inline]
    fn rotate_inplace(&self, vector: &mut [f32]) {
        let scale = 1.0 / (self.padded_dim as f32).sqrt();
        for round in 0..NUM_ROUNDS {
            Self::flip_signs_fast(&self.bits[round], vector);
            fht(vector);
            // Rescale after each FHT
            for x in vector.iter_mut() {
                *x *= scale;
            }
        }
    }

    /// Apply inverse rotation (same as forward due to orthogonality)
    ///
    /// For orthogonal matrices: R^(-1) = R^T, and for our construction
    /// using sign flips and Hadamard, R^T = R (symmetric orthogonal).
    #[allow(dead_code)] // Public API for future decompression/debugging
    pub fn unrotate(&self, vector: &mut [f32]) {
        // For Hadamard-based rotation with sign flips, the rotation is
        // symmetric orthogonal, so inverse = forward
        // However, we need to reverse the order of operations
        assert_eq!(
            vector.len(),
            self.dim,
            "Vector length {} must match rotator dimension {}",
            vector.len(),
            self.dim
        );

        // Fast path: power-of-2 dimensions work in-place (no allocation)
        if self.dim == self.padded_dim {
            self.unrotate_inplace(vector);
            return;
        }

        // Slow path: non-power-of-2 needs padding
        let mut padded = vec![0.0f32; self.padded_dim];
        padded[..self.dim].copy_from_slice(vector);

        // Inverse Kac's Walk first (if applied)
        kacs_walk_inverse(&mut padded, self.dim);

        self.unrotate_inplace(&mut padded);

        vector.copy_from_slice(&padded[..self.dim]);
    }

    /// Unrotate in-place (for power-of-2 or padded vectors)
    #[inline]
    #[allow(dead_code)]
    fn unrotate_inplace(&self, vector: &mut [f32]) {
        // Apply rounds in reverse order: rescale → FHT → sign flip
        let scale = 1.0 / (self.padded_dim as f32).sqrt();
        for round in (0..NUM_ROUNDS).rev() {
            for x in vector.iter_mut() {
                *x *= scale;
            }
            fht(vector);
            Self::flip_signs_fast(&self.bits[round], vector);
        }
    }

    /// Flip signs according to random bits (optimized: process 64 elements per word)
    #[inline]
    fn flip_signs_fast(bits: &[u64; MAX_WORDS], x: &mut [f32]) {
        // Process in chunks of 64 to avoid division/modulo per element
        for (word_idx, chunk) in x.chunks_mut(64).enumerate() {
            let word = bits[word_idx];
            for (bit_idx, val) in chunk.iter_mut().enumerate() {
                if (word >> bit_idx) & 1 == 1 {
                    *val = -*val;
                }
            }
        }
    }
}

/// Fast Hadamard Transform (in-place, iterative butterfly)
///
/// Computes H_n * x where H_n is the n×n Hadamard matrix.
/// Uses the butterfly pattern: O(n log n) complexity.
///
/// # Panics
/// Panics if x.len() is not a power of 2.
fn fht(x: &mut [f32]) {
    let n = x.len();
    assert!(n.is_power_of_two(), "FHT requires power-of-2 length");

    // Butterfly pattern: h = 1, 2, 4, ..., n/2
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..(i + h) {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// Kac's Walk for non-power-of-2 dimensions
///
/// Couples the truncated portion [0..trunc_dim] with the padding [trunc_dim..n]
/// to maintain orthogonality properties after truncation.
///
/// Operation: For pairs (x[i], x[i + half]):
///   new_x[i] = x[i] + x[i + half]
///   new_x[i + half] = x[i] - x[i + half]
fn kacs_walk(x: &mut [f32], trunc_dim: usize) {
    let n = x.len();
    let half = n / 2;

    // Only couple elements that span the truncation boundary
    let start = trunc_dim.saturating_sub(half);
    let end = trunc_dim.min(half);

    for i in start..end {
        let a = x[i];
        let b = x[i + half];
        x[i] = a + b;
        x[i + half] = a - b;
    }
}

/// Inverse Kac's Walk
#[allow(dead_code)]
fn kacs_walk_inverse(x: &mut [f32], trunc_dim: usize) {
    let n = x.len();
    let half = n / 2;

    let start = trunc_dim.saturating_sub(half);
    let end = trunc_dim.min(half);

    // Inverse: (a+b, a-b) → (a, b) = ((sum+diff)/2, (sum-diff)/2)
    for i in start..end {
        let sum = x[i];
        let diff = x[i + half];
        x[i] = (sum + diff) * 0.5;
        x[i + half] = (sum - diff) * 0.5;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fht_basic() {
        // FHT of [1, 1, 1, 1] should give [4, 0, 0, 0] (unnormalized)
        let mut x = vec![1.0, 1.0, 1.0, 1.0];
        fht(&mut x);
        assert!((x[0] - 4.0).abs() < 1e-6);
        assert!(x[1].abs() < 1e-6);
        assert!(x[2].abs() < 1e-6);
        assert!(x[3].abs() < 1e-6);
    }

    #[test]
    fn test_fht_roundtrip() {
        // FHT(FHT(x)) = n * x (Hadamard is self-inverse up to scaling)
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut x = original.clone();
        let n = x.len() as f32;

        fht(&mut x);
        fht(&mut x);

        for (o, r) in original.iter().zip(x.iter()) {
            assert!(
                (r / n - o).abs() < 1e-5,
                "FHT roundtrip failed: {} vs {}",
                r / n,
                o
            );
        }
    }

    #[test]
    fn test_rotator_deterministic() {
        let r1 = Rotator::new(128, 42);
        let r2 = Rotator::new(128, 42);

        // Same seed should produce same bits
        for round in 0..NUM_ROUNDS {
            for word in 0..r1.num_words {
                assert_eq!(r1.bits[round][word], r2.bits[round][word]);
            }
        }
    }

    #[test]
    fn test_rotation_preserves_norm() {
        let rotator = Rotator::new(64, 12345);

        // Random vector
        let mut rng = StdRng::seed_from_u64(999);
        let original: Vec<f32> = (0..64).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let original_norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();

        let mut rotated = original.clone();
        rotator.rotate(&mut rotated);
        let rotated_norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Norm should be preserved (orthogonal transformation)
        assert!(
            (original_norm - rotated_norm).abs() < 0.01,
            "Norm changed: {} -> {}",
            original_norm,
            rotated_norm
        );
    }

    #[test]
    fn test_rotation_roundtrip() {
        let rotator = Rotator::new(64, 54321);

        let mut rng = StdRng::seed_from_u64(888);
        let original: Vec<f32> = (0..64).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let mut v = original.clone();
        rotator.rotate(&mut v);
        rotator.unrotate(&mut v);

        for (o, r) in original.iter().zip(v.iter()) {
            assert!((o - r).abs() < 0.01, "Roundtrip failed: {} vs {}", o, r);
        }
    }

    #[test]
    fn test_non_power_of_2() {
        // Test with a dimension that isn't a power of 2
        let rotator = Rotator::new(100, 777);
        assert_eq!(rotator.dim(), 100);
        assert_eq!(rotator.padded_dim(), 128);

        let mut rng = StdRng::seed_from_u64(444);
        let original: Vec<f32> = (0..100).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let original_norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();

        let mut v = original.clone();
        rotator.rotate(&mut v);

        assert_eq!(v.len(), 100);

        // Norm should be approximately preserved
        let rotated_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rel_diff = (original_norm - rotated_norm).abs() / original_norm;
        assert!(
            rel_diff < 0.1,
            "Norm changed too much: {} -> {} ({}%)",
            original_norm,
            rotated_norm,
            rel_diff * 100.0
        );
    }
}
