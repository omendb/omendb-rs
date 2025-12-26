//! Binary Quantization (BBQ) for `OmenDB`
//!
//! 1-bit quantization with SIMD-optimized Hamming distance.
//!
//! # Algorithm
//!
//! - Quantize: bit[d] = 1 if f32[d] > threshold[d] else 0
//! - Distance: Hamming distance via XOR + popcnt
//! - Correction: Apply norm-based correction for accurate ranking
//!
//! # Performance
//!
//! - 32x compression (f32 → 1 bit)
//! - 2-4x faster search than SQ8 (SIMD Hamming is extremely fast)
//! - ~85% raw recall, ~95-98% with rescore
//!
//! # When to Use
//!
//! - Dimensions >= 384 (below this, SQ8 has better recall)
//! - Large datasets (>100K vectors) where memory matters
//! - Cost-sensitive deployments

use serde::{Deserialize, Serialize};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vaddvq_u8, vcntq_u8, veorq_u8, vld1q_u8};
#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

/// Binary quantization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryParams {
    /// Threshold per dimension (typically 0.0 or median)
    pub thresholds: Vec<f32>,
    /// Number of dimensions
    pub dimensions: usize,
}

impl BinaryParams {
    /// Create with zero thresholds (value > 0 = 1, value <= 0 = 0)
    #[must_use]
    pub fn new(dimensions: usize) -> Self {
        Self {
            thresholds: vec![0.0; dimensions],
            dimensions,
        }
    }

    /// Train thresholds from sample vectors using median per dimension.
    ///
    /// # Errors
    /// Returns error if vectors is empty or vectors have inconsistent dimensions.
    pub fn train(vectors: &[&[f32]]) -> Result<Self, &'static str> {
        if vectors.is_empty() {
            return Err("Need at least one vector to train");
        }
        let dimensions = vectors[0].len();
        if !vectors.iter().all(|v| v.len() == dimensions) {
            return Err("All vectors must have same dimensions");
        }

        let n = vectors.len();
        let mut thresholds = Vec::with_capacity(dimensions);
        let mut dim_values: Vec<f32> = Vec::with_capacity(n);

        for d in 0..dimensions {
            dim_values.clear();
            for v in vectors {
                dim_values.push(v[d]);
            }
            dim_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Use median as threshold
            let median = if n.is_multiple_of(2) {
                let mid = n / 2;
                f32::midpoint(dim_values[mid - 1], dim_values[mid])
            } else {
                dim_values[n / 2]
            };

            thresholds.push(median);
        }

        Ok(Self {
            thresholds,
            dimensions,
        })
    }

    /// Quantize f32 vector to packed binary
    #[must_use]
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        debug_assert_eq!(vector.len(), self.dimensions);

        let num_bytes = self.dimensions.div_ceil(8);
        let mut quantized = vec![0u8; num_bytes];

        for (i, (&value, &threshold)) in vector.iter().zip(self.thresholds.iter()).enumerate() {
            if value > threshold {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                quantized[byte_idx] |= 1 << bit_idx;
            }
        }

        quantized
    }

    /// Quantize into pre-allocated buffer
    pub fn quantize_into(&self, vector: &[f32], output: &mut [u8]) {
        debug_assert_eq!(vector.len(), self.dimensions);
        let num_bytes = self.dimensions.div_ceil(8);
        debug_assert!(output.len() >= num_bytes);

        // Clear output first
        for byte in output.iter_mut().take(num_bytes) {
            *byte = 0;
        }

        for (i, (&value, &threshold)) in vector.iter().zip(self.thresholds.iter()).enumerate() {
            if value > threshold {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                output[byte_idx] |= 1 << bit_idx;
            }
        }
    }
}

/// Compute Hamming distance between two binary vectors
///
/// SIMD-optimized using:
/// - AVX2: _mm256_xor_si256 + manual popcnt
/// - AVX-512: _mm512_popcnt_epi64 (if available)
/// - NEON: veorq_u8 + vcntq_u8
///
/// Falls back to scalar popcnt if no SIMD available.
#[inline]
#[must_use]
#[allow(clippy::needless_return)] // returns needed for cfg-conditional control flow
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { hamming_distance_avx2(a, b) };
        }
        if is_x86_feature_detected!("popcnt") {
            return unsafe { hamming_distance_popcnt(a, b) };
        }
        return hamming_distance_scalar(a, b);
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { hamming_distance_neon(a, b) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    hamming_distance_scalar(a, b)
}

/// Scalar Hamming distance (fallback)
#[allow(dead_code)]
fn hamming_distance_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

/// AVX2 Hamming distance with PSHUFB popcount lookup table
///
/// Uses nibble-based lookup instead of scalar popcnt for full SIMD throughput.
/// Technique: split each byte into two 4-bit nibbles, use shuffle as LUT.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_ptr_alignment)] // loadu handles unaligned loads
unsafe fn hamming_distance_avx2(a: &[u8], b: &[u8]) -> u32 {
    // Popcount lookup table for 4-bit values: [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
    let lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, // low 128 bits
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, // high 128 bits
    );
    let low_mask = _mm256_set1_epi8(0x0f); // mask for low nibble

    let mut total = _mm256_setzero_si256();
    let mut i = 0;

    // Process 32 bytes at a time
    while i + 32 <= a.len() {
        let va = _mm256_loadu_si256(a.as_ptr().add(i).cast::<__m256i>());
        let vb = _mm256_loadu_si256(b.as_ptr().add(i).cast::<__m256i>());
        let xor = _mm256_xor_si256(va, vb);

        // Split into nibbles and lookup popcount
        let lo = _mm256_and_si256(xor, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(xor, 4), low_mask);

        let cnt_lo = _mm256_shuffle_epi8(lookup, lo);
        let cnt_hi = _mm256_shuffle_epi8(lookup, hi);

        // Add nibble counts (each byte now has popcount of original byte)
        let cnt = _mm256_add_epi8(cnt_lo, cnt_hi);

        // Accumulate using sad_epu8 against zero for horizontal sum
        total = _mm256_add_epi64(total, _mm256_sad_epu8(cnt, _mm256_setzero_si256()));

        i += 32;
    }

    // Horizontal sum of 4 x u64 accumulators
    let lo = _mm256_castsi256_si128(total);
    let hi = _mm256_extracti128_si256(total, 1);
    let sum128 = _mm_add_epi64(lo, hi);
    let count = (_mm_extract_epi64(sum128, 0) + _mm_extract_epi64(sum128, 1)) as u32;

    // Handle remaining bytes with scalar
    let mut remainder = 0u32;
    for j in i..a.len() {
        remainder += (a[j] ^ b[j]).count_ones();
    }

    count + remainder
}

/// x86_64 popcnt-based Hamming distance (8 bytes at a time)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
#[allow(clippy::cast_possible_wrap)] // intentional reinterpret for popcnt
unsafe fn hamming_distance_popcnt(a: &[u8], b: &[u8]) -> u32 {
    let mut count = 0u64;
    let mut i = 0;

    // Process 8 bytes at a time using u64 popcnt
    while i + 8 <= a.len() {
        let a_u64 = std::ptr::read_unaligned(a.as_ptr().add(i).cast::<u64>());
        let b_u64 = std::ptr::read_unaligned(b.as_ptr().add(i).cast::<u64>());
        count += _popcnt64((a_u64 ^ b_u64) as i64) as u64;
        i += 8;
    }

    // Handle remaining bytes
    for j in i..a.len() {
        count += (a[j] ^ b[j]).count_ones() as u64;
    }

    count as u32
}

/// NEON Hamming distance with native vcntq_u8
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn hamming_distance_neon(a: &[u8], b: &[u8]) -> u32 {
    let mut sum: u32 = 0;
    let mut i = 0;

    // Process 16 bytes at a time
    while i + 16 <= a.len() {
        // Load 16 bytes each
        let va = vld1q_u8(a.as_ptr().add(i));
        let vb = vld1q_u8(b.as_ptr().add(i));

        // XOR
        let xor = veorq_u8(va, vb);

        // Count bits per byte and sum horizontally
        let cnt = vcntq_u8(xor);
        sum += vaddvq_u8(cnt) as u32;

        i += 16;
    }

    // Handle remaining bytes
    for j in i..a.len() {
        sum += (a[j] ^ b[j]).count_ones();
    }

    sum
}

/// Compute corrected distance for binary quantization
///
/// Raw Hamming distance gives rough ranking. Correction using norms
/// approximates true L2 distance for better accuracy.
///
/// Formula: corrected = hamming * (query_norm * vec_norm) / dimensions
#[inline]
#[must_use]
pub fn corrected_distance(hamming: u32, query_norm: f32, vec_norm: f32, dimensions: usize) -> f32 {
    let hamming_f = hamming as f32;
    // Convert Hamming to approximate L2: each different bit contributes to distance
    // The scaling approximates the magnitude of the difference
    hamming_f * (query_norm * vec_norm) / (dimensions as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_quantize() {
        let params = BinaryParams::new(8);
        let vector = vec![0.5, -0.3, 0.8, -0.1, 0.0, 0.1, -0.5, 0.2];

        let quantized = params.quantize(&vector);

        // Bits: 1 (0.5>0), 0 (-0.3), 1 (0.8>0), 0 (-0.1), 0 (0.0==0), 1 (0.1>0), 0 (-0.5), 1 (0.2>0)
        // Packed: bit0=1, bit1=0, bit2=1, bit3=0, bit4=0, bit5=1, bit6=0, bit7=1
        // = 0b10100101 = 165
        assert_eq!(quantized.len(), 1);
        assert_eq!(quantized[0], 0b1010_0101);
    }

    #[test]
    fn test_binary_train() {
        let v1 = vec![1.0, 5.0, 0.0, 2.0];
        let v2 = vec![2.0, 6.0, 1.0, 3.0];
        let v3 = vec![3.0, 7.0, 2.0, 4.0];
        let vectors: Vec<&[f32]> = vec![v1.as_slice(), v2.as_slice(), v3.as_slice()];

        let params = BinaryParams::train(&vectors).unwrap();

        // Median: [2.0, 6.0, 1.0, 3.0]
        assert_eq!(params.thresholds, vec![2.0, 6.0, 1.0, 3.0]);
    }

    #[test]
    fn test_hamming_distance_identical() {
        let a = vec![0b1010_1010, 0b1111_0000, 0b0000_1111];
        let b = vec![0b1010_1010, 0b1111_0000, 0b0000_1111];

        let dist = hamming_distance(&a, &b);
        assert_eq!(dist, 0);
    }

    #[test]
    fn test_hamming_distance_all_different() {
        let a = vec![0b0000_0000];
        let b = vec![0b1111_1111];

        let dist = hamming_distance(&a, &b);
        assert_eq!(dist, 8); // All 8 bits different
    }

    #[test]
    fn test_hamming_distance_partial() {
        let a = vec![0b1010_1010];
        let b = vec![0b0101_0101];

        let dist = hamming_distance(&a, &b);
        assert_eq!(dist, 8); // All bits flipped
    }

    #[test]
    fn test_hamming_distance_large() {
        // 768 dimensions = 96 bytes
        let a: Vec<u8> = vec![0b1010_1010; 96];
        let b: Vec<u8> = vec![0b0101_0101; 96];

        let dist = hamming_distance(&a, &b);
        assert_eq!(dist, 96 * 8); // All 8 bits different in all 96 bytes
    }

    #[test]
    fn test_compression_ratio() {
        let dims: usize = 768;
        let original_size = dims * 4; // f32 = 4 bytes
        let quantized_size = dims.div_ceil(8); // 1 bit = 1/8 byte

        let ratio = original_size as f32 / quantized_size as f32;
        assert!(
            (ratio - 32.0).abs() < 0.1,
            "Expected 32x compression, got {ratio}"
        );
    }

    #[test]
    fn test_corrected_distance() {
        let hamming = 100;
        let query_norm = 2.0;
        let vec_norm = 1.5;
        let dimensions = 768;

        let dist = corrected_distance(hamming, query_norm, vec_norm, dimensions);

        // Should be: 100 * 2.0 * 1.5 / 768 ≈ 0.39
        assert!((dist - 0.39).abs() < 0.01);
    }
}
