//! SIMD-optimized operations for RaBitQ
//!
//! Provides optimized Hamming distance and popcount for binary codes.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vaddvq_u8, vcntq_u8, veorq_u8, vld1q_u8};
#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

/// Compute Hamming distance between two u64 arrays
///
/// Returns the number of differing bits.
#[inline]
#[must_use]
#[allow(clippy::needless_return)]
pub fn hamming_distance_u64(a: &[u64], b: &[u64]) -> u32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Slice lengths must match: {} vs {}",
        a.len(),
        b.len()
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("popcnt") {
            return unsafe { hamming_distance_popcnt(a, b) };
        }
        return hamming_distance_scalar(a, b);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON operates on u8 arrays, so we need to reinterpret
        let a_bytes: &[u8] = unsafe { std::slice::from_raw_parts(a.as_ptr().cast(), a.len() * 8) };
        let b_bytes: &[u8] = unsafe { std::slice::from_raw_parts(b.as_ptr().cast(), b.len() * 8) };
        return unsafe { hamming_distance_neon(a_bytes, b_bytes) };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    hamming_distance_scalar(a, b)
}

/// Scalar Hamming distance (fallback)
#[allow(dead_code)]
fn hamming_distance_scalar(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

/// x86_64 popcnt-based Hamming distance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
#[allow(clippy::cast_possible_wrap)]
unsafe fn hamming_distance_popcnt(a: &[u64], b: &[u64]) -> u32 {
    let mut count = 0u64;

    for (&x, &y) in a.iter().zip(b.iter()) {
        count += _popcnt64((x ^ y) as i64) as u64;
    }

    count as u32
}

/// NEON Hamming distance (for u8 arrays)
///
/// Optimized: accumulates in vector register, does single horizontal sum at end.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn hamming_distance_neon(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::aarch64::vdupq_n_u8;

    let mut i = 0;

    // Use wider accumulators to avoid overflow and defer horizontal sum
    // u8 popcounts -> u16 pairwise add -> u32 pairwise add -> u64 pairwise add
    let mut acc_lo: u64 = 0;
    let mut acc_hi: u64 = 0;

    // Process 64 bytes at a time (4x16) to maximize register utilization
    while i + 64 <= a.len() {
        // Load and process 4 vectors
        let mut sum16 = vdupq_n_u8(0);

        for _ in 0..4 {
            let va = vld1q_u8(a.as_ptr().add(i));
            let vb = vld1q_u8(b.as_ptr().add(i));
            let xor = veorq_u8(va, vb);
            let cnt = vcntq_u8(xor);
            // Accumulate popcounts (safe: max 64*8 = 512, fits in u8 sum of 4 iterations)
            sum16 = std::arch::aarch64::vaddq_u8(sum16, cnt);
            i += 16;
        }

        // Widen and add to accumulator
        acc_lo += vaddvq_u8(sum16) as u64;
    }

    // Process remaining 16-byte chunks
    while i + 16 <= a.len() {
        let va = vld1q_u8(a.as_ptr().add(i));
        let vb = vld1q_u8(b.as_ptr().add(i));
        let xor = veorq_u8(va, vb);
        let cnt = vcntq_u8(xor);
        acc_hi += vaddvq_u8(cnt) as u64;
        i += 16;
    }

    // Handle remaining bytes (scalar)
    let mut scalar_sum: u32 = 0;
    for j in i..a.len() {
        scalar_sum += (a[j] ^ b[j]).count_ones();
    }

    (acc_lo + acc_hi) as u32 + scalar_sum
}

/// Popcount for a single u64 array (total set bits)
#[inline]
#[must_use]
#[allow(dead_code)] // Public API for binary operations
pub fn popcount_u64(a: &[u64]) -> u32 {
    a.iter().map(|&x| x.count_ones()).sum()
}

/// XOR two u64 arrays into a destination
#[inline]
#[allow(dead_code)] // Public API for binary operations
pub fn xor_u64(a: &[u64], b: &[u64], dest: &mut [u64]) {
    assert_eq!(a.len(), b.len(), "Input slice lengths must match");
    assert_eq!(a.len(), dest.len(), "Output slice length must match input");

    for ((d, &x), &y) in dest.iter_mut().zip(a.iter()).zip(b.iter()) {
        *d = x ^ y;
    }
}

/// AND two u64 arrays and return popcount (for binary inner product)
#[inline]
#[must_use]
#[allow(clippy::needless_return)]
#[allow(dead_code)] // Public API for binary inner product
pub fn and_popcount_u64(a: &[u64], b: &[u64]) -> u32 {
    assert_eq!(a.len(), b.len(), "Slice lengths must match");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("popcnt") {
            return unsafe { and_popcount_popcnt(a, b) };
        }
        return and_popcount_scalar(a, b);
    }

    #[cfg(not(target_arch = "x86_64"))]
    and_popcount_scalar(a, b)
}

#[allow(dead_code)]
fn and_popcount_scalar(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x & y).count_ones())
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
#[allow(clippy::cast_possible_wrap)]
unsafe fn and_popcount_popcnt(a: &[u64], b: &[u64]) -> u32 {
    let mut count = 0u64;

    for (&x, &y) in a.iter().zip(b.iter()) {
        count += _popcnt64((x & y) as i64) as u64;
    }

    count as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_identical() {
        let a = vec![0xFFFF_FFFF_FFFF_FFFFu64; 4];
        let b = vec![0xFFFF_FFFF_FFFF_FFFFu64; 4];
        assert_eq!(hamming_distance_u64(&a, &b), 0);
    }

    #[test]
    fn test_hamming_all_different() {
        let a = vec![0u64; 4];
        let b = vec![0xFFFF_FFFF_FFFF_FFFFu64; 4];
        // 4 words * 64 bits = 256 bits different
        assert_eq!(hamming_distance_u64(&a, &b), 256);
    }

    #[test]
    fn test_hamming_partial() {
        let a = vec![0b1010_1010u64];
        let b = vec![0b0101_0101u64];
        // All 8 bits different in the low byte
        assert_eq!(hamming_distance_u64(&a, &b), 8);
    }

    #[test]
    fn test_hamming_single_bit() {
        let a = vec![0b0000_0001u64];
        let b = vec![0b0000_0000u64];
        assert_eq!(hamming_distance_u64(&a, &b), 1);
    }

    #[test]
    fn test_popcount() {
        let a = vec![0b1111_0000u64, 0b0000_1111u64];
        assert_eq!(popcount_u64(&a), 8);
    }

    #[test]
    fn test_and_popcount() {
        let a = vec![0b1111_0000u64];
        let b = vec![0b1111_1111u64];
        // AND gives 0b1111_0000, popcount = 4
        assert_eq!(and_popcount_u64(&a, &b), 4);
    }

    #[test]
    fn test_xor() {
        let a = vec![0b1111_0000u64];
        let b = vec![0b0000_1111u64];
        let mut dest = vec![0u64; 1];
        xor_u64(&a, &b, &mut dest);
        assert_eq!(dest[0], 0b1111_1111);
    }

    #[test]
    fn test_large_hamming() {
        // 768 dimensions = 12 u64 words
        let a: Vec<u64> = (0..12).map(|i| i * 0x0101_0101_0101_0101).collect();
        let b: Vec<u64> = (0..12).map(|i| (11 - i) * 0x0101_0101_0101_0101).collect();

        let dist = hamming_distance_u64(&a, &b);
        assert!(dist > 0);
        assert!(dist <= 12 * 64);
    }
}
