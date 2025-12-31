//! FastScan SIMD-accelerated distance computation for quantized vectors
//!
//! FastScan uses SIMD shuffle instructions (pshufb/vqtbl1q) to perform
//! parallel LUT lookups, computing distances for 32 neighbors at once.
//!
//! # Performance
//!
//! Benchmark on M3 Max showed 5x speedup vs per-neighbor ADC:
//! - Per-neighbor ADC: 1.93 µs for 32 neighbors
//! - FastScan NEON: 390 ns for 32 neighbors
//!
//! # Memory Layout
//!
//! FastScan requires codes to be interleaved by sub-quantizer position:
//! ```text
//! [n0_sq0, n1_sq0, ..., n31_sq0]  // 32 bytes - sub-quantizer 0 for all neighbors
//! [n0_sq1, n1_sq1, ..., n31_sq1]  // 32 bytes - sub-quantizer 1 for all neighbors
//! ```
//!
//! For 4-bit RaBitQ with 768 dimensions:
//! - code_size = 768 / 2 = 384 bytes per vector
//! - 384 sub-quantizers, each holding 2 dimension codes (lo/hi nibbles)
//!
//! # LUT Format
//!
//! For 4-bit quantization, each sub-quantizer has TWO 16-entry u8 LUTs:
//! - `luts_lo[sq][code]` for the lo nibble (even dimension)
//! - `luts_hi[sq][code]` for the hi nibble (odd dimension)

use crate::compression::ADCTable;

/// Batch size for FastScan - AVX2/NEON process 32 bytes at a time
pub const BATCH_SIZE: usize = 32;

/// Quantized LUT for FastScan (u8 distances for SIMD efficiency)
///
/// Contains pre-computed distance contributions for each possible code value.
/// For 4-bit quantization: 16 entries per sub-quantizer, separate LUTs for lo/hi nibbles.
#[derive(Debug, Clone)]
pub struct FastScanLUT {
    /// Lo nibble LUTs: luts_lo[sq][code] = quantized distance for even dimension
    luts_lo: Vec<[u8; 16]>,

    /// Hi nibble LUTs: luts_hi[sq][code] = quantized distance for odd dimension
    luts_hi: Vec<[u8; 16]>,

    /// Scale factor to convert accumulated u16 back to approximate f32 distance
    scale: f32,

    /// Offset to add after scaling (for accurate reconstruction)
    offset: f32,
}

impl FastScanLUT {
    /// Build FastScan LUT from RaBitQ ADC table
    ///
    /// ADC table format: table[dim][code] = partial squared distance
    /// For 4-bit quantization with D dimensions:
    /// - D/2 sub-quantizers (each byte packs 2 dimensions)
    /// - table[sq*2][code] = distance contribution for lo nibble (even dim)
    /// - table[sq*2+1][code] = distance contribution for hi nibble (odd dim)
    #[must_use]
    pub fn from_adc_table(adc: &ADCTable) -> Option<Self> {
        // Only support 4-bit for now
        if adc.bits() != 4 {
            return None;
        }

        let dimensions = adc.dimensions();
        if dimensions == 0 || !dimensions.is_multiple_of(2) {
            return None;
        }

        let num_sq = dimensions / 2;

        // Find global min/max across all LUT entries for uniform scaling
        // We sum lo + hi contributions, so find min/max of sums
        let mut global_min = f32::MAX;
        let mut global_max = f32::MIN;

        for sq in 0..num_sq {
            for lo_code in 0..16 {
                for hi_code in 0..16 {
                    let dist_lo = adc.get(sq * 2, lo_code);
                    let dist_hi = adc.get(sq * 2 + 1, hi_code);
                    let sum = dist_lo + dist_hi;
                    global_min = global_min.min(sum);
                    global_max = global_max.max(sum);
                }
            }
        }

        // Calculate safe max per nibble to prevent u16 overflow
        // Max accumulation = num_sq * 2 * max_per_nibble <= 65535
        // Formula: max_per_nibble = floor(65535 / (num_sq * 2))
        // Examples: 128D->127, 512D->127, 768D->85, 1536D->42
        let safe_max_per_nibble = (65535.0 / (num_sq * 2) as f32).floor().min(127.0);

        // Each sub-quantizer contributes to the sum, so scale per-sq contributions
        // to fit in u8 such that the total sum fits in u16
        let range = global_max - global_min;
        let scale_factor = if range > 1e-7 {
            safe_max_per_nibble / (range / 2.0) // Divide range by 2 since lo+hi both contribute
        } else {
            1.0
        };

        let offset = global_min;

        // Build separate LUTs for lo and hi nibbles
        let mut luts_lo = Vec::with_capacity(num_sq);
        let mut luts_hi = Vec::with_capacity(num_sq);

        for sq in 0..num_sq {
            let dim_lo = sq * 2;
            let dim_hi = sq * 2 + 1;

            // Lo nibble LUT (even dimension)
            let mut lut_lo = [0u8; 16];
            for (code, entry) in lut_lo.iter_mut().enumerate() {
                let dist = adc.get(dim_lo, code);
                // Subtract per-dimension share of offset, then scale
                *entry = ((dist - offset / 2.0) * scale_factor)
                    .round()
                    .clamp(0.0, safe_max_per_nibble) as u8;
            }

            // Hi nibble LUT (odd dimension)
            let mut lut_hi = [0u8; 16];
            for (code, entry) in lut_hi.iter_mut().enumerate() {
                let dist = adc.get(dim_hi, code);
                *entry = ((dist - offset / 2.0) * scale_factor)
                    .round()
                    .clamp(0.0, safe_max_per_nibble) as u8;
            }

            luts_lo.push(lut_lo);
            luts_hi.push(lut_hi);
        }

        Some(Self {
            luts_lo,
            luts_hi,
            scale: 1.0 / scale_factor,
            offset,
        })
    }

    /// Get number of sub-quantizers
    #[must_use]
    pub fn num_sq(&self) -> usize {
        self.luts_lo.len()
    }

    /// Get lo nibble LUTs
    #[must_use]
    pub fn luts_lo(&self) -> &[[u8; 16]] {
        &self.luts_lo
    }

    /// Get hi nibble LUTs
    #[must_use]
    pub fn luts_hi(&self) -> &[[u8; 16]] {
        &self.luts_hi
    }

    /// Convert accumulated u16 distance back to approximate f32
    #[must_use]
    pub fn to_f32(&self, accumulated: u16) -> f32 {
        accumulated as f32 * self.scale + self.offset
    }
}

/// Compute batched distances using FastScan NEON (ARM)
///
/// # Arguments
/// * `luts_lo` - Lo nibble LUTs (one 16-byte LUT per sub-quantizer)
/// * `luts_hi` - Hi nibble LUTs (one 16-byte LUT per sub-quantizer)
/// * `interleaved_codes` - Interleaved neighbor codes (num_sq * 32 bytes)
///
/// # Returns
/// Array of 32 accumulated u16 distances
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn fastscan_batch_neon(
    luts_lo: &[[u8; 16]],
    luts_hi: &[[u8; 16]],
    interleaved_codes: &[u8],
) -> [u16; BATCH_SIZE] {
    use std::arch::aarch64::{
        uint16x8_t, vaddl_u8, vaddq_u16, vandq_u8, vdupq_n_u16, vdupq_n_u8, vget_high_u8,
        vget_low_u8, vld1q_u8, vqtbl1q_u8, vshrq_n_u8, vst1q_u16,
    };

    unsafe {
        let low_mask = vdupq_n_u8(0x0F);

        // Four accumulators for 32 results (NEON processes 8 u16 at a time)
        let mut accum0: uint16x8_t = vdupq_n_u16(0);
        let mut accum1: uint16x8_t = vdupq_n_u16(0);
        let mut accum2: uint16x8_t = vdupq_n_u16(0);
        let mut accum3: uint16x8_t = vdupq_n_u16(0);

        // Process each sub-quantizer
        for sq in 0..luts_lo.len() {
            let base = sq * BATCH_SIZE;

            // Load separate LUTs for lo and hi nibbles
            let lut_lo_vec = vld1q_u8(luts_lo[sq].as_ptr());
            let lut_hi_vec = vld1q_u8(luts_hi[sq].as_ptr());

            // Load 32 bytes of codes (32 neighbors' codes for this sub-quantizer)
            let codes_0_15 = vld1q_u8(interleaved_codes.as_ptr().add(base));
            let codes_16_31 = vld1q_u8(interleaved_codes.as_ptr().add(base + 16));

            // Extract lo nibbles and lookup in lut_lo
            let idx_lo_0 = vandq_u8(codes_0_15, low_mask);
            let idx_lo_1 = vandq_u8(codes_16_31, low_mask);
            let vals_lo_0 = vqtbl1q_u8(lut_lo_vec, idx_lo_0);
            let vals_lo_1 = vqtbl1q_u8(lut_lo_vec, idx_lo_1);

            // Extract hi nibbles and lookup in lut_hi
            let idx_hi_0 = vshrq_n_u8(codes_0_15, 4);
            let idx_hi_1 = vshrq_n_u8(codes_16_31, 4);
            let vals_hi_0 = vqtbl1q_u8(lut_hi_vec, idx_hi_0);
            let vals_hi_1 = vqtbl1q_u8(lut_hi_vec, idx_hi_1);

            // Accumulate as u16 to avoid overflow
            // Neighbors 0-7
            accum0 = vaddq_u16(
                accum0,
                vaddl_u8(vget_low_u8(vals_lo_0), vget_low_u8(vals_hi_0)),
            );
            // Neighbors 8-15
            accum1 = vaddq_u16(
                accum1,
                vaddl_u8(vget_high_u8(vals_lo_0), vget_high_u8(vals_hi_0)),
            );
            // Neighbors 16-23
            accum2 = vaddq_u16(
                accum2,
                vaddl_u8(vget_low_u8(vals_lo_1), vget_low_u8(vals_hi_1)),
            );
            // Neighbors 24-31
            accum3 = vaddq_u16(
                accum3,
                vaddl_u8(vget_high_u8(vals_lo_1), vget_high_u8(vals_hi_1)),
            );
        }

        // Extract results
        let mut results = [0u16; BATCH_SIZE];
        vst1q_u16(results.as_mut_ptr(), accum0);
        vst1q_u16(results.as_mut_ptr().add(8), accum1);
        vst1q_u16(results.as_mut_ptr().add(16), accum2);
        vst1q_u16(results.as_mut_ptr().add(24), accum3);

        results
    }
}

/// Compute batched distances using FastScan AVX2 (x86_64)
#[cfg(target_arch = "x86_64")]
#[allow(clippy::cast_ptr_alignment)] // loadu/storeu intrinsics handle unaligned access
#[must_use]
pub fn fastscan_batch_avx2(
    luts_lo: &[[u8; 16]],
    luts_hi: &[[u8; 16]],
    interleaved_codes: &[u8],
) -> [u16; BATCH_SIZE] {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm256_add_epi16, _mm256_and_si256, _mm256_broadcastsi128_si256,
        _mm256_cvtepu8_epi16, _mm256_loadu_si256, _mm256_set1_epi8, _mm256_setzero_si256,
        _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_storeu_si256, _mm_loadu_si128,
    };

    unsafe {
        if !std::is_x86_feature_detected!("avx2") {
            return fastscan_batch_scalar(luts_lo, luts_hi, interleaved_codes);
        }

        let low_mask = _mm256_set1_epi8(0x0F);

        // Two accumulators for 32 u16 results
        let mut accum_lo = _mm256_setzero_si256(); // neighbors 0-15
        let mut accum_hi = _mm256_setzero_si256(); // neighbors 16-31

        for sq in 0..luts_lo.len() {
            let base = sq * BATCH_SIZE;

            // Broadcast 16-byte LUTs to 256-bit registers
            let lut_lo_128 = _mm_loadu_si128(luts_lo[sq].as_ptr() as *const __m128i);
            let lut_hi_128 = _mm_loadu_si128(luts_hi[sq].as_ptr() as *const __m128i);
            let lut_lo_vec = _mm256_broadcastsi128_si256(lut_lo_128);
            let lut_hi_vec = _mm256_broadcastsi128_si256(lut_hi_128);

            // Load 32 codes
            let codes = _mm256_loadu_si256(interleaved_codes.as_ptr().add(base) as *const __m256i);

            // Lo nibble lookups using lut_lo
            let idx_lo = _mm256_and_si256(codes, low_mask);
            let vals_lo = _mm256_shuffle_epi8(lut_lo_vec, idx_lo);

            // Hi nibble lookups using lut_hi
            let idx_hi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), low_mask);
            let vals_hi = _mm256_shuffle_epi8(lut_hi_vec, idx_hi);

            // Add lo + hi as u8 (safe because max is 127+127=254)
            // Then widen to u16 and accumulate
            // Note: We need to handle the 32 u8 -> 32 u16 conversion carefully
            // AVX2 can only widen 16 u8 -> 16 u16 at a time

            // Extract low 16 bytes, widen to u16, accumulate
            let vals_lo_128 = _mm256_castsi256_si128(vals_lo);
            let vals_hi_128 = _mm256_castsi256_si128(vals_hi);
            let sum_lo_16 = _mm256_cvtepu8_epi16(vals_lo_128);
            let sum_hi_16 = _mm256_cvtepu8_epi16(vals_hi_128);
            accum_lo = _mm256_add_epi16(accum_lo, sum_lo_16);
            accum_lo = _mm256_add_epi16(accum_lo, sum_hi_16);

            // Extract high 16 bytes, widen to u16, accumulate
            let vals_lo_high = _mm256_extracti128_si256(vals_lo, 1);
            let vals_hi_high = _mm256_extracti128_si256(vals_hi, 1);
            let sum_lo_high_16 = _mm256_cvtepu8_epi16(vals_lo_high);
            let sum_hi_high_16 = _mm256_cvtepu8_epi16(vals_hi_high);
            accum_hi = _mm256_add_epi16(accum_hi, sum_lo_high_16);
            accum_hi = _mm256_add_epi16(accum_hi, sum_hi_high_16);
        }

        let mut results = [0u16; BATCH_SIZE];
        _mm256_storeu_si256(results.as_mut_ptr() as *mut __m256i, accum_lo);
        _mm256_storeu_si256(results.as_mut_ptr().add(16) as *mut __m256i, accum_hi);
        results
    }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm256_castsi256_si128, _mm256_extracti128_si256};

/// Scalar fallback for platforms without SIMD
#[must_use]
pub fn fastscan_batch_scalar(
    luts_lo: &[[u8; 16]],
    luts_hi: &[[u8; 16]],
    interleaved_codes: &[u8],
) -> [u16; BATCH_SIZE] {
    let mut results = [0u16; BATCH_SIZE];

    for (sq, (lut_lo, lut_hi)) in luts_lo.iter().zip(luts_hi.iter()).enumerate() {
        let base = sq * BATCH_SIZE;
        for n in 0..BATCH_SIZE {
            let code = interleaved_codes[base + n];
            let lo_idx = (code & 0x0F) as usize;
            let hi_idx = ((code >> 4) & 0x0F) as usize;
            results[n] += lut_lo[lo_idx] as u16 + lut_hi[hi_idx] as u16;
        }
    }

    results
}

/// Choose the best FastScan implementation for the current platform
#[inline]
#[must_use]
pub fn fastscan_batch(
    luts_lo: &[[u8; 16]],
    luts_hi: &[[u8; 16]],
    interleaved_codes: &[u8],
) -> [u16; BATCH_SIZE] {
    #[cfg(target_arch = "aarch64")]
    {
        fastscan_batch_neon(luts_lo, luts_hi, interleaved_codes)
    }
    #[cfg(target_arch = "x86_64")]
    {
        fastscan_batch_avx2(luts_lo, luts_hi, interleaved_codes)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        fastscan_batch_scalar(luts_lo, luts_hi, interleaved_codes)
    }
}

/// Convenience wrapper using FastScanLUT struct
#[inline]
#[must_use]
pub fn fastscan_batch_with_lut(lut: &FastScanLUT, interleaved_codes: &[u8]) -> [u16; BATCH_SIZE] {
    fastscan_batch(lut.luts_lo(), lut.luts_hi(), interleaved_codes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fastscan_scalar() {
        // Create simple LUTs (distance = code value)
        let luts_lo: Vec<[u8; 16]> = (0..4).map(|_| core::array::from_fn(|i| i as u8)).collect();
        let luts_hi: Vec<[u8; 16]> = (0..4).map(|_| core::array::from_fn(|i| i as u8)).collect();

        // Create interleaved codes: all zeros
        let codes = vec![0u8; 4 * BATCH_SIZE];

        let results = fastscan_batch_scalar(&luts_lo, &luts_hi, &codes);

        // All distances should be 0 (code 0 maps to distance 0)
        for &r in &results {
            assert_eq!(r, 0);
        }
    }

    #[test]
    fn test_fastscan_scalar_nonzero() {
        // LUT where each code maps to its value
        let luts_lo: Vec<[u8; 16]> = (0..2).map(|_| core::array::from_fn(|i| i as u8)).collect();
        let luts_hi: Vec<[u8; 16]> = (0..2).map(|_| core::array::from_fn(|i| i as u8)).collect();

        // Create codes: first neighbor has all 0x11 (lo=1, hi=1)
        let mut codes = vec![0u8; 2 * BATCH_SIZE];
        codes[0] = 0x11; // sq0, neighbor 0: lo=1, hi=1
        codes[BATCH_SIZE] = 0x22; // sq1, neighbor 0: lo=2, hi=2

        let results = fastscan_batch_scalar(&luts_lo, &luts_hi, &codes);

        // Neighbor 0: (1+1) + (2+2) = 6
        assert_eq!(results[0], 6);
        // Other neighbors: all 0
        assert_eq!(results[1], 0);
    }

    #[test]
    fn test_fastscan_matches_scalar() {
        // Create random-ish LUTs
        let luts_lo: Vec<[u8; 16]> = (0..8)
            .map(|sq| core::array::from_fn(|i| ((sq * 17 + i * 7) % 100) as u8))
            .collect();
        let luts_hi: Vec<[u8; 16]> = (0..8)
            .map(|sq| core::array::from_fn(|i| ((sq * 13 + i * 11) % 100) as u8))
            .collect();

        // Create random-ish codes
        let mut codes = vec![0u8; 8 * BATCH_SIZE];
        for (i, code) in codes.iter_mut().enumerate() {
            *code = ((i * 31 + 17) % 256) as u8;
        }

        let scalar_results = fastscan_batch_scalar(&luts_lo, &luts_hi, &codes);
        let simd_results = fastscan_batch(&luts_lo, &luts_hi, &codes);

        // SIMD should match scalar
        for (i, (&scalar, &simd)) in scalar_results.iter().zip(simd_results.iter()).enumerate() {
            assert_eq!(
                scalar, simd,
                "Mismatch at neighbor {i}: scalar={scalar}, simd={simd}"
            );
        }
    }
}
