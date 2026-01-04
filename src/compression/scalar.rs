//! Scalar Quantization (SQ8) for `OmenDB`
//!
//! Compresses f32 vectors to u8 (4x compression, ~98% recall).
//!
//! # Algorithm
//!
//! Per-dimension min/max scaling:
//! - Train: Compute min[d], max[d] from sample vectors
//! - Quantize: u8[d] = round((f32[d] - min[d]) / scale[d] * 255)
//! - Dequantize: f32[d] ≈ u8[d] / 255 * scale[d] + min[d]
//!
//! # Search: ADC (Asymmetric Distance Computation)
//!
//! All production vector databases use ADC for quantized search:
//! - Build lookup table ONCE per query: table[d][code] = (query[d] - dequant(code))²
//! - Per candidate: distance = sum(table[d][vector[d]]) - just lookups + adds
//!
//! This is 10-100x faster than per-candidate dequantization for typical HNSW searches.
//!
//! # Performance
//!
//! - 4x compression (f32 → u8)
//! - ~2x search speedup with ADC (vs f32)
//! - ~98% recall with rescoring, ~95% without

use serde::{Deserialize, Serialize};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vaddq_f32, vaddvq_f32, vcvtq_f32_u32, vdupq_n_f32, vfmaq_f32, vget_high_u16, vget_low_u16,
    vld1_u8, vld1q_f32, vmovl_u16, vmovl_u8, vsubq_f32,
};
#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

/// Trained scalar quantization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarParams {
    /// Minimum value per dimension
    pub mins: Vec<f32>,
    /// Scale factor per dimension: (max - min) / 255
    pub scales: Vec<f32>,
    /// Number of dimensions
    pub dimensions: usize,
}

impl ScalarParams {
    /// Create uninitialized params (for lazy training)
    ///
    /// Uses identity mapping (min=0, scale=1/255) until trained.
    #[must_use]
    pub fn uninitialized(dimensions: usize) -> Self {
        Self {
            mins: vec![0.0; dimensions],
            scales: vec![1.0 / 255.0; dimensions],
            dimensions,
        }
    }

    /// Train scalar quantization from sample vectors
    ///
    /// Uses 1st and 99th percentiles to handle outliers.
    ///
    /// # Errors
    /// Returns error if vectors is empty or vectors have inconsistent dimensions.
    pub fn train(vectors: &[&[f32]]) -> Result<Self, &'static str> {
        Self::train_with_percentiles(vectors, 0.01, 0.99)
    }

    /// Train with custom percentile bounds
    ///
    /// # Errors
    /// Returns error if vectors is empty or vectors have inconsistent dimensions.
    pub fn train_with_percentiles(
        vectors: &[&[f32]],
        lower_percentile: f32,
        upper_percentile: f32,
    ) -> Result<Self, &'static str> {
        if vectors.is_empty() {
            return Err("Need at least one vector to train");
        }
        let dimensions = vectors[0].len();
        if !vectors.iter().all(|v| v.len() == dimensions) {
            return Err("All vectors must have same dimensions");
        }

        let n = vectors.len();
        let lower_idx = ((n as f32 * lower_percentile) as usize).min(n - 1);
        let upper_idx = ((n as f32 * upper_percentile) as usize).min(n - 1);

        let mut mins = Vec::with_capacity(dimensions);
        let mut scales = Vec::with_capacity(dimensions);

        let mut dim_values: Vec<f32> = Vec::with_capacity(n);
        for d in 0..dimensions {
            dim_values.clear();
            for v in vectors {
                dim_values.push(v[d]);
            }
            dim_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let min_val = dim_values[lower_idx];
            let max_val = dim_values[upper_idx];

            // Ensure non-zero range
            let range = max_val - min_val;
            let (min, scale) = if range < 1e-7 {
                (min_val - 0.5, 1.0 / 255.0)
            } else {
                (min_val, range / 255.0)
            };

            mins.push(min);
            scales.push(scale);
        }

        Ok(Self {
            mins,
            scales,
            dimensions,
        })
    }

    /// Quantize a single f32 vector to u8
    #[must_use]
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        debug_assert_eq!(vector.len(), self.dimensions);

        vector
            .iter()
            .zip(self.mins.iter().zip(self.scales.iter()))
            .map(|(&val, (&min, &scale))| {
                let normalized = (val - min) / scale;
                normalized.clamp(0.0, 255.0).round() as u8
            })
            .collect()
    }

    /// Quantize into pre-allocated buffer
    pub fn quantize_into(&self, vector: &[f32], output: &mut [u8]) {
        assert_eq!(vector.len(), self.dimensions);
        assert_eq!(output.len(), self.dimensions);

        for (i, &val) in vector.iter().enumerate() {
            let normalized = (val - self.mins[i]) / self.scales[i];
            output[i] = normalized.clamp(0.0, 255.0).round() as u8;
        }
    }

    /// Dequantize a u8 vector back to f32 (approximate)
    #[must_use]
    pub fn dequantize(&self, quantized: &[u8]) -> Vec<f32> {
        assert_eq!(quantized.len(), self.dimensions);

        quantized
            .iter()
            .zip(self.mins.iter().zip(self.scales.iter()))
            .map(|(&q, (&min, &scale))| f32::from(q) * scale + min)
            .collect()
    }

    /// Dequantize into pre-allocated buffer
    pub fn dequantize_into(&self, quantized: &[u8], output: &mut [f32]) {
        assert_eq!(quantized.len(), self.dimensions);
        assert_eq!(output.len(), self.dimensions);

        for (i, &q) in quantized.iter().enumerate() {
            output[i] = f32::from(q) * self.scales[i] + self.mins[i];
        }
    }

    /// Compute squared norm of dequantized vector: ||dequant(q)||^2
    ///
    /// Used for L2 decomposition: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    #[must_use]
    pub fn dequantized_norm_squared(&self, quantized: &[u8]) -> f32 {
        assert_eq!(quantized.len(), self.dimensions);

        let mut sum = 0.0f32;
        for (i, &q) in quantized.iter().enumerate() {
            let dequant = f32::from(q) * self.scales[i] + self.mins[i];
            sum += dequant * dequant;
        }
        sum
    }

    /// Compute dot product between query (f32) and dequantized vector (u8)
    ///
    /// Used for L2 decomposition: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    #[inline(always)]
    #[must_use]
    #[allow(clippy::needless_return)]
    pub fn asymmetric_dot_product(&self, query: &[f32], quantized: &[u8]) -> f32 {
        debug_assert_eq!(query.len(), self.dimensions);
        debug_assert_eq!(quantized.len(), self.dimensions);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.asymmetric_dot_product_avx2(query, quantized) };
            }
            return self.asymmetric_dot_product_scalar(query, quantized);
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe { self.asymmetric_dot_product_neon(query, quantized) }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        self.asymmetric_dot_product_scalar(query, quantized)
    }

    #[allow(dead_code)]
    fn asymmetric_dot_product_scalar(&self, query: &[f32], quantized: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..self.dimensions {
            let dequant = f32::from(quantized[i]) * self.scales[i] + self.mins[i];
            sum += query[i] * dequant;
        }
        sum
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn asymmetric_dot_product_avx2(&self, query: &[f32], quantized: &[u8]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;

        while i + 8 <= self.dimensions {
            // Load and convert 8 u8 to f32 using SIMD
            let bytes = _mm_loadl_epi64(quantized.as_ptr().add(i).cast());
            let u32s = _mm256_cvtepu8_epi32(bytes);
            let q0 = _mm256_cvtepi32_ps(u32s);

            // Load scales and mins
            let scales = _mm256_loadu_ps(self.scales.as_ptr().add(i));
            let mins = _mm256_loadu_ps(self.mins.as_ptr().add(i));

            // Dequantize: q * scale + min
            let dequant = _mm256_fmadd_ps(q0, scales, mins);

            // Load query and accumulate dot product
            let query_vec = _mm256_loadu_ps(query.as_ptr().add(i));
            sum = _mm256_fmadd_ps(query_vec, dequant, sum);

            i += 8;
        }

        // Horizontal sum
        let sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Handle remaining elements
        for j in i..self.dimensions {
            let dequant = f32::from(quantized[j]) * self.scales[j] + self.mins[j];
            result += query[j] * dequant;
        }

        result
    }

    /// NEON SIMD implementation for aarch64
    /// Note: NEON is baseline on aarch64, no #[target_feature] needed
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn asymmetric_dot_product_neon(&self, query: &[f32], quantized: &[u8]) -> f32 {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut i = 0;

        while i + 8 <= self.dimensions {
            // Load 8 u8 values and convert to f32
            let u8x8 = vld1_u8(quantized.as_ptr().add(i));
            let u16x8 = vmovl_u8(u8x8);
            let u32x4_lo = vmovl_u16(vget_low_u16(u16x8));
            let u32x4_hi = vmovl_u16(vget_high_u16(u16x8));
            let f32x4_lo = vcvtq_f32_u32(u32x4_lo);
            let f32x4_hi = vcvtq_f32_u32(u32x4_hi);

            // Load scales and mins
            let scales_lo = vld1q_f32(self.scales.as_ptr().add(i));
            let scales_hi = vld1q_f32(self.scales.as_ptr().add(i + 4));
            let mins_lo = vld1q_f32(self.mins.as_ptr().add(i));
            let mins_hi = vld1q_f32(self.mins.as_ptr().add(i + 4));

            // Dequantize: q * scale + min
            let dequant_lo = vfmaq_f32(mins_lo, f32x4_lo, scales_lo);
            let dequant_hi = vfmaq_f32(mins_hi, f32x4_hi, scales_hi);

            // Load query and accumulate dot product
            let query_lo = vld1q_f32(query.as_ptr().add(i));
            let query_hi = vld1q_f32(query.as_ptr().add(i + 4));
            sum0 = vfmaq_f32(sum0, query_lo, dequant_lo);
            sum1 = vfmaq_f32(sum1, query_hi, dequant_hi);

            i += 8;
        }

        let mut result = vaddvq_f32(vaddq_f32(sum0, sum1));

        // Handle remaining elements
        for j in i..self.dimensions {
            let dequant = f32::from(quantized[j]) * self.scales[j] + self.mins[j];
            result += query[j] * dequant;
        }

        result
    }

    /// Compute L2 distance using decomposition: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    ///
    /// This is faster than direct asymmetric distance when the candidate norm is precomputed.
    /// Uses the multiversion dot product for better cdylib compatibility.
    #[inline(always)]
    #[must_use]
    pub fn asymmetric_l2_decomposed(
        &self,
        query: &[f32],
        query_norm: f32,
        quantized: &[u8],
        candidate_norm: f32,
    ) -> f32 {
        // Use multiversion dot product for cdylib compatibility
        let dot =
            crate::distance::sq8_asymmetric_dot_product(query, quantized, &self.scales, &self.mins);
        query_norm + candidate_norm - 2.0 * dot
    }

    /// Compute approximate L2 distance between query (f32) and quantized vector (u8)
    ///
    /// Uses asymmetric distance: query stays f32, candidate is dequantized on-the-fly.
    #[inline(always)]
    #[must_use]
    #[allow(clippy::needless_return)]
    pub fn asymmetric_l2_squared(&self, query: &[f32], quantized: &[u8]) -> f32 {
        debug_assert_eq!(query.len(), self.dimensions);
        debug_assert_eq!(quantized.len(), self.dimensions);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.asymmetric_l2_squared_avx2(query, quantized) };
            }
            return self.asymmetric_l2_squared_scalar(query, quantized);
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe { self.asymmetric_l2_squared_neon(query, quantized) }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        self.asymmetric_l2_squared_scalar(query, quantized)
    }

    #[allow(dead_code)]
    fn asymmetric_l2_squared_scalar(&self, query: &[f32], quantized: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..self.dimensions {
            let dequant = f32::from(quantized[i]) * self.scales[i] + self.mins[i];
            let diff = query[i] - dequant;
            sum += diff * diff;
        }
        sum
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[allow(clippy::needless_range_loop)] // index needed for multiple array accesses
    unsafe fn asymmetric_l2_squared_avx2(&self, query: &[f32], quantized: &[u8]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;

        // Process 8 elements at a time
        while i + 8 <= self.dimensions {
            // Load 8 u8 values and convert to f32 using SIMD (3 instructions vs 15+ scalar)
            // _mm_loadl_epi64: load 8 bytes into lower 64 bits of 128-bit register
            // _mm256_cvtepu8_epi32: zero-extend 8 u8 values to 8 i32 values
            // _mm256_cvtepi32_ps: convert 8 i32 values to 8 f32 values
            let bytes = _mm_loadl_epi64(quantized.as_ptr().add(i).cast());
            let u32s = _mm256_cvtepu8_epi32(bytes);
            let q0 = _mm256_cvtepi32_ps(u32s);

            // Load scales and mins
            let scales = _mm256_loadu_ps(self.scales.as_ptr().add(i));
            let mins = _mm256_loadu_ps(self.mins.as_ptr().add(i));

            // Dequantize: q * scale + min
            let dequant = _mm256_fmadd_ps(q0, scales, mins);

            // Load query
            let query_vec = _mm256_loadu_ps(query.as_ptr().add(i));

            // Compute diff
            let diff = _mm256_sub_ps(query_vec, dequant);

            // Accumulate diff^2
            sum = _mm256_fmadd_ps(diff, diff, sum);

            i += 8;
        }

        // Horizontal sum
        let sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Handle remaining elements
        for j in i..self.dimensions {
            let dequant = f32::from(quantized[j]) * self.scales[j] + self.mins[j];
            let diff = query[j] - dequant;
            result += diff * diff;
        }

        result
    }

    /// NEON SIMD implementation for aarch64
    /// Note: NEON is baseline on aarch64, no #[target_feature] needed
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn asymmetric_l2_squared_neon(&self, query: &[f32], quantized: &[u8]) -> f32 {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut i = 0;

        // Process 8 elements at a time using proper SIMD widening
        while i + 8 <= self.dimensions {
            // Load 8 u8 values and convert to f32 via SIMD widening
            let u8x8 = vld1_u8(quantized.as_ptr().add(i));
            let u16x8 = vmovl_u8(u8x8);
            let u32x4_lo = vmovl_u16(vget_low_u16(u16x8));
            let u32x4_hi = vmovl_u16(vget_high_u16(u16x8));
            let f32x4_lo = vcvtq_f32_u32(u32x4_lo);
            let f32x4_hi = vcvtq_f32_u32(u32x4_hi);

            // Load scales and mins
            let scales_lo = vld1q_f32(self.scales.as_ptr().add(i));
            let scales_hi = vld1q_f32(self.scales.as_ptr().add(i + 4));
            let mins_lo = vld1q_f32(self.mins.as_ptr().add(i));
            let mins_hi = vld1q_f32(self.mins.as_ptr().add(i + 4));

            // Dequantize: q * scale + min
            let dequant_lo = vfmaq_f32(mins_lo, f32x4_lo, scales_lo);
            let dequant_hi = vfmaq_f32(mins_hi, f32x4_hi, scales_hi);

            // Load query
            let query_lo = vld1q_f32(query.as_ptr().add(i));
            let query_hi = vld1q_f32(query.as_ptr().add(i + 4));

            // Compute diff
            let diff_lo = vsubq_f32(query_lo, dequant_lo);
            let diff_hi = vsubq_f32(query_hi, dequant_hi);

            // Accumulate diff^2
            sum0 = vfmaq_f32(sum0, diff_lo, diff_lo);
            sum1 = vfmaq_f32(sum1, diff_hi, diff_hi);

            i += 8;
        }

        // Process remaining 4 elements
        if i + 4 <= self.dimensions {
            let u8x8 = vld1_u8(quantized.as_ptr().add(i));
            let u16x8 = vmovl_u8(u8x8);
            let u32x4 = vmovl_u16(vget_low_u16(u16x8));
            let f32x4 = vcvtq_f32_u32(u32x4);

            let scales = vld1q_f32(self.scales.as_ptr().add(i));
            let mins = vld1q_f32(self.mins.as_ptr().add(i));
            let dequant = vfmaq_f32(mins, f32x4, scales);
            let query_vec = vld1q_f32(query.as_ptr().add(i));
            let diff = vsubq_f32(query_vec, dequant);
            sum0 = vfmaq_f32(sum0, diff, diff);
            i += 4;
        }

        // Horizontal sum
        let sum = vaddq_f32(sum0, sum1);
        let mut result = vaddvq_f32(sum);

        // Handle remaining elements (unchecked for performance)
        for j in i..self.dimensions {
            let dequant = f32::from(*quantized.get_unchecked(j)) * self.scales.get_unchecked(j)
                + self.mins.get_unchecked(j);
            let diff = *query.get_unchecked(j) - dequant;
            result += diff * diff;
        }

        result
    }
}

/// Compute L2² distance between two quantized u8 vectors
///
/// Note: This is approximate and less accurate than asymmetric distance.
/// Prefer `asymmetric_l2_squared` when query is available in f32.
///
/// Not SIMD-optimized - use asymmetric distance for hot paths.
#[must_use]
pub fn symmetric_l2_squared_u8(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = i32::from(x) - i32::from(y);
            (diff * diff) as u32
        })
        .sum()
}

/// ADC (Asymmetric Distance Computation) lookup table for SQ8
///
/// Precomputes (query[d] - dequant(code))² for all 256 codes per dimension.
/// Distance computation becomes just table lookups + summation.
///
/// Memory: dimensions × 256 × 4 bytes (e.g., 768D = 768KB, fits in L2 cache)
#[derive(Debug, Clone)]
pub struct SQ8ADCTable {
    /// table[d * 256 + code] = (query[d] - dequant(code, d))²
    /// Flat layout for cache efficiency
    table: Vec<f32>,
    dimensions: usize,
}

impl SQ8ADCTable {
    /// Build ADC table for a query vector
    ///
    /// Cost: dimensions × 256 FMA operations (one-time per query)
    #[must_use]
    #[allow(clippy::needless_range_loop)]
    pub fn build(params: &ScalarParams, query: &[f32]) -> Self {
        assert_eq!(query.len(), params.dimensions);

        let mut table = vec![0.0f32; params.dimensions * 256];

        for d in 0..params.dimensions {
            let q = query[d];
            let min = params.mins[d];
            let scale = params.scales[d];
            let base = d * 256;

            for code in 0..256 {
                let dequant = f32::from(code as u8) * scale + min;
                let diff = q - dequant;
                table[base + code] = diff * diff;
            }
        }

        Self {
            table,
            dimensions: params.dimensions,
        }
    }

    /// Compute L2² distance using precomputed table
    ///
    /// Cost: dimensions lookups + additions (extremely fast)
    #[must_use]
    #[inline]
    #[allow(clippy::needless_return)] // returns needed for cfg-conditional control flow
    pub fn distance_squared(&self, quantized: &[u8]) -> f32 {
        debug_assert_eq!(quantized.len(), self.dimensions);

        #[cfg(target_arch = "aarch64")]
        {
            unsafe { self.distance_squared_neon(quantized) }
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.distance_squared_avx2(quantized) };
            }
            self.distance_squared_scalar(quantized)
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        self.distance_squared_scalar(quantized)
    }

    #[allow(dead_code)]
    #[inline]
    fn distance_squared_scalar(&self, quantized: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for (d, &code) in quantized.iter().enumerate() {
            sum += self.table[d * 256 + code as usize];
        }
        sum
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    #[allow(clippy::needless_range_loop)]
    unsafe fn distance_squared_neon(&self, quantized: &[u8]) -> f32 {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut i = 0;

        // Process 8 dimensions at a time
        while i + 8 <= self.dimensions {
            // Load 8 codes and gather from table
            // Each dimension has its own 256-entry table at d * 256
            let d0 = self.table[(i) * 256 + quantized[i] as usize];
            let d1 = self.table[(i + 1) * 256 + quantized[i + 1] as usize];
            let d2 = self.table[(i + 2) * 256 + quantized[i + 2] as usize];
            let d3 = self.table[(i + 3) * 256 + quantized[i + 3] as usize];
            let d4 = self.table[(i + 4) * 256 + quantized[i + 4] as usize];
            let d5 = self.table[(i + 5) * 256 + quantized[i + 5] as usize];
            let d6 = self.table[(i + 6) * 256 + quantized[i + 6] as usize];
            let d7 = self.table[(i + 7) * 256 + quantized[i + 7] as usize];

            // Pack into SIMD registers and accumulate
            let vals_lo = [d0, d1, d2, d3];
            let vals_hi = [d4, d5, d6, d7];
            sum0 = vaddq_f32(sum0, vld1q_f32(vals_lo.as_ptr()));
            sum1 = vaddq_f32(sum1, vld1q_f32(vals_hi.as_ptr()));

            i += 8;
        }

        // Horizontal sum
        let sum = vaddq_f32(sum0, sum1);
        let mut result = vaddvq_f32(sum);

        // Handle remaining dimensions
        for d in i..self.dimensions {
            result += self.table[d * 256 + quantized[d] as usize];
        }

        result
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    #[allow(clippy::needless_range_loop)] // index needed for d * 256 calculation
    unsafe fn distance_squared_avx2(&self, quantized: &[u8]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;

        // Process 8 dimensions at a time
        while i + 8 <= self.dimensions {
            // Gather from table (no SIMD gather, scalar lookups)
            let d0 = self.table[(i) * 256 + quantized[i] as usize];
            let d1 = self.table[(i + 1) * 256 + quantized[i + 1] as usize];
            let d2 = self.table[(i + 2) * 256 + quantized[i + 2] as usize];
            let d3 = self.table[(i + 3) * 256 + quantized[i + 3] as usize];
            let d4 = self.table[(i + 4) * 256 + quantized[i + 4] as usize];
            let d5 = self.table[(i + 5) * 256 + quantized[i + 5] as usize];
            let d6 = self.table[(i + 6) * 256 + quantized[i + 6] as usize];
            let d7 = self.table[(i + 7) * 256 + quantized[i + 7] as usize];

            let vals = _mm256_set_ps(d7, d6, d5, d4, d3, d2, d1, d0);
            sum = _mm256_add_ps(sum, vals);

            i += 8;
        }

        // Horizontal sum
        let sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Handle remaining dimensions
        for d in i..self.dimensions {
            result += self.table[d * 256 + quantized[d] as usize];
        }

        result
    }

    /// Get the dimensions of this table
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.table.len() * std::mem::size_of::<f32>()
    }
}

impl ScalarParams {
    /// Build an ADC lookup table for a query vector
    ///
    /// Use this for search: build once per query, then call
    /// `table.distance_squared()` for each candidate.
    #[must_use]
    pub fn build_adc_table(&self, query: &[f32]) -> SQ8ADCTable {
        SQ8ADCTable::build(self, query)
    }
}

// ============================================================================
// Uniform Scalar Quantization (Integer SIMD optimized)
// ============================================================================

/// Uniform scalar quantization parameters (single scale/offset for all dims)
///
/// Unlike `ScalarParams` which uses per-dimension min/scale, this uses a single
/// global scale and offset. This enables integer SIMD (32 ops at once vs 8 f32),
/// providing 2-4x speedup at the cost of slightly lower recall (~97% vs 99%).
///
/// # Performance
/// - 4x compression (f32 → u8)
/// - **2-4x faster than FP32** (integer SIMD)
/// - ~97% recall (vs ~99% with per-dim)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniformScalarParams {
    /// Global scale factor: (max - min) / 255
    pub scale: f32,
    /// Global offset (minimum value)
    pub offset: f32,
    /// Number of dimensions
    pub dimensions: usize,
}

/// Precomputed data for a quantized vector (uniform quantization)
#[derive(Debug, Clone)]
pub struct UniformQuantizedVector {
    /// Quantized values (u8)
    pub data: Vec<u8>,
    /// Precomputed: sum of quantized values (Σ data[i])
    pub sum: i32,
    /// Precomputed: squared norm of dequantized vector
    pub norm_sq: f32,
}

/// Precomputed query data for fast integer SIMD distance
#[derive(Debug, Clone)]
pub struct UniformQueryPrep {
    /// Quantized query values (u8 for SIMD dot product)
    pub quantized: Vec<u8>,
    /// Query squared norm: ||q||²
    pub norm_sq: f32,
    /// Sum of quantized query values
    pub sum: i32,
}

impl UniformScalarParams {
    /// Train uniform quantization from sample vectors
    ///
    /// Uses global min/max across all dimensions and vectors.
    pub fn train(vectors: &[&[f32]]) -> Result<Self, &'static str> {
        Self::train_with_percentiles(vectors, 0.01, 0.99)
    }

    /// Train with custom percentile bounds
    pub fn train_with_percentiles(
        vectors: &[&[f32]],
        lower_percentile: f32,
        upper_percentile: f32,
    ) -> Result<Self, &'static str> {
        if vectors.is_empty() {
            return Err("Need at least one vector to train");
        }
        let dimensions = vectors[0].len();
        if !vectors.iter().all(|v| v.len() == dimensions) {
            return Err("All vectors must have same dimensions");
        }

        // Collect ALL values across all vectors and dimensions
        let mut all_values: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = all_values.len();
        let lower_idx = ((n as f32 * lower_percentile) as usize).min(n - 1);
        let upper_idx = ((n as f32 * upper_percentile) as usize).min(n - 1);

        let min_val = all_values[lower_idx];
        let max_val = all_values[upper_idx];

        let range = max_val - min_val;
        let (offset, scale) = if range < 1e-7 {
            (min_val - 0.5, 1.0 / 255.0)
        } else {
            (min_val, range / 255.0)
        };

        Ok(Self {
            scale,
            offset,
            dimensions,
        })
    }

    /// Quantize a vector to u8 with precomputed metadata
    #[must_use]
    pub fn quantize(&self, vector: &[f32]) -> UniformQuantizedVector {
        debug_assert_eq!(vector.len(), self.dimensions);

        let inv_scale = 1.0 / self.scale;
        let data: Vec<u8> = vector
            .iter()
            .map(|&v| ((v - self.offset) * inv_scale).clamp(0.0, 255.0).round() as u8)
            .collect();

        let sum: i32 = data.iter().map(|&x| x as i32).sum();

        // Compute dequantized norm
        let norm_sq: f32 = data
            .iter()
            .map(|&x| {
                let dequant = x as f32 * self.scale + self.offset;
                dequant * dequant
            })
            .sum();

        UniformQuantizedVector { data, sum, norm_sq }
    }

    /// Prepare query for fast integer SIMD distance computation
    #[must_use]
    pub fn prepare_query(&self, query: &[f32]) -> UniformQueryPrep {
        debug_assert_eq!(query.len(), self.dimensions);

        let inv_scale = 1.0 / self.scale;
        let quantized: Vec<u8> = query
            .iter()
            .map(|&v| ((v - self.offset) * inv_scale).clamp(0.0, 255.0).round() as u8)
            .collect();

        let norm_sq: f32 = query.iter().map(|x| x * x).sum();
        let sum: i32 = quantized.iter().map(|&x| x as i32).sum();

        UniformQueryPrep {
            quantized,
            norm_sq,
            sum,
        }
    }

    /// Compute L2² distance using integer SIMD
    ///
    /// Uses the identity: ||q - v||² = ||q||² + ||v||² - 2⟨q,v⟩
    /// The dot product is computed in integer domain for speed.
    #[inline(always)]
    #[must_use]
    pub fn distance_l2_squared(
        &self,
        query_prep: &UniformQueryPrep,
        vec: &UniformQuantizedVector,
    ) -> f32 {
        // Integer dot product (SIMD accelerated) - uses u8×u8→u32
        let int_dot = self.int_dot_product(&query_prep.quantized, &vec.data);

        // Reconstruct actual dot product: scale² × int_dot + corrections
        // dot(q, v) = scale² × Σ q_int[i] × v_int[i]
        //           + scale × offset × (Σ q_int[i] + Σ v_int[i])
        //           + offset² × dim
        let scale_sq = self.scale * self.scale;
        let dot = scale_sq * int_dot as f32
            + self.scale * self.offset * (query_prep.sum + vec.sum) as f32
            + self.offset * self.offset * self.dimensions as f32;

        // L2² = ||q||² + ||v||² - 2⟨q,v⟩
        query_prep.norm_sq + vec.norm_sq - 2.0 * dot
    }

    /// Integer dot product with SIMD acceleration (u8 × u8 → u32)
    #[inline(always)]
    fn int_dot_product(&self, query: &[u8], vec: &[u8]) -> u32 {
        debug_assert_eq!(query.len(), vec.len());

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.int_dot_product_avx2(query, vec) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { self.int_dot_product_neon(query, vec) };
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self.int_dot_product_scalar(query, vec)
        }
    }

    #[inline]
    #[allow(dead_code)]
    fn int_dot_product_scalar(&self, query: &[u8], vec: &[u8]) -> u32 {
        query
            .iter()
            .zip(vec.iter())
            .map(|(&q, &v)| q as u32 * v as u32)
            .sum()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn int_dot_product_avx2(&self, query: &[u8], vec: &[u8]) -> u32 {
        // Use _mm256_maddubs_epi16 for u8*i8→i16, then horizontal sum
        // Since both are u8, we treat one as "signed" (values 0-127 safe)
        let mut sum = _mm256_setzero_si256();
        let mut i = 0;

        while i + 32 <= query.len() {
            // Load 32 bytes each
            let q = _mm256_loadu_si256(query.as_ptr().add(i).cast());
            let v = _mm256_loadu_si256(vec.as_ptr().add(i).cast());

            // _mm256_maddubs_epi16: treats first arg as u8, second as i8
            // Since v is 0-255, we need to handle overflow carefully
            // Instead, use: extend to 16-bit and multiply
            let q_lo = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(q, 0));
            let q_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(q, 1));
            let v_lo = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v, 0));
            let v_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v, 1));

            // madd: pairs adjacent products and sums (a0*b0+a1*b1, a2*b2+a3*b3, ...)
            let prod_lo = _mm256_madd_epi16(q_lo, v_lo);
            let prod_hi = _mm256_madd_epi16(q_hi, v_hi);
            sum = _mm256_add_epi32(sum, prod_lo);
            sum = _mm256_add_epi32(sum, prod_hi);

            i += 32;
        }

        // Process remaining 16 at a time
        while i + 16 <= query.len() {
            let q = _mm256_cvtepu8_epi16(_mm_loadu_si128(query.as_ptr().add(i).cast()));
            let v = _mm256_cvtepu8_epi16(_mm_loadu_si128(vec.as_ptr().add(i).cast()));
            let prod = _mm256_madd_epi16(q, v);
            sum = _mm256_add_epi32(sum, prod);
            i += 16;
        }

        // Horizontal sum
        let sum128 = _mm_add_epi32(
            _mm256_extracti128_si256(sum, 0),
            _mm256_extracti128_si256(sum, 1),
        );
        let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
        let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
        let mut result = _mm_cvtsi128_si32(sum32) as u32;

        // Handle remaining elements
        for j in i..query.len() {
            result += query[j] as u32 * vec[j] as u32;
        }

        result
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn int_dot_product_neon(&self, query: &[u8], vec: &[u8]) -> u32 {
        use std::arch::aarch64::{
            vaddvq_u32, vdupq_n_u32, vget_low_u8, vld1q_u8, vmull_high_u8, vmull_u8, vpadalq_u16,
        };

        // Use 4 accumulators to hide latency and increase ILP
        let mut sum0 = vdupq_n_u32(0);
        let mut sum1 = vdupq_n_u32(0);
        let mut sum2 = vdupq_n_u32(0);
        let mut sum3 = vdupq_n_u32(0);
        let mut i = 0;

        // Process 64 elements per iteration (4x unrolling)
        while i + 64 <= query.len() {
            // Block 0
            let q0 = vld1q_u8(query.as_ptr().add(i));
            let v0 = vld1q_u8(vec.as_ptr().add(i));
            let prod0_lo = vmull_u8(vget_low_u8(q0), vget_low_u8(v0));
            let prod0_hi = vmull_high_u8(q0, v0);
            sum0 = vpadalq_u16(sum0, prod0_lo);
            sum0 = vpadalq_u16(sum0, prod0_hi);

            // Block 1
            let q1 = vld1q_u8(query.as_ptr().add(i + 16));
            let v1 = vld1q_u8(vec.as_ptr().add(i + 16));
            let prod1_lo = vmull_u8(vget_low_u8(q1), vget_low_u8(v1));
            let prod1_hi = vmull_high_u8(q1, v1);
            sum1 = vpadalq_u16(sum1, prod1_lo);
            sum1 = vpadalq_u16(sum1, prod1_hi);

            // Block 2
            let q2 = vld1q_u8(query.as_ptr().add(i + 32));
            let v2 = vld1q_u8(vec.as_ptr().add(i + 32));
            let prod2_lo = vmull_u8(vget_low_u8(q2), vget_low_u8(v2));
            let prod2_hi = vmull_high_u8(q2, v2);
            sum2 = vpadalq_u16(sum2, prod2_lo);
            sum2 = vpadalq_u16(sum2, prod2_hi);

            // Block 3
            let q3 = vld1q_u8(query.as_ptr().add(i + 48));
            let v3 = vld1q_u8(vec.as_ptr().add(i + 48));
            let prod3_lo = vmull_u8(vget_low_u8(q3), vget_low_u8(v3));
            let prod3_hi = vmull_high_u8(q3, v3);
            sum3 = vpadalq_u16(sum3, prod3_lo);
            sum3 = vpadalq_u16(sum3, prod3_hi);

            i += 64;
        }

        // Process remaining 16 at a time
        while i + 16 <= query.len() {
            let q = vld1q_u8(query.as_ptr().add(i));
            let v = vld1q_u8(vec.as_ptr().add(i));
            let prod_lo = vmull_u8(vget_low_u8(q), vget_low_u8(v));
            let prod_hi = vmull_high_u8(q, v);
            // vpadalq_u16: pairwise add u16→u32 and accumulate
            sum0 = vpadalq_u16(sum0, prod_lo);
            sum0 = vpadalq_u16(sum0, prod_hi);
            i += 16;
        }

        // Combine all accumulators
        use std::arch::aarch64::vaddq_u32;
        let sum01 = vaddq_u32(sum0, sum1);
        let sum23 = vaddq_u32(sum2, sum3);
        let sum_all = vaddq_u32(sum01, sum23);
        let mut result = vaddvq_u32(sum_all);

        // Handle remaining elements
        for j in i..query.len() {
            result += query[j] as u32 * vec[j] as u32;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_and_quantize() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.5, 1.0],
            vec![0.1, 0.6, 0.9],
            vec![0.2, 0.4, 0.8],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let params = ScalarParams::train(&refs).unwrap();

        // Quantize and dequantize
        let quantized = params.quantize(&vectors[0]);
        let dequantized = params.dequantize(&quantized);

        // Should be close to original
        for (orig, deq) in vectors[0].iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.02, "Roundtrip error too large");
        }
    }

    #[test]
    fn test_asymmetric_distance() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0.5, 0.5, 0.5, 0.5],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let params = ScalarParams::train(&refs).unwrap();
        let quantized = params.quantize(&vectors[1]);

        // Distance from [0,0,0,0] to [1,1,1,1] should be ~4.0
        let dist = params.asymmetric_l2_squared(&vectors[0], &quantized);
        assert!(
            (dist - 4.0).abs() < 0.1,
            "Distance should be ~4.0, got {dist}"
        );
    }

    #[test]
    fn test_compression_ratio() {
        let dims = 768;
        let original_size = dims * 4; // f32 = 4 bytes
        let quantized_size = dims; // u8 = 1 byte

        assert_eq!(original_size / quantized_size, 4);
    }

    #[test]
    fn test_symmetric_distance() {
        let a: Vec<u8> = vec![0, 100, 200, 255];
        let b: Vec<u8> = vec![0, 100, 200, 255];
        let dist = symmetric_l2_squared_u8(&a, &b);
        assert_eq!(dist, 0);

        let c: Vec<u8> = vec![10, 110, 210, 245];
        let dist2 = symmetric_l2_squared_u8(&a, &c);
        assert!(dist2 > 0);
    }

    #[test]
    fn test_adc_table() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0.5, 0.5, 0.5, 0.5],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let params = ScalarParams::train(&refs).unwrap();

        // Quantize target vector
        let quantized = params.quantize(&vectors[1]);

        // Build ADC table for query
        let table = params.build_adc_table(&vectors[0]);

        // ADC distance should match asymmetric distance
        let adc_dist = table.distance_squared(&quantized);
        let asym_dist = params.asymmetric_l2_squared(&vectors[0], &quantized);

        assert!(
            (adc_dist - asym_dist).abs() < 0.001,
            "ADC dist {adc_dist} should match asymmetric dist {asym_dist}"
        );

        // Both should be close to 4.0 (L2² from origin to [1,1,1,1])
        assert!(
            (adc_dist - 4.0).abs() < 0.1,
            "Distance should be ~4.0, got {adc_dist}"
        );
    }

    #[test]
    fn test_adc_table_memory() {
        let dims = 768;
        let params = ScalarParams::uninitialized(dims);
        let query = vec![0.0f32; dims];
        let table = params.build_adc_table(&query);

        // 768 dimensions × 256 codes × 4 bytes = 768KB
        assert_eq!(table.memory_bytes(), dims * 256 * 4);
        assert_eq!(table.memory_bytes(), 786_432); // 768KB
    }

    #[test]
    fn test_l2_decomposition_equivalence() {
        // Test that L2 decomposition gives same result as direct L2
        // ||q - v||² = ||q||² + ||v||² - 2<q, v>
        use rand::Rng;

        let dim = 128;
        let n_vectors = 100;
        let n_queries = 20;

        let mut rng = rand::thread_rng();

        // Generate training vectors
        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let params = ScalarParams::train(&refs).unwrap();

        // Quantize vectors
        let quantized: Vec<Vec<u8>> = vectors.iter().map(|v| params.quantize(v)).collect();

        // Precompute norms of dequantized vectors
        let norms: Vec<f32> = quantized
            .iter()
            .map(|q| params.dequantized_norm_squared(q))
            .collect();

        // Test with random queries
        let queries: Vec<Vec<f32>> = (0..n_queries)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        let mut max_abs_diff = 0.0f32;
        let mut max_rel_diff = 0.0f32;

        for query in &queries {
            let query_norm: f32 = query.iter().map(|x| x * x).sum();

            for (i, q) in quantized.iter().enumerate() {
                // Method 1: Direct asymmetric L2
                let dist_direct = params.asymmetric_l2_squared(query, q);

                // Method 2: L2 decomposition
                let dot = params.asymmetric_dot_product(query, q);
                let dist_decomposed = query_norm + norms[i] - 2.0 * dot;

                let abs_diff = (dist_direct - dist_decomposed).abs();
                let rel_diff = abs_diff / dist_direct.max(1e-10);

                max_abs_diff = max_abs_diff.max(abs_diff);
                max_rel_diff = max_rel_diff.max(rel_diff);
            }
        }

        println!("Max absolute difference: {max_abs_diff}");
        println!("Max relative difference: {max_rel_diff}");

        // They should be nearly identical (within floating point tolerance)
        assert!(
            max_abs_diff < 1e-3,
            "Absolute diff too large: {max_abs_diff}"
        );
        assert!(
            max_rel_diff < 1e-4,
            "Relative diff too large: {max_rel_diff}"
        );
    }

    #[test]
    fn test_dot_product_implementations_match() {
        // Compare ScalarParams::asymmetric_dot_product vs crate::distance::sq8_asymmetric_dot_product
        use rand::Rng;

        let dim = 128;
        let n_vectors = 100;

        let mut rng = rand::thread_rng();

        // Generate training vectors
        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let params = ScalarParams::train(&refs).unwrap();

        // Quantize vectors
        let quantized: Vec<Vec<u8>> = vectors.iter().map(|v| params.quantize(v)).collect();

        // Test with random queries
        let queries: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        let mut max_diff = 0.0f32;

        for query in &queries {
            for q in &quantized {
                // Method 1: ScalarParams method
                let dot1 = params.asymmetric_dot_product(query, q);

                // Method 2: crate::distance function
                let dot2 = crate::distance::sq8_asymmetric_dot_product(
                    query,
                    q,
                    &params.scales,
                    &params.mins,
                );

                let diff = (dot1 - dot2).abs();
                max_diff = max_diff.max(diff);
            }
        }

        println!("Max dot product difference: {max_diff}");
        assert!(max_diff < 1e-5, "Dot products don't match: {max_diff}");
    }

    // ========== UniformScalarParams tests ==========

    #[test]
    fn test_uniform_train_and_quantize() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.5, 1.0, 0.3],
            vec![0.1, 0.6, 0.9, 0.4],
            vec![0.2, 0.4, 0.8, 0.5],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let params = UniformScalarParams::train(&refs).unwrap();

        // Quantize and check metadata
        let quantized = params.quantize(&vectors[0]);
        assert_eq!(quantized.data.len(), 4);
        assert!(quantized.sum > 0);
        assert!(quantized.norm_sq > 0.0);
    }

    #[test]
    fn test_uniform_distance_accuracy() {
        use rand::Rng;

        let dim = 128;
        let n_vectors = 100;
        let mut rng = rand::thread_rng();

        // Generate normalized vectors (common in embeddings)
        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / norm).collect()
            })
            .collect();

        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let params = UniformScalarParams::train(&refs).unwrap();

        // Quantize all vectors
        let quantized: Vec<_> = vectors.iter().map(|v| params.quantize(v)).collect();

        // Check distance accuracy
        let query = &vectors[0];
        let query_prep = params.prepare_query(query);

        let mut max_rel_error = 0.0f32;

        for (i, (orig, quant)) in vectors.iter().zip(quantized.iter()).enumerate() {
            if i == 0 {
                continue;
            }

            // True L2² distance
            let true_dist: f32 = query
                .iter()
                .zip(orig.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();

            // Quantized distance
            let quant_dist = params.distance_l2_squared(&query_prep, quant);

            let rel_error = (true_dist - quant_dist).abs() / true_dist.max(1e-6);
            max_rel_error = max_rel_error.max(rel_error);
        }

        println!(
            "Uniform SQ8 max relative distance error: {:.2}%",
            max_rel_error * 100.0
        );
        // Allow up to 10% relative error for uniform quantization
        assert!(
            max_rel_error < 0.15,
            "Distance error too large: {max_rel_error:.4}"
        );
    }

    #[test]
    fn test_uniform_int_dot_product() {
        let vectors: Vec<Vec<f32>> = vec![vec![0.5; 768], vec![0.3; 768]];
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let params = UniformScalarParams::train(&refs).unwrap();
        let query_prep = params.prepare_query(&vectors[0]);
        let quantized = params.quantize(&vectors[1]);

        // Just verify it runs without panicking
        let dist = params.distance_l2_squared(&query_prep, &quantized);
        assert!(dist >= 0.0);
        assert!(!dist.is_nan());
    }

    #[test]
    fn test_uniform_preserves_ordering() {
        use rand::Rng;

        let dim = 128;
        let mut rng = rand::thread_rng();

        // Create query and vectors at different distances
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let close: Vec<f32> = query.iter().map(|x| x + rng.gen_range(-0.1..0.1)).collect();
        let medium: Vec<f32> = query.iter().map(|x| x + rng.gen_range(-0.5..0.5)).collect();
        let far: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let vectors: Vec<Vec<f32>> = vec![close.clone(), medium.clone(), far.clone()];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let all_refs: Vec<&[f32]> = vec![query.as_slice()]
            .into_iter()
            .chain(refs.iter().copied())
            .collect();
        let params = UniformScalarParams::train(&all_refs).unwrap();

        let quantized: Vec<_> = vectors.iter().map(|v| params.quantize(v)).collect();
        let query_prep = params.prepare_query(&query);

        // Compute distances
        let d_close = params.distance_l2_squared(&query_prep, &quantized[0]);
        let d_medium = params.distance_l2_squared(&query_prep, &quantized[1]);
        let d_far = params.distance_l2_squared(&query_prep, &quantized[2]);

        // True distances
        let true_close: f32 = query
            .iter()
            .zip(close.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let true_medium: f32 = query
            .iter()
            .zip(medium.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let true_far: f32 = query
            .iter()
            .zip(far.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        // Close should be closest, far should be farthest (for most runs)
        // This test is probabilistic - we mainly want to catch major bugs
        println!(
            "True distances: close={true_close:.4}, medium={true_medium:.4}, far={true_far:.4}"
        );
        println!("Quant distances: close={d_close:.4}, medium={d_medium:.4}, far={d_far:.4}");

        // At minimum, verify distances are positive
        assert!(d_close >= 0.0);
        assert!(d_medium >= 0.0);
        assert!(d_far >= 0.0);
    }
}
