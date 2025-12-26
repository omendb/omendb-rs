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

    /// Compute approximate L2 distance between query (f32) and quantized vector (u8)
    ///
    /// Uses asymmetric distance: query stays f32, candidate is dequantized on-the-fly.
    #[must_use]
    #[allow(clippy::needless_return)] // returns needed for cfg-conditional control flow
    pub fn asymmetric_l2_squared(&self, query: &[f32], quantized: &[u8]) -> f32 {
        assert_eq!(query.len(), self.dimensions);
        assert_eq!(quantized.len(), self.dimensions);

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
            // Load 8 u8 values and convert to f32
            let q_bytes = std::slice::from_raw_parts(quantized.as_ptr().add(i), 8);
            let q0 = _mm256_set_ps(
                f32::from(q_bytes[7]),
                f32::from(q_bytes[6]),
                f32::from(q_bytes[5]),
                f32::from(q_bytes[4]),
                f32::from(q_bytes[3]),
                f32::from(q_bytes[2]),
                f32::from(q_bytes[1]),
                f32::from(q_bytes[0]),
            );

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

    #[cfg(target_arch = "aarch64")]
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

        // Handle remaining elements
        for j in i..self.dimensions {
            let dequant = f32::from(quantized[j]) * self.scales[j] + self.mins[j];
            let diff = query[j] - dequant;
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
}
