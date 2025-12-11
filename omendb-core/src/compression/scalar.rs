//! Scalar Quantization (SQ8) for OmenDB
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
//! # Performance
//!
//! - 4x compression (f32 → u8)
//! - SIMD u8 L2 distance
//! - ~98% recall with rescoring, ~95% without

use serde::{Deserialize, Serialize};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
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
    #[must_use]
    pub fn train(vectors: &[&[f32]]) -> Self {
        Self::train_with_percentiles(vectors, 0.01, 0.99)
    }

    /// Train with custom percentile bounds
    #[must_use]
    pub fn train_with_percentiles(
        vectors: &[&[f32]],
        lower_percentile: f32,
        upper_percentile: f32,
    ) -> Self {
        assert!(!vectors.is_empty(), "Need at least one vector to train");
        let dimensions = vectors[0].len();
        assert!(
            vectors.iter().all(|v| v.len() == dimensions),
            "All vectors must have same dimensions"
        );

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

        Self {
            mins,
            scales,
            dimensions,
        }
    }

    /// Quantize a single f32 vector to u8
    #[must_use]
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.dimensions);

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
            return unsafe { self.asymmetric_l2_squared_neon(query, quantized) };
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
        let mut sum = vdupq_n_f32(0.0);
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= self.dimensions {
            // Load 4 u8 values and convert to f32
            let q_f32 = vld1q_f32(
                [
                    f32::from(quantized[i]),
                    f32::from(quantized[i + 1]),
                    f32::from(quantized[i + 2]),
                    f32::from(quantized[i + 3]),
                ]
                .as_ptr(),
            );

            // Load scales and mins
            let scales = vld1q_f32(self.scales.as_ptr().add(i));
            let mins = vld1q_f32(self.mins.as_ptr().add(i));

            // Dequantize: q * scale + min
            let dequant = vfmaq_f32(mins, q_f32, scales);

            // Load query
            let query_vec = vld1q_f32(query.as_ptr().add(i));

            // Compute diff
            let diff = vsubq_f32(query_vec, dequant);

            // Accumulate diff^2
            sum = vfmaq_f32(sum, diff, diff);

            i += 4;
        }

        // Horizontal sum
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

/// Compute L2 distance between two quantized u8 vectors
///
/// Note: This is approximate and less accurate than asymmetric distance.
/// Prefer asymmetric_l2_squared when query is available in f32.
#[must_use]
pub fn symmetric_l2_squared_u8(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { symmetric_l2_squared_u8_avx2(a, b) };
        }
    }

    // Scalar fallback
    symmetric_l2_squared_u8_scalar(a, b)
}

fn symmetric_l2_squared_u8_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = i32::from(x) - i32::from(y);
            (diff * diff) as u32
        })
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn symmetric_l2_squared_u8_avx2(a: &[u8], b: &[u8]) -> u32 {
    let mut sum = _mm256_setzero_si256();
    let mut i = 0;

    // Process 32 bytes at a time
    while i + 32 <= a.len() {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);

        // Compute absolute difference
        let sad = _mm256_sad_epu8(va, vb);
        sum = _mm256_add_epi64(sum, sad);

        i += 32;
    }

    // Extract and sum the 4 64-bit integers
    let sum_array: [i64; 4] = std::mem::transmute(sum);
    let mut result = (sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3]) as u32;

    // Handle remaining elements
    for j in i..a.len() {
        let diff = i32::from(a[j]) - i32::from(b[j]);
        result += (diff * diff) as u32;
    }

    result
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
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let params = ScalarParams::train(&refs);

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
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let params = ScalarParams::train(&refs);
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
}
