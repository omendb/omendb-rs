//! Multi-bit Scalar Quantization for `OmenDB`
//!
//! Provides flexible vector compression with arbitrary bit rates (2-8 bits per
//! dimension) using per-dimension min/max quantization with trained parameters.
//!
//! **Note:** This module is named `rabitq` for historical reasons but implements
//! standard scalar quantization, NOT the RaBitQ algorithm from arXiv:2405.12497.
//! True RaBitQ requires random orthogonal rotation + binary quantization.
//!
//! # Tiered Compression Strategy
//!
//! - L0-L2 (hot): Full precision f32 (no compression)
//! - L3-L4 (warm): 4-bit (8× compression)
//! - L5-L6 (cold): 2-bit (16× compression)
//!
//! # Key Features
//!
//! - Flexible compression (2, 3, 4, 5, 7, 8 bits/dimension)
//! - Per-dimension min/max training (percentile-based for outlier robustness)
//! - SIMD-accelerated distance (AVX2/NEON)
//! - Same query speed as 8-bit scalar quantization
//! - Better accuracy than binary quantization

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::fmt;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vsubq_f32};
#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

/// Maximum number of codes per subspace (16 for 4-bit quantization)
const MAX_CODES: usize = 16;

/// Number of bits per dimension for quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationBits {
    /// 1 bit per dimension (32x compression) - Binary/BBQ
    Bits1,
    /// 2 bits per dimension (16x compression)
    Bits2,
    /// 3 bits per dimension (~10x compression)
    Bits3,
    /// 4 bits per dimension (8x compression)
    Bits4,
    /// 5 bits per dimension (~6x compression)
    Bits5,
    /// 7 bits per dimension (~4x compression)
    Bits7,
    /// 8 bits per dimension (4x compression)
    Bits8,
}

impl QuantizationBits {
    /// Convert to number of bits
    #[must_use]
    pub fn to_u8(self) -> u8 {
        match self {
            QuantizationBits::Bits1 => 1,
            QuantizationBits::Bits2 => 2,
            QuantizationBits::Bits3 => 3,
            QuantizationBits::Bits4 => 4,
            QuantizationBits::Bits5 => 5,
            QuantizationBits::Bits7 => 7,
            QuantizationBits::Bits8 => 8,
        }
    }

    /// Get number of quantization levels (2^bits)
    #[must_use]
    pub fn levels(self) -> usize {
        1 << self.to_u8()
    }

    /// Get compression ratio vs f32 (32 bits / `bits_per_dim`)
    #[must_use]
    pub fn compression_ratio(self) -> f32 {
        32.0 / self.to_u8() as f32
    }

    /// Get number of values that fit in one byte
    #[must_use]
    pub fn values_per_byte(self) -> usize {
        8 / self.to_u8() as usize
    }
}

impl fmt::Display for QuantizationBits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-bit", self.to_u8())
    }
}

/// Configuration for `RaBitQ` quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitQParams {
    /// Number of bits per dimension
    pub bits_per_dim: QuantizationBits,

    /// Number of rescaling factors to try (DEPRECATED: use trained quantizer)
    ///
    /// Higher values = better quantization quality but slower
    /// Typical range: 8-16
    pub num_rescale_factors: usize,

    /// Range of rescaling factors to try (DEPRECATED: use trained quantizer)
    ///
    /// Typical range: (0.5, 2.0) means try scales from 0.5x to 2.0x
    pub rescale_range: (f32, f32),
}

/// Trained quantization parameters computed from data
///
/// Stores per-dimension min/max values learned from a representative sample.
/// This enables consistent quantization across all vectors and correct ADC distances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainedParams {
    /// Minimum value per dimension
    pub mins: Vec<f32>,
    /// Maximum value per dimension
    pub maxs: Vec<f32>,
    /// Number of dimensions
    pub dimensions: usize,
}

impl TrainedParams {
    /// Train quantization parameters from sample vectors
    ///
    /// Computes per-dimension min/max using percentiles to exclude outliers.
    /// Uses 1st and 99th percentiles by default for robustness.
    ///
    /// # Arguments
    /// * `vectors` - Sample vectors to train from (should be representative)
    ///
    /// # Errors
    /// Returns error if vectors is empty or vectors have inconsistent dimensions.
    pub fn train(vectors: &[&[f32]]) -> Result<Self, &'static str> {
        Self::train_with_percentiles(vectors, 0.01, 0.99)
    }

    /// Train with custom percentile bounds
    ///
    /// # Arguments
    /// * `vectors` - Sample vectors to train from
    /// * `lower_percentile` - Lower bound percentile (e.g., 0.01 for 1st percentile)
    /// * `upper_percentile` - Upper bound percentile (e.g., 0.99 for 99th percentile)
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
        let mut maxs = Vec::with_capacity(dimensions);

        // For each dimension, collect values and compute percentiles
        let mut dim_values: Vec<f32> = Vec::with_capacity(n);
        for d in 0..dimensions {
            dim_values.clear();
            for v in vectors {
                dim_values.push(v[d]);
            }
            dim_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let min_val = dim_values[lower_idx];
            let max_val = dim_values[upper_idx];

            // Ensure non-zero range (add small epsilon if needed)
            let range = max_val - min_val;
            if range < 1e-7 {
                mins.push(min_val - 0.5);
                maxs.push(max_val + 0.5);
            } else {
                mins.push(min_val);
                maxs.push(max_val);
            }
        }

        Ok(Self {
            mins,
            maxs,
            dimensions,
        })
    }

    /// Quantize a single value using trained parameters for given dimension
    #[inline]
    #[must_use]
    pub fn quantize_value(&self, value: f32, dim: usize, levels: usize) -> u8 {
        let min = self.mins[dim];
        let max = self.maxs[dim];
        let range = max - min;

        // Map value to [0, 1] range, then to [0, levels-1]
        let normalized = (value - min) / range;
        let level = (normalized * (levels - 1) as f32).round();
        level.clamp(0.0, (levels - 1) as f32) as u8
    }

    /// Dequantize a code to reconstructed value for given dimension
    #[inline]
    #[must_use]
    pub fn dequantize_value(&self, code: u8, dim: usize, levels: usize) -> f32 {
        let min = self.mins[dim];
        let max = self.maxs[dim];
        let range = max - min;

        // Map code back to original range
        (code as f32 / (levels - 1) as f32) * range + min
    }
}

impl Default for RaBitQParams {
    fn default() -> Self {
        Self {
            bits_per_dim: QuantizationBits::Bits4, // 8x compression
            num_rescale_factors: 12,               // Good balance
            rescale_range: (0.5, 2.0),             // Paper recommendation
        }
    }
}

impl RaBitQParams {
    /// Create parameters for 2-bit quantization (16x compression)
    #[must_use]
    pub fn bits2() -> Self {
        Self {
            bits_per_dim: QuantizationBits::Bits2,
            ..Default::default()
        }
    }

    /// Create parameters for 4-bit quantization (8x compression, recommended)
    #[must_use]
    pub fn bits4() -> Self {
        Self {
            bits_per_dim: QuantizationBits::Bits4,
            ..Default::default()
        }
    }

    /// Create parameters for 8-bit quantization (4x compression, highest quality)
    #[must_use]
    pub fn bits8() -> Self {
        Self {
            bits_per_dim: QuantizationBits::Bits8,
            num_rescale_factors: 16,   // More factors for higher precision
            rescale_range: (0.7, 1.5), // Narrower range for 8-bit
        }
    }
}

/// A quantized vector with optimal rescaling
///
/// Storage format:
/// - data: Packed quantized values (multiple values per byte)
/// - scale: Optimal rescaling factor for this vector
/// - bits: Number of bits per dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedVector {
    /// Packed quantized values
    ///
    /// Format depends on `bits_per_dim`:
    /// - 2-bit: 4 values per byte
    /// - 3-bit: Not byte-aligned, needs special packing
    /// - 4-bit: 2 values per byte
    /// - 8-bit: 1 value per byte
    pub data: Vec<u8>,

    /// Optimal rescaling factor for this vector
    ///
    /// This is the scale factor that minimized quantization error
    /// during the rescaling search.
    pub scale: f32,

    /// Number of bits per dimension
    pub bits: u8,

    /// Original vector dimensions (for unpacking)
    pub dimensions: usize,
}

impl QuantizedVector {
    /// Create a new quantized vector
    #[must_use]
    pub fn new(data: Vec<u8>, scale: f32, bits: u8, dimensions: usize) -> Self {
        Self {
            data,
            scale,
            bits,
            dimensions,
        }
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.data.len()
    }

    /// Get compression ratio vs original f32 vector
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.dimensions * 4; // f32 = 4 bytes
        let compressed_bytes = self.data.len() + 4 + 1; // data + scale + bits
        original_bytes as f32 / compressed_bytes as f32
    }
}

/// `RaBitQ` quantizer
///
/// Implements scalar quantization with trained per-dimension ranges.
/// Training computes min/max per dimension from sample data, enabling
/// consistent quantization across all vectors and correct ADC distances.
///
/// # Usage
///
/// ```ignore
/// // Create and train quantizer
/// let mut quantizer = RaBitQ::new(RaBitQParams::bits4());
/// quantizer.train(&sample_vectors);
///
/// // Quantize vectors
/// let quantized = quantizer.quantize(&vector);
///
/// // ADC search (distances are mathematically correct)
/// let adc = quantizer.build_adc_table(&query);
/// let dist = adc.distance(&quantized.data);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitQ {
    params: RaBitQParams,
    /// Trained parameters (per-dimension min/max)
    /// When None, falls back to legacy per-vector scaling (deprecated)
    trained: Option<TrainedParams>,
}

/// Asymmetric Distance Computation (ADC) lookup table for fast quantized search
///
/// Precomputes partial squared distances from a query vector to all possible
/// quantized codes. This enables O(1) distance computation per dimension instead
/// of O(dim) decompression + distance calculation.
///
/// # Performance
///
/// - **Memory**: For 4-bit: dim * 16 * 4 bytes (e.g., 1536D = 96KB per query)
/// - **Speedup**: 5-10x faster distance computation vs full decompression
/// - **Use case**: Scanning many candidate vectors during HNSW search
///
/// # Algorithm
///
/// Instead of:
/// ```ignore
/// for each candidate:
///     decompress(candidate) -> O(dim)
///     distance(query, decompressed) -> O(dim)
/// ```
///
/// With ADC:
/// ```ignore
/// precompute table[code] = (query_value - dequantize(code))^2 for all codes
/// for each candidate:
///     sum(table[candidate[i]]) -> O(1) per dimension
/// ```
#[derive(Debug, Clone)]
pub struct ADCTable {
    /// Lookup table: `table[dim_idx][code] = partial squared distance`
    /// For 4-bit: each inner array has 16 entries (codes 0-15)
    /// For 2-bit: each inner array has 4 entries (codes 0-3)
    table: Vec<SmallVec<[f32; MAX_CODES]>>,

    /// Quantization parameters (bits per dimension)
    bits: u8,

    /// Number of dimensions
    dimensions: usize,
}

impl ADCTable {
    /// Build ADC lookup table using trained quantization parameters
    ///
    /// This is the production method that computes correct distances.
    /// Uses per-dimension min/max ranges from training.
    ///
    /// # Arguments
    ///
    /// * `query` - The uncompressed query vector
    /// * `trained` - Trained parameters with per-dimension min/max
    /// * `params` - Quantization parameters (bits per dimension)
    #[must_use]
    pub fn new_trained(query: &[f32], trained: &TrainedParams, params: &RaBitQParams) -> Self {
        let bits = params.bits_per_dim.to_u8();
        let num_codes = params.bits_per_dim.levels();
        let dimensions = query.len();

        let mut table = Vec::with_capacity(dimensions);

        // For each dimension, compute distances to all possible codes
        for (d, &q_value) in query.iter().enumerate() {
            let mut dim_table = SmallVec::new();

            for code in 0..num_codes {
                // Dequantize using trained min/max for this dimension
                let reconstructed = trained.dequantize_value(code as u8, d, num_codes);

                // Compute squared difference (partial L2 distance)
                let diff = q_value - reconstructed;
                dim_table.push(diff * diff);
            }

            table.push(dim_table);
        }

        Self {
            table,
            bits,
            dimensions,
        }
    }

    /// Build ADC lookup table for a query vector (DEPRECATED)
    ///
    /// For each dimension and each possible quantized code, precomputes the
    /// squared distance contribution: (query[i] - dequantize(code, scale))^2
    ///
    /// WARNING: This method uses a fixed scale which produces incorrect distances
    /// when vectors were quantized with different scales. Use `new_trained()` instead.
    ///
    /// # Arguments
    ///
    /// * `query` - The uncompressed query vector
    /// * `scale` - The scale factor used for quantization (from training or default)
    /// * `params` - Quantization parameters (bits per dimension)
    ///
    /// # Returns
    ///
    /// An `ADCTable` that can compute distances via `distance()` method
    ///
    /// # Note
    /// Prefer `ADCTable::new_trained()` with `TrainedParams` for correct ADC distances.
    /// This method uses per-vector scale which gives lower accuracy.
    #[must_use]
    pub fn new(query: &[f32], scale: f32, params: &RaBitQParams) -> Self {
        let bits = params.bits_per_dim.to_u8();
        let num_codes = params.bits_per_dim.levels();
        let dimensions = query.len();

        let mut table = Vec::with_capacity(dimensions);

        // Dequantization factor: value = code / (levels - 1) / scale
        let levels = num_codes as f32;
        let dequant_factor = 1.0 / ((levels - 1.0) * scale);

        // For each dimension, compute distances to all possible codes
        for &q_value in query {
            let mut dim_table = SmallVec::new();

            for code in 0..num_codes {
                // Dequantize the code to get the reconstructed value
                let reconstructed = (code as f32) * dequant_factor;

                // Compute squared difference (partial L2 distance)
                let diff = q_value - reconstructed;
                dim_table.push(diff * diff);
            }

            table.push(dim_table);
        }

        Self {
            table,
            bits,
            dimensions,
        }
    }

    /// Compute approximate L2 squared distance using lookup table
    ///
    /// This is the hot path for search! Instead of decompressing and computing
    /// distance, we just sum up precomputed values from the table.
    ///
    /// # Performance
    ///
    /// - 4-bit: ~5-10x faster than decompression + distance
    /// - Cache-friendly: sequential access patterns
    /// - SIMD-friendly: can vectorize the summation
    ///
    /// # Arguments
    ///
    /// * `data` - Packed quantized bytes
    ///
    /// # Returns
    ///
    /// Approximate squared L2 distance (not square-rooted for efficiency)
    #[inline]
    #[must_use]
    pub fn distance_squared(&self, data: &[u8]) -> f32 {
        match self.bits {
            4 => self.distance_squared_4bit(data),
            2 => self.distance_squared_2bit(data),
            8 => self.distance_squared_8bit(data),
            _ => self.distance_squared_generic(data),
        }
    }

    /// Compute distance and return square root (actual L2 distance)
    ///
    /// Uses SIMD-accelerated distance computation (AVX2 on `x86_64`, NEON on aarch64).
    #[inline]
    #[must_use]
    pub fn distance(&self, data: &[u8]) -> f32 {
        self.distance_squared_simd(data).sqrt()
    }

    /// Fast path for 4-bit quantization (most common case)
    ///
    /// # Safety invariants (maintained by `ADCTable::new`)
    /// - `self.table.len() == self.dimensions`
    /// - Each `table[i]` has exactly 16 entries (4-bit = 2^4 codes)
    /// - Input `data` has `ceil(dimensions/2)` bytes (2 values per byte)
    #[inline]
    fn distance_squared_4bit(&self, data: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        let num_pairs = self.dimensions / 2;

        // Process pairs of dimensions (2 codes per byte)
        for i in 0..num_pairs {
            if i >= data.len() {
                break;
            }

            // SAFETY: i < num_pairs <= data.len() (checked above)
            let byte = unsafe { *data.get_unchecked(i) };
            let code_hi = (byte >> 4) as usize; // 0..=15
            let code_lo = (byte & 0x0F) as usize; // 0..=15

            // SAFETY:
            // - i*2 < dimensions (since i < num_pairs = dimensions/2)
            // - i*2+1 < dimensions (same reasoning)
            // - code_hi, code_lo in 0..16 (4-bit mask guarantees this)
            // - table has 16 entries per dimension (4-bit quantization)
            sum += unsafe {
                *self.table.get_unchecked(i * 2).get_unchecked(code_hi)
                    + *self.table.get_unchecked(i * 2 + 1).get_unchecked(code_lo)
            };
        }

        // Handle odd dimension
        if self.dimensions % 2 == 1 && num_pairs < data.len() {
            // SAFETY: num_pairs < data.len() checked above
            let byte = unsafe { *data.get_unchecked(num_pairs) };
            let code_hi = (byte >> 4) as usize; // 0..=15
                                                // SAFETY:
                                                // - dimensions-1 is valid index (dimensions >= 1 when odd)
                                                // - code_hi in 0..16 (4-bit mask)
            sum += unsafe {
                *self
                    .table
                    .get_unchecked(self.dimensions - 1)
                    .get_unchecked(code_hi)
            };
        }

        sum
    }

    /// Fast path for 2-bit quantization
    ///
    /// # Safety invariants (maintained by `ADCTable::new`)
    /// - `self.table.len() == self.dimensions`
    /// - Each `table[i]` has exactly 4 entries (2-bit = 2^2 codes)
    /// - Input `data` has `ceil(dimensions/4)` bytes (4 values per byte)
    #[inline]
    fn distance_squared_2bit(&self, data: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        let num_quads = self.dimensions / 4;

        // Process quads of dimensions (4 codes per byte)
        for i in 0..num_quads {
            if i >= data.len() {
                break;
            }

            // SAFETY: i < num_quads <= data.len() (checked above)
            let byte = unsafe { *data.get_unchecked(i) };

            // SAFETY:
            // - i*4+k < dimensions for k in 0..4 (since i < num_quads = dimensions/4)
            // - all codes in 0..4 (2-bit mask guarantees this)
            // - table has 4 entries per dimension (2-bit quantization)
            sum += unsafe {
                *self
                    .table
                    .get_unchecked(i * 4)
                    .get_unchecked((byte & 0b11) as usize)
                    + *self
                        .table
                        .get_unchecked(i * 4 + 1)
                        .get_unchecked(((byte >> 2) & 0b11) as usize)
                    + *self
                        .table
                        .get_unchecked(i * 4 + 2)
                        .get_unchecked(((byte >> 4) & 0b11) as usize)
                    + *self
                        .table
                        .get_unchecked(i * 4 + 3)
                        .get_unchecked(((byte >> 6) & 0b11) as usize)
            };
        }

        // Handle remainder
        let remaining = self.dimensions % 4;
        if remaining > 0 && num_quads < data.len() {
            // SAFETY: num_quads < data.len() checked above
            let byte = unsafe { *data.get_unchecked(num_quads) };
            for j in 0..remaining {
                let code = ((byte >> (j * 2)) & 0b11) as usize; // 0..=3
                                                                // SAFETY:
                                                                // - num_quads*4+j < dimensions (j < remaining = dimensions%4)
                                                                // - code in 0..4 (2-bit mask)
                sum += unsafe {
                    *self
                        .table
                        .get_unchecked(num_quads * 4 + j)
                        .get_unchecked(code)
                };
            }
        }

        sum
    }

    /// Fast path for 8-bit quantization
    ///
    /// # Safety invariants (maintained by `ADCTable::new`)
    /// - `self.table.len() == self.dimensions`
    /// - Each `table[i]` has exactly 256 entries (8-bit = 2^8 codes)
    /// - Input `data` has `dimensions` bytes (1 value per byte)
    #[inline]
    fn distance_squared_8bit(&self, data: &[u8]) -> f32 {
        let mut sum = 0.0f32;

        for (i, &byte) in data.iter().enumerate().take(self.dimensions) {
            // SAFETY:
            // - i < dimensions (take() ensures this)
            // - byte as usize in 0..256 (u8 range)
            // - table has 256 entries per dimension (8-bit quantization)
            sum += unsafe { *self.table.get_unchecked(i).get_unchecked(byte as usize) };
        }

        sum
    }

    /// Generic fallback for other bit widths
    #[inline]
    fn distance_squared_generic(&self, data: &[u8]) -> f32 {
        // For non-standard bit widths, fall back to bounds-checked access
        let mut sum = 0.0f32;

        for (i, dim_table) in self.table.iter().enumerate() {
            if i >= data.len() {
                break;
            }
            let code = data[i] as usize;
            if let Some(&dist) = dim_table.get(code) {
                sum += dist;
            }
        }

        sum
    }

    /// SIMD-accelerated distance computation for 4-bit quantization
    ///
    /// Uses AVX2 on `x86_64` or NEON on `aarch64` to process multiple lookups in parallel.
    /// Falls back to scalar implementation if SIMD not available.
    #[inline]
    #[must_use]
    pub fn distance_squared_simd(&self, data: &[u8]) -> f32 {
        match self.bits {
            4 => {
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx2") {
                        unsafe { self.distance_squared_4bit_avx2(data) }
                    } else {
                        // x86_64 fallback to scalar
                        self.distance_squared_4bit(data)
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    // NEON is always available on aarch64
                    unsafe { self.distance_squared_4bit_neon(data) }
                }
                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                {
                    // Other architectures fallback to scalar
                    self.distance_squared_4bit(data)
                }
            }
            2 => {
                // For 2-bit, scalar is already quite fast
                self.distance_squared_2bit(data)
            }
            8 => {
                // For 8-bit, could use SIMD gather but scalar is reasonable
                self.distance_squared_8bit(data)
            }
            _ => self.distance_squared_generic(data),
        }
    }

    /// AVX2 implementation for 4-bit ADC distance
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn distance_squared_4bit_avx2(&self, data: &[u8]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let num_pairs = self.dimensions / 2;

        // Process 8 pairs (16 dimensions) at a time using AVX2
        let chunks = num_pairs / 8;
        for chunk_idx in 0..chunks {
            let byte_idx = chunk_idx * 8;
            if byte_idx + 8 > data.len() {
                break;
            }

            // Load 8 bytes (16 4-bit codes)
            let mut values = [0.0f32; 8];
            for (i, value) in values.iter_mut().enumerate() {
                let byte = *data.get_unchecked(byte_idx + i);
                let code_hi = (byte >> 4) as usize;
                let code_lo = (byte & 0x0F) as usize;

                let dist_hi = *self
                    .table
                    .get_unchecked((byte_idx + i) * 2)
                    .get_unchecked(code_hi);
                let dist_lo = *self
                    .table
                    .get_unchecked((byte_idx + i) * 2 + 1)
                    .get_unchecked(code_lo);
                *value = dist_hi + dist_lo;
            }

            let vec = _mm256_loadu_ps(values.as_ptr());
            sum = _mm256_add_ps(sum, vec);
        }

        // Horizontal sum of AVX2 register
        let mut result = horizontal_sum_avx2(sum);

        // Handle remainder with scalar
        for i in (chunks * 8)..num_pairs {
            if i >= data.len() {
                break;
            }
            let byte = *data.get_unchecked(i);
            let code_hi = (byte >> 4) as usize;
            let code_lo = (byte & 0x0F) as usize;

            result += *self.table.get_unchecked(i * 2).get_unchecked(code_hi)
                + *self.table.get_unchecked(i * 2 + 1).get_unchecked(code_lo);
        }

        // Handle odd dimension
        if self.dimensions % 2 == 1 && num_pairs < data.len() {
            let byte = *data.get_unchecked(num_pairs);
            let code_hi = (byte >> 4) as usize;
            result += *self
                .table
                .get_unchecked(self.dimensions - 1)
                .get_unchecked(code_hi);
        }

        result
    }

    /// NEON implementation for 4-bit ADC distance
    #[cfg(target_arch = "aarch64")]
    unsafe fn distance_squared_4bit_neon(&self, data: &[u8]) -> f32 {
        let mut sum = vdupq_n_f32(0.0);
        let num_pairs = self.dimensions / 2;

        // Process 4 pairs (8 dimensions) at a time using NEON
        let chunks = num_pairs / 4;
        for chunk_idx in 0..chunks {
            let byte_idx = chunk_idx * 4;
            if byte_idx + 4 > data.len() {
                break;
            }

            let mut values = [0.0f32; 4];
            for (i, value) in values.iter_mut().enumerate() {
                let byte = *data.get_unchecked(byte_idx + i);
                let code_hi = (byte >> 4) as usize;
                let code_lo = (byte & 0x0F) as usize;

                let dist_hi = *self
                    .table
                    .get_unchecked((byte_idx + i) * 2)
                    .get_unchecked(code_hi);
                let dist_lo = *self
                    .table
                    .get_unchecked((byte_idx + i) * 2 + 1)
                    .get_unchecked(code_lo);
                *value = dist_hi + dist_lo;
            }

            let vec = vld1q_f32(values.as_ptr());
            sum = vaddq_f32(sum, vec);
        }

        let mut result = vaddvq_f32(sum);

        // Handle remainder with scalar
        for i in (chunks * 4)..num_pairs {
            if i >= data.len() {
                break;
            }
            let byte = *data.get_unchecked(i);
            let code_hi = (byte >> 4) as usize;
            let code_lo = (byte & 0x0F) as usize;

            result += *self.table.get_unchecked(i * 2).get_unchecked(code_hi)
                + *self.table.get_unchecked(i * 2 + 1).get_unchecked(code_lo);
        }

        // Handle odd dimension
        if self.dimensions % 2 == 1 && num_pairs < data.len() {
            let byte = *data.get_unchecked(num_pairs);
            let code_hi = (byte >> 4) as usize;
            result += *self
                .table
                .get_unchecked(self.dimensions - 1)
                .get_unchecked(code_hi);
        }

        result
    }

    /// Get bits per dimension
    #[must_use]
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Get number of dimensions
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get partial distance for a dimension and code
    ///
    /// Returns 0.0 if indices are out of bounds.
    #[must_use]
    pub fn get(&self, dim: usize, code: usize) -> f32 {
        self.table
            .get(dim)
            .and_then(|t| t.get(code))
            .copied()
            .unwrap_or(0.0)
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.table.len() * std::mem::size_of::<SmallVec<[f32; MAX_CODES]>>()
            + self.table.iter().map(|t| t.len() * 4).sum::<usize>()
    }
}

impl RaBitQ {
    /// Create a new `RaBitQ` quantizer (untrained)
    ///
    /// Call `train()` before use to enable correct ADC distances.
    #[must_use]
    pub fn new(params: RaBitQParams) -> Self {
        Self {
            params,
            trained: None,
        }
    }

    /// Create a trained `RaBitQ` quantizer
    #[must_use]
    pub fn new_trained(params: RaBitQParams, trained: TrainedParams) -> Self {
        Self {
            params,
            trained: Some(trained),
        }
    }

    /// Create with default 4-bit quantization
    #[must_use]
    pub fn default_4bit() -> Self {
        Self::new(RaBitQParams::bits4())
    }

    /// Get quantization parameters
    #[must_use]
    pub fn params(&self) -> &RaBitQParams {
        &self.params
    }

    /// Check if quantizer has been trained
    #[must_use]
    pub fn is_trained(&self) -> bool {
        self.trained.is_some()
    }

    /// Get trained parameters (if any)
    #[must_use]
    pub fn trained_params(&self) -> Option<&TrainedParams> {
        self.trained.as_ref()
    }

    /// Train quantizer on sample vectors
    ///
    /// Computes per-dimension min/max ranges from the sample.
    /// Must be called before quantization for correct ADC distances.
    ///
    /// # Arguments
    /// * `vectors` - Representative sample of vectors to train from
    ///
    /// # Errors
    /// Returns error if vectors is empty or have inconsistent dimensions.
    pub fn train(&mut self, vectors: &[&[f32]]) -> Result<(), &'static str> {
        self.trained = Some(TrainedParams::train(vectors)?);
        Ok(())
    }

    /// Train with owned vectors (convenience method)
    ///
    /// # Errors
    /// Returns error if vectors is empty or have inconsistent dimensions.
    pub fn train_owned(&mut self, vectors: &[Vec<f32>]) -> Result<(), &'static str> {
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        self.train(&refs)
    }

    /// Quantize a vector using trained parameters
    ///
    /// If trained: uses per-dimension min/max for consistent quantization
    /// If untrained: falls back to legacy per-vector scaling (deprecated)
    #[must_use]
    pub fn quantize(&self, vector: &[f32]) -> QuantizedVector {
        // Use trained quantization if available
        if let Some(ref trained) = self.trained {
            return self.quantize_trained(vector, trained);
        }

        // Legacy fallback: per-vector scale search (deprecated)
        let mut best_error = f32::MAX;
        let mut best_quantized = Vec::new();
        let mut best_scale = 1.0;

        // Generate rescaling factors to try
        let scales = self.generate_scales();

        // Try each scale and find the one with minimum error
        for scale in scales {
            let quantized = self.quantize_with_scale(vector, scale);
            let error = self.compute_error(vector, &quantized, scale);

            if error < best_error {
                best_error = error;
                best_quantized = quantized;
                best_scale = scale;
            }
        }

        QuantizedVector::new(
            best_quantized,
            best_scale,
            self.params.bits_per_dim.to_u8(),
            vector.len(),
        )
    }

    /// Quantize using trained per-dimension min/max ranges
    ///
    /// This is the production path that enables correct ADC distances.
    fn quantize_trained(&self, vector: &[f32], trained: &TrainedParams) -> QuantizedVector {
        let bits = self.params.bits_per_dim.to_u8();
        let levels = self.params.bits_per_dim.levels();

        // Quantize each dimension using trained min/max
        let quantized: Vec<u8> = vector
            .iter()
            .enumerate()
            .map(|(d, &v)| trained.quantize_value(v, d, levels))
            .collect();

        // Pack into bytes
        let packed = self.pack_quantized(&quantized, bits);

        // Scale=1.0 for trained quantization (min/max handles the range)
        QuantizedVector::new(packed, 1.0, bits, vector.len())
    }

    /// Generate rescaling factors to try
    ///
    /// Returns a vector of scale factors evenly spaced between
    /// `rescale_range.0` and `rescale_range.1`
    fn generate_scales(&self) -> Vec<f32> {
        let (min_scale, max_scale) = self.params.rescale_range;
        let n = self.params.num_rescale_factors;

        if n == 1 {
            return vec![f32::midpoint(min_scale, max_scale)];
        }

        let step = (max_scale - min_scale) / (n - 1) as f32;
        (0..n).map(|i| min_scale + i as f32 * step).collect()
    }

    /// Quantize a vector with a specific scale factor
    ///
    /// Algorithm (Extended RaBitQ):
    /// 1. Scale: v' = v * scale
    /// 2. Quantize to grid: q = round(v' * (2^bits - 1))
    /// 3. Clamp to valid range
    /// 4. Pack into bytes
    fn quantize_with_scale(&self, vector: &[f32], scale: f32) -> Vec<u8> {
        let bits = self.params.bits_per_dim.to_u8();
        let levels = self.params.bits_per_dim.levels() as f32;
        let max_level = (levels - 1.0) as u8;

        // Scale and quantize directly (no normalization needed)
        let quantized: Vec<u8> = vector
            .iter()
            .map(|&v| {
                // Scale the value
                let scaled = v * scale;
                // Quantize to grid [0, levels-1]
                let level = (scaled * (levels - 1.0)).round();
                // Clamp to valid range
                level.clamp(0.0, max_level as f32) as u8
            })
            .collect();

        // Pack into bytes
        self.pack_quantized(&quantized, bits)
    }

    /// Pack quantized values into bytes
    ///
    /// Packing depends on bits per dimension:
    /// - 2-bit: 4 values per byte (00 00 00 00)
    /// - 4-bit: 2 values per byte (0000 0000)
    /// - 8-bit: 1 value per byte
    #[allow(clippy::unused_self)]
    fn pack_quantized(&self, values: &[u8], bits: u8) -> Vec<u8> {
        match bits {
            2 => {
                // 4 values per byte
                let mut packed = Vec::with_capacity(values.len().div_ceil(4));
                for chunk in values.chunks(4) {
                    let mut byte = 0u8;
                    for (i, &val) in chunk.iter().enumerate() {
                        byte |= (val & 0b11) << (i * 2);
                    }
                    packed.push(byte);
                }
                packed
            }
            4 => {
                // 2 values per byte
                let mut packed = Vec::with_capacity(values.len().div_ceil(2));
                for chunk in values.chunks(2) {
                    let byte = if chunk.len() == 2 {
                        (chunk[0] << 4) | (chunk[1] & 0x0F)
                    } else {
                        chunk[0] << 4
                    };
                    packed.push(byte);
                }
                packed
            }
            8 => {
                // 1 value per byte (no packing needed)
                values.to_vec()
            }
            _ => {
                // 3, 5, 7-bit: fall back to 8-bit storage
                // Not implementing proper bit-packing because:
                // - Public API only exposes 2, 4, 8-bit (see python/src/lib.rs)
                // - Cross-byte packing is complex with marginal compression benefit
                // - 4-bit (8x) vs 5-bit (~6x) isn't worth the code complexity
                values.to_vec()
            }
        }
    }

    /// Unpack quantized bytes into individual values
    #[must_use]
    pub fn unpack_quantized(&self, packed: &[u8], bits: u8, dimensions: usize) -> Vec<u8> {
        match bits {
            2 => {
                // 4 values per byte
                let mut values = Vec::with_capacity(dimensions);
                for &byte in packed {
                    for i in 0..4 {
                        if values.len() < dimensions {
                            values.push((byte >> (i * 2)) & 0b11);
                        }
                    }
                }
                values
            }
            4 => {
                // 2 values per byte
                let mut values = Vec::with_capacity(dimensions);
                for &byte in packed {
                    values.push(byte >> 4);
                    if values.len() < dimensions {
                        values.push(byte & 0x0F);
                    }
                }
                values.truncate(dimensions);
                values
            }
            8 => {
                // 1 value per byte
                packed[..dimensions.min(packed.len())].to_vec()
            }
            _ => {
                // For other bit widths, assume 8-bit storage
                packed[..dimensions.min(packed.len())].to_vec()
            }
        }
    }

    /// Compute quantization error (reconstruction error)
    ///
    /// Error = ||original - reconstructed||²
    fn compute_error(&self, original: &[f32], quantized: &[u8], scale: f32) -> f32 {
        let reconstructed = self.reconstruct(quantized, scale, original.len());

        original
            .iter()
            .zip(reconstructed.iter())
            .map(|(o, r)| (o - r).powi(2))
            .sum()
    }

    /// Reconstruct (dequantize) a quantized vector
    ///
    /// Algorithm (Extended RaBitQ):
    /// 1. Unpack bytes to quantized values [0, 2^bits-1]
    /// 2. Denormalize: v' = q / (2^bits - 1)
    /// 3. Unscale: v = v' / scale
    #[must_use]
    pub fn reconstruct(&self, quantized: &[u8], scale: f32, dimensions: usize) -> Vec<f32> {
        let bits = self.params.bits_per_dim.to_u8();
        let levels = self.params.bits_per_dim.levels() as f32;

        // Unpack bytes
        let values = self.unpack_quantized(quantized, bits, dimensions);

        // Dequantize: reverse the quantization process
        values
            .iter()
            .map(|&q| {
                // Denormalize from [0, levels-1] to [0, 1]
                let denorm = q as f32 / (levels - 1.0);
                // Unscale
                denorm / scale
            })
            .collect()
    }

    /// Compute L2 (Euclidean) distance between two quantized vectors
    ///
    /// This reconstructs both vectors and computes standard L2 distance.
    /// For maximum accuracy, use this with original vectors for reranking.
    #[must_use]
    pub fn distance_l2(&self, qv1: &QuantizedVector, qv2: &QuantizedVector) -> f32 {
        let v1 = self.reconstruct(&qv1.data, qv1.scale, qv1.dimensions);
        let v2 = self.reconstruct(&qv2.data, qv2.scale, qv2.dimensions);

        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Compute cosine distance between two quantized vectors
    ///
    /// Cosine distance = 1 - cosine similarity
    #[must_use]
    pub fn distance_cosine(&self, qv1: &QuantizedVector, qv2: &QuantizedVector) -> f32 {
        let v1 = self.reconstruct(&qv1.data, qv1.scale, qv1.dimensions);
        let v2 = self.reconstruct(&qv2.data, qv2.scale, qv2.dimensions);

        let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
        let norm2: f32 = v2.iter().map(|b| b * b).sum::<f32>().sqrt();

        if norm1 < 1e-10 || norm2 < 1e-10 {
            return 1.0; // Maximum distance for zero vectors
        }

        let cosine_sim = dot / (norm1 * norm2);
        1.0 - cosine_sim
    }

    /// Compute dot product between two quantized vectors
    #[must_use]
    pub fn distance_dot(&self, qv1: &QuantizedVector, qv2: &QuantizedVector) -> f32 {
        let v1 = self.reconstruct(&qv1.data, qv1.scale, qv1.dimensions);
        let v2 = self.reconstruct(&qv2.data, qv2.scale, qv2.dimensions);

        // Return negative dot product (for nearest neighbor search)
        -v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum::<f32>()
    }

    /// Compute approximate distance using quantized values directly (fast path)
    ///
    /// This computes distance in the quantized space without full reconstruction.
    /// Faster but less accurate than `distance_l2`.
    #[must_use]
    pub fn distance_approximate(&self, qv1: &QuantizedVector, qv2: &QuantizedVector) -> f32 {
        // Unpack to quantized values (u8)
        let v1 = self.unpack_quantized(&qv1.data, qv1.bits, qv1.dimensions);
        let v2 = self.unpack_quantized(&qv2.data, qv2.bits, qv2.dimensions);

        // Compute L2 distance in quantized space
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| {
                let diff = (*a as i16 - *b as i16) as f32;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }

    /// Compute asymmetric L2 distance (query vs quantized) without full reconstruction
    ///
    /// This is the hot path for search! It unpacks quantized values on the fly and
    /// computes distance against the uncompressed query vector.
    ///
    /// When trained: Uses per-dimension min/max for correct distance computation.
    /// When untrained: Falls back to per-vector scale (deprecated, lower accuracy).
    #[must_use]
    pub fn distance_asymmetric_l2(&self, query: &[f32], quantized: &QuantizedVector) -> f32 {
        // Use trained parameters when available for correct distance computation
        if let Some(trained) = &self.trained {
            return self.distance_asymmetric_l2_trained(query, &quantized.data, trained);
        }
        // Fallback to per-vector scale (deprecated, only for untrained quantizers)
        self.distance_asymmetric_l2_raw(query, &quantized.data, quantized.scale, quantized.bits)
    }

    /// Asymmetric L2 distance from flat storage (no QuantizedVector wrapper)
    ///
    /// This is the preferred method for flat contiguous storage layouts.
    /// When trained: Uses per-dimension min/max (scale ignored).
    /// When untrained: Falls back to per-vector scale.
    #[must_use]
    #[inline]
    pub fn distance_asymmetric_l2_flat(&self, query: &[f32], data: &[u8], scale: f32) -> f32 {
        if let Some(trained) = &self.trained {
            return self.distance_asymmetric_l2_trained(query, data, trained);
        }
        // Fallback to per-vector scale (deprecated, only for untrained quantizers)
        self.distance_asymmetric_l2_raw(query, data, scale, self.params.bits_per_dim.to_u8())
    }

    /// Asymmetric L2 distance using trained per-dimension parameters
    ///
    /// This is the correct distance computation for trained quantizers.
    /// Each dimension uses its own min/max range for accurate dequantization.
    #[must_use]
    fn distance_asymmetric_l2_trained(
        &self,
        query: &[f32],
        data: &[u8],
        trained: &TrainedParams,
    ) -> f32 {
        let levels = self.params.bits_per_dim.levels() as f32;
        let bits = self.params.bits_per_dim.to_u8();

        // Use SmallVec for stack allocation when possible
        let mut buffer: SmallVec<[f32; 256]> = SmallVec::with_capacity(query.len());

        match bits {
            4 => {
                let num_pairs = query.len() / 2;
                if data.len() < query.len().div_ceil(2) {
                    return f32::MAX;
                }

                for i in 0..num_pairs {
                    let byte = unsafe { *data.get_unchecked(i) };
                    let d0 = i * 2;
                    let d1 = i * 2 + 1;

                    // Dequantize using per-dimension min/max
                    let code0 = (byte >> 4) as f32;
                    let code1 = (byte & 0x0F) as f32;

                    let range0 = trained.maxs[d0] - trained.mins[d0];
                    let range1 = trained.maxs[d1] - trained.mins[d1];

                    buffer.push((code0 / (levels - 1.0)) * range0 + trained.mins[d0]);
                    buffer.push((code1 / (levels - 1.0)) * range1 + trained.mins[d1]);
                }

                if !query.len().is_multiple_of(2) {
                    let byte = unsafe { *data.get_unchecked(num_pairs) };
                    let d = num_pairs * 2;
                    let code = (byte >> 4) as f32;
                    let range = trained.maxs[d] - trained.mins[d];
                    buffer.push((code / (levels - 1.0)) * range + trained.mins[d]);
                }
            }
            2 => {
                let num_quads = query.len() / 4;
                if data.len() < query.len().div_ceil(4) {
                    return f32::MAX;
                }

                for i in 0..num_quads {
                    let byte = unsafe { *data.get_unchecked(i) };
                    for j in 0..4 {
                        let d = i * 4 + j;
                        let code = ((byte >> (j * 2)) & 0b11) as f32;
                        let range = trained.maxs[d] - trained.mins[d];
                        buffer.push((code / (levels - 1.0)) * range + trained.mins[d]);
                    }
                }

                let remaining = query.len() % 4;
                if remaining > 0 {
                    let byte = unsafe { *data.get_unchecked(num_quads) };
                    for j in 0..remaining {
                        let d = num_quads * 4 + j;
                        let code = ((byte >> (j * 2)) & 0b11) as f32;
                        let range = trained.maxs[d] - trained.mins[d];
                        buffer.push((code / (levels - 1.0)) * range + trained.mins[d]);
                    }
                }
            }
            8 => {
                if data.len() < query.len() {
                    return f32::MAX;
                }
                for (d, &byte) in data.iter().enumerate().take(query.len()) {
                    let code = byte as f32;
                    let range = trained.maxs[d] - trained.mins[d];
                    buffer.push((code / (levels - 1.0)) * range + trained.mins[d]);
                }
            }
            _ => {
                // Generic fallback for other bit widths
                let unpacked = self.unpack_quantized(data, bits, query.len());
                for (d, &code) in unpacked.iter().enumerate().take(query.len()) {
                    let range = trained.maxs[d] - trained.mins[d];
                    buffer.push((code as f32 / (levels - 1.0)) * range + trained.mins[d]);
                }
            }
        }

        simd_l2_distance(query, &buffer)
    }

    /// Build an ADC (Asymmetric Distance Computation) lookup table for a query
    ///
    /// If trained: uses per-dimension min/max for correct distances
    /// If untrained: uses provided scale (deprecated, incorrect distances)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Preferred: train first, then build ADC table
    /// quantizer.train(&sample_vectors);
    /// let adc_table = quantizer.build_adc_table(&query);
    /// for candidate in candidates {
    ///     let dist = adc_table.distance(&candidate.data);
    /// }
    /// ```
    #[must_use]
    pub fn build_adc_table(&self, query: &[f32]) -> Option<ADCTable> {
        self.trained
            .as_ref()
            .map(|trained| ADCTable::new_trained(query, trained, &self.params))
    }

    /// Build ADC table with explicit scale
    ///
    /// # Note
    /// Prefer `build_adc_table()` on a trained quantizer for correct ADC distances.
    /// This method uses per-vector scale which gives lower accuracy.
    #[must_use]
    pub fn build_adc_table_with_scale(&self, query: &[f32], scale: f32) -> ADCTable {
        ADCTable::new(query, scale, &self.params)
    }

    /// Compute distance using ADC table (convenience wrapper)
    ///
    /// Returns None if quantizer is not trained.
    #[must_use]
    pub fn distance_with_adc(&self, query: &[f32], quantized: &QuantizedVector) -> Option<f32> {
        let adc = self.build_adc_table(query)?;
        Some(adc.distance(&quantized.data))
    }

    /// Low-level asymmetric distance computation on raw bytes
    ///
    /// Enables zero-copy access from mmap (no `QuantizedVector` allocation needed).
    /// Uses `SmallVec` to unpack on stack and SIMD for distance computation.
    #[must_use]
    pub fn distance_asymmetric_l2_raw(
        &self,
        query: &[f32],
        data: &[u8],
        scale: f32,
        bits: u8,
    ) -> f32 {
        let levels = self.params.bits_per_dim.levels() as f32;

        // Dequantization factor: value = q * factor
        // derived from: q / (levels - 1) / scale
        // So factor = 1.0 / ((levels - 1.0) * scale)
        let factor = 1.0 / ((levels - 1.0) * scale);

        match bits {
            4 => {
                // Unpack to stack buffer (up to 256 dims = 1KB). Falls back to heap for larger.
                let mut buffer: SmallVec<[f32; 256]> = SmallVec::with_capacity(query.len());

                let num_pairs = query.len() / 2;

                // Check bounds once to avoid checks in loop
                if data.len() < query.len().div_ceil(2) {
                    // Fallback if data is truncated (shouldn't happen in valid storage)
                    return f32::MAX;
                }

                for i in 0..num_pairs {
                    let byte = unsafe { *data.get_unchecked(i) };
                    buffer.push((byte >> 4) as f32 * factor);
                    buffer.push((byte & 0x0F) as f32 * factor);
                }

                if !query.len().is_multiple_of(2) {
                    let byte = unsafe { *data.get_unchecked(num_pairs) };
                    buffer.push((byte >> 4) as f32 * factor);
                }

                simd_l2_distance(query, &buffer)
            }
            2 => {
                let mut buffer: SmallVec<[f32; 256]> = SmallVec::with_capacity(query.len());
                let num_quads = query.len() / 4;

                if data.len() < query.len().div_ceil(4) {
                    return f32::MAX;
                }

                for i in 0..num_quads {
                    let byte = unsafe { *data.get_unchecked(i) };
                    buffer.push((byte & 0b11) as f32 * factor);
                    buffer.push(((byte >> 2) & 0b11) as f32 * factor);
                    buffer.push(((byte >> 4) & 0b11) as f32 * factor);
                    buffer.push(((byte >> 6) & 0b11) as f32 * factor);
                }

                // Handle remainder
                let remaining = query.len() % 4;
                if remaining > 0 {
                    let byte = unsafe { *data.get_unchecked(num_quads) };
                    for i in 0..remaining {
                        buffer.push(((byte >> (i * 2)) & 0b11) as f32 * factor);
                    }
                }

                simd_l2_distance(query, &buffer)
            }
            _ => {
                // Generic fallback using existing unpack (allocates Vec if > 256, but correct)
                // Actually unpack_quantized returns Vec<u8>, so this path allocates.
                // That's fine for non-optimized bit widths.
                let unpacked = self.unpack_quantized(data, bits, query.len());
                let mut buffer: SmallVec<[f32; 256]> = SmallVec::with_capacity(query.len());

                for &q in &unpacked {
                    buffer.push(q as f32 * factor);
                }

                simd_l2_distance(query, &buffer)
            }
        }
    }

    // SIMD-optimized distance functions

    /// Compute L2 distance using SIMD acceleration
    ///
    /// Uses runtime CPU detection to select the best SIMD implementation:
    /// - `x86_64`: AVX2 > SSE2 > scalar
    /// - aarch64: NEON > scalar
    #[inline]
    #[must_use]
    pub fn distance_l2_simd(&self, qv1: &QuantizedVector, qv2: &QuantizedVector) -> f32 {
        // Reconstruct to f32 vectors
        let v1 = self.reconstruct(&qv1.data, qv1.scale, qv1.dimensions);
        let v2 = self.reconstruct(&qv2.data, qv2.scale, qv2.dimensions);

        // Use SIMD distance computation
        simd_l2_distance(&v1, &v2)
    }

    /// Compute cosine distance using SIMD acceleration
    #[inline]
    #[must_use]
    pub fn distance_cosine_simd(&self, qv1: &QuantizedVector, qv2: &QuantizedVector) -> f32 {
        let v1 = self.reconstruct(&qv1.data, qv1.scale, qv1.dimensions);
        let v2 = self.reconstruct(&qv2.data, qv2.scale, qv2.dimensions);

        simd_cosine_distance(&v1, &v2)
    }
}

// SIMD distance computation functions

/// Compute L2 distance using SIMD
#[inline]
fn simd_l2_distance(v1: &[f32], v2: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { l2_distance_avx2(v1, v2) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { l2_distance_sse2(v1, v2) };
        }
        // Scalar fallback for x86_64 without SIMD
        l2_distance_scalar(v1, v2)
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON always available on aarch64
        unsafe { l2_distance_neon(v1, v2) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Scalar fallback for other architectures
        l2_distance_scalar(v1, v2)
    }
}

/// Compute cosine distance using SIMD
#[inline]
fn simd_cosine_distance(v1: &[f32], v2: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { cosine_distance_avx2(v1, v2) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { cosine_distance_sse2(v1, v2) };
        }
        // Scalar fallback for x86_64 without SIMD
        cosine_distance_scalar(v1, v2)
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON always available on aarch64
        unsafe { cosine_distance_neon(v1, v2) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Scalar fallback for other architectures
        cosine_distance_scalar(v1, v2)
    }
}

// Scalar implementations

#[inline]
#[allow(dead_code)] // Used as SIMD fallback on x86_64 without AVX2/SSE2
fn l2_distance_scalar(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[inline]
#[allow(dead_code)] // Used as SIMD fallback on x86_64 without AVX2/SSE2
fn cosine_distance_scalar(v1: &[f32], v2: &[f32]) -> f32 {
    let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let norm2: f32 = v2.iter().map(|b| b * b).sum::<f32>().sqrt();

    if norm1 < 1e-10 || norm2 < 1e-10 {
        return 1.0;
    }

    let cosine_sim = dot / (norm1 * norm2);
    1.0 - cosine_sim
}

// AVX2 implementations (x86_64)

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn l2_distance_avx2(v1: &[f32], v2: &[f32]) -> f32 {
    unsafe {
        let len = v1.len().min(v2.len());
        let mut sum = _mm256_setzero_ps();

        let chunks = len / 8;
        for i in 0..chunks {
            let a = _mm256_loadu_ps(v1.as_ptr().add(i * 8));
            let b = _mm256_loadu_ps(v2.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(a, b);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        // Horizontal sum
        let sum_high = _mm256_extractf128_ps(sum, 1);
        let sum_low = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(sum_low, sum_high);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Handle remainder
        for i in (chunks * 8)..len {
            let diff = v1[i] - v2[i];
            result += diff * diff;
        }

        result.sqrt()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn cosine_distance_avx2(v1: &[f32], v2: &[f32]) -> f32 {
    unsafe {
        let len = v1.len().min(v2.len());
        let mut dot_sum = _mm256_setzero_ps();
        let mut norm1_sum = _mm256_setzero_ps();
        let mut norm2_sum = _mm256_setzero_ps();

        let chunks = len / 8;
        for i in 0..chunks {
            let a = _mm256_loadu_ps(v1.as_ptr().add(i * 8));
            let b = _mm256_loadu_ps(v2.as_ptr().add(i * 8));
            dot_sum = _mm256_fmadd_ps(a, b, dot_sum);
            norm1_sum = _mm256_fmadd_ps(a, a, norm1_sum);
            norm2_sum = _mm256_fmadd_ps(b, b, norm2_sum);
        }

        // Horizontal sums
        let mut dot = horizontal_sum_avx2(dot_sum);
        let mut norm1 = horizontal_sum_avx2(norm1_sum);
        let mut norm2 = horizontal_sum_avx2(norm2_sum);

        // Handle remainder
        for i in (chunks * 8)..len {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }

        if norm1 < 1e-10 || norm2 < 1e-10 {
            return 1.0;
        }

        let cosine_sim = dot / (norm1.sqrt() * norm2.sqrt());
        1.0 - cosine_sim
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    unsafe {
        let sum_high = _mm256_extractf128_ps(v, 1);
        let sum_low = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(sum_low, sum_high);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        _mm_cvtss_f32(sum32)
    }
}

// SSE2 implementations (x86_64 fallback)

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn l2_distance_sse2(v1: &[f32], v2: &[f32]) -> f32 {
    unsafe {
        let len = v1.len().min(v2.len());
        let mut sum = _mm_setzero_ps();

        let chunks = len / 4;
        for i in 0..chunks {
            let a = _mm_loadu_ps(v1.as_ptr().add(i * 4));
            let b = _mm_loadu_ps(v2.as_ptr().add(i * 4));
            let diff = _mm_sub_ps(a, b);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }

        // Horizontal sum
        let sum64 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Handle remainder
        for i in (chunks * 4)..len {
            let diff = v1[i] - v2[i];
            result += diff * diff;
        }

        result.sqrt()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn cosine_distance_sse2(v1: &[f32], v2: &[f32]) -> f32 {
    unsafe {
        let len = v1.len().min(v2.len());
        let mut dot_sum = _mm_setzero_ps();
        let mut norm1_sum = _mm_setzero_ps();
        let mut norm2_sum = _mm_setzero_ps();

        let chunks = len / 4;
        for i in 0..chunks {
            let a = _mm_loadu_ps(v1.as_ptr().add(i * 4));
            let b = _mm_loadu_ps(v2.as_ptr().add(i * 4));
            dot_sum = _mm_add_ps(dot_sum, _mm_mul_ps(a, b));
            norm1_sum = _mm_add_ps(norm1_sum, _mm_mul_ps(a, a));
            norm2_sum = _mm_add_ps(norm2_sum, _mm_mul_ps(b, b));
        }

        // Horizontal sums
        let mut dot = horizontal_sum_sse2(dot_sum);
        let mut norm1 = horizontal_sum_sse2(norm1_sum);
        let mut norm2 = horizontal_sum_sse2(norm2_sum);

        // Handle remainder
        for i in (chunks * 4)..len {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }

        if norm1 < 1e-10 || norm2 < 1e-10 {
            return 1.0;
        }

        let cosine_sim = dot / (norm1.sqrt() * norm2.sqrt());
        1.0 - cosine_sim
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum_sse2(v: __m128) -> f32 {
    unsafe {
        let sum64 = _mm_add_ps(v, _mm_movehl_ps(v, v));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        _mm_cvtss_f32(sum32)
    }
}

// NEON implementations (aarch64)

#[cfg(target_arch = "aarch64")]
unsafe fn l2_distance_neon(v1: &[f32], v2: &[f32]) -> f32 {
    let len = v1.len().min(v2.len());

    // SAFETY: All SIMD operations wrapped in unsafe block for Rust 2024
    unsafe {
        let mut sum = vdupq_n_f32(0.0);

        let chunks = len / 4;
        for i in 0..chunks {
            let a = vld1q_f32(v1.as_ptr().add(i * 4));
            let b = vld1q_f32(v2.as_ptr().add(i * 4));
            let diff = vsubq_f32(a, b);
            sum = vfmaq_f32(sum, diff, diff);
        }

        // Horizontal sum
        let mut result = vaddvq_f32(sum);

        // Handle remainder
        for i in (chunks * 4)..len {
            let diff = v1[i] - v2[i];
            result += diff * diff;
        }

        result.sqrt()
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn cosine_distance_neon(v1: &[f32], v2: &[f32]) -> f32 {
    let len = v1.len().min(v2.len());

    // SAFETY: All SIMD operations wrapped in unsafe block for Rust 2024
    unsafe {
        let mut dot_sum = vdupq_n_f32(0.0);
        let mut norm1_sum = vdupq_n_f32(0.0);
        let mut norm2_sum = vdupq_n_f32(0.0);

        let chunks = len / 4;
        for i in 0..chunks {
            let a = vld1q_f32(v1.as_ptr().add(i * 4));
            let b = vld1q_f32(v2.as_ptr().add(i * 4));
            dot_sum = vfmaq_f32(dot_sum, a, b);
            norm1_sum = vfmaq_f32(norm1_sum, a, a);
            norm2_sum = vfmaq_f32(norm2_sum, b, b);
        }

        // Horizontal sums
        let mut dot = vaddvq_f32(dot_sum);
        let mut norm1 = vaddvq_f32(norm1_sum);
        let mut norm2 = vaddvq_f32(norm2_sum);

        // Handle remainder
        for i in (chunks * 4)..len {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }

        if norm1 < 1e-10 || norm2 < 1e-10 {
            return 1.0;
        }

        let cosine_sim = dot / (norm1.sqrt() * norm2.sqrt());
        1.0 - cosine_sim
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_bits_conversion() {
        assert_eq!(QuantizationBits::Bits2.to_u8(), 2);
        assert_eq!(QuantizationBits::Bits4.to_u8(), 4);
        assert_eq!(QuantizationBits::Bits8.to_u8(), 8);
    }

    #[test]
    fn test_quantization_bits_levels() {
        assert_eq!(QuantizationBits::Bits2.levels(), 4); // 2^2
        assert_eq!(QuantizationBits::Bits4.levels(), 16); // 2^4
        assert_eq!(QuantizationBits::Bits8.levels(), 256); // 2^8
    }

    #[test]
    fn test_quantization_bits_compression() {
        assert_eq!(QuantizationBits::Bits2.compression_ratio(), 16.0); // 32/2
        assert_eq!(QuantizationBits::Bits4.compression_ratio(), 8.0); // 32/4
        assert_eq!(QuantizationBits::Bits8.compression_ratio(), 4.0); // 32/8
    }

    #[test]
    fn test_quantization_bits_values_per_byte() {
        assert_eq!(QuantizationBits::Bits2.values_per_byte(), 4); // 8/2
        assert_eq!(QuantizationBits::Bits4.values_per_byte(), 2); // 8/4
        assert_eq!(QuantizationBits::Bits8.values_per_byte(), 1); // 8/8
    }

    #[test]
    fn test_default_params() {
        let params = RaBitQParams::default();
        assert_eq!(params.bits_per_dim, QuantizationBits::Bits4);
        assert_eq!(params.num_rescale_factors, 12);
        assert_eq!(params.rescale_range, (0.5, 2.0));
    }

    #[test]
    fn test_preset_params() {
        let params2 = RaBitQParams::bits2();
        assert_eq!(params2.bits_per_dim, QuantizationBits::Bits2);

        let params4 = RaBitQParams::bits4();
        assert_eq!(params4.bits_per_dim, QuantizationBits::Bits4);

        let params8 = RaBitQParams::bits8();
        assert_eq!(params8.bits_per_dim, QuantizationBits::Bits8);
        assert_eq!(params8.num_rescale_factors, 16);
    }

    #[test]
    fn test_quantized_vector_creation() {
        let data = vec![0u8, 128, 255];
        let qv = QuantizedVector::new(data.clone(), 1.5, 8, 3);

        assert_eq!(qv.data, data);
        assert_eq!(qv.scale, 1.5);
        assert_eq!(qv.bits, 8);
        assert_eq!(qv.dimensions, 3);
    }

    #[test]
    fn test_quantized_vector_memory() {
        let data = vec![0u8; 16]; // 16 bytes
        let qv = QuantizedVector::new(data, 1.0, 4, 32);

        // Should be: struct overhead + data length
        let expected_min = 16; // At least the data
        assert!(qv.memory_bytes() >= expected_min);
    }

    #[test]
    fn test_quantized_vector_compression_ratio() {
        // 128 dimensions, 4-bit = 64 bytes
        let data = vec![0u8; 64];
        let qv = QuantizedVector::new(data, 1.0, 4, 128);

        // Original: 128 * 4 = 512 bytes
        // Compressed: 64 + 4 (scale) + 1 (bits) = 69 bytes
        // Ratio: 512 / 69 ≈ 7.4x
        let ratio = qv.compression_ratio();
        assert!(ratio > 7.0 && ratio < 8.0);
    }

    #[test]
    fn test_create_quantizer() {
        let quantizer = RaBitQ::default_4bit();
        assert_eq!(quantizer.params().bits_per_dim, QuantizationBits::Bits4);
    }

    // Phase 2 Tests: Core Algorithm

    #[test]
    fn test_generate_scales() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits4,
            num_rescale_factors: 5,
            rescale_range: (0.5, 1.5),
        });

        let scales = quantizer.generate_scales();
        assert_eq!(scales.len(), 5);
        assert_eq!(scales[0], 0.5);
        assert_eq!(scales[4], 1.5);
        assert!((scales[2] - 1.0).abs() < 0.01); // Middle should be ~1.0
    }

    #[test]
    fn test_generate_scales_single() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits4,
            num_rescale_factors: 1,
            rescale_range: (0.5, 1.5),
        });

        let scales = quantizer.generate_scales();
        assert_eq!(scales.len(), 1);
        assert_eq!(scales[0], 1.0); // Average of min and max
    }

    #[test]
    fn test_pack_unpack_2bit() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits2,
            ..Default::default()
        });

        // 8 values (2 bits each) = 2 bytes
        let values = vec![0u8, 1, 2, 3, 0, 1, 2, 3];
        let packed = quantizer.pack_quantized(&values, 2);
        assert_eq!(packed.len(), 2); // 8 values / 4 per byte = 2 bytes

        let unpacked = quantizer.unpack_quantized(&packed, 2, 8);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_pack_unpack_4bit() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits4,
            ..Default::default()
        });

        // 8 values (4 bits each) = 4 bytes
        let values = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let packed = quantizer.pack_quantized(&values, 4);
        assert_eq!(packed.len(), 4); // 8 values / 2 per byte = 4 bytes

        let unpacked = quantizer.unpack_quantized(&packed, 4, 8);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_pack_unpack_8bit() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8,
            ..Default::default()
        });

        // 8 values (8 bits each) = 8 bytes
        let values = vec![0u8, 10, 20, 30, 40, 50, 60, 70];
        let packed = quantizer.pack_quantized(&values, 8);
        assert_eq!(packed.len(), 8); // 8 values = 8 bytes

        let unpacked = quantizer.unpack_quantized(&packed, 8, 8);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_quantize_simple_vector() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits4,
            num_rescale_factors: 4,
            rescale_range: (0.5, 1.5),
        });

        // Simple vector: [0.0, 0.25, 0.5, 0.75, 1.0]
        let vector = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let quantized = quantizer.quantize(&vector);

        // Check structure
        assert_eq!(quantized.dimensions, 5);
        assert_eq!(quantized.bits, 4);
        assert!(quantized.scale > 0.0);

        // Check compression: 5 floats * 4 bytes = 20 bytes original
        // Quantized: 5 values * 4 bits = 20 bits = 3 bytes (rounded up)
        assert!(quantized.data.len() <= 4);
    }

    #[test]
    fn test_quantize_reconstruct_accuracy() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8, // High precision
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
        });

        // Test vector
        let vector = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let quantized = quantizer.quantize(&vector);

        // Reconstruct
        let reconstructed = quantizer.reconstruct(&quantized.data, quantized.scale, vector.len());

        // Check reconstruction is close (8-bit should be accurate)
        for (orig, recon) in vector.iter().zip(reconstructed.iter()) {
            let error = (orig - recon).abs();
            assert!(error < 0.1, "Error too large: {orig} vs {recon}");
        }
    }

    #[test]
    fn test_quantize_uniform_vector() {
        let quantizer = RaBitQ::default_4bit();

        // All values the same
        let vector = vec![0.5; 10];
        let quantized = quantizer.quantize(&vector);

        // Reconstruct should also be uniform
        let reconstructed = quantizer.reconstruct(&quantized.data, quantized.scale, vector.len());

        // All values should be similar
        let avg = reconstructed.iter().sum::<f32>() / reconstructed.len() as f32;
        for &val in &reconstructed {
            assert!((val - avg).abs() < 0.2);
        }
    }

    #[test]
    fn test_compute_error() {
        let quantizer = RaBitQ::default_4bit();

        let original = vec![0.1, 0.2, 0.3, 0.4];
        let quantized_vec = quantizer.quantize(&original);

        // Compute error
        let error = quantizer.compute_error(&original, &quantized_vec.data, quantized_vec.scale);

        // Error should be non-negative and finite
        assert!(error >= 0.0);
        assert!(error.is_finite());
    }

    #[test]
    fn test_quantize_different_bit_widths() {
        let test_vector = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        // Test 2-bit
        let q2 = RaBitQ::new(RaBitQParams::bits2());
        let qv2 = q2.quantize(&test_vector);
        assert_eq!(qv2.bits, 2);

        // Test 4-bit
        let q4 = RaBitQ::default_4bit();
        let qv4 = q4.quantize(&test_vector);
        assert_eq!(qv4.bits, 4);

        // Test 8-bit
        let q8 = RaBitQ::new(RaBitQParams::bits8());
        let qv8 = q8.quantize(&test_vector);
        assert_eq!(qv8.bits, 8);

        // Higher bits = larger packed size (for same dimensions)
        assert!(qv2.data.len() <= qv4.data.len());
        assert!(qv4.data.len() <= qv8.data.len());
    }

    #[test]
    fn test_quantize_high_dimensional() {
        let quantizer = RaBitQ::default_4bit();

        // 128D vector (like small embeddings)
        let vector: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
        let quantized = quantizer.quantize(&vector);

        assert_eq!(quantized.dimensions, 128);
        assert_eq!(quantized.bits, 4);

        // 128 dimensions * 4 bits = 512 bits = 64 bytes
        assert_eq!(quantized.data.len(), 64);

        // Verify reconstruction
        let reconstructed = quantizer.reconstruct(&quantized.data, quantized.scale, 128);
        assert_eq!(reconstructed.len(), 128);
    }

    // Phase 3 Tests: Distance Computation

    #[test]
    fn test_distance_l2() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8, // High precision
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
        });

        let v1 = vec![0.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist = quantizer.distance_l2(&qv1, &qv2);

        // Distance should be approximately 1.0
        assert!((dist - 1.0).abs() < 0.2, "Distance: {dist}");
    }

    #[test]
    fn test_distance_l2_identical() {
        let quantizer = RaBitQ::default_4bit();

        let v = vec![0.5, 0.3, 0.8, 0.2];
        let qv1 = quantizer.quantize(&v);
        let qv2 = quantizer.quantize(&v);

        let dist = quantizer.distance_l2(&qv1, &qv2);

        // Identical vectors should have near-zero distance
        assert!(dist < 0.3, "Distance should be near zero, got: {dist}");
    }

    #[test]
    fn test_distance_cosine() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8,
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
        });

        // Orthogonal vectors
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist = quantizer.distance_cosine(&qv1, &qv2);

        // Orthogonal vectors: cosine = 0, distance = 1
        assert!((dist - 1.0).abs() < 0.3, "Distance: {dist}");
    }

    #[test]
    fn test_distance_cosine_identical() {
        let quantizer = RaBitQ::default_4bit();

        let v = vec![0.5, 0.3, 0.8];
        let qv1 = quantizer.quantize(&v);
        let qv2 = quantizer.quantize(&v);

        let dist = quantizer.distance_cosine(&qv1, &qv2);

        // Identical vectors: cosine = 1, distance = 0
        assert!(dist < 0.2, "Distance should be near zero, got: {dist}");
    }

    #[test]
    fn test_distance_dot() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8,
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
        });

        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist = quantizer.distance_dot(&qv1, &qv2);

        // Dot product of [1,0,0] with itself = 1, negated = -1
        assert!((dist + 1.0).abs() < 0.3, "Distance: {dist}");
    }

    #[test]
    fn test_distance_approximate() {
        let quantizer = RaBitQ::default_4bit();

        let v1 = vec![0.0, 0.0, 0.0];
        let v2 = vec![0.5, 0.5, 0.5];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist_approx = quantizer.distance_approximate(&qv1, &qv2);
        let dist_exact = quantizer.distance_l2(&qv1, &qv2);

        // Approximate should be non-negative and finite
        assert!(dist_approx >= 0.0);
        assert!(dist_approx.is_finite());

        // Approximate and exact should be correlated (not exact match)
        // Just verify both increase/decrease together
        let v3 = vec![1.0, 1.0, 1.0];
        let qv3 = quantizer.quantize(&v3);

        let dist_approx2 = quantizer.distance_approximate(&qv1, &qv3);
        let dist_exact2 = quantizer.distance_l2(&qv1, &qv3);

        // If v3 is farther from v1 than v2, both metrics should reflect that
        if dist_exact2 > dist_exact {
            assert!(dist_approx2 > dist_approx * 0.5); // Allow some variance
        }
    }

    #[test]
    fn test_distance_correlation() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8, // High precision for correlation
            num_rescale_factors: 12,
            rescale_range: (0.8, 1.2),
        });

        // Create multiple vectors
        let vectors = [
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        // Quantize all
        let quantized: Vec<QuantizedVector> =
            vectors.iter().map(|v| quantizer.quantize(v)).collect();

        // Ground truth L2 distances
        let ground_truth_01 = vectors[0]
            .iter()
            .zip(vectors[1].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        let ground_truth_02 = vectors[0]
            .iter()
            .zip(vectors[2].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Quantized distances
        let quantized_01 = quantizer.distance_l2(&quantized[0], &quantized[1]);
        let quantized_02 = quantizer.distance_l2(&quantized[0], &quantized[2]);

        // Check correlation: if ground truth says v2 > v1, quantized should too
        if ground_truth_02 > ground_truth_01 {
            assert!(
                quantized_02 > quantized_01 * 0.8,
                "Order not preserved: {quantized_01} vs {quantized_02}"
            );
        }
    }

    #[test]
    fn test_distance_zero_vectors() {
        let quantizer = RaBitQ::default_4bit();

        let v_zero = vec![0.0, 0.0, 0.0];
        let qv_zero = quantizer.quantize(&v_zero);

        // Distance to itself should be zero
        let dist = quantizer.distance_l2(&qv_zero, &qv_zero);
        assert!(dist < 0.1);

        // Cosine distance with zero vector should handle gracefully
        let dist_cosine = quantizer.distance_cosine(&qv_zero, &qv_zero);
        assert!(dist_cosine.is_finite());
    }

    #[test]
    fn test_distance_high_dimensional() {
        let quantizer = RaBitQ::default_4bit();

        // 128D vectors
        let v1: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
        let v2: Vec<f32> = (0..128).map(|i| ((i + 10) as f32) / 128.0).collect();

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        // All distance metrics should work on high-dimensional vectors
        let dist_l2 = quantizer.distance_l2(&qv1, &qv2);
        let dist_cosine = quantizer.distance_cosine(&qv1, &qv2);
        let dist_dot = quantizer.distance_dot(&qv1, &qv2);
        let dist_approx = quantizer.distance_approximate(&qv1, &qv2);

        assert!(dist_l2 > 0.0 && dist_l2.is_finite());
        assert!(dist_cosine >= 0.0 && dist_cosine.is_finite());
        assert!(dist_dot.is_finite());
        assert!(dist_approx > 0.0 && dist_approx.is_finite());
    }

    #[test]
    fn test_distance_asymmetric_l2() {
        let quantizer = RaBitQ::default_4bit();

        let query = vec![0.1, 0.2, 0.3, 0.4];
        // Vector close to query
        let vector = vec![0.12, 0.22, 0.32, 0.42];

        let quantized = quantizer.quantize(&vector);

        // Symmetric distance (allocates)
        let dist_sym = quantizer.distance_l2_simd(&quantized, &quantizer.quantize(&query));

        // Asymmetric distance (no allocation)
        let dist_asym = quantizer.distance_asymmetric_l2(&query, &quantized);

        // Should be reasonably close (asymmetric is actually MORE accurate because query is exact)
        // But for this test, just ensure it's sane
        assert!(dist_asym >= 0.0);
        assert!((dist_asym - dist_sym).abs() < 0.2);
    }

    // Phase 4 Tests: SIMD Optimizations

    #[test]
    fn test_simd_l2_matches_scalar() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8, // High precision
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
        });

        let v1 = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let v2 = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist_scalar = quantizer.distance_l2(&qv1, &qv2);
        let dist_simd = quantizer.distance_l2_simd(&qv1, &qv2);

        // SIMD should match scalar within floating point precision
        let diff = (dist_scalar - dist_simd).abs();
        assert!(diff < 0.01, "SIMD vs scalar: {dist_simd} vs {dist_scalar}");
    }

    #[test]
    fn test_simd_cosine_matches_scalar() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8,
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
        });

        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist_scalar = quantizer.distance_cosine(&qv1, &qv2);
        let dist_simd = quantizer.distance_cosine_simd(&qv1, &qv2);

        // SIMD should match scalar within floating point precision
        let diff = (dist_scalar - dist_simd).abs();
        assert!(diff < 0.01, "SIMD vs scalar: {dist_simd} vs {dist_scalar}");
    }

    #[test]
    fn test_simd_high_dimensional() {
        let quantizer = RaBitQ::default_4bit();

        // 128D vectors (realistic embeddings)
        let v1: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
        let v2: Vec<f32> = (0..128).map(|i| ((i + 1) as f32) / 128.0).collect();

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist_scalar = quantizer.distance_l2(&qv1, &qv2);
        let dist_simd = quantizer.distance_l2_simd(&qv1, &qv2);

        // Should be close (allow for quantization + FP variance)
        let diff = (dist_scalar - dist_simd).abs();
        assert!(
            diff < 0.1,
            "High-D SIMD vs scalar: {dist_simd} vs {dist_scalar}"
        );
    }

    #[test]
    fn test_simd_scalar_fallback() {
        let quantizer = RaBitQ::default_4bit();

        // Small vector (tests remainder handling)
        let v1 = vec![0.1, 0.2, 0.3];
        let v2 = vec![0.4, 0.5, 0.6];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        // Should not crash on small vectors
        let dist_l2 = quantizer.distance_l2_simd(&qv1, &qv2);
        let dist_cosine = quantizer.distance_cosine_simd(&qv1, &qv2);

        assert!(dist_l2.is_finite());
        assert!(dist_cosine.is_finite());
    }

    #[test]
    fn test_simd_performance_improvement() {
        let quantizer = RaBitQ::default_4bit();

        // Large vectors (1536D like OpenAI embeddings)
        let v1: Vec<f32> = (0..1536).map(|i| (i as f32) / 1536.0).collect();
        let v2: Vec<f32> = (0..1536).map(|i| ((i + 10) as f32) / 1536.0).collect();

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        // Just verify SIMD works on large vectors
        let dist_simd = quantizer.distance_l2_simd(&qv1, &qv2);
        assert!(dist_simd > 0.0 && dist_simd.is_finite());

        // Note: Actual performance benchmarks in Phase 6
    }

    #[test]
    fn test_scalar_distance_functions() {
        // Test the scalar fallback functions directly
        let v1 = vec![0.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];

        let dist = l2_distance_scalar(&v1, &v2);
        assert!((dist - 1.0).abs() < 0.001);

        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];

        let dist = cosine_distance_scalar(&v1, &v2);
        assert!((dist - 1.0).abs() < 0.001);
    }

    // ADC Tests

    #[test]
    fn test_adc_table_creation() {
        let quantizer = RaBitQ::default_4bit();
        let query = vec![0.1, 0.2, 0.3, 0.4];
        let scale = 1.0;

        let adc = quantizer.build_adc_table_with_scale(&query, scale);

        // Check structure
        assert_eq!(adc.dimensions, 4);
        assert_eq!(adc.bits, 4);
        assert_eq!(adc.table.len(), 4);

        // Each dimension should have 16 codes (4-bit)
        for dim_table in &adc.table {
            assert_eq!(dim_table.len(), 16);
        }
    }

    #[test]
    fn test_adc_table_2bit() {
        let quantizer = RaBitQ::new(RaBitQParams::bits2());
        let query = vec![0.1, 0.2, 0.3, 0.4];
        let scale = 1.0;

        let adc = quantizer.build_adc_table_with_scale(&query, scale);

        // Each dimension should have 4 codes (2-bit)
        for dim_table in &adc.table {
            assert_eq!(dim_table.len(), 4);
        }
    }

    #[test]
    fn test_adc_distance_matches_asymmetric() {
        let quantizer = RaBitQ::default_4bit();

        // Create query and vector
        let query = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let vector = vec![0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85];

        // Quantize the vector
        let quantized = quantizer.quantize(&vector);

        // Compute distance with asymmetric method
        let dist_asymmetric = quantizer.distance_asymmetric_l2(&query, &quantized);

        // Compute distance with ADC (using build_adc_table_with_scale for untrained quantizer)
        let adc = quantizer.build_adc_table_with_scale(&query, quantized.scale);
        let dist_adc = adc.distance(&quantized.data);

        // ADC should give similar results to asymmetric distance
        // They use different computation paths but should be close
        let diff = (dist_asymmetric - dist_adc).abs();
        assert!(
            diff < 0.1,
            "ADC vs asymmetric: {dist_adc} vs {dist_asymmetric}, diff: {diff}"
        );
    }

    #[test]
    fn test_adc_distance_accuracy() {
        let quantizer = RaBitQ::new(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8, // High precision
            num_rescale_factors: 16,
            rescale_range: (0.8, 1.2),
        });

        let query = vec![0.1, 0.2, 0.3, 0.4];
        let vector = vec![0.1, 0.2, 0.3, 0.4]; // Same as query

        let quantized = quantizer.quantize(&vector);

        // Build ADC table
        let adc = quantizer.build_adc_table_with_scale(&query, quantized.scale);

        // Distance should be near zero (same vector)
        let dist = adc.distance(&quantized.data);
        assert!(dist < 0.2, "Distance should be near zero, got: {dist}");
    }

    #[test]
    fn test_adc_distance_ordering() {
        let quantizer = RaBitQ::default_4bit();

        let query = vec![0.5, 0.5, 0.5, 0.5];
        let v1 = vec![0.5, 0.5, 0.5, 0.5]; // Closest
        let v2 = vec![0.6, 0.6, 0.6, 0.6]; // Medium
        let v3 = vec![0.9, 0.9, 0.9, 0.9]; // Farthest

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);
        let qv3 = quantizer.quantize(&v3);

        // Build ADC tables with respective scales
        let adc1 = quantizer.build_adc_table_with_scale(&query, qv1.scale);
        let adc2 = quantizer.build_adc_table_with_scale(&query, qv2.scale);
        let adc3 = quantizer.build_adc_table_with_scale(&query, qv3.scale);

        let dist1 = adc1.distance(&qv1.data);
        let dist2 = adc2.distance(&qv2.data);
        let dist3 = adc3.distance(&qv3.data);

        // Order should be preserved
        assert!(
            dist1 < dist2,
            "v1 should be closer than v2: {dist1} vs {dist2}"
        );
        assert!(
            dist2 < dist3,
            "v2 should be closer than v3: {dist2} vs {dist3}"
        );
    }

    #[test]
    fn test_adc_high_dimensional() {
        let quantizer = RaBitQ::default_4bit();

        // 128D vectors (realistic embedding size)
        let query: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
        let vector: Vec<f32> = (0..128).map(|i| ((i + 5) as f32) / 128.0).collect();

        let quantized = quantizer.quantize(&vector);

        // Build ADC table
        let adc = quantizer.build_adc_table_with_scale(&query, quantized.scale);

        // Should handle high dimensions without panic
        let dist = adc.distance(&quantized.data);
        assert!(dist > 0.0 && dist.is_finite());
    }

    #[test]
    fn test_adc_batch_search() {
        let quantizer = RaBitQ::default_4bit();

        let query = vec![0.5, 0.5, 0.5, 0.5];
        let candidates = [
            vec![0.5, 0.5, 0.5, 0.5],
            vec![0.6, 0.6, 0.6, 0.6],
            vec![0.4, 0.4, 0.4, 0.4],
            vec![0.7, 0.7, 0.7, 0.7],
        ];

        // Quantize all candidates
        let quantized: Vec<QuantizedVector> =
            candidates.iter().map(|v| quantizer.quantize(v)).collect();

        // Scan all candidates using ADC tables
        let mut results: Vec<(usize, f32)> = quantized
            .iter()
            .enumerate()
            .map(|(i, qv)| {
                let adc = quantizer.build_adc_table_with_scale(&query, qv.scale);
                (i, adc.distance(&qv.data))
            })
            .collect();

        // Sort by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // First result should be index 0 (identical to query)
        assert_eq!(results[0].0, 0, "Results: {results:?}");
    }

    #[test]
    fn test_adc_distance_squared() {
        let quantizer = RaBitQ::default_4bit();

        let query = vec![0.0, 0.0, 0.0];
        let vector = vec![1.0, 0.0, 0.0];

        let quantized = quantizer.quantize(&vector);
        let adc = quantizer.build_adc_table_with_scale(&query, quantized.scale);

        let dist_squared = adc.distance_squared(&quantized.data);
        let dist = adc.distance(&quantized.data);

        // distance_squared should be dist^2 (approximately)
        let diff = (dist_squared - dist * dist).abs();
        assert!(
            diff < 0.01,
            "distance_squared != dist^2: {} vs {}",
            dist_squared,
            dist * dist
        );
    }

    #[test]
    fn test_adc_simd_matches_scalar() {
        let quantizer = RaBitQ::default_4bit();

        let query = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let vector = vec![0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85];

        let quantized = quantizer.quantize(&vector);
        let adc = quantizer.build_adc_table_with_scale(&query, quantized.scale);

        let dist_scalar = adc.distance_squared(&quantized.data);
        let dist_simd = adc.distance_squared_simd(&quantized.data);

        // SIMD should match scalar within floating point precision
        let diff = (dist_scalar - dist_simd).abs();
        assert!(diff < 0.01, "SIMD vs scalar: {dist_simd} vs {dist_scalar}");
    }

    #[test]
    fn test_adc_simd_high_dimensional() {
        let quantizer = RaBitQ::default_4bit();

        // 1536D vectors (OpenAI embeddings)
        let query: Vec<f32> = (0..1536).map(|i| (i as f32) / 1536.0).collect();
        let vector: Vec<f32> = (0..1536).map(|i| ((i + 10) as f32) / 1536.0).collect();

        let quantized = quantizer.quantize(&vector);
        let adc = quantizer.build_adc_table_with_scale(&query, quantized.scale);

        // Should handle large dimensions efficiently
        let dist_simd = adc.distance_squared_simd(&quantized.data);
        assert!(dist_simd > 0.0 && dist_simd.is_finite());
    }

    #[test]
    fn test_adc_memory_usage() {
        let quantizer = RaBitQ::default_4bit();

        let query: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
        let adc = quantizer.build_adc_table_with_scale(&query, 1.0);

        let memory = adc.memory_bytes();

        // For 128D, 4-bit: 128 * 16 * 4 bytes = 8KB (plus overhead)
        let expected_min = 128 * 16 * 4;
        assert!(
            memory >= expected_min,
            "Memory {memory} should be at least {expected_min}"
        );
    }

    #[test]
    fn test_adc_different_scales() {
        let quantizer = RaBitQ::default_4bit();

        let query = vec![0.5, 0.5, 0.5, 0.5];
        let vector = vec![0.6, 0.6, 0.6, 0.6];

        let quantized = quantizer.quantize(&vector);

        // Build ADC tables with different scales
        let adc1 = quantizer.build_adc_table_with_scale(&query, 0.5);
        let adc2 = quantizer.build_adc_table_with_scale(&query, 1.0);
        let adc3 = quantizer.build_adc_table_with_scale(&query, 2.0);

        // Distances should differ based on scale
        let dist1 = adc1.distance(&quantized.data);
        let dist2 = adc2.distance(&quantized.data);
        let dist3 = adc3.distance(&quantized.data);

        // All should be valid finite numbers
        assert!(dist1.is_finite());
        assert!(dist2.is_finite());
        assert!(dist3.is_finite());
    }

    #[test]
    fn test_adc_edge_cases() {
        let quantizer = RaBitQ::default_4bit();

        // Test with very small vector
        let query = vec![0.5];
        let vector = vec![0.6];
        let quantized = quantizer.quantize(&vector);
        let adc = quantizer.build_adc_table_with_scale(&query, quantized.scale);
        let dist = adc.distance(&quantized.data);
        assert!(dist.is_finite());

        // Test with all zeros
        let query = vec![0.0, 0.0, 0.0, 0.0];
        let vector = vec![0.0, 0.0, 0.0, 0.0];
        let quantized = quantizer.quantize(&vector);
        let adc = quantizer.build_adc_table_with_scale(&query, quantized.scale);
        let dist = adc.distance(&quantized.data);
        assert!(dist.is_finite());
    }

    #[test]
    fn test_adc_2bit_accuracy() {
        let quantizer = RaBitQ::new(RaBitQParams::bits2());

        let query = vec![0.1, 0.2, 0.3, 0.4];
        let vector = vec![0.12, 0.22, 0.32, 0.42];

        let quantized = quantizer.quantize(&vector);

        // Test ADC for 2-bit quantization
        let adc = quantizer.build_adc_table_with_scale(&query, quantized.scale);
        let dist_adc = adc.distance(&quantized.data);
        let dist_asymmetric = quantizer.distance_asymmetric_l2(&query, &quantized);

        // Should be reasonably close despite lower precision
        let diff = (dist_adc - dist_asymmetric).abs();
        assert!(diff < 0.2, "2-bit ADC diff too large: {diff}");
    }

    #[test]
    fn test_adc_8bit_accuracy() {
        let quantizer = RaBitQ::new(RaBitQParams::bits8());

        let query = vec![0.1, 0.2, 0.3, 0.4];
        let vector = vec![0.12, 0.22, 0.32, 0.42];

        let quantized = quantizer.quantize(&vector);

        // Test ADC for 8-bit quantization (highest precision)
        let adc = quantizer.build_adc_table_with_scale(&query, quantized.scale);
        let dist_adc = adc.distance(&quantized.data);
        let dist_asymmetric = quantizer.distance_asymmetric_l2(&query, &quantized);

        // 8-bit should be very accurate
        let diff = (dist_adc - dist_asymmetric).abs();
        assert!(
            diff < 0.05,
            "8-bit ADC should be highly accurate, diff: {diff}"
        );
    }
}
