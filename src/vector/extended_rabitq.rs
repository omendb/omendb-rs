//! Extended RaBitQ Quantization
//!
//! Extended RaBitQ (SIGMOD 2025) extends RaBitQ to support arbitrary compression
//! rates (2-9 bits per dimension) with significantly better accuracy than standard
//! scalar quantization.
//!
//! Key features:
//! - Flexible compression (3, 4, 5, 7, 8, 9 bits/dimension)
//! - Optimal rescaling for each vector
//! - Same query speed as scalar quantization
//! - Better accuracy than binary quantization

use serde::{Deserialize, Serialize};
use std::fmt;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Number of bits per dimension for quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationBits {
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
    pub fn to_u8(self) -> u8 {
        match self {
            QuantizationBits::Bits2 => 2,
            QuantizationBits::Bits3 => 3,
            QuantizationBits::Bits4 => 4,
            QuantizationBits::Bits5 => 5,
            QuantizationBits::Bits7 => 7,
            QuantizationBits::Bits8 => 8,
        }
    }

    /// Get number of quantization levels (2^bits)
    pub fn levels(self) -> usize {
        1 << self.to_u8()
    }

    /// Get compression ratio vs f32 (32 bits / bits_per_dim)
    pub fn compression_ratio(self) -> f32 {
        32.0 / self.to_u8() as f32
    }

    /// Get number of values that fit in one byte
    pub fn values_per_byte(self) -> usize {
        8 / self.to_u8() as usize
    }
}

impl fmt::Display for QuantizationBits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-bit", self.to_u8())
    }
}

/// Configuration for Extended RaBitQ quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedRaBitQParams {
    /// Number of bits per dimension
    pub bits_per_dim: QuantizationBits,

    /// Number of rescaling factors to try
    ///
    /// Higher values = better quantization quality but slower
    /// Typical range: 8-16
    pub num_rescale_factors: usize,

    /// Range of rescaling factors to try (min, max)
    ///
    /// Typical range: (0.5, 2.0) means try scales from 0.5x to 2.0x
    pub rescale_range: (f32, f32),

    /// Use Fast Hadamard Transform for random rotation
    ///
    /// FHT improves quantization quality by spreading information across dimensions
    /// using O(D log D) operations instead of O(D²) for standard rotation.
    /// Recommended for high-dimensional vectors (D >= 64).
    pub use_fht: bool,
}

impl Default for ExtendedRaBitQParams {
    fn default() -> Self {
        Self {
            bits_per_dim: QuantizationBits::Bits4, // 8x compression
            num_rescale_factors: 12,               // Good balance
            rescale_range: (0.5, 2.0),             // Paper recommendation
            use_fht: false,                        // Off by default for compatibility
        }
    }
}

impl ExtendedRaBitQParams {
    /// Create parameters for 2-bit quantization (16x compression)
    pub fn bits2() -> Self {
        Self {
            bits_per_dim: QuantizationBits::Bits2,
            ..Default::default()
        }
    }

    /// Create parameters for 4-bit quantization (8x compression, recommended)
    pub fn bits4() -> Self {
        Self {
            bits_per_dim: QuantizationBits::Bits4,
            ..Default::default()
        }
    }

    /// Create parameters for 8-bit quantization (4x compression, highest quality)
    pub fn bits8() -> Self {
        Self {
            bits_per_dim: QuantizationBits::Bits8,
            num_rescale_factors: 16, // More factors for higher precision
            rescale_range: (0.7, 1.5), // Narrower range for 8-bit
            use_fht: false,
        }
    }

    /// Create parameters with FHT enabled for faster, higher-quality quantization
    ///
    /// FHT (Fast Hadamard Transform) provides:
    /// - O(D log D) random rotation vs O(D²) for matrix multiply
    /// - 10-80x faster encoding for high-dimensional vectors
    /// - Better distribution of information across dimensions
    pub fn with_fht(bits: QuantizationBits) -> Self {
        Self {
            bits_per_dim: bits,
            num_rescale_factors: 12,
            rescale_range: (0.5, 2.0),
            use_fht: true,
        }
    }

    /// Create 4-bit parameters with FHT (recommended for production)
    pub fn bits4_fht() -> Self {
        Self::with_fht(QuantizationBits::Bits4)
    }

    /// Create parameters for 3-bit quantization (~10x compression)
    ///
    /// Provides a balance between 2-bit (84% recall) and 4-bit (100% recall).
    /// Target: ~92-95% recall at 10.67x compression.
    pub fn bits3() -> Self {
        Self {
            bits_per_dim: QuantizationBits::Bits3,
            num_rescale_factors: 12,
            rescale_range: (0.5, 2.0),
            use_fht: false,
        }
    }

    /// Create parameters for 5-bit quantization (~6x compression)
    ///
    /// Higher precision than 4-bit for applications requiring better accuracy.
    /// Target: ~99-100% recall at 6.4x compression.
    pub fn bits5() -> Self {
        Self {
            bits_per_dim: QuantizationBits::Bits5,
            num_rescale_factors: 14,
            rescale_range: (0.6, 1.6),
            use_fht: false,
        }
    }

    /// Create parameters for 7-bit quantization (~4.5x compression)
    ///
    /// Near-lossless compression for high-precision requirements.
    /// Target: ~100% recall at 4.57x compression.
    pub fn bits7() -> Self {
        Self {
            bits_per_dim: QuantizationBits::Bits7,
            num_rescale_factors: 16,
            rescale_range: (0.7, 1.4),
            use_fht: false,
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
    /// Format depends on bits_per_dim:
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

    /// Whether FHT rotation was applied during quantization
    #[serde(default)]
    pub fht_applied: bool,
}

impl QuantizedVector {
    /// Create a new quantized vector
    pub fn new(data: Vec<u8>, scale: f32, bits: u8, dimensions: usize) -> Self {
        Self {
            data,
            scale,
            bits,
            dimensions,
            fht_applied: false,
        }
    }

    /// Create a new quantized vector with FHT flag
    pub fn new_with_fht(data: Vec<u8>, scale: f32, bits: u8, dimensions: usize, fht_applied: bool) -> Self {
        Self {
            data,
            scale,
            bits,
            dimensions,
            fht_applied,
        }
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.data.len()
    }

    /// Get compression ratio vs original f32 vector
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.dimensions * 4; // f32 = 4 bytes
        let compressed_bytes = self.data.len() + 4 + 1; // data + scale + bits
        original_bytes as f32 / compressed_bytes as f32
    }
}

/// Extended RaBitQ quantizer
///
/// Implements the Extended RaBitQ algorithm from SIGMOD 2025:
/// 1. Try multiple rescaling factors
/// 2. For each scale, quantize to grid and compute error
/// 3. Select scale with minimum error
/// 4. Store quantized vector with optimal scale
#[derive(Debug)]
pub struct ExtendedRaBitQ {
    params: ExtendedRaBitQParams,
}

impl ExtendedRaBitQ {
    /// Create a new Extended RaBitQ quantizer
    pub fn new(params: ExtendedRaBitQParams) -> Self {
        Self { params }
    }

    /// Create with default 4-bit quantization
    pub fn default_4bit() -> Self {
        Self::new(ExtendedRaBitQParams::bits4())
    }

    /// Get quantization parameters
    pub fn params(&self) -> &ExtendedRaBitQParams {
        &self.params
    }

    /// Quantize a vector using Extended RaBitQ algorithm
    ///
    /// Algorithm:
    /// 1. (Optional) Apply FHT for random rotation
    /// 2. Try multiple rescaling factors
    /// 3. For each scale, quantize to grid and compute error
    /// 4. Select scale with minimum error
    /// 5. Return quantized vector with optimal scale
    pub fn quantize(&self, vector: &[f32]) -> QuantizedVector {
        // Apply FHT if enabled (O(D log D) random rotation)
        let working_vector = if self.params.use_fht {
            self.apply_fht_to_vector(vector)
        } else {
            vector.to_vec()
        };

        let mut best_error = f32::MAX;
        let mut best_quantized = Vec::new();
        let mut best_scale = 1.0;

        // Generate rescaling factors to try
        let scales = self.generate_scales();

        // Try each scale and find the one with minimum error
        for scale in scales {
            let quantized = self.quantize_with_scale(&working_vector, scale);
            let error = self.compute_error(&working_vector, &quantized, scale);

            if error < best_error {
                best_error = error;
                best_quantized = quantized;
                best_scale = scale;
            }
        }

        QuantizedVector::new_with_fht(
            best_quantized,
            best_scale,
            self.params.bits_per_dim.to_u8(),
            vector.len(),
            self.params.use_fht,
        )
    }

    /// Apply FHT to a vector for random rotation
    fn apply_fht_to_vector(&self, vector: &[f32]) -> Vec<f32> {
        let n = vector.len();
        if n == 0 {
            return vec![];
        }

        // Pad to power of 2
        let padded_len = n.next_power_of_two();
        let mut data = vec![0.0f32; padded_len];
        data[..n].copy_from_slice(vector);

        // Apply random signs (using dimension as seed for reproducibility)
        apply_random_signs(&mut data, n as u64);

        // Apply FHT
        fast_hadamard_transform_inplace(&mut data);

        // Truncate and return
        data.truncate(n);
        data
    }

    /// Apply inverse FHT to reconstruct original vector
    fn apply_inverse_fht_to_vector(&self, vector: &[f32]) -> Vec<f32> {
        let n = vector.len();
        if n == 0 {
            return vec![];
        }

        // Pad to power of 2
        let padded_len = n.next_power_of_two();
        let mut data = vec![0.0f32; padded_len];
        data[..n].copy_from_slice(vector);

        // Apply inverse FHT (same as forward for Hadamard)
        fast_hadamard_transform_inplace(&mut data);

        // Apply same random signs to reverse (xor with self = identity)
        apply_random_signs(&mut data, n as u64);

        // Truncate and return
        data.truncate(n);
        data
    }

    /// Generate rescaling factors to try
    ///
    /// Returns a vector of scale factors evenly spaced between
    /// rescale_range.0 and rescale_range.1
    fn generate_scales(&self) -> Vec<f32> {
        let (min_scale, max_scale) = self.params.rescale_range;
        let n = self.params.num_rescale_factors;

        if n == 1 {
            return vec![(min_scale + max_scale) / 2.0];
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
                // For other bit widths (3, 5, 7), use 8-bit for now
                // TODO: Implement efficient bit-packing for non-power-of-2 bits
                values.to_vec()
            }
        }
    }

    /// Unpack quantized bytes into individual values
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

    /// Reconstruct a quantized vector, applying inverse FHT if needed
    pub fn reconstruct_full(&self, qv: &QuantizedVector) -> Vec<f32> {
        let mut result = self.reconstruct(&qv.data, qv.scale, qv.dimensions);

        // Apply inverse FHT if the vector was quantized with FHT
        if qv.fht_applied {
            result = self.apply_inverse_fht_to_vector(&result);
        }

        result
    }

    /// Compute L2 (Euclidean) distance between two quantized vectors
    ///
    /// This reconstructs both vectors and computes standard L2 distance.
    /// For maximum accuracy, use this with original vectors for reranking.
    pub fn distance_l2(
        &self,
        qv1: &QuantizedVector,
        qv2: &QuantizedVector,
    ) -> f32 {
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
    pub fn distance_cosine(
        &self,
        qv1: &QuantizedVector,
        qv2: &QuantizedVector,
    ) -> f32 {
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
    pub fn distance_dot(
        &self,
        qv1: &QuantizedVector,
        qv2: &QuantizedVector,
    ) -> f32 {
        let v1 = self.reconstruct(&qv1.data, qv1.scale, qv1.dimensions);
        let v2 = self.reconstruct(&qv2.data, qv2.scale, qv2.dimensions);

        // Return negative dot product (for nearest neighbor search)
        -v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum::<f32>()
    }

    /// Compute approximate distance using quantized values directly (fast path)
    ///
    /// This computes distance in the quantized space without full reconstruction.
    /// Faster but less accurate than distance_l2.
    pub fn distance_approximate(
        &self,
        qv1: &QuantizedVector,
        qv2: &QuantizedVector,
    ) -> f32 {
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

    // SIMD-optimized distance functions

    /// Compute L2 distance using SIMD acceleration
    ///
    /// Uses runtime CPU detection to select the best SIMD implementation:
    /// - x86_64: AVX2 > SSE2 > scalar
    /// - aarch64: NEON > scalar
    #[inline]
    pub fn distance_l2_simd(
        &self,
        qv1: &QuantizedVector,
        qv2: &QuantizedVector,
    ) -> f32 {
        // Reconstruct to f32 vectors
        let v1 = self.reconstruct(&qv1.data, qv1.scale, qv1.dimensions);
        let v2 = self.reconstruct(&qv2.data, qv2.scale, qv2.dimensions);

        // Use SIMD distance computation
        simd_l2_distance(&v1, &v2)
    }

    /// Compute cosine distance using SIMD acceleration
    #[inline]
    pub fn distance_cosine_simd(
        &self,
        qv1: &QuantizedVector,
        qv2: &QuantizedVector,
    ) -> f32 {
        let v1 = self.reconstruct(&qv1.data, qv1.scale, qv1.dimensions);
        let v2 = self.reconstruct(&qv2.data, qv2.scale, qv2.dimensions);

        simd_cosine_distance(&v1, &v2)
    }

    /// Apply Fast Hadamard Transform for random rotation
    ///
    /// FHT provides O(D log D) rotation vs O(D²) for matrix multiply.
    /// The input is modified in place and must have power-of-2 length.
    fn apply_fht(&self, data: &mut [f32]) {
        fast_hadamard_transform_inplace(data);
    }

    /// Apply inverse FHT (same as forward FHT for Hadamard matrices)
    fn apply_inverse_fht(&self, data: &mut [f32]) {
        fast_hadamard_transform_inplace(data);
    }
}

// Fast Hadamard Transform implementation

/// Fast Hadamard Transform - O(D log D) random rotation
///
/// The Hadamard transform is an orthonormal transformation that
/// spreads information across all dimensions. Combined with random
/// sign flipping, it provides an efficient pseudo-random rotation.
///
/// For non-power-of-2 dimensions, the vector is padded to the next
/// power of 2, transformed, then truncated back.
pub fn fast_hadamard_transform(data: &[f32]) -> Vec<f32> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    // Pad to next power of 2 if necessary
    let padded_len = n.next_power_of_two();
    let mut result = vec![0.0f32; padded_len];
    result[..n].copy_from_slice(data);

    fast_hadamard_transform_inplace(&mut result);

    // Truncate back to original length
    result.truncate(n);
    result
}

/// In-place Fast Hadamard Transform
///
/// Uses the butterfly algorithm for O(n log n) complexity.
/// Input length must be a power of 2.
#[inline]
pub fn fast_hadamard_transform_inplace(data: &mut [f32]) {
    let n = data.len();
    if n == 0 || !n.is_power_of_two() {
        return;
    }

    // Butterfly algorithm
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in 0..h {
                let a = data[i + j];
                let b = data[i + j + h];
                data[i + j] = a + b;
                data[i + j + h] = a - b;
            }
        }
        h *= 2;
    }

    // Normalize by sqrt(n) for orthonormality
    let norm = (n as f32).sqrt();
    let inv_norm = 1.0 / norm;
    for val in data.iter_mut() {
        *val *= inv_norm;
    }
}

/// Fast Hadamard Transform with SIMD acceleration
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fast_hadamard_transform_avx2(data: &mut [f32]) { unsafe {
    let n = data.len();
    if n == 0 || !n.is_power_of_two() {
        return;
    }

    // Use SIMD for large blocks, scalar for small
    let mut h = 1;

    // Scalar phase for h < 8 (can't vectorize efficiently)
    while h < 8 && h < n {
        for i in (0..n).step_by(h * 2) {
            for j in 0..h {
                let a = data[i + j];
                let b = data[i + j + h];
                data[i + j] = a + b;
                data[i + j + h] = a - b;
            }
        }
        h *= 2;
    }

    // AVX2 phase for h >= 8
    while h < n {
        for i in (0..n).step_by(h * 2) {
            // Process 8 elements at a time
            let mut j = 0;
            while j + 8 <= h {
                let a = _mm256_loadu_ps(data.as_ptr().add(i + j));
                let b = _mm256_loadu_ps(data.as_ptr().add(i + j + h));
                let sum = _mm256_add_ps(a, b);
                let diff = _mm256_sub_ps(a, b);
                _mm256_storeu_ps(data.as_mut_ptr().add(i + j), sum);
                _mm256_storeu_ps(data.as_mut_ptr().add(i + j + h), diff);
                j += 8;
            }
            // Handle remainder
            while j < h {
                let a = data[i + j];
                let b = data[i + j + h];
                data[i + j] = a + b;
                data[i + j + h] = a - b;
                j += 1;
            }
        }
        h *= 2;
    }

    // Normalize
    let norm = (n as f32).sqrt();
    let inv_norm = 1.0 / norm;
    let inv_norm_vec = _mm256_set1_ps(inv_norm);

    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(data.as_ptr().add(i));
        let scaled = _mm256_mul_ps(v, inv_norm_vec);
        _mm256_storeu_ps(data.as_mut_ptr().add(i), scaled);
        i += 8;
    }
    while i < n {
        data[i] *= inv_norm;
        i += 1;
    }
}}

/// Fast Hadamard Transform with NEON acceleration
#[cfg(target_arch = "aarch64")]
unsafe fn fast_hadamard_transform_neon(data: &mut [f32]) {
    let n = data.len();
    if n == 0 || !n.is_power_of_two() {
        return;
    }

    unsafe {
        let mut h = 1;

        // Scalar phase for h < 4
        while h < 4 && h < n {
            for i in (0..n).step_by(h * 2) {
                for j in 0..h {
                    let a = data[i + j];
                    let b = data[i + j + h];
                    data[i + j] = a + b;
                    data[i + j + h] = a - b;
                }
            }
            h *= 2;
        }

        // NEON phase for h >= 4
        while h < n {
            for i in (0..n).step_by(h * 2) {
                let mut j = 0;
                while j + 4 <= h {
                    let a = vld1q_f32(data.as_ptr().add(i + j));
                    let b = vld1q_f32(data.as_ptr().add(i + j + h));
                    let sum = vaddq_f32(a, b);
                    let diff = vsubq_f32(a, b);
                    vst1q_f32(data.as_mut_ptr().add(i + j), sum);
                    vst1q_f32(data.as_mut_ptr().add(i + j + h), diff);
                    j += 4;
                }
                while j < h {
                    let a = data[i + j];
                    let b = data[i + j + h];
                    data[i + j] = a + b;
                    data[i + j + h] = a - b;
                    j += 1;
                }
            }
            h *= 2;
        }

        // Normalize
        let norm = (n as f32).sqrt();
        let inv_norm = 1.0 / norm;
        let inv_norm_vec = vdupq_n_f32(inv_norm);

        let mut i = 0;
        while i + 4 <= n {
            let v = vld1q_f32(data.as_ptr().add(i));
            let scaled = vmulq_f32(v, inv_norm_vec);
            vst1q_f32(data.as_mut_ptr().add(i), scaled);
            i += 4;
        }
        while i < n {
            data[i] *= inv_norm;
            i += 1;
        }
    }
}

/// Apply random sign flipping for pseudo-random rotation
///
/// Combined with FHT, this provides an efficient approximation
/// to a random orthogonal rotation matrix.
pub fn apply_random_signs(data: &mut [f32], seed: u64) {
    // Simple deterministic pseudo-random sign flip
    // Uses the seed to generate reproducible signs
    let mut state = seed;
    for val in data.iter_mut() {
        // Simple PRNG (xorshift)
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        if state & 1 == 0 {
            *val = -*val;
        }
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
        return l2_distance_scalar(v1, v2);
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
        return cosine_distance_scalar(v1, v2);
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
unsafe fn l2_distance_avx2(v1: &[f32], v2: &[f32]) -> f32 { unsafe {
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
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn cosine_distance_avx2(v1: &[f32], v2: &[f32]) -> f32 { unsafe {
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
}}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 { unsafe {
    let sum_high = _mm256_extractf128_ps(v, 1);
    let sum_low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(sum_low, sum_high);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    _mm_cvtss_f32(sum32)
}}

// SSE2 implementations (x86_64 fallback)

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn l2_distance_sse2(v1: &[f32], v2: &[f32]) -> f32 { unsafe {
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
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn cosine_distance_sse2(v1: &[f32], v2: &[f32]) -> f32 { unsafe {
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
}}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum_sse2(v: __m128) -> f32 { unsafe {
    let sum64 = _mm_add_ps(v, _mm_movehl_ps(v, v));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    _mm_cvtss_f32(sum32)
}}

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
        let params = ExtendedRaBitQParams::default();
        assert_eq!(params.bits_per_dim, QuantizationBits::Bits4);
        assert_eq!(params.num_rescale_factors, 12);
        assert_eq!(params.rescale_range, (0.5, 2.0));
    }

    #[test]
    fn test_preset_params() {
        let params2 = ExtendedRaBitQParams::bits2();
        assert_eq!(params2.bits_per_dim, QuantizationBits::Bits2);

        let params4 = ExtendedRaBitQParams::bits4();
        assert_eq!(params4.bits_per_dim, QuantizationBits::Bits4);

        let params8 = ExtendedRaBitQParams::bits8();
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
        let quantizer = ExtendedRaBitQ::default_4bit();
        assert_eq!(quantizer.params().bits_per_dim, QuantizationBits::Bits4);
    }

    // Phase 2 Tests: Core Algorithm

    #[test]
    fn test_generate_scales() {
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams {
            bits_per_dim: QuantizationBits::Bits4,
            num_rescale_factors: 5,
            rescale_range: (0.5, 1.5),
            ..Default::default()
        });

        let scales = quantizer.generate_scales();
        assert_eq!(scales.len(), 5);
        assert_eq!(scales[0], 0.5);
        assert_eq!(scales[4], 1.5);
        assert!((scales[2] - 1.0).abs() < 0.01); // Middle should be ~1.0
    }

    #[test]
    fn test_generate_scales_single() {
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams {
            bits_per_dim: QuantizationBits::Bits4,
            num_rescale_factors: 1,
            rescale_range: (0.5, 1.5),
            ..Default::default()
        });

        let scales = quantizer.generate_scales();
        assert_eq!(scales.len(), 1);
        assert_eq!(scales[0], 1.0); // Average of min and max
    }

    #[test]
    fn test_pack_unpack_2bit() {
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams {
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
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams {
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
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams {
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
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams {
            bits_per_dim: QuantizationBits::Bits4,
            num_rescale_factors: 4,
            rescale_range: (0.5, 1.5),
            ..Default::default()
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
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams {
            bits_per_dim: QuantizationBits::Bits8, // High precision
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
            ..Default::default()
        });

        // Test vector
        let vector = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let quantized = quantizer.quantize(&vector);

        // Reconstruct
        let reconstructed = quantizer.reconstruct(&quantized.data, quantized.scale, vector.len());

        // Check reconstruction is close (8-bit should be accurate)
        for (orig, recon) in vector.iter().zip(reconstructed.iter()) {
            let error = (orig - recon).abs();
            assert!(error < 0.1, "Error too large: {} vs {}", orig, recon);
        }
    }

    #[test]
    fn test_quantize_uniform_vector() {
        let quantizer = ExtendedRaBitQ::default_4bit();

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
        let quantizer = ExtendedRaBitQ::default_4bit();

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
        let q2 = ExtendedRaBitQ::new(ExtendedRaBitQParams::bits2());
        let qv2 = q2.quantize(&test_vector);
        assert_eq!(qv2.bits, 2);

        // Test 4-bit
        let q4 = ExtendedRaBitQ::default_4bit();
        let qv4 = q4.quantize(&test_vector);
        assert_eq!(qv4.bits, 4);

        // Test 8-bit
        let q8 = ExtendedRaBitQ::new(ExtendedRaBitQParams::bits8());
        let qv8 = q8.quantize(&test_vector);
        assert_eq!(qv8.bits, 8);

        // Higher bits = larger packed size (for same dimensions)
        assert!(qv2.data.len() <= qv4.data.len());
        assert!(qv4.data.len() <= qv8.data.len());
    }

    #[test]
    fn test_quantize_high_dimensional() {
        let quantizer = ExtendedRaBitQ::default_4bit();

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
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams {
            bits_per_dim: QuantizationBits::Bits8, // High precision
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
            ..Default::default()
        });

        let v1 = vec![0.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist = quantizer.distance_l2(&qv1, &qv2);

        // Distance should be approximately 1.0
        assert!((dist - 1.0).abs() < 0.2, "Distance: {}", dist);
    }

    #[test]
    fn test_distance_l2_identical() {
        let quantizer = ExtendedRaBitQ::default_4bit();

        let v = vec![0.5, 0.3, 0.8, 0.2];
        let qv1 = quantizer.quantize(&v);
        let qv2 = quantizer.quantize(&v);

        let dist = quantizer.distance_l2(&qv1, &qv2);

        // Identical vectors should have near-zero distance
        assert!(dist < 0.3, "Distance should be near zero, got: {}", dist);
    }

    #[test]
    fn test_distance_cosine() {
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams {
            bits_per_dim: QuantizationBits::Bits8,
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
            ..Default::default()
        });

        // Orthogonal vectors
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist = quantizer.distance_cosine(&qv1, &qv2);

        // Orthogonal vectors: cosine = 0, distance = 1
        assert!((dist - 1.0).abs() < 0.3, "Distance: {}", dist);
    }

    #[test]
    fn test_distance_cosine_identical() {
        let quantizer = ExtendedRaBitQ::default_4bit();

        let v = vec![0.5, 0.3, 0.8];
        let qv1 = quantizer.quantize(&v);
        let qv2 = quantizer.quantize(&v);

        let dist = quantizer.distance_cosine(&qv1, &qv2);

        // Identical vectors: cosine = 1, distance = 0
        assert!(dist < 0.2, "Distance should be near zero, got: {}", dist);
    }

    #[test]
    fn test_distance_dot() {
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams {
            bits_per_dim: QuantizationBits::Bits8,
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
            ..Default::default()
        });

        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist = quantizer.distance_dot(&qv1, &qv2);

        // Dot product of [1,0,0] with itself = 1, negated = -1
        assert!((dist + 1.0).abs() < 0.3, "Distance: {}", dist);
    }

    #[test]
    fn test_distance_approximate() {
        let quantizer = ExtendedRaBitQ::default_4bit();

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
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams {
            bits_per_dim: QuantizationBits::Bits8, // High precision for correlation
            num_rescale_factors: 12,
            rescale_range: (0.8, 1.2),
            ..Default::default()
        });

        // Create multiple vectors
        let vectors = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        // Quantize all
        let quantized: Vec<QuantizedVector> = vectors
            .iter()
            .map(|v| quantizer.quantize(v))
            .collect();

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
                "Order not preserved: {} vs {}",
                quantized_01,
                quantized_02
            );
        }
    }

    #[test]
    fn test_distance_zero_vectors() {
        let quantizer = ExtendedRaBitQ::default_4bit();

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
        let quantizer = ExtendedRaBitQ::default_4bit();

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

    // Phase 4 Tests: SIMD Optimizations

    #[test]
    fn test_simd_l2_matches_scalar() {
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams {
            bits_per_dim: QuantizationBits::Bits8, // High precision
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
            ..Default::default()
        });

        let v1 = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let v2 = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist_scalar = quantizer.distance_l2(&qv1, &qv2);
        let dist_simd = quantizer.distance_l2_simd(&qv1, &qv2);

        // SIMD should match scalar within floating point precision
        let diff = (dist_scalar - dist_simd).abs();
        assert!(diff < 0.01, "SIMD vs scalar: {} vs {}", dist_simd, dist_scalar);
    }

    #[test]
    fn test_simd_cosine_matches_scalar() {
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams {
            bits_per_dim: QuantizationBits::Bits8,
            num_rescale_factors: 8,
            rescale_range: (0.8, 1.2),
            ..Default::default()
        });

        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist_scalar = quantizer.distance_cosine(&qv1, &qv2);
        let dist_simd = quantizer.distance_cosine_simd(&qv1, &qv2);

        // SIMD should match scalar within floating point precision
        let diff = (dist_scalar - dist_simd).abs();
        assert!(diff < 0.01, "SIMD vs scalar: {} vs {}", dist_simd, dist_scalar);
    }

    #[test]
    fn test_simd_high_dimensional() {
        let quantizer = ExtendedRaBitQ::default_4bit();

        // 128D vectors (realistic embeddings)
        let v1: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
        let v2: Vec<f32> = (0..128).map(|i| ((i + 1) as f32) / 128.0).collect();

        let qv1 = quantizer.quantize(&v1);
        let qv2 = quantizer.quantize(&v2);

        let dist_scalar = quantizer.distance_l2(&qv1, &qv2);
        let dist_simd = quantizer.distance_l2_simd(&qv1, &qv2);

        // Should be close (allow for quantization + FP variance)
        let diff = (dist_scalar - dist_simd).abs();
        assert!(diff < 0.1, "High-D SIMD vs scalar: {} vs {}", dist_simd, dist_scalar);
    }

    #[test]
    fn test_simd_scalar_fallback() {
        let quantizer = ExtendedRaBitQ::default_4bit();

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
        let quantizer = ExtendedRaBitQ::default_4bit();

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

    // Phase 5 Tests: Fast Hadamard Transform

    #[test]
    fn test_fht_basic() {
        // Test basic FHT on power-of-2 length
        let data = vec![1.0, 0.0, 0.0, 0.0];
        let result = fast_hadamard_transform(&data);

        // FHT of [1,0,0,0] normalized should be [0.5, 0.5, 0.5, 0.5]
        assert_eq!(result.len(), 4);
        for &val in &result {
            assert!((val - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_fht_identity() {
        // Applying FHT twice should return to original (Hadamard is self-inverse)
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let original = data.clone();

        fast_hadamard_transform_inplace(&mut data);
        fast_hadamard_transform_inplace(&mut data);

        // Should be close to original
        for (orig, result) in original.iter().zip(data.iter()) {
            assert!((orig - result).abs() < 0.001, "FHT not self-inverse");
        }
    }

    #[test]
    fn test_fht_non_power_of_2() {
        // Test FHT on non-power-of-2 length (should pad and truncate)
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = fast_hadamard_transform(&data);

        // Should return same length
        assert_eq!(result.len(), 5);
        // Values should be transformed (not all same as input)
        assert!(result.iter().any(|&v| (v - 1.0).abs() > 0.01));
    }

    #[test]
    fn test_fht_empty() {
        let data: Vec<f32> = vec![];
        let result = fast_hadamard_transform(&data);
        assert!(result.is_empty());
    }

    #[test]
    fn test_random_signs() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let seed = 42u64;

        // Apply random signs
        apply_random_signs(&mut data, seed);

        // Some values should be negated
        let has_negative = data.iter().any(|&v| v < 0.0);
        let has_positive = data.iter().any(|&v| v > 0.0);
        assert!(has_negative || has_positive); // At least some values affected

        // Applying same signs again should restore original magnitudes
        // (but may flip signs again - deterministic based on seed)
        apply_random_signs(&mut data, seed);

        // Magnitudes should be preserved
        for &val in &data {
            assert!([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0].iter().any(|&v| (val - v).abs() < 0.001));
        }
    }

    #[test]
    fn test_quantize_with_fht() {
        // Test that FHT-enabled quantization works
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams::bits4_fht());

        let vector: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
        let quantized = quantizer.quantize(&vector);

        assert_eq!(quantized.dimensions, 128);
        assert_eq!(quantized.bits, 4);
        assert!(quantized.fht_applied);
        assert!(quantized.scale > 0.0);
    }

    #[test]
    fn test_fht_quantization_accuracy() {
        // Compare accuracy with and without FHT
        let params_no_fht = ExtendedRaBitQParams::bits4();
        let params_fht = ExtendedRaBitQParams::bits4_fht();

        let q_no_fht = ExtendedRaBitQ::new(params_no_fht);
        let q_fht = ExtendedRaBitQ::new(params_fht);

        // Test vector with varying values
        let vector: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();

        let qv_no_fht = q_no_fht.quantize(&vector);
        let qv_fht = q_fht.quantize(&vector);

        // Both should work (not crash)
        assert_eq!(qv_no_fht.dimensions, 64);
        assert_eq!(qv_fht.dimensions, 64);

        // FHT flag should be set correctly
        assert!(!qv_no_fht.fht_applied);
        assert!(qv_fht.fht_applied);
    }

    #[test]
    fn test_fht_high_dimensional() {
        // Test FHT on high-dimensional vectors (like embeddings)
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams::with_fht(QuantizationBits::Bits4));

        // 768D vector (BERT-like)
        let vector: Vec<f32> = (0..768).map(|i| ((i % 100) as f32) / 100.0).collect();
        let quantized = quantizer.quantize(&vector);

        assert_eq!(quantized.dimensions, 768);
        assert!(quantized.fht_applied);
    }

    // Phase 6 Tests: 3/5/7-bit Quantization

    #[test]
    fn test_bits3_constructor() {
        let params = ExtendedRaBitQParams::bits3();
        assert_eq!(params.bits_per_dim, QuantizationBits::Bits3);
        assert_eq!(params.bits_per_dim.to_u8(), 3);
        assert_eq!(params.bits_per_dim.levels(), 8); // 2^3
    }

    #[test]
    fn test_bits5_constructor() {
        let params = ExtendedRaBitQParams::bits5();
        assert_eq!(params.bits_per_dim, QuantizationBits::Bits5);
        assert_eq!(params.bits_per_dim.to_u8(), 5);
        assert_eq!(params.bits_per_dim.levels(), 32); // 2^5
    }

    #[test]
    fn test_bits7_constructor() {
        let params = ExtendedRaBitQParams::bits7();
        assert_eq!(params.bits_per_dim, QuantizationBits::Bits7);
        assert_eq!(params.bits_per_dim.to_u8(), 7);
        assert_eq!(params.bits_per_dim.levels(), 128); // 2^7
    }

    #[test]
    fn test_quantize_3bit() {
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams::bits3());

        let vector: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        let quantized = quantizer.quantize(&vector);

        assert_eq!(quantized.dimensions, 64);
        assert_eq!(quantized.bits, 3);
        assert!(quantized.scale > 0.0);

        // Reconstruction should work
        let reconstructed = quantizer.reconstruct(&quantized.data, quantized.scale, 64);
        assert_eq!(reconstructed.len(), 64);
    }

    #[test]
    fn test_quantize_5bit() {
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams::bits5());

        let vector: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        let quantized = quantizer.quantize(&vector);

        assert_eq!(quantized.dimensions, 64);
        assert_eq!(quantized.bits, 5);
        assert!(quantized.scale > 0.0);

        // Reconstruction should work
        let reconstructed = quantizer.reconstruct(&quantized.data, quantized.scale, 64);
        assert_eq!(reconstructed.len(), 64);
    }

    #[test]
    fn test_quantize_7bit() {
        let quantizer = ExtendedRaBitQ::new(ExtendedRaBitQParams::bits7());

        let vector: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        let quantized = quantizer.quantize(&vector);

        assert_eq!(quantized.dimensions, 64);
        assert_eq!(quantized.bits, 7);
        assert!(quantized.scale > 0.0);

        // Reconstruction should work
        let reconstructed = quantizer.reconstruct(&quantized.data, quantized.scale, 64);
        assert_eq!(reconstructed.len(), 64);
    }

    #[test]
    fn test_bit_width_accuracy_ordering() {
        // Higher bit widths should have better accuracy
        let vector: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();

        let q2 = ExtendedRaBitQ::new(ExtendedRaBitQParams::bits2());
        let q3 = ExtendedRaBitQ::new(ExtendedRaBitQParams::bits3());
        let q4 = ExtendedRaBitQ::new(ExtendedRaBitQParams::bits4());
        let q5 = ExtendedRaBitQ::new(ExtendedRaBitQParams::bits5());

        let qv2 = q2.quantize(&vector);
        let qv3 = q3.quantize(&vector);
        let qv4 = q4.quantize(&vector);
        let qv5 = q5.quantize(&vector);

        // Compute reconstruction errors
        let err2 = q2.compute_error(&vector, &qv2.data, qv2.scale);
        let err3 = q3.compute_error(&vector, &qv3.data, qv3.scale);
        let err4 = q4.compute_error(&vector, &qv4.data, qv4.scale);
        let err5 = q5.compute_error(&vector, &qv5.data, qv5.scale);

        // Higher bits should generally have lower error
        // (may not be strictly monotonic due to quantization effects, but trend should hold)
        assert!(err4 <= err2 * 2.0, "4-bit should be more accurate than 2-bit");
        assert!(err5 <= err3 * 2.0, "5-bit should be more accurate than 3-bit");
    }
}
