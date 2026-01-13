//! 1-bit quantization with corrective factors for unbiased distance estimation

use super::rotate::Rotator;
use serde::{Deserialize, Serialize};

/// Per-vector metadata for distance correction
///
/// These factors enable unbiased distance estimation from 1-bit codes.
/// Computed during quantization based on the vector's relationship to the centroid.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CodeMetadata {
    /// Additive offset: incorporates centroid distance
    pub f_add: f32,
    /// Multiplicative rescale factor
    pub f_rescale: f32,
    /// Error bound factor for confidence intervals
    pub f_error: f32,
}

impl Default for CodeMetadata {
    fn default() -> Self {
        Self {
            f_add: 0.0,
            f_rescale: 1.0,
            f_error: 0.0,
        }
    }
}

/// RaBitQ index with per-index rotation and centroid
///
/// Trained from sample vectors to compute:
/// - Centroid (mean vector)
/// - Random rotation matrix (reproducible from seed)
/// - Pre-rotated centroid (cached for performance)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitQIndex {
    /// Vector dimension
    dim: usize,
    /// Random rotator (stored via seed for reproducibility)
    rotator: Rotator,
    /// Centroid (mean of training vectors)
    centroid: Vec<f32>,
    /// Pre-rotated centroid (cached to avoid recomputation per quantize)
    rotated_centroid: Vec<f32>,
    /// Seed used to create rotator (for serialization)
    seed: u64,
}

impl RaBitQIndex {
    /// Train RaBitQ index from sample vectors
    ///
    /// Computes the centroid and generates a random rotation matrix.
    ///
    /// # Arguments
    /// * `vectors` - Sample vectors (should be representative of the dataset)
    /// * `seed` - Random seed for reproducible rotation
    ///
    /// # Errors
    /// Returns error if vectors is empty or has inconsistent dimensions.
    pub fn train(vectors: &[&[f32]], seed: u64) -> Result<Self, &'static str> {
        if vectors.is_empty() {
            return Err("Need at least one vector to train");
        }

        let dim = vectors[0].len();
        if dim == 0 {
            return Err("Vectors must have positive dimension");
        }
        if !vectors.iter().all(|v| v.len() == dim) {
            return Err("All vectors must have same dimensions");
        }

        // Compute centroid (mean)
        let n = vectors.len() as f32;
        let mut centroid = vec![0.0f32; dim];
        for v in vectors {
            for (c, &x) in centroid.iter_mut().zip(v.iter()) {
                *c += x;
            }
        }
        for c in &mut centroid {
            *c /= n;
        }

        let rotator = Rotator::new(dim, seed);

        // Pre-compute rotated centroid (cached for quantize performance)
        let mut rotated_centroid = centroid.clone();
        rotator.rotate(&mut rotated_centroid);

        Ok(Self {
            dim,
            rotator,
            centroid,
            rotated_centroid,
            seed,
        })
    }

    /// Create index with known centroid (for deserialization or testing)
    ///
    /// # Panics
    /// Panics if centroid is empty.
    #[must_use]
    pub fn with_centroid(centroid: Vec<f32>, seed: u64) -> Self {
        assert!(!centroid.is_empty(), "Centroid cannot be empty");
        let dim = centroid.len();
        let rotator = Rotator::new(dim, seed);

        // Pre-compute rotated centroid
        let mut rotated_centroid = centroid.clone();
        rotator.rotate(&mut rotated_centroid);

        Self {
            dim,
            rotator,
            centroid,
            rotated_centroid,
            seed,
        }
    }

    /// Get vector dimension
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get centroid reference
    #[inline]
    #[must_use]
    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }

    /// Get rotator reference
    #[inline]
    #[must_use]
    pub fn rotator(&self) -> &Rotator {
        &self.rotator
    }

    /// Get seed
    #[inline]
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Number of u64 words needed for binary codes
    #[inline]
    #[must_use]
    pub fn code_words(&self) -> usize {
        self.dim.div_ceil(64)
    }

    /// Bytes per quantized vector (codes + metadata)
    #[must_use]
    pub fn bytes_per_vector(&self) -> usize {
        // ceil(dim/64) * 8 bytes for bits + 12 bytes for metadata (3 x f32)
        self.code_words() * 8 + 12
    }

    /// Quantize a vector to 1-bit codes with metadata
    ///
    /// # Process
    /// 1. Compute residual: x - centroid
    /// 2. Rotate residual via FFHT
    /// 3. Extract signs as binary code
    /// 4. Compute corrective factors
    ///
    /// # Returns
    /// (binary_codes, metadata) where binary_codes is packed u64s
    #[must_use]
    pub fn quantize(&self, vector: &[f32]) -> (Vec<u64>, CodeMetadata) {
        assert_eq!(
            vector.len(),
            self.dim,
            "Vector dimension {} must match index dimension {}",
            vector.len(),
            self.dim
        );

        // Step 1: Compute residual (x - centroid)
        let mut residual: Vec<f32> = vector
            .iter()
            .zip(self.centroid.iter())
            .map(|(&v, &c)| v - c)
            .collect();

        // Compute L2 norm squared of residual before rotation
        let l2_sqr: f32 = residual.iter().map(|x| x * x).sum();
        let l2_norm = l2_sqr.sqrt();

        // Step 2: Rotate residual
        self.rotator.rotate(&mut residual);

        // Step 3: Extract signs as binary code
        let num_words = self.code_words();
        let mut codes = vec![0u64; num_words];

        // Also compute sum of absolute values for correction factors
        let mut sum_abs: f32 = 0.0;

        for (i, &val) in residual.iter().enumerate() {
            let word = i / 64;
            let bit = i % 64;

            if val > 0.0 {
                codes[word] |= 1u64 << bit;
            }
            sum_abs += val.abs();
        }

        // Step 4: Compute corrective factors
        // Based on RaBitQ paper (arXiv:2405.12497)
        //
        // The distance formula is:
        //   ||o_r - q_r||^2 = sqr_x + sqr_y - 2 * dist_o * dist_q * <q, o>_est
        //
        // Where <q, o>_est = (binary_ip / D) / x_0
        // And x_0 = sum_abs / (dist_o * sqrt(D))
        //
        // Simplifying:
        //   dist = sqr_x + sqr_y + f_rescale * g_rescale * binary_ip
        // Where:
        //   f_add = sqr_x = ||data - centroid||^2
        //   f_rescale = -2 * sqr_x / sum_abs (data-side coefficient)
        //   g_rescale = sqrt(sqr_y / D) (query-side, computed at query time)

        // f_add: simply ||data - centroid||^2
        let f_add = l2_sqr;

        // f_rescale: multiplicative correction (data-side)
        // f_rescale = -2 * ||data - centroid||^2 / sum_abs
        let f_rescale = if sum_abs.abs() > 1e-10 {
            -2.0 * l2_sqr / sum_abs
        } else {
            0.0
        };

        // f_error: error bound factor
        // Based on concentration inequality from the paper
        // xu_cb_norm_sqr = ||binary_code||^2 where each component is ±1
        // This equals dim since each squared component is 1
        let xu_cb_norm_sqr = self.dim as f32;
        let variance_term = if sum_abs.abs() > 1e-10 {
            (l2_sqr * xu_cb_norm_sqr) / (sum_abs * sum_abs) - 1.0
        } else {
            0.0
        };
        let f_error = if self.dim > 1 && variance_term > 0.0 {
            2.0 * l2_norm * 1.9 * (variance_term / (self.dim - 1) as f32).sqrt()
        } else {
            0.0
        };

        let metadata = CodeMetadata {
            f_add,
            f_rescale,
            f_error,
        };

        (codes, metadata)
    }

    /// Quantize vector and pack into bytes for storage
    #[must_use]
    pub fn quantize_to_bytes(&self, vector: &[f32]) -> Vec<u8> {
        let (codes, metadata) = self.quantize(vector);

        let mut bytes = Vec::with_capacity(self.bytes_per_vector());

        // Pack binary codes
        for code in codes {
            bytes.extend_from_slice(&code.to_le_bytes());
        }

        // Pack metadata
        bytes.extend_from_slice(&metadata.f_add.to_le_bytes());
        bytes.extend_from_slice(&metadata.f_rescale.to_le_bytes());
        bytes.extend_from_slice(&metadata.f_error.to_le_bytes());

        bytes
    }

    /// Unpack binary codes and metadata from bytes
    ///
    /// # Panics
    /// Panics if bytes is too short (must be at least `bytes_per_vector()` bytes).
    #[must_use]
    pub fn unpack_bytes(&self, bytes: &[u8]) -> (Vec<u64>, CodeMetadata) {
        let expected_len = self.bytes_per_vector();
        assert!(
            bytes.len() >= expected_len,
            "Input bytes too short: got {} bytes, need at least {}",
            bytes.len(),
            expected_len
        );

        let num_words = self.code_words();
        let code_bytes = num_words * 8;

        let mut codes = Vec::with_capacity(num_words);
        for i in 0..num_words {
            let offset = i * 8;
            let word = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            codes.push(word);
        }

        let f_add = f32::from_le_bytes(bytes[code_bytes..code_bytes + 4].try_into().unwrap());
        let f_rescale =
            f32::from_le_bytes(bytes[code_bytes + 4..code_bytes + 8].try_into().unwrap());
        let f_error =
            f32::from_le_bytes(bytes[code_bytes + 8..code_bytes + 12].try_into().unwrap());

        let metadata = CodeMetadata {
            f_add,
            f_rescale,
            f_error,
        };

        (codes, metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_train_basic() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let index = RaBitQIndex::train(&refs, 42).unwrap();

        // Centroid should be [0.25, 0.25, 0.25, 0.25]
        assert_eq!(index.dim(), 4);
        for &c in index.centroid() {
            assert!((c - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_quantize_produces_bits() {
        let dim = 64;
        let mut rng = StdRng::seed_from_u64(123);
        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let index = RaBitQIndex::train(&refs, 999).unwrap();

        let (codes, metadata) = index.quantize(&vectors[0]);

        // Should have exactly 1 u64 for 64 dimensions
        assert_eq!(codes.len(), 1);
        // Metadata should have reasonable values
        assert!(metadata.f_add.is_finite());
        assert!(metadata.f_rescale.is_finite());
        assert!(metadata.f_error.is_finite());
    }

    #[test]
    fn test_quantize_768d() {
        let dim = 768;
        let mut rng = StdRng::seed_from_u64(456);
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / norm).collect()
            })
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let index = RaBitQIndex::train(&refs, 777).unwrap();

        // 768 dims = 12 u64 words
        assert_eq!(index.code_words(), 12);
        assert_eq!(index.bytes_per_vector(), 12 * 8 + 12); // 96 + 12 = 108 bytes

        let (codes, _metadata) = index.quantize(&vectors[0]);
        assert_eq!(codes.len(), 12);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let dim = 128;
        let mut rng = StdRng::seed_from_u64(789);
        let vectors: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let index = RaBitQIndex::train(&refs, 111).unwrap();

        let (original_codes, original_meta) = index.quantize(&vectors[5]);
        let bytes = index.quantize_to_bytes(&vectors[5]);
        let (unpacked_codes, unpacked_meta) = index.unpack_bytes(&bytes);

        assert_eq!(original_codes, unpacked_codes);
        assert!((original_meta.f_add - unpacked_meta.f_add).abs() < 1e-6);
        assert!((original_meta.f_rescale - unpacked_meta.f_rescale).abs() < 1e-6);
        assert!((original_meta.f_error - unpacked_meta.f_error).abs() < 1e-6);
    }

    #[test]
    fn test_compression_ratio() {
        let dim = 768;
        let original_bytes = dim * 4; // f32 = 4 bytes
        let mut rng = StdRng::seed_from_u64(999);
        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let index = RaBitQIndex::train(&refs, 888).unwrap();
        let quantized_bytes = index.bytes_per_vector();

        let ratio = original_bytes as f32 / quantized_bytes as f32;
        // Should be close to 28x (3072 / 108)
        assert!(
            ratio > 25.0 && ratio < 35.0,
            "Compression ratio {} not in expected range",
            ratio
        );
    }
}
