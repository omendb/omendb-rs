//! RaBitQ: Randomly transformed Binary Quantization
//!
//! 1-bit quantization with FFHT rotation for 32x compression and unbiased
//! distance estimation. Uses asymmetric distance: data is 1-bit, query is
//! scalar-quantized to 6 bits, distance computed via binarized popcount.
//!
//! # Algorithm
//!
//! 1. **Training**: Compute centroid, generate random sign-flip vectors
//! 2. **Quantization**: residual → normalize → rotate (FFHT) → sign extraction
//! 3. **Query prep**: residual → normalize → rotate → 6-bit scalar quantize → binarize
//! 4. **Distance**: popcount across 6 bit planes + corrective factors
//!
//! # Reference
//!
//! Based on RaBitQ (arXiv:2405.12497) and VectorChord implementation.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Number of rotation rounds
const NUM_ROUNDS: usize = 4;

/// Minimum vectors for training
const MIN_TRAINING_VECTORS: usize = 256;

/// RaBitQ parameters (trained from data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitQParams {
    /// Vector dimensions
    pub dimensions: usize,
    /// Centroid (mean of training vectors)
    pub centroid: Vec<f32>,
    /// Random sign-flip bits: 4 rounds x ceil(D/64) u64s
    pub rotation_bits: [Vec<u64>; NUM_ROUNDS],
    /// Seed used for rotation (for reproducibility)
    seed: u64,
}

/// Precomputed query data for asymmetric distance
#[derive(Debug, Clone)]
pub struct RaBitQQueryPrep {
    /// ||q - centroid||^2
    pub dis_v_2: f32,
    /// ||q - centroid||
    pub dis_v: f32,
    /// Scalar quantization scale: range / 63
    pub k: f32,
    /// Scalar quantization offset: min value
    pub b: f32,
    /// Sum of quantized query values
    pub qvec_sum: f32,
    /// Binarized 6-bit query: 6 bit planes of ceil(D/64) u64s each
    pub binary_lut: [Vec<u64>; 6],
}

impl RaBitQParams {
    /// Train RaBitQ from sample vectors
    ///
    /// Computes centroid and generates deterministic rotation bits.
    pub fn train(vectors: &[&[f32]]) -> Result<Self, &'static str> {
        if vectors.is_empty() {
            return Err("Need at least one vector to train RaBitQ");
        }
        if vectors.len() < MIN_TRAINING_VECTORS {
            return Err("Need at least 256 vectors to train RaBitQ");
        }

        let dimensions = vectors[0].len();
        if dimensions == 0 {
            return Err("Vectors must have at least one dimension");
        }
        if !vectors.iter().all(|v| v.len() == dimensions) {
            return Err("All vectors must have same dimensions");
        }

        // Compute centroid
        let n = vectors.len() as f32;
        let mut centroid = vec![0.0f32; dimensions];
        for v in vectors {
            for (c, &x) in centroid.iter_mut().zip(v.iter()) {
                *c += x;
            }
        }
        for c in &mut centroid {
            *c /= n;
        }

        // Generate rotation bits deterministically from dimensions
        let seed = dimensions as u64 ^ 0x5A5A_5A5A_5A5A_5A5A;
        let padded_dim = dimensions.next_power_of_two();
        let num_words = padded_dim.div_ceil(64);

        let mut rng = StdRng::seed_from_u64(seed);
        let rotation_bits =
            std::array::from_fn(|_| (0..num_words).map(|_| rng.gen::<u64>()).collect());

        Ok(Self {
            dimensions,
            centroid,
            rotation_bits,
            seed,
        })
    }

    /// Number of u64 words per binary code
    #[inline]
    #[must_use]
    pub fn code_words(&self) -> usize {
        self.dimensions.div_ceil(64)
    }

    /// Bytes per vector for binary codes only (stored in colocated nodes)
    #[inline]
    #[must_use]
    pub fn code_bytes(&self) -> usize {
        self.code_words() * 8
    }

    /// Quantize a vector to 1-bit codes with metadata
    ///
    /// Returns (packed_signs, dis_u_2, factor_ip, factor_ppc, factor_err)
    #[must_use]
    pub fn quantize(&self, vector: &[f32]) -> (Vec<u64>, f32, f32, f32, f32) {
        debug_assert_eq!(vector.len(), self.dimensions);

        // Compute residual: v - centroid
        let mut residual: Vec<f32> = vector
            .iter()
            .zip(self.centroid.iter())
            .map(|(&v, &c)| v - c)
            .collect();

        let dis_u_2: f32 = residual.iter().map(|x| x * x).sum();
        let dis_u = dis_u_2.sqrt();

        // Handle zero-norm vectors
        if dis_u < 1e-10 {
            let codes = vec![0u64; self.code_words()];
            return (codes, 0.0, 0.0, 0.0, 0.0);
        }

        // Normalize residual
        for x in &mut residual {
            *x /= dis_u;
        }

        // Rotate
        self.rotate(&mut residual);

        // Extract signs and compute factors
        let num_words = self.code_words();
        let mut codes = vec![0u64; num_words];
        let mut sum_abs: f32 = 0.0;
        let mut cnt_pos: f32 = 0.0;
        let mut cnt_neg: f32 = 0.0;

        for (i, &val) in residual.iter().enumerate() {
            if val >= 0.0 {
                codes[i / 64] |= 1u64 << (i % 64);
                cnt_pos += 1.0;
            } else {
                cnt_neg += 1.0;
            }
            sum_abs += val.abs();
        }

        // x_0 = sum_of_abs / sqrt(D)
        let x_0 = sum_abs / (self.dimensions as f32).sqrt();

        // factor_ip: since ||o|| = 1 after normalization, factor_ip = 1/sum_abs
        let factor_ip = 1.0 / sum_abs;

        // factor_ppc: cnt_pos - cnt_neg (used in distance formula)
        let factor_ppc = cnt_pos - cnt_neg;

        // factor_err: error bound
        let factor_err = if self.dimensions > 1 && x_0 > 1e-10 {
            1.9 * ((1.0 / (x_0 * x_0) - 1.0) / (self.dimensions as f32 - 1.0)).sqrt()
        } else {
            0.0
        };

        (codes, dis_u_2, factor_ip, factor_ppc, factor_err)
    }

    /// Prepare query for asymmetric distance computation
    ///
    /// Rotates and scalar-quantizes query to 6 bits, then binarizes into
    /// 6 bit planes for fast popcount-based distance.
    #[must_use]
    pub fn prepare_query(&self, query: &[f32]) -> RaBitQQueryPrep {
        debug_assert_eq!(query.len(), self.dimensions);

        // Compute residual: q - centroid
        let mut residual: Vec<f32> = query
            .iter()
            .zip(self.centroid.iter())
            .map(|(&q, &c)| q - c)
            .collect();

        let dis_v_2: f32 = residual.iter().map(|x| x * x).sum();
        let dis_v = dis_v_2.sqrt();

        // Handle zero query
        if dis_v < 1e-10 {
            let num_words = self.code_words();
            return RaBitQQueryPrep {
                dis_v_2: 0.0,
                dis_v: 0.0,
                k: 0.0,
                b: 0.0,
                qvec_sum: 0.0,
                binary_lut: std::array::from_fn(|_| vec![0u64; num_words]),
            };
        }

        // Normalize
        for x in &mut residual {
            *x /= dis_v;
        }

        // Rotate
        self.rotate(&mut residual);

        // Scalar quantize to 6 bits (0-63)
        let (min_val, max_val) = residual
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &x| {
                (mn.min(x), mx.max(x))
            });

        let range = max_val - min_val;
        let k = if range > 1e-10 { range / 63.0 } else { 1.0 };
        let b = min_val;

        let quantized: Vec<u8> = residual
            .iter()
            .map(|&x| ((x - b) / k).round().clamp(0.0, 63.0) as u8)
            .collect();

        let qvec_sum: f32 = quantized.iter().map(|&x| x as f32).sum();

        // Binarize: split into 6 bit planes
        let num_words = self.code_words();
        let binary_lut: [Vec<u64>; 6] = std::array::from_fn(|bit| {
            let mut packed = vec![0u64; num_words];
            for (i, &v) in quantized.iter().enumerate() {
                if (v >> bit) & 1 == 1 {
                    packed[i / 64] |= 1u64 << (i % 64);
                }
            }
            packed
        });

        RaBitQQueryPrep {
            dis_v_2,
            dis_v,
            k,
            b,
            qvec_sum,
            binary_lut,
        }
    }

    /// Estimate L2 squared distance between query and quantized vector
    ///
    /// Uses asymmetric binarized popcount for the inner product estimation.
    #[inline]
    #[must_use]
    pub fn distance(
        prep: &RaBitQQueryPrep,
        signs: &[u64],
        dis_u_2: f32,
        factor_ip: f32,
        factor_ppc: f32,
    ) -> f32 {
        // Binary inner product across 6 bit planes
        let sum = binary_inner_product(signs, &prep.binary_lut);

        // Reconstruct inner product estimate
        // e = k * (2 * sum - qvec_sum) + b * factor_ppc
        let e = prep.k * (2.0 * sum as f32 - prep.qvec_sum) + prep.b * factor_ppc;

        // Apply corrective factor
        let ip_estimate = e * factor_ip;

        // L2 distance: ||u||^2 + ||v||^2 - 2 * ||u|| * ||v|| * ip_estimate
        // where ip_estimate approximates <normalized_u_rotated, normalized_v_rotated>
        let rough = dis_u_2 + prep.dis_v_2 - 2.0 * dis_u_2.sqrt() * prep.dis_v * ip_estimate;

        rough.max(0.0)
    }

    /// Batch compute distances
    #[inline]
    pub fn distance_batch(
        prep: &RaBitQQueryPrep,
        signs_batch: &[u64],
        metadata_batch: &[f32],
        code_words: usize,
        distances: &mut [f32],
    ) {
        for (i, dist) in distances.iter_mut().enumerate() {
            let code_start = i * code_words;
            let code_end = code_start + code_words;
            let meta_start = i * 4;

            let signs = &signs_batch[code_start..code_end];
            let dis_u_2 = metadata_batch[meta_start];
            let factor_ip = metadata_batch[meta_start + 1];
            let factor_ppc = metadata_batch[meta_start + 2];

            *dist = Self::distance(prep, signs, dis_u_2, factor_ip, factor_ppc);
        }
    }

    /// Serialize params to bytes
    pub fn serialize_params(&self) -> Vec<u8> {
        // Format: [dimensions:u32][seed:u64][centroid:f32*D][rotation_bits:4*num_words*u64]
        let num_words = self.rotation_bits[0].len();
        let mut buf = Vec::with_capacity(4 + 8 + self.dimensions * 4 + NUM_ROUNDS * num_words * 8);

        buf.extend_from_slice(&(self.dimensions as u32).to_le_bytes());
        buf.extend_from_slice(&self.seed.to_le_bytes());

        for &val in &self.centroid {
            buf.extend_from_slice(&val.to_le_bytes());
        }

        for round in &self.rotation_bits {
            for &word in round {
                buf.extend_from_slice(&word.to_le_bytes());
            }
        }

        buf
    }

    /// Deserialize params from bytes
    pub fn deserialize_params(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < 12 {
            return Err("RaBitQ data too short for header");
        }

        let dimensions = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let seed = u64::from_le_bytes(data[4..12].try_into().unwrap());

        let padded_dim = dimensions.next_power_of_two();
        let num_words = padded_dim.div_ceil(64);

        let expected_len = 12 + dimensions * 4 + NUM_ROUNDS * num_words * 8;
        if data.len() < expected_len {
            return Err("RaBitQ data too short for params");
        }

        let mut pos = 12;
        let mut centroid = vec![0.0f32; dimensions];
        for val in &mut centroid {
            *val = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
        }

        let rotation_bits: [Vec<u64>; NUM_ROUNDS] = std::array::from_fn(|_| {
            let mut words = Vec::with_capacity(num_words);
            for _ in 0..num_words {
                words.push(u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()));
                pos += 8;
            }
            words
        });

        Ok(Self {
            dimensions,
            centroid,
            rotation_bits,
            seed,
        })
    }

    // =========================================================================
    // FFHT rotation internals
    // =========================================================================

    /// Apply 4-round FFHT rotation in-place
    ///
    /// For non-power-of-2 dimensions: pad, rotate, Kac's walk, truncate.
    fn rotate(&self, vector: &mut [f32]) {
        let dim = vector.len();
        let padded_dim = dim.next_power_of_two();

        if dim == padded_dim {
            // Power-of-2: in-place
            let scale = 1.0 / (padded_dim as f32).sqrt();
            for round in 0..NUM_ROUNDS {
                flip_signs(&self.rotation_bits[round], vector);
                fht(vector);
                for x in vector.iter_mut() {
                    *x *= scale;
                }
            }
        } else {
            // Non-power-of-2: pad with zeros, rotate, Kac's walk, truncate
            let mut padded = vec![0.0f32; padded_dim];
            padded[..dim].copy_from_slice(vector);

            let base = dim.ilog2() as usize;
            let base_len = 1 << base;
            let scale = 1.0 / (base_len as f32).sqrt();

            for round in 0..NUM_ROUNDS {
                flip_signs(&self.rotation_bits[round], &mut padded);

                // Alternating FHT segments (VectorChord approach)
                if round % 2 == 0 {
                    fht(&mut padded[..base_len]);
                    for x in &mut padded[..base_len] {
                        *x *= scale;
                    }
                } else {
                    fht(&mut padded[padded_dim - base_len..]);
                    for x in &mut padded[padded_dim - base_len..] {
                        *x *= scale;
                    }
                }

                kacs_walk(&mut padded);
            }

            vector.copy_from_slice(&padded[..dim]);
        }
    }
}

/// Binary inner product via popcount across 6 bit planes
///
/// For each bit plane b (0..6), compute popcount(signs AND query_plane_b),
/// then weight by 2^b.
#[inline]
fn binary_inner_product(data_signs: &[u64], query_lut: &[Vec<u64>; 6]) -> u32 {
    let mut result = 0u32;

    for (bit_idx, query_plane) in query_lut.iter().enumerate() {
        let plane_sum: u32 = data_signs
            .iter()
            .zip(query_plane.iter())
            .map(|(&d, &q)| (d & q).count_ones())
            .sum();

        result += plane_sum << bit_idx;
    }

    result
}

/// Fast Hadamard Transform (in-place, iterative butterfly)
///
/// Requires power-of-2 length.
fn fht(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two(), "FHT requires power-of-2 length");

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
fn kacs_walk(x: &mut [f32]) {
    let n = x.len();
    let m = n / 2;
    let scale = 1.0 / 2.0_f32.sqrt();

    for i in 0..m {
        let l = x[i];
        let r = x[i + m];
        x[i] = (l + r) * scale;
        x[i + m] = (l - r) * scale;
    }
}

/// Flip signs according to random bits
#[inline]
fn flip_signs(bits: &[u64], x: &mut [f32]) {
    for (word_idx, chunk) in x.chunks_mut(64).enumerate() {
        if word_idx >= bits.len() {
            break;
        }
        let word = bits[word_idx];
        for (bit_idx, val) in chunk.iter_mut().enumerate() {
            if (word >> bit_idx) & 1 == 1 {
                *val = -*val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    #[test]
    fn test_fht_self_inverse() {
        // FHT(FHT(x)) = n * x
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
    fn test_quantize_basic() {
        let vectors = make_random_vectors(512, 128, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let params = RaBitQParams::train(&refs).unwrap();
        let (codes, dis_u_2, factor_ip, _factor_ppc, _factor_err) = params.quantize(&vectors[0]);

        assert_eq!(codes.len(), 2); // 128/64 = 2 u64 words
        assert!(dis_u_2 >= 0.0);
        assert!(factor_ip.is_finite());
    }

    #[test]
    fn test_asymmetric_distance_self() {
        let vectors = make_random_vectors(512, 128, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let params = RaBitQParams::train(&refs).unwrap();
        let prep = params.prepare_query(&vectors[0]);
        let (codes, dis_u_2, factor_ip, factor_ppc, _factor_err) = params.quantize(&vectors[0]);

        let self_dist = RaBitQParams::distance(&prep, &codes, dis_u_2, factor_ip, factor_ppc);

        // Self-distance should be near zero (some error from quantization)
        assert!(
            self_dist < 5.0,
            "Self-distance should be near zero, got {self_dist}"
        );
    }

    #[test]
    fn test_distance_ordering() {
        let vectors = make_random_vectors(512, 128, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let params = RaBitQParams::train(&refs).unwrap();
        let query = &vectors[0];
        let prep = params.prepare_query(query);

        // Compute RaBitQ distances and exact distances
        let mut rabitq_dists: Vec<(usize, f32)> = (0..512)
            .map(|i| {
                let (codes, dis_u_2, factor_ip, factor_ppc, _) = params.quantize(&vectors[i]);
                let d = RaBitQParams::distance(&prep, &codes, dis_u_2, factor_ip, factor_ppc);
                (i, d)
            })
            .collect();

        let mut exact_dists: Vec<(usize, f32)> = (0..512)
            .map(|i| {
                let dist: f32 = query
                    .iter()
                    .zip(vectors[i].iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                (i, dist)
            })
            .collect();

        rabitq_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // recall@10
        let exact_top10: std::collections::HashSet<usize> =
            exact_dists.iter().take(10).map(|(i, _)| *i).collect();
        let rabitq_top10: std::collections::HashSet<usize> =
            rabitq_dists.iter().take(10).map(|(i, _)| *i).collect();

        let recall = exact_top10.intersection(&rabitq_top10).count() as f32 / 10.0;
        assert!(
            recall >= 0.3,
            "RaBitQ recall@10 too low: {:.1}% (expected >= 30%)",
            recall * 100.0
        );
    }

    #[test]
    fn test_serialize_roundtrip() {
        let vectors = make_random_vectors(512, 128, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let params = RaBitQParams::train(&refs).unwrap();
        let bytes = params.serialize_params();
        let params2 = RaBitQParams::deserialize_params(&bytes).unwrap();

        assert_eq!(params.dimensions, params2.dimensions);
        assert_eq!(params.centroid.len(), params2.centroid.len());
        for (a, b) in params.centroid.iter().zip(params2.centroid.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
        for round in 0..NUM_ROUNDS {
            assert_eq!(
                params.rotation_bits[round].len(),
                params2.rotation_bits[round].len()
            );
            for (a, b) in params.rotation_bits[round]
                .iter()
                .zip(params2.rotation_bits[round].iter())
            {
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn test_768d_compression() {
        let vectors = make_random_vectors(512, 768, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let params = RaBitQParams::train(&refs).unwrap();

        // 768 dims = 12 u64 words = 96 bytes for binary codes
        assert_eq!(params.code_words(), 12);
        assert_eq!(params.code_bytes(), 96);

        let (codes, _, _, _, _) = params.quantize(&vectors[0]);
        assert_eq!(codes.len(), 12);

        // 96 bytes codes + 16 bytes metadata = 112 bytes vs 3072 bytes f32
        // 27.4x compression
        let ratio = (768 * 4) as f32 / (96 + 16) as f32;
        assert!(ratio > 25.0 && ratio < 30.0, "Compression ratio: {ratio}");
    }

    #[test]
    fn test_recall_at_10() {
        // Larger test: 2K vectors, 128D
        let vectors = make_random_vectors(2048, 128, 12345);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let params = RaBitQParams::train(&refs).unwrap();

        // Use 100 different queries and average recall
        let mut total_recall = 0.0;
        let num_queries = 100;

        for q_idx in 0..num_queries {
            let query = &vectors[q_idx];
            let prep = params.prepare_query(query);

            let mut rabitq_dists: Vec<(usize, f32)> = (0..2048)
                .map(|i| {
                    let (codes, dis_u_2, factor_ip, factor_ppc, _) = params.quantize(&vectors[i]);
                    let d = RaBitQParams::distance(&prep, &codes, dis_u_2, factor_ip, factor_ppc);
                    (i, d)
                })
                .collect();

            let mut exact_dists: Vec<(usize, f32)> = (0..2048)
                .map(|i| {
                    let dist: f32 = query
                        .iter()
                        .zip(vectors[i].iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (i, dist)
                })
                .collect();

            rabitq_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let exact_top10: std::collections::HashSet<usize> =
                exact_dists.iter().take(10).map(|(i, _)| *i).collect();
            let rabitq_top10: std::collections::HashSet<usize> =
                rabitq_dists.iter().take(10).map(|(i, _)| *i).collect();

            total_recall += exact_top10.intersection(&rabitq_top10).count() as f32 / 10.0;
        }

        let avg_recall = total_recall / num_queries as f32;
        // Raw 1-bit quantization on uniform random 128D vectors gives ~40% recall@10.
        // This is expected: RaBitQ achieves >95% recall with HNSW oversampling + rescore.
        assert!(
            avg_recall >= 0.3,
            "Average RaBitQ recall@10 too low: {:.1}% (expected >= 30%)",
            avg_recall * 100.0
        );
    }
}
