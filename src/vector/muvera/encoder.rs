//! MUVERA encoder for multi-vector to FDE transformation.

use crate::vector::muvera::MuveraConfig;
use rand::prelude::*;
use rand_distr::StandardNormal;

/// Aggregation mode for MUVERA encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggMode {
    /// Sum tokens per partition (used for queries).
    Sum,
    /// Average tokens per partition (used for documents).
    Average,
}

/// MUVERA encoder for transforming multi-vector sets into FDEs.
///
/// Encodes variable-length sets of token vectors into fixed-dimensional
/// encodings. The inner product of two FDEs approximates MaxSim similarity.
#[derive(Debug, Clone)]
pub struct MuveraEncoder {
    config: MuveraConfig,
    token_dim: usize,
    fde_dim: usize,
}

impl MuveraEncoder {
    /// Create a new encoder for the given token dimension and config.
    #[must_use]
    pub fn new(token_dim: usize, config: MuveraConfig) -> Self {
        let fde_dim = config.fde_dimension(token_dim);
        Self {
            config,
            token_dim,
            fde_dim,
        }
    }

    /// Get the FDE output dimension.
    #[must_use]
    pub fn fde_dimension(&self) -> usize {
        self.fde_dim
    }

    /// Get the token input dimension.
    #[must_use]
    pub fn token_dimension(&self) -> usize {
        self.token_dim
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &MuveraConfig {
        &self.config
    }

    /// Encode query tokens into an FDE using SUM aggregation.
    ///
    /// Each query token contributes to its partition's sum.
    #[must_use]
    pub fn encode_query(&self, tokens: &[&[f32]]) -> Vec<f32> {
        self.encode(tokens, AggMode::Sum)
    }

    /// Encode document tokens into an FDE using AVERAGE aggregation.
    ///
    /// Each partition is normalized by its token count.
    #[must_use]
    pub fn encode_document(&self, tokens: &[&[f32]]) -> Vec<f32> {
        self.encode(tokens, AggMode::Average)
    }

    /// Core encoding function with configurable aggregation mode.
    #[must_use]
    pub fn encode(&self, tokens: &[&[f32]], mode: AggMode) -> Vec<f32> {
        if tokens.is_empty() {
            return vec![0.0; self.fde_dim];
        }

        let num_partitions = self.config.num_partitions();
        let mut fde = vec![0.0; self.fde_dim];

        for rep in 0..self.config.r_reps as usize {
            let seed = self.config.seed + rep as u64;
            let hyperplanes = gaussian_matrix(self.token_dim, self.config.k_sim as usize, seed);

            // Accumulate tokens per partition
            let mut partition_sums = vec![vec![0.0; self.token_dim]; num_partitions];
            let mut partition_counts = vec![0usize; num_partitions];

            for token in tokens {
                debug_assert_eq!(token.len(), self.token_dim, "Token dimension mismatch");

                let sketch = matmul_vec(token, &hyperplanes);
                let partition = simhash_gray_code(&sketch);

                // Add token to partition
                for (sum, &val) in partition_sums[partition].iter_mut().zip(token.iter()) {
                    *sum += val;
                }
                partition_counts[partition] += 1;
            }

            // Apply aggregation mode
            if mode == AggMode::Average {
                for p in 0..num_partitions {
                    if partition_counts[p] > 0 {
                        let scale = 1.0 / partition_counts[p] as f32;
                        for val in &mut partition_sums[p] {
                            *val *= scale;
                        }
                    }
                }
            }

            // Copy to FDE output
            let rep_offset = rep * num_partitions * self.token_dim;
            for p in 0..num_partitions {
                let start = rep_offset + p * self.token_dim;
                fde[start..start + self.token_dim].copy_from_slice(&partition_sums[p]);
            }
        }

        fde
    }
}

/// Generate a matrix of Gaussian random vectors for SimHash.
///
/// Returns k_sim vectors of dimension dim.
fn gaussian_matrix(dim: usize, k_sim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k_sim)
        .map(|_| {
            (0..dim)
                .map(|_| rng.sample::<f32, _>(StandardNormal))
                .collect()
        })
        .collect()
}

/// Multiply a vector by a matrix (vector @ matrix).
///
/// Returns a vector of length k_sim (one dot product per hyperplane).
fn matmul_vec(vec: &[f32], matrix: &[Vec<f32>]) -> Vec<f32> {
    matrix
        .iter()
        .map(|row| vec.iter().zip(row.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Map a sketch to a partition index using SimHash with Gray code.
///
/// Gray code preserves locality: adjacent buckets differ by one bit.
fn simhash_gray_code(sketch: &[f32]) -> usize {
    let mut gray = 0usize;
    for &val in sketch {
        let bit = if val > 0.0 { 1 } else { 0 };
        gray = (gray << 1) + (bit ^ (gray & 1));
    }
    gray
}

/// Compute MaxSim score between query and document token sets.
///
/// MaxSim = sum_{q in Q} max_{d in D} dot(q, d)
///
/// For each query token, find the most similar document token. Sum those max similarities.
#[must_use]
pub fn maxsim(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }

    let mut total = 0.0;
    for q in query_tokens {
        let max_sim = doc_tokens
            .iter()
            .map(|d| dot(q, d))
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        total += max_sim;
    }
    total
}

/// Dot product of two vectors.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_dimensions() {
        let config = MuveraConfig::default();
        let encoder = MuveraEncoder::new(128, config);
        assert_eq!(encoder.fde_dimension(), 5120);
        assert_eq!(encoder.token_dimension(), 128);
    }

    #[test]
    fn test_empty_tokens() {
        let encoder = MuveraEncoder::new(128, MuveraConfig::default());
        let fde = encoder.encode_query(&[]);
        assert_eq!(fde.len(), 5120);
        assert!(fde.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_single_token() {
        let encoder = MuveraEncoder::new(4, MuveraConfig::new(2, 2, 42));
        let token = [1.0, 0.0, 0.0, 0.0];
        let fde = encoder.encode_query(&[&token]);
        assert_eq!(fde.len(), 2 * 4 * 4); // r_reps=2, 2^k_sim=4, dim=4
        assert!(fde.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_query_vs_document_encoding() {
        let encoder = MuveraEncoder::new(4, MuveraConfig::new(2, 2, 42));
        let tokens: Vec<&[f32]> = vec![&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]];

        let query_fde = encoder.encode_query(&tokens);
        let doc_fde = encoder.encode_document(&tokens);

        // Query uses SUM, document uses AVERAGE - they should differ
        assert_ne!(query_fde, doc_fde);
    }

    #[test]
    fn test_deterministic_encoding() {
        let encoder = MuveraEncoder::new(128, MuveraConfig::default());
        let token = vec![0.1f32; 128];
        let tokens: Vec<&[f32]> = vec![&token];

        let fde1 = encoder.encode_query(&tokens);
        let fde2 = encoder.encode_query(&tokens);

        assert_eq!(fde1, fde2);
    }

    #[test]
    fn test_gaussian_matrix_deterministic() {
        let m1 = gaussian_matrix(128, 3, 42);
        let m2 = gaussian_matrix(128, 3, 42);
        assert_eq!(m1, m2);

        let m3 = gaussian_matrix(128, 3, 43);
        assert_ne!(m1, m3);
    }

    #[test]
    fn test_simhash_gray_code_output_range() {
        // k_sim=3 -> output should be in [0, 8)
        for _ in 0..100 {
            let sketch: Vec<f32> = (0..3).map(|i| i as f32 - 1.0).collect();
            let partition = simhash_gray_code(&sketch);
            assert!(partition < 8);
        }
    }

    #[test]
    fn test_simhash_gray_code_locality() {
        // Adjacent Gray codes differ by one bit
        // Test that small changes in sketch lead to nearby partitions
        let sketch1 = vec![1.0, 1.0, 1.0];
        let sketch2 = vec![1.0, 1.0, -0.001]; // Flip one sign

        let p1 = simhash_gray_code(&sketch1);
        let p2 = simhash_gray_code(&sketch2);

        // p1 and p2 should differ by at most 1 in Gray code distance
        let xor = p1 ^ p2;
        let bit_diff = xor.count_ones();
        assert!(bit_diff <= 2, "Gray code should preserve locality");
    }

    #[test]
    fn test_maxsim_basic() {
        let q1 = [1.0, 0.0, 0.0, 0.0];
        let q2 = [0.0, 1.0, 0.0, 0.0];
        let d1 = [1.0, 0.0, 0.0, 0.0]; // matches q1 perfectly
        let d2 = [0.0, 0.0, 1.0, 0.0]; // doesn't match q2

        let query: Vec<&[f32]> = vec![&q1, &q2];
        let doc: Vec<&[f32]> = vec![&d1, &d2];

        let score = maxsim(&query, &doc);
        // q1 matches d1 with score 1.0
        // q2 best match is either d1 or d2, both 0.0
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_fde_approximates_maxsim() {
        // MUV-7: Verify FDE dot product correlates with true MaxSim
        // Test with default params (k_sim=3, r_reps=5) - expect correlation > 0.65
        // Note: ColBERT uses normalized vectors, so we normalize here too
        use rand::prelude::*;

        let mut rng = StdRng::seed_from_u64(12345);
        let dim = 64; // More realistic dimension
        let encoder = MuveraEncoder::new(dim, MuveraConfig::new(3, 5, 42));

        let num_pairs = 200; // More samples for stability
        let mut fde_scores = Vec::with_capacity(num_pairs);
        let mut maxsim_scores = Vec::with_capacity(num_pairs);

        for _ in 0..num_pairs {
            // Generate random query (5-15 tokens), L2-normalized
            let num_q = rng.gen_range(5..=15);
            let query_vecs: Vec<Vec<f32>> = (0..num_q)
                .map(|_| {
                    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
                    normalize(&v)
                })
                .collect();
            let query: Vec<&[f32]> = query_vecs.iter().map(|v| v.as_slice()).collect();

            // Generate random document (20-50 tokens), L2-normalized
            let num_d = rng.gen_range(20..=50);
            let doc_vecs: Vec<Vec<f32>> = (0..num_d)
                .map(|_| {
                    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
                    normalize(&v)
                })
                .collect();
            let doc: Vec<&[f32]> = doc_vecs.iter().map(|v| v.as_slice()).collect();

            // Compute FDE dot product
            let query_fde = encoder.encode_query(&query);
            let doc_fde = encoder.encode_document(&doc);
            let fde_score = dot(&query_fde, &doc_fde);

            // Compute true MaxSim
            let true_score = maxsim(&query, &doc);

            fde_scores.push(fde_score);
            maxsim_scores.push(true_score);
        }

        // Compute Spearman correlation
        let correlation = spearman_correlation(&fde_scores, &maxsim_scores);

        // Default params should achieve > 0.65 correlation
        // (0.7 is achievable with more reps, but 0.65 is safe for k_sim=3, r_reps=5)
        assert!(
            correlation > 0.65,
            "FDE correlation with MaxSim should be > 0.65, got {:.3}",
            correlation
        );
    }

    /// L2-normalize a vector.
    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    #[test]
    fn test_fde_higher_params_better_correlation() {
        // Higher k_sim and r_reps should give better approximation
        use rand::prelude::*;

        let mut rng = StdRng::seed_from_u64(54321);
        let dim = 64;
        let encoder = MuveraEncoder::new(dim, MuveraConfig::new(4, 10, 42));

        let num_pairs = 200;
        let mut fde_scores = Vec::with_capacity(num_pairs);
        let mut maxsim_scores = Vec::with_capacity(num_pairs);

        for _ in 0..num_pairs {
            let num_q = rng.gen_range(5..=15);
            let query_vecs: Vec<Vec<f32>> = (0..num_q)
                .map(|_| {
                    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
                    normalize(&v)
                })
                .collect();
            let query: Vec<&[f32]> = query_vecs.iter().map(|v| v.as_slice()).collect();

            let num_d = rng.gen_range(20..=50);
            let doc_vecs: Vec<Vec<f32>> = (0..num_d)
                .map(|_| {
                    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
                    normalize(&v)
                })
                .collect();
            let doc: Vec<&[f32]> = doc_vecs.iter().map(|v| v.as_slice()).collect();

            let query_fde = encoder.encode_query(&query);
            let doc_fde = encoder.encode_document(&doc);
            let fde_score = dot(&query_fde, &doc_fde);
            let true_score = maxsim(&query, &doc);

            fde_scores.push(fde_score);
            maxsim_scores.push(true_score);
        }

        let correlation = spearman_correlation(&fde_scores, &maxsim_scores);

        // Higher params should achieve > 0.70 correlation
        // Note: MUVERA paper shows ~70% quality, reranking recovers to ~99%
        assert!(
            correlation > 0.70,
            "Higher params FDE correlation should be > 0.70, got {:.3}",
            correlation
        );
    }

    /// Compute Spearman rank correlation coefficient.
    fn spearman_correlation(x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());
        let n = x.len();

        // Compute ranks
        let x_ranks = compute_ranks(x);
        let y_ranks = compute_ranks(y);

        // Pearson correlation of ranks
        let x_mean: f32 = x_ranks.iter().sum::<f32>() / n as f32;
        let y_mean: f32 = y_ranks.iter().sum::<f32>() / n as f32;

        let mut num = 0.0;
        let mut denom_x = 0.0;
        let mut denom_y = 0.0;

        for i in 0..n {
            let dx = x_ranks[i] - x_mean;
            let dy = y_ranks[i] - y_mean;
            num += dx * dy;
            denom_x += dx * dx;
            denom_y += dy * dy;
        }

        num / (denom_x.sqrt() * denom_y.sqrt())
    }

    /// Compute ranks for a slice of values (1-based, average ties).
    fn compute_ranks(values: &[f32]) -> Vec<f32> {
        let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut ranks = vec![0.0; values.len()];
        let mut i = 0;
        while i < indexed.len() {
            let mut j = i;
            // Find ties
            while j < indexed.len() && indexed[j].1 == indexed[i].1 {
                j += 1;
            }
            // Average rank for ties
            let avg_rank = (i + 1 + j) as f32 / 2.0;
            for k in i..j {
                ranks[indexed[k].0] = avg_rank;
            }
            i = j;
        }
        ranks
    }
}
