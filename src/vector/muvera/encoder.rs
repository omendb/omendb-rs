//! MUVERA encoder for multi-vector to FDE transformation.

use crate::distance::dot_product as dot;
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
///
/// When d_proj is configured, tokens are projected to a lower dimension before
/// encoding, reducing FDE size (e.g., 16,384D → 2,048D with d_proj=16).
///
/// Hyperplanes and projection matrix are pre-computed and cached.
#[derive(Debug, Clone)]
pub struct MuveraEncoder {
    config: MuveraConfig,
    token_dim: usize,
    proj_dim: usize,
    fde_dim: usize,
    /// Cached hyperplanes: [repetition][hyperplane][proj_dim]
    hyperplanes: Vec<Vec<Vec<f32>>>,
    /// Projection matrix: [d_proj][token_dim], or None if d_proj not set
    projection: Option<Vec<Vec<f32>>>,
}

impl MuveraEncoder {
    /// Create a new encoder for the given token dimension and config.
    ///
    /// # Panics
    ///
    /// Panics if d_proj exceeds token_dim.
    #[must_use]
    pub fn new(token_dim: usize, config: MuveraConfig) -> Self {
        let proj_dim = config.proj_dim(token_dim);

        // Validate d_proj <= token_dim
        if let Some(d) = config.d_proj {
            assert!(
                (d as usize) <= token_dim,
                "d_proj ({d}) cannot exceed token_dim ({token_dim})"
            );
        }

        let fde_dim = config.encoded_dimension(token_dim);

        // Generate projection matrix if d_proj is set
        // Use seed offset to avoid correlation with hyperplanes
        let projection = config.d_proj.map(|d| {
            let proj_seed = config.seed.wrapping_add(1_000_000);
            projection_matrix(token_dim, d as usize, proj_seed)
        });

        // Hyperplanes operate on projected dimension
        let hyperplanes = (0..config.repetitions as usize)
            .map(|rep| {
                let seed = config.seed + rep as u64;
                gaussian_matrix(proj_dim, config.partition_bits as usize, seed)
            })
            .collect();

        Self {
            config,
            token_dim,
            proj_dim,
            fde_dim,
            hyperplanes,
            projection,
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

    /// Get the projection dimension (d_proj or token_dim).
    #[must_use]
    pub fn proj_dimension(&self) -> usize {
        self.proj_dim
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

        let num_partitions = self.config.partitions();
        let mut fde = vec![0.0; self.fde_dim];

        for (rep, hyperplanes) in self.hyperplanes.iter().enumerate() {
            // Accumulate tokens per partition (using proj_dim)
            let mut partition_sums = vec![vec![0.0; self.proj_dim]; num_partitions];
            let mut partition_counts = vec![0usize; num_partitions];

            for token in tokens {
                debug_assert_eq!(token.len(), self.token_dim, "Token dimension mismatch");

                // When projection is enabled, we allocate for projected values
                // When disabled, we use the token slice directly (no allocation)
                let partition = if let Some(proj) = &self.projection {
                    // Project token, compute sketch, and accumulate projected values
                    let projected: Vec<f32> = proj.iter().map(|row| dot(token, row)).collect();
                    let sketch = matmul_vec(&projected, hyperplanes);
                    let p = simhash_gray_code(&sketch);
                    for (sum, val) in partition_sums[p].iter_mut().zip(projected.iter()) {
                        *sum += val;
                    }
                    p
                } else {
                    // No projection: use token directly for sketch and accumulation
                    let sketch = matmul_vec(token, hyperplanes);
                    let p = simhash_gray_code(&sketch);
                    for (sum, &val) in partition_sums[p].iter_mut().zip(token.iter()) {
                        *sum += val;
                    }
                    p
                };
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

            // Copy to FDE output (using proj_dim)
            let rep_offset = rep * num_partitions * self.proj_dim;
            for (p, partition_sum) in partition_sums.iter().enumerate().take(num_partitions) {
                let start = rep_offset + p * self.proj_dim;
                fde[start..start + self.proj_dim].copy_from_slice(partition_sum);
            }
        }

        fde
    }
}

/// Generate projection matrix: d_proj rows × token_dim cols.
///
/// Each row is a Gaussian random vector scaled by 1/sqrt(d_proj) for variance preservation.
fn projection_matrix(token_dim: usize, d_proj: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let scale = 1.0 / (d_proj as f32).sqrt();
    (0..d_proj)
        .map(|_| {
            (0..token_dim)
                .map(|_| rng.sample::<f32, _>(StandardNormal) * scale)
                .collect()
        })
        .collect()
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
    matrix.iter().map(|row| dot(vec, row)).collect()
}

/// Map a sketch to a partition index using SimHash with Gray code.
///
/// Gray code preserves locality: adjacent buckets differ by one bit.
fn simhash_gray_code(sketch: &[f32]) -> usize {
    let mut gray = 0usize;
    for &val in sketch {
        let bit = usize::from(val > 0.0);
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
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        total += max_sim;
    }
    total
}

/// Compute MaxSim scores for multiple documents in batch (optimized).
///
/// Uses matrix multiply pattern: Q @ D.T → row-wise max → sum.
/// This is faster than calling `maxsim()` repeatedly when reranking many candidates.
///
/// # Arguments
///
/// * `query_tokens` - Query token embeddings (borrowed)
/// * `doc_tokens_list` - List of document token sets (each document is a Vec of borrowed slices)
///
/// # Returns
///
/// MaxSim score for each document in the same order as `doc_tokens_list`.
#[must_use]
pub fn maxsim_batch<'a, T: AsRef<[&'a [f32]]>>(
    query_tokens: &[&[f32]],
    doc_tokens_list: &[T],
) -> Vec<f32> {
    if query_tokens.is_empty() {
        return vec![0.0; doc_tokens_list.len()];
    }

    doc_tokens_list
        .iter()
        .map(|doc_tokens| {
            let doc_tokens = doc_tokens.as_ref();
            if doc_tokens.is_empty() {
                return 0.0;
            }
            // For each query token, find max similarity across all doc tokens
            // This is the matrix multiply pattern: Q @ D.T, then row-wise max
            let mut total = 0.0f32;
            for q in query_tokens {
                let max_sim = doc_tokens
                    .iter()
                    .map(|d: &&[f32]| dot(q, d))
                    .max_by(|a: &f32, b: &f32| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0.0);
                total += max_sim;
            }
            total
        })
        .collect()
}

/// Compute MaxSim scores for multiple documents in batch using parallel iteration.
///
/// Same as `maxsim_batch` but uses rayon for parallel computation.
/// Use when reranking many candidates (>100).
#[must_use]
pub fn maxsim_batch_par<'a, T: AsRef<[&'a [f32]]> + Sync>(
    query_tokens: &[&[f32]],
    doc_tokens_list: &[T],
) -> Vec<f32> {
    use rayon::prelude::*;

    if query_tokens.is_empty() {
        return vec![0.0; doc_tokens_list.len()];
    }

    doc_tokens_list
        .par_iter()
        .map(|doc_tokens| {
            let doc_tokens = doc_tokens.as_ref();
            if doc_tokens.is_empty() {
                return 0.0;
            }
            let mut total = 0.0f32;
            for q in query_tokens {
                let max_sim = doc_tokens
                    .iter()
                    .map(|d: &&[f32]| dot(q, d))
                    .max_by(|a: &f32, b: &f32| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0.0);
                total += max_sim;
            }
            total
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_dimensions_with_dproj() {
        let config = MuveraConfig::default(); // d_proj=Some(16)
        let encoder = MuveraEncoder::new(128, config);
        // 8 reps * 16 partitions * 16 (d_proj) = 2,048
        assert_eq!(encoder.fde_dimension(), 2048);
        assert_eq!(encoder.token_dimension(), 128);
        assert_eq!(encoder.proj_dimension(), 16);
    }

    #[test]
    fn test_encoder_dimensions_no_dproj() {
        let config = MuveraConfig {
            d_proj: None,
            ..Default::default()
        };
        let encoder = MuveraEncoder::new(128, config);
        // 8 reps * 16 partitions * 128 (token_dim) = 16,384
        assert_eq!(encoder.fde_dimension(), 16384);
        assert_eq!(encoder.token_dimension(), 128);
        assert_eq!(encoder.proj_dimension(), 128);
    }

    #[test]
    fn test_empty_tokens() {
        let encoder = MuveraEncoder::new(128, MuveraConfig::default());
        let fde = encoder.encode_query(&[]);
        assert_eq!(fde.len(), 2048); // With d_proj=16
        assert!(fde.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_single_token() {
        // Use d_proj=None since token_dim=4 < default d_proj=16
        let encoder = MuveraEncoder::new(
            4,
            MuveraConfig {
                repetitions: 2,
                partition_bits: 2,
                d_proj: None,
                seed: 42,
                pool_factor: None,
            },
        );
        let token = [1.0, 0.0, 0.0, 0.0];
        let fde = encoder.encode_query(&[&token]);
        assert_eq!(fde.len(), 2 * 4 * 4); // r_reps=2, 2^k_sim=4, dim=4
        assert!(fde.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_query_vs_document_encoding() {
        // Use d_proj=None since token_dim=4 < default d_proj=16
        let encoder = MuveraEncoder::new(
            4,
            MuveraConfig {
                repetitions: 2,
                partition_bits: 2,
                d_proj: None,
                seed: 42,
                pool_factor: None,
            },
        );
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
    #[should_panic(expected = "d_proj")]
    fn test_dproj_exceeds_token_dim() {
        let config = MuveraConfig {
            d_proj: Some(200),
            ..Default::default()
        };
        let _encoder = MuveraEncoder::new(128, config); // Should panic
    }

    #[test]
    fn test_projection_matrix_deterministic() {
        let m1 = projection_matrix(128, 16, 42);
        let m2 = projection_matrix(128, 16, 42);
        assert_eq!(m1, m2);

        let m3 = projection_matrix(128, 16, 43);
        assert_ne!(m1, m3);

        // Verify dimensions
        assert_eq!(m1.len(), 16); // d_proj rows
        assert_eq!(m1[0].len(), 128); // token_dim cols
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
        let encoder = MuveraEncoder::new(dim, MuveraConfig::default());

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
        let encoder = MuveraEncoder::new(
            dim,
            MuveraConfig {
                repetitions: 10,
                partition_bits: 4,
                d_proj: Some(16),
                seed: 42,
                pool_factor: None,
            },
        );

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

    #[test]
    fn test_maxsim_batch_matches_individual() {
        // maxsim_batch should produce same results as calling maxsim repeatedly
        let q1 = [1.0, 0.0, 0.0, 0.0];
        let q2 = [0.0, 1.0, 0.0, 0.0];
        let query: Vec<&[f32]> = vec![&q1, &q2];

        let d1_t1 = [1.0, 0.0, 0.0, 0.0];
        let d1_t2 = [0.0, 0.0, 1.0, 0.0];
        let doc1: Vec<&[f32]> = vec![&d1_t1, &d1_t2];

        let d2_t1 = [0.0, 1.0, 0.0, 0.0];
        let d2_t2 = [0.0, 0.0, 0.0, 1.0];
        let doc2: Vec<&[f32]> = vec![&d2_t1, &d2_t2];

        let d3_t1 = [0.5, 0.5, 0.0, 0.0];
        let doc3: Vec<&[f32]> = vec![&d3_t1];

        let docs = vec![doc1.clone(), doc2.clone(), doc3.clone()];

        // Batch computation
        let batch_scores = super::maxsim_batch(&query, &docs);

        // Individual computation
        let individual_scores: Vec<f32> =
            docs.iter().map(|doc| super::maxsim(&query, doc)).collect();

        assert_eq!(batch_scores.len(), 3);
        for (batch, individual) in batch_scores.iter().zip(individual_scores.iter()) {
            assert!(
                (batch - individual).abs() < 1e-6,
                "Batch {batch} != individual {individual}"
            );
        }
    }

    #[test]
    fn test_maxsim_batch_empty_inputs() {
        let q1 = [1.0, 0.0];
        let query: Vec<&[f32]> = vec![&q1];
        let empty_query: Vec<&[f32]> = vec![];

        let d1 = [1.0, 0.0];
        let doc1: Vec<&[f32]> = vec![&d1];
        let empty_doc: Vec<&[f32]> = vec![];

        // Empty query -> all zeros
        let scores = super::maxsim_batch(&empty_query, &[doc1.clone()]);
        assert_eq!(scores, vec![0.0]);

        // Empty doc -> zero score
        let scores = super::maxsim_batch(&query, &[empty_doc]);
        assert_eq!(scores, vec![0.0]);

        // Empty doc list -> empty scores
        let empty_docs: &[Vec<&[f32]>] = &[];
        let scores = super::maxsim_batch(&query, empty_docs);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_maxsim_batch_par_matches_sequential() {
        use rand::prelude::*;

        let mut rng = StdRng::seed_from_u64(99999);
        let dim = 32;

        // Generate query with 10 tokens
        let query_vecs: Vec<Vec<f32>> = (0..10)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
            .collect();
        let query: Vec<&[f32]> = query_vecs.iter().map(|v| v.as_slice()).collect();

        // Generate 50 documents with varying token counts
        let docs: Vec<Vec<Vec<f32>>> = (0..50)
            .map(|_| {
                let num_tokens = rng.gen_range(5..20);
                (0..num_tokens)
                    .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
                    .collect()
            })
            .collect();

        let docs_refs: Vec<Vec<&[f32]>> = docs
            .iter()
            .map(|doc| doc.iter().map(|v| v.as_slice()).collect())
            .collect();

        // Compare sequential and parallel
        let seq_scores = super::maxsim_batch(&query, &docs_refs);
        let par_scores = super::maxsim_batch_par(&query, &docs_refs);

        assert_eq!(seq_scores.len(), par_scores.len());
        for (seq, par) in seq_scores.iter().zip(par_scores.iter()) {
            assert!(
                (seq - par).abs() < 1e-6,
                "Sequential {seq} != parallel {par}"
            );
        }
    }

    #[test]
    fn test_maxsim_batch_ordering() {
        // Verify that batch scores maintain correct ordering for reranking
        let q1 = [1.0, 0.0, 0.0, 0.0];
        let query: Vec<&[f32]> = vec![&q1];

        // doc1: perfect match
        let d1 = [1.0, 0.0, 0.0, 0.0];
        let doc1: Vec<&[f32]> = vec![&d1];

        // doc2: partial match
        let d2 = [0.5, 0.5, 0.0, 0.0];
        let doc2: Vec<&[f32]> = vec![&d2];

        // doc3: no match
        let d3 = [0.0, 0.0, 1.0, 0.0];
        let doc3: Vec<&[f32]> = vec![&d3];

        let docs = vec![doc3.clone(), doc1.clone(), doc2.clone()]; // Scrambled order
        let scores = super::maxsim_batch(&query, &docs);

        // Scores should be: [0.0, 1.0, 0.5] (doc3, doc1, doc2)
        assert!((scores[0] - 0.0).abs() < 1e-6, "doc3 should score 0");
        assert!((scores[1] - 1.0).abs() < 1e-6, "doc1 should score 1");

        // doc2 score is dot(q1, d2) = 1.0*0.5 + 0 + 0 + 0 = 0.5
        let expected_d2 = 0.5;
        assert!(
            (scores[2] - expected_d2).abs() < 1e-6,
            "doc2 should score {expected_d2}, got {}",
            scores[2]
        );
    }
}
