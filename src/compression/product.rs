//! Product Quantization (PQ) for OmenDB
//!
//! Compresses vectors into M bytes (one codeword index per subspace).
//! Uses Asymmetric Distance Computation (ADC) with pre-computed lookup tables.
//!
//! # Algorithm
//!
//! 1. Divide D-dimensional space into M subspaces of D/M dimensions each
//! 2. k-means cluster each subspace into 256 centroids (1 byte per subspace)
//! 3. For search: pre-compute query-to-centroid distances per subspace (ADC table)
//! 4. Distance = sum of table lookups per code byte
//!
//! # Compression
//!
//! Example: 768D, M=96 → 96 bytes per vector (32x vs 3072 bytes for f32)
//!
//! # Performance
//!
//! ADC lookup is O(M) per distance, independent of D.
//! With rescore, recall is comparable to full precision.

use serde::{Deserialize, Serialize};

/// Number of codewords per subspace (fixed at 256 for u8 codes)
const NUM_CODEWORDS: usize = 256;

/// k-means iterations for codebook training
const KMEANS_ITERATIONS: usize = 25;

/// Minimum vectors required for training
const MIN_TRAINING_VECTORS: usize = 256;

/// Product Quantization parameters (trained codebooks)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQParams {
    /// Number of subspaces
    pub num_subspaces: usize,
    /// Dimensions per subspace (D / num_subspaces)
    pub subspace_dim: usize,
    /// Total dimensions
    pub dimensions: usize,
    /// Codebooks: [num_subspaces][256][subspace_dim]
    /// Flattened for cache efficiency: [num_subspaces * 256 * subspace_dim]
    pub codebooks: Vec<f32>,
}

/// Pre-computed ADC lookup table for a query
#[derive(Debug, Clone)]
pub struct PQQueryPrep {
    /// Distance lookup table: [num_subspaces][256]
    /// table[m][c] = ||q_m - centroid_m_c||^2
    pub table: Vec<f32>,
    /// Number of subspaces (for indexing)
    pub num_subspaces: usize,
}

impl PQParams {
    /// Train PQ codebooks from sample vectors using k-means
    ///
    /// # Arguments
    /// - `vectors`: Training vectors (each must have `dimensions` elements)
    /// - `num_subspaces`: Number of subspaces (must divide dimensions evenly)
    ///
    /// # Errors
    /// Returns error if vectors is empty, dimensions don't match, or
    /// num_subspaces doesn't divide dimensions evenly.
    pub fn train(vectors: &[&[f32]], num_subspaces: usize) -> Result<Self, &'static str> {
        if vectors.is_empty() {
            return Err("Need at least one vector to train PQ");
        }
        if vectors.len() < MIN_TRAINING_VECTORS {
            return Err("Need at least 256 vectors to train PQ");
        }

        let dimensions = vectors[0].len();
        if dimensions == 0 {
            return Err("Vectors must have at least one dimension");
        }
        if !dimensions.is_multiple_of(num_subspaces) {
            return Err("Dimensions must be evenly divisible by num_subspaces");
        }

        let subspace_dim = dimensions / num_subspaces;
        let n = vectors.len();

        // Flatten codebooks: [num_subspaces * 256 * subspace_dim]
        let mut codebooks = vec![0.0f32; num_subspaces * NUM_CODEWORDS * subspace_dim];

        // Train each subspace independently
        for m in 0..num_subspaces {
            let sub_offset = m * subspace_dim;
            let codebook_offset = m * NUM_CODEWORDS * subspace_dim;

            // Extract subspace vectors
            let mut sub_vectors: Vec<f32> = Vec::with_capacity(n * subspace_dim);
            for vec in vectors {
                sub_vectors.extend_from_slice(&vec[sub_offset..sub_offset + subspace_dim]);
            }

            // Run k-means for this subspace
            kmeans(
                &sub_vectors,
                n,
                subspace_dim,
                &mut codebooks[codebook_offset..codebook_offset + NUM_CODEWORDS * subspace_dim],
            );
        }

        Ok(Self {
            num_subspaces,
            subspace_dim,
            dimensions,
            codebooks,
        })
    }

    /// Quantize a vector to PQ codes (one u8 per subspace)
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        debug_assert_eq!(vector.len(), self.dimensions);
        let mut codes = Vec::with_capacity(self.num_subspaces);

        for m in 0..self.num_subspaces {
            let sub_offset = m * self.subspace_dim;
            let sub_vec = &vector[sub_offset..sub_offset + self.subspace_dim];
            let codebook_offset = m * NUM_CODEWORDS * self.subspace_dim;

            let mut best_code = 0u8;
            let mut best_dist = f32::MAX;

            for c in 0..NUM_CODEWORDS {
                let centroid_offset = codebook_offset + c * self.subspace_dim;
                let centroid =
                    &self.codebooks[centroid_offset..centroid_offset + self.subspace_dim];

                let dist: f32 = sub_vec
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();

                if dist < best_dist {
                    best_dist = dist;
                    best_code = c as u8;
                }
            }

            codes.push(best_code);
        }

        codes
    }

    /// Build ADC (Asymmetric Distance Computation) lookup table for a query
    ///
    /// Pre-computes query-to-centroid distances for each subspace.
    /// Total distance is then just a sum of M table lookups.
    pub fn build_adc_table(&self, query: &[f32]) -> PQQueryPrep {
        debug_assert_eq!(query.len(), self.dimensions);
        let mut table = vec![0.0f32; self.num_subspaces * NUM_CODEWORDS];

        for m in 0..self.num_subspaces {
            let sub_offset = m * self.subspace_dim;
            let query_sub = &query[sub_offset..sub_offset + self.subspace_dim];
            let codebook_offset = m * NUM_CODEWORDS * self.subspace_dim;
            let table_offset = m * NUM_CODEWORDS;

            for c in 0..NUM_CODEWORDS {
                let centroid_offset = codebook_offset + c * self.subspace_dim;
                let centroid =
                    &self.codebooks[centroid_offset..centroid_offset + self.subspace_dim];

                let dist: f32 = query_sub
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();

                table[table_offset + c] = dist;
            }
        }

        PQQueryPrep {
            table,
            num_subspaces: self.num_subspaces,
        }
    }

    /// Compute approximate L2 squared distance using ADC
    ///
    /// Sums pre-computed sub-distances from the lookup table.
    /// O(M) per distance, independent of D.
    #[inline]
    pub fn distance_adc(prep: &PQQueryPrep, codes: &[u8]) -> f32 {
        debug_assert_eq!(codes.len(), prep.num_subspaces);
        let mut dist = 0.0f32;
        let m = prep.num_subspaces;

        // Unroll by 4 for ILP
        let chunks = m / 4;
        let remainder = m % 4;

        for i in 0..chunks {
            let base = i * 4;
            unsafe {
                dist += *prep
                    .table
                    .get_unchecked(base * NUM_CODEWORDS + *codes.get_unchecked(base) as usize);
                dist += *prep.table.get_unchecked(
                    (base + 1) * NUM_CODEWORDS + *codes.get_unchecked(base + 1) as usize,
                );
                dist += *prep.table.get_unchecked(
                    (base + 2) * NUM_CODEWORDS + *codes.get_unchecked(base + 2) as usize,
                );
                dist += *prep.table.get_unchecked(
                    (base + 3) * NUM_CODEWORDS + *codes.get_unchecked(base + 3) as usize,
                );
            }
        }

        for i in (chunks * 4)..(chunks * 4 + remainder) {
            unsafe {
                dist += *prep
                    .table
                    .get_unchecked(i * NUM_CODEWORDS + *codes.get_unchecked(i) as usize);
            }
        }

        dist
    }

    /// Batch compute PQ distances for multiple code vectors
    #[inline]
    pub fn distance_adc_batch(
        prep: &PQQueryPrep,
        codes_batch: &[u8],
        num_subspaces: usize,
        distances: &mut [f32],
    ) {
        let n = distances.len();
        for i in 0..n {
            let codes = &codes_batch[i * num_subspaces..(i + 1) * num_subspaces];
            distances[i] = Self::distance_adc(prep, codes);
        }
    }

    /// Get the codebook for a specific subspace and codeword
    #[inline]
    fn centroid(&self, subspace: usize, codeword: usize) -> &[f32] {
        let offset = (subspace * NUM_CODEWORDS + codeword) * self.subspace_dim;
        &self.codebooks[offset..offset + self.subspace_dim]
    }

    /// Reconstruct an approximate vector from PQ codes
    pub fn reconstruct(&self, codes: &[u8]) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.dimensions);
        for (m, &code) in codes.iter().enumerate() {
            vector.extend_from_slice(self.centroid(m, code as usize));
        }
        vector
    }

    /// Compute the number of bytes per vector
    #[inline]
    #[must_use]
    pub fn code_size(&self) -> usize {
        self.num_subspaces
    }

    /// Serialize codebooks to bytes
    pub fn serialize_codebooks(&self) -> Vec<u8> {
        // Format: [num_subspaces:u32][subspace_dim:u32][dimensions:u32][codebooks:f32*N]
        let mut buf = Vec::with_capacity(12 + self.codebooks.len() * 4);
        buf.extend_from_slice(&(self.num_subspaces as u32).to_le_bytes());
        buf.extend_from_slice(&(self.subspace_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(self.dimensions as u32).to_le_bytes());
        for &val in &self.codebooks {
            buf.extend_from_slice(&val.to_le_bytes());
        }
        buf
    }

    /// Deserialize codebooks from bytes
    pub fn deserialize_codebooks(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < 12 {
            return Err("PQ data too short for header");
        }
        let num_subspaces = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let subspace_dim = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let dimensions = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;

        let expected_len = 12 + num_subspaces * NUM_CODEWORDS * subspace_dim * 4;
        if data.len() < expected_len {
            return Err("PQ data too short for codebooks");
        }

        let mut codebooks = vec![0.0f32; num_subspaces * NUM_CODEWORDS * subspace_dim];
        for (i, val) in codebooks.iter_mut().enumerate() {
            let offset = 12 + i * 4;
            *val = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        }

        Ok(Self {
            num_subspaces,
            subspace_dim,
            dimensions,
            codebooks,
        })
    }
}

/// Simple k-means clustering for a single subspace
///
/// Clusters `n` vectors of `dim` dimensions into 256 centroids.
/// Results written directly to `centroids` buffer.
fn kmeans(data: &[f32], n: usize, dim: usize, centroids: &mut [f32]) {
    debug_assert_eq!(data.len(), n * dim);
    debug_assert_eq!(centroids.len(), NUM_CODEWORDS * dim);

    let k = NUM_CODEWORDS.min(n); // Can't have more centroids than data points

    // Initialize centroids: pick k evenly-spaced data points
    let step = n / k;
    for c in 0..k {
        let src_idx = (c * step) * dim;
        let dst_idx = c * dim;
        centroids[dst_idx..dst_idx + dim].copy_from_slice(&data[src_idx..src_idx + dim]);
    }

    // If fewer data points than centroids, zero-fill remaining
    if k < NUM_CODEWORDS {
        centroids[k * dim..].fill(0.0);
    }

    let mut assignments = vec![0u8; n];
    let mut counts = vec![0u32; NUM_CODEWORDS];

    for _ in 0..KMEANS_ITERATIONS {
        // Assignment step: assign each vector to nearest centroid
        for (assignment, vec_chunk) in assignments.iter_mut().zip(data.chunks_exact(dim)) {
            let mut best = 0u8;
            let mut best_dist = f32::MAX;

            for c in 0..k {
                let cent_offset = c * dim;
                let cent = &centroids[cent_offset..cent_offset + dim];

                let dist: f32 = vec_chunk
                    .iter()
                    .zip(cent)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();

                if dist < best_dist {
                    best_dist = dist;
                    best = c as u8;
                }
            }
            *assignment = best;
        }

        // Update step: recompute centroids as mean of assigned vectors
        centroids[..k * dim].fill(0.0);
        counts[..k].fill(0);

        for (i, &assignment) in assignments.iter().enumerate() {
            let c = assignment as usize;
            counts[c] += 1;
            let vec_offset = i * dim;
            let cent_offset = c * dim;
            for d in 0..dim {
                centroids[cent_offset + d] += data[vec_offset + d];
            }
        }

        for (c, &count) in counts.iter().enumerate().take(k) {
            if count > 0 {
                let cent_offset = c * dim;
                let count_f = count as f32;
                for d in 0..dim {
                    centroids[cent_offset + d] /= count_f;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng_state = seed;
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        // Simple xorshift64 for reproducibility
                        rng_state ^= rng_state << 13;
                        rng_state ^= rng_state >> 7;
                        rng_state ^= rng_state << 17;
                        (rng_state as f32 / u64::MAX as f32) * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_pq_train_and_quantize() {
        let vectors = make_random_vectors(512, 128, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let params = PQParams::train(&refs, 16).unwrap();
        assert_eq!(params.num_subspaces, 16);
        assert_eq!(params.subspace_dim, 8);
        assert_eq!(params.dimensions, 128);

        let codes = params.quantize(&vectors[0]);
        assert_eq!(codes.len(), 16);
    }

    #[test]
    fn test_pq_adc_distance() {
        let vectors = make_random_vectors(512, 128, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let params = PQParams::train(&refs, 16).unwrap();

        let query = &vectors[0];
        let prep = params.build_adc_table(query);
        let codes = params.quantize(query);

        // Self-distance should be relatively small (PQ quantization introduces error)
        let self_dist = PQParams::distance_adc(&prep, &codes);
        assert!(
            self_dist < 50.0,
            "Self-distance should be small, got {}",
            self_dist
        );

        // Distance to different vector should be larger
        let other_codes = params.quantize(&vectors[100]);
        let other_dist = PQParams::distance_adc(&prep, &other_codes);
        assert!(other_dist > 0.0);
    }

    #[test]
    fn test_pq_reconstruction() {
        let vectors = make_random_vectors(512, 128, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let params = PQParams::train(&refs, 16).unwrap();
        let codes = params.quantize(&vectors[0]);
        let reconstructed = params.reconstruct(&codes);

        assert_eq!(reconstructed.len(), 128);

        // Reconstruction error should be bounded
        let error: f32 = vectors[0]
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        assert!(error < 50.0, "Reconstruction error too large: {}", error);
    }

    #[test]
    fn test_pq_serialize_roundtrip() {
        let vectors = make_random_vectors(512, 128, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let params = PQParams::train(&refs, 16).unwrap();
        let bytes = params.serialize_codebooks();
        let params2 = PQParams::deserialize_codebooks(&bytes).unwrap();

        assert_eq!(params.num_subspaces, params2.num_subspaces);
        assert_eq!(params.subspace_dim, params2.subspace_dim);
        assert_eq!(params.dimensions, params2.dimensions);
        assert_eq!(params.codebooks.len(), params2.codebooks.len());

        // Codebooks should match exactly
        for (a, b) in params.codebooks.iter().zip(params2.codebooks.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_pq_768d_compression() {
        // Realistic scenario: 768D vectors with 96 subspaces
        let vectors = make_random_vectors(512, 768, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let params = PQParams::train(&refs, 96).unwrap();
        assert_eq!(params.num_subspaces, 96);
        assert_eq!(params.subspace_dim, 8);

        let codes = params.quantize(&vectors[0]);
        assert_eq!(codes.len(), 96); // 96 bytes vs 3072 bytes (32x compression)

        // ADC should work
        let prep = params.build_adc_table(&vectors[0]);
        let dist = PQParams::distance_adc(&prep, &codes);
        assert!(dist < 100.0, "Self-distance too large: {}", dist);
    }

    #[test]
    fn test_pq_distance_ordering() {
        let vectors = make_random_vectors(512, 128, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let params = PQParams::train(&refs, 16).unwrap();
        let query = &vectors[0];
        let prep = params.build_adc_table(query);

        // Compute PQ distances and exact distances
        let mut pq_dists: Vec<(usize, f32)> = (0..512)
            .map(|i| {
                let codes = params.quantize(&vectors[i]);
                (i, PQParams::distance_adc(&prep, &codes))
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

        pq_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Check recall@10: how many of the true 10 nearest are in PQ's top 10
        let exact_top10: std::collections::HashSet<usize> =
            exact_dists.iter().take(10).map(|(i, _)| *i).collect();
        let pq_top10: std::collections::HashSet<usize> =
            pq_dists.iter().take(10).map(|(i, _)| *i).collect();

        let recall = exact_top10.intersection(&pq_top10).count() as f32 / 10.0;
        assert!(
            recall >= 0.5,
            "PQ recall@10 too low: {:.1}% (expected >= 50%)",
            recall * 100.0
        );
    }
}
