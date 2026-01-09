// HNSW Index - Main implementation
//
// Architecture:
// - Flattened index (contiguous nodes, u32 node IDs)
// - Separate neighbor storage (fetch only when needed)
// - Cache-optimized layout (64-byte aligned hot data)
//
// Module structure:
// - mod.rs: Core struct, constructors, getters, distance methods
// - insert.rs: Insert operations (single, batch, graph construction)
// - search.rs: Search operations (k-NN, filtered, layer-level)
// - persistence.rs: Save/load to disk
// - stats.rs: Statistics, memory usage, cache optimization

mod delete;
mod insert;
mod persistence;
mod search;
mod stats;

#[cfg(test)]
mod tests;

use super::error::{HNSWError, Result};
use super::graph_storage::GraphStorage;
use super::storage::VectorStorage;
use super::types::{Distance, DistanceFunction, HNSWNode, HNSWParams};
use crate::compression::RaBitQParams;
use serde::{Deserialize, Serialize};

/// Index statistics for monitoring and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Total number of vectors in index
    pub num_vectors: usize,

    /// Vector dimensionality
    pub dimensions: usize,

    /// Entry point node ID
    pub entry_point: Option<u32>,

    /// Maximum level in the graph
    pub max_level: u8,

    /// Level distribution (count of nodes at each level as their TOP level)
    pub level_distribution: Vec<usize>,

    /// Average neighbors per node (level 0)
    pub avg_neighbors_l0: f32,

    /// Max neighbors per node (level 0)
    pub max_neighbors_l0: usize,

    /// Memory usage in bytes
    pub memory_bytes: usize,

    /// HNSW parameters
    pub params: HNSWParams,

    /// Distance function
    pub distance_function: DistanceFunction,

    /// Whether quantization is enabled
    pub quantization_enabled: bool,
}

/// HNSW Index
///
/// Hierarchical graph index for approximate nearest neighbor search.
/// Optimized for cache locality and memory efficiency.
///
/// **Note**: Not Clone due to `GraphStorage` containing non-cloneable backends.
/// Use persistence APIs (save/load) instead of cloning.
#[derive(Debug, Serialize, Deserialize)]
pub struct HNSWIndex {
    /// Node metadata (cache-line aligned)
    pub(super) nodes: Vec<HNSWNode>,

    /// Graph storage (mode-dependent: in-memory or hybrid disk+cache)
    pub(super) neighbors: GraphStorage,

    /// Vector storage (full precision or quantized)
    pub(super) vectors: VectorStorage,

    /// Entry point (top-level node)
    pub(super) entry_point: Option<u32>,

    /// Construction parameters
    pub(super) params: HNSWParams,

    /// Distance function
    pub(super) distance_fn: DistanceFunction,

    /// Random number generator seed state
    pub(super) rng_state: u64,
}

impl HNSWIndex {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Build an HNSWIndex with pre-created vector storage
    fn build(vectors: VectorStorage, params: HNSWParams, distance_fn: DistanceFunction) -> Self {
        let neighbors = GraphStorage::new(params.max_level as usize);
        Self {
            nodes: Vec::new(),
            neighbors,
            vectors,
            entry_point: None,
            rng_state: params.seed,
            params,
            distance_fn,
        }
    }

    /// Validate params and check that distance function is L2 (required for quantized modes)
    fn validate_l2_required(
        params: &HNSWParams,
        distance_fn: DistanceFunction,
        mode_name: &str,
    ) -> Result<()> {
        params.validate().map_err(HNSWError::InvalidParams)?;
        if !matches!(distance_fn, DistanceFunction::L2) {
            return Err(HNSWError::InvalidParams(format!(
                "{mode_name} only supports L2 distance function"
            )));
        }
        Ok(())
    }

    /// Create a new empty HNSW index
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality
    /// * `params` - HNSW construction parameters
    /// * `distance_fn` - Distance function (L2, Cosine, Dot)
    /// * `use_quantization` - Whether to use binary quantization
    pub fn new(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: DistanceFunction,
        use_quantization: bool,
    ) -> Result<Self> {
        params.validate().map_err(HNSWError::InvalidParams)?;

        let vectors = if use_quantization {
            VectorStorage::new_binary_quantized(dimensions, true)
        } else {
            VectorStorage::new_full_precision(dimensions)
        };

        Ok(Self::build(vectors, params, distance_fn))
    }

    /// Create a new HNSW index with `RaBitQ` asymmetric search (CLOUD MOAT)
    ///
    /// This enables 2-3x faster search by using asymmetric distance computation:
    /// - Query vector stays full precision
    /// - Candidate vectors use `RaBitQ` quantization (8x smaller)
    /// - Final reranking uses full precision for accuracy
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality
    /// * `params` - HNSW construction parameters
    /// * `distance_fn` - Distance function (only L2 supported for asymmetric)
    /// * `rabitq_params` - `RaBitQ` quantization parameters (typically 4-bit)
    ///
    /// # Performance
    /// - Search: 2-3x faster than full precision
    /// - Memory: 8x smaller quantized storage (+ original for reranking)
    /// - Recall: 98%+ with reranking
    ///
    /// # Example
    /// ```ignore
    /// let params = HNSWParams::default();
    /// let rabitq = RaBitQParams::bits4(); // 4-bit, 8x compression
    /// let index = HNSWIndex::new_with_asymmetric(128, params, DistanceFunction::L2, rabitq)?;
    /// ```
    pub fn new_with_asymmetric(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: DistanceFunction,
        rabitq_params: RaBitQParams,
    ) -> Result<Self> {
        Self::validate_l2_required(&params, distance_fn, "RaBitQ asymmetric search")?;
        let vectors = VectorStorage::new_rabitq_quantized(dimensions, rabitq_params);
        Ok(Self::build(vectors, params, distance_fn))
    }

    /// Create new HNSW index with SQ8 (Scalar Quantization)
    ///
    /// SQ8 compresses f32 → u8 (4x smaller) and uses direct SIMD operations
    /// for ~2x faster search than full precision.
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality
    /// * `params` - HNSW parameters (m, `ef_construction`, `ef_search`)
    /// * `distance_fn` - Distance function (only L2 supported for SQ8)
    ///
    /// # Example
    /// ```ignore
    /// let params = HNSWParams::default();
    /// let index = HNSWIndex::new_with_sq8(768, params, DistanceFunction::L2)?;
    /// ```
    pub fn new_with_sq8(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: DistanceFunction,
    ) -> Result<Self> {
        Self::validate_l2_required(&params, distance_fn, "SQ8 quantization")?;
        let vectors = VectorStorage::new_sq8_quantized(dimensions);
        Ok(Self::build(vectors, params, distance_fn))
    }

    /// Create new HNSW index with binary (1-bit) quantization
    ///
    /// Uses SIMD-optimized Hamming distance for fast search.
    ///
    /// # Performance
    /// - Search: 2-4x faster than SQ8 (SIMD Hamming is extremely fast)
    /// - Memory: 32x smaller quantized storage (+ original for reranking)
    /// - Recall: ~85% raw, ~95-98% with reranking
    ///
    /// # Example
    /// ```ignore
    /// let params = HNSWParams::default();
    /// let index = HNSWIndex::new_with_binary(768, params, DistanceFunction::L2)?;
    /// ```
    pub fn new_with_binary(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: DistanceFunction,
    ) -> Result<Self> {
        Self::validate_l2_required(&params, distance_fn, "Binary quantization")?;
        let vectors = VectorStorage::new_binary_quantized(dimensions, true);
        Ok(Self::build(vectors, params, distance_fn))
    }

    // =========================================================================
    // Getters
    // =========================================================================

    /// Check if this index uses asymmetric search (`RaBitQ` or `SQ8`)
    #[must_use]
    pub fn is_asymmetric(&self) -> bool {
        self.vectors.is_asymmetric()
    }

    /// Check if this index uses SQ8 quantization
    #[must_use]
    pub fn is_sq8(&self) -> bool {
        self.vectors.is_sq8()
    }

    /// Train the quantizer from sample vectors
    pub fn train_quantizer(&mut self, sample_vectors: &[Vec<f32>]) -> Result<()> {
        self.vectors
            .train_quantization(sample_vectors)
            .map_err(HNSWError::InvalidParams)
    }

    /// Get number of vectors in index
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if index is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get dimensions
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.vectors.dimensions()
    }

    /// Get a vector by ID (full precision)
    ///
    /// Returns None if the ID is invalid or out of bounds.
    #[must_use]
    pub fn get_vector(&self, id: u32) -> Option<&[f32]> {
        self.vectors.get(id)
    }

    /// Get entry point
    #[must_use]
    pub fn entry_point(&self) -> Option<u32> {
        self.entry_point
    }

    /// Get node level
    #[must_use]
    pub fn node_level(&self, node_id: u32) -> Option<u8> {
        self.nodes.get(node_id as usize).map(|n| n.level)
    }

    /// Get neighbor count for a node at a level
    #[must_use]
    pub fn neighbor_count(&self, node_id: u32, level: u8) -> usize {
        self.neighbors.get_neighbors(node_id, level).len()
    }

    /// Get HNSW parameters
    #[must_use]
    pub fn params(&self) -> &HNSWParams {
        &self.params
    }

    /// Get neighbors at level 0 for a node
    ///
    /// Level 0 has the most connections (M*2) and is used for graph merging.
    #[must_use]
    pub fn get_neighbors_level0(&self, node_id: u32) -> Vec<u32> {
        self.neighbors.get_neighbors(node_id, 0)
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /// Assign random level to new node
    ///
    /// Uses exponential decay: P(level = l) = (1/M)^l
    /// This ensures most nodes are at level 0, fewer at higher levels.
    pub(super) fn random_level(&mut self) -> u8 {
        // Simple LCG for deterministic random numbers
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let rand_val = (self.rng_state >> 32) as f32 / u32::MAX as f32;

        // Exponential distribution: -ln(uniform) / ln(M)
        let level = (-rand_val.ln() * self.params.ml) as u8;
        level.min(self.params.max_level - 1)
    }

    // =========================================================================
    // Distance functions
    // =========================================================================

    /// Distance between nodes for ordering comparisons
    ///
    /// Uses dequantized vectors if storage is quantized (SQ8).
    #[inline]
    pub(super) fn distance_between_cmp(&self, id_a: u32, id_b: u32) -> Result<f32> {
        // Try asymmetric distance first (for SQ8/RaBitQ - use id_b as quantized candidate)
        if let Some(vec_a) = self.vectors.get_dequantized(id_a) {
            if let Some(dist) = self.vectors.distance_asymmetric_l2(&vec_a, id_b) {
                return Ok(dist);
            }
        }
        // Fallback to full precision
        let vec_a = self
            .vectors
            .get(id_a)
            .ok_or(HNSWError::VectorNotFound(id_a))?;
        let vec_b = self
            .vectors
            .get(id_b)
            .ok_or(HNSWError::VectorNotFound(id_b))?;
        Ok(self.distance_fn.distance_for_comparison(vec_a, vec_b))
    }

    /// Distance from query to node for ordering comparisons
    ///
    /// Tries asymmetric distance first (for SQ8/RaBitQ), falls back to full precision.
    #[inline(always)]
    pub(super) fn distance_cmp(&self, query: &[f32], id: u32) -> Result<f32> {
        // Try asymmetric distance first (for SQ8/RaBitQ storage)
        if let Some(dist) = self.vectors.distance_asymmetric_l2(query, id) {
            return Ok(dist);
        }
        // Fallback to full precision
        let vec = self.vectors.get(id).ok_or(HNSWError::VectorNotFound(id))?;
        Ok(self.distance_fn.distance_for_comparison(query, vec))
    }

    /// Monomorphized distance computation (static dispatch, no match)
    ///
    /// Critical for x86/ARM servers where branch misprediction hurts performance.
    /// The Distance trait enables compile-time specialization.
    #[inline(always)]
    pub(super) fn distance_cmp_mono<D: Distance>(&self, query: &[f32], id: u32) -> Result<f32> {
        // Try asymmetric distance first (for SQ8/RaBitQ storage)
        if let Some(dist) = self.vectors.distance_asymmetric_l2(query, id) {
            return Ok(dist);
        }
        // Fallback to full precision with static dispatch
        let vec = self.vectors.get(id).ok_or(HNSWError::VectorNotFound(id))?;
        Ok(D::distance(query, vec))
    }

    /// Distance from query to node using full precision (f32-to-f32)
    ///
    /// Used during graph construction where quantization noise hurts graph quality.
    /// For RaBitQ, uses stored originals. For SQ8, dequantizes.
    #[inline]
    pub(super) fn distance_cmp_full_precision(&self, query: &[f32], id: u32) -> Result<f32> {
        // Always use dequantized/original vectors for full precision comparison
        let vec = self
            .vectors
            .get_dequantized(id)
            .ok_or(HNSWError::VectorNotFound(id))?;
        Ok(self.distance_fn.distance_for_comparison(query, &vec))
    }

    /// Actual distance (with sqrt for L2)
    #[inline]
    pub(super) fn distance_exact(&self, query: &[f32], id: u32) -> Result<f32> {
        // For SQ8/RaBitQ: use asymmetric distance (returns squared L2)
        // For Binary: skip asymmetric (hamming is not L2), use original vectors
        if !self.vectors.is_binary_quantized() {
            if let Some(dist) = self.vectors.distance_asymmetric_l2(query, id) {
                return Ok(dist.sqrt());
            }
        }
        let vec = self.vectors.get(id).ok_or(HNSWError::VectorNotFound(id))?;
        Ok(self.distance_fn.distance(query, vec))
    }

    /// L2 distance using decomposition: ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
    ///
    /// ~7% faster than direct L2 by pre-computing vector norms during insert.
    /// Query norm is computed once per search and passed in.
    ///
    /// Returns None if decomposition is not available (non-FullPrecision storage).
    #[inline(always)]
    pub(super) fn distance_l2_decomposed(
        &self,
        query: &[f32],
        query_norm: f32,
        id: u32,
    ) -> Option<f32> {
        self.vectors.distance_l2_decomposed(query, query_norm, id)
    }

    /// Check if L2 decomposition optimization is available
    ///
    /// Returns true if storage supports L2 decomposition AND distance function is L2.
    #[inline]
    pub(super) fn supports_l2_decomposition(&self) -> bool {
        matches!(self.distance_fn, DistanceFunction::L2) && self.vectors.supports_l2_decomposition()
    }
}
