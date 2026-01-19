// HNSW Index - Main implementation
//
// V1 Architecture:
// - Unified NodeStorage with colocated vectors + neighbors (level 0)
// - Cache-line aligned nodes for optimal prefetch
// - Sparse upper level storage (only ~5% of nodes have level > 0)
// - SQ8 quantization support with integer SIMD
//
// Module structure:
// - mod.rs: Core struct, constructors, getters, distance methods
// - insert.rs: Insert operations (single, batch, graph construction)
// - search.rs: Search operations (k-NN, filtered, layer-level)
// - persistence.rs: Save/load to disk
// - stats.rs: Statistics, memory usage, cache optimization

/// Dispatch distance function to monomorphized implementations
macro_rules! dispatch_distance {
    ($distance_fn:expr, $Type:ident => $body:expr) => {
        match $distance_fn {
            crate::vector::hnsw::types::DistanceFunction::L2 => {
                type $Type = crate::vector::hnsw::types::L2;
                $body
            }
            crate::vector::hnsw::types::DistanceFunction::Cosine => {
                type $Type = crate::vector::hnsw::types::Cosine;
                $body
            }
            crate::vector::hnsw::types::DistanceFunction::NegativeDotProduct => {
                type $Type = crate::vector::hnsw::types::NegDot;
                $body
            }
        }
    };
}

mod delete;
mod insert;
mod parallel;
mod persistence;
mod search;
mod stats;

#[cfg(test)]
mod tests;

// Re-export parallel builder
pub use parallel::ParallelBuilder;

use super::error::{HNSWError, Result};
use super::node_storage::NodeStorage;
use super::types::{DistanceFunction, HNSWParams};
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
/// V1 architecture with unified NodeStorage for cache-efficient colocated storage.
///
/// **Note**: Not Clone due to raw pointer storage in NodeStorage.
/// Use persistence APIs (save/load) instead of cloning.
#[derive(Debug)]
pub struct HNSWIndex {
    /// Unified node storage (vectors + neighbors colocated at level 0)
    ///
    /// Replaces separate nodes/neighbors/vectors storage with cache-efficient
    /// colocated layout where a single prefetch covers both vector and neighbors.
    pub(super) storage: NodeStorage,

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

    /// Build an HNSWIndex with pre-created storage
    fn build(storage: NodeStorage, params: HNSWParams, distance_fn: DistanceFunction) -> Self {
        Self {
            storage,
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
    /// * `use_quantization` - Whether to use SQ8 quantization
    pub fn new(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: DistanceFunction,
        use_quantization: bool,
    ) -> Result<Self> {
        params.validate().map_err(HNSWError::InvalidParams)?;

        let storage = if use_quantization {
            NodeStorage::new_sq8(dimensions, params.m, params.max_level as usize)
        } else {
            NodeStorage::new(dimensions, params.m, params.max_level as usize)
        };

        Ok(Self::build(storage, params, distance_fn))
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
        let storage = NodeStorage::new_sq8(dimensions, params.m, params.max_level as usize);
        Ok(Self::build(storage, params, distance_fn))
    }

    // =========================================================================
    // Getters
    // =========================================================================

    /// Check if this index uses asymmetric search (SQ8)
    #[must_use]
    pub fn is_asymmetric(&self) -> bool {
        self.storage.is_sq8()
    }

    /// Check if this index uses SQ8 quantization
    #[must_use]
    pub fn is_sq8(&self) -> bool {
        self.storage.is_sq8()
    }

    /// Train the quantizer from sample vectors
    ///
    /// Note: SQ8 training is now automatic (lazy training after 256 vectors).
    /// This method is provided for API compatibility.
    pub fn train_quantizer(&mut self, _sample_vectors: &[Vec<f32>]) -> Result<()> {
        // NodeStorage handles training automatically via lazy training
        // This is a no-op for compatibility
        Ok(())
    }

    /// Get number of vectors in index
    #[must_use]
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if index is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Get dimensions
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.storage.dimensions()
    }

    /// Get a vector by ID (full precision)
    ///
    /// Returns None if the ID is invalid, out of bounds, or in SQ8 mode.
    /// For SQ8 mode, use `get_vector_dequantized()` instead.
    #[must_use]
    pub fn get_vector(&self, id: u32) -> Option<&[f32]> {
        if self.storage.is_sq8() || (id as usize) >= self.storage.len() {
            return None;
        }
        Some(self.storage.vector(id))
    }

    /// Get a dequantized vector by ID
    ///
    /// Works for both full precision and SQ8 modes.
    /// For full precision, returns a copy of the vector.
    /// For SQ8, returns the dequantized approximation.
    #[must_use]
    pub fn get_vector_dequantized(&self, id: u32) -> Option<Vec<f32>> {
        self.storage.get_dequantized(id)
    }

    /// Get entry point
    #[must_use]
    pub fn entry_point(&self) -> Option<u32> {
        self.entry_point
    }

    /// Get node level
    #[must_use]
    pub fn node_level(&self, node_id: u32) -> Option<u8> {
        if (node_id as usize) >= self.storage.len() {
            return None;
        }
        Some(self.storage.level(node_id))
    }

    /// Get neighbor count for a node at a level
    #[must_use]
    pub fn neighbor_count(&self, node_id: u32, level: u8) -> usize {
        self.storage.neighbor_count_at_level(node_id, level)
    }

    /// Serialize index to bytes (for in-memory persistence)
    ///
    /// This is more efficient than save() for embedding in other data structures.
    /// Uses the same format as save() but returns bytes instead of writing to file.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();

        // Magic bytes
        out.extend_from_slice(b"HNSWIDX\0");

        // Version
        out.extend_from_slice(&4u32.to_le_bytes());

        // Entry point
        match self.entry_point {
            Some(ep) => {
                out.push(1u8);
                out.extend_from_slice(&ep.to_le_bytes());
            }
            None => {
                out.push(0u8);
            }
        }

        // Distance function (length-prefixed postcard)
        let df_bytes = postcard::to_allocvec(&self.distance_fn)?;
        out.extend_from_slice(&(df_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&df_bytes);

        // Params (length-prefixed postcard)
        let params_bytes = postcard::to_allocvec(&self.params)?;
        out.extend_from_slice(&(params_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&params_bytes);

        // RNG state
        out.extend_from_slice(&self.rng_state.to_le_bytes());

        // Storage (complete NodeStorage serialization)
        let storage_bytes = self.storage.serialize_full();
        out.extend_from_slice(&(storage_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&storage_bytes);

        Ok(out)
    }

    /// Deserialize index from bytes
    ///
    /// Counterpart to `to_bytes()`. Use for loading from embedded byte storage.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 12 {
            return Err(HNSWError::Storage("Data too short for header".to_string()));
        }

        let mut pos = 0;

        // Magic bytes
        if &data[pos..pos + 8] != b"HNSWIDX\0" {
            return Err(HNSWError::Storage("Invalid magic bytes".to_string()));
        }
        pos += 8;

        // Version
        let version = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        if version != 4 {
            return Err(HNSWError::Storage(format!(
                "Unsupported version: {version} (expected 4)"
            )));
        }

        // Entry point
        let entry_point = if data[pos] == 1 {
            pos += 1;
            let ep = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            Some(ep)
        } else {
            pos += 1;
            None
        };

        // Distance function
        let df_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let distance_fn: DistanceFunction = postcard::from_bytes(&data[pos..pos + df_len])?;
        pos += df_len;

        // Params
        let params_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let params: HNSWParams = postcard::from_bytes(&data[pos..pos + params_len])?;
        pos += params_len;

        // RNG state
        let rng_state = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        // Storage
        let storage_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let storage = NodeStorage::deserialize_full(&data[pos..pos + storage_len])
            .map_err(|e| HNSWError::Storage(format!("Failed to deserialize storage: {e}")))?;

        Ok(Self {
            storage,
            entry_point,
            params,
            distance_fn,
            rng_state,
        })
    }

    /// Get HNSW parameters
    #[must_use]
    pub fn params(&self) -> &HNSWParams {
        &self.params
    }

    /// Get distance function
    #[must_use]
    pub fn distance_function(&self) -> DistanceFunction {
        self.distance_fn
    }

    /// Get neighbors at level 0 for a node
    ///
    /// Level 0 has the most connections (M*2) and is used for graph merging.
    #[must_use]
    pub fn get_neighbors_level0(&self, node_id: u32) -> Vec<u32> {
        self.storage.neighbors(node_id).to_vec()
    }

    /// Get neighbors at any level for a node
    #[must_use]
    pub fn get_neighbors(&self, node_id: u32, level: u8) -> Vec<u32> {
        self.storage.neighbors_at_level(node_id, level)
    }

    /// Get slot ID for a node (maps to RecordStore)
    #[must_use]
    pub fn slot(&self, node_id: u32) -> Option<u32> {
        if (node_id as usize) >= self.storage.len() {
            return None;
        }
        Some(self.storage.slot(node_id))
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
        if self.storage.is_sq8() {
            // SQ8 mode: use prepared query for id_a, compute distance to id_b
            if let Some(vec_a) = self.storage.get_dequantized(id_a) {
                if let Some(prep) = self.storage.prepare_query(&vec_a) {
                    if let Some(dist) = self.storage.distance_sq8(&prep, id_b) {
                        return Ok(dist);
                    }
                }
            }
            // SQ8 fallback: dequantize both (slow but necessary)
            let vec_a = self
                .storage
                .get_dequantized(id_a)
                .ok_or(HNSWError::VectorNotFound(id_a))?;
            let vec_b = self
                .storage
                .get_dequantized(id_b)
                .ok_or(HNSWError::VectorNotFound(id_b))?;
            Ok(self.distance_fn.distance_for_comparison(&vec_a, &vec_b))
        } else {
            // Full precision: use zero-copy references (no allocation)
            let vec_a = self.storage.vector(id_a);
            let vec_b = self.storage.vector(id_b);
            Ok(self.distance_fn.distance_for_comparison(vec_a, vec_b))
        }
    }

    /// Distance from query to node for ordering comparisons
    ///
    /// Tries SQ8 fast path first, falls back to full precision.
    #[inline(always)]
    pub(super) fn distance_cmp(&self, query: &[f32], id: u32) -> Result<f32> {
        if self.storage.is_sq8() {
            // SQ8 fast path
            if let Some(prep) = self.storage.prepare_query(query) {
                if let Some(dist) = self.storage.distance_sq8(&prep, id) {
                    return Ok(dist);
                }
            }
        }
        // Full precision path
        if self.storage.is_sq8() {
            let vec = self
                .storage
                .get_dequantized(id)
                .ok_or(HNSWError::VectorNotFound(id))?;
            Ok(self.distance_fn.distance_for_comparison(query, &vec))
        } else {
            let vec = self.storage.vector(id);
            Ok(self.distance_fn.distance_for_comparison(query, vec))
        }
    }

    /// Actual distance (with sqrt for L2)
    #[inline]
    pub(super) fn distance_exact(&self, query: &[f32], id: u32) -> Result<f32> {
        if self.storage.is_sq8() {
            // SQ8: use prepared query (returns squared L2)
            if let Some(prep) = self.storage.prepare_query(query) {
                if let Some(dist) = self.storage.distance_sq8(&prep, id) {
                    return Ok(dist.sqrt());
                }
            }
        }
        // Full precision path
        if self.storage.is_sq8() {
            let vec = self
                .storage
                .get_dequantized(id)
                .ok_or(HNSWError::VectorNotFound(id))?;
            Ok(self.distance_fn.distance(query, &vec))
        } else {
            let vec = self.storage.vector(id);
            Ok(self.distance_fn.distance(query, vec))
        }
    }

    /// Check if L2 decomposition optimization is available.
    ///
    /// Returns true if storage is SQ8 AND distance function is L2.
    #[inline]
    #[allow(dead_code)]
    pub(super) fn supports_l2_decomposition(&self) -> bool {
        matches!(self.distance_fn, DistanceFunction::L2) && self.storage.is_sq8()
    }

    /// Check if SQ8 training is complete (always true for full precision)
    #[inline]
    #[allow(dead_code)]
    pub(super) fn is_trained(&self) -> bool {
        self.storage.is_trained()
    }

    /// Prefetch node data for cache efficiency
    #[inline]
    #[allow(dead_code)]
    pub(super) fn prefetch(&self, node_id: u32) {
        self.storage.prefetch(node_id);
    }
}
