//! HNSW (Hierarchical Navigable Small World) index implementation
//!
//! High-performance vector index for approximate nearest neighbor search.
//!
//! Features:
//! - Cache-line aligned data structures (64-byte nodes)
//! - Fast binary serialization (<1 second for 100K vectors)
//! - Configurable parameters (M, `ef_construction`, `ef_search`)
//! - Multiple distance functions (L2, cosine, dot product)
//! - Optional binary quantization (32x memory reduction)

use super::hnsw::{DistanceFunction, HNSWIndex as CoreHNSW, HNSWParams as CoreParams};
use crate::compression::RaBitQParams;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Default `ef_search` value for deserialization
fn default_ef_search() -> usize {
    100
}

/// HNSW index for approximate nearest neighbor search
#[derive(Debug, Serialize, Deserialize)]
pub struct HNSWIndex {
    /// Core HNSW implementation
    index: CoreHNSW,

    /// Index parameters
    max_elements: usize,
    max_nb_connection: usize, // M parameter
    ef_construction: usize,
    dimensions: usize,

    /// Runtime search parameter (tunable, not persisted)
    #[serde(skip, default = "default_ef_search")]
    ef_search: usize,

    /// Number of vectors inserted
    num_vectors: usize,
}

/// HNSW construction and search parameters
#[derive(Debug, Clone)]
pub struct HNSWParams {
    pub max_elements: usize,
    pub max_nb_connection: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub dimensions: usize,
}

// ============================================================================
// Builder Pattern
// ============================================================================

/// Default HNSW M parameter (neighbors per node)
const DEFAULT_M: usize = 16;
/// Default HNSW ef_construction parameter (build quality)
const DEFAULT_EF_CONSTRUCTION: usize = 100;
/// Default HNSW ef_search parameter (search quality)
const DEFAULT_EF_SEARCH: usize = 100;
/// Default maximum elements
const DEFAULT_MAX_ELEMENTS: usize = 1_000_000;

/// Quantization mode for HNSW index
#[derive(Debug, Clone)]
pub enum HNSWQuantization {
    /// No quantization (full f32 precision)
    None,
    /// Binary quantization (32x compression, ~85% raw recall, ~95-98% with rescore)
    Binary,
    /// SQ8 scalar quantization (4x compression, ~99% recall)
    SQ8,
    /// RaBitQ asymmetric quantization (8x compression, ~98% recall)
    RaBitQ(RaBitQParams),
}

/// Builder for creating HNSWIndex with compile-time safety
///
/// Ensures all required parameters are provided and provides sensible defaults.
///
/// # Example
/// ```ignore
/// let index = HNSWIndex::builder()
///     .dimensions(768)
///     .m(16)
///     .ef_construction(100)
///     .metric(DistanceFunction::Cosine)
///     .quantization(HNSWQuantization::SQ8)
///     .build()?;
/// ```
#[derive(Debug, Clone)]
pub struct HNSWIndexBuilder {
    dimensions: Option<usize>,
    max_elements: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    metric: DistanceFunction,
    quantization: HNSWQuantization,
}

impl Default for HNSWIndexBuilder {
    fn default() -> Self {
        Self {
            dimensions: None,
            max_elements: DEFAULT_MAX_ELEMENTS,
            m: DEFAULT_M,
            ef_construction: DEFAULT_EF_CONSTRUCTION,
            ef_search: DEFAULT_EF_SEARCH,
            metric: DistanceFunction::L2,
            quantization: HNSWQuantization::None,
        }
    }
}

impl HNSWIndexBuilder {
    /// Create a new builder with default values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the vector dimensions (required)
    #[must_use]
    pub fn dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Set the maximum number of elements
    #[must_use]
    pub fn max_elements(mut self, max_elements: usize) -> Self {
        self.max_elements = max_elements;
        self
    }

    /// Set the M parameter (neighbors per node)
    ///
    /// Higher values improve recall but use more memory.
    /// Typical values: 16-48
    #[must_use]
    pub fn m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Set the ef_construction parameter (build quality)
    ///
    /// Higher values improve recall but slow down construction.
    /// Typical values: 100-400
    #[must_use]
    pub fn ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Set the ef_search parameter (search quality)
    ///
    /// Higher values improve recall but slow down search.
    /// Typical values: 100-400
    #[must_use]
    pub fn ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }

    /// Set the distance metric
    #[must_use]
    pub fn metric(mut self, metric: DistanceFunction) -> Self {
        self.metric = metric;
        self
    }

    /// Set the quantization mode
    #[must_use]
    pub fn quantization(mut self, quantization: HNSWQuantization) -> Self {
        self.quantization = quantization;
        self
    }

    /// Build the HNSWIndex
    ///
    /// # Errors
    /// Returns an error if dimensions is not set.
    pub fn build(self) -> Result<HNSWIndex> {
        let dimensions = self
            .dimensions
            .ok_or_else(|| anyhow::anyhow!("dimensions is required"))?;

        let params = CoreParams {
            m: self.m,
            ef_construction: self.ef_construction,
            ml: 1.0 / (self.m as f32).ln(),
            seed: 42,
            max_level: 8,
        };

        let index = match self.quantization {
            HNSWQuantization::None => CoreHNSW::new(dimensions, params, self.metric, false)?,
            HNSWQuantization::Binary => CoreHNSW::new_with_binary(dimensions, params, self.metric)
                .map_err(|e| anyhow::anyhow!(e))?,
            HNSWQuantization::SQ8 => CoreHNSW::new_with_sq8(dimensions, params, self.metric)
                .map_err(|e| anyhow::anyhow!(e))?,
            HNSWQuantization::RaBitQ(rabitq_params) => {
                CoreHNSW::new_with_asymmetric(dimensions, params, self.metric, rabitq_params)
                    .map_err(|e| anyhow::anyhow!(e))?
            }
        };

        Ok(HNSWIndex {
            index,
            max_elements: self.max_elements,
            max_nb_connection: self.m,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search,
            dimensions,
            num_vectors: 0,
        })
    }

    /// Build the HNSWIndex and train quantizer with sample vectors
    ///
    /// Use this when you have training vectors available at construction time.
    pub fn build_with_training(self, training_vectors: &[Vec<f32>]) -> Result<HNSWIndex> {
        let mut index = self.build()?;
        if !training_vectors.is_empty() && index.is_asymmetric() {
            index.train_quantizer(training_vectors)?;
        }
        Ok(index)
    }
}

impl HNSWIndex {
    /// Create a new builder for constructing an HNSWIndex
    ///
    /// This is the recommended way to create an HNSWIndex.
    ///
    /// # Example
    /// ```ignore
    /// let index = HNSWIndex::builder()
    ///     .dimensions(768)
    ///     .m(16)
    ///     .metric(DistanceFunction::Cosine)
    ///     .build()?;
    /// ```
    #[must_use]
    pub fn builder() -> HNSWIndexBuilder {
        HNSWIndexBuilder::new()
    }

    /// Create new HNSW index with adaptive parameters
    ///
    /// # Arguments
    /// * `max_elements` - Maximum number of vectors (e.g., `1_000_000`)
    /// * `dimensions` - Vector dimensionality (e.g., 1536 for `OpenAI` embeddings)
    ///
    /// # Adaptive Parameters
    /// Parameters automatically adjust based on expected dataset size:
    /// - <10K vectors: M=16, `ef_construction=100` (fast builds, 95%+ recall)
    /// - 10K-100K: M=24, `ef_construction=200` (balanced)
    /// - 100K+: M=32, `ef_construction=400` (maximum recall, 98%+)
    ///
    /// # Example
    /// ```ignore
    /// use omen::vector::HNSWIndex;
    ///
    /// let mut index = HNSWIndex::new(1_000_000, 1536, DistanceFunction::L2)?;
    /// index.insert(&vector)?;
    /// let results = index.search(&query, 10)?;
    /// ```
    pub fn new(
        max_elements: usize,
        dimensions: usize,
        distance_fn: DistanceFunction,
    ) -> Result<Self> {
        // Industry standard defaults (ChromaDB, hnswlib, Milvus, pgvector)
        // Users can override via new_with_params() if needed
        let m = 16;
        let ef_construction = 100;

        let params = CoreParams {
            m,
            ef_construction,
            ml: 1.0 / (m as f32).ln(),
            seed: 42,
            max_level: 8,
        };

        let index = CoreHNSW::new(dimensions, params, distance_fn, false)?;

        Ok(Self {
            index,
            max_elements,
            max_nb_connection: m,
            ef_construction,
            ef_search: ef_construction,
            dimensions,
            num_vectors: 0,
        })
    }

    /// Create new HNSW index with custom parameters
    ///
    /// # Arguments
    /// * `max_elements` - Maximum number of vectors
    /// * `dimensions` - Vector dimensionality
    /// * `m` - Number of bidirectional links per node (typical: 16-48)
    /// * `ef_construction` - Candidate list size during construction (typical: 200-800)
    /// * `ef_search` - Candidate list size during search (typical: 200-1000)
    /// * `distance_fn` - Distance function (L2, Cosine, NegDot)
    ///
    /// # Example
    /// ```ignore
    /// // Higher M for better recall at scale
    /// let mut index = HNSWIndex::new_with_params(1_000_000, 128, 32, 400, 600, DistanceFunction::L2)?;
    /// ```
    pub fn new_with_params(
        max_elements: usize,
        dimensions: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        distance_fn: DistanceFunction,
    ) -> Result<Self> {
        let params = CoreParams {
            m,
            ef_construction,
            ml: 1.0 / (m as f32).ln(),
            seed: 42,
            max_level: 8,
        };

        let index = CoreHNSW::new(dimensions, params, distance_fn, false)?;

        Ok(Self {
            index,
            max_elements,
            max_nb_connection: m,
            ef_construction,
            ef_search,
            dimensions,
            num_vectors: 0,
        })
    }

    /// Create new HNSW index with `RaBitQ` asymmetric search
    ///
    /// Uses asymmetric distance computation for 2-3x faster search:
    /// - Query vector stays full precision
    /// - Candidate vectors use `RaBitQ` quantization (8x smaller)
    /// - Original vectors stored for rescore accuracy
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality
    /// * `params` - HNSW parameters (m, `ef_construction`, `ef_search`)
    /// * `distance_fn` - Distance function (only L2 supported for asymmetric)
    /// * `rabitq_params` - `RaBitQ` quantization parameters (2, 4, or 8 bit)
    ///
    /// # Example
    /// ```ignore
    /// let index = HNSWIndex::new_with_asymmetric(
    ///     128,
    ///     CoreParams::default().with_m(16).with_ef_construction(100),
    ///     DistanceFunction::L2,
    ///     RaBitQParams::bits4(),
    /// )?;
    /// ```
    pub fn new_with_asymmetric(
        dimensions: usize,
        params: CoreParams,
        distance_fn: DistanceFunction,
        rabitq_params: RaBitQParams,
    ) -> Result<Self> {
        let index = CoreHNSW::new_with_asymmetric(dimensions, params, distance_fn, rabitq_params)
            .map_err(|e| anyhow::anyhow!(e))?;

        Ok(Self {
            index,
            max_elements: 1_000_000, // Default for asymmetric
            max_nb_connection: params.m,
            ef_construction: params.ef_construction,
            ef_search: params.ef_construction, // Match ef_construction initially
            dimensions,
            num_vectors: 0,
        })
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
    /// # Performance
    /// - Search: ~2x faster than full precision
    /// - Memory: 4x smaller quantized storage (+ original for reranking)
    /// - Recall: ~99% with reranking
    ///
    /// # Example
    /// ```ignore
    /// let index = HNSWIndex::new_with_sq8(
    ///     768,
    ///     CoreParams::default().with_m(16).with_ef_construction(100),
    ///     DistanceFunction::L2,
    /// )?;
    /// ```
    pub fn new_with_sq8(
        dimensions: usize,
        params: CoreParams,
        distance_fn: DistanceFunction,
    ) -> Result<Self> {
        let index = CoreHNSW::new_with_sq8(dimensions, params, distance_fn)
            .map_err(|e| anyhow::anyhow!(e))?;

        Ok(Self {
            index,
            max_elements: 1_000_000, // Default for SQ8
            max_nb_connection: params.m,
            ef_construction: params.ef_construction,
            ef_search: params.ef_construction, // Match ef_construction initially
            dimensions,
            num_vectors: 0,
        })
    }

    /// Create new HNSW index with Binary (1-bit) quantization
    ///
    /// Binary quantization uses SIMD-optimized Hamming distance for extremely
    /// fast search with 32x memory compression.
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality (384+ recommended)
    /// * `params` - HNSW parameters (m, `ef_construction`, `ef_search`)
    /// * `distance_fn` - Distance function (only L2 supported for binary)
    ///
    /// # Performance
    /// - Search: 2-4x faster than SQ8 (SIMD Hamming is extremely fast)
    /// - Memory: 32x smaller quantized storage (+ original for reranking)
    /// - Recall: ~85% raw, ~95-98% with reranking
    ///
    /// # Example
    /// ```ignore
    /// let index = HNSWIndex::new_with_binary(
    ///     768,
    ///     CoreParams::default().with_m(16).with_ef_construction(100),
    ///     DistanceFunction::L2,
    /// )?;
    /// ```
    pub fn new_with_binary(
        dimensions: usize,
        params: CoreParams,
        distance_fn: DistanceFunction,
    ) -> Result<Self> {
        let index = CoreHNSW::new_with_binary(dimensions, params, distance_fn)
            .map_err(|e| anyhow::anyhow!(e))?;

        Ok(Self {
            index,
            max_elements: 1_000_000, // Default for binary
            max_nb_connection: params.m,
            ef_construction: params.ef_construction,
            ef_search: params.ef_construction, // Match ef_construction initially
            dimensions,
            num_vectors: 0,
        })
    }

    /// Check if this index uses asymmetric search (`RaBitQ` or `SQ8`)
    #[must_use]
    pub fn is_asymmetric(&self) -> bool {
        self.index.is_asymmetric()
    }

    /// Check if this index uses SQ8 quantization
    #[must_use]
    pub fn is_sq8(&self) -> bool {
        self.index.is_sq8()
    }

    /// Train the quantizer from sample vectors
    ///
    /// Must be called before inserting vectors when using asymmetric search.
    pub fn train_quantizer(&mut self, sample_vectors: &[Vec<f32>]) -> Result<()> {
        self.index
            .train_quantizer(sample_vectors)
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Insert vector into index and return its ID
    ///
    /// # Arguments
    /// * `vector` - Vector to insert (must match index dimensions)
    ///
    /// # Returns
    /// Vector ID (sequential, starting from 0)
    pub fn insert(&mut self, vector: &[f32]) -> Result<usize> {
        if vector.len() != self.dimensions {
            anyhow::bail!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            );
        }

        let id = self.index.insert(vector).map_err(|e| anyhow::anyhow!(e))?;
        self.num_vectors += 1;
        Ok(id as usize)
    }

    /// Insert batch of vectors
    ///
    /// Currently inserts sequentially. Parallel insertion will be added
    /// in future optimization phase.
    ///
    /// # Arguments
    /// * `vectors` - Batch of vectors to insert
    ///
    /// # Returns
    /// Vector of IDs for inserted vectors
    pub fn batch_insert(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<usize>> {
        // Validate all vectors have correct dimensions
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != self.dimensions {
                anyhow::bail!(
                    "Vector {} dimension mismatch: expected {}, got {}",
                    i,
                    self.dimensions,
                    vector.len()
                );
            }
        }

        // Use parallel batch_insert from core HNSW implementation
        let core_ids = self
            .index
            .batch_insert(vectors.to_vec())
            .map_err(|e| anyhow::anyhow!(e))?;

        // Update vector count
        self.num_vectors += vectors.len();

        // Convert u32 IDs to usize
        let ids: Vec<usize> = core_ids.iter().map(|&id| id as usize).collect();

        Ok(ids)
    }

    /// Search for K nearest neighbors
    ///
    /// # Arguments
    /// * `query` - Query vector (must match index dimensions)
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    /// Vector of (ID, distance) tuples, sorted by distance (ascending)
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        self.search_with_ef(query, k, None)
    }

    /// Search for K nearest neighbors with optional ef override
    ///
    /// # Arguments
    /// * `query` - Query vector (must match index dimensions)
    /// * `k` - Number of nearest neighbors to return
    /// * `ef` - Search width override (None = use default, which auto-tunes to max(k*4, 64))
    ///
    /// # Returns
    /// Vector of (ID, distance) tuples, sorted by distance (ascending)
    #[inline]
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
    ) -> Result<Vec<(usize, f32)>> {
        // Pre-compute ef to avoid Option overhead on hot path
        let effective_ef = ef.unwrap_or_else(|| self.compute_ef(k));
        self.search_ef(query, k, effective_ef)
    }

    /// Compute default ef value for given k
    ///
    /// Returns `max(k*4, 64, ef_search)` - good balance of speed and recall.
    #[inline]
    #[must_use]
    pub fn compute_ef(&self, k: usize) -> usize {
        (k * 4).max(64).max(self.ef_search)
    }

    /// Fast search with concrete ef value (no Option overhead)
    ///
    /// Prefer this over `search_with_ef` in tight loops for ~40% better performance.
    /// Use `compute_ef(k)` to get a good default ef value.
    #[inline]
    pub fn search_ef(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<(usize, f32)>> {
        if query.len() != self.dimensions {
            anyhow::bail!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.len()
            );
        }

        // Search with HNSW
        let results = self
            .index
            .search(query, k, ef)
            .map_err(|e| anyhow::anyhow!(e))?;

        // Convert to (id, distance) tuples
        let neighbors: Vec<(usize, f32)> = results
            .iter()
            .map(|r| (r.id as usize, r.distance))
            .collect();

        Ok(neighbors)
    }

    /// Search using quantized (ADC) distances only - no exact distance calculation.
    ///
    /// Use when rescore=False for maximum speed (accepts quantization error).
    /// Falls back to regular search if not in asymmetric mode.
    #[inline]
    pub(crate) fn search_asymmetric_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(usize, f32)>> {
        if query.len() != self.dimensions {
            anyhow::bail!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.len()
            );
        }

        let results = self
            .index
            .search_asymmetric(query, k, ef)
            .map_err(|e| anyhow::anyhow!(e))?;

        let neighbors: Vec<(usize, f32)> = results
            .iter()
            .map(|r| (r.id as usize, r.distance))
            .collect();

        Ok(neighbors)
    }

    /// Search with metadata filter (ACORN-1)
    ///
    /// Uses ACORN-1 filtered search algorithm for efficient metadata-aware search.
    pub fn search_with_filter<F>(
        &self,
        query: &[f32],
        k: usize,
        filter_fn: F,
    ) -> Result<Vec<(usize, f32)>>
    where
        F: Fn(u32) -> bool,
    {
        self.search_with_filter_ef(query, k, None, filter_fn)
    }

    /// Search with metadata filter and optional ef override (ACORN-1)
    ///
    /// Uses ACORN-1 filtered search algorithm for efficient metadata-aware search.
    pub fn search_with_filter_ef<F>(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        filter_fn: F,
    ) -> Result<Vec<(usize, f32)>>
    where
        F: Fn(u32) -> bool,
    {
        if query.len() != self.dimensions {
            anyhow::bail!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.len()
            );
        }

        // Use provided ef or fall back to auto-tuned default
        let effective_ef = ef.unwrap_or_else(|| self.compute_ef(k));

        // Search with ACORN-1 filtered search
        let results = self
            .index
            .search_with_filter(query, k, effective_ef, filter_fn)
            .map_err(|e| anyhow::anyhow!(e))?;

        // Convert to (id, distance) tuples
        let neighbors: Vec<(usize, f32)> = results
            .iter()
            .map(|r| (r.id as usize, r.distance))
            .collect();

        Ok(neighbors)
    }

    /// Set `ef_search` parameter for runtime tuning
    ///
    /// Higher `ef_search` improves recall but increases query latency.
    ///
    /// # Guidelines
    /// - ef=50: ~85-90% recall, ~1ms
    /// - ef=100: ~90-95% recall, ~2ms (default)
    /// - ef=200: ~95-98% recall, ~5ms
    /// - ef=500: ~98-99% recall, ~10ms
    pub fn set_ef_search(&mut self, ef_search: usize) {
        self.ef_search = ef_search;
    }

    /// Get current `ef_search` value
    #[must_use]
    pub fn get_ef_search(&self) -> usize {
        self.ef_search
    }

    /// Optimize cache locality by reordering nodes using BFS
    ///
    /// Improves query performance by placing frequently-accessed neighbors
    /// close together in memory. Should be called after index construction
    /// and before querying for best performance.
    ///
    /// Returns the old-to-new node ID mapping. Callers must use this to update
    /// any external state (like VectorStore's id mappings).
    pub fn optimize_cache_locality(&mut self) -> Result<Vec<u32>> {
        self.index
            .optimize_cache_locality()
            .map_err(|e| anyhow::anyhow!("Optimization failed: {e}"))
    }

    /// Number of vectors in index
    #[must_use]
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Check if index is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// Get index parameters
    #[must_use]
    pub fn params(&self) -> HNSWParams {
        HNSWParams {
            max_elements: self.max_elements,
            max_nb_connection: self.max_nb_connection,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search,
            dimensions: self.dimensions,
        }
    }

    /// Save index to disk
    ///
    /// Uses fast binary serialization format. Saves both graph structure
    /// and vector data in a single file.
    ///
    /// # Performance
    /// - 100K vectors (1536D): ~500ms save, ~1s load
    /// - vs rebuild: 4175x faster loading
    ///
    /// # Format
    /// Versioned binary format (v1):
    /// - Magic bytes: "HNSWIDX\0"
    /// - Graph structure (serialized)
    /// - Vector data (full precision or quantized)
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.index.save(path).map_err(|e| anyhow::anyhow!(e))
    }

    /// Load index from disk
    ///
    /// Loads index saved with `save()` method.
    ///
    /// # Performance
    /// Fast loading: <1 second for 100K vectors (vs minutes for rebuild)
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let index = CoreHNSW::load(path).map_err(|e| anyhow::anyhow!(e))?;

        // Extract parameters from loaded index
        let dimensions = index.dimensions();
        let num_vectors = index.len();
        let params = index.params();
        let m = params.m;
        let ef_construction = params.ef_construction;

        Ok(Self {
            index,
            max_elements: num_vectors.max(1_000_000),
            max_nb_connection: m,
            ef_construction,
            ef_search: ef_construction, // Default ef_search to ef_construction
            dimensions,
            num_vectors,
        })
    }

    /// Get dimensions
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.index.memory_usage()
    }

    /// Merge another index into this one using IGTM algorithm
    ///
    /// Uses Iterative Greedy Tree Merging for 1.3-1.7x faster batch inserts
    /// compared to naive insertion.
    ///
    /// # Arguments
    /// * `other` - Index to merge from (will not be modified)
    ///
    /// # Returns
    /// Number of vectors merged
    ///
    /// # Performance
    /// ~1.3-1.7x faster than inserting vectors one by one
    pub fn merge_from(&mut self, other: &HNSWIndex) -> Result<usize> {
        use super::hnsw::{GraphMerger, MergeConfig};

        if other.dimensions != self.dimensions {
            anyhow::bail!(
                "Dimension mismatch: self={}, other={}",
                self.dimensions,
                other.dimensions
            );
        }

        let merger = GraphMerger::with_config(MergeConfig::default());
        let stats = merger
            .merge_graphs(&mut self.index, &other.index)
            .map_err(|e| anyhow::anyhow!(e))?;

        self.num_vectors += stats.vectors_merged;

        Ok(stats.vectors_merged)
    }

    /// Access the underlying core HNSW index
    ///
    /// Used for advanced operations like direct graph merging.
    #[must_use]
    pub fn core_index(&self) -> &CoreHNSW {
        &self.index
    }

    /// Access the underlying core HNSW index mutably
    pub fn core_index_mut(&mut self) -> &mut CoreHNSW {
        &mut self.index
    }

    /// Mark a node as deleted and repair the graph using MN-RU algorithm
    ///
    /// This method repairs the HNSW graph structure after a node is deleted,
    /// maintaining recall quality by reconnecting orphaned edges.
    ///
    /// # Arguments
    /// * `node_id` - The node ID to mark as deleted
    ///
    /// # Returns
    /// Number of edges repaired across all levels
    pub fn mark_deleted(&mut self, node_id: u32) -> Result<usize> {
        self.index
            .mark_deleted(node_id)
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Batch mark multiple nodes as deleted with graph repair
    ///
    /// More efficient than individual deletions when deleting many nodes.
    ///
    /// # Arguments
    /// * `node_ids` - Node IDs to delete
    ///
    /// # Returns
    /// Total number of edges repaired
    pub fn mark_deleted_batch(&mut self, node_ids: &[u32]) -> Result<usize> {
        self.index
            .mark_deleted_batch(node_ids)
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Check if a node is effectively deleted (has no neighbors)
    #[must_use]
    pub fn is_orphaned(&self, node_id: u32) -> bool {
        self.index.is_orphaned(node_id)
    }

    /// Count orphaned nodes (nodes with no neighbors)
    ///
    /// Useful for monitoring graph health after deletions.
    #[must_use]
    pub fn count_orphaned(&self) -> usize {
        self.index.count_orphaned()
    }

    /// Validate graph connectivity after deletions
    ///
    /// Returns (reachable_count, orphan_count).
    #[must_use]
    pub fn validate_connectivity(&self) -> (usize, usize) {
        self.index.validate_connectivity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Builder Pattern Tests
    // ========================================================================

    #[test]
    fn test_builder_basic() {
        let index = HNSWIndex::builder().dimensions(128).build().unwrap();

        assert_eq!(index.dimensions(), 128);
        assert_eq!(index.params().max_nb_connection, DEFAULT_M);
        assert_eq!(index.params().ef_construction, DEFAULT_EF_CONSTRUCTION);
    }

    #[test]
    fn test_builder_custom_params() {
        let index = HNSWIndex::builder()
            .dimensions(64)
            .m(32)
            .ef_construction(200)
            .ef_search(300)
            .max_elements(50_000)
            .metric(DistanceFunction::Cosine)
            .build()
            .unwrap();

        let params = index.params();
        assert_eq!(params.dimensions, 64);
        assert_eq!(params.max_nb_connection, 32);
        assert_eq!(params.ef_construction, 200);
        assert_eq!(params.ef_search, 300);
        assert_eq!(params.max_elements, 50_000);
    }

    #[test]
    fn test_builder_sq8_quantization() {
        let index = HNSWIndex::builder()
            .dimensions(128)
            .quantization(HNSWQuantization::SQ8)
            .build()
            .unwrap();

        // SQ8 IS in the asymmetric path for better recall
        // (L2 decomposition in search_layer_mono causes ~10% recall regression)
        assert!(index.is_asymmetric());
        assert!(index.is_sq8());
    }

    #[test]
    fn test_builder_rabitq_quantization() {
        let index = HNSWIndex::builder()
            .dimensions(128)
            .quantization(HNSWQuantization::RaBitQ(RaBitQParams::default()))
            .build()
            .unwrap();

        assert!(index.is_asymmetric());
        assert!(!index.is_sq8());
    }

    #[test]
    fn test_builder_requires_dimensions() {
        let result = HNSWIndex::builder().build();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("dimensions is required"));
    }

    #[test]
    fn test_builder_with_training() {
        let training_data: Vec<Vec<f32>> = (0..100).map(|i| vec![(i as f32) / 100.0; 64]).collect();

        let index = HNSWIndex::builder()
            .dimensions(64)
            .quantization(HNSWQuantization::SQ8)
            .build_with_training(&training_data)
            .unwrap();

        // SQ8 IS in asymmetric path for better recall
        assert!(index.is_asymmetric());
        assert!(index.is_sq8());
        assert_eq!(index.dimensions(), 64);
    }

    // ========================================================================
    // Legacy Constructor Tests
    // ========================================================================

    #[test]
    fn test_hnsw_basic() {
        let mut index = HNSWIndex::new(1000, 4, DistanceFunction::L2).unwrap();

        // Insert vectors
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];

        let id1 = index.insert(&v1).unwrap();
        let id2 = index.insert(&v2).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(index.len(), 2);

        // Search
        let query = vec![0.9, 0.1, 0.0, 0.0];
        let results = index.search(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0); // Closest to v1
    }

    #[test]
    fn test_hnsw_batch_insert() {
        let mut index = HNSWIndex::new(1000, 3, DistanceFunction::L2).unwrap();

        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let ids = index.batch_insert(&vectors).unwrap();

        assert_eq!(ids.len(), 3);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_hnsw_ef_search() {
        let mut index = HNSWIndex::new(1000, 4, DistanceFunction::L2).unwrap();

        assert_eq!(index.get_ef_search(), 100); // Default for <10K: M=16, ef=100

        index.set_ef_search(600);
        assert_eq!(index.get_ef_search(), 600);
    }
}
