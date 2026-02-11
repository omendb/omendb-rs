//! Vector storage with HNSW indexing
//!
//! `VectorStore` manages a collection of vectors and provides k-NN search
//! using HNSW (Hierarchical Navigable Small World) algorithm.
//!
//! Optional SQ8 quantization for memory-efficient storage.
//!
//! Optional tantivy-based full-text search for hybrid (vector + BM25) retrieval.

mod accessors;
mod crud;
mod filter;
mod helpers;
mod input;
mod lifecycle;
mod multivec_ops;
mod options;
mod persistence;
mod record_store;
mod search;
mod text_search;
mod thread_safe;

pub use crate::omen::Metric;
pub use filter::MetadataFilter;
pub use input::{
    BatchItem, HybridParams, QueryData, QueryInput, Rerank, SearchOptions, SearchParams,
    VectorData, VectorInput,
};
pub use options::VectorStoreOptions;
pub use record_store::{Record, RecordStore};
pub use thread_safe::ThreadSafeVectorStore;

use super::hnsw::{HNSWParams, SegmentConfig, SegmentManager};
use super::muvera::{MultiVecStorage, MultiVectorConfig, MuveraEncoder};
use super::types::Vector;
use crate::omen::OmenFile;
use crate::text::{TextIndex, TextSearchConfig};
use crate::vector::metadata::MetadataIndex;
use anyhow::Result;
use rayon::prelude::*;
use serde_json::Value as JsonValue;
use std::path::PathBuf;

/// Default HNSW M parameter (neighbors per node)
const DEFAULT_HNSW_M: usize = 16;
/// Default HNSW ef_construction parameter (build quality)
const DEFAULT_HNSW_EF_CONSTRUCTION: usize = 100;
/// Default HNSW ef_search parameter (search quality)
const DEFAULT_HNSW_EF_SEARCH: usize = 100;

#[cfg(test)]
mod stress_tests;
#[cfg(test)]
mod tests;

/// Search result with user ID, distance, and metadata
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// User-provided document ID
    pub id: String,
    /// Distance from query (lower = more similar for L2)
    pub distance: f32,
    /// Document metadata
    pub metadata: JsonValue,
}

impl SearchResult {
    /// Create a new search result
    #[inline]
    pub fn new(id: String, distance: f32, metadata: JsonValue) -> Self {
        Self {
            id,
            distance,
            metadata,
        }
    }
}

/// Vector store with HNSW indexing
pub struct VectorStore {
    /// Single source of truth for records (vectors, IDs, deleted, metadata)
    records: RecordStore,

    /// Segment manager for HNSW index (mutable + frozen segments)
    pub(crate) segments: Option<SegmentManager>,

    /// Roaring bitmap index for fast filtered search
    metadata_index: MetadataIndex,

    /// Persistent storage backend (.omen format)
    storage: Option<OmenFile>,

    /// Storage path (for `TextIndex` subdirectory)
    storage_path: Option<PathBuf>,

    /// Optional tantivy text index for hybrid search
    text_index: Option<TextIndex>,

    /// Text search configuration (used by `enable_text_search`)
    text_search_config: Option<TextSearchConfig>,

    /// SQ8 quantization enabled (deferred until first insert for training)
    pending_quantization: bool,

    /// HNSW parameters for lazy initialization
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_ef_search: usize,

    /// Distance metric for similarity search (default: L2)
    distance_metric: Metric,

    /// MUVERA encoder for multi-vector to FDE transformation.
    /// Present when store is created with `new_muvera()`.
    muvera_encoder: Option<MuveraEncoder>,

    /// Storage for original multi-vector tokens (for MaxSim reranking).
    multivec_storage: Option<MultiVecStorage>,

    /// Maximum tokens per document (default: 512, matches ColBERT).
    max_tokens: usize,

    /// Override for segment capacity (vectors per segment before freezing).
    /// None = use SegmentConfig::DEFAULT_CAPACITY (100K).
    segment_capacity: Option<usize>,
}

/// Default maximum tokens per multi-vector document.
const DEFAULT_MAX_TOKENS: usize = 512;

impl VectorStore {
    /// Create base VectorStore with default field values.
    ///
    /// All optional fields are None, HNSW params are defaults.
    /// Used by public constructors to avoid field duplication.
    fn with_defaults(dimensions: usize, distance_metric: Metric) -> Self {
        Self {
            records: RecordStore::new(dimensions as u32),
            segments: None,
            metadata_index: MetadataIndex::new(),
            storage: None,
            storage_path: None,
            text_index: None,
            text_search_config: None,
            pending_quantization: false,
            hnsw_m: DEFAULT_HNSW_M,
            hnsw_ef_construction: DEFAULT_HNSW_EF_CONSTRUCTION,
            hnsw_ef_search: DEFAULT_HNSW_EF_SEARCH,
            distance_metric,
            muvera_encoder: None,
            multivec_storage: None,
            max_tokens: DEFAULT_MAX_TOKENS,
            segment_capacity: None,
        }
    }

    /// Create new vector store
    #[must_use]
    pub fn new(dimensions: usize) -> Self {
        Self::with_defaults(dimensions, Metric::L2)
    }

    /// Create a multi-vector store for ColBERT-style token embeddings.
    ///
    /// Multi-vector stores let you index documents as sets of token embeddings,
    /// enabling late-interaction retrieval patterns like ColBERT's MaxSim scoring.
    ///
    /// # Arguments
    ///
    /// * `token_dim` - Dimension of each token embedding (e.g., 128 for ColBERT)
    ///
    /// # Example
    ///
    /// ```rust
    /// use omendb::VectorStore;
    /// use serde_json::json;
    ///
    /// // Create store for 128-dimensional token embeddings
    /// let mut store = VectorStore::multi_vector(128);
    ///
    /// // Insert document with token embeddings
    /// let tokens = vec![vec![0.1; 128]; 10]; // 10 tokens
    /// store.store("doc1", tokens.clone(), json!({})).unwrap();
    ///
    /// // Search with query tokens
    /// let results = store.query_with_options(&tokens, 10, &Default::default()).unwrap();
    /// ```
    #[must_use]
    pub fn multi_vector(token_dim: usize) -> Self {
        Self::multi_vector_with(token_dim, MultiVectorConfig::default())
    }

    /// Create a multi-vector store with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `token_dim` - Dimension of each token embedding
    /// * `config` - Configuration controlling quality/size tradeoff
    ///
    /// # Example
    ///
    /// ```rust
    /// use omendb::{VectorStore, MultiVectorConfig};
    ///
    /// // High-quality configuration for production
    /// let store = VectorStore::multi_vector_with(128, MultiVectorConfig::quality());
    ///
    /// // Fast configuration for prototyping
    /// let store = VectorStore::multi_vector_with(128, MultiVectorConfig::fast());
    /// ```
    #[must_use]
    pub fn multi_vector_with(token_dim: usize, config: MultiVectorConfig) -> Self {
        let encoder = MuveraEncoder::new(token_dim, config);
        let fde_dim = encoder.fde_dimension();

        let mut store = Self::with_defaults(fde_dim, Metric::InnerProduct);
        store.muvera_encoder = Some(encoder);
        store.multivec_storage = Some(MultiVecStorage::new(token_dim));
        store
    }

    /// Get the dimensionality of vectors stored in this database.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.records.dimensions() as usize
    }

    /// Get the distance metric used by this store.
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.distance_metric
    }

    /// Create new vector store with SQ8 quantization
    ///
    /// Quantization is trained on the first batch of vectors inserted.
    #[must_use]
    pub fn new_with_quantization(dimensions: usize) -> Self {
        let mut store = Self::with_defaults(dimensions, Metric::L2);
        store.pending_quantization = true;
        store
    }

    /// Create new vector store with custom HNSW parameters
    ///
    /// Parameters are stored and applied when segments are created on first insert.
    #[must_use]
    pub fn new_with_params(
        dimensions: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        distance_metric: Metric,
    ) -> Self {
        let mut store = Self::with_defaults(dimensions, distance_metric);
        store.hnsw_m = m;
        store.hnsw_ef_construction = ef_construction;
        store.hnsw_ef_search = ef_search;
        store
    }

    /// Set segment capacity (vectors per segment before freezing)
    #[must_use]
    pub fn with_segment_capacity(mut self, capacity: usize) -> Self {
        self.segment_capacity = Some(capacity);
        self
    }

    /// Build a SegmentConfig from current store settings
    fn segment_config(&self, dimensions: usize) -> SegmentConfig {
        let mut config = SegmentConfig::new(dimensions)
            .with_params(HNSWParams {
                m: self.hnsw_m,
                ef_construction: self.hnsw_ef_construction,
                ..Default::default()
            })
            .with_distance(self.distance_metric)
            .with_quantization(self.pending_quantization);
        if let Some(cap) = self.segment_capacity {
            config = config.with_capacity(cap);
        }
        config
    }

    /// Resolve dimensions from vector or existing store config.
    fn resolve_dimensions(&self, vector_dim: usize) -> Result<usize> {
        if self.dimensions() == 0 {
            Ok(vector_dim)
        } else if vector_dim != self.dimensions() {
            anyhow::bail!(
                "Vector dimension mismatch: store expects {}, got {}",
                self.dimensions(),
                vector_dim
            );
        } else {
            Ok(self.dimensions())
        }
    }

    /// K-nearest neighbors search using HNSW
    pub(crate) fn knn_search(&self, query: &Vector, k: usize) -> Result<Vec<(usize, f32)>> {
        self.knn_search_with_ef(query, k, None)
    }

    /// K-nearest neighbors search with optional ef override
    #[inline]
    pub(crate) fn knn_search_with_ef(
        &self,
        query: &Vector,
        k: usize,
        ef: Option<usize>,
    ) -> Result<Vec<(usize, f32)>> {
        let effective_ef = helpers::compute_effective_ef(ef, self.hnsw_ef_search, k);
        self.knn_search_ef(query, k, effective_ef)
    }

    /// K-nearest neighbors search with concrete ef value
    #[inline]
    pub(crate) fn knn_search_ef(
        &self,
        query: &Vector,
        k: usize,
        ef: usize,
    ) -> Result<Vec<(usize, f32)>> {
        if query.dim() != self.dimensions() {
            anyhow::bail!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions(),
                query.dim()
            );
        }

        search::knn_search_core(
            &self.records,
            self.segments.as_ref(),
            &query.data,
            k,
            ef,
            self.distance_metric,
        )
    }

    /// K-nearest neighbors search with metadata filtering
    pub(crate) fn knn_search_with_filter(
        &self,
        query: &Vector,
        k: usize,
        filter: &MetadataFilter,
    ) -> Result<Vec<SearchResult>> {
        self.knn_search_with_filter_ef(query, k, filter, None)
    }

    /// Filtered K-nearest neighbors search with optional ef override
    ///
    /// Uses Roaring bitmap index for O(1) filter evaluation when possible,
    /// falls back to JSON-based filtering for complex filters.
    pub(crate) fn knn_search_with_filter_ef(
        &self,
        query: &Vector,
        k: usize,
        filter: &MetadataFilter,
        ef: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        let effective_ef = helpers::compute_effective_ef(ef, self.hnsw_ef_search, k);

        search::knn_search_filtered_core(
            &self.records,
            &self.metadata_index,
            self.segments.as_ref(),
            &query.data,
            k,
            effective_ef,
            filter,
            self.distance_metric,
        )
    }

    /// Search with optional filter (convenience method)
    pub fn search(
        &self,
        query: &Vector,
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>> {
        self.search_with_options(query, k, filter, None, None)
    }

    /// Search with all options: filter, ef override, and max_distance
    pub fn search_with_options(
        &self,
        query: &Vector,
        k: usize,
        filter: Option<&MetadataFilter>,
        ef: Option<usize>,
        max_distance: Option<f32>,
    ) -> Result<Vec<SearchResult>> {
        let mut results = if let Some(f) = filter {
            self.knn_search_with_filter_ef(query, k, f, ef)?
        } else {
            let slot_results = self.knn_search_with_ef(query, k, ef)?;
            search::slots_to_results_with_fallback(
                &self.records,
                slot_results,
                &query.data,
                k,
                self.distance_metric,
            )
        };

        if let Some(max_dist) = max_distance {
            results.retain(|r| r.distance <= max_dist);
        }

        Ok(results)
    }

    /// Search with SearchOptions struct
    pub fn search_with_params(
        &self,
        query: &Vector,
        k: usize,
        params: &SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        self.search_with_options(
            query,
            k,
            params.filter.as_ref(),
            params.ef,
            params.max_distance,
        )
    }

    /// Parallel batch search with metadata
    #[must_use]
    pub fn search_batch_with_metadata(
        &self,
        queries: &[Vector],
        k: usize,
        ef: Option<usize>,
    ) -> Vec<Result<Vec<SearchResult>>> {
        queries
            .par_iter()
            .map(|q| self.search_with_options(q, k, None, ef, None))
            .collect()
    }

    /// Parallel batch search with a pre-parsed filter.
    ///
    /// The filter is parsed once and shared across all queries in the batch,
    /// avoiding redundant parsing per query.
    #[must_use]
    pub fn search_batch_with_filter(
        &self,
        queries: &[Vector],
        k: usize,
        filter: Option<&MetadataFilter>,
        ef: Option<usize>,
        max_distance: Option<f32>,
    ) -> Vec<Result<Vec<SearchResult>>> {
        queries
            .par_iter()
            .map(|q| self.search_with_options(q, k, filter, ef, max_distance))
            .collect()
    }

    /// Brute-force K-NN search (fallback, used in tests)
    #[allow(dead_code)]
    pub(crate) fn knn_search_brute_force(
        &self,
        query: &Vector,
        k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        if query.dim() != self.dimensions() {
            anyhow::bail!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions(),
                query.dim()
            );
        }

        Ok(search::brute_force_search(
            &self.records,
            &query.data,
            k,
            self.distance_metric,
        ))
    }
}
