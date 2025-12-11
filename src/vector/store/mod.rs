//! Vector storage with HNSW indexing
//!
//! `VectorStore` manages a collection of vectors and provides k-NN search
//! using HNSW (Hierarchical Navigable Small World) algorithm.
//!
//! Optional Extended `RaBitQ` quantization for memory-efficient storage.
//!
//! Optional tantivy-based full-text search for hybrid (vector + BM25) retrieval.

use super::hnsw::{DistanceFunction, HNSWParams};
use super::hnsw_index::HNSWIndex;
use super::storage::SeerDBStorage;
use super::types::Vector;
use super::QuantizationMode;
use crate::text::{weighted_reciprocal_rank_fusion, TextIndex, TextSearchConfig, DEFAULT_RRF_K};
use anyhow::Result;
use omendb_core::compression::RaBitQParams;
use omendb_core::distance::l2_distance;
use rayon::prelude::*;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(test)]
mod tests;

// ============================================================================
// VectorStoreOptions - Builder pattern for VectorStore configuration
// ============================================================================

/// Configuration options for opening or creating a vector store.
///
/// Follows the `std::fs::OpenOptions` pattern for familiar, ergonomic API.
///
/// # Examples
///
/// ```rust,no_run
/// use omendb::vector::store::VectorStoreOptions;
///
/// // Simple persistent store
/// let store = VectorStoreOptions::default()
///     .dimensions(384)
///     .open("./vectors")?;
///
/// // With custom HNSW parameters
/// let store = VectorStoreOptions::default()
///     .dimensions(384)
///     .m(32)
///     .ef_construction(400)
///     .ef_search(100)
///     .open("./vectors")?;
///
/// // In-memory store
/// let store = VectorStoreOptions::default()
///     .dimensions(384)
///     .build()?;
/// # Ok::<(), anyhow::Error>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct VectorStoreOptions {
    /// Vector dimensionality (0 = infer from first insert or existing data)
    dimensions: usize,

    /// HNSW M parameter: neighbors per node (default: 16)
    m: Option<usize>,

    /// HNSW `ef_construction`: build quality (default: 100)
    ef_construction: Option<usize>,

    /// HNSW `ef_search`: search quality/speed tradeoff (default: 100)
    ef_search: Option<usize>,

    /// Quantization mode (SQ8 or RaBitQ for asymmetric HNSW search)
    quantization: Option<QuantizationMode>,

    /// Rescore candidates with original vectors (default: true when quantization enabled)
    /// When true, search fetches `k * oversample` candidates using quantized distance,
    /// then reranks with full precision distance for final k results.
    rescore: Option<bool>,

    /// Oversampling factor for rescore (default: 3.0)
    /// Fetches `k * oversample` candidates during quantized search.
    oversample: Option<f32>,

    /// Text search configuration (None = disabled)
    text_search_config: Option<TextSearchConfig>,
}

impl VectorStoreOptions {
    /// Create new options with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set vector dimensionality.
    ///
    /// If not set, dimensions will be inferred from:
    /// 1. Existing data when opening a persistent store
    /// 2. First inserted vector
    #[must_use]
    pub fn dimensions(mut self, dim: usize) -> Self {
        self.dimensions = dim;
        self
    }

    /// Set HNSW M parameter (neighbors per node).
    ///
    /// Higher M = better recall, more memory. Range: 4-64, default: 16.
    #[must_use]
    pub fn m(mut self, m: usize) -> Self {
        self.m = Some(m);
        self
    }

    /// Set HNSW `ef_construction` (build quality).
    ///
    /// Higher = better graph quality, slower build. Default: 100.
    #[must_use]
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = Some(ef);
        self
    }

    /// Set HNSW `ef_search` (search quality/speed tradeoff).
    ///
    /// Higher = better recall, slower search. Default: 100.
    #[must_use]
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        self
    }

    /// Enable quantization for memory-efficient storage.
    ///
    /// # Modes
    /// - `QuantizationMode::SQ8`: 4x compression, ~2x faster search, ~99% recall (default)
    /// - `QuantizationMode::RaBitQ(params)`: 4-16x compression, ~0.5x slower, 93-99% recall
    ///
    /// # Example
    /// ```ignore
    /// // SQ8 (recommended for most cases)
    /// let store = VectorStoreOptions::default()
    ///     .dimensions(768)
    ///     .quantization(QuantizationMode::sq8())
    ///     .open("./vectors")?;
    ///
    /// // RaBitQ for higher compression
    /// let store = VectorStoreOptions::default()
    ///     .dimensions(768)
    ///     .quantization(QuantizationMode::rabitq())
    ///     .open("./vectors")?;
    /// ```
    #[must_use]
    pub fn quantization(mut self, mode: QuantizationMode) -> Self {
        self.quantization = Some(mode);
        self
    }

    /// Enable SQ8 quantization (4x compression, ~2x faster)
    ///
    /// Convenience method for the most common quantization mode.
    #[must_use]
    pub fn quantization_sq8(self) -> Self {
        self.quantization(QuantizationMode::SQ8)
    }

    /// Enable RaBitQ quantization with default 4-bit parameters (8x compression)
    #[must_use]
    pub fn quantization_rabitq(self) -> Self {
        self.quantization(QuantizationMode::rabitq())
    }

    /// Enable RaBitQ quantization with custom parameters
    #[must_use]
    pub fn quantization_rabitq_params(self, params: RaBitQParams) -> Self {
        self.quantization(QuantizationMode::RaBitQ(params))
    }

    /// Enable/disable rescoring with original vectors (default: true when quantization enabled).
    ///
    /// When rescoring is enabled, search uses quantized vectors for fast candidate selection,
    /// then reranks candidates using full-precision vectors for accuracy.
    ///
    /// # Arguments
    /// * `enable` - Whether to rescore candidates
    #[must_use]
    pub fn rescore(mut self, enable: bool) -> Self {
        self.rescore = Some(enable);
        self
    }

    /// Set oversampling factor for rescoring (default: 3.0).
    ///
    /// When rescoring, fetches `k * oversample` candidates during quantized search,
    /// then returns top k after reranking with full precision.
    ///
    /// Higher values improve recall but increase latency.
    ///
    /// # Arguments
    /// * `factor` - Oversampling multiplier (must be >= 1.0)
    #[must_use]
    pub fn oversample(mut self, factor: f32) -> Self {
        self.oversample = Some(factor.max(1.0));
        self
    }

    /// Enable tantivy-based full-text search with default configuration.
    ///
    /// When enabled, you can use `set_with_text()` to index text alongside vectors,
    /// and `hybrid_search()` to search both with RRF fusion.
    ///
    /// Uses 50MB writer buffer by default. For custom memory settings,
    /// use `text_search_config()` instead.
    #[must_use]
    pub fn text_search(mut self, enabled: bool) -> Self {
        self.text_search_config = if enabled {
            Some(TextSearchConfig::default())
        } else {
            None
        };
        self
    }

    /// Enable text search with custom configuration.
    ///
    /// # Example
    /// ```ignore
    /// // Mobile: lower memory
    /// let store = VectorStoreOptions::default()
    ///     .text_search_config(TextSearchConfig { writer_buffer_mb: 15 })
    ///     .open("./db")?;
    ///
    /// // Cloud: higher throughput
    /// let store = VectorStoreOptions::default()
    ///     .text_search_config(TextSearchConfig { writer_buffer_mb: 200 })
    ///     .open("./db")?;
    /// ```
    #[must_use]
    pub fn text_search_config(mut self, config: TextSearchConfig) -> Self {
        self.text_search_config = Some(config);
        self
    }

    /// Open or create a persistent vector store at the given path.
    ///
    /// Creates the directory if it doesn't exist.
    /// Loads existing data if the store already exists.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<VectorStore> {
        VectorStore::open_with_options(path, self)
    }

    /// Build an in-memory vector store (no persistence).
    pub fn build(&self) -> Result<VectorStore> {
        VectorStore::build_with_options(self)
    }
}

/// Metadata filter for vector search (MongoDB-style operators)
#[derive(Debug, Clone)]
pub enum MetadataFilter {
    /// Equality: field == value
    Eq(String, JsonValue),
    /// Not equal: field != value
    Ne(String, JsonValue),
    /// Greater than or equal: field >= value
    Gte(String, f64),
    /// Less than: field < value
    Lt(String, f64),
    /// Greater than: field > value
    Gt(String, f64),
    /// Less than or equal: field <= value
    Lte(String, f64),
    /// In list: field in [values]
    In(String, Vec<JsonValue>),
    /// Contains substring: field.contains(value)
    Contains(String, String),
    /// Logical AND: all filters must match
    And(Vec<MetadataFilter>),
    /// Logical OR: at least one filter must match
    Or(Vec<MetadataFilter>),
}

impl MetadataFilter {
    /// Evaluate filter against metadata
    #[must_use]
    pub fn matches(&self, metadata: &JsonValue) -> bool {
        match self {
            MetadataFilter::Eq(field, value) => metadata.get(field) == Some(value),
            MetadataFilter::Ne(field, value) => metadata.get(field) != Some(value),
            MetadataFilter::Gte(field, threshold) => metadata
                .get(field)
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|v| v >= *threshold),
            MetadataFilter::Lt(field, threshold) => metadata
                .get(field)
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|v| v < *threshold),
            MetadataFilter::Gt(field, threshold) => metadata
                .get(field)
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|v| v > *threshold),
            MetadataFilter::Lte(field, threshold) => metadata
                .get(field)
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|v| v <= *threshold),
            MetadataFilter::In(field, values) => {
                metadata.get(field).is_some_and(|v| values.contains(v))
            }
            MetadataFilter::Contains(field, substring) => metadata
                .get(field)
                .and_then(|v| v.as_str())
                .is_some_and(|s| s.contains(substring)),
            MetadataFilter::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            MetadataFilter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
        }
    }
}

/// Vector store with HNSW indexing
pub struct VectorStore {
    /// All vectors stored in memory (used for rescore when quantization enabled)
    pub vectors: Vec<Vector>,

    /// HNSW index for approximate nearest neighbor search
    pub hnsw_index: Option<HNSWIndex>,

    /// Vector dimensionality
    dimensions: usize,

    /// Whether to rescore candidates with original vectors (default: true when quantization enabled)
    rescore_enabled: bool,

    /// Oversampling factor for rescore (default: 3.0)
    oversample_factor: f32,

    /// Metadata storage (indexed by internal vector ID)
    metadata: HashMap<usize, JsonValue>,

    /// Map from string IDs to internal indices (public for Python bindings)
    pub id_to_index: HashMap<String, usize>,

    /// Reverse map from internal indices to string IDs (O(1) lookup for search results)
    index_to_id: HashMap<usize, String>,

    /// Deleted vector IDs (tombstones for MVCC)
    deleted: HashMap<usize, bool>,

    /// Persistent storage backend (seerdb LSM)
    storage: Option<SeerDBStorage>,

    /// Storage path (for `TextIndex` subdirectory)
    storage_path: Option<PathBuf>,

    /// Optional tantivy text index for hybrid search
    text_index: Option<TextIndex>,

    /// Text search configuration (used by `enable_text_search`)
    text_search_config: Option<TextSearchConfig>,

    /// Pending quantization mode (deferred until first insert for training)
    pending_quantization: Option<QuantizationMode>,

    /// HNSW parameters for lazy initialization
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_ef_search: usize,
}

impl VectorStore {
    /// Create new vector store
    #[must_use]
    pub fn new(dimensions: usize) -> Self {
        Self {
            vectors: Vec::new(),
            hnsw_index: None,
            dimensions,
            rescore_enabled: false,
            oversample_factor: 3.0,
            metadata: HashMap::new(),
            id_to_index: HashMap::new(),
            index_to_id: HashMap::new(),
            deleted: HashMap::new(),
            storage: None,
            storage_path: None,
            text_index: None,
            text_search_config: None,
            pending_quantization: None,
            hnsw_m: 16,
            hnsw_ef_construction: 100,
            hnsw_ef_search: 100,
        }
    }

    /// Create new vector store with quantization
    ///
    /// Quantization is trained on the first batch of vectors inserted.
    #[must_use]
    pub fn new_with_quantization(dimensions: usize, mode: QuantizationMode) -> Self {
        Self {
            vectors: Vec::new(),
            hnsw_index: None,
            dimensions,
            rescore_enabled: true,
            oversample_factor: 3.0,
            metadata: HashMap::new(),
            id_to_index: HashMap::new(),
            index_to_id: HashMap::new(),
            deleted: HashMap::new(),
            storage: None,
            storage_path: None,
            text_index: None,
            text_search_config: None,
            pending_quantization: Some(mode),
            hnsw_m: 16,
            hnsw_ef_construction: 100,
            hnsw_ef_search: 100,
        }
    }

    /// Create new vector store with custom HNSW parameters
    pub fn new_with_params(
        dimensions: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> Result<Self> {
        let hnsw_index = Some(HNSWIndex::new_with_params(
            1_000_000,
            dimensions,
            m,
            ef_construction,
            ef_search,
        )?);

        Ok(Self {
            vectors: Vec::new(),
            hnsw_index,
            dimensions,
            rescore_enabled: false,
            oversample_factor: 3.0,
            metadata: HashMap::new(),
            id_to_index: HashMap::new(),
            index_to_id: HashMap::new(),
            deleted: HashMap::new(),
            storage: None,
            storage_path: None,
            text_index: None,
            text_search_config: None,
            pending_quantization: None,
            hnsw_m: m,
            hnsw_ef_construction: ef_construction,
            hnsw_ef_search: ef_search,
        })
    }

    /// Open a persistent vector store at the given path
    ///
    /// Creates a new database if it doesn't exist, or loads existing data.
    /// All operations (insert, set, delete) are automatically persisted.
    ///
    /// # Arguments
    /// * `path` - Directory path for the database (e.g., "mydb.oadb")
    ///
    /// # Example
    /// ```ignore
    /// let mut store = VectorStore::open("mydb.oadb")?;
    /// store.set("doc1".to_string(), vector, metadata)?;
    /// // Data is automatically persisted
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let storage = SeerDBStorage::open(path)?;

        // Check if store was quantized - if so, skip loading vectors to RAM
        // (use seerdb for rescore instead)
        let is_quantized = storage.is_quantized()?;

        // Load metadata and mappings (always needed)
        let metadata = storage.load_all_metadata()?;
        let id_to_index = storage.load_all_id_mappings()?;
        let deleted = storage.load_all_deleted()?;

        // Get dimensions from config
        let dimensions = storage.get_config("dimensions")?.unwrap_or(0) as usize;

        // Load vectors to RAM only if NOT quantized
        // When quantized, use seerdb for rescore (Phase 1 change)
        let (vectors, real_indices) = if is_quantized {
            // Skip loading vectors to RAM - use seerdb for rescore
            (Vec::new(), std::collections::HashSet::new())
        } else {
            // Non-quantized: load vectors to RAM for HNSW
            let vectors_data = storage.load_all_vectors()?;
            let mut vectors: Vec<Vector> = Vec::new();
            let mut real_indices: std::collections::HashSet<usize> =
                std::collections::HashSet::new();

            for (id, data) in &vectors_data {
                // Fill gaps with placeholder vectors (will be marked as deleted)
                while vectors.len() < *id {
                    vectors.push(Vector::new(vec![0.0; dimensions.max(1)]));
                }
                vectors.push(Vector::new(data.clone()));
                real_indices.insert(*id);
            }
            (vectors, real_indices)
        };

        // Mark gap-filled vectors as deleted (they're placeholders, not real data)
        // This ensures they're filtered out during search
        let mut deleted = deleted; // Make mutable
        for idx in 0..vectors.len() {
            if !real_indices.contains(&idx) && !deleted.contains_key(&idx) {
                deleted.insert(idx, true);
            }
        }

        // Build HNSW index
        // NOTE: When reopening a quantized store, quantization is LOST because
        // we rebuild HNSW without the quantization settings. This is a limitation
        // that will be addressed by persisting HNSW index to disk.
        let hnsw_index = if !is_quantized && !vectors.is_empty() {
            // Non-quantized: build from vectors
            let mut index = HNSWIndex::new(vectors.len().max(10_000), dimensions)?;
            let vector_data: Vec<Vec<f32>> = vectors.iter().map(|v| v.data.clone()).collect();
            index.batch_insert(&vector_data)?;
            Some(index)
        } else if is_quantized && dimensions > 0 {
            // Quantized store reopened: load vectors from seerdb and rebuild
            // TODO: Persist HNSW index for proper quantization preservation
            let vectors_data = storage.load_all_vectors()?;
            if !vectors_data.is_empty() {
                let mut index = HNSWIndex::new(vectors_data.len().max(10_000), dimensions)?;
                let vector_data: Vec<Vec<f32>> =
                    vectors_data.iter().map(|(_, v)| v.clone()).collect();
                index.batch_insert(&vector_data)?;
                Some(index)
            } else {
                None
            }
        } else {
            None
        };

        // Try to open existing text index if it exists
        let text_index_path = path.join("text_index");
        let text_index = if text_index_path.exists() {
            Some(TextIndex::open(&text_index_path)?)
        } else {
            None
        };

        // Build reverse map for O(1) index→id lookup
        let index_to_id: HashMap<usize, String> = id_to_index
            .iter()
            .map(|(id, &idx)| (idx, id.clone()))
            .collect();

        Ok(Self {
            vectors,
            hnsw_index,
            dimensions,
            rescore_enabled: false,
            oversample_factor: 3.0,
            metadata,
            id_to_index,
            index_to_id,
            deleted,
            storage: Some(storage),
            storage_path: Some(path.to_path_buf()),
            text_index,
            text_search_config: None,
            pending_quantization: None,
            hnsw_m: 16,
            hnsw_ef_construction: 100,
            hnsw_ef_search: 100,
        })
    }

    /// Open a persistent vector store with specified dimensions
    ///
    /// Like `open()` but ensures dimensions are set for new databases.
    pub fn open_with_dimensions(path: impl AsRef<Path>, dimensions: usize) -> Result<Self> {
        let mut store = Self::open(path)?;
        if store.dimensions == 0 {
            store.dimensions = dimensions;
            if let Some(ref storage) = store.storage {
                storage.put_config("dimensions", dimensions as u64)?;
            }
        }
        Ok(store)
    }

    /// Open a persistent vector store with custom options.
    ///
    /// This is the internal implementation used by `VectorStoreOptions::open()`.
    pub fn open_with_options(path: impl AsRef<Path>, options: &VectorStoreOptions) -> Result<Self> {
        let path = path.as_ref();

        // If path exists, load existing data
        if path.exists() {
            let mut store = Self::open(path)?;

            // Apply dimension if specified and store has none
            if store.dimensions == 0 && options.dimensions > 0 {
                store.dimensions = options.dimensions;
                if let Some(ref storage) = store.storage {
                    storage.put_config("dimensions", options.dimensions as u64)?;
                }
            }

            // Apply ef_search if specified
            if let Some(ef) = options.ef_search {
                store.set_ef_search(ef);
            }

            return Ok(store);
        }

        // Create new persistent store with options
        let storage = SeerDBStorage::open(path)?;
        let dimensions = options.dimensions;

        // Determine HNSW parameters
        let m = options.m.unwrap_or(16);
        let ef_construction = options.ef_construction.unwrap_or(100);
        let ef_search = options.ef_search.unwrap_or(100);

        // Initialize HNSW - defer when quantization enabled (need vectors to train)
        // When quantization is enabled, we defer HNSW creation to set_batch()
        // so we can train the quantizer from actual vectors first
        let (hnsw_index, pending_quantization) = if options.quantization.is_some() {
            // Defer HNSW creation - will be created on first insert with trained quantizer
            (None, options.quantization.clone())
        } else if dimensions > 0 {
            if options.m.is_some() || options.ef_construction.is_some() {
                (
                    Some(HNSWIndex::new_with_params(
                        10_000,
                        dimensions,
                        m,
                        ef_construction,
                        ef_search,
                    )?),
                    None,
                )
            } else {
                (None, None) // Will be lazily initialized
            }
        } else {
            (None, None)
        };

        // Save dimensions to storage if set
        if dimensions > 0 {
            storage.put_config("dimensions", dimensions as u64)?;
        }

        // Initialize text index if enabled
        let text_index = if let Some(ref config) = options.text_search_config {
            let text_path = path.join("text_index");
            Some(TextIndex::open_with_config(&text_path, config)?)
        } else {
            None
        };

        // Determine rescore settings
        let rescore_enabled = options.rescore.unwrap_or(options.quantization.is_some());
        let oversample_factor = options.oversample.unwrap_or(3.0);

        Ok(Self {
            vectors: Vec::new(),
            hnsw_index,
            dimensions,
            rescore_enabled,
            oversample_factor,
            metadata: HashMap::new(),
            id_to_index: HashMap::new(),
            index_to_id: HashMap::new(),
            deleted: HashMap::new(),
            storage: Some(storage),
            storage_path: Some(path.to_path_buf()),
            text_index,
            text_search_config: options.text_search_config.clone(),
            pending_quantization,
            hnsw_m: m,
            hnsw_ef_construction: ef_construction,
            hnsw_ef_search: ef_search,
        })
    }

    /// Build an in-memory vector store with custom options.
    pub fn build_with_options(options: &VectorStoreOptions) -> Result<Self> {
        let dimensions = options.dimensions;

        // Determine HNSW parameters
        let m = options.m.unwrap_or(16);
        let ef_construction = options.ef_construction.unwrap_or(100);
        let ef_search = options.ef_search.unwrap_or(100);

        // Initialize HNSW - defer when quantization enabled (need vectors to train)
        let (hnsw_index, pending_quantization) = if options.quantization.is_some() {
            (None, options.quantization.clone())
        } else if dimensions > 0 {
            if options.m.is_some() || options.ef_construction.is_some() {
                (
                    Some(HNSWIndex::new_with_params(
                        10_000,
                        dimensions,
                        m,
                        ef_construction,
                        ef_search,
                    )?),
                    None,
                )
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // Initialize in-memory text index if enabled
        let text_index = if let Some(ref config) = options.text_search_config {
            Some(TextIndex::open_in_memory_with_config(config)?)
        } else {
            None
        };

        // Determine rescore settings
        let rescore_enabled = options.rescore.unwrap_or(options.quantization.is_some());
        let oversample_factor = options.oversample.unwrap_or(3.0);

        Ok(Self {
            vectors: Vec::new(),
            hnsw_index,
            dimensions,
            rescore_enabled,
            oversample_factor,
            metadata: HashMap::new(),
            id_to_index: HashMap::new(),
            index_to_id: HashMap::new(),
            deleted: HashMap::new(),
            storage: None,
            storage_path: None,
            text_index,
            text_search_config: options.text_search_config.clone(),
            pending_quantization,
            hnsw_m: m,
            hnsw_ef_construction: ef_construction,
            hnsw_ef_search: ef_search,
        })
    }

    /// Insert vector and return its ID
    pub fn insert(&mut self, vector: Vector) -> Result<usize> {
        let id = self.vectors.len();

        // Lazy initialize HNSW on first insert
        if self.hnsw_index.is_none() {
            let dimensions = if self.dimensions == 0 {
                vector.dim()
            } else {
                if vector.dim() != self.dimensions {
                    anyhow::bail!(
                        "Vector dimension mismatch: store expects {}, got {}",
                        self.dimensions,
                        vector.dim()
                    );
                }
                self.dimensions
            };

            // Check if we have pending quantization
            if let Some(quant_mode) = self.pending_quantization.take() {
                let hnsw_params = HNSWParams::default()
                    .with_m(self.hnsw_m)
                    .with_ef_construction(self.hnsw_ef_construction)
                    .with_ef_search(self.hnsw_ef_search);

                // Save quantization mode to storage for persistence
                // Mode values: 0=none, 1=sq8, 2=rabitq-4, 3=rabitq-2, 4=rabitq-8
                let quant_mode_id = match &quant_mode {
                    QuantizationMode::SQ8 => 1u64,
                    QuantizationMode::RaBitQ(p) => match p.bits_per_dim.to_u8() {
                        2 => 3u64, // rabitq-2
                        8 => 4u64, // rabitq-8
                        _ => 2u64, // default rabitq-4
                    },
                };
                if let Some(ref storage) = self.storage {
                    storage.put_quantization_mode(quant_mode_id)?;
                }

                let index = match quant_mode {
                    QuantizationMode::SQ8 => {
                        // SQ8 trains lazily on first 256 vectors
                        HNSWIndex::new_with_sq8(dimensions, hnsw_params, DistanceFunction::L2)?
                    }
                    QuantizationMode::RaBitQ(params) => {
                        // RaBitQ needs explicit training
                        let mut idx = HNSWIndex::new_with_asymmetric(
                            dimensions,
                            hnsw_params,
                            DistanceFunction::L2,
                            params,
                        )?;
                        idx.train_quantizer(&[vector.data.clone()])?;
                        idx
                    }
                };
                self.hnsw_index = Some(index);
            } else {
                self.hnsw_index = Some(HNSWIndex::new(10_000, dimensions)?);
            }
            self.dimensions = dimensions;
        } else {
            // Validate dimension matches existing HNSW index
            // NOTE: HNSW requires all vectors to have same dimension
            if vector.dim() != self.dimensions {
                anyhow::bail!(
                    "Vector dimension mismatch: store expects {}, got {}. All vectors in same store must have same dimension.",
                    self.dimensions,
                    vector.dim()
                );
            }
        }

        // Insert into HNSW index
        if let Some(ref mut index) = self.hnsw_index {
            index.insert(&vector.data)?;
        }

        // Persist to storage if available
        if let Some(ref storage) = self.storage {
            storage.put_vector(id, &vector.data)?;
            storage.increment_count()?;
            // Save dimensions on first insert
            if id == 0 {
                storage.put_config("dimensions", self.dimensions as u64)?;
            }
        }

        self.vectors.push(vector);
        Ok(id)
    }

    /// Insert vector with string ID and metadata
    ///
    /// This is the primary method for inserting vectors with metadata support.
    /// Returns error if ID already exists (use set for insert-or-update semantics).
    pub fn insert_with_metadata(
        &mut self,
        id: String,
        vector: Vector,
        metadata: JsonValue,
    ) -> Result<usize> {
        // Check if ID already exists
        if self.id_to_index.contains_key(&id) {
            anyhow::bail!("Vector with ID '{id}' already exists. Use set() to update.");
        }

        // Insert vector using existing insert method
        let index = self.insert(vector)?;

        // Store metadata and ID mapping
        self.metadata.insert(index, metadata.clone());
        self.id_to_index.insert(id.clone(), index);
        self.index_to_id.insert(index, id.clone());

        // Persist to storage if available
        if let Some(ref storage) = self.storage {
            storage.put_metadata(index, &metadata)?;
            storage.put_id_mapping(&id, index)?;
        }

        Ok(index)
    }

    /// Upsert vector (insert or update) with string ID and metadata
    ///
    /// This is the recommended method for most use cases.
    /// If the ID exists, updates the vector and metadata.
    /// If the ID doesn't exist, inserts a new vector.
    pub fn set(&mut self, id: String, vector: Vector, metadata: JsonValue) -> Result<usize> {
        // Check if ID already exists
        if let Some(&index) = self.id_to_index.get(&id) {
            // Update existing vector
            self.update_by_index(index, Some(vector), Some(metadata))?;
            Ok(index)
        } else {
            // Insert new vector
            self.insert_with_metadata(id, vector, metadata)
        }
    }

    /// Batch set vectors (insert or update multiple vectors at once)
    ///
    /// This is the recommended method for bulk operations. It provides significant
    /// performance improvements over calling `set()` repeatedly by:
    /// - Reducing function call overhead
    /// - Batching HNSW insertions
    /// - Amortizing metadata operations
    ///
    /// # Arguments
    /// * `batch` - Vector of (id, vector, metadata) tuples
    ///
    /// # Returns
    /// Vector of indices for all set vectors
    ///
    /// # Example
    /// ```ignore
    /// let batch = vec![
    ///     ("vec1".to_string(), Vector::new(vec![0.1, 0.2]), json!({"key": "value"})),
    ///     ("vec2".to_string(), Vector::new(vec![0.3, 0.4]), json!({"key": "value2"})),
    /// ];
    /// let indices = store.set_batch(batch)?;
    /// ```
    pub fn set_batch(&mut self, batch: Vec<(String, Vector, JsonValue)>) -> Result<Vec<usize>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        // Separate batch into updates and inserts
        let mut updates: Vec<(usize, Vector, JsonValue)> = Vec::new();
        let mut inserts: Vec<(String, Vector, JsonValue)> = Vec::new();

        for (id, vector, metadata) in batch {
            if let Some(&index) = self.id_to_index.get(&id) {
                // Existing vector - queue for update
                updates.push((index, vector, metadata));
            } else {
                // New vector - queue for insert
                inserts.push((id, vector, metadata));
            }
        }

        let mut result_indices = Vec::new();

        // Process updates first (modify in-place)
        for (index, vector, metadata) in updates {
            self.update_by_index(index, Some(vector), Some(metadata))?;
            result_indices.push(index);
        }

        // Process inserts in batch
        if !inserts.is_empty() {
            // Lazy initialize HNSW if needed
            if self.hnsw_index.is_none() {
                let dimensions = if self.dimensions == 0 {
                    inserts[0].1.dim()
                } else {
                    self.dimensions
                };

                // Check if we have pending quantization to train
                if let Some(quant_mode) = self.pending_quantization.take() {
                    let hnsw_params = HNSWParams::default()
                        .with_m(self.hnsw_m)
                        .with_ef_construction(self.hnsw_ef_construction)
                        .with_ef_search(self.hnsw_ef_search);

                    // Save quantization mode to storage for persistence
                    // Mode values: 0=none, 1=sq8, 2=rabitq-4, 3=rabitq-2, 4=rabitq-8
                    let quant_mode_id = match &quant_mode {
                        QuantizationMode::SQ8 => 1u64,
                        QuantizationMode::RaBitQ(p) => match p.bits_per_dim.to_u8() {
                            2 => 3u64, // rabitq-2
                            8 => 4u64, // rabitq-8
                            _ => 2u64, // default rabitq-4
                        },
                    };
                    if let Some(ref storage) = self.storage {
                        storage.put_quantization_mode(quant_mode_id)?;
                    }

                    let index = match quant_mode {
                        QuantizationMode::SQ8 => {
                            // SQ8 trains lazily on first 256 vectors
                            HNSWIndex::new_with_sq8(dimensions, hnsw_params, DistanceFunction::L2)?
                        }
                        QuantizationMode::RaBitQ(params) => {
                            // RaBitQ needs explicit training from first batch
                            let mut idx = HNSWIndex::new_with_asymmetric(
                                dimensions,
                                hnsw_params,
                                DistanceFunction::L2,
                                params,
                            )?;
                            let training_vectors: Vec<Vec<f32>> =
                                inserts.iter().map(|(_, v, _)| v.data.clone()).collect();
                            idx.train_quantizer(&training_vectors)?;
                            idx
                        }
                    };

                    self.hnsw_index = Some(index);
                } else {
                    // Standard HNSW without quantization
                    self.hnsw_index = Some(HNSWIndex::new(10_000, dimensions)?);
                }
                self.dimensions = dimensions;
            }

            // Validate all vectors have same dimensions
            for (i, (_, vector, _)) in inserts.iter().enumerate() {
                if vector.dim() != self.dimensions {
                    anyhow::bail!(
                        "Vector {} dimension mismatch: expected {}, got {}",
                        i,
                        self.dimensions,
                        vector.dim()
                    );
                }
            }

            // Extract vectors for batch HNSW insertion
            let vectors_data: Vec<Vec<f32>> =
                inserts.iter().map(|(_, v, _)| v.data.clone()).collect();

            // Sequential insert: parallel batch_insert searches incomplete graphs during construction
            let base_index = self.vectors.len();
            if let Some(ref mut index) = self.hnsw_index {
                for vector in &vectors_data {
                    index.insert(vector)?;
                }
            }

            // Batch persist to storage (atomic, high-performance)
            if let Some(ref storage) = self.storage {
                // Save dimensions on first insert
                if base_index == 0 {
                    storage.put_config("dimensions", self.dimensions as u64)?;
                }

                // Prepare batch items: (index, string_id, vector, metadata)
                let batch_items: Vec<(usize, String, Vec<f32>, serde_json::Value)> = inserts
                    .iter()
                    .enumerate()
                    .map(|(i, (id, vector, metadata))| {
                        (
                            base_index + i,
                            id.clone(),
                            vector.data.clone(),
                            metadata.clone(),
                        )
                    })
                    .collect();

                // Single atomic batch commit (replaces N individual puts)
                storage.put_batch(batch_items)?;
            }

            // Add vectors to in-memory structures
            for (i, (id, vector, metadata)) in inserts.into_iter().enumerate() {
                let idx = base_index + i;
                self.vectors.push(vector);
                self.metadata.insert(idx, metadata);
                self.index_to_id.insert(idx, id.clone());
                self.id_to_index.insert(id, idx);
                result_indices.push(idx);
            }
        }

        Ok(result_indices)
    }

    // ============================================================================
    // Text Search Methods (Hybrid Search)
    // ============================================================================

    /// Enable text search on this store (creates in-memory text index).
    ///
    /// For persistent stores, the text index is stored at `{path}/text_index`.
    /// For in-memory stores, the text index is also in-memory.
    pub fn enable_text_search(&mut self) -> Result<()> {
        self.enable_text_search_with_config(None)
    }

    /// Enable text search with custom configuration.
    ///
    /// # Arguments
    /// * `config` - Text search configuration (None = use store's default or system default)
    pub fn enable_text_search_with_config(
        &mut self,
        config: Option<TextSearchConfig>,
    ) -> Result<()> {
        if self.text_index.is_some() {
            return Ok(()); // Already enabled
        }

        let config = config
            .or_else(|| self.text_search_config.clone())
            .unwrap_or_default();

        self.text_index = if let Some(ref path) = self.storage_path {
            let text_path = path.join("text_index");
            Some(TextIndex::open_with_config(&text_path, &config)?)
        } else {
            Some(TextIndex::open_in_memory_with_config(&config)?)
        };

        Ok(())
    }

    /// Check if text search is enabled.
    #[must_use]
    pub fn has_text_search(&self) -> bool {
        self.text_index.is_some()
    }

    /// Upsert vector with text content for hybrid search.
    ///
    /// Like `set()`, but also indexes text content for BM25 search.
    /// Requires text search to be enabled.
    ///
    /// # Arguments
    /// * `id` - Unique string identifier
    /// * `vector` - Vector embedding
    /// * `text` - Text content for full-text search
    /// * `metadata` - Optional JSON metadata
    ///
    /// # Example
    /// ```ignore
    /// store.enable_text_search()?;
    /// store.set_with_text(
    ///     "doc1".to_string(),
    ///     embed("machine learning"),
    ///     "Machine learning is a branch of AI",
    ///     json!({"type": "article"})
    /// )?;
    /// store.flush()?; // Commit text index changes
    /// ```
    pub fn set_with_text(
        &mut self,
        id: String,
        vector: Vector,
        text: &str,
        metadata: JsonValue,
    ) -> Result<usize> {
        let Some(ref mut text_index) = self.text_index else {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        };

        // Index text content (commit deferred to flush() for batch efficiency)
        text_index.index_document(&id, text)?;

        // Store vector and metadata
        self.set(id, vector, metadata)
    }

    /// Batch upsert vectors with text content for hybrid search.
    ///
    /// Like `set_batch()`, but also indexes text content for BM25 search.
    /// More efficient than repeated `set_with_text()` calls.
    pub fn set_batch_with_text(
        &mut self,
        batch: Vec<(String, Vector, String, JsonValue)>,
    ) -> Result<Vec<usize>> {
        let Some(ref mut text_index) = self.text_index else {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        };

        // Index all text content
        for (id, _, text, _) in &batch {
            text_index.index_document(id, text)?;
        }

        // Convert to set_batch format (without text)
        let vector_batch: Vec<(String, Vector, JsonValue)> = batch
            .into_iter()
            .map(|(id, vector, _, metadata)| (id, vector, metadata))
            .collect();

        self.set_batch(vector_batch)
    }

    /// Search text index only (BM25 scoring).
    ///
    /// Returns Vec of (id, score) tuples, sorted by score descending.
    pub fn text_search(&self, query: &str, k: usize) -> Result<Vec<(String, f32)>> {
        let Some(ref text_index) = self.text_index else {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        };

        text_index.search(query, k)
    }

    /// Hybrid search combining vector similarity and BM25 text relevance.
    ///
    /// Uses Reciprocal Rank Fusion (RRF) to combine results from:
    /// - HNSW vector search (by embedding similarity)
    /// - Tantivy text search (by BM25 relevance)
    ///
    /// # Arguments
    /// * `query_vector` - Query embedding for vector search
    /// * `query_text` - Query text for BM25 search
    /// * `k` - Number of results to return
    /// * `alpha` - Weight for vector vs text (0.0 = text only, 1.0 = vector only, None = 0.5)
    ///
    /// # Returns
    /// Vec of (id, score) tuples, sorted by combined score descending.
    ///
    /// # Example
    /// ```ignore
    /// // Balanced hybrid search
    /// let results = store.hybrid_search(&query_embedding, "machine learning", 10, None)?;
    ///
    /// // Favor vector similarity (70% vector, 30% text)
    /// let results = store.hybrid_search(&query_embedding, "machine learning", 10, Some(0.7))?;
    /// ```
    pub fn hybrid_search(
        &mut self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        alpha: Option<f32>,
    ) -> Result<Vec<(String, f32, JsonValue)>> {
        self.hybrid_search_with_rrf_k(query_vector, query_text, k, alpha, None)
    }

    /// Hybrid search with configurable RRF k constant.
    ///
    /// # Arguments
    /// * `query_vector` - Query embedding for vector search
    /// * `query_text` - Query text for BM25 search
    /// * `k` - Number of results to return
    /// * `alpha` - Weight for vector vs text (0.0 = text only, 1.0 = vector only, None = 0.5)
    /// * `rrf_k` - RRF constant (None = 60, higher values reduce rank influence)
    pub fn hybrid_search_with_rrf_k(
        &mut self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> Result<Vec<(String, f32, JsonValue)>> {
        // Validate inputs
        if query_vector.data.len() != self.dimensions {
            anyhow::bail!(
                "Query vector dimension {} does not match store dimension {}",
                query_vector.data.len(),
                self.dimensions
            );
        }
        if self.text_index.is_none() {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        }

        // Over-fetch from both sources for better fusion
        let fetch_k = k * 2;

        // Vector search
        let vector_results = self.knn_search(query_vector, fetch_k)?;

        // Convert vector results to (id, distance) format - O(1) lookup via reverse map
        let vector_results: Vec<(String, f32)> = vector_results
            .into_iter()
            .filter_map(|(idx, distance)| {
                self.index_to_id.get(&idx).map(|id| (id.clone(), distance))
            })
            .collect();

        // Text search - propagate errors (hybrid requires text search enabled)
        let text_results = self.text_search(query_text, fetch_k)?;

        // Fuse results with weighted RRF
        let fused = weighted_reciprocal_rank_fusion(
            vector_results,
            text_results,
            k,
            rrf_k.unwrap_or(DEFAULT_RRF_K),
            alpha.unwrap_or(0.5),
        );

        // Attach metadata to results
        Ok(self.attach_metadata(fused))
    }

    /// Hybrid search with filter (combining vector + text + metadata filter).
    ///
    /// Like `hybrid_search()`, but also applies a metadata filter.
    ///
    /// # Arguments
    /// * `query_vector` - Query embedding for vector search
    /// * `query_text` - Query text for BM25 search
    /// * `k` - Number of results to return
    /// * `filter` - Metadata filter to apply
    /// * `alpha` - Weight for vector vs text (0.0 = text only, 1.0 = vector only, None = 0.5)
    pub fn hybrid_search_with_filter(
        &mut self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        filter: &MetadataFilter,
        alpha: Option<f32>,
    ) -> Result<Vec<(String, f32, JsonValue)>> {
        self.hybrid_search_with_filter_rrf_k(query_vector, query_text, k, filter, alpha, None)
    }

    /// Hybrid search with filter and configurable RRF k constant.
    pub fn hybrid_search_with_filter_rrf_k(
        &mut self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        filter: &MetadataFilter,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> Result<Vec<(String, f32, JsonValue)>> {
        // Validate inputs
        if query_vector.data.len() != self.dimensions {
            anyhow::bail!(
                "Query vector dimension {} does not match store dimension {}",
                query_vector.data.len(),
                self.dimensions
            );
        }
        if self.text_index.is_none() {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        }

        // Over-fetch 4x to account for filter eliminating candidates
        // Both sources use same multiplier for symmetric RRF ranking
        let fetch_k = k * 4;

        // Filtered vector search (filter applied during search)
        let vector_results = self.knn_search_with_filter(query_vector, fetch_k, filter)?;

        // Convert to (id, distance) format - O(1) lookup via reverse map
        let vector_results: Vec<(String, f32)> = vector_results
            .into_iter()
            .filter_map(|(idx, distance, _)| {
                self.index_to_id.get(&idx).map(|id| (id.clone(), distance))
            })
            .collect();

        // Text search (filter applied post-search since tantivy can't filter metadata)
        let text_results = self.text_search(query_text, fetch_k)?;

        // Filter text results by metadata
        let text_results: Vec<(String, f32)> = text_results
            .into_iter()
            .filter(|(id, _)| {
                self.id_to_index
                    .get(id)
                    .and_then(|&idx| self.metadata.get(&idx))
                    .is_some_and(|meta| filter.matches(meta))
            })
            .collect();

        let fused = weighted_reciprocal_rank_fusion(
            vector_results,
            text_results,
            k,
            rrf_k.unwrap_or(DEFAULT_RRF_K),
            alpha.unwrap_or(0.5),
        );

        // Attach metadata to results
        Ok(self.attach_metadata(fused))
    }

    /// Attach metadata to fused results.
    fn attach_metadata(&self, results: Vec<(String, f32)>) -> Vec<(String, f32, JsonValue)> {
        results
            .into_iter()
            .map(|(id, score)| {
                let metadata = self
                    .id_to_index
                    .get(&id)
                    .and_then(|&idx| self.metadata.get(&idx))
                    .cloned()
                    .unwrap_or(serde_json::json!({}));
                (id, score, metadata)
            })
            .collect()
    }

    // ============================================================================
    // Update Methods
    // ============================================================================

    /// Update existing vector by index (internal method)
    fn update_by_index(
        &mut self,
        index: usize,
        vector: Option<Vector>,
        metadata: Option<JsonValue>,
    ) -> Result<()> {
        // Check if vector exists and is not deleted
        if index >= self.vectors.len() {
            anyhow::bail!("Vector index {index} does not exist");
        }
        if self.deleted.contains_key(&index) {
            anyhow::bail!("Vector index {index} has been deleted");
        }

        // Update vector if provided
        if let Some(new_vector) = vector {
            // Validate dimensions
            if new_vector.dim() != self.dimensions {
                anyhow::bail!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.dimensions,
                    new_vector.dim()
                );
            }

            // Update in memory
            self.vectors[index] = new_vector.clone();

            // Persist to storage if available
            if let Some(ref storage) = self.storage {
                storage.put_vector(index, &new_vector.data)?;
            }
        }

        // Update metadata if provided
        if let Some(ref new_metadata) = metadata {
            self.metadata.insert(index, new_metadata.clone());

            // Persist to storage if available
            if let Some(ref storage) = self.storage {
                storage.put_metadata(index, new_metadata)?;
            }
        }

        Ok(())
    }

    /// Update existing vector by string ID
    pub fn update(
        &mut self,
        id: &str,
        vector: Option<Vector>,
        metadata: Option<JsonValue>,
    ) -> Result<()> {
        let index = self
            .id_to_index
            .get(id)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Vector with ID '{id}' not found"))?;

        self.update_by_index(index, vector, metadata)
    }

    /// Delete vector by string ID (marks as deleted, uses tombstone)
    pub fn delete(&mut self, id: &str) -> Result<()> {
        let index = self
            .id_to_index
            .get(id)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Vector with ID '{id}' not found"))?;

        // Mark as deleted
        self.deleted.insert(index, true);

        // Persist tombstone to storage if available
        if let Some(ref storage) = self.storage {
            storage.put_deleted(index)?;
            storage.delete_id_mapping(id)?;
        }

        // Remove from text index if enabled
        if let Some(ref mut text_index) = self.text_index {
            text_index.delete_document(id)?;
        }

        // Remove from ID mappings
        self.id_to_index.remove(id);
        self.index_to_id.remove(&index);

        Ok(())
    }

    /// Delete multiple vectors by string IDs
    pub fn delete_batch(&mut self, ids: &[String]) -> Result<usize> {
        let mut deleted_count = 0;
        for id in ids {
            if self.delete(id).is_ok() {
                deleted_count += 1;
            }
        }
        Ok(deleted_count)
    }

    /// Get vector by string ID
    pub fn get_by_id(&self, id: &str) -> Option<(&Vector, &JsonValue)> {
        self.id_to_index.get(id).and_then(|&index| {
            // Check if deleted
            if self.deleted.contains_key(&index) {
                return None;
            }
            // Return vector and metadata
            self.vectors
                .get(index)
                .and_then(|vec| self.metadata.get(&index).map(|meta| (vec, meta)))
        })
    }

    /// Get metadata by string ID (without loading vector data).
    ///
    /// More efficient than `get_by_id` when only metadata is needed.
    pub fn get_metadata_by_id(&self, id: &str) -> Option<&JsonValue> {
        self.id_to_index.get(id).and_then(|&index| {
            if self.deleted.contains_key(&index) {
                return None;
            }
            self.metadata.get(&index)
        })
    }

    /// Insert batch of vectors in parallel
    ///
    /// Automatically chunks vectors into optimal batch sizes for parallel insertion.
    /// Uses `hnsw_rs`'s `parallel_insert` with Rayon for multi-threaded building.
    ///
    /// Chunk size of 10,000 balances:
    /// - Parallelization overhead (want batches large enough)
    /// - Memory usage (smaller batches more memory-friendly)
    /// - Progress reporting (can log after each chunk)
    ///
    /// Returns Vec of IDs for inserted vectors
    pub fn batch_insert(&mut self, vectors: Vec<Vector>) -> Result<Vec<usize>> {
        // Chunk size for parallel insertion (recommended: 1000 × num_threads)
        // Using 10,000 as a good default (works well for 4-16 core machines)
        const CHUNK_SIZE: usize = 10_000;

        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Validate dimensions
        for (i, vector) in vectors.iter().enumerate() {
            if vector.dim() != self.dimensions {
                anyhow::bail!(
                    "Vector {} dimension mismatch: expected {}, got {}",
                    i,
                    self.dimensions,
                    vector.dim()
                );
            }
        }

        // Lazy initialize HNSW on first insert
        if self.hnsw_index.is_none() {
            if let Some(quant_mode) = self.pending_quantization.take() {
                let hnsw_params = HNSWParams::default()
                    .with_m(self.hnsw_m)
                    .with_ef_construction(self.hnsw_ef_construction)
                    .with_ef_search(self.hnsw_ef_search);

                let index = match quant_mode {
                    QuantizationMode::SQ8 => {
                        // SQ8 trains lazily on first 256 vectors
                        HNSWIndex::new_with_sq8(self.dimensions, hnsw_params, DistanceFunction::L2)?
                    }
                    QuantizationMode::RaBitQ(params) => {
                        // RaBitQ needs explicit training from batch
                        let mut idx = HNSWIndex::new_with_asymmetric(
                            self.dimensions,
                            hnsw_params,
                            DistanceFunction::L2,
                            params,
                        )?;
                        let training_vectors: Vec<Vec<f32>> =
                            vectors.iter().map(|v| v.data.clone()).collect();
                        idx.train_quantizer(&training_vectors)?;
                        idx
                    }
                };

                self.hnsw_index = Some(index);
            } else {
                let capacity = vectors.len().max(1_000_000);
                self.hnsw_index = Some(HNSWIndex::new(capacity, self.dimensions)?);
            }
        }

        let _start_id = self.vectors.len();
        let mut all_ids = Vec::with_capacity(vectors.len());

        // Process in chunks for better memory management and progress tracking
        for chunk in vectors.chunks(CHUNK_SIZE) {
            // Extract vector data for HNSW
            let vector_data: Vec<Vec<f32>> = chunk.iter().map(|v| v.data.clone()).collect();

            // Parallel insert this chunk
            if let Some(ref mut index) = self.hnsw_index {
                let chunk_ids = index.batch_insert(&vector_data)?;
                all_ids.extend(chunk_ids);
            }
        }

        self.vectors.extend(vectors);

        // Return IDs from HNSW
        Ok(all_ids)
    }

    /// Rebuild HNSW index from existing vectors
    ///
    /// This is needed when:
    /// - Vectors are loaded from disk but index wasn't persisted
    /// - Index needs to be rebuilt after batch inserts
    /// - Quantization is enabled/disabled after loading
    pub fn rebuild_index(&mut self) -> Result<()> {
        if self.vectors.is_empty() {
            return Ok(());
        }

        // Create new HNSW index
        let mut index = HNSWIndex::new(self.vectors.len().max(1_000_000), self.dimensions)?;

        // Insert all vectors
        for vector in &self.vectors {
            index.insert(&vector.data)?;
        }

        self.hnsw_index = Some(index);
        Ok(())
    }

    /// Merge another `VectorStore` into this one using IGTM algorithm
    ///
    /// Uses Iterative Greedy Tree Merging for 1.3-1.7x faster batch inserts
    /// compared to naive insertion.
    ///
    /// # Arguments
    /// * `other` - `VectorStore` to merge from (vectors and metadata will be copied)
    ///
    /// # Returns
    /// Number of vectors merged
    ///
    /// # Note
    /// String IDs from `other` are preserved. If there are conflicts,
    /// the existing ID in `self` takes precedence (other's vector is skipped).
    pub fn merge_from(&mut self, other: &VectorStore) -> Result<usize> {
        if other.dimensions != self.dimensions {
            anyhow::bail!(
                "Dimension mismatch: self={}, other={}",
                self.dimensions,
                other.dimensions
            );
        }

        if other.vectors.is_empty() {
            return Ok(0);
        }

        // Initialize HNSW if needed
        if self.hnsw_index.is_none() {
            let capacity = (self.vectors.len() + other.vectors.len()).max(1_000_000);
            self.hnsw_index = Some(HNSWIndex::new(capacity, self.dimensions)?);
        }

        // Track how many vectors we actually merge (skip ID conflicts)
        let mut merged_count = 0;
        let base_index = self.vectors.len();

        // Copy vectors and metadata, handling ID conflicts
        for (other_idx, vector) in other.vectors.iter().enumerate() {
            // Check for string ID conflict
            let has_conflict = other
                .id_to_index
                .iter()
                .find(|(_, &idx)| idx == other_idx)
                .is_some_and(|(string_id, _)| self.id_to_index.contains_key(string_id));

            if has_conflict {
                continue; // Skip vectors with conflicting string IDs
            }

            // Copy vector
            self.vectors.push(vector.clone());

            // Copy metadata if present
            if let Some(meta) = other.metadata.get(&other_idx) {
                self.metadata
                    .insert(base_index + merged_count, meta.clone());
            }

            // Copy string ID mapping
            if let Some((string_id, _)) =
                other.id_to_index.iter().find(|(_, &idx)| idx == other_idx)
            {
                self.id_to_index
                    .insert(string_id.clone(), base_index + merged_count);
            }

            merged_count += 1;
        }

        // Merge HNSW indexes using IGTM
        if let (Some(ref mut self_index), Some(ref other_index)) =
            (&mut self.hnsw_index, &other.hnsw_index)
        {
            self_index.merge_from(other_index)?;
        } else {
            // Fallback: rebuild index if other didn't have one
            self.rebuild_index()?;
        }

        Ok(merged_count)
    }

    /// Check if index needs to be rebuilt (read-only check)
    ///
    /// Returns true if index is missing and we have significant data.
    /// Use this to avoid write lock when index is already ready.
    #[inline]
    pub fn needs_index_rebuild(&self) -> bool {
        self.hnsw_index.is_none() && self.vectors.len() > 100
    }

    /// Ensure HNSW index is ready for search
    ///
    /// Rebuilds the index if it's missing but vectors exist (crash recovery case).
    /// Call this once after loading from disk before performing searches.
    pub fn ensure_index_ready(&mut self) -> Result<()> {
        if self.needs_index_rebuild() {
            self.rebuild_index()?;
        }
        Ok(())
    }

    /// K-nearest neighbors search using HNSW
    ///
    /// Quantization (if enabled) is for storage/memory savings only.
    /// Search always uses HNSW with original vectors for accuracy and speed.
    ///
    /// Note: May trigger index rebuild if index is missing. For parallel search,
    /// call `ensure_index_ready()` first, then use `knn_search_readonly()`.
    pub fn knn_search(&mut self, query: &Vector, k: usize) -> Result<Vec<(usize, f32)>> {
        self.knn_search_with_ef(query, k, None)
    }

    /// K-nearest neighbors search with optional ef override
    ///
    /// Note: May trigger index rebuild. For parallel search, use readonly version.
    pub fn knn_search_with_ef(
        &mut self,
        query: &Vector,
        k: usize,
        ef: Option<usize>,
    ) -> Result<Vec<(usize, f32)>> {
        self.ensure_index_ready()?;
        self.knn_search_readonly(query, k, ef)
    }

    /// Read-only K-nearest neighbors search (for parallel execution)
    ///
    /// This version takes `&self` instead of `&mut self`, enabling parallel search.
    /// Caller must ensure index is ready by calling `ensure_index_ready()` first.
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    /// * `ef` - Search width override (None = auto-tune to max(k*4, 64))
    #[inline]
    pub fn knn_search_readonly(
        &self,
        query: &Vector,
        k: usize,
        ef: Option<usize>,
    ) -> Result<Vec<(usize, f32)>> {
        // Compute ef early to avoid closure overhead in hot path
        // This is done before any checks to ensure the value is available
        let effective_ef = match ef {
            Some(e) => e,
            None => (k * 4).max(64).max(100), // Default ef_search is 100
        };
        self.knn_search_ef(query, k, effective_ef)
    }

    /// Fast K-nearest neighbors search with concrete ef value
    ///
    /// This is the optimized hot path - ~40% faster than using Option<usize>.
    /// Use this in tight loops where performance is critical.
    ///
    /// When rescore is enabled (with quantization), this:
    /// 1. Searches for k * oversample candidates using quantized distance
    /// 2. Rescores candidates with full-precision vectors
    /// 3. Returns top k results
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    /// * `ef` - Search width (higher = better recall, slower)
    #[inline]
    pub fn knn_search_ef(&self, query: &Vector, k: usize, ef: usize) -> Result<Vec<(usize, f32)>> {
        if query.dim() != self.dimensions {
            anyhow::bail!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.dim()
            );
        }

        // Check if we have any data (either in vectors or in HNSW)
        let has_data =
            !self.vectors.is_empty() || self.hnsw_index.as_ref().is_some_and(|idx| !idx.is_empty());

        if !has_data {
            return Ok(Vec::new());
        }

        // Use HNSW index if available
        if let Some(ref index) = self.hnsw_index {
            // Asymmetric mode with quantization
            if index.is_asymmetric() {
                if self.rescore_enabled && !self.vectors.is_empty() {
                    // Rescore: get candidates with ADC, rerank with exact L2
                    return self.knn_search_with_rescore(query, k, ef);
                }
                // No rescore: use ADC distances directly (fastest)
                return index.search_asymmetric_ef(&query.data, k, ef);
            }
            // Regular HNSW (no quantization)
            return index.search_ef(&query.data, k, ef);
        }

        // Fallback to brute-force if no index (small datasets only)
        self.knn_search_brute_force(query, k)
    }

    /// K-nearest neighbors search with rescore using original vectors
    ///
    /// Used when asymmetric HNSW is enabled with rescore=true.
    /// Fetches k * oversample candidates with ADC, then reranks with full precision L2.
    fn knn_search_with_rescore(
        &self,
        query: &Vector,
        k: usize,
        ef: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let index = self
            .hnsw_index
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("HNSW index required for rescore"))?;

        // Fetch k * oversample candidates using quantized (ADC) distances
        let oversample_k = ((k as f32) * self.oversample_factor).ceil() as usize;
        let candidates = index.search_asymmetric_ef(&query.data, oversample_k, ef)?;

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Rescore candidates with full-precision L2 distance
        // Prefer seerdb (disk) over self.vectors (RAM) to avoid duplication
        let mut rescored: Vec<(usize, f32)> = candidates
            .iter()
            .filter_map(|&(id, _quantized_dist)| {
                // Try seerdb first (disk-backed), fall back to RAM
                let vec_data = if let Some(ref storage) = self.storage {
                    storage.get_vector(id).ok().flatten()
                } else {
                    self.vectors.get(id).map(|v| v.data.clone())
                };

                vec_data.map(|data| {
                    let dist = l2_distance(&query.data, &data);
                    (id, dist)
                })
            })
            .collect();

        // Sort by distance and return top k
        rescored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        rescored.truncate(k);

        Ok(rescored)
    }

    /// K-nearest neighbors search with metadata filtering
    ///
    /// Performs HNSW search and filters results by metadata.
    /// Uses ACORN-1 algorithm for efficient filtered search.
    ///
    /// Returns Vec of (id, distance, metadata) tuples
    pub fn knn_search_with_filter(
        &mut self,
        query: &Vector,
        k: usize,
        filter: &MetadataFilter,
    ) -> Result<Vec<(usize, f32, JsonValue)>> {
        self.ensure_index_ready()?;
        self.knn_search_with_filter_ef_readonly(query, k, filter, None)
    }

    /// K-nearest neighbors search with metadata filtering and optional ef override
    ///
    /// Performs HNSW search and filters results by metadata.
    /// Uses ACORN-1 algorithm for efficient filtered search.
    ///
    /// Returns Vec of (id, distance, metadata) tuples
    pub fn knn_search_with_filter_ef(
        &mut self,
        query: &Vector,
        k: usize,
        filter: &MetadataFilter,
        ef: Option<usize>,
    ) -> Result<Vec<(usize, f32, JsonValue)>> {
        self.ensure_index_ready()?;
        self.knn_search_with_filter_ef_readonly(query, k, filter, ef)
    }

    /// Read-only filtered search (for parallel execution)
    ///
    /// This version takes `&self` instead of `&mut self`, enabling parallel search.
    /// Caller must ensure index is ready by calling `ensure_index_ready()` first.
    pub fn knn_search_with_filter_ef_readonly(
        &self,
        query: &Vector,
        k: usize,
        filter: &MetadataFilter,
        ef: Option<usize>,
    ) -> Result<Vec<(usize, f32, JsonValue)>> {
        // Use ACORN-1 filtered search if HNSW index is available
        if let Some(ref hnsw) = self.hnsw_index {
            // Create filter closure that checks metadata
            let metadata_map = &self.metadata;
            let deleted_map = &self.deleted;
            let filter_fn = |node_id: u32| -> bool {
                let index = node_id as usize;

                // Skip deleted vectors
                if deleted_map.contains_key(&index) {
                    return false;
                }

                // Get metadata (default to empty object if none)
                let metadata = metadata_map
                    .get(&index)
                    .cloned()
                    .unwrap_or(serde_json::json!({}));

                // Apply filter
                filter.matches(&metadata)
            };

            // Use ACORN-1 filtered search with optional ef override
            let search_results = hnsw.search_with_filter_ef(&query.data, k, ef, filter_fn)?;

            // Convert to (index, distance, metadata) format
            let filtered_results: Vec<(usize, f32, JsonValue)> = search_results
                .into_iter()
                .map(|(index, distance)| {
                    let metadata = self
                        .metadata
                        .get(&index)
                        .cloned()
                        .unwrap_or(serde_json::json!({}));
                    (index, distance, metadata)
                })
                .collect();

            return Ok(filtered_results);
        }

        // Fallback: Brute-force search with filtering (no HNSW index)
        let mut all_results: Vec<(usize, f32, JsonValue)> = self
            .vectors
            .iter()
            .enumerate()
            .filter_map(|(index, vec)| {
                // Skip deleted vectors
                if self.deleted.contains_key(&index) {
                    return None;
                }

                // Get metadata and check filter
                let metadata = self
                    .metadata
                    .get(&index)
                    .cloned()
                    .unwrap_or(serde_json::json!({}));

                if !filter.matches(&metadata) {
                    return None;
                }

                // Calculate distance
                let distance = query.l2_distance(vec).unwrap_or(f32::MAX);
                Some((index, distance, metadata))
            })
            .collect();

        // Sort by distance and take top k
        all_results.sort_by(|a, b| a.1.total_cmp(&b.1));
        all_results.truncate(k);

        Ok(all_results)
    }

    /// Search with optional filter (convenience method)
    pub fn search(
        &mut self,
        query: &Vector,
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<(usize, f32, JsonValue)>> {
        self.search_with_ef(query, k, filter, None)
    }

    /// Search with optional filter and ef override
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    /// * `filter` - Optional metadata filter
    /// * `ef` - Search width override (None = auto-tune to max(k*4, 64))
    pub fn search_with_ef(
        &mut self,
        query: &Vector,
        k: usize,
        filter: Option<&MetadataFilter>,
        ef: Option<usize>,
    ) -> Result<Vec<(usize, f32, JsonValue)>> {
        self.ensure_index_ready()?;
        self.search_with_ef_readonly(query, k, filter, ef)
    }

    /// Read-only search with optional filter (for parallel execution)
    ///
    /// This version takes `&self` instead of `&mut self`, enabling parallel search.
    /// Caller must ensure index is ready by calling `ensure_index_ready()` first.
    pub fn search_with_ef_readonly(
        &self,
        query: &Vector,
        k: usize,
        filter: Option<&MetadataFilter>,
        ef: Option<usize>,
    ) -> Result<Vec<(usize, f32, JsonValue)>> {
        if let Some(f) = filter {
            self.knn_search_with_filter_ef_readonly(query, k, f, ef)
        } else {
            // No filter - get all results with metadata
            let results = self.knn_search_readonly(query, k, ef)?;
            Ok(results
                .into_iter()
                .filter_map(|(index, distance)| {
                    // Skip deleted vectors
                    if self.deleted.contains_key(&index) {
                        return None;
                    }
                    // Get metadata (default to empty object)
                    let metadata = self
                        .metadata
                        .get(&index)
                        .cloned()
                        .unwrap_or(serde_json::json!({}));
                    Some((index, distance, metadata))
                })
                .collect())
        }
    }

    /// Parallel batch search for multiple queries (ChromaDB-style optimization)
    ///
    /// Executes all queries in parallel using rayon, achieving significant
    /// speedup on multi-core systems. Caller must call `ensure_index_ready()`
    /// before this method.
    ///
    /// # Arguments
    /// * `queries` - Slice of query vectors
    /// * `k` - Number of neighbors to return per query
    /// * `ef` - Search width override (None = auto-tune)
    ///
    /// # Returns
    /// Vec of results, one per query. Each result contains (index, distance) pairs.
    pub fn batch_search_parallel(
        &self,
        queries: &[Vector],
        k: usize,
        ef: Option<usize>,
    ) -> Vec<Result<Vec<(usize, f32)>>> {
        // Pre-compute ef once to avoid per-query Option overhead
        let effective_ef = match ef {
            Some(e) => e,
            None => (k * 4).max(64).max(100),
        };
        queries
            .par_iter()
            .map(|q| self.knn_search_ef(q, k, effective_ef))
            .collect()
    }

    /// Parallel batch search with metadata (ChromaDB-style optimization)
    ///
    /// Executes all queries in parallel using rayon, returning metadata with results.
    /// Caller must call `ensure_index_ready()` before this method.
    pub fn batch_search_parallel_with_metadata(
        &self,
        queries: &[Vector],
        k: usize,
        ef: Option<usize>,
    ) -> Vec<Result<Vec<(usize, f32, JsonValue)>>> {
        queries
            .par_iter()
            .map(|q| self.search_with_ef_readonly(q, k, None, ef))
            .collect()
    }

    /// Brute-force K-NN search (fallback, mainly for testing)
    pub fn knn_search_brute_force(&self, query: &Vector, k: usize) -> Result<Vec<(usize, f32)>> {
        if query.dim() != self.dimensions {
            anyhow::bail!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.dim()
            );
        }

        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Compute distances to all vectors
        let mut distances: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(id, vec)| {
                let dist = query.l2_distance(vec).unwrap_or(f32::MAX);
                (id, dist)
            })
            .collect();

        // Sort by distance and take top K
        distances.sort_by(|a, b| a.1.total_cmp(&b.1));
        Ok(distances.into_iter().take(k).collect())
    }

    /// Get vector by ID
    ///
    /// Returns the vector from RAM if available, otherwise fetches from disk (seerdb).
    /// Note: When vectors aren't loaded to RAM (quantized mode), this returns an owned
    /// vector from disk. Use `get_owned()` for consistent owned semantics.
    pub fn get(&self, id: usize) -> Option<&Vector> {
        self.vectors.get(id)
    }

    /// Get vector by ID (owned)
    ///
    /// Returns an owned vector, fetching from disk if not in RAM.
    /// Use this when you need the vector data regardless of storage location.
    pub fn get_owned(&self, id: usize) -> Option<Vector> {
        // Try RAM first
        if let Some(v) = self.vectors.get(id) {
            return Some(v.clone());
        }

        // Fall back to disk (seerdb)
        if let Some(ref storage) = self.storage {
            if let Ok(Some(data)) = storage.get_vector(id) {
                return Some(Vector::new(data));
            }
        }

        None
    }

    /// Number of vectors stored (excluding deleted vectors)
    pub fn len(&self) -> usize {
        // Return active vector count (excluding tombstones)
        self.vectors.len().saturating_sub(self.deleted.len())
    }

    /// Check if store is empty (no active vectors)
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Memory usage estimate (bytes)
    pub fn memory_usage(&self) -> usize {
        // Vector data: num_vectors * dim * 4 bytes (f32)
        self.vectors.iter().map(|v| v.dim() * 4).sum::<usize>()
    }

    /// Bytes per vector (average)
    pub fn bytes_per_vector(&self) -> f32 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        self.memory_usage() as f32 / self.vectors.len() as f32
    }

    /// Set HNSW `ef_search` parameter (runtime tuning)
    pub fn set_ef_search(&mut self, ef_search: usize) {
        if let Some(ref mut index) = self.hnsw_index {
            index.set_ef_search(ef_search);
        }
    }

    /// Get HNSW `ef_search` parameter
    pub fn get_ef_search(&self) -> Option<usize> {
        self.hnsw_index
            .as_ref()
            .map(super::hnsw_index::HNSWIndex::get_ef_search)
    }

    /// Save vector store to disk with HNSW graph serialization
    ///
    /// Uses `hnsw_rs` `file_dump()` to persist both vectors and graph structure.
    /// This enables fast loading (<1s) without rebuilding the index.
    ///
    /// File format:
    /// - `<basename>.hnsw`: HNSW index
    /// - `<basename>.vectors.bin`: Vector data
    /// - `<basename>.quantized.bin`: Quantized vectors (if quantization enabled)
    /// - `<basename>.quantizer.json`: Quantizer parameters (if quantization enabled)
    pub fn save_to_disk(&self, base_path: &str) -> Result<()> {
        use std::fs;
        use std::path::Path;

        let path = Path::new(base_path);
        let directory = path.parent().unwrap_or_else(|| Path::new("."));
        let filename = path
            .file_name()
            .and_then(|f| f.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid path: no filename in '{base_path}'"))?;

        // Create directory if needed
        fs::create_dir_all(directory)?;

        // Save vectors array
        let vectors_path = directory.join(format!("{filename}.vectors.bin"));
        let vectors_data: Vec<Vec<f32>> = self.vectors.iter().map(|v| v.data.clone()).collect();
        let encoded = bincode::serialize(&vectors_data)?;
        fs::write(&vectors_path, encoded)?;

        // Save metadata if present
        if !self.metadata.is_empty() {
            let metadata_path = directory.join(format!("{filename}.metadata.json"));
            let metadata_json = serde_json::to_string_pretty(&self.metadata)?;
            fs::write(&metadata_path, metadata_json)?;
        }

        // Save ID to index mapping if present
        if !self.id_to_index.is_empty() {
            let id_mapping_path = directory.join(format!("{filename}.id_mapping.json"));
            let id_mapping_json = serde_json::to_string_pretty(&self.id_to_index)?;
            fs::write(&id_mapping_path, id_mapping_json)?;
        }

        // Save deleted vectors tombstones if present
        if !self.deleted.is_empty() {
            let deleted_path = directory.join(format!("{filename}.deleted.json"));
            let deleted_json = serde_json::to_string_pretty(&self.deleted)?;
            fs::write(&deleted_path, deleted_json)?;
        }

        // Save HNSW index if present
        if let Some(ref index) = self.hnsw_index {
            let hnsw_path = directory.join(format!("{filename}.hnsw"));
            index.save(&hnsw_path)?;
        }

        Ok(())
    }

    /// Load vector store from disk with fast HNSW index loading
    ///
    /// Tries to load HNSW index first (fast: <1s).
    /// Falls back to loading vectors and rebuilding if index not found.
    ///
    /// Performance:
    /// - With HNSW index: <1 second load time (4175x faster than rebuild)
    /// - Fallback (rebuild): Several minutes for 100K+ vectors
    pub fn load_from_disk(base_path: &str, dimensions: usize) -> Result<Self> {
        use std::fs;
        use std::path::Path;

        let path = Path::new(base_path);
        let directory = path.parent().unwrap_or_else(|| Path::new("."));
        let filename = path
            .file_name()
            .and_then(|f| f.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid path: no filename in '{base_path}'"))?;

        // Check if HNSW index file exists
        let hnsw_path = directory.join(format!("{filename}.hnsw"));

        if hnsw_path.exists() {
            // Fast path: Load HNSW index directly
            let hnsw_index = HNSWIndex::load(&hnsw_path)?;

            // Load vectors array (needed for get/len/verification)
            let vectors_path = directory.join(format!("{filename}.vectors.bin"));
            let vectors = if vectors_path.exists() {
                let vectors_data = fs::read(&vectors_path)?;
                let vectors_raw: Vec<Vec<f32>> = bincode::deserialize(&vectors_data)?;
                vectors_raw.into_iter().map(Vector::new).collect()
            } else {
                Vec::new()
            };

            // Try to load metadata
            let metadata_path = directory.join(format!("{filename}.metadata.json"));
            let metadata = if metadata_path.exists() {
                let metadata_json = fs::read_to_string(&metadata_path)?;
                serde_json::from_str(&metadata_json)?
            } else {
                HashMap::new()
            };

            // Try to load ID to index mapping
            let id_mapping_path = directory.join(format!("{filename}.id_mapping.json"));
            let id_to_index: HashMap<String, usize> = if id_mapping_path.exists() {
                let id_mapping_json = fs::read_to_string(&id_mapping_path)?;
                serde_json::from_str(&id_mapping_json)?
            } else {
                HashMap::new()
            };

            // Try to load deleted tombstones
            let deleted_path = directory.join(format!("{filename}.deleted.json"));
            let deleted: HashMap<usize, bool> = if deleted_path.exists() {
                let deleted_json = fs::read_to_string(&deleted_path)?;
                serde_json::from_str(&deleted_json)?
            } else {
                HashMap::new()
            };

            // Build reverse map for O(1) lookup
            let index_to_id: HashMap<usize, String> = id_to_index
                .iter()
                .map(|(id, &idx)| (idx, id.clone()))
                .collect();

            Ok(Self {
                vectors,
                hnsw_index: Some(hnsw_index),
                dimensions,
                rescore_enabled: false,
                oversample_factor: 3.0,
                metadata,
                id_to_index,
                index_to_id,
                deleted,
                storage: None,
                storage_path: None,
                text_index: None,
                text_search_config: None,
                pending_quantization: None,
                hnsw_m: 16,
                hnsw_ef_construction: 100,
                hnsw_ef_search: 100,
            })
        } else {
            // Fallback: Load vectors and rebuild HNSW
            let vectors_path = directory.join(format!("{filename}.vectors.bin"));
            if !vectors_path.exists() {
                anyhow::bail!("Vector file not found: {}", vectors_path.display());
            }

            let vectors_data = fs::read(&vectors_path)?;
            let vectors_raw: Vec<Vec<f32>> = bincode::deserialize(&vectors_data)?;
            let vectors: Vec<Vector> = vectors_raw.into_iter().map(Vector::new).collect();

            // Try to load metadata
            let metadata_path = directory.join(format!("{filename}.metadata.json"));
            let metadata = if metadata_path.exists() {
                let metadata_json = fs::read_to_string(&metadata_path)?;
                serde_json::from_str(&metadata_json)?
            } else {
                HashMap::new()
            };

            // Try to load ID to index mapping
            let id_mapping_path = directory.join(format!("{filename}.id_mapping.json"));
            let id_to_index: HashMap<String, usize> = if id_mapping_path.exists() {
                let id_mapping_json = fs::read_to_string(&id_mapping_path)?;
                serde_json::from_str(&id_mapping_json)?
            } else {
                HashMap::new()
            };

            // Try to load deleted tombstones
            let deleted_path = directory.join(format!("{filename}.deleted.json"));
            let deleted: HashMap<usize, bool> = if deleted_path.exists() {
                let deleted_json = fs::read_to_string(&deleted_path)?;
                serde_json::from_str(&deleted_json)?
            } else {
                HashMap::new()
            };

            // Build reverse map for O(1) lookup
            let index_to_id: HashMap<usize, String> = id_to_index
                .iter()
                .map(|(id, &idx)| (idx, id.clone()))
                .collect();

            // Create VectorStore and rebuild HNSW index
            let mut store = Self {
                vectors,
                hnsw_index: None,
                dimensions,
                rescore_enabled: false,
                oversample_factor: 3.0,
                metadata,
                id_to_index,
                index_to_id,
                deleted,
                storage: None,
                storage_path: None,
                text_index: None,
                text_search_config: None,
                pending_quantization: None,
                hnsw_m: 16,
                hnsw_ef_construction: 100,
                hnsw_ef_search: 100,
            };

            if !store.vectors.is_empty() {
                store.rebuild_index()?;
            }

            Ok(store)
        }
    }

    /// Flush all pending changes to disk.
    ///
    /// This commits:
    /// - Vector/metadata changes to seerdb storage
    /// - Text index changes to tantivy (if enabled)
    ///
    /// Call after batch inserts for durability.
    pub fn flush(&mut self) -> Result<()> {
        // Flush vector storage
        if let Some(ref storage) = self.storage {
            storage.flush()?;
        }

        // Commit text index if enabled
        if let Some(ref mut text_index) = self.text_index {
            text_index.commit()?;
        }

        Ok(())
    }

    /// Check if this store has persistent storage enabled
    pub fn is_persistent(&self) -> bool {
        self.storage.is_some()
    }

    /// Get reference to the seerdb storage backend (if persistent)
    ///
    /// Returns None if storage is not persistent (in-memory mode).
    /// Use for profiling/stats access only.
    pub fn storage(&self) -> Option<&SeerDBStorage> {
        self.storage.as_ref()
    }
}
