//! Vector storage with HNSW indexing
//!
//! `VectorStore` manages a collection of vectors and provides k-NN search
//! using HNSW (Hierarchical Navigable Small World) algorithm.
//!
//! Optional SQ8 quantization for memory-efficient storage.
//!
//! Optional tantivy-based full-text search for hybrid (vector + BM25) retrieval.

mod filter;
mod helpers;
mod input;
mod multivec_ops;
mod options;
mod record_store;
mod search;
mod text_search;
mod thread_safe;

pub use crate::omen::Metric;
pub use filter::MetadataFilter;
pub use input::{BatchItem, QueryData, QueryInput, Rerank, SearchOptions, VectorData, VectorInput};
pub use options::VectorStoreOptions;
pub use record_store::{Record, RecordStore};
pub use thread_safe::ThreadSafeVectorStore;

// SearchResult is defined in this module and re-exported from lib.rs

use super::hnsw::{HNSWParams, SegmentConfig, SegmentManager};
use super::hnsw_index::HNSWIndex;
use super::muvera::{MultiVecStorage, MultiVectorConfig, MuveraEncoder};
use super::types::Vector;
use super::QuantizationMode;
use crate::omen::{
    parse_wal_delete, parse_wal_insert, CheckpointOptions, MetadataIndex, OmenFile, WalEntryType,
};
use crate::text::{TextIndex, TextSearchConfig};
use anyhow::Result;
use rayon::prelude::*;
use serde_json::Value as JsonValue;
use std::path::{Path, PathBuf};

// ============================================================================
// Constants
// ============================================================================

/// Default HNSW M parameter (neighbors per node)
const DEFAULT_HNSW_M: usize = 16;
/// Default HNSW ef_construction parameter (build quality)
const DEFAULT_HNSW_EF_CONSTRUCTION: usize = 100;
/// Default HNSW ef_search parameter (search quality)
const DEFAULT_HNSW_EF_SEARCH: usize = 100;
/// Default oversample factor for rescore
const DEFAULT_OVERSAMPLE_FACTOR: f32 = 3.0;

// ============================================================================
// Helper Functions (moved to helpers.rs)
// ============================================================================

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
    pub segments: Option<SegmentManager>,

    /// Direct HNSW index access (for backward compatibility during transition)
    /// TODO: Remove once segment integration is complete
    pub hnsw_index: Option<HNSWIndex>,

    /// Whether to rescore candidates with original vectors (default: true when quantization enabled)
    rescore_enabled: bool,

    /// Oversampling factor for rescore (default: 3.0)
    oversample_factor: f32,

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

    /// Pending quantization mode (deferred until first insert for training)
    pending_quantization: Option<QuantizationMode>,

    /// HNSW parameters for lazy initialization
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_ef_search: usize,

    /// Distance metric for similarity search (default: L2)
    distance_metric: Metric,

    // ============================================================================
    // MUVERA (Multi-Vector) Support
    // ============================================================================
    /// MUVERA encoder for multi-vector to FDE transformation.
    /// Present when store is created with `new_muvera()`.
    muvera_encoder: Option<MuveraEncoder>,

    /// Storage for original multi-vector tokens (for MaxSim reranking).
    multivec_storage: Option<MultiVecStorage>,

    /// Maximum tokens per document (default: 512, matches ColBERT).
    max_tokens: usize,
}

/// Default maximum tokens per multi-vector document.
const DEFAULT_MAX_TOKENS: usize = 512;

impl VectorStore {
    // ============================================================================
    // Constructors
    // ============================================================================

    /// Create new vector store
    #[must_use]
    pub fn new(dimensions: usize) -> Self {
        Self {
            records: RecordStore::new(dimensions as u32),
            segments: None,
            hnsw_index: None,
            rescore_enabled: false,
            oversample_factor: DEFAULT_OVERSAMPLE_FACTOR,
            metadata_index: MetadataIndex::new(),
            storage: None,
            storage_path: None,
            text_index: None,
            text_search_config: None,
            pending_quantization: None,
            hnsw_m: DEFAULT_HNSW_M,
            hnsw_ef_construction: DEFAULT_HNSW_EF_CONSTRUCTION,
            hnsw_ef_search: DEFAULT_HNSW_EF_SEARCH,
            distance_metric: Metric::L2,
            muvera_encoder: None,
            multivec_storage: None,
            max_tokens: DEFAULT_MAX_TOKENS,
        }
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
    ///
    /// // Create store for 128-dimensional token embeddings
    /// let mut store = VectorStore::multi_vector(128);
    ///
    /// // Insert document with token embeddings
    /// let tokens: Vec<Vec<f32>> = vec![vec![0.1; 128]; 10]; // 10 tokens
    /// let refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();
    /// store.set_multi("doc1", &refs, serde_json::json!({})).unwrap();
    ///
    /// // Search with query tokens
    /// let results = store.search_multi(&refs, 10).unwrap();
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

        Self {
            records: RecordStore::new(fde_dim as u32),
            segments: None,
            hnsw_index: None,
            rescore_enabled: false,
            oversample_factor: DEFAULT_OVERSAMPLE_FACTOR,
            metadata_index: MetadataIndex::new(),
            storage: None,
            storage_path: None,
            text_index: None,
            text_search_config: None,
            pending_quantization: None,
            hnsw_m: DEFAULT_HNSW_M,
            hnsw_ef_construction: DEFAULT_HNSW_EF_CONSTRUCTION,
            hnsw_ef_search: DEFAULT_HNSW_EF_SEARCH,
            distance_metric: Metric::InnerProduct, // FDEs use inner product
            muvera_encoder: Some(encoder),
            multivec_storage: Some(MultiVecStorage::new(token_dim)),
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }

    // Compatibility accessors for fields moved to RecordStore
    fn dimensions(&self) -> usize {
        self.records.dimensions() as usize
    }

    /// Create new vector store with quantization
    ///
    /// Quantization is trained on the first batch of vectors inserted.
    #[must_use]
    pub fn new_with_quantization(dimensions: usize, mode: QuantizationMode) -> Self {
        Self {
            records: RecordStore::new(dimensions as u32),
            segments: None,
            hnsw_index: None,
            rescore_enabled: true,
            oversample_factor: DEFAULT_OVERSAMPLE_FACTOR,
            metadata_index: MetadataIndex::new(),
            storage: None,
            storage_path: None,
            text_index: None,
            text_search_config: None,
            pending_quantization: Some(mode),
            hnsw_m: DEFAULT_HNSW_M,
            muvera_encoder: None,
            multivec_storage: None,
            max_tokens: DEFAULT_MAX_TOKENS,
            hnsw_ef_construction: DEFAULT_HNSW_EF_CONSTRUCTION,
            hnsw_ef_search: DEFAULT_HNSW_EF_SEARCH,
            distance_metric: Metric::L2,
        }
    }

    /// Create new vector store with custom HNSW parameters
    pub fn new_with_params(
        dimensions: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        distance_metric: Metric,
    ) -> Result<Self> {
        let hnsw_index = Some(HNSWIndex::new_with_params(
            1_000_000,
            dimensions,
            m,
            ef_construction,
            ef_search,
            distance_metric.into(),
        )?);

        Ok(Self {
            records: RecordStore::new(dimensions as u32),
            segments: None,
            hnsw_index,
            rescore_enabled: false,
            oversample_factor: DEFAULT_OVERSAMPLE_FACTOR,
            metadata_index: MetadataIndex::new(),
            storage: None,
            storage_path: None,
            text_index: None,
            text_search_config: None,
            pending_quantization: None,
            hnsw_m: m,
            hnsw_ef_construction: ef_construction,
            hnsw_ef_search: ef_search,
            distance_metric,
            muvera_encoder: None,
            multivec_storage: None,
            max_tokens: DEFAULT_MAX_TOKENS,
        })
    }

    // ============================================================================
    // Persistence: Open/Create
    // ============================================================================

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
        use roaring::RoaringBitmap;

        let path = path.as_ref();
        let omen_path = OmenFile::compute_omen_path(path);
        let mut storage = if omen_path.exists() {
            OmenFile::open(path)?
        } else {
            OmenFile::create(path, 0)?
        };

        // Load persisted snapshot (checkpoint data only, not WAL)
        let snapshot = storage.load_persisted_snapshot()?;
        let mut dimensions = snapshot.dimensions as usize;

        // Get HNSW parameters from header
        let header = storage.header();
        let distance_metric = header.metric;
        let hnsw_m = header.hnsw_m as usize;
        let hnsw_ef_construction = header.hnsw_ef_construction as usize;
        let hnsw_ef_search = header.hnsw_ef_search as usize;

        // Check quantization
        let _is_quantized = storage.is_quantized()?;
        let quantization_mode =
            helpers::quantization_mode_from_id(storage.get_quantization_mode()?.unwrap_or(0));

        // Build RecordStore from snapshot
        let mut deleted_bitmap: RoaringBitmap = snapshot.deleted.iter().copied().collect();
        let mut slots: Vec<Option<Record>> = Vec::with_capacity(snapshot.vectors.len());

        for (slot, vec_opt) in snapshot.vectors.iter().enumerate() {
            let slot_u32 = slot as u32;
            if deleted_bitmap.contains(slot_u32) {
                slots.push(None);
                continue;
            }

            if let Some(vec_data) = vec_opt {
                // Find the ID for this slot
                let id = snapshot
                    .id_to_slot
                    .iter()
                    .find(|(_, &s)| s == slot_u32)
                    .map_or_else(|| format!("__slot_{slot}"), |(id, _)| id.clone());

                let metadata = snapshot.metadata.get(&slot_u32).cloned();
                slots.push(Some(Record::new(id, vec_data.clone(), metadata)));
            } else {
                slots.push(None);
            }
        }

        let mut records =
            RecordStore::from_snapshot(slots, deleted_bitmap.clone(), dimensions as u32);

        // Replay WAL entries directly into RecordStore (Phase 5 architecture)
        let wal_entries = storage.pending_wal_entries()?;
        for entry in wal_entries {
            if !entry.verify() {
                tracing::warn!(
                    entry_type = ?entry.header.entry_type,
                    "Skipping corrupted WAL entry during recovery"
                );
                continue;
            }

            match entry.header.entry_type {
                WalEntryType::InsertNode => {
                    if let Ok(insert_data) = parse_wal_insert(&entry.data) {
                        // Infer dimensions from first WAL vector if needed
                        if dimensions == 0 && !insert_data.vector.is_empty() {
                            dimensions = insert_data.vector.len();
                            records = RecordStore::from_snapshot(
                                Vec::new(),
                                RoaringBitmap::new(),
                                dimensions as u32,
                            );
                        }

                        // Parse metadata
                        let metadata: Option<JsonValue> =
                            insert_data.metadata.as_ref().and_then(|bytes| {
                                match serde_json::from_slice(bytes) {
                                    Ok(json) => Some(json),
                                    Err(e) => {
                                        tracing::warn!(
                                            "Corrupt metadata for '{}' during WAL replay: {}",
                                            insert_data.id,
                                            e
                                        );
                                        None
                                    }
                                }
                            });

                        // Upsert into RecordStore
                        records.upsert(insert_data.id, insert_data.vector, metadata)?;
                    }
                }
                WalEntryType::DeleteNode => {
                    if let Ok(delete_data) = parse_wal_delete(&entry.data) {
                        records.delete(&delete_data.id);
                    }
                }
                WalEntryType::UpdateNeighbors
                | WalEntryType::UpdateMetadata
                | WalEntryType::Checkpoint => {
                    // No-op: neighbors managed by HNSW, metadata/checkpoint are markers
                }
            }
        }

        // Update deleted bitmap after WAL replay
        deleted_bitmap.clone_from(records.deleted_bitmap());

        // Build HNSW index - must maintain slot index correspondence
        let slot_count = records.slot_count() as usize;
        let active_count = records.len() as usize;

        let hnsw_index = if let Some(hnsw_bytes) = snapshot.hnsw_bytes {
            match HNSWIndex::from_bytes(&hnsw_bytes) {
                Ok(index) => {
                    // Compare with total slots, not just live count, since HNSW includes deleted
                    if index.len() != slot_count && slot_count > 0 {
                        tracing::info!(
                            "HNSW index count ({}) differs from slot count ({}), rebuilding",
                            index.len(),
                            slot_count
                        );
                        Some(helpers::rebuild_hnsw_with_slots(
                            &records,
                            &deleted_bitmap,
                            dimensions,
                            hnsw_m,
                            hnsw_ef_construction,
                            hnsw_ef_search,
                            distance_metric,
                            quantization_mode.as_ref(),
                        )?)
                    } else {
                        Some(index)
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to deserialize HNSW index, rebuilding: {}", e);
                    if active_count > 0 {
                        Some(helpers::rebuild_hnsw_with_slots(
                            &records,
                            &deleted_bitmap,
                            dimensions,
                            hnsw_m,
                            hnsw_ef_construction,
                            hnsw_ef_search,
                            distance_metric,
                            quantization_mode.as_ref(),
                        )?)
                    } else {
                        None
                    }
                }
            }
        } else if active_count > 0 {
            Some(helpers::rebuild_hnsw_with_slots(
                &records,
                &deleted_bitmap,
                dimensions,
                hnsw_m,
                hnsw_ef_construction,
                hnsw_ef_search,
                distance_metric,
                quantization_mode.as_ref(),
            )?)
        } else {
            None
        };

        // Try to open existing text index
        let text_index_path = path.join("text_index");
        let text_index = if text_index_path.exists() {
            Some(TextIndex::open(&text_index_path)?)
        } else {
            None
        };

        // Load or rebuild metadata index
        let metadata_index = if let Some(ref bytes) = snapshot.metadata_index_bytes {
            match MetadataIndex::from_bytes(bytes) {
                Ok(index) => {
                    tracing::debug!("Loaded MetadataIndex from disk");
                    index
                }
                Err(e) => {
                    tracing::warn!("Failed to deserialize MetadataIndex, rebuilding: {}", e);
                    let mut index = MetadataIndex::new();
                    for (slot, record) in records.iter_live() {
                        if let Some(ref meta) = record.metadata {
                            index.index_json(slot, meta);
                        }
                    }
                    index
                }
            }
        } else {
            // No persisted index, build from scratch
            let mut index = MetadataIndex::new();
            for (slot, record) in records.iter_live() {
                if let Some(ref meta) = record.metadata {
                    index.index_json(slot, meta);
                }
            }
            index
        };

        // Enable rescore if quantized
        let rescore_enabled = hnsw_index
            .as_ref()
            .is_some_and(super::hnsw_index::HNSWIndex::is_asymmetric);

        // Reconstruct multi-vector state if config is present
        let (muvera_encoder, multivec_storage, distance_metric) =
            if let Some((reps, bits, seed, token_dim)) = snapshot.multivec_config {
                let config = MultiVectorConfig {
                    repetitions: reps,
                    partition_bits: bits,
                    seed,
                };
                let encoder = MuveraEncoder::new(token_dim, config);

                // Reconstruct storage from persisted bytes
                let storage = match (&snapshot.multivec_bytes, &snapshot.multivec_offsets) {
                    (Some(vec_bytes), Some(off_bytes)) => {
                        match MultiVecStorage::from_bytes(vec_bytes, off_bytes, token_dim) {
                            Ok(s) => Some(s),
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to restore MultiVecStorage, creating empty: {}",
                                    e
                                );
                                Some(MultiVecStorage::new(token_dim))
                            }
                        }
                    }
                    _ => Some(MultiVecStorage::new(token_dim)),
                };

                // FDEs use inner product
                (Some(encoder), storage, Metric::InnerProduct)
            } else {
                (None, None, distance_metric)
            };

        Ok(Self {
            records,
            segments: None,
            hnsw_index,
            rescore_enabled,
            oversample_factor: DEFAULT_OVERSAMPLE_FACTOR,
            metadata_index,
            storage: Some(storage),
            storage_path: Some(path.to_path_buf()),
            text_index,
            text_search_config: None,
            pending_quantization: quantization_mode,
            hnsw_m: hnsw_m.max(DEFAULT_HNSW_M),
            hnsw_ef_construction: hnsw_ef_construction.max(DEFAULT_HNSW_EF_CONSTRUCTION),
            hnsw_ef_search: hnsw_ef_search.max(DEFAULT_HNSW_EF_SEARCH),
            distance_metric,
            muvera_encoder,
            multivec_storage,
            max_tokens: DEFAULT_MAX_TOKENS,
        })
    }

    /// Open a persistent vector store with specified dimensions
    ///
    /// Like `open()` but ensures dimensions are set for new databases.
    pub fn open_with_dimensions(path: impl AsRef<Path>, dimensions: usize) -> Result<Self> {
        let mut store = Self::open(path)?;
        if store.dimensions() == 0 {
            store.records.set_dimensions(dimensions as u32);
            if let Some(ref mut storage) = store.storage {
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
        let omen_path = OmenFile::compute_omen_path(path);

        // If path or .omen file exists, load existing data
        if path.exists() || omen_path.exists() {
            let mut store = Self::open(path)?;

            // Apply dimension if specified and store has none
            if store.dimensions() == 0 && options.dimensions > 0 {
                store.records.set_dimensions(options.dimensions as u32);
                if let Some(ref mut storage) = store.storage {
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
        let mut storage = OmenFile::create(path, options.dimensions as u32)?;
        let dimensions = options.dimensions;

        // Determine HNSW parameters
        let m = options.m.unwrap_or(16);
        let ef_construction = options.ef_construction.unwrap_or(100);
        let ef_search = options.ef_search.unwrap_or(100);

        // Get distance metric from options (default: L2)
        let distance_metric = options.metric.unwrap_or(Metric::L2);

        // Initialize HNSW - defer when quantization enabled
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
                        distance_metric.into(),
                    )?),
                    None,
                )
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // Save dimensions to storage if set
        if dimensions > 0 {
            storage.put_config("dimensions", dimensions as u64)?;
        }

        // Save quantization mode to storage if set
        if let Some(ref q) = options.quantization {
            storage.put_quantization_mode(helpers::quantization_mode_to_id(q))?;
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
        let oversample_factor = options.oversample.unwrap_or_else(|| {
            helpers::default_oversample_for_quantization(options.quantization.as_ref())
        });

        Ok(Self {
            records: RecordStore::new(dimensions as u32),
            segments: None,
            hnsw_index,
            rescore_enabled,
            oversample_factor,
            metadata_index: MetadataIndex::new(),
            storage: Some(storage),
            storage_path: Some(path.to_path_buf()),
            text_index,
            text_search_config: options.text_search_config.clone(),
            pending_quantization,
            hnsw_m: m,
            hnsw_ef_construction: ef_construction,
            hnsw_ef_search: ef_search,
            distance_metric,
            muvera_encoder: None,
            multivec_storage: None,
            max_tokens: DEFAULT_MAX_TOKENS,
        })
    }

    /// Build an in-memory vector store with custom options.
    pub fn build_with_options(options: &VectorStoreOptions) -> Result<Self> {
        let dimensions = options.dimensions;

        // Determine HNSW parameters
        let m = options.m.unwrap_or(16);
        let ef_construction = options.ef_construction.unwrap_or(100);
        let ef_search = options.ef_search.unwrap_or(100);

        // Get distance metric from options (default: L2)
        let distance_metric = options.metric.unwrap_or(Metric::L2);

        // Initialize HNSW - defer when quantization enabled
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
                        distance_metric.into(),
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
        let oversample_factor = options.oversample.unwrap_or_else(|| {
            helpers::default_oversample_for_quantization(options.quantization.as_ref())
        });

        Ok(Self {
            records: RecordStore::new(dimensions as u32),
            segments: None,
            hnsw_index,
            rescore_enabled,
            oversample_factor,
            metadata_index: MetadataIndex::new(),
            storage: None,
            storage_path: None,
            text_index,
            text_search_config: options.text_search_config.clone(),
            pending_quantization,
            hnsw_m: m,
            hnsw_ef_construction: ef_construction,
            hnsw_ef_search: ef_search,
            distance_metric,
            muvera_encoder: None,
            multivec_storage: None,
            max_tokens: DEFAULT_MAX_TOKENS,
        })
    }

    // ============================================================================
    // Private Helpers
    // ============================================================================

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

    /// Create initial HNSW index, handling pending quantization.
    #[allow(dead_code)]
    fn create_initial_hnsw(
        &mut self,
        dimensions: usize,
        training_vectors: &[Vec<f32>],
    ) -> Result<HNSWIndex> {
        self.create_initial_hnsw_with_capacity(dimensions, training_vectors, 10_000)
    }

    /// Create initial HNSW index with custom capacity.
    #[allow(dead_code)]
    fn create_initial_hnsw_with_capacity(
        &mut self,
        dimensions: usize,
        training_vectors: &[Vec<f32>],
        capacity: usize,
    ) -> Result<HNSWIndex> {
        if let Some(quant_mode) = self.pending_quantization.take() {
            if let Some(ref mut storage) = self.storage {
                storage.put_quantization_mode(helpers::quantization_mode_to_id(&quant_mode))?;
            }
            helpers::initialize_quantized_hnsw(
                dimensions,
                self.hnsw_m,
                self.hnsw_ef_construction,
                self.hnsw_ef_search,
                self.distance_metric,
                quant_mode,
                training_vectors,
            )
        } else {
            helpers::initialize_standard_hnsw(
                dimensions,
                self.hnsw_m,
                self.hnsw_ef_construction,
                self.hnsw_ef_search,
                self.distance_metric,
                capacity,
            )
        }
    }

    // ============================================================================
    // Insert/Set Methods
    // ============================================================================

    /// Insert vector and return its slot ID
    pub fn insert(&mut self, vector: Vector) -> Result<usize> {
        // Generate a unique ID for unnamed vectors
        let slot = self.records.slot_count();
        let id = format!("__auto_{slot}");

        self.set(id, vector, helpers::default_metadata())
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
        if self.records.get_slot(&id).is_some() {
            anyhow::bail!("Vector with ID '{id}' already exists. Use set() to update.");
        }

        self.set(id, vector, metadata)
    }

    /// Upsert vector (insert or update) with string ID and metadata
    ///
    /// This is the recommended method for most use cases.
    ///
    /// # Durability
    ///
    /// Individual writes are buffered in the WAL but NOT synced to disk immediately.
    /// For guaranteed durability, call [`flush()`](Self::flush) after critical writes.
    /// Batch operations ([`set_batch`](Self::set_batch)) sync the WAL at batch end.
    ///
    /// Without explicit flush:
    /// - Data is recoverable after normal shutdown
    /// - Data may be lost on crash/power failure between set() and next flush/batch
    pub fn set(&mut self, id: String, vector: Vector, metadata: JsonValue) -> Result<usize> {
        // Initialize segments if needed
        if self.segments.is_none() && self.hnsw_index.is_none() {
            let dimensions = self.resolve_dimensions(vector.dim())?;
            self.records.set_dimensions(dimensions as u32);

            // Create segment manager with initial config
            let config = SegmentConfig::new(dimensions)
                .with_params(HNSWParams {
                    m: self.hnsw_m,
                    ef_construction: self.hnsw_ef_construction,
                    ..Default::default()
                })
                .with_distance(self.distance_metric.into())
                .with_quantization(self.pending_quantization.is_some());

            self.segments = Some(
                SegmentManager::new(config)
                    .map_err(|e| anyhow::anyhow!("Failed to create segment manager: {e}"))?,
            );
        } else if vector.dim() != self.dimensions() {
            anyhow::bail!(
                "Vector dimension mismatch: store expects {}, got {}",
                self.dimensions(),
                vector.dim()
            );
        }

        // Check if this is an update
        let old_slot = self.records.get_slot(&id);

        // Upsert into RecordStore - creates new slot (both for insert and update)
        // RecordStore marks old slot deleted internally to maintain slot == HNSW node ID
        let slot = self
            .records
            .upsert(id.clone(), vector.data.clone(), Some(metadata.clone()))?
            as usize;

        // Insert into segments (preferred) or hnsw_index (legacy)
        if let Some(ref mut segments) = self.segments {
            // Note: mark_deleted not needed - RecordStore filtering handles deleted nodes
            segments
                .insert_with_slot(&vector.data, slot as u32)
                .map_err(|e| anyhow::anyhow!("Segment insert failed: {e}"))?;
        } else if let Some(ref mut index) = self.hnsw_index {
            // Legacy path: direct hnsw_index
            if let Some(old) = old_slot {
                if let Err(e) = index.mark_deleted(old) {
                    tracing::warn!(
                        id = %id,
                        slot = old,
                        error = ?e,
                        "Failed to mark old node as deleted in HNSW during update"
                    );
                }
            }
            index.insert(&vector.data)?;
        }

        // Update metadata index
        if let Some(old) = old_slot {
            self.metadata_index.remove(old);
        }
        self.metadata_index.index_json(slot as u32, &metadata);

        // WAL for crash durability
        if let Some(ref mut storage) = self.storage {
            let metadata_bytes = serde_json::to_vec(&metadata)?;
            storage.wal_append_insert(&id, &vector.data, Some(&metadata_bytes))?;
        }

        Ok(slot)
    }

    /// Batch set vectors (insert or update multiple vectors at once)
    ///
    /// This is the recommended method for bulk operations.
    /// Uses parallel HNSW construction for new indexes.
    pub fn set_batch(&mut self, batch: Vec<(String, Vector, JsonValue)>) -> Result<Vec<usize>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        // Separate batch into updates and inserts
        let mut updates: Vec<(u32, String, Vector, JsonValue)> = Vec::new();
        let mut inserts: Vec<(String, Vector, JsonValue)> = Vec::new();

        for (id, vector, metadata) in batch {
            if let Some(slot) = self.records.get_slot(&id) {
                updates.push((slot, id, vector, metadata));
            } else {
                inserts.push((id, vector, metadata));
            }
        }

        let mut result_indices = Vec::with_capacity(updates.len() + inserts.len());

        // Process updates individually
        for (old_slot, id, vector, metadata) in updates {
            // Update RecordStore - creates new slot, marks old as deleted
            let new_slot =
                self.records
                    .upsert(id.clone(), vector.data.clone(), Some(metadata.clone()))?;

            // Insert into segments (preferred) or hnsw_index (legacy)
            if let Some(ref mut segments) = self.segments {
                segments
                    .insert_with_slot(&vector.data, new_slot)
                    .map_err(|e| anyhow::anyhow!("Segment insert failed: {e}"))?;
            } else if let Some(ref mut index) = self.hnsw_index {
                if let Err(e) = index.mark_deleted(old_slot) {
                    tracing::warn!(
                        slot = old_slot,
                        error = ?e,
                        "Failed to mark old node as deleted in HNSW during batch update"
                    );
                }
                index.insert(&vector.data)?;
            }

            // Update metadata index (remove old, add new)
            self.metadata_index.remove(old_slot);
            self.metadata_index.index_json(new_slot, &metadata);

            // WAL for crash durability
            if let Some(ref mut storage) = self.storage {
                let metadata_bytes = serde_json::to_vec(&metadata)?;
                storage.wal_append_insert(&id, &vector.data, Some(&metadata_bytes))?;
            }

            result_indices.push(new_slot as usize);
        }

        // Process inserts with batch optimization
        if !inserts.is_empty() {
            let vectors_data: Vec<Vec<f32>> =
                inserts.iter().map(|(_, v, _)| v.data.clone()).collect();

            // Check if this is a new index (no existing segments or hnsw_index)
            let is_new_index = self.segments.is_none() && self.hnsw_index.is_none();

            if is_new_index {
                let dimensions = self.resolve_dimensions(inserts[0].1.dim())?;
                self.records.set_dimensions(dimensions as u32);

                // Insert into RecordStore first to get slots
                let mut slots = Vec::with_capacity(inserts.len());
                for (id, vector, metadata) in &inserts {
                    let slot = self.records.upsert(
                        id.clone(),
                        vector.data.clone(),
                        Some(metadata.clone()),
                    )?;
                    slots.push(slot);
                    self.metadata_index.index_json(slot, metadata);
                }

                // Build segment config
                let config = SegmentConfig::new(dimensions)
                    .with_params(HNSWParams {
                        m: self.hnsw_m,
                        ef_construction: self.hnsw_ef_construction,
                        ..Default::default()
                    })
                    .with_distance(self.distance_metric.into())
                    .with_quantization(self.pending_quantization.is_some());

                // Use parallel build with slot mapping
                self.segments = Some(
                    SegmentManager::build_parallel_with_slots(config, vectors_data.clone(), &slots)
                        .map_err(|e| anyhow::anyhow!("Segment parallel build failed: {e}"))?,
                );

                // Handle quantization mode persistence
                if let Some(quant_mode) = self.pending_quantization.take() {
                    if let Some(ref mut storage) = self.storage {
                        storage
                            .put_quantization_mode(helpers::quantization_mode_to_id(&quant_mode))?;
                    }
                }

                // WAL for crash durability
                if let Some(ref mut storage) = self.storage {
                    for (id, vector, metadata) in &inserts {
                        let metadata_bytes = serde_json::to_vec(metadata)?;
                        storage.wal_append_insert(id, &vector.data, Some(&metadata_bytes))?;
                    }
                }

                result_indices.extend(slots.iter().map(|&s| s as usize));
            } else {
                // Existing index - validate dimensions and insert one by one
                let expected_dims = self.dimensions();
                for (i, (_, vector, _)) in inserts.iter().enumerate() {
                    if vector.dim() != expected_dims {
                        anyhow::bail!(
                            "Vector {} dimension mismatch: expected {}, got {}",
                            i,
                            expected_dims,
                            vector.dim()
                        );
                    }
                }

                // Insert into RecordStore and index
                let mut slots = Vec::with_capacity(inserts.len());
                for (id, vector, metadata) in &inserts {
                    let slot = self.records.upsert(
                        id.clone(),
                        vector.data.clone(),
                        Some(metadata.clone()),
                    )?;
                    slots.push(slot);
                    self.metadata_index.index_json(slot, metadata);

                    // Insert into segments or hnsw_index
                    if let Some(ref mut segments) = self.segments {
                        segments
                            .insert_with_slot(&vector.data, slot)
                            .map_err(|e| anyhow::anyhow!("Segment insert failed: {e}"))?;
                    } else if let Some(ref mut index) = self.hnsw_index {
                        index.insert(&vector.data)?;
                    }
                }

                // WAL for crash durability
                if let Some(ref mut storage) = self.storage {
                    for (id, vector, metadata) in &inserts {
                        let metadata_bytes = serde_json::to_vec(metadata)?;
                        storage.wal_append_insert(id, &vector.data, Some(&metadata_bytes))?;
                    }
                }

                result_indices.extend(slots.iter().map(|&s| s as usize));
            }
        }

        // Sync WAL once at end of batch for durability
        if let Some(ref mut storage) = self.storage {
            storage.wal_sync()?;
        }

        Ok(result_indices)
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
        let slot = index as u32;

        // Check bounds and deleted status
        if !self.records.is_live(slot) {
            anyhow::bail!("Vector index {index} does not exist or has been deleted");
        }

        if let Some(new_vector) = vector {
            if new_vector.dim() != self.dimensions() {
                anyhow::bail!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.dimensions(),
                    new_vector.dim()
                );
            }

            // Update in RecordStore
            self.records.update_vector(slot, new_vector.data.clone())?;

            if let Some(ref mut storage) = self.storage {
                storage.put_vector(index, &new_vector.data)?;
            }
        }

        if let Some(ref new_metadata) = metadata {
            // Re-index metadata: remove old values, add new ones
            self.metadata_index.remove(slot);
            self.metadata_index.index_json(slot, new_metadata);
            self.records.update_metadata(slot, new_metadata.clone())?;

            if let Some(ref mut storage) = self.storage {
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
        let slot = self
            .records
            .get_slot(id)
            .ok_or_else(|| anyhow::anyhow!("Vector with ID '{id}' not found"))?;

        self.update_by_index(slot as usize, vector, metadata)
    }

    /// Delete vector by string ID (lazy delete)
    ///
    /// This method:
    /// 1. Marks the vector as deleted in bitmap (O(1) soft delete)
    /// 2. Marks node as deleted in HNSW (filtered during search)
    /// 3. Removes from text index if present
    /// 4. Persists to WAL
    ///
    /// Deleted vectors are filtered during search. Call `compact()` to reclaim space.
    pub fn delete(&mut self, id: &str) -> Result<()> {
        // Delete from RecordStore (single source of truth)
        let slot = self
            .records
            .delete(id)
            .ok_or_else(|| anyhow::anyhow!("Vector with ID '{id}' not found"))?;

        self.metadata_index.remove(slot);

        // Mark as deleted in HNSW (lazy - no graph repair, filtered during search)
        if let Some(ref mut hnsw) = self.hnsw_index {
            if let Err(e) = hnsw.mark_deleted(slot) {
                tracing::warn!(
                    id = id,
                    slot = slot,
                    error = ?e,
                    "Failed to mark node as deleted in HNSW"
                );
            }
        }

        // Use OmenFile::delete for WAL-backed persistence
        if let Some(ref mut storage) = self.storage {
            storage.delete(id)?;
        }

        if let Some(ref mut text_index) = self.text_index {
            text_index.delete_document(id)?;
        }

        Ok(())
    }

    /// Delete multiple vectors by string IDs (lazy delete)
    ///
    /// Marks vectors as deleted in bitmap. Deleted vectors are filtered during search.
    /// Call `compact()` to reclaim space after bulk deletes.
    pub fn delete_batch(&mut self, ids: &[String]) -> Result<usize> {
        // Delete from RecordStore and collect slots
        let mut slots: Vec<u32> = Vec::with_capacity(ids.len());
        let mut valid_ids: Vec<String> = Vec::with_capacity(ids.len());

        for id in ids {
            if let Some(slot) = self.records.delete(id) {
                self.metadata_index.remove(slot);
                slots.push(slot);
                valid_ids.push(id.clone());
            }
        }

        // Mark as deleted in HNSW (lazy - filtered during search)
        if !slots.is_empty() {
            if let Some(ref mut hnsw) = self.hnsw_index {
                if let Err(e) = hnsw.mark_deleted_batch(&slots) {
                    tracing::warn!(
                        count = slots.len(),
                        error = ?e,
                        "Failed to batch mark nodes as deleted in HNSW"
                    );
                }
            }
        }

        // Persist deletions
        for id in &valid_ids {
            if let Some(ref mut storage) = self.storage {
                if let Err(e) = storage.delete(id) {
                    tracing::warn!(id = %id, error = ?e, "Failed to persist deletion to storage");
                }
            }
            if let Some(ref mut text_index) = self.text_index {
                if let Err(e) = text_index.delete_document(id) {
                    tracing::warn!(id = %id, error = ?e, "Failed to delete from text index");
                }
            }
        }

        Ok(valid_ids.len())
    }

    /// Delete vectors matching a metadata filter
    ///
    /// Evaluates the filter against all vectors and deletes those that match.
    /// This is more efficient than manually iterating and calling delete_batch.
    ///
    /// # Arguments
    /// * `filter` - MongoDB-style metadata filter
    ///
    /// # Returns
    /// Number of vectors deleted
    pub fn delete_by_filter(&mut self, filter: &MetadataFilter) -> Result<usize> {
        // Find matching IDs
        let ids_to_delete: Vec<String> = self
            .records
            .iter_live()
            .filter_map(|(_, record)| {
                let metadata = record.metadata.as_ref()?;
                if filter.matches(metadata) {
                    Some(record.id.clone())
                } else {
                    None
                }
            })
            .collect();

        if ids_to_delete.is_empty() {
            return Ok(0);
        }

        self.delete_batch(&ids_to_delete)
    }

    /// Count vectors matching a metadata filter
    ///
    /// Evaluates the filter against all vectors and returns the count of matches.
    /// More efficient than iterating and counting manually.
    ///
    /// # Arguments
    /// * `filter` - MongoDB-style metadata filter
    ///
    /// # Returns
    /// Number of vectors matching the filter
    #[must_use]
    pub fn count_by_filter(&self, filter: &MetadataFilter) -> usize {
        self.records
            .iter_live()
            .filter(|(_, record)| {
                record
                    .metadata
                    .as_ref()
                    .is_some_and(|metadata| filter.matches(metadata))
            })
            .count()
    }

    /// Get vector by string ID
    ///
    /// Returns owned data since vectors may be loaded from disk for quantized stores.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<(Vector, JsonValue)> {
        let record = self.records.get(id)?;
        let metadata = record
            .metadata
            .clone()
            .unwrap_or_else(helpers::default_metadata);
        Some((Vector::new(record.vector.clone()), metadata))
    }

    /// Get multiple vectors by string IDs
    ///
    /// Returns a vector of results in the same order as input IDs.
    /// Missing/deleted IDs return None in their position.
    #[must_use]
    pub fn get_batch(&self, ids: &[impl AsRef<str>]) -> Vec<Option<(Vector, JsonValue)>> {
        ids.iter().map(|id| self.get(id.as_ref())).collect()
    }

    /// Get metadata by string ID (without loading vector data)
    #[must_use]
    pub fn get_metadata_by_id(&self, id: &str) -> Option<&JsonValue> {
        self.records.get(id).and_then(|r| r.metadata.as_ref())
    }

    // ============================================================================
    // Batch Insert / Index Rebuild
    // ============================================================================

    /// Insert batch of vectors in parallel
    ///
    /// NOTE: This method generates synthetic IDs for the vectors.
    /// For explicit IDs, use `set_batch` instead.
    pub fn batch_insert(&mut self, vectors: Vec<Vector>) -> Result<Vec<usize>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let dimensions = self.dimensions();
        for (i, vector) in vectors.iter().enumerate() {
            if vector.dim() != dimensions {
                anyhow::bail!(
                    "Vector {} dimension mismatch: expected {}, got {}",
                    i,
                    dimensions,
                    vector.dim()
                );
            }
        }

        // Insert into RecordStore with generated IDs
        let mut all_slots = Vec::with_capacity(vectors.len());
        let base_slot = self.records.slot_count();

        for (i, vector) in vectors.iter().enumerate() {
            let id = format!("_batch_{}", base_slot + i as u32);
            let slot = self.records.upsert(id, vector.data.clone(), None)?;
            all_slots.push(slot as usize);
        }

        // Build or extend segments
        let vector_data: Vec<Vec<f32>> = vectors.iter().map(|v| v.data.clone()).collect();
        let slots: Vec<u32> = all_slots.iter().map(|&s| s as u32).collect();

        if self.segments.is_none() && self.hnsw_index.is_none() {
            // Build new segment with parallel construction
            let config = SegmentConfig::new(dimensions)
                .with_params(HNSWParams {
                    m: self.hnsw_m,
                    ef_construction: self.hnsw_ef_construction,
                    ..Default::default()
                })
                .with_distance(self.distance_metric.into())
                .with_quantization(self.pending_quantization.is_some());

            self.segments = Some(
                SegmentManager::build_parallel_with_slots(config, vector_data, &slots)
                    .map_err(|e| anyhow::anyhow!("Segment build failed: {e}"))?,
            );
        } else if let Some(ref mut segments) = self.segments {
            // Insert into existing segments
            for (vector, &slot) in vector_data.iter().zip(slots.iter()) {
                segments
                    .insert_with_slot(vector, slot)
                    .map_err(|e| anyhow::anyhow!("Segment insert failed: {e}"))?;
            }
        } else if let Some(ref mut index) = self.hnsw_index {
            // Legacy path: insert into hnsw_index
            index.batch_insert(&vector_data)?;
        }

        Ok(all_slots)
    }

    /// Rebuild HNSW index from existing vectors
    pub fn rebuild_index(&mut self) -> Result<()> {
        if self.records.is_empty() {
            return Ok(());
        }

        // Collect live vectors and their slots
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(self.records.len() as usize);
        let mut slots: Vec<u32> = Vec::with_capacity(self.records.len() as usize);
        for (slot, record) in self.records.iter_live() {
            vectors.push(record.vector.clone());
            slots.push(slot);
        }

        // Build segment config
        let config = SegmentConfig::new(self.dimensions())
            .with_params(HNSWParams {
                m: self.hnsw_m,
                ef_construction: self.hnsw_ef_construction,
                ..Default::default()
            })
            .with_distance(self.distance_metric.into())
            .with_quantization(self.pending_quantization.is_some());

        // Rebuild with parallel construction
        self.segments = Some(
            SegmentManager::build_parallel_with_slots(config, vectors, &slots)
                .map_err(|e| anyhow::anyhow!("Segment rebuild failed: {e}"))?,
        );

        // Clear legacy hnsw_index
        self.hnsw_index = None;

        Ok(())
    }

    /// Merge another `VectorStore` into this one using IGTM algorithm
    pub fn merge_from(&mut self, other: &VectorStore) -> Result<usize> {
        if other.dimensions() != self.dimensions() {
            anyhow::bail!(
                "Dimension mismatch: self={}, other={}",
                self.dimensions(),
                other.dimensions()
            );
        }

        if other.records.is_empty() {
            return Ok(0);
        }

        if self.hnsw_index.is_none() {
            let capacity =
                (self.records.len() as usize + other.records.len() as usize).max(1_000_000);
            self.hnsw_index = Some(HNSWIndex::new_with_params(
                capacity,
                self.dimensions(),
                self.hnsw_m,
                self.hnsw_ef_construction,
                self.hnsw_ef_search,
                self.distance_metric.into(),
            )?);
        }

        let mut merged_count = 0;

        // Merge records, skipping conflicts
        for (_, record) in other.records.iter_live() {
            // Skip if ID already exists in self
            if self.records.get_slot(&record.id).is_some() {
                continue;
            }

            // Insert into our RecordStore
            self.records.upsert(
                record.id.clone(),
                record.vector.clone(),
                record.metadata.clone(),
            )?;
            merged_count += 1;
        }

        // Rebuild index after merge to ensure consistency
        self.rebuild_index()?;

        Ok(merged_count)
    }

    /// Check if index needs to be rebuilt
    #[inline]
    #[must_use]
    pub fn needs_index_rebuild(&self) -> bool {
        self.segments.is_none() && self.hnsw_index.is_none() && self.records.len() > 100
    }

    /// Ensure HNSW index is ready for search
    pub fn ensure_index_ready(&mut self) -> Result<()> {
        if self.needs_index_rebuild() {
            self.rebuild_index()?;
        }
        Ok(())
    }

    // ============================================================================
    // Search Methods
    // ============================================================================

    /// K-nearest neighbors search using HNSW
    ///
    /// Takes `&self` for concurrent read access. Index initialization happens
    /// on first insert, not first search.
    pub fn knn_search(&self, query: &Vector, k: usize) -> Result<Vec<(usize, f32)>> {
        self.knn_search_readonly(query, k, None)
    }

    /// K-nearest neighbors search with optional ef override
    ///
    /// Takes `&self` for concurrent read access.
    pub fn knn_search_with_ef(
        &self,
        query: &Vector,
        k: usize,
        ef: Option<usize>,
    ) -> Result<Vec<(usize, f32)>> {
        self.knn_search_readonly(query, k, ef)
    }

    /// Read-only K-nearest neighbors search (for parallel execution)
    #[inline]
    pub fn knn_search_readonly(
        &self,
        query: &Vector,
        k: usize,
        ef: Option<usize>,
    ) -> Result<Vec<(usize, f32)>> {
        // Use provided ef, or fall back to stored hnsw_ef_search
        // Ensure ef >= k (HNSW requirement)
        let effective_ef = helpers::compute_effective_ef(ef, self.hnsw_ef_search, k);
        self.knn_search_ef(query, k, effective_ef)
    }

    /// Fast K-nearest neighbors search with concrete ef value
    #[inline]
    pub fn knn_search_ef(&self, query: &Vector, k: usize, ef: usize) -> Result<Vec<(usize, f32)>> {
        if query.dim() != self.dimensions() {
            anyhow::bail!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions(),
                query.dim()
            );
        }

        let config = search::SearchConfig {
            rescore_enabled: self.rescore_enabled,
            oversample_factor: self.oversample_factor,
        };

        search::knn_search_core(
            &self.records,
            self.segments.as_ref(),
            self.hnsw_index.as_ref(),
            &query.data,
            k,
            ef,
            &config,
        )
    }

    /// K-nearest neighbors search with metadata filtering
    ///
    /// Takes `&self` for concurrent read access.
    pub fn knn_search_with_filter(
        &self,
        query: &Vector,
        k: usize,
        filter: &MetadataFilter,
    ) -> Result<Vec<SearchResult>> {
        self.knn_search_with_filter_ef_readonly(query, k, filter, None)
    }

    /// K-nearest neighbors search with metadata filtering and optional ef override
    ///
    /// Takes `&self` for concurrent read access.
    pub fn knn_search_with_filter_ef(
        &self,
        query: &Vector,
        k: usize,
        filter: &MetadataFilter,
        ef: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        self.knn_search_with_filter_ef_readonly(query, k, filter, ef)
    }

    /// Read-only filtered search (for parallel execution)
    ///
    /// Uses Roaring bitmap index for O(1) filter evaluation when possible,
    /// falls back to JSON-based filtering for complex filters.
    pub fn knn_search_with_filter_ef_readonly(
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
            self.hnsw_index.as_ref(),
            &query.data,
            k,
            effective_ef,
            filter,
        )
    }

    /// Search with optional filter (convenience method)
    ///
    /// Takes `&self` for concurrent read access.
    pub fn search(
        &self,
        query: &Vector,
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>> {
        self.search_with_options_readonly(query, k, filter, None, None)
    }

    /// Search with optional filter and ef override
    ///
    /// Takes `&self` for concurrent read access.
    pub fn search_with_ef(
        &self,
        query: &Vector,
        k: usize,
        filter: Option<&MetadataFilter>,
        ef: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        self.search_with_options_readonly(query, k, filter, ef, None)
    }

    /// Search with all options: filter, ef override, and max_distance
    ///
    /// Takes `&self` for concurrent read access.
    pub fn search_with_options(
        &self,
        query: &Vector,
        k: usize,
        filter: Option<&MetadataFilter>,
        ef: Option<usize>,
        max_distance: Option<f32>,
    ) -> Result<Vec<SearchResult>> {
        self.search_with_options_readonly(query, k, filter, ef, max_distance)
    }

    /// Read-only search with optional filter (for parallel execution)
    pub fn search_with_ef_readonly(
        &self,
        query: &Vector,
        k: usize,
        filter: Option<&MetadataFilter>,
        ef: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        self.search_with_options_readonly(query, k, filter, ef, None)
    }

    /// Read-only search with all options (for parallel execution)
    pub fn search_with_options_readonly(
        &self,
        query: &Vector,
        k: usize,
        filter: Option<&MetadataFilter>,
        ef: Option<usize>,
        max_distance: Option<f32>,
    ) -> Result<Vec<SearchResult>> {
        let mut results = if let Some(f) = filter {
            self.knn_search_with_filter_ef_readonly(query, k, f, ef)?
        } else {
            let slot_results = self.knn_search_readonly(query, k, ef)?;
            search::slots_to_results_with_fallback(&self.records, slot_results, &query.data, k)
        };

        if let Some(max_dist) = max_distance {
            results.retain(|r| r.distance <= max_dist);
        }

        Ok(results)
    }

    /// Parallel batch search for multiple queries
    #[must_use]
    pub fn search_batch(
        &self,
        queries: &[Vector],
        k: usize,
        ef: Option<usize>,
    ) -> Vec<Result<Vec<(usize, f32)>>> {
        // Use provided ef, or fall back to stored hnsw_ef_search
        // Ensure ef >= k (HNSW requirement)
        let effective_ef = helpers::compute_effective_ef(ef, self.hnsw_ef_search, k);
        queries
            .par_iter()
            .map(|q| self.knn_search_ef(q, k, effective_ef))
            .collect()
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
            .map(|q| self.search_with_ef_readonly(q, k, None, ef))
            .collect()
    }

    /// Brute-force K-NN search (fallback)
    pub fn knn_search_brute_force(&self, query: &Vector, k: usize) -> Result<Vec<(usize, f32)>> {
        if query.dim() != self.dimensions() {
            anyhow::bail!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions(),
                query.dim()
            );
        }

        Ok(search::brute_force_search(&self.records, &query.data, k))
    }

    // ============================================================================
    // Optimization
    // ============================================================================

    /// Optimize index for cache-efficient search
    ///
    /// Reorders graph nodes using BFS traversal to improve memory locality.
    /// Nodes that are frequently accessed together during search will be stored
    /// adjacently in memory, reducing cache misses and improving QPS.
    ///
    /// Call this after loading/building the index and before querying for best results.
    /// Based on NeurIPS 2021 "Graph Reordering for Cache-Efficient Near Neighbor Search".
    ///
    /// Returns the number of nodes reordered, or 0 if index is empty/not initialized.
    pub fn optimize(&mut self) -> Result<usize> {
        let Some(ref mut index) = self.hnsw_index else {
            return Ok(0);
        };

        // Get the old-to-new mapping from HNSW reordering
        let old_to_new = index
            .optimize_cache_locality()
            .map_err(|e| anyhow::anyhow!("Optimization failed: {e}"))?;

        if old_to_new.is_empty() {
            return Ok(0);
        }

        // HNSW reordering changes its internal indices, but RecordStore keeps
        // its slot indices stable. This works because HNSW search returns
        // indices that map to RecordStore slots via the stored node data.
        // No RecordStore reordering needed - HNSW handles the graph optimization.
        Ok(old_to_new.len())
    }

    // ============================================================================
    // Accessors
    // ============================================================================

    /// Get vector by internal index (used by FFI bindings)
    #[must_use]
    #[allow(dead_code)] // Used by FFI feature
    pub(crate) fn get_by_internal_index(&self, idx: usize) -> Option<Vector> {
        self.records
            .get_vector(idx as u32)
            .map(|v| Vector::new(v.to_vec()))
    }

    /// Get vector by internal index, owned (used by FFI bindings)
    #[must_use]
    #[allow(dead_code)] // Used by FFI feature
    pub(crate) fn get_by_internal_index_owned(&self, idx: usize) -> Option<Vector> {
        self.records
            .get_vector(idx as u32)
            .map(|v| Vector::new(v.to_vec()))
    }

    /// Number of vectors stored (excluding deleted vectors)
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len() as usize
    }

    /// Count of vectors stored (excluding deleted vectors)
    ///
    /// Alias for `len()` - preferred for database-style APIs.
    #[must_use]
    pub fn count(&self) -> usize {
        self.len()
    }

    /// Check if store is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// List all non-deleted IDs
    ///
    /// Returns vector IDs without loading vector data.
    /// O(n) time, O(n) memory for strings only.
    #[must_use]
    pub fn ids(&self) -> Vec<String> {
        self.records
            .iter_live()
            .map(|(_, record)| record.id.clone())
            .collect()
    }

    /// Get all items as (id, vector, metadata) tuples
    ///
    /// Returns all non-deleted items. O(n) time and memory.
    #[must_use]
    pub fn items(&self) -> Vec<(String, Vec<f32>, JsonValue)> {
        self.records
            .iter_live()
            .map(|(_, record)| {
                let metadata = record.metadata.clone().unwrap_or_default();
                (record.id.clone(), record.vector.clone(), metadata)
            })
            .collect()
    }

    /// Check if an ID exists (not deleted)
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.records.get(id).is_some()
    }

    /// Memory usage estimate (bytes)
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.records
            .iter_live()
            .map(|(_, r)| r.vector.len() * 4)
            .sum()
    }

    /// Bytes per vector (average)
    #[must_use]
    pub fn bytes_per_vector(&self) -> f32 {
        let count = self.records.len();
        if count == 0 {
            return 0.0;
        }
        self.memory_usage() as f32 / count as f32
    }

    /// Set HNSW `ef_search` parameter (runtime tuning)
    pub fn set_ef_search(&mut self, ef_search: usize) {
        self.hnsw_ef_search = ef_search;
        if let Some(ref mut index) = self.hnsw_index {
            index.set_ef_search(ef_search);
        }
    }

    /// Get HNSW `ef_search` parameter
    #[must_use]
    pub fn get_ef_search(&self) -> Option<usize> {
        // Return stored value even if no index yet
        Some(self.hnsw_ef_search)
    }

    /// Get index-to-ID mapping (for FFI bindings)
    ///
    /// Returns a HashMap mapping internal slot indices to string IDs.
    #[must_use]
    pub fn index_to_id_mapping(&self) -> std::collections::HashMap<usize, String> {
        self.records
            .iter_live()
            .map(|(slot, record)| (slot as usize, record.id.clone()))
            .collect()
    }

    /// Get ID-to-index mapping (for FFI bindings)
    ///
    /// Returns a HashMap mapping string IDs to internal slot indices.
    #[must_use]
    pub fn id_to_index_mapping(&self) -> std::collections::HashMap<String, usize> {
        self.records
            .iter_live()
            .map(|(slot, record)| (record.id.clone(), slot as usize))
            .collect()
    }

    // ============================================================================
    // Compaction
    // ============================================================================

    /// Compact the database by removing deleted records and reclaiming space.
    ///
    /// This operation:
    /// 1. Removes all tombstoned (deleted) records from storage
    /// 2. Reassigns slot indices to be contiguous
    /// 3. Rebuilds the HNSW index with new slot assignments
    /// 4. Rebuilds the metadata index
    ///
    /// Returns the number of deleted records that were removed.
    ///
    /// # Persistence
    ///
    /// **Important:** Compaction modifies in-memory state only. You MUST call
    /// [`flush()`](Self::flush) after compact() to persist the compacted state.
    /// Without flush, a crash will recover the pre-compaction state from disk.
    ///
    /// # Example
    /// ```ignore
    /// // After deleting many records
    /// db.delete_batch(&old_ids)?;
    ///
    /// // Reclaim space (in-memory only)
    /// let removed = db.compact()?;
    /// println!("Removed {} deleted records", removed);
    ///
    /// // REQUIRED: Persist the compacted state
    /// db.flush()?;
    /// ```
    ///
    /// # Performance
    /// Compaction rebuilds the HNSW index, which is O(n log n) where n is the
    /// number of live records. Call periodically after bulk deletes, not after
    /// every delete.
    pub fn compact(&mut self) -> Result<usize> {
        // Count tombstones before compacting
        let removed_count = self.records.deleted_count() as usize;

        if removed_count == 0 {
            return Ok(0);
        }

        // Compact RecordStore - reassigns slots, clears tombstones
        let old_to_new = self.records.compact();

        // Compact multi-vector storage if present
        if let Some(ref mut multivec_storage) = self.multivec_storage {
            multivec_storage.compact(&old_to_new);
        }

        // Rebuild HNSW index with new contiguous slots
        if self.records.is_empty() {
            self.hnsw_index = None;
        } else {
            self.rebuild_index()?;
        }

        // Rebuild metadata index from compacted records
        self.metadata_index = MetadataIndex::new();
        for (slot, record) in self.records.iter_live() {
            if let Some(ref meta) = record.metadata {
                self.metadata_index.index_json(slot, meta);
            }
        }

        Ok(removed_count)
    }

    // ============================================================================
    // Persistence
    // ============================================================================

    /// Flush all pending changes to disk
    ///
    /// Commits vector/metadata changes and HNSW index to `.omen` storage.
    /// Uses RecordStore as single source of truth (no duplicated state in OmenFile).
    pub fn flush(&mut self) -> Result<()> {
        if let Some(ref mut storage) = self.storage {
            // Ensure dimensions are set in storage header
            let dims = self.records.dimensions();
            if dims > 0 {
                storage.set_dimensions(dims);
            }

            // Persist HNSW parameters to header
            storage.set_hnsw_params(
                self.hnsw_m as u16,
                self.hnsw_ef_construction as u16,
                self.hnsw_ef_search as u16,
            );

            // Export data from RecordStore (single source of truth)
            let vectors = self.records.export_vectors();
            let id_to_slot = self.records.export_id_to_slot();
            let deleted = self.records.export_deleted();
            let metadata = self.records.export_metadata();

            // Serialize HNSW index
            let hnsw_bytes = self
                .hnsw_index
                .as_ref()
                .map(super::hnsw_index::HNSWIndex::to_bytes)
                .transpose()?;

            // Serialize MetadataIndex for fast recovery
            let metadata_index_bytes = self.metadata_index.to_bytes().ok();

            // Export multi-vector data if present
            let (multivec_bytes, multivec_offsets, multivec_config) =
                if let (Some(ref mvs), Some(ref enc)) =
                    (&self.multivec_storage, &self.muvera_encoder)
                {
                    let config = enc.config();
                    (
                        Some(mvs.vectors_to_bytes()),
                        Some(mvs.offsets_to_bytes()),
                        Some((
                            config.repetitions,
                            config.partition_bits,
                            config.seed,
                            enc.token_dimension(),
                        )),
                    )
                } else {
                    (None, None, None)
                };

            // Checkpoint from RecordStore data (not OmenFile's internal state)
            storage.checkpoint_from_snapshot(
                &vectors,
                &id_to_slot,
                &deleted,
                &metadata,
                CheckpointOptions {
                    hnsw_bytes: hnsw_bytes.as_deref(),
                    metadata_index_bytes: metadata_index_bytes.as_deref(),
                    multivec_bytes: multivec_bytes.as_deref(),
                    multivec_offsets: multivec_offsets.as_deref(),
                    multivec_config,
                },
            )?;
        }

        if let Some(ref mut text_index) = self.text_index {
            text_index.commit()?;
        }

        Ok(())
    }

    /// Check if this store has persistent storage enabled
    #[must_use]
    pub fn is_persistent(&self) -> bool {
        self.storage.is_some()
    }

    /// Get reference to the .omen storage backend (if persistent)
    #[must_use]
    pub fn storage(&self) -> Option<&OmenFile> {
        self.storage.as_ref()
    }

    /// Enable persistence for this store (builder pattern).
    ///
    /// Creates or opens an .omen file at the given path. Use `flush()` to persist data.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut store = VectorStore::multi_vector(128);
    /// store = store.persist("my_store.omen")?;
    /// store.store("doc1", tokens, metadata)?;
    /// store.flush()?;
    /// ```
    pub fn persist(mut self, path: impl AsRef<Path>) -> Result<Self> {
        if self.storage.is_some() {
            anyhow::bail!("Store already has persistence enabled");
        }

        let path = path.as_ref();
        let omen_path = OmenFile::compute_omen_path(path);
        let storage = if omen_path.exists() {
            OmenFile::open(path)?
        } else {
            OmenFile::create(path, self.dimensions() as u32)?
        };

        self.storage = Some(storage);
        self.storage_path = Some(path.to_path_buf());
        Ok(self)
    }

}
