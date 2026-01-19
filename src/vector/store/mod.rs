//! Vector storage with HNSW indexing
//!
//! `VectorStore` manages a collection of vectors and provides k-NN search
//! using HNSW (Hierarchical Navigable Small World) algorithm.
//!
//! Optional SQ8 quantization for memory-efficient storage.
//!
//! Optional tantivy-based full-text search for hybrid (vector + BM25) retrieval.

mod filter;
mod options;
mod record_store;
mod thread_safe;

pub use crate::omen::Metric;
pub use filter::MetadataFilter;
pub use options::VectorStoreOptions;
pub use record_store::{Record, RecordStore};
pub use thread_safe::ThreadSafeVectorStore;

// SearchResult is defined in this module and re-exported from lib.rs

use super::hnsw::{HNSWParams, SegmentConfig, SegmentManager};
use super::hnsw_index::HNSWIndex;
use super::types::Vector;
use super::QuantizationMode;
use crate::distance::l2_distance;
use crate::omen::{parse_wal_delete, parse_wal_insert, MetadataIndex, OmenFile, WalEntryType};
use crate::text::{
    weighted_reciprocal_rank_fusion, weighted_reciprocal_rank_fusion_with_subscores, HybridResult,
    TextIndex, TextSearchConfig, DEFAULT_RRF_K,
};
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
// Helper Functions
// ============================================================================

/// Compute effective ef_search value.
///
/// Ensures ef >= k (HNSW requirement) and falls back to default if not specified.
#[inline]
fn compute_effective_ef(ef: Option<usize>, stored_ef: usize, k: usize) -> usize {
    ef.unwrap_or(stored_ef).max(k)
}

#[cfg(test)]
mod stress_tests;
#[cfg(test)]
mod tests;

/// Compute optimal oversample factor based on quantization mode.
///
/// Different quantization modes have different baseline recall:
/// - SQ8: ~99% accurate, needs minimal oversampling (2.0x)
/// - No quantization: 1.0 (rescore disabled)
fn default_oversample_for_quantization(mode: Option<&QuantizationMode>) -> f32 {
    match mode {
        None => 1.0,
        Some(QuantizationMode::SQ8) => 2.0,
    }
}

/// Convert stored quantization mode ID to QuantizationMode.
///
/// Mode IDs: 0=none, 1=sq8
fn quantization_mode_from_id(mode_id: u64) -> Option<QuantizationMode> {
    match mode_id {
        1 => Some(QuantizationMode::SQ8),
        _ => None,
    }
}

/// Convert QuantizationMode to storage mode ID.
fn quantization_mode_to_id(mode: &QuantizationMode) -> u64 {
    match mode {
        QuantizationMode::SQ8 => 1,
    }
}

/// Create HNSW index with quantization mode.
fn create_hnsw_index(
    dimensions: usize,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_ef_search: usize,
    distance_metric: Metric,
    quantization_mode: Option<&QuantizationMode>,
    training_vectors: &[Vec<f32>],
) -> Result<HNSWIndex> {
    use super::hnsw_index::HNSWQuantization;

    let m = hnsw_m.max(DEFAULT_HNSW_M);
    let ef_construction = hnsw_ef_construction.max(DEFAULT_HNSW_EF_CONSTRUCTION);
    let ef_search = hnsw_ef_search.max(DEFAULT_HNSW_EF_SEARCH);

    let quantization = match quantization_mode {
        Some(QuantizationMode::SQ8) => HNSWQuantization::SQ8,
        None => HNSWQuantization::None,
    };

    HNSWIndex::builder()
        .dimensions(dimensions)
        .max_elements(training_vectors.len().max(10_000))
        .m(m)
        .ef_construction(ef_construction)
        .ef_search(ef_search)
        .metric(distance_metric.into())
        .quantization(quantization)
        .build_with_training(training_vectors)
}

/// Rebuild HNSW index maintaining slot-index correspondence
///
/// Inserts vectors in slot order so HNSW indices match RecordStore slots.
/// For deleted slots, inserts zero vectors and marks them deleted.
#[allow(clippy::too_many_arguments)]
fn rebuild_hnsw_with_slots(
    records: &RecordStore,
    deleted: &roaring::RoaringBitmap,
    dimensions: usize,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_ef_search: usize,
    distance_metric: Metric,
    quantization_mode: Option<&QuantizationMode>,
) -> Result<HNSWIndex> {
    // Collect live vectors for training (PQ/SQ codebooks)
    let training_vectors: Vec<Vec<f32>> = records.collect_vectors();

    let mut index = create_hnsw_index(
        dimensions,
        hnsw_m,
        hnsw_ef_construction,
        hnsw_ef_search,
        distance_metric,
        quantization_mode,
        &training_vectors,
    )?;

    // Insert vectors in slot order to maintain index == slot correspondence
    let zero_vector = vec![0.0f32; dimensions];
    let mut deleted_slots = Vec::new();

    for slot in 0..records.slot_count() {
        if deleted.contains(slot) {
            // Insert placeholder for deleted slot, mark deleted after
            index.insert(&zero_vector)?;
            deleted_slots.push(slot);
        } else if let Some(record) = records.get_by_slot(slot) {
            index.insert(&record.vector)?;
        } else {
            // Empty slot without delete marker - shouldn't happen but handle it
            index.insert(&zero_vector)?;
            deleted_slots.push(slot);
        }
    }

    // Mark all deleted slots in HNSW
    if !deleted_slots.is_empty() {
        index.mark_deleted_batch(&deleted_slots)?;
    }

    Ok(index)
}

/// Initialize HNSW index from quantization mode.
#[allow(dead_code)]
fn initialize_quantized_hnsw(
    dimensions: usize,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_ef_search: usize,
    distance_metric: Metric,
    quant_mode: QuantizationMode,
    _training_vectors: &[Vec<f32>],
) -> Result<HNSWIndex> {
    // Note: ef_search is a runtime parameter passed to search(), not stored in HNSWParams
    let _ = hnsw_ef_search; // Silence unused warning - caller passes it to search() at runtime
    let hnsw_params = HNSWParams::default()
        .with_m(hnsw_m)
        .with_ef_construction(hnsw_ef_construction);

    match quant_mode {
        QuantizationMode::SQ8 => {
            HNSWIndex::new_with_sq8(dimensions, hnsw_params, distance_metric.into())
        }
    }
}

/// Initialize standard (non-quantized) HNSW index.
#[allow(dead_code)]
fn initialize_standard_hnsw(
    dimensions: usize,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_ef_search: usize,
    distance_metric: Metric,
    capacity: usize,
) -> Result<HNSWIndex> {
    HNSWIndex::new_with_params(
        capacity,
        dimensions,
        hnsw_m,
        hnsw_ef_construction,
        hnsw_ef_search,
        distance_metric.into(),
    )
}

/// Default empty JSON object for missing metadata.
#[inline]
fn default_metadata() -> JsonValue {
    serde_json::json!({})
}

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
}

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
            quantization_mode_from_id(storage.get_quantization_mode()?.unwrap_or(0));

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
                        Some(rebuild_hnsw_with_slots(
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
                        Some(rebuild_hnsw_with_slots(
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
            Some(rebuild_hnsw_with_slots(
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
            storage.put_quantization_mode(quantization_mode_to_id(q))?;
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
        let oversample_factor = options
            .oversample
            .unwrap_or_else(|| default_oversample_for_quantization(options.quantization.as_ref()));

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
        let oversample_factor = options
            .oversample
            .unwrap_or_else(|| default_oversample_for_quantization(options.quantization.as_ref()));

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
                storage.put_quantization_mode(quantization_mode_to_id(&quant_mode))?;
            }
            initialize_quantized_hnsw(
                dimensions,
                self.hnsw_m,
                self.hnsw_ef_construction,
                self.hnsw_ef_search,
                self.distance_metric,
                quant_mode,
                training_vectors,
            )
        } else {
            initialize_standard_hnsw(
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

        self.set(id, vector, default_metadata())
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
                        storage.put_quantization_mode(quantization_mode_to_id(&quant_mode))?;
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
    // Text Search Methods (Hybrid Search)
    // ============================================================================

    /// Enable text search on this store
    pub fn enable_text_search(&mut self) -> Result<()> {
        self.enable_text_search_with_config(None)
    }

    /// Enable text search with custom configuration
    pub fn enable_text_search_with_config(
        &mut self,
        config: Option<TextSearchConfig>,
    ) -> Result<()> {
        if self.text_index.is_some() {
            return Ok(());
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

    /// Check if text search is enabled
    #[must_use]
    pub fn has_text_search(&self) -> bool {
        self.text_index.is_some()
    }

    /// Upsert vector with text content for hybrid search
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

        text_index.index_document(&id, text)?;
        self.set(id, vector, metadata)
    }

    /// Batch upsert vectors with text content for hybrid search
    pub fn set_batch_with_text(
        &mut self,
        batch: Vec<(String, Vector, String, JsonValue)>,
    ) -> Result<Vec<usize>> {
        let Some(ref mut text_index) = self.text_index else {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        };

        for (id, _, text, _) in &batch {
            text_index.index_document(id, text)?;
        }

        let vector_batch: Vec<(String, Vector, JsonValue)> = batch
            .into_iter()
            .map(|(id, vector, _, metadata)| (id, vector, metadata))
            .collect();

        self.set_batch(vector_batch)
    }

    /// Search text index only (BM25 scoring)
    pub fn text_search(&self, query: &str, k: usize) -> Result<Vec<(String, f32)>> {
        let Some(ref text_index) = self.text_index else {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        };

        text_index.search(query, k)
    }

    /// Hybrid search combining vector similarity and BM25 text relevance
    pub fn hybrid_search(
        &self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        alpha: Option<f32>,
    ) -> Result<Vec<(String, f32, JsonValue)>> {
        self.hybrid_search_with_rrf_k(query_vector, query_text, k, alpha, None)
    }

    /// Hybrid search with configurable RRF k constant
    pub fn hybrid_search_with_rrf_k(
        &self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> Result<Vec<(String, f32, JsonValue)>> {
        if query_vector.data.len() != self.dimensions() {
            anyhow::bail!(
                "Query vector dimension {} does not match store dimension {}",
                query_vector.data.len(),
                self.dimensions()
            );
        }
        if self.text_index.is_none() {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        }

        let fetch_k = k * 2;

        let vector_results = self.knn_search(query_vector, fetch_k)?;
        let vector_results: Vec<(String, f32)> = vector_results
            .into_iter()
            .filter_map(|(idx, distance)| {
                self.records
                    .get_id(idx as u32)
                    .map(|id| (id.to_string(), distance))
            })
            .collect();

        let text_results = self.text_search(query_text, fetch_k)?;

        let fused = weighted_reciprocal_rank_fusion(
            vector_results,
            text_results,
            k,
            rrf_k.unwrap_or(DEFAULT_RRF_K),
            alpha.unwrap_or(0.5),
        );

        Ok(self.attach_metadata(fused))
    }

    /// Hybrid search with filter
    pub fn hybrid_search_with_filter(
        &self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        filter: &MetadataFilter,
        alpha: Option<f32>,
    ) -> Result<Vec<(String, f32, JsonValue)>> {
        self.hybrid_search_with_filter_rrf_k(query_vector, query_text, k, filter, alpha, None)
    }

    /// Hybrid search with filter and configurable RRF k constant
    pub fn hybrid_search_with_filter_rrf_k(
        &self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        filter: &MetadataFilter,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> Result<Vec<(String, f32, JsonValue)>> {
        if query_vector.data.len() != self.dimensions() {
            anyhow::bail!(
                "Query vector dimension {} does not match store dimension {}",
                query_vector.data.len(),
                self.dimensions()
            );
        }
        if self.text_index.is_none() {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        }

        let fetch_k = k * 4;

        let vector_results = self.knn_search_with_filter(query_vector, fetch_k, filter)?;
        let vector_results: Vec<(String, f32)> = vector_results
            .into_iter()
            .map(|r| (r.id, r.distance))
            .collect();

        let text_results = self.text_search(query_text, fetch_k)?;
        let text_results: Vec<(String, f32)> = text_results
            .into_iter()
            .filter(|(id, _)| {
                self.records
                    .get(id)
                    .and_then(|r| r.metadata.as_ref())
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

        Ok(self.attach_metadata(fused))
    }

    /// Attach metadata to fused results
    fn attach_metadata(&self, results: Vec<(String, f32)>) -> Vec<(String, f32, JsonValue)> {
        results
            .into_iter()
            .map(|(id, score)| {
                let metadata = self
                    .records
                    .get(&id)
                    .and_then(|r| r.metadata.clone())
                    .unwrap_or_else(default_metadata);
                (id, score, metadata)
            })
            .collect()
    }

    /// Hybrid search returning separate keyword and semantic scores.
    ///
    /// Returns [`HybridResult`] with `keyword_score` (BM25) and `semantic_score` (vector distance)
    /// for each result, enabling custom post-processing or debugging.
    pub fn hybrid_search_with_subscores(
        &self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> Result<Vec<(HybridResult, JsonValue)>> {
        if query_vector.data.len() != self.dimensions() {
            anyhow::bail!(
                "Query vector dimension {} does not match store dimension {}",
                query_vector.data.len(),
                self.dimensions()
            );
        }
        if self.text_index.is_none() {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        }

        let fetch_k = k * 2;

        let vector_results = self.knn_search(query_vector, fetch_k)?;
        let vector_results: Vec<(String, f32)> = vector_results
            .into_iter()
            .filter_map(|(idx, distance)| {
                self.records
                    .get_id(idx as u32)
                    .map(|id| (id.to_string(), distance))
            })
            .collect();

        let text_results = self.text_search(query_text, fetch_k)?;

        let fused = weighted_reciprocal_rank_fusion_with_subscores(
            vector_results,
            text_results,
            k,
            rrf_k.unwrap_or(DEFAULT_RRF_K),
            alpha.unwrap_or(0.5),
        );

        Ok(self.attach_metadata_to_hybrid_results(fused))
    }

    /// Hybrid search with filter returning separate keyword and semantic scores.
    pub fn hybrid_search_with_filter_subscores(
        &self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        filter: &MetadataFilter,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> Result<Vec<(HybridResult, JsonValue)>> {
        if query_vector.data.len() != self.dimensions() {
            anyhow::bail!(
                "Query vector dimension {} does not match store dimension {}",
                query_vector.data.len(),
                self.dimensions()
            );
        }
        if self.text_index.is_none() {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        }

        let fetch_k = k * 4;

        let vector_results = self.knn_search_with_filter(query_vector, fetch_k, filter)?;
        let vector_results: Vec<(String, f32)> = vector_results
            .into_iter()
            .map(|r| (r.id, r.distance))
            .collect();

        let text_results = self.text_search(query_text, fetch_k)?;
        let text_results: Vec<(String, f32)> = text_results
            .into_iter()
            .filter(|(id, _)| {
                self.records
                    .get(id)
                    .and_then(|r| r.metadata.as_ref())
                    .is_some_and(|meta| filter.matches(meta))
            })
            .collect();

        let fused = weighted_reciprocal_rank_fusion_with_subscores(
            vector_results,
            text_results,
            k,
            rrf_k.unwrap_or(DEFAULT_RRF_K),
            alpha.unwrap_or(0.5),
        );

        Ok(self.attach_metadata_to_hybrid_results(fused))
    }

    /// Attach metadata to hybrid results with subscores
    fn attach_metadata_to_hybrid_results(
        &self,
        results: Vec<HybridResult>,
    ) -> Vec<(HybridResult, JsonValue)> {
        results
            .into_iter()
            .map(|result| {
                let metadata = self
                    .records
                    .get(&result.id)
                    .and_then(|r| r.metadata.clone())
                    .unwrap_or_else(default_metadata);
                (result, metadata)
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
        let metadata = record.metadata.clone().unwrap_or_else(default_metadata);
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
        let effective_ef = compute_effective_ef(ef, self.hnsw_ef_search, k);
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

        let has_data = !self.records.is_empty()
            || self.segments.as_ref().is_some_and(|s| !s.is_empty())
            || self.hnsw_index.as_ref().is_some_and(|idx| !idx.is_empty());

        if !has_data {
            return Ok(Vec::new());
        }

        // Use segments if available (preferred path)
        if let Some(ref segments) = self.segments {
            let segment_results = segments
                .search(&query.data, k, ef)
                .map_err(|e| anyhow::anyhow!("Segment search failed: {e}"))?;

            // Convert SegmentSearchResult to (slot, distance)
            let results: Vec<(usize, f32)> = segment_results
                .into_iter()
                .map(|r| (r.slot as usize, r.distance))
                .collect();

            // Fall back to brute force if segments return nothing but we have data
            if results.is_empty() && self.has_live_vectors() {
                return self.knn_search_brute_force(query, k);
            }
            return Ok(results);
        }

        // Legacy path: use hnsw_index directly
        if let Some(ref index) = self.hnsw_index {
            let results = if index.is_asymmetric() {
                // Rescore if we have storage (fetch from disk) OR records in RAM
                let can_rescore = self.storage.is_some() || !self.records.is_empty();
                if self.rescore_enabled && can_rescore {
                    self.knn_search_with_rescore(query, k, ef)?
                } else {
                    index.search_ef(&query.data, k, ef)?
                }
            } else {
                index.search_ef(&query.data, k, ef)?
            };

            // Fall back to brute force if HNSW returns nothing but we have data
            if results.is_empty() && self.has_live_vectors() {
                return self.knn_search_brute_force(query, k);
            }
            return Ok(results);
        }

        self.knn_search_brute_force(query, k)
    }

    /// K-nearest neighbors search with rescore using original vectors
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

        let oversample_k = ((k as f32) * self.oversample_factor).ceil() as usize;
        let candidates = index.search_ef(&query.data, oversample_k, ef)?;

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Rescore candidates with exact L2 distance from RecordStore (source of truth)
        let mut rescored: Vec<(usize, f32)> = candidates
            .iter()
            .filter_map(|&(id, _quantized_dist)| {
                self.records
                    .get_vector(id as u32)
                    .map(|v| (id, l2_distance(&query.data, v)))
            })
            .collect();

        rescored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        rescored.truncate(k);

        Ok(rescored)
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
        // Use provided ef, or fall back to stored hnsw_ef_search
        // Ensure ef >= k (HNSW requirement)
        let effective_ef = compute_effective_ef(ef, self.hnsw_ef_search, k);

        // Try bitmap-based filtering (O(1) per candidate)
        let filter_bitmap = filter.evaluate_bitmap(&self.metadata_index);

        if let Some(ref hnsw) = self.hnsw_index {
            let records = &self.records;

            let search_results = if let Some(ref bitmap) = filter_bitmap {
                // Fast path: bitmap-based filtering
                let filter_fn =
                    |node_id: u32| -> bool { records.is_live(node_id) && bitmap.contains(node_id) };
                hnsw.search_with_filter_ef(&query.data, k, Some(effective_ef), filter_fn)?
            } else {
                // Slow path: JSON-based filtering
                let filter_fn = |node_id: u32| -> bool {
                    if !records.is_live(node_id) {
                        return false;
                    }
                    let metadata = records
                        .get_by_slot(node_id)
                        .and_then(|r| r.metadata.clone())
                        .unwrap_or_else(default_metadata);
                    filter.matches(&metadata)
                };
                hnsw.search_with_filter_ef(&query.data, k, Some(effective_ef), filter_fn)?
            };

            let filtered_results: Vec<SearchResult> = search_results
                .into_iter()
                .filter_map(|(slot, distance)| {
                    let record = self.records.get_by_slot(slot as u32)?;
                    let metadata = record.metadata.clone().unwrap_or_else(default_metadata);
                    Some(SearchResult::new(record.id.clone(), distance, metadata))
                })
                .collect();

            return Ok(filtered_results);
        }

        // Fallback: brute-force search with filtering
        let mut all_results: Vec<SearchResult> = self
            .records
            .iter_live()
            .filter_map(|(slot, record)| {
                // Use bitmap if available, otherwise JSON
                let passes_filter = if let Some(ref bitmap) = filter_bitmap {
                    bitmap.contains(slot)
                } else {
                    let metadata = record.metadata.clone().unwrap_or_else(default_metadata);
                    filter.matches(&metadata)
                };

                if !passes_filter {
                    return None;
                }

                let metadata = record.metadata.clone().unwrap_or_else(default_metadata);
                let distance = l2_distance(&query.data, &record.vector);
                Some(SearchResult::new(record.id.clone(), distance, metadata))
            })
            .collect();

        all_results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        all_results.truncate(k);

        Ok(all_results)
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
            let results = self.knn_search_readonly(query, k, ef)?;
            let filtered: Vec<SearchResult> = results
                .into_iter()
                .filter_map(|(slot, distance)| {
                    let record = self.records.get_by_slot(slot as u32)?;
                    let metadata = record.metadata.clone().unwrap_or_else(default_metadata);
                    Some(SearchResult::new(record.id.clone(), distance, metadata))
                })
                .collect();

            // Fall back to brute force if HNSW results were all deleted
            if filtered.is_empty() && self.has_live_vectors() {
                self.knn_search_brute_force_with_metadata(query, k)?
            } else {
                filtered
            }
        };

        if let Some(max_dist) = max_distance {
            results.retain(|r| r.distance <= max_dist);
        }

        Ok(results)
    }

    /// Check if there are any non-deleted vectors
    fn has_live_vectors(&self) -> bool {
        !self.records.is_empty()
    }

    /// Brute-force search with metadata (fallback for orphaned nodes)
    fn knn_search_brute_force_with_metadata(
        &self,
        query: &Vector,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let results = self.knn_search_brute_force(query, k)?;
        Ok(results
            .into_iter()
            .filter_map(|(slot, distance)| {
                let record = self.records.get_by_slot(slot as u32)?;
                let metadata = record.metadata.clone().unwrap_or_else(default_metadata);
                Some(SearchResult::new(record.id.clone(), distance, metadata))
            })
            .collect())
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
        let effective_ef = compute_effective_ef(ef, self.hnsw_ef_search, k);
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

        // Brute force search using RecordStore
        if self.records.is_empty() {
            return Ok(Vec::new());
        }

        let mut distances: Vec<(usize, f32)> = self
            .records
            .iter_live()
            .map(|(slot, record)| {
                let dist = l2_distance(&query.data, &record.vector);
                (slot as usize, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.total_cmp(&b.1));
        Ok(distances.into_iter().take(k).collect())
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
        let _old_to_new = self.records.compact();

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

            // Checkpoint from RecordStore data (not OmenFile's internal state)
            storage.checkpoint_from_snapshot(
                &vectors,
                &id_to_slot,
                &deleted,
                &metadata,
                hnsw_bytes.as_deref(),
                metadata_index_bytes.as_deref(),
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
}
