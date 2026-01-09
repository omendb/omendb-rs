//! Vector storage with HNSW indexing
//!
//! `VectorStore` manages a collection of vectors and provides k-NN search
//! using HNSW (Hierarchical Navigable Small World) algorithm.
//!
//! Optional Extended `RaBitQ` quantization for memory-efficient storage.
//!
//! Optional tantivy-based full-text search for hybrid (vector + BM25) retrieval.

mod filter;
mod options;

pub use crate::omen::Metric;
pub use filter::MetadataFilter;
pub use options::VectorStoreOptions;

use super::hnsw::HNSWParams;
use super::hnsw_index::HNSWIndex;
use super::types::Vector;
use super::QuantizationMode;
use crate::compression::{QuantizationBits, RaBitQParams};
use crate::distance::l2_distance;
use crate::omen::{MetadataIndex, OmenFile};
use crate::text::{
    weighted_reciprocal_rank_fusion, weighted_reciprocal_rank_fusion_with_subscores, HybridResult,
    TextIndex, TextSearchConfig, DEFAULT_RRF_K,
};
use anyhow::Result;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
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

/// Assert ID mapping consistency (debug builds only).
///
/// Verifies that id_to_index and index_to_id are inverse mappings.
#[cfg(debug_assertions)]
fn debug_assert_mapping_consistency(
    id_to_index: &FxHashMap<String, usize>,
    index_to_id: &FxHashMap<usize, String>,
) {
    // Both maps must have same size
    debug_assert_eq!(
        id_to_index.len(),
        index_to_id.len(),
        "ID mapping size mismatch: id_to_index={}, index_to_id={}",
        id_to_index.len(),
        index_to_id.len()
    );

    // Every entry in id_to_index must have inverse in index_to_id
    for (id, &idx) in id_to_index {
        debug_assert_eq!(
            index_to_id.get(&idx),
            Some(id),
            "Mapping inconsistency: id_to_index[{id}]={idx} but index_to_id[{idx}]={:?}",
            index_to_id.get(&idx)
        );
    }
}

#[cfg(not(debug_assertions))]
#[inline]
fn debug_assert_mapping_consistency(
    _id_to_index: &FxHashMap<String, usize>,
    _index_to_id: &FxHashMap<usize, String>,
) {
    // No-op in release builds
}

#[cfg(test)]
mod tests;

/// Compute optimal oversample factor based on quantization mode.
///
/// Different quantization modes have different baseline recall:
/// - Binary: ~85% accurate, needs higher oversampling (5.0x)
/// - SQ8: ~99% accurate, needs minimal oversampling (2.0x)
/// - RaBitQ 2-bit: ~93% accurate, needs more candidates (4.0x)
/// - RaBitQ 4-bit: ~96% accurate, moderate oversampling (3.0x)
/// - RaBitQ 8-bit: ~99% accurate, minimal oversampling (2.0x)
/// - No quantization: 1.0 (rescore disabled, oversample unused)
fn default_oversample_for_quantization(mode: Option<&QuantizationMode>) -> f32 {
    match mode {
        None => 1.0,
        Some(QuantizationMode::Binary) => 5.0, // ~85% recall baseline
        Some(QuantizationMode::SQ8) => 2.0,
        Some(QuantizationMode::RaBitQ(params)) => match params.bits_per_dim.to_u8() {
            2 => 4.0, // ~93% recall baseline
            8 => 2.0, // ~99% recall baseline
            _ => 3.0, // 4-bit default: ~96% recall baseline
        },
    }
}

/// Convert stored quantization mode ID to QuantizationMode.
///
/// Mode IDs: 0=none, 1=sq8, 2=rabitq-4, 3=rabitq-2, 4=rabitq-8, 5=binary
fn quantization_mode_from_id(mode_id: u64) -> Option<QuantizationMode> {
    match mode_id {
        1 => Some(QuantizationMode::SQ8),
        2 => Some(QuantizationMode::RaBitQ(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits4,
            ..RaBitQParams::default()
        })),
        3 => Some(QuantizationMode::RaBitQ(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits2,
            ..RaBitQParams::default()
        })),
        4 => Some(QuantizationMode::RaBitQ(RaBitQParams {
            bits_per_dim: QuantizationBits::Bits8,
            ..RaBitQParams::default()
        })),
        5 => Some(QuantizationMode::Binary),
        _ => None, // 0 and unknown values
    }
}

/// Convert QuantizationMode to storage mode ID.
///
/// Mode IDs: 0=none, 1=sq8, 2=rabitq-4, 3=rabitq-2, 4=rabitq-8, 5=binary
fn quantization_mode_to_id(mode: &QuantizationMode) -> u64 {
    match mode {
        QuantizationMode::Binary => 5,
        QuantizationMode::SQ8 => 1,
        QuantizationMode::RaBitQ(p) => match p.bits_per_dim.to_u8() {
            2 => 3,
            8 => 4,
            _ => 2, // 4-bit default
        },
    }
}

/// Create HNSW index with proper quantization mode.
///
/// This ensures rebuilt indexes preserve the original quantization settings.
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

    // Ensure minimum values for HNSW parameters
    let m = hnsw_m.max(DEFAULT_HNSW_M);
    let ef_construction = hnsw_ef_construction.max(DEFAULT_HNSW_EF_CONSTRUCTION);
    let ef_search = hnsw_ef_search.max(DEFAULT_HNSW_EF_SEARCH);

    // Convert QuantizationMode to HNSWQuantization
    let quantization = match quantization_mode {
        Some(QuantizationMode::Binary) => HNSWQuantization::Binary,
        Some(QuantizationMode::SQ8) => HNSWQuantization::SQ8,
        Some(QuantizationMode::RaBitQ(params)) => HNSWQuantization::RaBitQ(params.clone()),
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

/// Initialize HNSW index from pending quantization mode.
fn initialize_quantized_hnsw(
    dimensions: usize,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_ef_search: usize,
    distance_metric: Metric,
    quant_mode: QuantizationMode,
    training_vectors: &[Vec<f32>],
) -> Result<HNSWIndex> {
    let hnsw_params = HNSWParams::default()
        .with_m(hnsw_m)
        .with_ef_construction(hnsw_ef_construction)
        .with_ef_search(hnsw_ef_search);

    match quant_mode {
        QuantizationMode::Binary => {
            let mut idx =
                HNSWIndex::new_with_binary(dimensions, hnsw_params, distance_metric.into())?;
            idx.train_quantizer(training_vectors)?;
            Ok(idx)
        }
        QuantizationMode::SQ8 => {
            HNSWIndex::new_with_sq8(dimensions, hnsw_params, distance_metric.into())
        }
        QuantizationMode::RaBitQ(params) => {
            let mut idx = HNSWIndex::new_with_asymmetric(
                dimensions,
                hnsw_params,
                distance_metric.into(),
                params,
            )?;
            idx.train_quantizer(training_vectors)?;
            Ok(idx)
        }
    }
}

/// Initialize standard (non-quantized) HNSW index.
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
    pub id_to_index: FxHashMap<String, usize>,

    /// Reverse map from internal indices to string IDs (O(1) lookup for search results)
    index_to_id: FxHashMap<usize, String>,

    /// Deleted vector IDs (tombstones for MVCC)
    deleted: HashMap<usize, bool>,

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

    /// Next available index for vectors (reliable counter even when skip_ram enabled)
    next_index: usize,
}

impl VectorStore {
    // ============================================================================
    // Constructors
    // ============================================================================

    /// Create new vector store
    #[must_use]
    pub fn new(dimensions: usize) -> Self {
        Self {
            vectors: Vec::new(),
            hnsw_index: None,
            dimensions,
            rescore_enabled: false,
            oversample_factor: DEFAULT_OVERSAMPLE_FACTOR,
            metadata: HashMap::new(),
            id_to_index: FxHashMap::default(),
            index_to_id: FxHashMap::default(),
            deleted: HashMap::new(),
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
            next_index: 0,
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
            oversample_factor: DEFAULT_OVERSAMPLE_FACTOR,
            metadata: HashMap::new(),
            id_to_index: FxHashMap::default(),
            index_to_id: FxHashMap::default(),
            deleted: HashMap::new(),
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
            next_index: 0,
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
            vectors: Vec::new(),
            hnsw_index,
            dimensions,
            rescore_enabled: false,
            oversample_factor: DEFAULT_OVERSAMPLE_FACTOR,
            metadata: HashMap::new(),
            id_to_index: FxHashMap::default(),
            index_to_id: FxHashMap::default(),
            deleted: HashMap::new(),
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
            next_index: 0,
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
        let path = path.as_ref();
        let omen_path = OmenFile::compute_omen_path(path);
        let storage = if omen_path.exists() {
            OmenFile::open(path)?
        } else {
            OmenFile::create(path, 0)?
        };

        // Check if store was quantized - if so, skip loading vectors to RAM
        let is_quantized = storage.is_quantized()?;
        let quantization_mode =
            quantization_mode_from_id(storage.get_quantization_mode()?.unwrap_or(0));

        // Load metadata and mappings (always needed)
        let metadata = storage.load_all_metadata()?;
        let id_to_index: FxHashMap<String, usize> =
            storage.load_all_id_mappings()?.into_iter().collect();
        let deleted = storage.load_all_deleted()?;

        // Get dimensions from config
        let dimensions = storage.get_config("dimensions")?.unwrap_or(0) as usize;

        // Get HNSW parameters from header (for rebuilding HNSW if needed)
        let header = storage.header();
        let distance_metric = header.distance_fn;
        let hnsw_m = header.m as usize;
        let hnsw_ef_construction = header.ef_construction as usize;
        let hnsw_ef_search = header.ef_search as usize;

        // Load vectors to RAM only if NOT quantized
        let (vectors, real_indices) = if is_quantized {
            (Vec::new(), std::collections::HashSet::new())
        } else {
            let vectors_data = storage.load_all_vectors()?;
            let mut vectors: Vec<Vector> = Vec::new();
            let mut real_indices: std::collections::HashSet<usize> =
                std::collections::HashSet::new();

            for (id, data) in &vectors_data {
                while vectors.len() < *id {
                    vectors.push(Vector::new(vec![0.0; dimensions.max(1)]));
                }
                vectors.push(Vector::new(data.clone()));
                real_indices.insert(*id);
            }
            (vectors, real_indices)
        };

        // Mark gap-filled vectors as deleted
        let mut deleted = deleted;
        for idx in 0..vectors.len() {
            if !real_indices.contains(&idx) && !deleted.contains_key(&idx) {
                deleted.insert(idx, true);
            }
        }

        // Load or rebuild HNSW index
        // Count non-deleted vectors
        let active_vector_count = vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| !deleted.contains_key(i))
            .count();

        let hnsw_index = if let Some(hnsw_bytes) = storage.get_hnsw_index() {
            match postcard::from_bytes::<HNSWIndex>(hnsw_bytes) {
                Ok(index) => {
                    // Check if HNSW index matches loaded vectors (WAL recovery may add more)
                    if index.len() != active_vector_count && !vectors.is_empty() {
                        tracing::info!(
                            "HNSW index count ({}) differs from vector count ({}), rebuilding",
                            index.len(),
                            active_vector_count
                        );
                        let vector_data: Vec<Vec<f32>> =
                            vectors.iter().map(|v| v.data.clone()).collect();
                        let mut new_index = create_hnsw_index(
                            dimensions,
                            hnsw_m,
                            hnsw_ef_construction,
                            hnsw_ef_search,
                            distance_metric,
                            quantization_mode.as_ref(),
                            &vector_data,
                        )?;
                        new_index.batch_insert(&vector_data)?;
                        Some(new_index)
                    } else {
                        Some(index)
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to deserialize HNSW index, rebuilding: {}", e);
                    None
                }
            }
        } else if !vectors.is_empty() {
            let vector_data: Vec<Vec<f32>> = vectors.iter().map(|v| v.data.clone()).collect();
            let mut index = create_hnsw_index(
                dimensions,
                hnsw_m,
                hnsw_ef_construction,
                hnsw_ef_search,
                distance_metric,
                quantization_mode.as_ref(),
                &vector_data,
            )?;
            index.batch_insert(&vector_data)?;
            Some(index)
        } else if is_quantized && dimensions > 0 {
            let vectors_data = storage.load_all_vectors()?;
            if vectors_data.is_empty() {
                None
            } else {
                let vector_data: Vec<Vec<f32>> =
                    vectors_data.iter().map(|(_, v)| v.clone()).collect();
                let mut index = create_hnsw_index(
                    dimensions,
                    hnsw_m,
                    hnsw_ef_construction,
                    hnsw_ef_search,
                    distance_metric,
                    quantization_mode.as_ref(),
                    &vector_data,
                )?;
                index.batch_insert(&vector_data)?;
                Some(index)
            }
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

        // Build reverse map for O(1) index→id lookup
        let index_to_id: FxHashMap<usize, String> = id_to_index
            .iter()
            .map(|(id, &idx)| (idx, id.clone()))
            .collect();

        // Build metadata index from loaded metadata (for fast filtered search)
        let mut metadata_index = MetadataIndex::new();
        for (&idx, meta) in &metadata {
            if !deleted.contains_key(&idx) {
                metadata_index.index_json(idx as u32, meta);
            }
        }

        // Enable rescore if the loaded index is quantized
        let rescore_enabled = hnsw_index
            .as_ref()
            .is_some_and(super::hnsw_index::HNSWIndex::is_asymmetric);

        // Verify mapping consistency before returning
        debug_assert_mapping_consistency(&id_to_index, &index_to_id);

        // Calculate next_index from loaded mappings (max index + 1)
        let next_index = id_to_index.values().max().map_or(0, |&max| max + 1);

        Ok(Self {
            vectors,
            hnsw_index,
            dimensions,
            rescore_enabled,
            oversample_factor: DEFAULT_OVERSAMPLE_FACTOR,
            metadata,
            id_to_index,
            index_to_id,
            deleted,
            metadata_index,
            storage: Some(storage),
            storage_path: Some(path.to_path_buf()),
            text_index,
            text_search_config: None,
            pending_quantization: None,
            hnsw_m: hnsw_m.max(DEFAULT_HNSW_M),
            hnsw_ef_construction: hnsw_ef_construction.max(DEFAULT_HNSW_EF_CONSTRUCTION),
            hnsw_ef_search: hnsw_ef_search.max(DEFAULT_HNSW_EF_SEARCH),
            distance_metric,
            next_index,
        })
    }

    /// Open a persistent vector store with specified dimensions
    ///
    /// Like `open()` but ensures dimensions are set for new databases.
    pub fn open_with_dimensions(path: impl AsRef<Path>, dimensions: usize) -> Result<Self> {
        let mut store = Self::open(path)?;
        if store.dimensions == 0 {
            store.dimensions = dimensions;
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
            if store.dimensions == 0 && options.dimensions > 0 {
                store.dimensions = options.dimensions;
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
            vectors: Vec::new(),
            hnsw_index,
            dimensions,
            rescore_enabled,
            oversample_factor,
            metadata: HashMap::new(),
            id_to_index: FxHashMap::default(),
            index_to_id: FxHashMap::default(),
            deleted: HashMap::new(),
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
            next_index: 0,
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
            vectors: Vec::new(),
            hnsw_index,
            dimensions,
            rescore_enabled,
            oversample_factor,
            metadata: HashMap::new(),
            id_to_index: FxHashMap::default(),
            index_to_id: FxHashMap::default(),
            deleted: HashMap::new(),
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
            next_index: 0,
        })
    }

    // ============================================================================
    // Private Helpers
    // ============================================================================

    /// Resolve dimensions from vector or existing store config.
    fn resolve_dimensions(&self, vector_dim: usize) -> Result<usize> {
        if self.dimensions == 0 {
            Ok(vector_dim)
        } else if vector_dim != self.dimensions {
            anyhow::bail!(
                "Vector dimension mismatch: store expects {}, got {}",
                self.dimensions,
                vector_dim
            );
        } else {
            Ok(self.dimensions)
        }
    }

    /// Create initial HNSW index, handling pending quantization.
    fn create_initial_hnsw(
        &mut self,
        dimensions: usize,
        training_vectors: &[Vec<f32>],
    ) -> Result<HNSWIndex> {
        self.create_initial_hnsw_with_capacity(dimensions, training_vectors, 10_000)
    }

    /// Create initial HNSW index with custom capacity.
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

    /// Insert vector and return its ID
    pub fn insert(&mut self, vector: Vector) -> Result<usize> {
        let id = self.next_index;

        if self.hnsw_index.is_none() {
            let dimensions = self.resolve_dimensions(vector.dim())?;
            self.hnsw_index =
                Some(self.create_initial_hnsw(dimensions, std::slice::from_ref(&vector.data))?);
            self.dimensions = dimensions;
        } else if vector.dim() != self.dimensions {
            anyhow::bail!(
                "Vector dimension mismatch: store expects {}, got {}. All vectors in same store must have same dimension.",
                self.dimensions,
                vector.dim()
            );
        }

        if let Some(ref mut index) = self.hnsw_index {
            index.insert(&vector.data)?;
        }

        if let Some(ref mut storage) = self.storage {
            storage.put_vector(id, &vector.data)?;
            storage.increment_count()?;
            if id == 0 {
                storage.put_config("dimensions", self.dimensions as u64)?;
            }
        }

        if !self.is_quantized() || self.storage.is_none() {
            self.vectors.push(vector);
        }

        // Increment next_index for the next insert
        self.next_index += 1;

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
        if self.id_to_index.contains_key(&id) {
            anyhow::bail!("Vector with ID '{id}' already exists. Use set() to update.");
        }

        let index = self.insert(vector)?;

        self.metadata.insert(index, metadata.clone());
        self.metadata_index.index_json(index as u32, &metadata);
        self.id_to_index.insert(id.clone(), index);
        self.index_to_id.insert(index, id.clone());

        // Verify mapping consistency
        debug_assert_mapping_consistency(&self.id_to_index, &self.index_to_id);

        if let Some(ref mut storage) = self.storage {
            storage.put_metadata(index, &metadata)?;
            storage.put_id_mapping(&id, index)?;
        }

        Ok(index)
    }

    /// Upsert vector (insert or update) with string ID and metadata
    ///
    /// This is the recommended method for most use cases.
    pub fn set(&mut self, id: String, vector: Vector, metadata: JsonValue) -> Result<usize> {
        if let Some(&index) = self.id_to_index.get(&id) {
            self.update_by_index(index, Some(vector), Some(metadata))?;
            Ok(index)
        } else {
            self.insert_with_metadata(id, vector, metadata)
        }
    }

    /// Batch set vectors (insert or update multiple vectors at once)
    ///
    /// This is the recommended method for bulk operations.
    pub fn set_batch(&mut self, batch: Vec<(String, Vector, JsonValue)>) -> Result<Vec<usize>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        // Separate batch into updates and inserts
        let mut updates: Vec<(usize, Vector, JsonValue)> = Vec::new();
        let mut inserts: Vec<(String, Vector, JsonValue)> = Vec::new();

        for (id, vector, metadata) in batch {
            if let Some(&index) = self.id_to_index.get(&id) {
                updates.push((index, vector, metadata));
            } else {
                inserts.push((id, vector, metadata));
            }
        }

        let mut result_indices = Vec::new();

        // Process updates first
        for (index, vector, metadata) in updates {
            self.update_by_index(index, Some(vector), Some(metadata))?;
            result_indices.push(index);
        }

        if !inserts.is_empty() {
            let vectors_data: Vec<Vec<f32>> =
                inserts.iter().map(|(_, v, _)| v.data.clone()).collect();

            if self.hnsw_index.is_none() {
                let dimensions = self.resolve_dimensions(inserts[0].1.dim())?;
                self.hnsw_index = Some(self.create_initial_hnsw(dimensions, &vectors_data)?);
                self.dimensions = dimensions;
            }

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

            let base_index = self.next_index;
            let insert_count = inserts.len();
            if let Some(ref mut index) = self.hnsw_index {
                index.batch_insert(&vectors_data)?;
            }

            // Batch persist to storage
            if let Some(ref mut storage) = self.storage {
                if base_index == 0 {
                    storage.put_config("dimensions", self.dimensions as u64)?;
                }

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

                storage.put_batch(batch_items)?;
            }

            // Add vectors to in-memory structures
            // Skip RAM storage when quantized with disk storage (fetch from disk for rescore)
            let skip_ram = self.is_quantized() && self.storage.is_some();
            for (i, (id, vector, metadata)) in inserts.into_iter().enumerate() {
                let idx = base_index + i;
                if !skip_ram {
                    self.vectors.push(vector);
                }
                self.metadata.insert(idx, metadata.clone());
                self.metadata_index.index_json(idx as u32, &metadata);
                self.index_to_id.insert(idx, id.clone());
                self.id_to_index.insert(id, idx);
                result_indices.push(idx);
            }

            // Update next_index counter
            self.next_index += insert_count;

            // Verify mapping consistency after batch insert
            debug_assert_mapping_consistency(&self.id_to_index, &self.index_to_id);
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
        &mut self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        alpha: Option<f32>,
    ) -> Result<Vec<(String, f32, JsonValue)>> {
        self.hybrid_search_with_rrf_k(query_vector, query_text, k, alpha, None)
    }

    /// Hybrid search with configurable RRF k constant
    pub fn hybrid_search_with_rrf_k(
        &mut self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> Result<Vec<(String, f32, JsonValue)>> {
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

        let fetch_k = k * 2;

        let vector_results = self.knn_search(query_vector, fetch_k)?;
        let vector_results: Vec<(String, f32)> = vector_results
            .into_iter()
            .filter_map(|(idx, distance)| {
                self.index_to_id.get(&idx).map(|id| (id.clone(), distance))
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
        &mut self,
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
        &mut self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        filter: &MetadataFilter,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> Result<Vec<(String, f32, JsonValue)>> {
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

        let fetch_k = k * 4;

        let vector_results = self.knn_search_with_filter(query_vector, fetch_k, filter)?;
        let vector_results: Vec<(String, f32)> = vector_results
            .into_iter()
            .filter_map(|(idx, distance, _)| {
                self.index_to_id.get(&idx).map(|id| (id.clone(), distance))
            })
            .collect();

        let text_results = self.text_search(query_text, fetch_k)?;
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

        Ok(self.attach_metadata(fused))
    }

    /// Attach metadata to fused results
    fn attach_metadata(&self, results: Vec<(String, f32)>) -> Vec<(String, f32, JsonValue)> {
        results
            .into_iter()
            .map(|(id, score)| {
                let metadata = self
                    .id_to_index
                    .get(&id)
                    .and_then(|&idx| self.metadata.get(&idx))
                    .cloned()
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
        &mut self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> Result<Vec<(HybridResult, JsonValue)>> {
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

        let fetch_k = k * 2;

        let vector_results = self.knn_search(query_vector, fetch_k)?;
        let vector_results: Vec<(String, f32)> = vector_results
            .into_iter()
            .filter_map(|(idx, distance)| {
                self.index_to_id.get(&idx).map(|id| (id.clone(), distance))
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
        &mut self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        filter: &MetadataFilter,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> Result<Vec<(HybridResult, JsonValue)>> {
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

        let fetch_k = k * 4;

        let vector_results = self.knn_search_with_filter(query_vector, fetch_k, filter)?;
        let vector_results: Vec<(String, f32)> = vector_results
            .into_iter()
            .filter_map(|(idx, distance, _)| {
                self.index_to_id.get(&idx).map(|id| (id.clone(), distance))
            })
            .collect();

        let text_results = self.text_search(query_text, fetch_k)?;
        let text_results: Vec<(String, f32)> = text_results
            .into_iter()
            .filter(|(id, _)| {
                self.id_to_index
                    .get(id)
                    .and_then(|&idx| self.metadata.get(&idx))
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
                    .id_to_index
                    .get(&result.id)
                    .and_then(|&idx| self.metadata.get(&idx))
                    .cloned()
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
        // Use next_index for bounds check (works for quantized stores where vectors is empty)
        if index >= self.next_index {
            anyhow::bail!("Vector index {index} does not exist");
        }
        if self.deleted.contains_key(&index) {
            anyhow::bail!("Vector index {index} has been deleted");
        }

        if let Some(new_vector) = vector {
            if new_vector.dim() != self.dimensions {
                anyhow::bail!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.dimensions,
                    new_vector.dim()
                );
            }

            // Update in RAM if vectors are stored there (non-quantized or in-memory mode)
            if let Some(v) = self.vectors.get_mut(index) {
                *v = new_vector.clone();
            }

            if let Some(ref mut storage) = self.storage {
                storage.put_vector(index, &new_vector.data)?;
            }
        }

        if let Some(ref new_metadata) = metadata {
            // Re-index metadata: remove old values, add new ones
            self.metadata_index.remove(index as u32);
            self.metadata_index.index_json(index as u32, new_metadata);
            self.metadata.insert(index, new_metadata.clone());

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
        let index = self
            .id_to_index
            .get(id)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Vector with ID '{id}' not found"))?;

        self.update_by_index(index, vector, metadata)
    }

    /// Delete vector by string ID
    ///
    /// This method:
    /// 1. Marks the vector as deleted (soft delete)
    /// 2. Repairs the HNSW graph using MN-RU algorithm to maintain recall quality
    /// 3. Removes from text index if present
    /// 4. Persists to WAL
    pub fn delete(&mut self, id: &str) -> Result<()> {
        let index = self
            .id_to_index
            .get(id)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Vector with ID '{id}' not found"))?;

        self.deleted.insert(index, true);
        self.metadata_index.remove(index as u32);

        // Repair HNSW graph using MN-RU algorithm
        // This maintains graph connectivity and recall quality after deletion
        if let Some(ref mut hnsw) = self.hnsw_index {
            if let Err(e) = hnsw.mark_deleted(index as u32) {
                tracing::warn!(
                    id = id,
                    index = index,
                    error = ?e,
                    "Failed to repair HNSW graph after deletion"
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

        self.id_to_index.remove(id);
        self.index_to_id.remove(&index);

        // Verify mapping consistency
        debug_assert_mapping_consistency(&self.id_to_index, &self.index_to_id);

        Ok(())
    }

    /// Delete multiple vectors by string IDs
    ///
    /// Uses batch MN-RU graph repair for better efficiency than individual deletes.
    pub fn delete_batch(&mut self, ids: &[String]) -> Result<usize> {
        // Collect valid indices first
        let mut node_ids: Vec<u32> = Vec::with_capacity(ids.len());
        let mut valid_ids: Vec<String> = Vec::with_capacity(ids.len());

        for id in ids {
            if let Some(&index) = self.id_to_index.get(id) {
                self.deleted.insert(index, true);
                self.metadata_index.remove(index as u32);
                node_ids.push(index as u32);
                valid_ids.push(id.clone());
            }
        }

        // Batch repair HNSW graph using MN-RU algorithm
        if !node_ids.is_empty() {
            if let Some(ref mut hnsw) = self.hnsw_index {
                if let Err(e) = hnsw.mark_deleted_batch(&node_ids) {
                    tracing::warn!(
                        count = node_ids.len(),
                        error = ?e,
                        "Failed to batch repair HNSW graph after deletion"
                    );
                }
            }
        }

        // Persist deletions and clean up mappings
        for (id, &node_id) in valid_ids.iter().zip(node_ids.iter()) {
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
            self.id_to_index.remove(id);
            self.index_to_id.remove(&(node_id as usize));
        }

        // Verify mapping consistency
        debug_assert_mapping_consistency(&self.id_to_index, &self.index_to_id);

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
            .id_to_index
            .iter()
            .filter_map(|(id, &idx)| {
                if self.deleted.contains_key(&idx) {
                    return None;
                }
                let metadata = self.metadata.get(&idx)?;
                if filter.matches(metadata) {
                    Some(id.clone())
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
        self.id_to_index
            .iter()
            .filter(|(_, &idx)| {
                if self.deleted.contains_key(&idx) {
                    return false;
                }
                self.metadata
                    .get(&idx)
                    .is_some_and(|metadata| filter.matches(metadata))
            })
            .count()
    }

    /// Get vector by string ID
    ///
    /// Returns owned data since vectors may be loaded from disk for quantized stores.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<(Vector, JsonValue)> {
        let &index = self.id_to_index.get(id)?;
        if self.deleted.contains_key(&index) {
            return None;
        }

        // Try in-memory vectors first
        if let Some(vec) = self.vectors.get(index) {
            return self
                .metadata
                .get(&index)
                .map(|meta| (vec.clone(), meta.clone()));
        }

        // Fall back to storage for quantized stores (vectors not in RAM)
        if let Some(ref storage) = self.storage {
            if let Ok(Some(vec_data)) = storage.get_vector(index) {
                return self
                    .metadata
                    .get(&index)
                    .map(|meta| (Vector::new(vec_data), meta.clone()));
            }
        }

        None
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
        self.id_to_index.get(id).and_then(|&index| {
            if self.deleted.contains_key(&index) {
                return None;
            }
            self.metadata.get(&index)
        })
    }

    // ============================================================================
    // Batch Insert / Index Rebuild
    // ============================================================================

    /// Insert batch of vectors in parallel
    pub fn batch_insert(&mut self, vectors: Vec<Vector>) -> Result<Vec<usize>> {
        const CHUNK_SIZE: usize = 10_000;

        if vectors.is_empty() {
            return Ok(Vec::new());
        }

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

        if self.hnsw_index.is_none() {
            let training: Vec<Vec<f32>> = vectors.iter().map(|v| v.data.clone()).collect();
            let capacity = vectors.len().max(1_000_000);
            self.hnsw_index = Some(self.create_initial_hnsw_with_capacity(
                self.dimensions,
                &training,
                capacity,
            )?);
        }

        let mut all_ids = Vec::with_capacity(vectors.len());
        for chunk in vectors.chunks(CHUNK_SIZE) {
            let vector_data: Vec<Vec<f32>> = chunk.iter().map(|v| v.data.clone()).collect();
            if let Some(ref mut index) = self.hnsw_index {
                all_ids.extend(index.batch_insert(&vector_data)?);
            }
        }

        self.vectors.extend(vectors);
        Ok(all_ids)
    }

    /// Rebuild HNSW index from existing vectors
    pub fn rebuild_index(&mut self) -> Result<()> {
        if self.vectors.is_empty() {
            return Ok(());
        }

        let mut index = HNSWIndex::new_with_params(
            self.vectors.len().max(1_000_000),
            self.dimensions,
            self.hnsw_m,
            self.hnsw_ef_construction,
            self.hnsw_ef_search,
            self.distance_metric.into(),
        )?;

        for vector in &self.vectors {
            index.insert(&vector.data)?;
        }

        self.hnsw_index = Some(index);
        Ok(())
    }

    /// Merge another `VectorStore` into this one using IGTM algorithm
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

        if self.hnsw_index.is_none() {
            let capacity = (self.vectors.len() + other.vectors.len()).max(1_000_000);
            self.hnsw_index = Some(HNSWIndex::new_with_params(
                capacity,
                self.dimensions,
                self.hnsw_m,
                self.hnsw_ef_construction,
                self.hnsw_ef_search,
                self.distance_metric.into(),
            )?);
        }

        let mut merged_count = 0;
        let base_index = self.vectors.len();

        for (other_idx, vector) in other.vectors.iter().enumerate() {
            let has_conflict = other
                .id_to_index
                .iter()
                .find(|(_, &idx)| idx == other_idx)
                .is_some_and(|(string_id, _)| self.id_to_index.contains_key(string_id));

            if has_conflict {
                continue;
            }

            self.vectors.push(vector.clone());

            if let Some(meta) = other.metadata.get(&other_idx) {
                self.metadata
                    .insert(base_index + merged_count, meta.clone());
            }

            if let Some((string_id, _)) =
                other.id_to_index.iter().find(|(_, &idx)| idx == other_idx)
            {
                self.id_to_index
                    .insert(string_id.clone(), base_index + merged_count);
            }

            merged_count += 1;
        }

        // Always rebuild index after merge to ensure consistency
        // (HNSW merge would include conflicting vectors that were skipped above)
        self.rebuild_index()?;

        Ok(merged_count)
    }

    /// Check if index needs to be rebuilt
    #[inline]
    #[must_use]
    pub fn needs_index_rebuild(&self) -> bool {
        self.hnsw_index.is_none() && self.vectors.len() > 100
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
    pub fn knn_search(&mut self, query: &Vector, k: usize) -> Result<Vec<(usize, f32)>> {
        self.knn_search_with_ef(query, k, None)
    }

    /// K-nearest neighbors search with optional ef override
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
        if query.dim() != self.dimensions {
            anyhow::bail!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.dim()
            );
        }

        let has_data =
            !self.vectors.is_empty() || self.hnsw_index.as_ref().is_some_and(|idx| !idx.is_empty());

        if !has_data {
            return Ok(Vec::new());
        }

        if let Some(ref index) = self.hnsw_index {
            let results = if index.is_asymmetric() {
                // Rescore if we have storage (fetch from disk) OR vectors in RAM
                let can_rescore = self.storage.is_some() || !self.vectors.is_empty();
                if self.rescore_enabled && can_rescore {
                    self.knn_search_with_rescore(query, k, ef)?
                } else {
                    index.search_asymmetric_ef(&query.data, k, ef)?
                }
            } else {
                index.search_ef(&query.data, k, ef)?
            };

            // Fall back to brute force if HNSW returns nothing but we have data
            // This can happen after heavy deletions leave the graph disconnected
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
        let candidates = index.search_asymmetric_ef(&query.data, oversample_k, ef)?;

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Rescore candidates with exact L2 distance
        // Avoid cloning: compute distance inline using references where possible
        let mut rescored: Vec<(usize, f32)> = candidates
            .iter()
            .filter_map(|&(id, _quantized_dist)| {
                // Storage path: get_vector returns owned Vec
                // Memory path: compute distance directly from reference (no clone)
                if let Some(ref storage) = self.storage {
                    storage
                        .get_vector(id)
                        .ok()
                        .flatten()
                        .map(|data| (id, l2_distance(&query.data, &data)))
                } else {
                    self.vectors
                        .get(id)
                        .map(|v| (id, l2_distance(&query.data, &v.data)))
                }
            })
            .collect();

        rescored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        rescored.truncate(k);

        Ok(rescored)
    }

    /// K-nearest neighbors search with metadata filtering
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
    /// Uses Roaring bitmap index for O(1) filter evaluation when possible,
    /// falls back to JSON-based filtering for complex filters.
    pub fn knn_search_with_filter_ef_readonly(
        &self,
        query: &Vector,
        k: usize,
        filter: &MetadataFilter,
        ef: Option<usize>,
    ) -> Result<Vec<(usize, f32, JsonValue)>> {
        // Use provided ef, or fall back to stored hnsw_ef_search
        // Ensure ef >= k (HNSW requirement)
        let effective_ef = compute_effective_ef(ef, self.hnsw_ef_search, k);

        // Try bitmap-based filtering (O(1) per candidate)
        let filter_bitmap = filter.evaluate_bitmap(&self.metadata_index);

        if let Some(ref hnsw) = self.hnsw_index {
            let metadata_map = &self.metadata;
            let deleted_map = &self.deleted;

            let search_results = if let Some(ref bitmap) = filter_bitmap {
                // Fast path: bitmap-based filtering
                let filter_fn = |node_id: u32| -> bool {
                    let index = node_id as usize;
                    !deleted_map.contains_key(&index) && bitmap.contains(node_id)
                };
                hnsw.search_with_filter_ef(&query.data, k, Some(effective_ef), filter_fn)?
            } else {
                // Slow path: JSON-based filtering
                let filter_fn = |node_id: u32| -> bool {
                    let index = node_id as usize;
                    if deleted_map.contains_key(&index) {
                        return false;
                    }
                    let metadata = metadata_map
                        .get(&index)
                        .cloned()
                        .unwrap_or_else(default_metadata);
                    filter.matches(&metadata)
                };
                hnsw.search_with_filter_ef(&query.data, k, Some(effective_ef), filter_fn)?
            };

            let filtered_results: Vec<(usize, f32, JsonValue)> = search_results
                .into_iter()
                .map(|(index, distance)| {
                    let metadata = self
                        .metadata
                        .get(&index)
                        .cloned()
                        .unwrap_or_else(default_metadata);
                    (index, distance, metadata)
                })
                .collect();

            return Ok(filtered_results);
        }

        // Fallback: brute-force search with filtering
        let mut all_results: Vec<(usize, f32, JsonValue)> = self
            .vectors
            .iter()
            .enumerate()
            .filter_map(|(index, vec)| {
                if self.deleted.contains_key(&index) {
                    return None;
                }

                // Use bitmap if available, otherwise JSON
                let passes_filter = if let Some(ref bitmap) = filter_bitmap {
                    bitmap.contains(index as u32)
                } else {
                    let metadata = self
                        .metadata
                        .get(&index)
                        .cloned()
                        .unwrap_or_else(default_metadata);
                    filter.matches(&metadata)
                };

                if !passes_filter {
                    return None;
                }

                let metadata = self
                    .metadata
                    .get(&index)
                    .cloned()
                    .unwrap_or_else(default_metadata);
                let distance = query.l2_distance(vec).unwrap_or(f32::MAX);
                Some((index, distance, metadata))
            })
            .collect();

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
        self.search_with_options(query, k, filter, None, None)
    }

    /// Search with optional filter and ef override
    pub fn search_with_ef(
        &mut self,
        query: &Vector,
        k: usize,
        filter: Option<&MetadataFilter>,
        ef: Option<usize>,
    ) -> Result<Vec<(usize, f32, JsonValue)>> {
        self.search_with_options(query, k, filter, ef, None)
    }

    /// Search with all options: filter, ef override, and max_distance
    pub fn search_with_options(
        &mut self,
        query: &Vector,
        k: usize,
        filter: Option<&MetadataFilter>,
        ef: Option<usize>,
        max_distance: Option<f32>,
    ) -> Result<Vec<(usize, f32, JsonValue)>> {
        self.ensure_index_ready()?;
        self.search_with_options_readonly(query, k, filter, ef, max_distance)
    }

    /// Read-only search with optional filter (for parallel execution)
    pub fn search_with_ef_readonly(
        &self,
        query: &Vector,
        k: usize,
        filter: Option<&MetadataFilter>,
        ef: Option<usize>,
    ) -> Result<Vec<(usize, f32, JsonValue)>> {
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
    ) -> Result<Vec<(usize, f32, JsonValue)>> {
        let mut results = if let Some(f) = filter {
            self.knn_search_with_filter_ef_readonly(query, k, f, ef)?
        } else {
            let results = self.knn_search_readonly(query, k, ef)?;
            let filtered: Vec<(usize, f32, JsonValue)> = results
                .into_iter()
                .filter_map(|(index, distance)| {
                    if self.deleted.contains_key(&index) {
                        return None;
                    }
                    let metadata = self
                        .metadata
                        .get(&index)
                        .cloned()
                        .unwrap_or_else(default_metadata);
                    Some((index, distance, metadata))
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
            results.retain(|(_, distance, _)| *distance <= max_dist);
        }

        Ok(results)
    }

    /// Check if there are any non-deleted vectors
    fn has_live_vectors(&self) -> bool {
        let total = self
            .vectors
            .len()
            .max(self.hnsw_index.as_ref().map_or(0, HNSWIndex::len));
        total > self.deleted.len()
    }

    /// Check if this store has quantization enabled (affects RAM storage)
    fn is_quantized(&self) -> bool {
        self.pending_quantization.is_some()
            || self
                .hnsw_index
                .as_ref()
                .is_some_and(|idx| idx.is_asymmetric() || idx.is_sq8())
    }

    /// Brute-force search with metadata (fallback for orphaned nodes)
    fn knn_search_brute_force_with_metadata(
        &self,
        query: &Vector,
        k: usize,
    ) -> Result<Vec<(usize, f32, JsonValue)>> {
        let results = self.knn_search_brute_force(query, k)?;
        Ok(results
            .into_iter()
            .filter_map(|(index, distance)| {
                if self.deleted.contains_key(&index) {
                    return None;
                }
                let metadata = self
                    .metadata
                    .get(&index)
                    .cloned()
                    .unwrap_or_else(default_metadata);
                Some((index, distance, metadata))
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
    ) -> Vec<Result<Vec<(usize, f32, JsonValue)>>> {
        queries
            .par_iter()
            .map(|q| self.search_with_ef_readonly(q, k, None, ef))
            .collect()
    }

    /// Brute-force K-NN search (fallback)
    pub fn knn_search_brute_force(&self, query: &Vector, k: usize) -> Result<Vec<(usize, f32)>> {
        if query.dim() != self.dimensions {
            anyhow::bail!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.dim()
            );
        }

        // Determine total vector count (in-memory or storage)
        let total_count = if !self.vectors.is_empty() {
            self.vectors.len()
        } else if let Some(ref idx) = self.hnsw_index {
            idx.len()
        } else {
            return Ok(Vec::new());
        };

        if total_count == 0 {
            return Ok(Vec::new());
        }

        let mut distances: Vec<(usize, f32)> = (0..total_count)
            .filter_map(|id| {
                // Get vector data from memory or storage
                let data = if let Some(vec) = self.vectors.get(id) {
                    Some(vec.data.clone())
                } else if let Some(ref storage) = self.storage {
                    storage.get_vector(id).ok().flatten()
                } else {
                    None
                };

                data.map(|vec_data| {
                    let dist = l2_distance(&query.data, &vec_data);
                    (id, dist)
                })
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
    /// Reorders graph nodes and vectors using BFS traversal to improve memory locality.
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

        let num_reordered = old_to_new.len();

        // Reorder VectorStore's own vectors (used for rescore)
        if !self.vectors.is_empty() {
            let old_vectors = std::mem::take(&mut self.vectors);
            let mut new_vectors = Vec::with_capacity(old_vectors.len());
            new_vectors.resize_with(old_vectors.len(), || Vector::new(Vec::new()));

            for (old_idx, &new_idx) in old_to_new.iter().enumerate() {
                let new_idx = new_idx as usize;
                if old_idx < old_vectors.len() && new_idx < new_vectors.len() {
                    new_vectors[new_idx] = old_vectors[old_idx].clone();
                }
            }
            self.vectors = new_vectors;
        }

        // Update ID mappings: id_to_index and index_to_id
        let mut new_id_to_index: FxHashMap<String, usize> =
            FxHashMap::with_capacity_and_hasher(self.id_to_index.len(), rustc_hash::FxBuildHasher);
        let mut new_index_to_id: FxHashMap<usize, String> =
            FxHashMap::with_capacity_and_hasher(self.index_to_id.len(), rustc_hash::FxBuildHasher);

        for (string_id, &old_idx) in &self.id_to_index {
            if old_idx < old_to_new.len() {
                let new_idx = old_to_new[old_idx] as usize;
                new_id_to_index.insert(string_id.clone(), new_idx);
                new_index_to_id.insert(new_idx, string_id.clone());
            }
        }

        self.id_to_index = new_id_to_index;
        self.index_to_id = new_index_to_id;

        // Update deleted tombstones
        if !self.deleted.is_empty() {
            let mut new_deleted = HashMap::with_capacity(self.deleted.len());
            for (&old_idx, &is_deleted) in &self.deleted {
                if old_idx < old_to_new.len() {
                    let new_idx = old_to_new[old_idx] as usize;
                    new_deleted.insert(new_idx, is_deleted);
                }
            }
            self.deleted = new_deleted;
        }

        // Note: metadata_index uses string IDs, not internal indices, so no update needed

        Ok(num_reordered)
    }

    // ============================================================================
    // Accessors
    // ============================================================================

    /// Get vector by internal index (used by FFI bindings)
    #[must_use]
    #[allow(dead_code)] // Used by FFI feature
    pub(crate) fn get_by_internal_index(&self, idx: usize) -> Option<&Vector> {
        self.vectors.get(idx)
    }

    /// Get vector by internal index, owned (used by FFI bindings)
    #[must_use]
    #[allow(dead_code)] // Used by FFI feature
    pub(crate) fn get_by_internal_index_owned(&self, idx: usize) -> Option<Vector> {
        if let Some(v) = self.vectors.get(idx) {
            return Some(v.clone());
        }

        if let Some(ref storage) = self.storage {
            if let Ok(Some(data)) = storage.get_vector(idx) {
                return Some(Vector::new(data));
            }
        }

        None
    }

    /// Number of vectors stored (excluding deleted vectors)
    #[must_use]
    pub fn len(&self) -> usize {
        if let Some(ref index) = self.hnsw_index {
            let hnsw_len = index.len();
            if hnsw_len > 0 {
                return hnsw_len.saturating_sub(self.deleted.len());
            }
        }
        self.vectors.len().saturating_sub(self.deleted.len())
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
        self.id_to_index
            .iter()
            .filter_map(|(id, &idx)| {
                if self.deleted.contains_key(&idx) {
                    None
                } else {
                    Some(id.clone())
                }
            })
            .collect()
    }

    /// Get all items as (id, vector, metadata) tuples
    ///
    /// Returns all non-deleted items. O(n) time and memory.
    #[must_use]
    pub fn items(&self) -> Vec<(String, Vec<f32>, JsonValue)> {
        self.id_to_index
            .iter()
            .filter_map(|(id, &idx)| {
                if self.deleted.contains_key(&idx) {
                    return None;
                }

                // Try in-memory vectors first
                let vec_data = if let Some(vec) = self.vectors.get(idx) {
                    vec.data.clone()
                } else if let Some(ref storage) = self.storage {
                    // Fall back to storage for quantized stores
                    storage.get_vector(idx).ok().flatten()?
                } else {
                    return None;
                };

                let metadata = self.metadata.get(&idx).cloned().unwrap_or_default();
                Some((id.clone(), vec_data, metadata))
            })
            .collect()
    }

    /// Check if an ID exists (not deleted)
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.id_to_index
            .get(id)
            .is_some_and(|&idx| !self.deleted.contains_key(&idx))
    }

    /// Memory usage estimate (bytes)
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.vectors.iter().map(|v| v.dim() * 4).sum::<usize>()
    }

    /// Bytes per vector (average)
    #[must_use]
    pub fn bytes_per_vector(&self) -> f32 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        self.memory_usage() as f32 / self.vectors.len() as f32
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

    // ============================================================================
    // Persistence
    // ============================================================================

    /// Flush all pending changes to disk
    ///
    /// Commits vector/metadata changes and HNSW index to `.omen` storage.
    pub fn flush(&mut self) -> Result<()> {
        let hnsw_bytes = self
            .hnsw_index
            .as_ref()
            .map(postcard::to_allocvec)
            .transpose()?;

        if let Some(ref mut storage) = self.storage {
            // Persist HNSW parameters to header
            storage.set_hnsw_params(
                self.hnsw_m as u16,
                self.hnsw_ef_construction as u16,
                self.hnsw_ef_search as u16,
            );

            if let Some(bytes) = hnsw_bytes {
                storage.put_hnsw_index(bytes);
            }
            storage.flush()?;
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
