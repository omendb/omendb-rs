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

pub use filter::MetadataFilter;
pub use options::VectorStoreOptions;

use super::hnsw::{DistanceFunction, HNSWParams};
use super::hnsw_index::HNSWIndex;
use super::types::Vector;
use super::QuantizationMode;
use crate::omen::{MetadataIndex, OmenFile};
use crate::text::{weighted_reciprocal_rank_fusion, TextIndex, TextSearchConfig, DEFAULT_RRF_K};
use anyhow::Result;
use omendb_core::distance::l2_distance;
use rayon::prelude::*;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(test)]
mod tests;

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
            oversample_factor: 3.0,
            metadata: HashMap::new(),
            id_to_index: HashMap::new(),
            index_to_id: HashMap::new(),
            deleted: HashMap::new(),
            metadata_index: MetadataIndex::new(),
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
            metadata_index: MetadataIndex::new(),
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
            metadata_index: MetadataIndex::new(),
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

        // Load metadata and mappings (always needed)
        let metadata = storage.load_all_metadata()?;
        let id_to_index = storage.load_all_id_mappings()?;
        let deleted = storage.load_all_deleted()?;

        // Get dimensions from config
        let dimensions = storage.get_config("dimensions")?.unwrap_or(0) as usize;

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
            match bincode::deserialize::<HNSWIndex>(hnsw_bytes) {
                Ok(index) => {
                    // Check if HNSW index matches loaded vectors (WAL recovery may add more)
                    if index.len() != active_vector_count && !vectors.is_empty() {
                        tracing::info!(
                            "HNSW index count ({}) differs from vector count ({}), rebuilding",
                            index.len(),
                            active_vector_count
                        );
                        let mut new_index = HNSWIndex::new(vectors.len().max(10_000), dimensions)?;
                        let vector_data: Vec<Vec<f32>> =
                            vectors.iter().map(|v| v.data.clone()).collect();
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
            let mut index = HNSWIndex::new(vectors.len().max(10_000), dimensions)?;
            let vector_data: Vec<Vec<f32>> = vectors.iter().map(|v| v.data.clone()).collect();
            index.batch_insert(&vector_data)?;
            Some(index)
        } else if is_quantized && dimensions > 0 {
            let vectors_data = storage.load_all_vectors()?;
            if vectors_data.is_empty() {
                None
            } else {
                let mut index = HNSWIndex::new(vectors_data.len().max(10_000), dimensions)?;
                let vector_data: Vec<Vec<f32>> =
                    vectors_data.iter().map(|(_, v)| v.clone()).collect();
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
        let index_to_id: HashMap<usize, String> = id_to_index
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

        Ok(Self {
            vectors,
            hnsw_index,
            dimensions,
            rescore_enabled,
            oversample_factor: 3.0,
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
            metadata_index: MetadataIndex::new(),
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
            metadata_index: MetadataIndex::new(),
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

    // ============================================================================
    // Insert/Set Methods
    // ============================================================================

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
                let quant_mode_id = match &quant_mode {
                    QuantizationMode::SQ8 => 1u64,
                    QuantizationMode::RaBitQ(p) => match p.bits_per_dim.to_u8() {
                        2 => 3u64,
                        8 => 4u64,
                        _ => 2u64,
                    },
                };
                if let Some(ref mut storage) = self.storage {
                    storage.put_quantization_mode(quant_mode_id)?;
                }

                let index = match quant_mode {
                    QuantizationMode::SQ8 => {
                        HNSWIndex::new_with_sq8(dimensions, hnsw_params, DistanceFunction::L2)?
                    }
                    QuantizationMode::RaBitQ(params) => {
                        let mut idx = HNSWIndex::new_with_asymmetric(
                            dimensions,
                            hnsw_params,
                            DistanceFunction::L2,
                            params,
                        )?;
                        idx.train_quantizer(std::slice::from_ref(&vector.data))?;
                        idx
                    }
                };
                self.hnsw_index = Some(index);
            } else {
                self.hnsw_index = Some(HNSWIndex::new(10_000, dimensions)?);
            }
            self.dimensions = dimensions;
        } else if vector.dim() != self.dimensions {
            anyhow::bail!(
                "Vector dimension mismatch: store expects {}, got {}. All vectors in same store must have same dimension.",
                self.dimensions,
                vector.dim()
            );
        }

        // Insert into HNSW index
        if let Some(ref mut index) = self.hnsw_index {
            index.insert(&vector.data)?;
        }

        // Persist to storage if available
        if let Some(ref mut storage) = self.storage {
            storage.put_vector(id, &vector.data)?;
            storage.increment_count()?;
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
        if self.id_to_index.contains_key(&id) {
            anyhow::bail!("Vector with ID '{id}' already exists. Use set() to update.");
        }

        let index = self.insert(vector)?;

        self.metadata.insert(index, metadata.clone());
        self.metadata_index.index_json(index as u32, &metadata);
        self.id_to_index.insert(id.clone(), index);
        self.index_to_id.insert(index, id.clone());

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

        // Process inserts in batch
        if !inserts.is_empty() {
            // Lazy initialize HNSW if needed
            if self.hnsw_index.is_none() {
                let dimensions = if self.dimensions == 0 {
                    inserts[0].1.dim()
                } else {
                    self.dimensions
                };

                if let Some(quant_mode) = self.pending_quantization.take() {
                    let hnsw_params = HNSWParams::default()
                        .with_m(self.hnsw_m)
                        .with_ef_construction(self.hnsw_ef_construction)
                        .with_ef_search(self.hnsw_ef_search);

                    let quant_mode_id = match &quant_mode {
                        QuantizationMode::SQ8 => 1u64,
                        QuantizationMode::RaBitQ(p) => match p.bits_per_dim.to_u8() {
                            2 => 3u64,
                            8 => 4u64,
                            _ => 2u64,
                        },
                    };
                    if let Some(ref mut storage) = self.storage {
                        storage.put_quantization_mode(quant_mode_id)?;
                    }

                    let index = match quant_mode {
                        QuantizationMode::SQ8 => {
                            HNSWIndex::new_with_sq8(dimensions, hnsw_params, DistanceFunction::L2)?
                        }
                        QuantizationMode::RaBitQ(params) => {
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

            // Insert vectors into HNSW using batch_insert for optimal graph construction
            // batch_insert works for all modes (f32, SQ8, RaBitQ) after fix to use get_dequantized
            let base_index = self.vectors.len();
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
            for (i, (id, vector, metadata)) in inserts.into_iter().enumerate() {
                let idx = base_index + i;
                self.vectors.push(vector);
                self.metadata.insert(idx, metadata.clone());
                self.metadata_index.index_json(idx as u32, &metadata);
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
        if index >= self.vectors.len() {
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

            self.vectors[index] = new_vector.clone();

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
    pub fn delete(&mut self, id: &str) -> Result<()> {
        let index = self
            .id_to_index
            .get(id)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Vector with ID '{id}' not found"))?;

        self.deleted.insert(index, true);
        self.metadata_index.remove(index as u32);

        // Use OmenFile::delete for WAL-backed persistence
        if let Some(ref mut storage) = self.storage {
            storage.delete(id)?;
        }

        if let Some(ref mut text_index) = self.text_index {
            text_index.delete_document(id)?;
        }

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
    #[must_use]
    pub fn get_by_id(&self, id: &str) -> Option<(&Vector, &JsonValue)> {
        self.id_to_index.get(id).and_then(|&index| {
            if self.deleted.contains_key(&index) {
                return None;
            }
            self.vectors
                .get(index)
                .and_then(|vec| self.metadata.get(&index).map(|meta| (vec, meta)))
        })
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
            if let Some(quant_mode) = self.pending_quantization.take() {
                let hnsw_params = HNSWParams::default()
                    .with_m(self.hnsw_m)
                    .with_ef_construction(self.hnsw_ef_construction)
                    .with_ef_search(self.hnsw_ef_search);

                let index = match quant_mode {
                    QuantizationMode::SQ8 => {
                        HNSWIndex::new_with_sq8(self.dimensions, hnsw_params, DistanceFunction::L2)?
                    }
                    QuantizationMode::RaBitQ(params) => {
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

        for chunk in vectors.chunks(CHUNK_SIZE) {
            let vector_data: Vec<Vec<f32>> = chunk.iter().map(|v| v.data.clone()).collect();

            if let Some(ref mut index) = self.hnsw_index {
                let chunk_ids = index.batch_insert(&vector_data)?;
                all_ids.extend(chunk_ids);
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

        let mut index = HNSWIndex::new(self.vectors.len().max(1_000_000), self.dimensions)?;

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
            self.hnsw_index = Some(HNSWIndex::new(capacity, self.dimensions)?);
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
        let effective_ef = match ef {
            Some(e) => e,
            None => (k * 4).max(64).max(100),
        };
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
            if index.is_asymmetric() {
                if self.rescore_enabled && !self.vectors.is_empty() {
                    return self.knn_search_with_rescore(query, k, ef);
                }
                return index.search_asymmetric_ef(&query.data, k, ef);
            }
            return index.search_ef(&query.data, k, ef);
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

        let mut rescored: Vec<(usize, f32)> = candidates
            .iter()
            .filter_map(|&(id, _quantized_dist)| {
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
                hnsw.search_with_filter_ef(&query.data, k, ef, filter_fn)?
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
                        .unwrap_or(serde_json::json!({}));
                    filter.matches(&metadata)
                };
                hnsw.search_with_filter_ef(&query.data, k, ef, filter_fn)?
            };

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
                        .unwrap_or(serde_json::json!({}));
                    filter.matches(&metadata)
                };

                if !passes_filter {
                    return None;
                }

                let metadata = self
                    .metadata
                    .get(&index)
                    .cloned()
                    .unwrap_or(serde_json::json!({}));
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
        self.search_with_ef(query, k, filter, None)
    }

    /// Search with optional filter and ef override
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
            let results = self.knn_search_readonly(query, k, ef)?;
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
                        .unwrap_or(serde_json::json!({}));
                    Some((index, distance, metadata))
                })
                .collect())
        }
    }

    /// Parallel batch search for multiple queries
    #[must_use]
    pub fn batch_search_parallel(
        &self,
        queries: &[Vector],
        k: usize,
        ef: Option<usize>,
    ) -> Vec<Result<Vec<(usize, f32)>>> {
        let effective_ef = match ef {
            Some(e) => e,
            None => (k * 4).max(64).max(100),
        };
        queries
            .par_iter()
            .map(|q| self.knn_search_ef(q, k, effective_ef))
            .collect()
    }

    /// Parallel batch search with metadata
    #[must_use]
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

    /// Brute-force K-NN search (fallback)
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

        let mut distances: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(id, vec)| {
                let dist = query.l2_distance(vec).unwrap_or(f32::MAX);
                (id, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.total_cmp(&b.1));
        Ok(distances.into_iter().take(k).collect())
    }

    // ============================================================================
    // Accessors
    // ============================================================================

    /// Get vector by ID
    #[must_use]
    pub fn get(&self, id: usize) -> Option<&Vector> {
        self.vectors.get(id)
    }

    /// Get vector by ID (owned)
    #[must_use]
    pub fn get_owned(&self, id: usize) -> Option<Vector> {
        if let Some(v) = self.vectors.get(id) {
            return Some(v.clone());
        }

        if let Some(ref storage) = self.storage {
            if let Ok(Some(data)) = storage.get_vector(id) {
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

    /// Check if store is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
            .map(bincode::serialize)
            .transpose()?;

        if let Some(ref mut storage) = self.storage {
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
