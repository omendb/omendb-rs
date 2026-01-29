//! Persistence operations for VectorStore.
//!
//! This module contains methods for opening, saving, and managing persistent
//! vector stores using the `.omen` file format.

use super::helpers;
use super::record_store::{Record, RecordStore};
use super::VectorStore;
use super::{
    DEFAULT_HNSW_EF_CONSTRUCTION, DEFAULT_HNSW_EF_SEARCH, DEFAULT_HNSW_M, DEFAULT_MAX_TOKENS,
    DEFAULT_OVERSAMPLE_FACTOR,
};
use crate::omen::{
    parse_wal_delete, parse_wal_insert, CheckpointOptions, MetadataIndex, OmenFile, WalEntryType,
};
use crate::text::TextIndex;
use crate::vector::hnsw::{HNSWParams, SegmentConfig, SegmentManager};
use crate::vector::muvera::{MultiVecStorage, MultiVectorConfig, MuveraEncoder};
use crate::vector::store::options::VectorStoreOptions;
use anyhow::Result;
use roaring::RoaringBitmap;
use serde_json::Value as JsonValue;
use std::path::Path;

use crate::omen::Metric;

impl VectorStore {
    // ========================================================================
    // Open/Create
    // ========================================================================

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

        // Build segments from vectors
        let active_count = records.len() as usize;

        let segments = if active_count > 0 && dimensions > 0 {
            // Collect vectors and slots for parallel build
            let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(active_count);
            let mut slots: Vec<u32> = Vec::with_capacity(active_count);
            for (slot, record) in records.iter_live() {
                vectors.push(record.vector.clone());
                slots.push(slot);
            }

            // Build segment config
            let config = SegmentConfig::new(dimensions)
                .with_params(HNSWParams {
                    m: hnsw_m,
                    ef_construction: hnsw_ef_construction,
                    ..Default::default()
                })
                .with_distance(distance_metric.into())
                .with_quantization(quantization_mode.is_some());

            // Build segments with parallel construction
            Some(
                SegmentManager::build_parallel_with_slots(config, vectors, &slots)
                    .map_err(|e| anyhow::anyhow!("Segment build failed: {e}"))?,
            )
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
        let rescore_enabled = quantization_mode.is_some();

        // Reconstruct multi-vector state if config is present
        let (muvera_encoder, multivec_storage, distance_metric) =
            if let Some((reps, bits, seed, token_dim, d_proj, pool_factor)) =
                snapshot.multivec_config
            {
                let config = MultiVectorConfig {
                    repetitions: reps,
                    partition_bits: bits,
                    d_proj,
                    seed,
                    pool_factor,
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
            segments,
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

        // Quantization is deferred until first insert
        let pending_quantization = options.quantization.clone();

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

        // Quantization is deferred until first insert
        let pending_quantization = options.quantization.clone();

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

    // ========================================================================
    // Flush/Save
    // ========================================================================

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
                            config.d_proj,
                            config.pool_factor,
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
                    hnsw_bytes: None,
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

    // ========================================================================
    // Persistence Accessors
    // ========================================================================

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
