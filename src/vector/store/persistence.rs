//! Persistence operations for VectorStore.
//!
//! This module contains methods for opening, saving, and managing persistent
//! vector stores using the `.omen` file format.

use super::VectorStore;
use super::helpers;
use super::record_store::{RecordStore, SnapshotRecord};
use super::{DEFAULT_HNSW_EF_CONSTRUCTION, DEFAULT_HNSW_EF_SEARCH, DEFAULT_HNSW_M};
use crate::catalog::{
    CollectionSchema, DenseSchema, FrozenDenseIndexKind, MultiEncoderKind, MultiSchema,
    MutableDenseIndexKind, QuantizationMode, SparseIndexKind, TextSchema,
};
use crate::omen::{
    CheckpointOptions, OmenFile, OmenSnapshot, PersistedMuveraConfig, WalEntryType,
    parse_wal_delete, parse_wal_delete_edge, parse_wal_insert, parse_wal_insert_edge,
    parse_wal_multi, parse_wal_sparse,
};
use crate::text::{TextIndex, TextSearchConfig};
use crate::vector::VectorEngineView;
use crate::vector::hnsw::{HNSWParams, SegmentConfig, SegmentManager};
use crate::vector::metadata::MetadataIndex;
use crate::vector::muvera::{MultiVecStorage, MultiVectorConfig, MuveraEncoder};
use crate::vector::sparse::SparseIndex;
use crate::vector::store::edge_store::{Edge, EdgeStore};
use crate::vector::store::options::VectorStoreOptions;
use anyhow::Result;
use arc_swap::ArcSwap;
use parking_lot::RwLock;
use roaring::RoaringBitmap;
use rustc_hash::FxHashMap;
use serde_json::Value as JsonValue;
use std::path::Path;
use std::sync::Arc;

use crate::omen::Metric;
use std::path::PathBuf;

/// Compute the segments directory path for a given store path.
///
/// Appends `.segments` to the full path. For example, "mydb.omen"
/// becomes "mydb.omen.segments".
pub(super) fn segments_dir_for(path: &Path) -> PathBuf {
    let mut seg_path = path.as_os_str().to_os_string();
    seg_path.push(".segments");
    PathBuf::from(seg_path)
}

type ReplayedWalData = (Vec<u32>, Option<EdgeStore>, Vec<(String, String, String)>);

fn tokenizer_preset_from_id(id: u64) -> Result<crate::text::TokenizerPreset> {
    match id {
        0 => Ok(crate::text::TokenizerPreset::Default),
        1 => Ok(crate::text::TokenizerPreset::Code),
        2 => Ok(crate::text::TokenizerPreset::Raw),
        other => anyhow::bail!("Unknown tokenizer preset id: {other}"),
    }
}

fn load_text_search_config(storage: &OmenFile) -> Result<Option<crate::text::TextSearchConfig>> {
    if let Some(text) = storage.schema().and_then(|schema| schema.text.as_ref()) {
        return Ok(Some(crate::text::TextSearchConfig {
            writer_buffer_mb: text.writer_buffer_mb as usize,
            tokenizer: text.tokenizer,
        }));
    }

    let tokenizer = storage.get_config("text_tokenizer")?;
    let writer_buffer_mb = storage.get_config("text_writer_buffer_mb")?;
    match (tokenizer, writer_buffer_mb) {
        (Some(tokenizer), Some(writer_buffer_mb)) => Ok(Some(crate::text::TextSearchConfig {
            writer_buffer_mb: writer_buffer_mb as usize,
            tokenizer: tokenizer_preset_from_id(tokenizer)?,
        })),
        (None, None) => Ok(None),
        _ => Ok(Some(crate::text::TextSearchConfig {
            writer_buffer_mb: writer_buffer_mb.unwrap_or(50) as usize,
            tokenizer: tokenizer
                .map(tokenizer_preset_from_id)
                .transpose()?
                .unwrap_or_default(),
        })),
    }
}

fn runtime_dimensions_from_schema(schema: &CollectionSchema) -> usize {
    if let Some(ref dense) = schema.dense {
        return dense.dim as usize;
    }
    if let Some(ref multi) = schema.multi {
        let token_dim = multi.token_dim as usize;
        let proj_dim = multi.d_proj.map_or(token_dim, usize::from);
        return usize::from(multi.repetitions) * (1usize << multi.partition_bits) * proj_dim;
    }
    0
}

fn multi_vector_config_from_schema(schema: &MultiSchema) -> MultiVectorConfig {
    match schema.encoder {
        MultiEncoderKind::Muvera => MultiVectorConfig {
            repetitions: schema.repetitions,
            partition_bits: schema.partition_bits,
            d_proj: schema.d_proj,
            seed: schema.seed,
            pool_factor: schema.pool_factor,
            max_tokens: schema.max_tokens.map(|v| v as usize),
        },
    }
}

fn text_search_config_from_schema(schema: &CollectionSchema) -> Option<TextSearchConfig> {
    schema.text.as_ref().map(|text| TextSearchConfig {
        writer_buffer_mb: text.writer_buffer_mb as usize,
        tokenizer: text.tokenizer,
    })
}

fn validate_collection_schema(schema: &CollectionSchema) -> Result<()> {
    if schema.dense.is_none()
        && schema.sparse.is_none()
        && schema.multi.is_none()
        && schema.text.is_none()
    {
        anyhow::bail!("collection schema must enable at least one modality");
    }

    if schema.dense.is_some() && schema.multi.is_some() {
        anyhow::bail!("dense and multi-vector primary schemas cannot both be enabled");
    }

    if let Some(ref dense) = schema.dense {
        match dense.mutable_index {
            MutableDenseIndexKind::Hnsw => {}
        }
        match dense.frozen_index {
            FrozenDenseIndexKind::Hnsw => {}
        }
    }

    if let Some(ref sparse) = schema.sparse {
        match sparse.index_kind {
            SparseIndexKind::InvertedExact => {}
        }
    }

    if let Some(ref multi) = schema.multi {
        if schema.metric != Metric::InnerProduct {
            anyhow::bail!("multi-vector schemas require metric='dot'");
        }
        if schema.sparse.is_some() {
            anyhow::bail!("multi-vector + sparse schema creation is not supported yet");
        }
        if multi.d_proj.is_some_and(|d| usize::from(d) > multi.token_dim as usize) {
            anyhow::bail!("multi-vector d_proj cannot exceed token_dim");
        }
        let _ = multi_vector_config_from_schema(multi);
    }

    Ok(())
}

fn base_store_from_schema(schema: &CollectionSchema) -> Result<VectorStore> {
    validate_collection_schema(schema)?;

    let mut store = if let Some(ref multi) = schema.multi {
        VectorStore::multi_vector_with(
            multi.token_dim as usize,
            multi_vector_config_from_schema(multi),
        )?
    } else if let Some(ref dense) = schema.dense {
        match dense.quantization {
            QuantizationMode::Sq8 => VectorStore::new_with_quantization(dense.dim as usize),
            QuantizationMode::None => VectorStore::new_with_params(
                dense.dim as usize,
                DEFAULT_HNSW_M,
                DEFAULT_HNSW_EF_CONSTRUCTION,
                DEFAULT_HNSW_EF_SEARCH,
                schema.metric,
            ),
        }
    } else {
        VectorStore::with_defaults(0, schema.metric)
    };

    if schema.sparse.is_some() {
        store.enable_sparse();
    }

    Ok(store)
}

fn schema_from_options(options: &VectorStoreOptions) -> CollectionSchema {
    CollectionSchema {
        name: String::new(),
        metric: options.metric.unwrap_or(Metric::L2),
        dense: Some(DenseSchema {
            dim: options.dimensions as u32,
            quantization: if options.quantization {
                QuantizationMode::Sq8
            } else {
                QuantizationMode::None
            },
            mutable_index: MutableDenseIndexKind::Hnsw,
            frozen_index: FrozenDenseIndexKind::Hnsw,
        }),
        sparse: None,
        multi: None,
        text: options.text_search_config.as_ref().map(|config| TextSchema {
            tokenizer: config.tokenizer,
            writer_buffer_mb: config.writer_buffer_mb as u32,
        }),
    }
}

fn legacy_dense_schema(dimensions: usize, quantized: bool) -> DenseSchema {
    DenseSchema {
        dim: dimensions as u32,
        quantization: if quantized {
            QuantizationMode::Sq8
        } else {
            QuantizationMode::None
        },
        mutable_index: MutableDenseIndexKind::Hnsw,
        frozen_index: FrozenDenseIndexKind::Hnsw,
    }
}

impl VectorStore {
    /// Create a new persistent vector store from an explicit collection schema.
    pub fn create(path: impl AsRef<Path>, schema: CollectionSchema) -> Result<Self> {
        let path = path.as_ref();
        let omen_path = OmenFile::compute_omen_path(path);
        if path.exists() || omen_path.exists() {
            anyhow::bail!("store already exists at {}", path.display());
        }

        let mut store = base_store_from_schema(&schema)?;
        let mut storage = OmenFile::create(path, runtime_dimensions_from_schema(&schema) as u32)?;
        storage.set_metric(schema.metric);
        storage.set_schema(schema.clone());
        if schema
            .dense
            .as_ref()
            .is_some_and(|dense| dense.quantization == QuantizationMode::Sq8)
        {
            storage.put_quantization_mode(helpers::quantization_to_id(true))?;
        }

        let storage: Arc<RwLock<dyn crate::omen::StorageBackend>> = Arc::new(RwLock::new(storage));
        store.storage = Some(storage);
        store.storage_path = Some(path.to_path_buf());
        store.schema_name = Some(schema.name.clone());
        store.dense_schema.write().clone_from(&schema.dense);

        if let Some(config) = text_search_config_from_schema(&schema) {
            store.enable_text_search_with_config(Some(config))?;
        }

        Ok(store)
    }

    /// Create a new in-memory vector store from an explicit collection schema.
    pub fn create_in_memory(schema: CollectionSchema) -> Result<Self> {
        let mut store = base_store_from_schema(&schema)?;
        store.schema_name = Some(schema.name.clone());
        store.dense_schema.write().clone_from(&schema.dense);
        if let Some(config) = text_search_config_from_schema(&schema) {
            store.enable_text_search_with_config(Some(config))?;
        }
        Ok(store)
    }

    /// Open a persistent vector store at the given path
    ///
    /// Creates a new database if it doesn't exist, or loads existing data.
    /// All operations (insert, set, delete) are automatically persisted.
    ///
    /// # Arguments
    /// * `path` - Directory path for the database (e.g., "mydb.omen")
    ///
    /// # Example
    /// ```ignore
    /// let mut store = VectorStore::open("mydb.omen")?;
    /// store.set("doc1", vector, metadata)?;
    /// // Data is automatically persisted
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let omen_path = OmenFile::compute_omen_path(path);
        let mut storage_local = if omen_path.exists() {
            OmenFile::open(path)?
        } else {
            OmenFile::create(path, 0)?
        };
        let text_search_config = load_text_search_config(&storage_local)?;

        // Scoped block to ensure ctx (which borrows from storage_local) is dropped
        // before we move storage_local into the Arc<RwLock>.
        let (records, engine, ancillary, quantization, hnsw_params, ef_search, _dimensions) = {
            let ctx = Self::load_recovery_context(&mut storage_local)?;

            // Build initial RecordStore from snapshot
            let mut records = Self::build_initial_records(
                &ctx.snapshot,
                ctx.dimensions,
                ctx.vec_mmap.clone(),
                ctx.main_mmap.clone(),
            );

            // Replay WAL entries into RecordStore and collect auxiliary deltas
            let (wal_modified_slots, wal_edge_store, wal_edge_deletes) =
                Self::replay_wal_into_records(&mut records, &mut storage_local, &ctx)?;

            // Initialize or rebuild segments (HNSW indexes)
            let modified_slots = if ctx.slim_snapshot_loaded && !ctx.slim_dirty_slots.is_empty() {
                &ctx.slim_dirty_slots
            } else {
                &wal_modified_slots
            };
            let engine =
                Self::initialize_segments(path, &storage_local, &records, modified_slots, &ctx)?;

            // Initialize auxiliary indexes (text, metadata, sparse, edge, muvera)
            let ancillary = Self::initialize_ancillary_indexes(
                path,
                &records,
                &ctx.snapshot,
                wal_edge_store,
                &wal_edge_deletes,
                ctx.distance_metric,
                text_search_config.as_ref(),
            )?;
            if let Some(ref multivec_storage) = ancillary.multivec_storage {
                for slot in 0..multivec_storage.len() as u32 {
                    if let Some(tokens) = multivec_storage.get_tokens(slot) {
                        let tokens: Vec<Vec<f32>> =
                            tokens.into_iter().map(<[f32]>::to_vec).collect();
                        records.update_multi(slot, Some(tokens))?;
                    }
                }
            }
            let ancillary = if let Some(ref encoder) = ancillary.muvera_encoder {
                let rebuilt = MultiVecStorage::from_slot_tokens(
                    encoder.token_dimension(),
                    &records.iter_multi(),
                );
                AncillaryIndexes {
                    multivec_storage: Some(rebuilt),
                    ..ancillary
                }
            } else {
                ancillary
            };
            (
                records,
                engine,
                ancillary,
                ctx.quantization,
                ctx.hnsw_params,
                ctx.ef_search,
                ctx.dimensions,
            )
        };

        let persisted_schema = storage_local.schema().cloned();
        let schema_name = persisted_schema.as_ref().map(|schema| schema.name.clone());
        let storage: Arc<RwLock<dyn crate::omen::StorageBackend>> =
            Arc::new(RwLock::new(storage_local));

        let mut engine = engine;
        if let Some(ref mut engine) = engine {
            engine.set_storage(Arc::clone(&storage));
        }
        let published_view = engine.as_ref().map(SegmentManager::read_view);

        let records_dimensions = records.dimensions() as usize;

        Ok(Self {
            records,
            engine: RwLock::new(engine),
            published_view: ArcSwap::new(Arc::new(published_view)),
            metadata_index: RwLock::new(ancillary.metadata_index),
            storage: Some(storage),
            storage_path: Some(path.to_path_buf()),
            schema_name,
            dense_schema: RwLock::new(
                persisted_schema
                    .as_ref()
                    .and_then(|schema| schema.dense.clone())
                    .or_else(|| {
                        if ancillary.muvera_encoder.is_none() {
                            Some(legacy_dense_schema(records_dimensions, quantization))
                        } else {
                            None
                        }
                    }),
            ),
            text_index: RwLock::new(ancillary.text_index),
            text_search_config: RwLock::new(text_search_config),
            pending_quantization: quantization.into(),
            hnsw_m: (if hnsw_params.m > 0 {
                hnsw_params.m
            } else {
                DEFAULT_HNSW_M
            })
            .into(),
            hnsw_ef_construction: (if hnsw_params.ef_construction > 0 {
                hnsw_params.ef_construction
            } else {
                DEFAULT_HNSW_EF_CONSTRUCTION
            })
            .into(),
            hnsw_ef_search: (if ef_search > 0 {
                ef_search
            } else {
                DEFAULT_HNSW_EF_SEARCH
            })
            .into(),
            distance_metric: ancillary.distance_metric,
            muvera_encoder: ancillary.muvera_encoder,
            multivec_storage: RwLock::new(ancillary.multivec_storage),
            sparse_index: RwLock::new(ancillary.sparse_index),
            edge_store: RwLock::new(ancillary.edge_store),
            segment_capacity: None,
            rescore: (quantization).into(),
            oversample: (3.0f32.to_bits()).into(),
            max_memory_bytes: None,
            auto_compact_threshold: (0.25f32.to_bits()).into(),
            write_lock: RwLock::new(()),
        })
    }

    /// Open a persistent vector store with specified dimensions
    ///
    /// Like `open()` but ensures dimensions are set for new databases.
    pub fn open_with_dimensions(path: impl AsRef<Path>, dimensions: usize) -> Result<Self> {
        let path = path.as_ref();
        let omen_path = OmenFile::compute_omen_path(path);
        if path.exists() || omen_path.exists() {
            return Self::open(path);
        }
        Self::create(
            path,
            CollectionSchema {
                name: String::new(),
                metric: Metric::L2,
                dense: Some(DenseSchema {
                    dim: dimensions as u32,
                    quantization: QuantizationMode::None,
                    mutable_index: MutableDenseIndexKind::Hnsw,
                    frozen_index: FrozenDenseIndexKind::Hnsw,
                }),
                sparse: None,
                multi: None,
                text: None,
            },
        )
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

            store
                .hnsw_m
                .store(options.m.unwrap_or(DEFAULT_HNSW_M), std::sync::atomic::Ordering::Relaxed);
            store.hnsw_ef_construction.store(
                options
                    .ef_construction
                    .unwrap_or(DEFAULT_HNSW_EF_CONSTRUCTION),
                std::sync::atomic::Ordering::Relaxed,
            );

            // Apply ef_search if specified
            store.set_ef_search(options.ef_search.unwrap_or(DEFAULT_HNSW_EF_SEARCH));

            // Apply rescore/oversample options
            store.set_rescore(options.rescore.unwrap_or(options.quantization));
            store.set_oversample(options.oversample.unwrap_or(3.0));
            store.max_memory_bytes = options.max_memory_bytes;

            return Ok(store);
        }
        let mut store = Self::create(path, schema_from_options(options))?;
        store
            .hnsw_m
            .store(options.m.unwrap_or(DEFAULT_HNSW_M), std::sync::atomic::Ordering::Relaxed);
        store.hnsw_ef_construction.store(
            options
                .ef_construction
                .unwrap_or(DEFAULT_HNSW_EF_CONSTRUCTION),
            std::sync::atomic::Ordering::Relaxed,
        );
        store.set_ef_search(options.ef_search.unwrap_or(DEFAULT_HNSW_EF_SEARCH));
        store.set_rescore(options.rescore.unwrap_or(options.quantization));
        store.set_oversample(options.oversample.unwrap_or(3.0));
        store.max_memory_bytes = options.max_memory_bytes;
        Ok(store)
    }

    /// Build an in-memory vector store with custom options.
    pub fn build_with_options(options: &VectorStoreOptions) -> Result<Self> {
        let mut store = Self::create_in_memory(schema_from_options(options))?;
        store
            .hnsw_m
            .store(options.m.unwrap_or(DEFAULT_HNSW_M), std::sync::atomic::Ordering::Relaxed);
        store.hnsw_ef_construction.store(
            options
                .ef_construction
                .unwrap_or(DEFAULT_HNSW_EF_CONSTRUCTION),
            std::sync::atomic::Ordering::Relaxed,
        );
        store.set_ef_search(options.ef_search.unwrap_or(DEFAULT_HNSW_EF_SEARCH));
        store.set_rescore(options.rescore.unwrap_or(options.quantization));
        store.set_oversample(options.oversample.unwrap_or(3.0));
        store.max_memory_bytes = options.max_memory_bytes;
        Ok(store)
    }

    /// Flush all pending changes to disk
    ///
    /// Commits vector/metadata changes and HNSW index to `.omen` storage.
    /// Uses RecordStore as single source of truth (no duplicated state in OmenFile).
    ///
    /// If the tombstone ratio exceeds `auto_compact_threshold` (default 25%),
    /// compaction runs automatically before persisting.
    pub fn flush(&self) -> Result<()> {
        let _lock = self.write_lock.write();
        let auto_compact_threshold = f32::from_bits(
            self.auto_compact_threshold
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        if self.records.deleted_count() > 0 && self.tombstone_ratio() > auto_compact_threshold {
            self.compact_locked()?;
        }
        self.flush_internal(false)
    }

    /// Auto-checkpoint: persist dirty vectors without full manifest rewrite.
    ///
    /// When .vecs exists, writes only dirty vector slots and syncs WAL — skipping
    /// the expensive manifest rewrite. On crash, WAL replay recovers IDs/metadata.
    /// Falls back to full flush on the first checkpoint (creates .vecs + manifest).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn checkpoint_wal(&self) -> Result<()> {
        let _lock = self.write_lock.write();
        self.checkpoint_wal_locked()
    }

    pub(super) fn checkpoint_wal_locked(&self) -> Result<()> {
        let has_vec = self
            .storage
            .as_ref()
            .is_some_and(|s| s.read().has_vec_file());
        let requires_full_checkpoint = self
            .sparse_index
            .read()
            .as_ref()
            .is_some_and(|index| !index.is_empty())
            || self
                .multivec_storage
                .read()
                .as_ref()
                .is_some_and(|storage| !storage.is_empty())
            || self
                .edge_store
                .read()
                .as_ref()
                .is_some_and(|store| store.edge_count() > 0);

        if has_vec {
            if requires_full_checkpoint {
                // The fast checkpoint path persists only dense slots + the slim `.records`
                // snapshot. Sparse postings, multivec token payloads, and edge state are
                // durable only via the full manifest path, so truncating the WAL here would
                // otherwise drop that state on reopen.
                return self.flush_internal(false);
            }

            // Fast path: write dirty .vecs slots, sync WAL, skip manifest
            let dirty = self.records.take_dirty_slots();
            if dirty.is_empty() {
                return Ok(());
            }
            if let Some(ref storage) = self.storage {
                let mut storage = storage.write();
                if let Err(e) = storage.checkpoint_vectors_only(&self.records, &dirty) {
                    // Restore dirty slots so the next checkpoint retries writing them.
                    // Without this, failed .vecs writes leave slots in id_to_slot but
                    // missing from dirty_since_flush, making them invisible to ANN search
                    // after recovery via slim snapshot.
                    self.records.restore_dirty_slots(dirty);
                    return Err(e);
                }
            }
            Ok(())
        } else {
            // First checkpoint: full flush to create .vecs + manifest
            self.flush_internal(true)
        }
    }

    // Caller must hold write_lock when mutations can race with search/CRUD.
    fn flush_internal(&self, skip_segments: bool) -> Result<()> {
        // Save engine FIRST so their generation is current when the manifest is written.
        // Saving after the checkpoint writes the old generation to the manifest, causing a
        // generation mismatch on every open that forces a full HNSW rebuild.
        if !skip_segments {
            self.flush_engine()?;
        }

        if let Some(ref storage) = self.storage {
            let mut storage = storage.write();
            self.update_storage_header_impl(&mut *storage);
            storage.set_schema(self.schema())?;

            // Get dirty slots for incremental checkpoint
            let dirty = self.records.take_dirty_slots();
            let prepared = self.prepare_flush_data()?;
            let options = prepared.as_options();

            let result = if storage.has_vec_file() {
                storage.checkpoint_incremental(&self.records, &dirty, options)
            } else {
                storage.checkpoint_full(&self.records, options)
            };
            if let Err(e) = result {
                // Restore dirty slots so the next flush retries writing them.
                // Without this, a failed checkpoint silently drops the slots —
                // same pattern as checkpoint_wal (persistence.rs:652).
                self.records.restore_dirty_slots(dirty);
                return Err(e);
            }
        }

        if let Some(ref mut text_index) = *self.text_index.write() {
            text_index.commit()?;
        }

        Ok(())
    }

    /// Check if this store has persistent storage enabled
    #[must_use]
    pub fn is_persistent(&self) -> bool {
        self.storage.is_some()
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
        let mut storage_local = if omen_path.exists() {
            OmenFile::open(path)?
        } else {
            OmenFile::create(path, self.dimensions() as u32)?
        };

        storage_local.set_metric(self.distance_metric);
        storage_local.set_hnsw_params(
            self.hnsw_m.load(std::sync::atomic::Ordering::Relaxed) as u16,
            self.hnsw_ef_construction
                .load(std::sync::atomic::Ordering::Relaxed) as u16,
            self.hnsw_ef_search
                .load(std::sync::atomic::Ordering::Relaxed) as u16,
        );
        if let Some(ref encoder) = self.muvera_encoder {
            let config = encoder.config();
            storage_local.put_config("muvera_repetitions", config.repetitions as u64)?;
            storage_local.put_config("muvera_partition_bits", config.partition_bits as u64)?;
            storage_local.put_config("muvera_seed", config.seed)?;
            storage_local.put_config("muvera_token_dim", encoder.token_dimension() as u64)?;
            if let Some(d_proj) = config.d_proj {
                storage_local.put_config("muvera_d_proj", d_proj as u64)?;
            }
            if let Some(pool_factor) = config.pool_factor {
                storage_local.put_config("muvera_pool_factor", pool_factor as u64)?;
            }
            if let Some(max_tokens) = config.max_tokens {
                storage_local.put_config("muvera_max_tokens", max_tokens as u64)?;
            }
        }
        let storage: Arc<RwLock<dyn crate::omen::StorageBackend>> =
            Arc::new(RwLock::new(storage_local));

        // Wire storage to existing engine if any
        if let Some(ref mut engine) = *self.engine.write() {
            engine.set_storage(Arc::clone(&storage));
        }

        self.storage = Some(storage);
        self.storage_path = Some(path.to_path_buf());
        Ok(self)
    }

    /// Load recovery context from storage, handling slim snapshots and manifest fallbacks.
    fn load_recovery_context(storage: &mut OmenFile) -> Result<RecoveryContext<'static>> {
        let mut slim_dirty_slots: Vec<u32> = Vec::new();
        let mut slim_snapshot_loaded = false;
        let mut slim_wal_epoch: u64 = 0;
        let mut slim_live_slots = RoaringBitmap::new();
        let mut slim_dense_slots = RoaringBitmap::new();
        let mut slim_snapshot = None;
        let manifest_wal_cutoff = storage.manifest_wal_replay_cutoff();

        if storage.records_newer_than_omen() {
            match storage.load_records_snapshot() {
                Ok(Some(slim)) => {
                    slim_live_slots = slim.id_to_slot.values().copied().collect();
                    slim_dense_slots = slim.dense_slots.iter().copied().collect();
                    slim_snapshot = Some(slim);
                }
                Ok(None) => {}
                Err(e) => {
                    println!("Failed to load slim records snapshot, using manifest: {e}");
                }
            }
        }

        // Use borrowed loading for zero-copy vectors
        let vec_mmap = storage.get_vec_mmap_arc();
        let main_mmap = storage.get_main_mmap_arc();
        let mut snapshot = storage.load_persisted_snapshot_borrowed(
            vec_mmap.as_ref(),
            main_mmap.as_ref(),
            (!slim_live_slots.is_empty()).then_some(&slim_live_slots),
            (!slim_dense_slots.is_empty()).then_some(&slim_dense_slots),
        );

        if let Some(slim) = slim_snapshot {
            tracing::info!(
                records = slim.id_to_slot.len(),
                dirty_slots = slim.dirty_since_flush.len(),
                wal_epoch = slim.wal_truncation_epoch,
                "Loaded slim records snapshot for recovery"
            );
            slim_wal_epoch = slim.wal_truncation_epoch;
            slim_dirty_slots = slim.dirty_since_flush.into_iter().collect();
            snapshot.id_to_slot = slim.id_to_slot;
            snapshot.deleted = slim.deleted;
            snapshot.metadata = slim.metadata;
            snapshot.dense_slots = slim.dense_slots;
            slim_snapshot_loaded = true;
        }

        if snapshot.multivec_config.is_none() {
            let reps = storage.get_config("muvera_repetitions")?;
            let bits = storage.get_config("muvera_partition_bits")?;
            let seed = storage.get_config("muvera_seed")?;
            let token_dim = storage.get_config("muvera_token_dim")?;
            let d_proj = storage.get_config("muvera_d_proj")?.map(|v| v as u8);
            let pool_factor = storage.get_config("muvera_pool_factor")?.map(|v| v as u8);
            let max_tokens = storage.get_config("muvera_max_tokens")?.map(|v| v as usize);
            if let (Some(repetitions), Some(partition_bits), Some(seed), Some(token_dim)) =
                (reps, bits, seed, token_dim)
            {
                snapshot.multivec_config = Some(PersistedMuveraConfig {
                    repetitions: repetitions as u8,
                    partition_bits: partition_bits as u8,
                    seed,
                    token_dim: token_dim as usize,
                    d_proj,
                    pool_factor,
                    max_tokens,
                });
            }
        }

        let dimensions = snapshot.dimensions as usize;
        let current_wal_epoch = storage.wal_truncation_epoch();

        if !slim_snapshot_loaded
            && let Some((manifest_wal_epoch, _)) = manifest_wal_cutoff
            && current_wal_epoch > manifest_wal_epoch
        {
            let referenced_slots: RoaringBitmap = snapshot.id_to_slot.values().copied().collect();
            for (slot, vector) in snapshot.vectors.iter_mut().enumerate() {
                let slot_u32 = slot as u32;
                if !referenced_slots.contains(slot_u32) {
                    *vector = None;
                    snapshot.metadata.remove(&slot_u32);
                }
            }
        }

        // Get HNSW parameters from header
        let header = storage.header();
        let quantization =
            helpers::quantization_from_id(storage.get_quantization_mode()?.unwrap_or(0));

        Ok(RecoveryContext {
            snapshot,
            slim_snapshot_loaded,
            slim_wal_epoch,
            slim_dirty_slots,
            hnsw_params: HNSWParams {
                m: header.hnsw_m as usize,
                ef_construction: header.hnsw_ef_construction as usize,
                ..Default::default()
            },
            ef_search: header.hnsw_ef_search as usize,
            distance_metric: header.metric,
            quantization,
            dimensions,
            vec_mmap,
            main_mmap,
        })
    }

    /// Build initial RecordStore from snapshot.
    fn build_initial_records(
        snapshot: &OmenSnapshot,
        dimensions: usize,
        vec_mmap: Option<Arc<Mmap>>,
        main_mmap: Option<Arc<Mmap>>,
    ) -> RecordStore {
        let deleted_bitmap: RoaringBitmap = snapshot.deleted.iter().copied().collect();
        let mut records = RecordStore::new(dimensions as u32);

        // Determine required slot count from vectors and id_to_slot
        let max_slot_from_ids = snapshot
            .id_to_slot
            .values()
            .copied()
            .max()
            .map_or(0, |m| m + 1) as usize;
        let slot_capacity = snapshot.vectors.len().max(max_slot_from_ids);
        let mut slots: Vec<Option<SnapshotRecord>> = Vec::with_capacity(slot_capacity);
        let mut live_count = 0u32;

        // Invert mapping for O(1) slot-to-ID lookup (O(N) construction)
        let slot_to_id: FxHashMap<u32, String> = snapshot
            .id_to_slot
            .iter()
            .map(|(id, &slot)| (slot, id.clone()))
            .collect();

        for slot in 0..slot_capacity {
            let slot_u32 = slot as u32;
            if deleted_bitmap.contains(slot_u32) {
                slots.push(None);
                continue;
            }

            let vec_data = snapshot.vectors.get(slot).and_then(|v| v.as_ref());

            // Find the ID for this slot (O(1) lookup)
            let id = slot_to_id.get(&slot_u32).cloned();

            if let Some(id) = id {
                let metadata = snapshot.metadata.get(&slot_u32).cloned();
                let record = SnapshotRecord {
                    id,
                    vector: vec_data.cloned(),
                    metadata,
                };
                slots.push(Some(record));
                live_count += 1;
            } else if let Some(vec_data) = vec_data {
                let id = format!("__slot_{slot}");
                let metadata = snapshot.metadata.get(&slot_u32).cloned();
                slots.push(Some(SnapshotRecord {
                    id,
                    vector: Some(vec_data.clone()),
                    metadata,
                }));
                live_count += 1;
            } else {
                slots.push(None);
            }
        }

        records.restore_snapshot_records(slots, deleted_bitmap, live_count, vec_mmap, main_mmap);
        records
    }

    /// Replay WAL entries directly into RecordStore.
    fn replay_wal_into_records(
        records: &mut RecordStore,
        storage: &mut OmenFile,
        ctx: &RecoveryContext,
    ) -> Result<ReplayedWalData> {
        let current_wal_epoch = storage.wal_truncation_epoch();
        let manifest_wal_cutoff = storage.manifest_wal_replay_cutoff();

        let wal_entries = if ctx.slim_snapshot_loaded && current_wal_epoch <= ctx.slim_wal_epoch {
            tracing::debug!(
                current_wal_epoch,
                slim_wal_epoch = ctx.slim_wal_epoch,
                "Slim snapshot: WAL epoch unchanged, skipping stale WAL replay"
            );
            vec![]
        } else {
            if ctx.slim_snapshot_loaded {
                tracing::debug!(
                    current_wal_epoch,
                    slim_wal_epoch = ctx.slim_wal_epoch,
                    "Slim snapshot: WAL epoch advanced, replaying new WAL entries"
                );
            }
            let mut entries = storage.pending_wal_entries()?;
            if !ctx.slim_snapshot_loaded
                && let Some((manifest_wal_epoch, manifest_wal_max_ts)) = manifest_wal_cutoff
                && current_wal_epoch == manifest_wal_epoch
                && let Some(max_ts) = manifest_wal_max_ts
            {
                entries.retain(|entry| entry.header.timestamp > max_ts);
            }
            entries
        };

        let mut wal_modified_slots: Vec<u32> = Vec::new();
        let mut wal_edge_store: Option<EdgeStore> = None;
        let mut wal_edge_deletes: Vec<(String, String, String)> = Vec::new();

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
                        // Infer dimensions from first WAL vector if needed (RecordStore::set handles this)
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
                        let slot = records.set(insert_data.id, insert_data.vector, metadata)?;
                        wal_modified_slots.push(slot);
                    }
                }
                WalEntryType::DeleteNode => {
                    if let Ok(delete_data) = parse_wal_delete(&entry.data) {
                        records.delete(&delete_data.id);
                    }
                }
                WalEntryType::UpsertSparse => {
                    if let Ok(data) = parse_wal_sparse(&entry.data) {
                        let metadata: Option<JsonValue> = data
                            .metadata
                            .as_ref()
                            .and_then(|bytes| serde_json::from_slice(bytes).ok());
                        let sparse =
                            crate::vector::sparse::SparseVector::new(data.indices, data.values)?;
                        let slot = if let Some(slot) = records.get_slot(&data.id) {
                            if let Some(metadata) = metadata.clone() {
                                records.update_metadata(slot, metadata)?;
                            }
                            records.update_sparse(slot, Some(sparse))?;
                            slot
                        } else {
                            let slot = records.set_without_vector(data.id, metadata);
                            records.update_sparse(slot, Some(sparse))?;
                            slot
                        };
                        wal_modified_slots.push(slot);
                    }
                }
                WalEntryType::UpsertMulti => {
                    if let Ok(data) = parse_wal_multi(&entry.data) {
                        let metadata: Option<JsonValue> = data
                            .metadata
                            .as_ref()
                            .and_then(|bytes| serde_json::from_slice(bytes).ok());
                        let slot = if let Some(slot) = records.get_slot(&data.id) {
                            if let Some(metadata) = metadata.clone() {
                                records.update_metadata(slot, metadata)?;
                            }
                            records.update_multi(slot, Some(data.tokens))?;
                            slot
                        } else {
                            let slot = records.set_without_vector(data.id, metadata);
                            records.update_multi(slot, Some(data.tokens))?;
                            slot
                        };
                        wal_modified_slots.push(slot);
                    }
                }
                WalEntryType::Checkpoint => {}
                WalEntryType::InsertEdge => {
                    if let Ok(data) = parse_wal_insert_edge(&entry.data) {
                        let metadata: Option<serde_json::Value> =
                            data.metadata.as_ref().and_then(|b| {
                                serde_json::from_slice(b)
                                    .map_err(|e| {
                                        tracing::warn!(
                                            "Corrupt edge metadata during WAL replay: {e}"
                                        );
                                    })
                                    .ok()
                            });
                        wal_edge_store
                            .get_or_insert_with(EdgeStore::new)
                            .add_edge(Edge {
                                from_id: data.from_id,
                                to_id: data.to_id,
                                edge_type: data.edge_type,
                                weight: data.weight,
                                metadata,
                            });
                    }
                }
                WalEntryType::DeleteEdge => {
                    if let Ok(data) = parse_wal_delete_edge(&entry.data) {
                        if let Some(ref mut es) = wal_edge_store {
                            es.remove_edge(&data.from_id, &data.to_id, &data.edge_type);
                        }
                        wal_edge_deletes.push((data.from_id, data.to_id, data.edge_type));
                    }
                }
            }
        }

        Ok((wal_modified_slots, wal_edge_store, wal_edge_deletes))
    }

    /// Initialize or rebuild search engine from RecordStore.
    fn initialize_segments(
        path: &Path,
        storage: &OmenFile,
        records: &RecordStore,
        modified_slots: &[u32],
        ctx: &RecoveryContext<'_>,
    ) -> Result<Option<SegmentManager>> {
        let dense_slots = records.dense_slots();
        let dense_count = dense_slots.len();
        if dense_count == 0 || ctx.dimensions == 0 {
            return Ok(None);
        }

        let segments_dir = segments_dir_for(path);
        let stored_generation = storage.get_config("segments_generation")?.unwrap_or(0);

        // Fast path: load persisted segments if available and up-to-date
        let loaded = if segments_dir.exists() {
            #[cfg(feature = "mmap")]
            let load_result = SegmentManager::load_mmap(&segments_dir);
            #[cfg(not(feature = "mmap"))]
            let load_result = SegmentManager::load(&segments_dir);

            match load_result {
                Ok(loaded)
                    if loaded.len() == dense_count
                        && loaded.generation() == stored_generation
                        && modified_slots.is_empty() =>
                {
                    let loaded_view = loaded.read_view();
                    tracing::info!(
                        segments = loaded_view.frozen_count(),
                        total_vectors = dense_count,
                        generation = stored_generation,
                        "Loaded persisted segments (skipped rebuild)"
                    );
                    Some(loaded)
                }
                Ok(mut loaded)
                    if loaded.generation() == stored_generation
                        && loaded.len() < dense_count
                        && !modified_slots.is_empty() =>
                {
                    let delta = dense_count - loaded.len();
                    let mut inserted = 0;
                    for &slot in modified_slots {
                        if records.deleted_bitmap().contains(slot)
                            || records.get_vector(slot).is_none()
                        {
                            continue;
                        }
                        records.with_vector_by_slot(slot, |vector| {
                            if let Some(vector) = vector {
                                loaded.insert_with_slot(vector, slot).map_err(|e| {
                                    anyhow::anyhow!("Failed to insert WAL delta vector: {e}")
                                })?;
                                inserted += 1;
                            }
                            Ok::<_, anyhow::Error>(())
                        })?;
                    }
                    let loaded_view = loaded.read_view();
                    tracing::info!(
                        frozen_segments = loaded_view.frozen_count(),
                        frozen_vectors = loaded.len() - inserted,
                        wal_delta = delta,
                        inserted,
                        "Partial rebuild: kept frozen segments, added WAL delta"
                    );
                    Some(loaded)
                }
                Ok(loaded) => {
                    tracing::info!(
                        segment_vectors = loaded.len(),
                        record_vectors = dense_count,
                        segment_generation = loaded.generation(),
                        stored_generation,
                        "Segment count or generation mismatch, rebuilding index"
                    );
                    None
                }
                Err(e) => {
                    tracing::warn!("Failed to load segments, rebuilding: {e}");
                    None
                }
            }
        } else {
            None
        };

        if let Some(mut engine) = loaded {
            engine.set_pending_merge_dir(segments_dir);
            Ok(Some(engine))
        } else {
            // Slow path: rebuild from vectors
            let slots = dense_slots;

            let config = SegmentConfig::new(ctx.dimensions)
                .with_params(ctx.hnsw_params)
                .with_distance(ctx.distance_metric)
                .with_quantization(ctx.quantization);

            let segs = records.with_vectors_by_slots(&slots, |vectors| {
                SegmentManager::build_parallel_with_slots_from_refs(config, vectors, &slots)
                    .map_err(|e| anyhow::anyhow!("Segment build failed: {e}"))
            })?;
            let mut segs = segs;
            segs.set_pending_merge_dir(segments_dir);
            Ok(Some(segs))
        }
    }

    /// Initialize auxiliary indexes (text, metadata, sparse, edge) from snapshot and WAL state.
    fn initialize_ancillary_indexes(
        path: &Path,
        records: &RecordStore,
        snapshot: &OmenSnapshot<'_>,
        wal_edge_store: Option<EdgeStore>,
        wal_edge_deletes: &[(String, String, String)],
        base_distance_metric: Metric,
        text_search_config: Option<&TextSearchConfig>,
    ) -> Result<AncillaryIndexes> {
        // Try to open existing text index
        let text_index_path = path.join("text_index");
        let text_index = if text_index_path.exists() {
            Some(if let Some(config) = text_search_config {
                TextIndex::open_with_config(&text_index_path, config)?
            } else {
                TextIndex::open(&text_index_path)?
            })
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

        // Reconstruct multi-vector state if config is present
        let (muvera_encoder, multivec_storage, distance_metric) =
            if let Some(ref mv_cfg) = snapshot.multivec_config {
                let config = MultiVectorConfig {
                    repetitions: mv_cfg.repetitions,
                    partition_bits: mv_cfg.partition_bits,
                    d_proj: mv_cfg.d_proj,
                    seed: mv_cfg.seed,
                    pool_factor: mv_cfg.pool_factor,
                    max_tokens: mv_cfg.max_tokens,
                };
                let encoder = MuveraEncoder::new(mv_cfg.token_dim, config)?;

                // Load persisted helper bytes, then rebuild helper state from RecordStore.
                let token_dim = mv_cfg.token_dim;
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
                (None, None, base_distance_metric)
            };

        // Reconstruct sparse index if persisted
        let sparse_index = snapshot
            .sparse_index_bytes
            .as_deref()
            .map(|bytes| {
                let (index, payloads) =
                    crate::vector::sparse::SparseIndex::from_bytes_with_payloads(bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize SparseIndex: {e}"))?;
                for (slot, sparse) in payloads {
                    records.update_sparse(slot, Some(sparse))?;
                }
                tracing::info!(vectors = index.len(), "Loaded SparseIndex from disk");
                Ok::<_, anyhow::Error>(index)
            })
            .transpose()?
            .or_else(|| {
                let payloads = records.iter_sparse();
                if payloads.is_empty() {
                    None
                } else {
                    let mut index = SparseIndex::new();
                    for (slot, sparse) in payloads {
                        index.insert(slot, &sparse);
                    }
                    Some(index)
                }
            });

        // Reconstruct edge store: start from persisted snapshot, then apply WAL delta
        let edge_store = {
            let mut base = snapshot
                .edge_store_bytes
                .as_deref()
                .map(|bytes| {
                    EdgeStore::from_bytes(bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize EdgeStore: {e}"))
                })
                .transpose()?;

            // Apply WAL deletions to the manifest base FIRST.
            if !wal_edge_deletes.is_empty()
                && let Some(ref mut store) = base
            {
                for (from_id, to_id, edge_type) in wal_edge_deletes {
                    store.remove_edge(from_id, to_id, edge_type);
                }
            }

            if let Some(wal_es) = wal_edge_store {
                // Merge WAL edges into base (or use WAL store directly if no base)
                let merged = base.get_or_insert_with(EdgeStore::new);
                for edge in wal_es.all_edges() {
                    merged.add_edge(edge);
                }
            }

            base
        };

        Ok(AncillaryIndexes {
            text_index,
            metadata_index,
            muvera_encoder,
            multivec_storage,
            distance_metric,
            sparse_index,
            edge_store,
        })
    }

    /// Flush HNSW engine to disk and update generation in storage.
    fn flush_engine(&self) -> Result<()> {
        self.with_engine_mut(|engine| {
            if let Some(engine) = engine.as_mut() {
                // Flush pending changes
                engine.flush()?;

                if let Some(ref path) = self.storage_path {
                    let segments_dir = segments_dir_for(path);
                    engine.set_pending_merge_dir(segments_dir.clone());
                    engine.save(&segments_dir)?;
                    // Update config with new generation so the manifest checkpoint below
                    // writes the correct value.
                    if let Some(ref storage) = self.storage {
                        storage
                            .write()
                            .put_config("segments_generation", engine.generation())?;
                    }
                }
            }
            Ok(())
        })
    }

    /// Update storage header with current store parameters (dimensions, HNSW, metric).
    fn update_storage_header_impl(&self, storage: &mut dyn crate::omen::StorageBackend) {
        let dims = self.records.dimensions();
        if dims > 0 {
            let _ = storage.set_dimensions(dims);
        }

        // Persist HNSW parameters and metric to header
        let _ = storage.set_hnsw_params(
            self.hnsw_m.load(std::sync::atomic::Ordering::Relaxed) as u16,
            self.hnsw_ef_construction
                .load(std::sync::atomic::Ordering::Relaxed) as u16,
            self.hnsw_ef_search
                .load(std::sync::atomic::Ordering::Relaxed) as u16,
        );
        let _ = storage.set_metric(self.distance_metric);
        let _ = storage.put_config(
            "quantization",
            helpers::quantization_to_id(self.pending_quantization.load(
                std::sync::atomic::Ordering::Relaxed,
            )),
        );
    }

    /// Prepare serialized data from all subsystems for the manifest checkpoint.
    fn prepare_flush_data(&self) -> Result<PreparedFlush> {
        // Serialize MetadataIndex for fast recovery
        let metadata_index_bytes = Some(self.metadata_index.read().to_bytes()?);

        // Export multi-vector data if present
        let (multivec_bytes, multivec_offsets, multivec_config) =
            if let Some(enc) = self.muvera_encoder.as_ref() {
                let helper = MultiVecStorage::from_slot_tokens(
                    enc.token_dimension(),
                    &self.records.iter_multi(),
                );
                let config = enc.config();
                (
                    Some(helper.vectors_to_bytes()),
                    Some(helper.offsets_to_bytes()),
                    Some(PersistedMuveraConfig {
                        repetitions: config.repetitions,
                        partition_bits: config.partition_bits,
                        seed: config.seed,
                        token_dim: enc.token_dimension(),
                        d_proj: config.d_proj,
                        pool_factor: config.pool_factor,
                        max_tokens: config.max_tokens,
                    }),
                )
            } else {
                (None, None, None)
            };

        // Export sparse index if present
        let sparse_payloads = self.records.iter_sparse();
        let sparse_index_bytes = self
            .sparse_index
            .read()
            .as_ref()
            .map(|index| index.to_bytes_with_payloads(sparse_payloads))
            .transpose()?;

        // Export edge store if present
        let edge_store_bytes = self
            .edge_store
            .read()
            .as_ref()
            .map(EdgeStore::to_bytes)
            .transpose()?;

        Ok(PreparedFlush {
            schema: self.schema(),
            metadata_index_bytes,
            multivec_bytes,
            multivec_offsets,
            multivec_config,
            sparse_index_bytes,
            edge_store_bytes,
        })
    }
}

use memmap2::Mmap;

/// Context for database recovery during open()
struct RecoveryContext<'a> {
    snapshot: OmenSnapshot<'a>,
    slim_snapshot_loaded: bool,
    slim_wal_epoch: u64,
    slim_dirty_slots: Vec<u32>,
    hnsw_params: HNSWParams,
    ef_search: usize,
    distance_metric: Metric,
    quantization: bool,
    dimensions: usize,
    vec_mmap: Option<Arc<Mmap>>,
    main_mmap: Option<Arc<Mmap>>,
}

/// Ancillary indexes recovered during open()
struct AncillaryIndexes {
    text_index: Option<TextIndex>,
    metadata_index: MetadataIndex,
    muvera_encoder: Option<MuveraEncoder>,
    multivec_storage: Option<MultiVecStorage>,
    distance_metric: Metric,
    sparse_index: Option<SparseIndex>,
    edge_store: Option<EdgeStore>,
}

/// Data prepared for flushing to disk
struct PreparedFlush {
    schema: CollectionSchema,
    metadata_index_bytes: Option<Vec<u8>>,
    multivec_bytes: Option<Vec<u8>>,
    multivec_offsets: Option<Vec<u8>>,
    multivec_config: Option<PersistedMuveraConfig>,
    sparse_index_bytes: Option<Vec<u8>>,
    edge_store_bytes: Option<Vec<u8>>,
}

impl PreparedFlush {
    fn as_options(&self) -> CheckpointOptions<'_> {
        CheckpointOptions {
            schema: Some(&self.schema),
            hnsw_bytes: None,
            metadata_index_bytes: self.metadata_index_bytes.as_deref(),
            multivec_bytes: self.multivec_bytes.as_deref(),
            multivec_offsets: self.multivec_offsets.as_deref(),
            multivec_config: self.multivec_config.clone(),
            sparse_index_bytes: self.sparse_index_bytes.as_deref(),
            edge_store_bytes: self.edge_store_bytes.as_deref(),
        }
    }
}
