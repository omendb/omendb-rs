//! Persistence operations for VectorStore.
//!
//! This module contains methods for opening, saving, and managing persistent
//! vector stores using the `.omen` file format.

use super::VectorStore;
use super::helpers;
use super::record_store::{Record, RecordStore};
use super::{DEFAULT_HNSW_EF_CONSTRUCTION, DEFAULT_HNSW_EF_SEARCH, DEFAULT_HNSW_M};
use crate::omen::{
    CheckpointOptions, OmenFile, PersistedMuveraConfig, WalEntryType, parse_wal_delete,
    parse_wal_delete_edge, parse_wal_insert, parse_wal_insert_edge,
};
use crate::text::TextIndex;
use crate::vector::hnsw::{HNSWParams, SegmentConfig, SegmentManager};
use crate::vector::metadata::MetadataIndex;
use crate::vector::muvera::{MultiVecStorage, MultiVectorConfig, MuveraEncoder};
use crate::vector::sparse::SparseIndex;
use crate::vector::store::edge_store::{Edge, EdgeStore};
use crate::vector::store::options::VectorStoreOptions;
use anyhow::Result;
use parking_lot::RwLock;
use roaring::RoaringBitmap;
use serde_json::Value as JsonValue;
use std::path::Path;

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

impl VectorStore {
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
        let mut storage = if omen_path.exists() {
            OmenFile::open(path)?
        } else {
            OmenFile::create(path, 0)?
        };

        // Load slim records snapshot if it was written more recently than the manifest.
        // This happens when auto-checkpoint ran after the last flush(). The snapshot captures
        // full id_to_slot/metadata state, and dirty_since_flush drives partial segment rebuild.
        let mut slim_dirty_slots: Vec<u32> = Vec::new();
        let mut slim_snapshot_loaded = false;
        // When slim snapshot is loaded, compare WAL truncation epochs to decide whether
        // to replay WAL entries:
        //   WAL epoch > snapshot epoch → WAL was truncated after snapshot, entries are new → replay
        //   WAL epoch == snapshot epoch → WAL not truncated after snapshot, entries stale → skip
        let mut slim_wal_epoch: u64 = 0;
        let mut slim_live_slots = RoaringBitmap::new();
        let mut slim_snapshot = None;
        let manifest_wal_cutoff = storage.manifest_wal_replay_cutoff();
        if storage.records_newer_than_omen() {
            match storage.load_records_snapshot() {
                Ok(Some(slim)) => {
                    slim_live_slots = slim.id_to_slot.values().copied().collect();
                    slim_snapshot = Some(slim);
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!("Failed to load slim records snapshot, using manifest: {e}");
                }
            }
        }
        let mut snapshot = storage.load_persisted_snapshot_with_live_slots(
            (!slim_live_slots.is_empty()).then_some(&slim_live_slots),
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
            slim_snapshot_loaded = true;
        }

        let mut dimensions = snapshot.dimensions as usize;
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
        let distance_metric = header.metric;
        let hnsw_m = header.hnsw_m as usize;
        let hnsw_ef_construction = header.hnsw_ef_construction as usize;
        let hnsw_ef_search = header.hnsw_ef_search as usize;

        // Check quantization
        let _is_quantized = storage.is_quantized()?;
        let quantization =
            helpers::quantization_from_id(storage.get_quantization_mode()?.unwrap_or(0));

        // Build RecordStore from snapshot
        let mut deleted_bitmap: RoaringBitmap = snapshot.deleted.iter().copied().collect();

        // Determine required slot count from vectors and id_to_slot
        let max_slot_from_ids = snapshot
            .id_to_slot
            .values()
            .copied()
            .max()
            .map_or(0, |m| m + 1) as usize;
        let slot_capacity = snapshot.vectors.len().max(max_slot_from_ids);
        let mut slots: Vec<Option<Record>> = Vec::with_capacity(slot_capacity);

        for slot in 0..slot_capacity {
            let slot_u32 = slot as u32;
            if deleted_bitmap.contains(slot_u32) {
                slots.push(None);
                continue;
            }

            let vec_data = snapshot.vectors.get(slot).and_then(|v| v.as_ref());

            // Find the ID for this slot
            let id = snapshot
                .id_to_slot
                .iter()
                .find(|&(_, &s)| s == slot_u32)
                .map(|(id, _)| id.clone());

            if let Some(id) = id {
                let metadata = snapshot.metadata.get(&slot_u32).cloned();
                let vector = vec_data.cloned().unwrap_or_default();
                slots.push(Some(Record::new(id, vector, metadata)));
            } else if let Some(vec_data) = vec_data {
                let id = format!("__slot_{slot}");
                let metadata = snapshot.metadata.get(&slot_u32).cloned();
                slots.push(Some(Record::new(id, vec_data.clone(), metadata)));
            } else {
                slots.push(None);
            }
        }

        let mut records =
            RecordStore::from_snapshot(slots, deleted_bitmap.clone(), dimensions as u32);

        // Replay WAL entries directly into RecordStore (Phase 5 architecture)
        // Track slots modified by WAL replay for partial segment rebuild.
        //
        // WAL replay decision when a slim snapshot is loaded uses truncation epochs:
        //   WAL epoch > snapshot epoch → WAL was truncated after the snapshot (new entries) → replay
        //   WAL epoch == snapshot epoch → WAL not truncated after snapshot (stale entries) → skip
        //
        // RecordStore::set() is NOT idempotent: replaying an existing ID allocates a new
        // slot and marks the old one deleted, corrupting the segment rebuild. Skipping stale
        // WAL entries prevents this. Old snapshots (wal_truncation_epoch == 0) conservatively
        // skip WAL replay to maintain the prior safe-but-lossy behavior.
        let wal_entries = if slim_snapshot_loaded && current_wal_epoch <= slim_wal_epoch {
            tracing::debug!(
                current_wal_epoch,
                slim_wal_epoch,
                "Slim snapshot: WAL epoch unchanged, skipping stale WAL replay"
            );
            vec![]
        } else {
            if slim_snapshot_loaded {
                tracing::debug!(
                    current_wal_epoch,
                    slim_wal_epoch,
                    "Slim snapshot: WAL epoch advanced, replaying new WAL entries"
                );
            }
            let mut entries = storage.pending_wal_entries()?;
            if !slim_snapshot_loaded
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
        // Track WAL edge deletions separately so they can be applied to the manifest base.
        // DeleteEdge in wal_edge_store only removes edges added *in the same WAL session*;
        // deletions of manifest-loaded edges must be applied after the merge.
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
                        let slot = records.set(insert_data.id, insert_data.vector, metadata)?;
                        wal_modified_slots.push(slot);
                    }
                }
                WalEntryType::DeleteNode => {
                    if let Ok(delete_data) = parse_wal_delete(&entry.data) {
                        records.delete(&delete_data.id);
                    }
                }
                WalEntryType::Checkpoint => {
                    // No-op: checkpoint entries mark safe recovery points
                }
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
                        // Remove from wal_edge_store (handles same-session inserts)
                        if let Some(ref mut es) = wal_edge_store {
                            es.remove_edge(&data.from_id, &data.to_id, &data.edge_type);
                        }
                        // Track for post-merge removal against the manifest base
                        wal_edge_deletes.push((data.from_id, data.to_id, data.edge_type));
                    }
                }
            }
        }

        // Use slim snapshot's accumulated dirty slots instead of WAL-derived slots when
        // the slim snapshot was loaded. slim_dirty_slots covers all dirty slots since the last
        // flush, accumulated across multiple auto-checkpoints, even though the WAL was truncated.
        let modified_slots: Vec<u32> = if slim_snapshot_loaded && !slim_dirty_slots.is_empty() {
            slim_dirty_slots
        } else {
            wal_modified_slots
        };

        // Update deleted bitmap after WAL replay
        deleted_bitmap.clone_from(&records.deleted_bitmap());

        // Build segments from vectors
        let active_count = records.len() as usize;

        let segments = if active_count > 0 && dimensions > 0 {
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
                        if loaded.len() == active_count
                            && loaded.generation() == stored_generation
                            && modified_slots.is_empty() =>
                    {
                        // Fast path: segments match RecordStore exactly and no WAL delta.
                        // Must require modified_slots.is_empty() — WAL updates (set on existing
                        // key) tombstone the old slot and allocate a new slot, keeping active_count
                        // unchanged. If modified_slots is non-empty, those new slots are not in the
                        // loaded segments and must be inserted via partial or full rebuild.
                        tracing::info!(
                            segments = loaded.frozen_count(),
                            total_vectors = active_count,
                            generation = stored_generation,
                            "Loaded persisted segments (skipped rebuild)"
                        );
                        Some(loaded)
                    }
                    Ok(mut loaded)
                        if loaded.generation() == stored_generation
                            && loaded.len() < active_count
                            && !modified_slots.is_empty() =>
                    {
                        // Partial rebuild: keep frozen segments, insert WAL delta
                        let delta = active_count - loaded.len();
                        let mut inserted = 0;
                        for &slot in &modified_slots {
                            if let Some(record) = records.get_by_slot(slot)
                                && !records.deleted_bitmap().contains(slot)
                            {
                                loaded.insert_with_slot(&record.vector, slot).map_err(|e| {
                                    anyhow::anyhow!("Failed to insert WAL delta vector: {e}")
                                })?;
                                inserted += 1;
                            }
                        }
                        tracing::info!(
                            frozen_segments = loaded.frozen_count(),
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
                            record_vectors = active_count,
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

            if let Some(mut segments) = loaded {
                // Enable background merge persistence so merge results survive crashes
                segments.set_pending_merge_dir(segments_dir_for(path));
                Some(segments)
            } else {
                // Slow path: rebuild from vectors
                let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(active_count);
                let mut slots: Vec<u32> = Vec::with_capacity(active_count);
                for (slot, record) in records.iter_live() {
                    vectors.push(record.vector.clone());
                    slots.push(slot);
                }

                let config = SegmentConfig::new(dimensions)
                    .with_params(HNSWParams {
                        m: hnsw_m,
                        ef_construction: hnsw_ef_construction,
                        ..Default::default()
                    })
                    .with_distance(distance_metric)
                    .with_quantization(quantization);

                let mut segs = SegmentManager::build_parallel_with_slots(config, vectors, &slots)
                    .map_err(|e| anyhow::anyhow!("Segment build failed: {e}"))?;
                segs.set_pending_merge_dir(segments_dir_for(path));
                Some(segs)
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
            if let Some(mv_cfg) = snapshot.multivec_config {
                let config = MultiVectorConfig {
                    repetitions: mv_cfg.repetitions,
                    partition_bits: mv_cfg.partition_bits,
                    d_proj: mv_cfg.d_proj,
                    seed: mv_cfg.seed,
                    pool_factor: mv_cfg.pool_factor,
                    max_tokens: mv_cfg.max_tokens,
                };
                let encoder = MuveraEncoder::new(mv_cfg.token_dim, config)?;

                // Reconstruct storage from persisted bytes
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
                (None, None, distance_metric)
            };

        // Reconstruct sparse index if persisted
        let sparse_index = snapshot
            .sparse_index_bytes
            .as_deref()
            .map(|bytes| {
                let index = crate::vector::sparse::SparseIndex::from_bytes(bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize SparseIndex: {e}"))?;
                tracing::info!(vectors = index.len(), "Loaded SparseIndex from disk");
                Ok::<_, anyhow::Error>(index)
            })
            .transpose()?;

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
            // This handles edges that were in the manifest snapshot but deleted via cascade.
            // Must happen before merging WAL inserts so that a delete→re-add sequence
            // correctly preserves the re-added edge.
            if !wal_edge_deletes.is_empty()
                && let Some(ref mut store) = base
            {
                for (from_id, to_id, edge_type) in &wal_edge_deletes {
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

        Ok(Self {
            records,
            segments: RwLock::new(segments),
            metadata_index: RwLock::new(metadata_index),
            storage: RwLock::new(Some(storage)),
            storage_path: Some(path.to_path_buf()),
            text_index: RwLock::new(text_index),
            text_search_config: RwLock::new(None),
            pending_quantization: (quantization).into(),
            hnsw_m: (if hnsw_m > 0 { hnsw_m } else { DEFAULT_HNSW_M }).into(),
            hnsw_ef_construction: (if hnsw_ef_construction > 0 {
                hnsw_ef_construction
            } else {
                DEFAULT_HNSW_EF_CONSTRUCTION
            })
            .into(),
            hnsw_ef_search: (if hnsw_ef_search > 0 {
                hnsw_ef_search
            } else {
                DEFAULT_HNSW_EF_SEARCH
            })
            .into(),
            distance_metric,
            muvera_encoder,
            multivec_storage: RwLock::new(multivec_storage),
            sparse_index: RwLock::new(sparse_index),
            edge_store: RwLock::new(edge_store),
            segment_capacity: None,
            rescore: (quantization).into(),
            oversample: (3.0f32.to_bits()).into(),
            max_memory_bytes: None,
            auto_compact_threshold: (0.25f32.to_bits()).into(),
            write_lock: parking_lot::Mutex::new(()),
        })
    }

    /// Open a persistent vector store with specified dimensions
    ///
    /// Like `open()` but ensures dimensions are set for new databases.
    pub fn open_with_dimensions(path: impl AsRef<Path>, dimensions: usize) -> Result<Self> {
        let store = Self::open(path)?;
        if store.dimensions() == 0 {
            store.records.set_dimensions(dimensions as u32);
            if let Some(ref mut storage) = *store.storage.write() {
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
                if let Some(ref mut storage) = *store.storage.write() {
                    storage.put_config("dimensions", options.dimensions as u64)?;
                }
            }

            // Apply ef_search if specified
            if let Some(ef) = options.ef_search {
                store.set_ef_search(ef);
            }

            // Apply rescore/oversample options
            if let Some(rescore) = options.rescore {
                store.set_rescore(rescore);
            }
            if let Some(oversample) = options.oversample {
                store.set_oversample(oversample);
            }
            store.max_memory_bytes = options.max_memory_bytes;

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
        let pending_quantization = options.quantization;

        // Save dimensions to storage if set
        if dimensions > 0 {
            storage.put_config("dimensions", dimensions as u64)?;
        }

        // Save quantization mode to storage if set
        if options.quantization {
            storage.put_quantization_mode(helpers::quantization_to_id(true))?;
        }

        // Initialize text index if enabled
        let text_index = if let Some(ref config) = options.text_search_config {
            let text_path = path.join("text_index");
            Some(TextIndex::open_with_config(&text_path, config)?)
        } else {
            None
        };

        let rescore = if let Some(r) = options.rescore {
            r
        } else {
            options.quantization
        };
        let oversample = options.oversample.unwrap_or(3.0);

        Ok(Self {
            records: RecordStore::new(dimensions as u32),
            segments: RwLock::new(None),
            metadata_index: RwLock::new(MetadataIndex::new()),
            storage: RwLock::new(Some(storage)),
            storage_path: Some(path.to_path_buf()),
            text_index: RwLock::new(text_index),
            text_search_config: RwLock::new(options.text_search_config.clone()),
            pending_quantization: pending_quantization.into(),
            hnsw_m: m.into(),
            hnsw_ef_construction: ef_construction.into(),
            hnsw_ef_search: ef_search.into(),
            distance_metric,
            muvera_encoder: None,
            multivec_storage: RwLock::new(None),
            sparse_index: RwLock::new(None),
            edge_store: RwLock::new(None),
            segment_capacity: None,
            rescore: rescore.into(),
            oversample: oversample.to_bits().into(),
            max_memory_bytes: options.max_memory_bytes,
            auto_compact_threshold: (0.25f32.to_bits()).into(),
            write_lock: parking_lot::Mutex::new(()),
        })
    }

    /// Build an in-memory vector store with custom options.
    pub fn build_with_options(options: &VectorStoreOptions) -> Result<Self> {
        let dimensions = options.dimensions;

        let m = options.m.unwrap_or(16);
        let ef_construction = options.ef_construction.unwrap_or(100);
        let ef_search = options.ef_search.unwrap_or(100);

        let distance_metric = options.metric.unwrap_or(Metric::L2);

        let pending_quantization = options.quantization;

        let text_index = if let Some(ref config) = options.text_search_config {
            Some(TextIndex::open_in_memory_with_config(config)?)
        } else {
            None
        };

        let rescore = if let Some(r) = options.rescore {
            r
        } else {
            options.quantization
        };
        let oversample = options.oversample.unwrap_or(3.0);

        Ok(Self {
            records: RecordStore::new(dimensions as u32),
            segments: RwLock::new(None),
            metadata_index: RwLock::new(MetadataIndex::new()),
            storage: RwLock::new(None),
            storage_path: None,
            text_index: RwLock::new(text_index),
            text_search_config: RwLock::new(options.text_search_config.clone()),
            pending_quantization: pending_quantization.into(),
            hnsw_m: m.into(),
            hnsw_ef_construction: ef_construction.into(),
            hnsw_ef_search: ef_search.into(),
            distance_metric,
            muvera_encoder: None,
            multivec_storage: RwLock::new(None),
            sparse_index: RwLock::new(None),
            edge_store: RwLock::new(None),
            segment_capacity: None,
            rescore: rescore.into(),
            oversample: oversample.to_bits().into(),
            max_memory_bytes: options.max_memory_bytes,
            auto_compact_threshold: (0.25f32.to_bits()).into(),
            write_lock: parking_lot::Mutex::new(()),
        })
    }

    /// Flush all pending changes to disk
    ///
    /// Commits vector/metadata changes and HNSW index to `.omen` storage.
    /// Uses RecordStore as single source of truth (no duplicated state in OmenFile).
    ///
    /// If the tombstone ratio exceeds `auto_compact_threshold` (default 25%),
    /// compaction runs automatically before persisting.
    pub fn flush(&self) -> Result<()> {
        let _lock = self.write_lock.lock();
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
        let _lock = self.write_lock.lock();
        self.checkpoint_wal_locked()
    }

    pub(super) fn checkpoint_wal_locked(&self) -> Result<()> {
        let has_vec = self
            .storage
            .read()
            .as_ref()
            .is_some_and(OmenFile::has_vec_file);
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
            if let Some(ref mut storage) = *self.storage.write()
                && let Err(e) = storage.checkpoint_vectors_only(&self.records, &dirty)
            {
                // Restore dirty slots so the next checkpoint retries writing them.
                // Without this, failed .vecs writes leave slots in id_to_slot but
                // missing from dirty_since_flush, making them invisible to ANN search
                // after recovery via slim snapshot.
                self.records.restore_dirty_slots(dirty);
                return Err(e.into());
            }
            Ok(())
        } else {
            // First checkpoint: full flush to create .vecs + manifest
            self.flush_internal(true)
        }
    }

    // Caller must hold write_lock when mutations can race with search/CRUD.
    fn flush_internal(&self, skip_segments: bool) -> Result<()> {
        // Save segments FIRST so their generation is current when the manifest is written.
        // Saving after the checkpoint writes the old generation to the manifest, causing a
        // generation mismatch on every open that forces a full HNSW rebuild.
        if !skip_segments {
            self.with_segments_mut(|segments| {
                if let Some(segments) = segments.as_mut() {
                    // Wait for any background merge to finish before saving
                    segments.drain_pending_merge();

                    if let Some(ref path) = self.storage_path {
                        let segments_dir = segments_dir_for(path);
                        segments.set_pending_merge_dir(segments_dir.clone());
                        segments
                            .save(&segments_dir)
                            .map_err(|e| anyhow::anyhow!("Failed to save segments: {e}"))?;
                        // Update config with new generation so the manifest checkpoint below
                        // writes the correct value.
                        if let Some(ref mut storage) = *self.storage.write() {
                            storage.put_config("segments_generation", segments.generation())?;
                        }
                    }
                }
                Ok(())
            })?;
        }

        if let Some(ref mut storage) = *self.storage.write() {
            // Ensure dimensions are set in storage header
            let dims = self.records.dimensions();
            if dims > 0 {
                storage.set_dimensions(dims);
            }

            // Persist HNSW parameters and metric to header
            storage.set_hnsw_params(
                self.hnsw_m.load(std::sync::atomic::Ordering::Relaxed) as u16,
                self.hnsw_ef_construction
                    .load(std::sync::atomic::Ordering::Relaxed) as u16,
                self.hnsw_ef_search
                    .load(std::sync::atomic::Ordering::Relaxed) as u16,
            );
            storage.set_metric(self.distance_metric);

            // Get dirty slots for incremental checkpoint
            let dirty = self.records.take_dirty_slots();

            // Serialize MetadataIndex for fast recovery
            let metadata_index_bytes = Some(self.metadata_index.read().to_bytes()?);

            // Export multi-vector data if present
            let multivec_guard = self.multivec_storage.read();
            let (multivec_bytes, multivec_offsets, multivec_config) =
                if let (Some(mvs), Some(enc)) =
                    (multivec_guard.as_ref(), self.muvera_encoder.as_ref())
                {
                    let config = enc.config();
                    (
                        Some(mvs.vectors_to_bytes()),
                        Some(mvs.offsets_to_bytes()),
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
            let sparse_index_bytes = self
                .sparse_index
                .read()
                .as_ref()
                .map(SparseIndex::to_bytes)
                .transpose()?;

            // Export edge store if present
            let edge_store_bytes = self
                .edge_store
                .read()
                .as_ref()
                .map(EdgeStore::to_bytes)
                .transpose()?;

            let options = CheckpointOptions {
                hnsw_bytes: None,
                metadata_index_bytes: metadata_index_bytes.as_deref(),
                multivec_bytes: multivec_bytes.as_deref(),
                multivec_offsets: multivec_offsets.as_deref(),
                multivec_config,
                sparse_index_bytes: sparse_index_bytes.as_deref(),
                edge_store_bytes: edge_store_bytes.as_deref(),
            };

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
                return Err(e.into());
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
        self.storage.read().is_some()
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
        if self.storage.read().is_some() {
            anyhow::bail!("Store already has persistence enabled");
        }

        let path = path.as_ref();
        let omen_path = OmenFile::compute_omen_path(path);
        let mut storage = if omen_path.exists() {
            OmenFile::open(path)?
        } else {
            OmenFile::create(path, self.dimensions() as u32)?
        };

        storage.set_metric(self.distance_metric);
        storage.set_hnsw_params(
            self.hnsw_m.load(std::sync::atomic::Ordering::Relaxed) as u16,
            self.hnsw_ef_construction
                .load(std::sync::atomic::Ordering::Relaxed) as u16,
            self.hnsw_ef_search
                .load(std::sync::atomic::Ordering::Relaxed) as u16,
        );
        *self.storage.write() = Some(storage);
        self.storage_path = Some(path.to_path_buf());
        Ok(self)
    }
}
