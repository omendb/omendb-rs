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

// SearchResult is defined in this module and re-exported from lib.rs

use super::hnsw::{HNSWParams, SegmentConfig, SegmentManager};
use super::muvera::{MultiVecStorage, MultiVectorConfig, MuveraEncoder};
use super::types::Vector;
use super::QuantizationMode;
use crate::omen::{MetadataIndex, OmenFile};
use crate::text::{TextIndex, TextSearchConfig};
use anyhow::Result;
use rayon::prelude::*;
use serde_json::Value as JsonValue;
use std::path::PathBuf;

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

    /// Create base VectorStore with default field values.
    ///
    /// All optional fields are None, HNSW params are defaults.
    /// Used by public constructors to avoid field duplication.
    fn with_defaults(dimensions: usize, distance_metric: Metric) -> Self {
        Self {
            records: RecordStore::new(dimensions as u32),
            segments: None,
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
            distance_metric,
            muvera_encoder: None,
            multivec_storage: None,
            max_tokens: DEFAULT_MAX_TOKENS,
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

        let mut store = Self::with_defaults(fde_dim, Metric::InnerProduct);
        store.muvera_encoder = Some(encoder);
        store.multivec_storage = Some(MultiVecStorage::new(token_dim));
        store
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
        let mut store = Self::with_defaults(dimensions, Metric::L2);
        store.rescore_enabled = true;
        store.pending_quantization = Some(mode);
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
        if self.segments.is_none() {
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

        // Insert into segments
        if let Some(ref mut segments) = self.segments {
            // Note: mark_deleted not needed - RecordStore filtering handles deleted nodes
            segments
                .insert_with_slot(&vector.data, slot as u32)
                .map_err(|e| anyhow::anyhow!("Segment insert failed: {e}"))?;
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
    ///
    /// # Ordering Guarantee
    ///
    /// This method processes updates before inserts to ensure slot ordering is predictable.
    /// **WARNING:** `set_multi_batch` in `multivec_ops.rs` depends on this ordering to
    /// correctly align tokens with FDEs. Do not change without updating that code.
    pub fn set_batch(&mut self, batch: Vec<(String, Vector, JsonValue)>) -> Result<Vec<usize>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        // Separate batch into updates and inserts (updates processed first - see docstring)
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

            // Insert into segments
            if let Some(ref mut segments) = self.segments {
                segments
                    .insert_with_slot(&vector.data, new_slot)
                    .map_err(|e| anyhow::anyhow!("Segment insert failed: {e}"))?;
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

            // Check if this is a new index (no existing segments)
            let is_new_index = self.segments.is_none();

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
                    SegmentManager::build_parallel_with_slots(config, vectors_data, &slots)
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

                    // Insert into segments
                    if let Some(ref mut segments) = self.segments {
                        segments
                            .insert_with_slot(&vector.data, slot)
                            .map_err(|e| anyhow::anyhow!("Segment insert failed: {e}"))?;
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

        if self.segments.is_none() {
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
        self.segments.is_none() && self.records.len() > 100
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
            self.segments.as_ref(),
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

    /// Search with SearchParams/SearchOptions struct
    ///
    /// Unified method using builder pattern for search options.
    ///
    /// # Example
    /// ```ignore
    /// let params = SearchParams::new().filter(my_filter).ef(200);
    /// let results = store.search_with_params(&query, k, &params)?;
    /// ```
    pub fn search_with_params(
        &self,
        query: &Vector,
        k: usize,
        params: &SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        self.search_with_options_readonly(
            query,
            k,
            params.filter.as_ref(),
            params.ef,
            params.max_distance,
        )
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
    ///
    /// For segment-based storage, this merges all frozen segments into one
    /// for better search locality. Returns the number of vectors in the merged segment.
    pub fn optimize(&mut self) -> Result<usize> {
        if let Some(ref mut segments) = self.segments {
            // Flush mutable segment first
            segments.flush().map_err(|e| anyhow::anyhow!("{e}"))?;
            // Merge all frozen segments
            if let Some(stats) = segments
                .merge_all_frozen()
                .map_err(|e| anyhow::anyhow!("{e}"))?
            {
                return Ok(stats.vectors_merged);
            }
        }
        Ok(0)
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
    }

    /// Get HNSW `ef_search` parameter
    #[must_use]
    pub fn get_ef_search(&self) -> Option<usize> {
        // Return stored value even if no index yet
        Some(self.hnsw_ef_search)
    }

    /// Get HNSW `ef_search` parameter (Rust API guidelines naming)
    #[must_use]
    pub fn ef_search(&self) -> usize {
        self.hnsw_ef_search
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

        // Rebuild segments with new contiguous slots
        if self.records.is_empty() {
            self.segments = None;
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
}
