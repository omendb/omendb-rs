use anyhow::Result;
use serde_json::Value as JsonValue;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::{MetadataFilter, VectorStore};
use crate::catalog::{
    CollectionSchema, MultiEncoderKind, MultiSchema, SparseIndexKind, SparseSchema, TextSchema,
};
use crate::omen::Metric;
use crate::vector::VectorEngineView;
use crate::vector::hnsw::{PublishedSegmentView, SegmentManager};

/// Comprehensive store diagnostics.
///
/// Returned by [`VectorStore::info()`]. All memory estimates are in bytes.
#[derive(Debug, Clone)]
pub struct StoreInfo {
    // Counts
    pub vector_count: usize,
    pub deleted_count: usize,
    pub dimensions: usize,
    pub metric: Metric,
    // Segments
    pub frozen_segment_count: usize,
    pub mutable_segment_vectors: usize,
    // Memory
    pub vector_bytes: usize,
    pub graph_bytes: usize,
    pub total_memory_bytes: usize,
    // Storage
    pub wal_entries: u64,
    pub is_persistent: bool,
    // Config
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
    pub quantization: bool,
    pub segment_capacity: usize,
    pub schema: CollectionSchema,
}

impl VectorStore {
    fn modality_error(schema: &CollectionSchema, operation: &str, required: &str) -> anyhow::Error {
        anyhow::anyhow!(
            "{operation} requires {required} in the collection schema (dense={}, sparse={}, multi={}, text={})",
            schema.dense.is_some(),
            schema.sparse.is_some(),
            schema.multi.is_some(),
            schema.text.is_some(),
        )
    }

    pub(crate) fn require_dense_schema(&self, operation: &str) -> Result<()> {
        let schema = self.schema();
        if schema.dense.is_none() {
            return Err(Self::modality_error(&schema, operation, "dense vectors"));
        }
        Ok(())
    }

    pub(crate) fn require_sparse_schema(&self, operation: &str) -> Result<()> {
        let schema = self.schema();
        if schema.sparse.is_none() {
            return Err(Self::modality_error(&schema, operation, "sparse vectors"));
        }
        Ok(())
    }

    pub(crate) fn require_multi_schema(&self, operation: &str) -> Result<()> {
        let schema = self.schema();
        if schema.multi.is_none() {
            return Err(Self::modality_error(
                &schema,
                operation,
                "multi-vector tokens",
            ));
        }
        Ok(())
    }

    pub(crate) fn require_text_schema(&self, operation: &str) -> Result<()> {
        let schema = self.schema();
        if schema.text.is_none() {
            return Err(Self::modality_error(&schema, operation, "text search"));
        }
        Ok(())
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

    /// Count vectors matching a metadata filter.
    #[must_use]
    pub fn count_by_filter(&self, filter: &MetadataFilter) -> usize {
        let metadata_index = self.metadata_index.read();
        if let Some(bitmap) = filter.evaluate_bitmap(&metadata_index) {
            return bitmap
                .iter()
                .filter(|&slot| self.records.is_live(slot))
                .count();
        }

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
            .filter_map(|(_, record)| {
                let vector = record.vector?;
                let metadata = record.metadata.clone().unwrap_or_default();
                Some((record.id.clone(), vector.to_vec(), metadata))
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
            .filter_map(|(_, r)| r.vector.map(|vector| vector.as_slice().len() * 4))
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
    pub fn set_ef_search(&self, ef_search: usize) {
        self.hnsw_ef_search.store(ef_search, Ordering::Relaxed);
    }

    /// Get HNSW `ef_search` parameter
    #[must_use]
    pub fn ef_search(&self) -> usize {
        self.hnsw_ef_search.load(Ordering::Relaxed)
    }

    /// Get HNSW M parameter (neighbors per node)
    #[must_use]
    pub fn hnsw_m(&self) -> usize {
        self.hnsw_m.load(Ordering::Relaxed)
    }

    /// Get HNSW ef_construction parameter (build quality)
    #[must_use]
    pub fn hnsw_ef_construction(&self) -> usize {
        self.hnsw_ef_construction.load(Ordering::Relaxed)
    }

    /// Check if SQ8 quantization is enabled
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        self.pending_quantization.load(Ordering::Relaxed)
    }

    /// Enable or disable SQ8 rescoring
    pub fn set_rescore(&self, rescore: bool) {
        self.rescore.store(rescore, Ordering::Relaxed);
    }

    /// Set oversample factor for SQ8 rescoring
    pub fn set_oversample(&self, oversample: f32) {
        self.oversample
            .store(oversample.to_bits(), Ordering::Relaxed);
    }

    /// Check if SQ8 rescoring is enabled
    #[must_use]
    pub fn rescore(&self) -> bool {
        self.rescore.load(Ordering::Relaxed)
    }

    /// Get oversample factor for SQ8 rescoring
    #[must_use]
    pub fn oversample(&self) -> f32 {
        f32::from_bits(self.oversample.load(Ordering::Relaxed))
    }

    /// Number of deleted (tombstoned) records
    #[must_use]
    pub fn deleted_count(&self) -> usize {
        self.records.deleted_count() as usize
    }

    /// Run a closure against the current immutable segment read view.
    ///
    /// This exposes the published search state without leaking the mutable
    /// `VectorEngine` lock shape to callers.
    pub fn with_segment_view<T>(
        &self,
        f: impl FnOnce(Option<Arc<PublishedSegmentView>>) -> T,
    ) -> T {
        let view = self.published_view.load();
        f((**view).clone())
    }

    /// Check whether a search engine is initialized.
    pub(crate) fn has_engine(&self) -> bool {
        self.engine.read().is_some()
    }

    /// Run a closure against the mutable search engine slot.
    ///
    /// This centralizes serialized write-side access so topology mutations can
    /// gradually move behind one explicit coordination seam.
    pub(crate) fn with_engine_mut<T>(
        &self,
        f: impl FnOnce(&mut Option<SegmentManager>) -> Result<T>,
    ) -> Result<T> {
        let mut engine = self.engine.write();
        let result = f(&mut engine);

        // Sync published view ArcSwap after mutation
        let new_view = engine.as_ref().map(SegmentManager::read_view);
        self.published_view.store(Arc::new(new_view));

        result
    }

    /// Get comprehensive store diagnostics.
    #[must_use]
    pub fn info(&self) -> StoreInfo {
        let vector_bytes = self
            .records
            .iter_live()
            .filter_map(|(_, r)| r.vector.map(|vector| vector.as_slice().len() * 4))
            .sum::<usize>();

        let (frozen_count, mutable_vecs, graph_bytes, segment_capacity) =
            self.with_segment_view(|engine_view| {
                if let Some(engine_view) = engine_view {
                    let total_mem = engine_view.total_memory();
                    let graph = total_mem.saturating_sub(vector_bytes);
                    (
                        engine_view.frozen_count(),
                        engine_view.mutable_len(),
                        graph,
                        engine_view.segment_capacity(),
                    )
                } else {
                    (0, 0, 0, 0)
                }
            });

        let wal_entries = self.storage.as_ref().map_or(0, |s| s.read().wal_len());

        StoreInfo {
            vector_count: self.records.len() as usize,
            deleted_count: self.records.deleted_count() as usize,
            dimensions: self.dimensions(),
            metric: self.distance_metric,
            frozen_segment_count: frozen_count,
            mutable_segment_vectors: mutable_vecs,
            vector_bytes,
            graph_bytes,
            total_memory_bytes: vector_bytes + graph_bytes,
            wal_entries: wal_entries as u64,
            is_persistent: self.storage.is_some(),
            hnsw_m: self.hnsw_m.load(Ordering::Relaxed),
            hnsw_ef_construction: self.hnsw_ef_construction.load(Ordering::Relaxed),
            hnsw_ef_search: self.hnsw_ef_search.load(Ordering::Relaxed),
            quantization: self.pending_quantization.load(Ordering::Relaxed),
            segment_capacity,
            schema: self.schema(),
        }
    }

    /// Derive the current collection schema from the live runtime state.
    ///
    /// Standalone `VectorStore` instances are not catalog-attached yet, so the
    /// returned schema name is best-effort: persistent stores use the backing
    /// path's final component, in-memory stores use an empty name.
    #[must_use]
    pub fn schema(&self) -> CollectionSchema {
        let name = self.schema_name.clone().unwrap_or_else(|| {
            self.storage_path
                .as_ref()
                .and_then(|path| path.file_name())
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        });

        let dense = self.dense_schema.read().clone().map(|mut dense| {
            dense.dim = self.dimensions() as u32;
            dense
        });

        let sparse = self.has_sparse().then_some(SparseSchema {
            index_kind: SparseIndexKind::InvertedExact,
            max_nonzero: None,
        });

        let multi = self.multi_vector_config().map(|config| MultiSchema {
            token_dim: self.token_dimension().unwrap_or(self.dimensions()) as u32,
            encoder: MultiEncoderKind::Muvera,
            repetitions: config.repetitions,
            partition_bits: config.partition_bits,
            d_proj: config.d_proj,
            seed: config.seed,
            max_tokens: config.max_tokens.map(|v| v as u32),
            pool_factor: config.pool_factor,
        });

        let text = if self.has_text_search() {
            let config = self.text_search_config.read().clone().unwrap_or_default();
            Some(TextSchema {
                tokenizer: config.tokenizer,
                writer_buffer_mb: config.writer_buffer_mb as u32,
            })
        } else {
            None
        };

        CollectionSchema {
            name,
            metric: self.metric(),
            dense,
            sparse,
            multi,
            text,
            graph: self
                .graph_schema
                .read()
                .clone()
                .or_else(|| self.has_edges().then_some(super::default_graph_schema())),
        }
    }
}
