use serde_json::Value as JsonValue;
use parking_lot::MappedRwLockReadGuard;
use std::sync::atomic::Ordering;

use super::VectorStore;
use crate::omen::{Metric, OmenFile};

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
}

impl VectorStore {
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
        self.oversample.store(oversample.to_bits(), Ordering::Relaxed);
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

    /// Get the segment manager (for benchmarking/diagnostics)
    #[must_use]
    pub fn segments(&self) -> Option<MappedRwLockReadGuard<'_, crate::vector::hnsw::SegmentManager>> {
        parking_lot::RwLockReadGuard::try_map(self.segments.read(), |segments| segments.as_ref())
            .ok()
    }

    /// Get comprehensive store diagnostics.
    #[must_use]
    pub fn info(&self) -> StoreInfo {
        let vector_bytes = self
            .records
            .iter_live()
            .map(|(_, r)| r.vector.len() * 4)
            .sum::<usize>();

        let (frozen_count, mutable_vecs, graph_bytes, segment_capacity) =
            if let Some(segments) = self.segments.read().as_ref() {
                let config = segments.config();
                let graph = segments.total_memory() - vector_bytes;
                (
                    segments.frozen_count(),
                    segments.mutable_len(),
                    graph,
                    config.segment_capacity,
                )
            } else {
                (
                    0,
                    0,
                    0,
                    self.segment_capacity
                        .unwrap_or(crate::vector::hnsw::SegmentConfig::DEFAULT_CAPACITY),
                )
            };

        let wal_entries = self.storage.read().as_ref().map_or(0, OmenFile::wal_len);

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
            wal_entries,
            is_persistent: self.storage.read().is_some(),
            hnsw_m: self.hnsw_m.load(Ordering::Relaxed),
            hnsw_ef_construction: self.hnsw_ef_construction.load(Ordering::Relaxed),
            hnsw_ef_search: self.hnsw_ef_search.load(Ordering::Relaxed),
            quantization: self.pending_quantization.load(Ordering::Relaxed),
            segment_capacity,
        }
    }
}
