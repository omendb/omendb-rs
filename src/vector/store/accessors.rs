use serde_json::Value as JsonValue;

use super::VectorStore;

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
    pub fn set_ef_search(&mut self, ef_search: usize) {
        self.hnsw_ef_search = ef_search;
    }

    /// Get HNSW `ef_search` parameter
    #[must_use]
    pub fn ef_search(&self) -> usize {
        self.hnsw_ef_search
    }

    /// Get HNSW M parameter (neighbors per node)
    #[must_use]
    pub fn hnsw_m(&self) -> usize {
        self.hnsw_m
    }

    /// Get HNSW ef_construction parameter (build quality)
    #[must_use]
    pub fn hnsw_ef_construction(&self) -> usize {
        self.hnsw_ef_construction
    }

    /// Check if SQ8 quantization is enabled
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        self.pending_quantization
    }

    /// Number of deleted (tombstoned) records
    #[must_use]
    pub fn deleted_count(&self) -> usize {
        self.records.deleted_count() as usize
    }

    /// Get the segment manager (for benchmarking/diagnostics)
    #[must_use]
    pub fn segments(&self) -> Option<&crate::vector::hnsw::SegmentManager> {
        self.segments.as_ref()
    }
}
