//! .omen single-file storage format for `OmenDB`
//!
//! Layout:
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ HEADER (4KB, page 0)                                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │ VECTOR SECTION (page-aligned, mmap)                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │ GRAPH SECTION (page-aligned, mmap)                          │
//! ├─────────────────────────────────────────────────────────────┤
//! │ METADATA SECTION                                            │
//! ├─────────────────────────────────────────────────────────────┤
//! │ WAL SECTION (append-only, at end)                           │
//! └─────────────────────────────────────────────────────────────┘
//! ```

mod file;
mod graph;
mod header;
mod manifest;
pub mod mmap_backend;
pub mod mock_backend;
mod vectors;
mod wal;

use anyhow::Result;

/// Core trait for storage backends.
///
/// Abstracting OmenFile allows us to swap between:
/// - OmenFile (Single file, custom layout, WAL)
/// - MmapBackend (Simple file-per-segment, no WAL)
/// - MemoryBackend (Testing only)
pub trait StorageBackend: Send + Sync {
    /// Dimension of vectors in this storage.
    fn dimensions(&self) -> usize;

    /// Distance metric used by this storage.
    fn metric(&self) -> Metric;

    /// Set dimensions.
    fn set_dimensions(&mut self, dimensions: u32) -> Result<()>;

    /// Set distance metric.
    fn set_metric(&mut self, metric: Metric) -> Result<()>;

    /// Set HNSW parameters.
    fn set_hnsw_params(&mut self, m: u16, ef_construction: u16, ef_search: u16) -> Result<()>;

    /// Append a vector insert to the WAL.
    fn log_insert(&mut self, id: &str, vector: &[f32], metadata: &serde_json::Value) -> Result<()>;

    /// Append a vector delete to the WAL.
    fn log_delete(&mut self, id: &str) -> Result<()>;

    /// Append a sparse upsert to the WAL.
    fn log_upsert_sparse(
        &mut self,
        id: &str,
        sparse: &crate::vector::sparse::SparseVector,
        metadata: &serde_json::Value,
    ) -> Result<()>;

    /// Append a multivector token upsert to the WAL.
    fn log_upsert_multi(
        &mut self,
        id: &str,
        tokens: &[Vec<f32>],
        metadata: &serde_json::Value,
    ) -> Result<()>;

    /// Append an edge insert to the WAL.
    fn log_insert_edge(
        &mut self,
        from_id: &str,
        to_id: &str,
        edge_type: &str,
        weight: f32,
        metadata: Option<&[u8]>,
    ) -> Result<()>;

    /// Append an edge delete to the WAL.
    fn log_delete_edge(&mut self, from_id: &str, to_id: &str, edge_type: &str) -> Result<()>;

    /// Checkpoint the storage (flush WAL to segments).
    fn checkpoint(&mut self) -> Result<()>;

    /// Sync to disk.
    fn sync(&mut self) -> Result<()>;

    /// Store configuration value.
    fn put_config(&mut self, key: &str, value: u64) -> Result<()>;

    /// Get current WAL length (number of entries).
    fn wal_len(&self) -> usize;

    /// Check if storage has a vectors file (.vecs).
    fn has_vec_file(&self) -> bool;

    /// Fast path: write dirty .vecs slots, sync WAL, skip manifest.
    fn checkpoint_vectors_only(
        &mut self,
        records: &crate::vector::store::record_store::RecordStore,
        dirty_slots: &roaring::RoaringBitmap,
    ) -> Result<()>;

    /// Incremental checkpoint: write dirty slots and update manifest.
    fn checkpoint_incremental(
        &mut self,
        records: &crate::vector::store::record_store::RecordStore,
        dirty_slots: &roaring::RoaringBitmap,
        options: CheckpointOptions,
    ) -> Result<()>;

    /// Full checkpoint: write all live records and update manifest.
    fn checkpoint_full(
        &mut self,
        records: &crate::vector::store::record_store::RecordStore,
        options: CheckpointOptions,
    ) -> Result<()>;
}

pub use file::{
    CheckpointOptions, OmenFile, OmenSnapshot, PersistedMuveraConfig, SlimRecordsSnapshot,
    WalDeleteData, WalDeleteEdgeData, WalInsertData, WalInsertEdgeData, WalMultiData,
    WalSparseData, parse_wal_delete, parse_wal_delete_edge, parse_wal_insert,
    parse_wal_insert_edge, parse_wal_multi, parse_wal_sparse,
};
pub use graph::GraphSection;
pub use header::{HEADER_SIZE, MAGIC, Metric, OmenHeader, VERSION_MAJOR, VERSION_MINOR};
pub use manifest::{ManifestHeader, NodeLocation, OmenFooter, OmenManifest, SegmentType};
pub use vectors::VectorSection;
pub use wal::{Wal, WalEntry, WalEntryType};

/// Page size for alignment (8KB optimal for `NVMe`)
pub const PAGE_SIZE: usize = 8192;

/// Align a value to page boundary
#[inline]
#[must_use]
pub const fn align_to_page(value: usize) -> usize {
    (value + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_to_page() {
        assert_eq!(align_to_page(0), 0);
        assert_eq!(align_to_page(1), PAGE_SIZE);
        assert_eq!(align_to_page(PAGE_SIZE), PAGE_SIZE);
        assert_eq!(align_to_page(PAGE_SIZE + 1), PAGE_SIZE * 2);
    }
}
