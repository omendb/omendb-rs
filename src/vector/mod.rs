//! Vector storage with HNSW indexing for approximate nearest neighbor search.

pub mod hnsw;
pub mod metadata;
pub mod muvera;
pub mod sparse;
pub mod store;
pub mod types;

use anyhow::Result;

// Re-export main types
pub use hnsw::{HNSWIndex, HNSWIndexBuilder, HNSWQuantization};
pub use metadata::{FieldIndex, Filter, FilterValue, MetadataIndex};
pub use store::{
    MetadataFilter, SearchResult, ThreadSafeVectorStore, VectorStore, VectorStoreOptions,
};
pub use types::Vector;

/// Result from a VectorEngine search
#[derive(Debug, Clone)]
pub struct EngineSearchResult {
    /// Slot ID of the record
    pub slot: u32,
    /// Distance from the query
    pub distance: f32,
}

impl EngineSearchResult {
    pub fn new(slot: u32, distance: f32) -> Self {
        Self { slot, distance }
    }
}

/// Core trait for a read-only view of a vector engine.
///
/// Allows lock-free read access to search results while the main engine
/// may be undergoing mutations.
pub trait VectorEngineView: Send + Sync {
    /// Total number of vectors visible in this view.
    fn len(&self) -> usize;

    /// Check if the view is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Search for k-nearest neighbors in this view.
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<EngineSearchResult>>;

    /// Search with a filter predicate in this view.
    fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_fn: &(dyn Fn(u32) -> bool + Sync + Send),
    ) -> Result<Vec<EngineSearchResult>>;

    /// Number of frozen segments visible in this view.
    fn frozen_count(&self) -> usize;

    /// Number of vectors visible in the mutable segment.
    fn mutable_len(&self) -> usize;

    /// Generation of the currently published engine state.
    fn generation(&self) -> u64;

    /// Segment capacity for the currently published topology.
    fn segment_capacity(&self) -> usize;

    /// Total graph memory visible across mutable and frozen segments.
    fn total_memory(&self) -> usize;
}

/// Marker trait for frozen dense generations.
///
/// This is intentionally the same capability set as `VectorEngineView`, but the
/// separate name makes the mutable/frozen split explicit in the codebase.
pub trait FrozenVectorEngineView: VectorEngineView {}

impl<T: VectorEngineView + ?Sized> FrozenVectorEngineView for T {}

/// Statistics from an optimization operation
pub struct OptimizationStats {
    pub vectors_reordered: usize,
    pub segments_merged: usize,
}

/// Core trait for a mutable dense retrieval engine.
pub trait MutableVectorEngine: Send + Sync {
    /// Dimension of vectors this engine supports.
    fn dimensions(&self) -> usize;

    /// Distance metric used by this engine.
    fn metric(&self) -> crate::Metric;

    /// Number of vectors currently indexed.
    fn len(&self) -> usize;

    /// Check if the engine is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert a vector with a specific slot ID.
    fn insert(&mut self, vector: &[f32], slot: u32) -> Result<u32>;

    /// Search for k-nearest neighbors.
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<EngineSearchResult>>;

    /// Search with a filter predicate.
    fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_fn: &(dyn Fn(u32) -> bool + Sync + Send),
    ) -> Result<Vec<EngineSearchResult>>;

    /// Flush pending changes (e.g., freeze mutable segment).
    fn flush(&mut self) -> Result<()>;

    /// Optimize the index (e.g., merge segments, reorder for cache locality).
    fn optimize(&mut self) -> Result<OptimizationStats>;

    /// Persist the engine state to a directory.
    fn checkpoint(&mut self, path: &std::path::Path) -> Result<()>;

    /// Set a storage backend for persistence.
    fn set_storage(
        &mut self,
        storage: std::sync::Arc<parking_lot::RwLock<dyn crate::omen::StorageBackend>>,
    );

    /// Set the directory for background merge artifacts.
    fn set_pending_merge_dir(&mut self, dir: std::path::PathBuf);

    /// Estimated memory usage in bytes.
    fn memory_usage(&self) -> usize;

    /// Number of vectors in the mutable portion of the engine.
    fn mutable_len(&self) -> usize;

    /// Force freeze the mutable portion into a read-only state.
    fn freeze_mutable(&mut self) -> Result<()>;

    /// Add multiple vectors in parallel.
    fn insert_batch_parallel(&mut self, vectors: Vec<Vec<f32>>, slots: &[u32]) -> Result<()>;

    /// Generation counter for the current engine state.
    fn generation(&self) -> u64;
}

/// Full dense engine interface.
///
/// This extends the mutable engine role with a read-only published view.
pub trait VectorEngine: MutableVectorEngine {
    /// Get a thread-safe read-only view of the current engine state.
    fn read_view(&self) -> std::sync::Arc<dyn VectorEngineView>;
}
