//! Vector storage with HNSW indexing for approximate nearest neighbor search.

pub mod hnsw;
pub mod metadata;
pub mod muvera;
pub mod sparse;
pub mod store;
pub mod types;

use anyhow::Result;
use crate::omen::Metric;

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

/// Statistics from an optimization operation
pub struct OptimizationStats {
    pub vectors_reordered: usize,
    pub segments_merged: usize,
}

/// Core trait for dense vector retrieval engines.
pub trait VectorEngine: Send + Sync {
    /// Dimension of vectors this engine supports.
    fn dimensions(&self) -> usize;

    /// Distance metric used by this engine.
    fn metric(&self) -> Metric;

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
}
