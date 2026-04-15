mod error;
mod index;
pub mod storage;
pub mod types;
pub mod visited;

use crate::omen::StorageBackend;
use crate::vector::{EngineSearchResult, MutableVectorEngine, VectorEngine, VectorEngineView};
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SegmentManager {
    pub(crate) index: HNSWIndex,
}

impl SegmentManager {
    pub fn new(_config: SegmentConfig) -> Self {
        let index = HNSWIndex::new(
            0,
            HNSWParams::default(),
            crate::vector::hnsw::types::Metric::L2,
        );
        Self { index }
    }

    pub fn build_parallel_with_slots_from_refs(
        _config: SegmentConfig,
        vectors: Vec<&[f32]>,
        _slots: &[u32],
    ) -> Result<Self> {
        let mut index = HNSWIndex::new(
            vectors[0].len(),
            HNSWParams::default(),
            crate::vector::hnsw::types::Metric::L2,
        );
        for v in vectors {
            index.insert(v)?;
        }
        Ok(Self { index })
    }

    pub fn load_mmap(path: &std::path::Path) -> Result<Self> {
        let index = HNSWIndex::load(path)?;
        Ok(Self { index })
    }

    pub fn read_view(&self) -> Arc<PublishedSegmentView> {
        Arc::new(PublishedSegmentView {
            index: self.index.clone_for_view(),
        })
    }

    pub fn insert_batch_parallel_from_refs(
        &mut self,
        vectors: &[&[f32]],
        slots: &[u32],
    ) -> Result<()> {
        self.index.insert_batch_parallel_from_refs(vectors, slots)
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        self.index.save(path)
    }

    pub fn insert_with_slot(&mut self, vector: &[f32], slot: u32) -> Result<u32> {
        self.index.insert_with_slot(vector, slot)
    }
}

impl MutableVectorEngine for SegmentManager {
    fn dimensions(&self) -> usize {
        self.index.storage.vectors.dim
    }
    fn metric(&self) -> crate::Metric {
        self.index.distance_fn
    }
    fn len(&self) -> usize {
        VectorEngineView::len(&self.index)
    }
    fn insert(&mut self, vector: &[f32], slot: u32) -> anyhow::Result<u32> {
        MutableVectorEngine::insert(&mut self.index, vector, slot)
    }
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> anyhow::Result<Vec<EngineSearchResult>> {
        MutableVectorEngine::search(&self.index, query, k, ef)
    }
    fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_fn: &(dyn Fn(u32) -> bool + Sync + Send),
    ) -> anyhow::Result<Vec<EngineSearchResult>> {
        MutableVectorEngine::search_with_filter(&self.index, query, k, ef, filter_fn)
    }
    fn flush(&mut self) -> anyhow::Result<()> {
        MutableVectorEngine::flush(&mut self.index)
    }
    fn checkpoint(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        MutableVectorEngine::checkpoint(&mut self.index, path)
    }
    fn set_storage(&mut self, storage: Arc<RwLock<dyn StorageBackend>>) {
        MutableVectorEngine::set_storage(&mut self.index, storage)
    }
    fn set_pending_merge_dir(&mut self, dir: std::path::PathBuf) {
        MutableVectorEngine::set_pending_merge_dir(&mut self.index, dir)
    }
    fn memory_usage(&self) -> usize {
        MutableVectorEngine::memory_usage(&self.index)
    }
    fn mutable_len(&self) -> usize {
        MutableVectorEngine::mutable_len(&self.index)
    }
    fn freeze_mutable(&mut self) -> anyhow::Result<()> {
        MutableVectorEngine::freeze_mutable(&mut self.index)
    }
    fn insert_batch_parallel(
        &mut self,
        vectors: Vec<Vec<f32>>,
        slots: &[u32],
    ) -> anyhow::Result<()> {
        MutableVectorEngine::insert_batch_parallel(&mut self.index, vectors, slots)
    }
    fn generation(&self) -> u64 {
        MutableVectorEngine::generation(&self.index)
    }
    fn optimize(&mut self) -> anyhow::Result<crate::vector::OptimizationStats> {
        MutableVectorEngine::optimize(&mut self.index)
    }
}

impl VectorEngine for SegmentManager {
    fn read_view(&self) -> Arc<dyn VectorEngineView> {
        Arc::new(PublishedSegmentView {
            index: self.index.clone_for_view(),
        })
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct PublishedSegmentView {
    pub(crate) index: HNSWIndex,
}

impl VectorEngineView for PublishedSegmentView {
    fn total_memory(&self) -> usize {
        VectorEngineView::total_memory(&self.index)
    }
    fn frozen_count(&self) -> usize {
        VectorEngineView::frozen_count(&self.index)
    }
    fn mutable_len(&self) -> usize {
        VectorEngineView::mutable_len(&self.index)
    }
    fn segment_capacity(&self) -> usize {
        VectorEngineView::segment_capacity(&self.index)
    }
    fn len(&self) -> usize {
        VectorEngineView::len(&self.index)
    }
    fn generation(&self) -> u64 {
        VectorEngineView::generation(&self.index)
    }
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> anyhow::Result<Vec<EngineSearchResult>> {
        VectorEngineView::search(&self.index, query, k, ef)
    }
    fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_fn: &(dyn Fn(u32) -> bool + Sync + Send),
    ) -> anyhow::Result<Vec<EngineSearchResult>> {
        VectorEngineView::search_with_filter(&self.index, query, k, ef, filter_fn)
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct SegmentConfig {
    pub capacity: usize,
}

impl SegmentConfig {
    pub fn new(_dim: usize) -> Self {
        Self {
            capacity: usize::MAX,
        }
    }
    pub fn with_params(self, _params: crate::vector::hnsw::types::HNSWParams) -> Self {
        self
    }
    pub fn with_distance(self, _metric: crate::Metric) -> Self {
        self
    }
    pub fn with_quantization(self, _enabled: bool) -> Self {
        self
    }
    pub fn with_capacity(mut self, cap: usize) -> Self {
        self.capacity = cap;
        self
    }
}

// Public API exports
pub use error::{HNSWError, Result};
pub use index::{HNSWIndex, HNSWIndexBuilder, HNSWQuantization, ParallelBuilder};
pub use storage::HNSWStorage;
pub use types::{Candidate, HNSWParams, Metric};
pub use types::{Cosine, Distance, L2, NegDot};
