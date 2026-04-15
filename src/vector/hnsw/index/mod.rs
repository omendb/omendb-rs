// HNSW Index - Unified Storage Architecture
//
// Optimized for SOTA performance with contiguous matrices and zero-copy traversal.

macro_rules! dispatch_distance {
    ($distance_fn:expr, $Type:ident => $body:expr) => {
        match $distance_fn {
            crate::vector::hnsw::types::Metric::L2 => {
                type $Type = crate::vector::hnsw::types::L2;
                $body
            }
            crate::vector::hnsw::types::Metric::Cosine => {
                type $Type = crate::vector::hnsw::types::Cosine;
                $body
            }
            crate::vector::hnsw::types::Metric::InnerProduct => {
                type $Type = crate::vector::hnsw::types::NegDot;
                $body
            }
        }
    };
}

mod insert;
mod parallel;
mod persistence;
mod search;
mod stats;

#[cfg(test)]
mod tests;

pub use parallel::ParallelBuilder;

use super::error::{HNSWError, Result};
use super::storage::HNSWStorage;
use super::types::{HNSWParams, Metric};
use crate::omen::StorageBackend;
use crate::vector::{EngineSearchResult, MutableVectorEngine, VectorEngine, VectorEngineView};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Serialize, Deserialize)]
pub struct HNSWIndex {
    pub(super) storage: HNSWStorage,
    pub(super) entry_point: Option<u32>,
    pub(super) params: HNSWParams,
    pub(super) distance_fn: Metric,
    pub(super) rng_state: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub num_vectors: usize,
    pub dimensions: usize,
    pub entry_point: Option<u32>,
    pub max_level: u8,
    pub level_distribution: Vec<usize>,
    pub avg_neighbors_l0: f32,
    pub max_neighbors_l0: usize,
    pub memory_bytes: usize,
    pub params: HNSWParams,
    pub distance_function: Metric,
    pub quantization_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum HNSWQuantization {
    #[default]
    None,
    SQ8,
}

pub struct HNSWIndexBuilder {
    dimensions: Option<usize>,
    params: HNSWParams,
    metric: Metric,
}

impl HNSWIndexBuilder {
    pub fn new() -> Self {
        Self {
            dimensions: None,
            params: HNSWParams::default(),
            metric: Metric::L2,
        }
    }
    pub fn dimensions(mut self, d: usize) -> Self {
        self.dimensions = Some(d);
        self
    }
    pub fn m(mut self, m: usize) -> Self {
        self.params.m = m;
        self
    }
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.params.ef_construction = ef;
        self
    }
    pub fn metric(mut self, m: Metric) -> Self {
        self.metric = m;
        self
    }
    pub fn build(self) -> Result<HNSWIndex> {
        let d = self
            .dimensions
            .ok_or_else(|| HNSWError::InvalidParams("dim required".into()))?;
        Ok(HNSWIndex::new(d, self.params, self.metric))
    }
}

impl HNSWIndex {
    pub fn new(dim: usize, params: HNSWParams, distance_fn: Metric) -> Self {
        Self {
            storage: HNSWStorage::new(dim, params.max_level as usize, params.m),
            entry_point: None,
            rng_state: params.seed,
            params,
            distance_fn,
        }
    }

    pub fn builder() -> HNSWIndexBuilder {
        HNSWIndexBuilder::new()
    }

    pub fn clone_for_view(&self) -> Self {
        Self {
            storage: postcard::from_bytes(&postcard::to_allocvec(&self.storage).unwrap()).unwrap(),
            entry_point: self.entry_point,
            params: self.params.clone(),
            distance_fn: self.distance_fn,
            rng_state: self.rng_state,
        }
    }

    pub fn insert_batch_parallel_from_refs(
        &mut self,
        vectors: &[&[f32]],
        _slots: &[u32],
    ) -> Result<()> {
        for v in vectors {
            self.insert(v)?;
        }
        Ok(())
    }

    pub fn insert_with_slot(&mut self, vector: &[f32], _slot: u32) -> Result<u32> {
        self.insert(vector)
    }

    pub fn dimensions(&self) -> usize {
        self.storage.vectors.dim
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }
}

impl VectorEngineView for HNSWIndex {
    fn total_memory(&self) -> usize {
        self.storage.memory_usage()
    }
    fn frozen_count(&self) -> usize {
        0
    }
    fn mutable_len(&self) -> usize {
        self.storage.len()
    }
    fn segment_capacity(&self) -> usize {
        usize::MAX
    }
    fn len(&self) -> usize {
        self.storage.len()
    }
    fn generation(&self) -> u64 {
        0
    }
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> anyhow::Result<Vec<EngineSearchResult>> {
        let results = self
            .search(query, k, ef)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(results
            .into_iter()
            .map(|r| EngineSearchResult {
                slot: r.id as u32,
                distance: r.distance,
            })
            .collect())
    }
    fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        _filter_fn: &(dyn Fn(u32) -> bool + Sync + Send),
    ) -> anyhow::Result<Vec<EngineSearchResult>> {
        VectorEngineView::search(self, query, k, ef)
    }
}

impl MutableVectorEngine for HNSWIndex {
    fn dimensions(&self) -> usize {
        self.storage.vectors.dim
    }
    fn metric(&self) -> crate::Metric {
        self.distance_fn
    }
    fn len(&self) -> usize {
        VectorEngineView::len(self)
    }
    fn insert(&mut self, vector: &[f32], _slot: u32) -> anyhow::Result<u32> {
        self.insert(vector)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> anyhow::Result<Vec<EngineSearchResult>> {
        VectorEngineView::search(self, query, k, ef)
    }
    fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_fn: &(dyn Fn(u32) -> bool + Sync + Send),
    ) -> anyhow::Result<Vec<EngineSearchResult>> {
        VectorEngineView::search_with_filter(self, query, k, ef, filter_fn)
    }
    fn flush(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
    fn checkpoint(&mut self, _path: &std::path::Path) -> anyhow::Result<()> {
        Ok(())
    }
    fn set_storage(&mut self, _storage: Arc<RwLock<dyn StorageBackend>>) {}
    fn set_pending_merge_dir(&mut self, _dir: std::path::PathBuf) {}
    fn memory_usage(&self) -> usize {
        VectorEngineView::total_memory(self)
    }
    fn mutable_len(&self) -> usize {
        VectorEngineView::mutable_len(self)
    }
    fn freeze_mutable(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
    fn insert_batch_parallel(
        &mut self,
        vectors: Vec<Vec<f32>>,
        _slots: &[u32],
    ) -> anyhow::Result<()> {
        for v in vectors {
            self.insert(&v)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        }
        Ok(())
    }
    fn generation(&self) -> u64 {
        VectorEngineView::generation(self)
    }
    fn optimize(&mut self) -> anyhow::Result<crate::vector::OptimizationStats> {
        Ok(crate::vector::OptimizationStats {
            vectors_reordered: 0,
            segments_merged: 0,
        })
    }
}

impl VectorEngine for HNSWIndex {
    fn read_view(&self) -> Arc<dyn VectorEngineView> {
        Arc::new(self.clone_for_view())
    }
}
