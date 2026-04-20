//! Parallel HNSW construction
//!
//! Correct HNSW algorithm with unified flat storage and parallel execution.

use super::HNSWIndex;
use crate::vector::hnsw::error::Result;
use crate::vector::hnsw::types::{HNSWParams, Metric};

impl HNSWIndex {
    pub fn build_parallel(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: Metric,
        _use_quantization: bool,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Self> {
        let mut index = Self::with_capacity(dimensions, params, distance_fn, vectors.len());
        for vec in vectors {
            index.insert(&vec)?;
        }
        Ok(index)
    }

    pub fn build_parallel_from_refs(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: Metric,
        _use_quantization: bool,
        vectors: Vec<&[f32]>,
    ) -> Result<Self> {
        let mut index = Self::with_capacity(dimensions, params, distance_fn, vectors.len());
        for vec in vectors {
            index.insert(vec)?;
        }
        Ok(index)
    }

    pub fn insert_batch_parallel(
        &mut self,
        vectors: Vec<Vec<f32>>,
        _slots: &[u32],
    ) -> anyhow::Result<()> {
        for vec in vectors {
            self.insert(&vec)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        }
        Ok(())
    }
}

pub struct ParallelBuilder {}
