//! Parallel HNSW construction
//!
//! Correct HNSW algorithm with unified flat storage and parallel execution.

use super::HNSWIndex;
use crate::vector::hnsw::error::Result;
use crate::vector::hnsw::types::{HNSWParams, Metric};
use parking_lot::Mutex;
use rayon::prelude::*;

const PARALLEL_BUILD_WARM_START: usize = 10_000;
const PARALLEL_BUILD_EF_MULTIPLIER: usize = 2;

impl HNSWIndex {
    pub fn build_parallel(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: Metric,
        _use_quantization: bool,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Self> {
        let refs = vectors.iter().map(Vec::as_slice).collect();
        Self::build_parallel_from_refs(dimensions, params, distance_fn, _use_quantization, refs)
    }

    pub fn build_parallel_from_refs(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: Metric,
        _use_quantization: bool,
        vectors: Vec<&[f32]>,
    ) -> Result<Self> {
        let mut index = Self::with_capacity(dimensions, params, distance_fn, vectors.len());

        for vector in &vectors {
            index.validate_vector(vector)?;
            let level = index.random_level();
            index.storage.add_node(vector, level);
        }

        if vectors.is_empty() {
            return Ok(index);
        }

        index.entry_point = Some(0);

        let warm_start = vectors.len().min(PARALLEL_BUILD_WARM_START);
        for (node_id, vector) in vectors.iter().enumerate().take(warm_start).skip(1) {
            let node_id = node_id as u32;
            let level = index.storage.get_node_level(node_id);
            let entry_point = index.entry_point.expect("entry point initialized");
            index.insert_into_graph_from_entry(
                node_id,
                vector,
                level,
                entry_point,
                params.ef_construction,
            )?;

            if level > index.storage.get_node_level(entry_point) {
                index.entry_point = Some(node_id);
            }
        }

        if warm_start == vectors.len() {
            return Ok(index);
        }

        let initial_entry = index.entry_point.expect("entry point initialized");
        let entry_point = Mutex::new((initial_entry, index.storage.get_node_level(initial_entry)));

        (warm_start..vectors.len())
            .into_par_iter()
            .try_for_each(|idx| {
                let node_id = idx as u32;
                let vector = vectors[idx];
                let level = index.storage.get_node_level(node_id);
                let current_entry = entry_point.lock().0;

                index.insert_into_graph_from_entry(
                    node_id,
                    vector,
                    level,
                    current_entry,
                    params.ef_construction * PARALLEL_BUILD_EF_MULTIPLIER,
                )?;

                let mut entry = entry_point.lock();
                if level > entry.1 {
                    *entry = (node_id, level);
                }
                Ok::<(), crate::vector::hnsw::error::HNSWError>(())
            })?;

        index.entry_point = Some(entry_point.into_inner().0);
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
