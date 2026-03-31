//! Query planner for coordinating retrieval from multiple engines.
//!
//! The QueryPlanner is responsible for:
//! - Coordinating dense (VectorEngine) and text (TextEngine) retrieval.
//! - Executing hybrid fusion (RRF).
//! - Managing multi-vector reranking (MaxSim).
//! - Applying metadata filters across retrieval paths.

use crate::text::{TextEngine, weighted_reciprocal_rank_fusion, HybridResult, weighted_reciprocal_rank_fusion_with_subscores};
use crate::vector::{VectorEngine, EngineSearchResult};
use crate::vector::store::SearchResult;
use crate::vector::store::record_store::RecordStore;
use crate::vector::metadata::MetadataIndex;
use crate::vector::store::helpers;
use crate::vector::store::input::HybridParams;
use anyhow::Result;
use std::sync::Arc;

pub struct QueryPlanner<'a> {
    records: &'a RecordStore,
    vector_engine: &'a dyn VectorEngine,
    text_engine: Option<&'a dyn TextEngine>,
    metadata_index: Option<&'a MetadataIndex>,
}

impl<'a> QueryPlanner<'a> {
    pub fn new(
        records: &'a RecordStore,
        vector_engine: &'a dyn VectorEngine,
        text_engine: Option<&'a dyn TextEngine>,
        metadata_index: Option<&'a MetadataIndex>,
    ) -> Self {
        Self {
            records,
            vector_engine,
            text_engine,
            metadata_index,
        }
    }

    /// Execute a standard dense search.
    pub fn search_dense(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<SearchResult>> {
        let results = self.vector_engine.search(query, k, ef)?;
        Ok(self.map_engine_results(results))
    }

    /// Execute a filtered dense search.
    pub fn search_dense_filtered(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_fn: &(dyn Fn(u32) -> bool + Sync + Send),
    ) -> Result<Vec<SearchResult>> {
        let results = self.vector_engine.search_with_filter(query, k, ef, filter_fn)?;
        Ok(self.map_engine_results(results))
    }

    /// Execute a hybrid search (vector + text) with optional metadata filtering.
    pub fn search_hybrid(
        &self,
        query_vector: &[f32],
        query_text: &str,
        k: usize,
        params: &HybridParams,
    ) -> Result<Vec<SearchResult>> {
        let text_engine = self.text_engine.ok_or_else(|| {
            anyhow::anyhow!("Text search not enabled")
        })?;

        // 1. Fetch candidates from both engines
        // We fetch more candidates than k to improve fusion quality
        let fetch_k = if params.filter.is_some() { k * 4 } else { k * 2 };
        let ef = params.ef.unwrap_or(fetch_k * 2);
// Vector retrieval with optional filtering
let vector_results = if let Some(ref filter) = params.filter {
    let filter_bitmap = self.metadata_index.and_then(|idx| filter.evaluate_bitmap(idx));

    if let Some(bitmap) = filter_bitmap {
        let filter_fn = move |slot: u32| -> bool { bitmap.contains(slot) };
        self.vector_engine.search_with_filter(query_vector, fetch_k, ef, &filter_fn)?
    } else {
        let filter_fn = |slot: u32| -> bool {
            if let Some(rec) = self.records.get_by_slot(slot) {
                if let Some(ref meta) = rec.metadata {
                    return filter.matches(meta);
                }
            }
            false
        };        self.vector_engine.search_with_filter(query_vector, fetch_k, ef, &filter_fn)?
    }
} else {
    self.vector_engine.search(query_vector, fetch_k, ef)?
};

        let mut text_results = text_engine.search(query_text, fetch_k)?;

        // Apply metadata filter to text results if present
        if let Some(ref filter) = params.filter {
            text_results.retain(|r| {
                if let Some(rec) = self.records.get(&r.id) {
                    if let Some(ref meta) = rec.metadata {
                        return filter.matches(meta);
                    }
                }
                false
            });
        }

        // 2. Convert to (id, score) for fusion
        let vector_scored_ids: Vec<(String, f32)> = vector_results
            .into_iter()
            .filter_map(|r| {
                self.records.get_id(r.slot).map(|id| (id.clone(), r.distance))
            })
            .collect();

        let text_scored_ids: Vec<(String, f32)> = text_results
            .into_iter()
            .map(|r| (r.id, r.score))
            .collect();

        // 3. Fuse results using RRF
        let fused = weighted_reciprocal_rank_fusion(
            vector_scored_ids,
            text_scored_ids,
            k,
            params.rrf_k,
            params.alpha,
        );

        // 4. Attach metadata and return
        Ok(fused
            .into_iter()
            .map(|(id, score)| {
                let metadata = self.records.get(&id)
                    .and_then(|r| r.metadata.clone())
                    .unwrap_or_else(helpers::default_metadata);
                SearchResult::new(id, score, metadata)
            })
            .collect())
    }

    /// Execute a hybrid search returning separate keyword and semantic scores.
    pub fn search_hybrid_with_subscores(
        &self,
        query_vector: &[f32],
        query_text: &str,
        k: usize,
        params: &HybridParams,
    ) -> Result<Vec<(HybridResult, serde_json::Value)>> {
        let text_engine = self.text_engine.ok_or_else(|| {
            anyhow::anyhow!("Text search not enabled")
        })?;

        let fetch_k = if params.filter.is_some() { k * 4 } else { k * 2 };
        let ef = params.ef.unwrap_or(fetch_k * 2);
// Vector retrieval with optional filtering
let vector_results = if let Some(ref filter) = params.filter {
    let filter_bitmap = self.metadata_index.and_then(|idx| filter.evaluate_bitmap(idx));

    if let Some(bitmap) = filter_bitmap {
        let filter_fn = move |slot: u32| -> bool { bitmap.contains(slot) };
        self.vector_engine.search_with_filter(query_vector, fetch_k, ef, &filter_fn)?
    } else {
        let filter_fn = |slot: u32| -> bool {
            if let Some(rec) = self.records.get_by_slot(slot) {
                if let Some(ref meta) = rec.metadata {
                    return filter.matches(meta);
                }
            }
            false
        };        self.vector_engine.search_with_filter(query_vector, fetch_k, ef, &filter_fn)?
    }
} else {
    self.vector_engine.search(query_vector, fetch_k, ef)?
};

        let mut text_results = text_engine.search(query_text, fetch_k)?;

        if let Some(ref filter) = params.filter {
            text_results.retain(|r| {
                if let Some(rec) = self.records.get(&r.id) {
                    if let Some(ref meta) = rec.metadata {
                        return filter.matches(meta);
                    }
                }
                false
            });
        }

        let vector_scored_ids: Vec<(String, f32)> = vector_results
            .into_iter()
            .filter_map(|r| {
                self.records.get_id(r.slot).map(|id| (id.clone(), r.distance))
            })
            .collect();

        let text_scored_ids: Vec<(String, f32)> = text_results
            .into_iter()
            .map(|r| (r.id, r.score))
            .collect();

        let fused = weighted_reciprocal_rank_fusion_with_subscores(
            vector_scored_ids,
            text_scored_ids,
            k,
            params.rrf_k,
            params.alpha,
        );

        Ok(fused
            .into_iter()
            .map(|result| {
                let metadata = self.records.get(&result.id)
                    .and_then(|r| r.metadata.clone())
                    .unwrap_or_else(helpers::default_metadata);
                (result, metadata)
            })
            .collect())
    }

    /// Helper to map EngineSearchResult to SearchResult by resolving IDs and metadata.
    fn map_engine_results(&self, results: Vec<EngineSearchResult>) -> Vec<SearchResult> {
        results
            .into_iter()
            .filter_map(|r| {
                let (id, metadata) = self.records.get_result_fields_by_slot(r.slot)?;
                Some(SearchResult::new(
                    id,
                    r.distance,
                    metadata.unwrap_or_else(helpers::default_metadata),
                ))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
