//! Search implementation functions for VectorStore.
//!
//! These functions take explicit dependencies and can be tested in isolation.
//! VectorStore methods become thin wrappers calling these implementations.

use super::helpers;
use super::record_store::RecordStore;
use super::{MetadataFilter, SearchResult};
use crate::distance::l2_distance;
use crate::omen::MetadataIndex;
use crate::vector::hnsw::SegmentManager;
use crate::vector::hnsw_index::HNSWIndex;
use anyhow::Result;
use rayon::prelude::*;

// ============================================================================
// Brute Force Search
// ============================================================================

/// Brute-force K-NN search implementation.
///
/// Scans all live records and returns k nearest neighbors.
/// Used as fallback when HNSW index is empty or returns no results.
pub fn brute_force_search(records: &RecordStore, query: &[f32], k: usize) -> Vec<(usize, f32)> {
    if records.is_empty() {
        return Vec::new();
    }

    let mut distances: Vec<(usize, f32)> = records
        .iter_live()
        .map(|(slot, record)| {
            let dist = l2_distance(query, &record.vector);
            (slot as usize, dist)
        })
        .collect();

    distances.sort_by(|a, b| a.1.total_cmp(&b.1));
    distances.into_iter().take(k).collect()
}

// ============================================================================
// Rescore
// ============================================================================

/// Rescore HNSW candidates using original vectors from RecordStore.
///
/// Takes quantized HNSW candidates and computes exact L2 distances
/// from the original vectors stored in RecordStore.
pub fn rescore_candidates(
    records: &RecordStore,
    candidates: &[(usize, f32)],
    query: &[f32],
    k: usize,
) -> Vec<(usize, f32)> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut rescored: Vec<(usize, f32)> = candidates
        .iter()
        .filter_map(|&(id, _quantized_dist)| {
            records
                .get_vector(id as u32)
                .map(|v| (id, l2_distance(query, v)))
        })
        .collect();

    rescored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    rescored.truncate(k);
    rescored
}

// ============================================================================
// Result Conversion
// ============================================================================

/// Convert slot-distance pairs to SearchResult with metadata.
pub fn slots_to_search_results(
    records: &RecordStore,
    results: Vec<(usize, f32)>,
) -> Vec<SearchResult> {
    results
        .into_iter()
        .filter_map(|(slot, distance)| {
            let record = records.get_by_slot(slot as u32)?;
            let metadata = record
                .metadata
                .clone()
                .unwrap_or_else(helpers::default_metadata);
            Some(SearchResult::new(record.id.clone(), distance, metadata))
        })
        .collect()
}

/// Convert slot-distance pairs to SearchResult, falling back to brute force if empty.
pub fn slots_to_results_with_fallback(
    records: &RecordStore,
    results: Vec<(usize, f32)>,
    query: &[f32],
    k: usize,
) -> Vec<SearchResult> {
    let filtered = slots_to_search_results(records, results);

    // Fall back to brute force if HNSW results were all deleted
    if filtered.is_empty() && !records.is_empty() {
        let brute_results = brute_force_search(records, query, k);
        slots_to_search_results(records, brute_results)
    } else {
        filtered
    }
}

// ============================================================================
// Core Search Implementation
// ============================================================================

/// Configuration for search operations.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Whether to rescore quantized results
    pub rescore_enabled: bool,
    /// Oversample factor for rescore (fetch more candidates than k)
    pub oversample_factor: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            rescore_enabled: true,
            oversample_factor: 3.0,
        }
    }
}

/// Core K-NN search using HNSW or segments.
///
/// Tries segments first (if available), then HNSW, then falls back to brute force.
/// Handles rescore for quantized indices.
#[allow(clippy::too_many_arguments)]
pub fn knn_search_core(
    records: &RecordStore,
    segments: Option<&SegmentManager>,
    hnsw_index: Option<&HNSWIndex>,
    query: &[f32],
    k: usize,
    ef: usize,
    config: &SearchConfig,
) -> Result<Vec<(usize, f32)>> {
    let has_data = !records.is_empty()
        || segments.as_ref().is_some_and(|s| !s.is_empty())
        || hnsw_index.as_ref().is_some_and(|idx| !idx.is_empty());

    if !has_data {
        return Ok(Vec::new());
    }

    // Use segments if available (preferred path)
    if let Some(segments) = segments {
        let segment_results = segments
            .search(query, k, ef)
            .map_err(|e| anyhow::anyhow!("Segment search failed: {e}"))?;

        let results: Vec<(usize, f32)> = segment_results
            .into_iter()
            .map(|r| (r.slot as usize, r.distance))
            .collect();

        // Fall back to brute force if segments return nothing but we have data
        if results.is_empty() && !records.is_empty() {
            return Ok(brute_force_search(records, query, k));
        }
        return Ok(results);
    }

    // Legacy path: use hnsw_index directly
    if let Some(index) = hnsw_index {
        let results = if index.is_asymmetric() {
            let can_rescore = !records.is_empty();
            if config.rescore_enabled && can_rescore {
                knn_search_with_rescore(records, index, query, k, ef, config.oversample_factor)?
            } else {
                index.search_ef(query, k, ef)?
            }
        } else {
            index.search_ef(query, k, ef)?
        };

        // Fall back to brute force if HNSW returns nothing but we have data
        if results.is_empty() && !records.is_empty() {
            return Ok(brute_force_search(records, query, k));
        }
        return Ok(results);
    }

    Ok(brute_force_search(records, query, k))
}

/// K-NN search with rescore using original vectors.
fn knn_search_with_rescore(
    records: &RecordStore,
    index: &HNSWIndex,
    query: &[f32],
    k: usize,
    ef: usize,
    oversample_factor: f32,
) -> Result<Vec<(usize, f32)>> {
    let oversample_k = ((k as f32) * oversample_factor).ceil() as usize;
    let candidates = index.search_ef(query, oversample_k, ef)?;

    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    Ok(rescore_candidates(records, &candidates, query, k))
}

// ============================================================================
// Filtered Search
// ============================================================================

/// Core filtered search implementation.
///
/// Uses bitmap-based filtering when possible, falls back to JSON matching.
/// Prefers segments (ACORN-1) when available, falls back to hnsw_index or brute-force.
#[allow(clippy::too_many_arguments)]
pub fn knn_search_filtered_core(
    records: &RecordStore,
    metadata_index: &MetadataIndex,
    segments: Option<&SegmentManager>,
    hnsw_index: Option<&HNSWIndex>,
    query: &[f32],
    k: usize,
    ef: usize,
    filter: &MetadataFilter,
) -> Result<Vec<SearchResult>> {
    // Try bitmap-based filtering (O(1) per candidate)
    let filter_bitmap = filter.evaluate_bitmap(metadata_index);

    // Try segments first (ACORN-1 filtered search)
    if let Some(seg_mgr) = segments {
        if !seg_mgr.is_empty() {
            let segment_results = if let Some(ref bitmap) = filter_bitmap {
                // Fast path: bitmap-based filtering
                let filter_fn =
                    |slot: u32| -> bool { records.is_live(slot) && bitmap.contains(slot) };
                seg_mgr.search_with_filter(query, k, ef, filter_fn)?
            } else {
                // Slow path: JSON-based filtering
                let filter_fn = |slot: u32| -> bool {
                    if !records.is_live(slot) {
                        return false;
                    }
                    let metadata = records
                        .get_by_slot(slot)
                        .and_then(|r| r.metadata.clone())
                        .unwrap_or_else(helpers::default_metadata);
                    filter.matches(&metadata)
                };
                seg_mgr.search_with_filter(query, k, ef, filter_fn)?
            };

            // Convert segment results to search results
            let results: Vec<SearchResult> = segment_results
                .into_iter()
                .filter_map(|r| {
                    records.get_by_slot(r.slot).map(|record| SearchResult {
                        id: record.id.clone(),
                        distance: r.distance,
                        metadata: record
                            .metadata
                            .clone()
                            .unwrap_or_else(helpers::default_metadata),
                    })
                })
                .collect();

            if !results.is_empty() {
                return Ok(results);
            }
            // Fall through to legacy path if no results
        }
    }

    // Legacy path: use hnsw_index
    if let Some(hnsw) = hnsw_index {
        let search_results = if let Some(ref bitmap) = filter_bitmap {
            // Fast path: bitmap-based filtering
            let filter_fn =
                |node_id: u32| -> bool { records.is_live(node_id) && bitmap.contains(node_id) };
            hnsw.search_with_filter_ef(query, k, Some(ef), filter_fn)?
        } else {
            // Slow path: JSON-based filtering
            let filter_fn = |node_id: u32| -> bool {
                if !records.is_live(node_id) {
                    return false;
                }
                let metadata = records
                    .get_by_slot(node_id)
                    .and_then(|r| r.metadata.clone())
                    .unwrap_or_else(helpers::default_metadata);
                filter.matches(&metadata)
            };
            hnsw.search_with_filter_ef(query, k, Some(ef), filter_fn)?
        };

        let filtered_results = slots_to_search_results(records, search_results);
        return Ok(filtered_results);
    }

    // Fallback: brute-force search with filtering
    Ok(brute_force_filtered(
        records,
        query,
        k,
        filter,
        filter_bitmap.as_ref(),
    ))
}

/// Brute-force search with metadata filtering.
fn brute_force_filtered(
    records: &RecordStore,
    query: &[f32],
    k: usize,
    filter: &MetadataFilter,
    filter_bitmap: Option<&roaring::RoaringBitmap>,
) -> Vec<SearchResult> {
    let mut all_results: Vec<SearchResult> = records
        .iter_live()
        .filter_map(|(slot, record)| {
            // Use bitmap if available, otherwise JSON
            let passes_filter = if let Some(bitmap) = filter_bitmap {
                bitmap.contains(slot)
            } else {
                let metadata = record
                    .metadata
                    .clone()
                    .unwrap_or_else(helpers::default_metadata);
                filter.matches(&metadata)
            };

            if !passes_filter {
                return None;
            }

            let metadata = record
                .metadata
                .clone()
                .unwrap_or_else(helpers::default_metadata);
            let distance = l2_distance(query, &record.vector);
            Some(SearchResult::new(record.id.clone(), distance, metadata))
        })
        .collect();

    all_results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    all_results.truncate(k);

    all_results
}

// ============================================================================
// Batch Search
// ============================================================================

/// Parallel batch search for multiple queries.
///
/// Uses rayon for parallel execution across queries.
#[allow(dead_code)]
pub fn search_batch_parallel<F>(
    queries: &[&[f32]],
    k: usize,
    ef: usize,
    search_fn: F,
) -> Vec<Result<Vec<(usize, f32)>>>
where
    F: Fn(&[f32], usize, usize) -> Result<Vec<(usize, f32)>> + Sync,
{
    queries.par_iter().map(|q| search_fn(q, k, ef)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brute_force_empty() {
        let records = RecordStore::new(3);
        let query = vec![1.0, 2.0, 3.0];
        let results = brute_force_search(&records, &query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_rescore_empty_candidates() {
        let records = RecordStore::new(3);
        let candidates: Vec<(usize, f32)> = vec![];
        let query = vec![1.0, 2.0, 3.0];
        let results = rescore_candidates(&records, &candidates, &query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_slots_to_results_empty() {
        let records = RecordStore::new(3);
        let slots: Vec<(usize, f32)> = vec![];
        let results = slots_to_search_results(&records, slots);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_config_default() {
        let config = SearchConfig::default();
        assert!(config.rescore_enabled);
        assert!((config.oversample_factor - 3.0).abs() < f32::EPSILON);
    }
}
