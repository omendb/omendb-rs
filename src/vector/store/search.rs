//! Search implementation functions for VectorStore.
//!
//! These functions take explicit dependencies and can be tested in isolation.
//! VectorStore methods become thin wrappers calling these implementations.

use super::helpers;
use super::record_store::RecordStore;
use super::{MetadataFilter, SearchResult};
use crate::distance::{cosine_distance_precomputed, dot_product, l2_distance, norm_squared};
use crate::omen::Metric;
use crate::vector::VectorEngineView;
use crate::vector::hnsw::PublishedSegmentView;
use crate::vector::metadata::MetadataIndex;
use anyhow::Result;
use std::sync::Arc;

/// Compute distance with precomputed query norm.
///
/// For cosine metric, the query norm is constant across all candidates in a search,
/// so precomputing it avoids redundant `dot_product(query, query)` per candidate.
/// For L2 and InnerProduct, the query_norm parameter is ignored.
#[inline]
fn compute_distance(metric: Metric, query: &[f32], candidate: &[f32], query_norm: f32) -> f32 {
    match metric {
        Metric::L2 => l2_distance(query, candidate),
        Metric::Cosine => cosine_distance_precomputed(query, candidate, query_norm),
        Metric::InnerProduct => -dot_product(query, candidate),
    }
}

/// Brute-force K-NN search implementation.
///
/// Scans all live records and returns k nearest neighbors.
/// Uses precomputed query norm for cosine metric and partial sort for large result sets.
pub(crate) fn brute_force_search(
    records: &RecordStore,
    query: &[f32],
    k: usize,
    metric: Metric,
) -> Vec<(usize, f32)> {
    if records.is_empty() {
        return Vec::new();
    }

    let query_norm = norm_squared(query).sqrt();

    let mut distances: Vec<(usize, f32)> = records
        .iter_live()
        .filter_map(|(slot, record)| {
            let vector = record.vector?;
            let dist = compute_distance(metric, query, vector.as_slice(), query_norm);
            Some((slot as usize, dist))
        })
        .collect();

    // Use partial sort when result set is much larger than k
    if distances.len() > k * 4 {
        distances.select_nth_unstable_by(k, |a, b| a.1.total_cmp(&b.1));
        distances.truncate(k);
        distances.sort_by(|a, b| a.1.total_cmp(&b.1));
    } else {
        distances.sort_by(|a, b| a.1.total_cmp(&b.1));
        distances.truncate(k);
    }

    distances
}

/// Convert slot-distance pairs to SearchResult with metadata.
pub(crate) fn slots_to_search_results(
    records: &RecordStore,
    results: Vec<(usize, f32)>,
) -> Vec<SearchResult> {
    results
        .into_iter()
        .filter_map(|(slot, distance)| {
            let (id, metadata) = records.get_result_fields_by_slot(slot as u32)?;
            Some(SearchResult::new(
                id,
                distance,
                metadata.unwrap_or_else(helpers::default_metadata),
            ))
        })
        .collect()
}

/// Convert slot-distance pairs to SearchResult, falling back to brute force if empty.
pub(crate) fn slots_to_results_with_fallback(
    records: &RecordStore,
    results: Vec<(usize, f32)>,
    query: &[f32],
    k: usize,
    metric: Metric,
) -> Vec<SearchResult> {
    let filtered = slots_to_search_results(records, results);

    // Fall back to brute force if HNSW results were all deleted
    if filtered.is_empty() && !records.is_empty() {
        let brute_results = brute_force_search(records, query, k, metric);
        slots_to_search_results(records, brute_results)
    } else {
        filtered
    }
}

/// Core K-NN search using search engine view.
///
/// Uses engine view if available, falls back to brute force.
pub(crate) fn knn_search_core(
    records: &RecordStore,
    engine: Option<Arc<PublishedSegmentView>>,
    query: &[f32],
    k: usize,
    ef: usize,
    metric: Metric,
) -> Result<Vec<(usize, f32)>> {
    let has_data = !records.is_empty()
        || engine.as_ref().is_some_and(
            |e: &std::sync::Arc<crate::vector::hnsw::PublishedSegmentView>| e.index.len() > 0,
        );

    if !has_data {
        return Ok(Vec::new());
    }

    // Use engine if available
    if let Some(engine) = engine {
        let engine_results = engine
            .search(query, k, ef)
            .map_err(|e| anyhow::anyhow!("Engine search failed: {e}"))?;

        let results: Vec<(usize, f32)> = engine_results
            .into_iter()
            .map(|r| (r.slot as usize, r.distance))
            .collect();

        // Fall back to brute force if engine returns nothing but we have data
        if results.is_empty() && !records.is_empty() {
            return Ok(brute_force_search(records, query, k, metric));
        }
        return Ok(results);
    }

    // No engine - use brute force
    Ok(brute_force_search(records, query, k, metric))
}

/// Core filtered search implementation.
///
/// Uses bitmap-based filtering when possible, falls back to JSON matching.
/// Uses segments (ACORN-1) when available, falls back to brute-force.
#[allow(clippy::too_many_arguments)]
pub(crate) fn knn_search_filtered_core(
    records: &RecordStore,
    metadata_index: &MetadataIndex,
    engine: Option<Arc<PublishedSegmentView>>,
    query: &[f32],
    k: usize,
    ef: usize,
    filter: &MetadataFilter,
    metric: Metric,
) -> Result<Vec<SearchResult>> {
    // Try bitmap-based filtering (O(1) per candidate)
    let filter_bitmap = filter.evaluate_bitmap(metadata_index);

    // Use engine (ACORN-1 filtered search)
    if let Some(engine) = engine
        && !engine.is_empty()
    {
        let engine_results = if let Some(ref bitmap) = filter_bitmap {
            // Fast path: bitmap-based filtering
            let filter_fn = |slot: u32| -> bool { records.is_live(slot) && bitmap.contains(slot) };
            PublishedSegmentView::search_with_filter(engine.as_ref(), query, k, ef, &filter_fn)?
        } else {
            // Slow path: JSON-based filtering (borrow metadata, no clone)
            let filter_fn = |slot: u32| -> bool {
                if !records.is_live(slot) {
                    return false;
                }
                let metadata = records
                    .get_by_slot(slot)
                    .and_then(|r| r.metadata)
                    .unwrap_or_else(|| helpers::DEFAULT_METADATA.clone());
                filter.matches(&metadata)
            };
            PublishedSegmentView::search_with_filter(engine.as_ref(), query, k, ef, &filter_fn)?
        };

        // Convert engine results to search results (metadata resolved only for final k)
        let slot_distances: Vec<(usize, f32)> = engine_results
            .into_iter()
            .map(|r| (r.slot as usize, r.distance))
            .collect();
        let results = slots_to_search_results(records, slot_distances);

        if !results.is_empty() {
            return Ok(results);
        }
    }

    // Fallback: brute-force search with filtering
    Ok(brute_force_filtered(
        records,
        query,
        k,
        filter,
        filter_bitmap.as_ref(),
        metric,
    ))
}

/// Brute-force search with metadata filtering.
///
/// Collects (slot, distance) pairs first, truncates to k, then resolves metadata
/// only for the final results. This avoids cloning metadata for candidates that
/// get sorted away.
fn brute_force_filtered(
    records: &RecordStore,
    query: &[f32],
    k: usize,
    filter: &MetadataFilter,
    filter_bitmap: Option<&roaring::RoaringBitmap>,
    metric: Metric,
) -> Vec<SearchResult> {
    let query_norm = norm_squared(query).sqrt();

    // Phase 1: Collect (slot, distance) for passing candidates — no metadata cloning
    let mut candidates: Vec<(u32, f32)> = records
        .iter_live()
        .filter_map(|(slot, record)| {
            let vector = record.vector?;
            let passes_filter = if let Some(bitmap) = filter_bitmap {
                bitmap.contains(slot)
            } else {
                let metadata = record
                    .metadata
                    .as_ref()
                    .unwrap_or(&helpers::DEFAULT_METADATA);
                filter.matches(metadata)
            };

            if !passes_filter {
                return None;
            }

            let distance = compute_distance(metric, query, vector.as_slice(), query_norm);
            Some((slot, distance))
        })
        .collect();

    // Phase 2: Partial sort + truncate to k
    if candidates.len() > k * 4 {
        candidates.select_nth_unstable_by(k, |a, b| a.1.total_cmp(&b.1));
        candidates.truncate(k);
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
    } else {
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        candidates.truncate(k);
    }

    // Phase 3: Resolve metadata only for final k results
    slots_to_search_results(
        records,
        candidates
            .into_iter()
            .map(|(s, d)| (s as usize, d))
            .collect(),
    )
}

/// Rescore quantized search candidates with original fp32 vectors.
///
/// Takes candidates from SQ8 search, recomputes distances using the original
/// fp32 vectors stored in RecordStore, sorts by true distance, and returns top-k.
pub(crate) fn rescore_results(
    records: &RecordStore,
    candidates: Vec<(usize, f32)>,
    query: &[f32],
    k: usize,
    metric: Metric,
) -> Vec<(usize, f32)> {
    let query_norm = norm_squared(query).sqrt();

    let mut rescored: Vec<(usize, f32)> = candidates
        .into_iter()
        .filter_map(|(slot, _sq8_dist)| {
            let record = records.get_by_slot(slot as u32)?;
            let vector = record.vector?;
            let dist = compute_distance(metric, query, vector.as_slice(), query_norm);
            Some((slot, dist))
        })
        .collect();

    rescored.sort_by(|a, b| a.1.total_cmp(&b.1));
    rescored.truncate(k);
    rescored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brute_force_empty() {
        let records = RecordStore::new(3);
        let query = vec![1.0, 2.0, 3.0];
        let results = brute_force_search(&records, &query, 10, Metric::L2);
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
    fn test_brute_force_cosine() {
        let records = RecordStore::new(3);
        records
            .set("a".to_string(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        records
            .set("b".to_string(), vec![0.0, 1.0, 0.0], None)
            .unwrap();
        let query = vec![1.0, 0.0, 0.0];
        let results = brute_force_search(&records, &query, 2, Metric::Cosine);
        assert_eq!(results.len(), 2);
        // First result should be "a" (cosine distance ~0)
        assert!(results[0].1 < 0.01);
        // Second result should be "b" (cosine distance ~1)
        assert!(results[1].1 > 0.99);
    }

    #[test]
    fn test_brute_force_inner_product() {
        let records = RecordStore::new(3);
        records
            .set("a".to_string(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        records
            .set("b".to_string(), vec![0.5, 0.0, 0.0], None)
            .unwrap();
        let query = vec![1.0, 0.0, 0.0];
        let results = brute_force_search(&records, &query, 2, Metric::InnerProduct);
        assert_eq!(results.len(), 2);
        // InnerProduct uses -dot_product, so higher dot = lower (more negative) distance
        // "a" has dot=1.0 -> distance=-1.0, "b" has dot=0.5 -> distance=-0.5
        assert!(results[0].1 < results[1].1); // "a" first (lower distance)
    }

    #[test]
    fn test_rescore_results_basic() {
        let records = RecordStore::new(3);
        records
            .set("a".to_string(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        records
            .set("b".to_string(), vec![0.0, 1.0, 0.0], None)
            .unwrap();
        records
            .set("c".to_string(), vec![0.5, 0.5, 0.0], None)
            .unwrap();

        // Simulate SQ8 candidates with wrong ordering
        let candidates = vec![(2, 0.1), (0, 0.2), (1, 0.3)];
        let query = vec![1.0, 0.0, 0.0];

        let rescored = rescore_results(&records, candidates, &query, 2, Metric::L2);

        assert_eq!(rescored.len(), 2);
        // "a" at slot 0 should be nearest to [1,0,0]
        assert_eq!(rescored[0].0, 0);
    }

    #[test]
    fn test_rescore_empty() {
        let records = RecordStore::new(3);
        let candidates: Vec<(usize, f32)> = vec![];
        let query = vec![1.0, 0.0, 0.0];
        let rescored = rescore_results(&records, candidates, &query, 5, Metric::L2);
        assert!(rescored.is_empty());
    }
}
