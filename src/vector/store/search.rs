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
use anyhow::Result;

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
///
/// Fields reserved for future quantization-based rescoring.
#[allow(dead_code)]
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

/// Core K-NN search using segments.
///
/// Uses segments if available, falls back to brute force.
#[allow(dead_code)] // config parameter reserved for future quantization rescore
pub fn knn_search_core(
    records: &RecordStore,
    segments: Option<&SegmentManager>,
    query: &[f32],
    k: usize,
    ef: usize,
    _config: &SearchConfig,
) -> Result<Vec<(usize, f32)>> {
    let has_data = !records.is_empty() || segments.as_ref().is_some_and(|s| !s.is_empty());

    if !has_data {
        return Ok(Vec::new());
    }

    // Use segments if available
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

    // No segments - use brute force
    Ok(brute_force_search(records, query, k))
}

// ============================================================================
// Filtered Search
// ============================================================================

/// Core filtered search implementation.
///
/// Uses bitmap-based filtering when possible, falls back to JSON matching.
/// Uses segments (ACORN-1) when available, falls back to brute-force.
pub fn knn_search_filtered_core(
    records: &RecordStore,
    metadata_index: &MetadataIndex,
    segments: Option<&SegmentManager>,
    query: &[f32],
    k: usize,
    ef: usize,
    filter: &MetadataFilter,
) -> Result<Vec<SearchResult>> {
    // Try bitmap-based filtering (O(1) per candidate)
    let filter_bitmap = filter.evaluate_bitmap(metadata_index);

    // Use segments (ACORN-1 filtered search)
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
        }
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
