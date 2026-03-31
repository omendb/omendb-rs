//! HNSW search operations
//!
//! Implements k-NN search, filtered search (ACORN-1), and layer-level search.

use super::HNSWIndex;
use crate::distance::dot_product;
use crate::vector::hnsw::error::{HNSWError, Result};
use crate::vector::hnsw::node_storage::{NodeStorage, QueryPrep};
use crate::vector::hnsw::types::{Candidate, Distance, SearchResult};
use tracing::{debug, error, instrument};

/// Context for distance computation during search
///
/// Encapsulates all data needed for optimized distance computation (SQ8, L2 decomposition, etc.)
/// to avoid repeated branching and parameter passing in the hot loop.
///
/// When `sq8_prep` is `Some`, the SQ8 integer SIMD path is used.
/// When `sq8_prep` is `None`, full-precision f32 distance is computed.
/// Callers that need full precision simply pass `force_full_precision: true`
/// to `new()`, which prevents `sq8_prep` from being populated.
struct DistanceContext<'a> {
    query: &'a [f32],
    query_norm: f32,
    sq8_prep: Option<QueryPrep>,
    storage: &'a NodeStorage,
}

impl<'a> DistanceContext<'a> {
    /// Create a new distance context for the current search
    ///
    /// If `force_full_precision` is true, the SQ8 path is disabled regardless
    /// of storage mode. This is used during graph construction where quantization
    /// noise would hurt graph quality.
    fn new(query: &'a [f32], index: &'a HNSWIndex, force_full_precision: bool) -> Self {
        let query_norm = dot_product(query, query).sqrt();
        let sq8_prep = if force_full_precision {
            None
        } else {
            index.storage.prepare_query(query)
        };

        Self {
            query,
            query_norm,
            sq8_prep,
            storage: &index.storage,
        }
    }

    /// Compute distance to a node using the best available method
    #[inline(always)]
    fn compute<D: Distance>(&self, node_id: u32) -> Result<f32> {
        // SQ8 fast path
        if let Some(ref prep) = self.sq8_prep
            && let Some(dist) = self.storage.distance_sq8(prep, node_id)
        {
            return Ok(dist);
        }

        // Full precision path (also fallback for untrained SQ8)
        if self.storage.is_sq8() {
            let vec = self
                .storage
                .get_dequantized(node_id)
                .ok_or(HNSWError::VectorNotFound(node_id))?;
            Ok(D::distance_precomputed(self.query, &vec, self.query_norm))
        } else {
            let vec = self.storage.vector(node_id);
            Ok(D::distance_precomputed(self.query, vec, self.query_norm))
        }
    }

    /// Check if batch distance computation is available (SQ8 mode)
    #[inline(always)]
    fn has_batch(&self) -> bool {
        self.sq8_prep.is_some()
    }

    /// Batch compute distances to multiple nodes (SQ8 fast path)
    ///
    /// Returns the number of distances computed. Caller must provide output buffer
    /// large enough to hold distances for all IDs.
    #[inline]
    fn compute_batch(&self, ids: &[u32], distances: &mut [f32]) -> usize {
        if let Some(ref prep) = self.sq8_prep {
            return self.storage.distance_sq8_batch(prep, ids, distances);
        }
        0
    }
}

/// Trait for collecting neighbors during HNSW traversal
trait NeighborCollector {
    /// Collect unvisited neighbors into the output buffer
    fn collect(
        &self,
        node_id: u32,
        level: u8,
        visited: &crate::vector::hnsw::query_buffers::VisitedList,
        output: &mut Vec<u32>,
    );

    /// Get initial entry points (some collectors may expand them)
    fn prepare_entry_points(
        &self,
        entry_points: &[u32],
        level: u8,
        visited: &mut crate::vector::hnsw::query_buffers::VisitedList,
        output: &mut Vec<u32>,
    );
}

/// Standard HNSW neighbor collector using NodeStorage
struct StandardCollector<'a> {
    storage: &'a NodeStorage,
}

impl NeighborCollector for StandardCollector<'_> {
    #[inline(always)]
    fn collect(
        &self,
        node_id: u32,
        level: u8,
        visited: &crate::vector::hnsw::query_buffers::VisitedList,
        output: &mut Vec<u32>,
    ) {
        output.clear();
        if level == 0 {
            // Level 0: colocated neighbors
            for &id in self.storage.neighbors(node_id) {
                if !visited.contains(id) {
                    output.push(id);
                }
            }
        } else {
            // Upper levels: sparse storage (zero-copy via Cow)
            for &id in &*self.storage.neighbors_at_level_cow(node_id, level) {
                if !visited.contains(id) {
                    output.push(id);
                }
            }
        }
    }

    #[inline(always)]
    fn prepare_entry_points(
        &self,
        entry_points: &[u32],
        _level: u8,
        visited: &mut crate::vector::hnsw::query_buffers::VisitedList,
        output: &mut Vec<u32>,
    ) {
        output.clear();
        for &ep in entry_points {
            if !visited.contains(ep) {
                visited.insert(ep);
                output.push(ep);
            }
        }
    }
}

/// ACORN-1 filtered neighbor collector (arXiv:2403.04871)
///
/// Uses shared acorn module for 2-hop neighbor expansion.
struct AcornCollector<'a, F>
where
    F: Fn(u32) -> bool,
{
    storage: &'a NodeStorage,
    filter_fn: &'a F,
    m: usize,
}

impl<F> NeighborCollector for AcornCollector<'_, F>
where
    F: Fn(u32) -> bool,
{
    #[inline(always)]
    fn collect(
        &self,
        node_id: u32,
        level: u8,
        visited: &crate::vector::hnsw::query_buffers::VisitedList,
        output: &mut Vec<u32>,
    ) {
        crate::vector::hnsw::acorn::collect_matching_neighbors(
            self.storage,
            node_id,
            level,
            visited,
            self.filter_fn,
            self.m,
            output,
        );
    }

    #[inline(always)]
    fn prepare_entry_points(
        &self,
        entry_points: &[u32],
        level: u8,
        visited: &mut crate::vector::hnsw::query_buffers::VisitedList,
        output: &mut Vec<u32>,
    ) {
        output.clear();
        let mut matching = Vec::new();
        for &ep in entry_points {
            if visited.contains(ep) {
                continue;
            }
            visited.insert(ep);

            if (self.filter_fn)(ep) {
                output.push(ep);
            } else {
                // Expand entry point to find matching neighbors
                crate::vector::hnsw::acorn::collect_matching_neighbors(
                    self.storage,
                    ep,
                    level,
                    visited,
                    self.filter_fn,
                    self.m,
                    &mut matching,
                );
                output.extend(matching.iter().copied());
            }
        }
    }
}

/// Validation result for search parameters
enum SearchValidation {
    /// Validation passed, continue with search
    Continue,
    /// Index is empty, return empty results immediately
    Empty,
}

impl HNSWIndex {
    /// Unified search layer loop for both standard and filtered search.
    #[inline(always)]
    fn search_layer_internal<D, C>(
        &self,
        entry_points: &[u32],
        ctx: &DistanceContext,
        collector: &C,
        ef: usize,
        level: u8,
    ) -> Result<Vec<(u32, f32)>>
    where
        D: Distance,
        C: NeighborCollector,
    {
        use super::super::query_buffers;
        use std::cmp::Reverse;

        query_buffers::with_buffers(|buffers| {
            let visited = &mut buffers.visited;
            let candidates = &mut buffers.candidates;
            let working = &mut buffers.working;
            let unvisited = &mut buffers.unvisited;
            let results_buf = &mut buffers.results;
            let batch_distances = &mut buffers.batch_distances;

            // Prepare entry points
            collector.prepare_entry_points(entry_points, level, visited, unvisited);
            for &ep in unvisited.iter() {
                let dist = ctx.compute::<D>(ep)?;
                let candidate = Candidate::new(ep, dist);
                candidates.push(Reverse(candidate));
                working.push(candidate);
            }

            if candidates.is_empty() {
                return Ok(Vec::new());
            }

            // Check if batch distance computation is available (SQ8 mode)
            let use_batch = ctx.has_batch();

            // Greedy search
            while let Some(Reverse(current)) = candidates.pop() {
                if let Some(&farthest) = working.peek()
                    && current.distance > farthest.distance
                {
                    break;
                }

                // Collect neighbors using specialized collector
                collector.collect(current.node_id, level, visited, unvisited);

                let neighbors_slice = unvisited.as_slice();
                let num_neighbors = neighbors_slice.len();

                if num_neighbors == 0 {
                    continue;
                }

                if use_batch {
                    // Batch path: compute all distances at once (SQ8 mode)
                    // Ensure buffer is large enough
                    if batch_distances.len() < num_neighbors {
                        batch_distances.resize(num_neighbors, 0.0);
                    }

                    let computed = ctx.compute_batch(neighbors_slice, batch_distances);
                    debug_assert_eq!(computed, num_neighbors, "batch distance count mismatch");

                    // Process all computed distances, marking visited as we go
                    for (i, &neighbor_id) in neighbors_slice.iter().enumerate() {
                        // Guard against duplicates in neighbor list
                        if visited.contains(neighbor_id) {
                            continue;
                        }
                        visited.insert(neighbor_id);

                        let dist = batch_distances[i];
                        let neighbor = Candidate::new(neighbor_id, dist);

                        if let Some(&farthest) = working.peek() {
                            if neighbor.distance < farthest.distance || working.len() < ef {
                                candidates.push(Reverse(neighbor));
                                working.push(neighbor);
                                if working.len() > ef {
                                    working.pop();
                                }
                            }
                        } else {
                            candidates.push(Reverse(neighbor));
                            working.push(neighbor);
                        }
                    }
                } else {
                    // Per-neighbor path (full precision or single neighbor)
                    use crate::vector::hnsw::prefetch::PrefetchConfig;
                    const PREFETCH_ENABLED: bool = PrefetchConfig::enabled();
                    const PREFETCH_DISTANCE: usize = PrefetchConfig::stride();

                    if PREFETCH_ENABLED {
                        for &id in neighbors_slice.iter().take(PREFETCH_DISTANCE) {
                            self.storage.prefetch(id);
                            visited.prefetch(id);
                        }
                    }

                    for (i, &neighbor_id) in neighbors_slice.iter().enumerate() {
                        if PREFETCH_ENABLED && i + PREFETCH_DISTANCE < num_neighbors {
                            let prefetch_id = neighbors_slice[i + PREFETCH_DISTANCE];
                            self.storage.prefetch(prefetch_id);
                            visited.prefetch(prefetch_id);
                        }

                        // Guard against duplicates in neighbor list
                        if visited.contains(neighbor_id) {
                            continue;
                        }
                        visited.insert(neighbor_id);

                        let dist = ctx.compute::<D>(neighbor_id)?;
                        let neighbor = Candidate::new(neighbor_id, dist);

                        if let Some(&farthest) = working.peek() {
                            if neighbor.distance < farthest.distance || working.len() < ef {
                                candidates.push(Reverse(neighbor));
                                working.push(neighbor);
                                if working.len() > ef {
                                    working.pop();
                                }
                            }
                        } else {
                            candidates.push(Reverse(neighbor));
                            working.push(neighbor);
                        }
                    }
                }
            }

            // Return (node_id, distance) pairs sorted by distance
            results_buf.extend(working.drain());
            results_buf.sort_unstable_by_key(|c| c.distance);
            let output: Vec<(u32, f32)> = results_buf
                .iter()
                .map(|c| (c.node_id, c.distance.into_inner()))
                .collect();
            Ok(output)
        })
    }

    /// Validate search parameters (k, ef, query dimensions, finite values, cosine query norm)
    ///
    /// Returns `SearchValidation::Empty` if index is empty (caller should return empty results).
    /// Returns `SearchValidation::Continue` if validation passes and search should proceed.
    fn validate_search_params(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<SearchValidation> {
        if k == 0 {
            error!(k, ef, "Invalid search parameters: k must be > 0");
            return Err(HNSWError::InvalidSearchParams { k, ef });
        }

        if ef < k {
            error!(k, ef, "Invalid search parameters: ef must be >= k");
            return Err(HNSWError::InvalidSearchParams { k, ef });
        }

        if query.len() != self.dimensions() {
            error!(
                expected_dim = self.dimensions(),
                actual_dim = query.len(),
                "Dimension mismatch during search"
            );
            return Err(HNSWError::DimensionMismatch {
                expected: self.dimensions(),
                actual: query.len(),
            });
        }

        if query.iter().any(|x| !x.is_finite()) {
            error!("Invalid query vector: contains NaN or Inf values");
            return Err(HNSWError::InvalidVector);
        }

        if matches!(self.distance_fn, crate::vector::hnsw::types::Metric::Cosine)
            && dot_product(query, query) == 0.0
        {
            error!("Invalid cosine query vector: zero norm");
            return Err(HNSWError::InvalidParams(
                "Cosine query vector must have non-zero norm".to_string(),
            ));
        }

        // Check both empty storage AND no entry point (all nodes deleted)
        if self.is_empty() || self.entry_point.is_none() {
            debug!("Search on empty index, returning empty results");
            return Ok(SearchValidation::Empty);
        }

        Ok(SearchValidation::Continue)
    }

    /// Search for k nearest neighbors
    ///
    /// Returns up to k nearest neighbors sorted by distance (closest first).
    #[instrument(skip(self, query), fields(k, ef, dimensions = query.len(), index_size = self.len()))]
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        if matches!(
            self.validate_search_params(query, k, ef)?,
            SearchValidation::Empty
        ) {
            return Ok(Vec::new());
        }

        let entry_point = self.entry_point.ok_or(HNSWError::EmptyIndex)?;
        let entry_level = self.storage.level(entry_point);

        // Beam search at layer 0 (find ef nearest)
        let search_ef = ef.max(k);

        // Start from entry point, descend to layer 0.
        // Use a stack array to avoid heap allocation when entry_level == 0 (95% of nodes):
        // the greedy traversal loop is skipped entirely and we search directly from the
        // entry point without any allocation.
        let candidates = if entry_level == 0 {
            self.search_layer(query, std::slice::from_ref(&entry_point), search_ef, 0)?
        } else {
            let mut nearest = vec![entry_point];
            for level in (1..=entry_level).rev() {
                nearest = self
                    .search_layer(query, &nearest, 1, level)?
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect();
            }
            self.search_layer(query, &nearest, search_ef, 0)?
        };

        // Convert comparison distances to actual (e.g., sqrt for L2)
        // Already sorted: comparison_to_actual() is monotonic for all distance functions
        let results: Vec<SearchResult> = candidates
            .iter()
            .take(k)
            .map(|&(id, dist)| {
                let slot = self.storage.slot(id);
                SearchResult::new(slot, self.distance_fn.comparison_to_actual(dist))
            })
            .collect();

        debug!(
            num_results = results.len(),
            closest_distance = results.first().map(|r| r.distance),
            "Search completed successfully"
        );

        Ok(results)
    }

    /// Search for k nearest neighbors with metadata filtering (ACORN-1)
    ///
    /// Implements ACORN-1 filtered search algorithm (arXiv:2403.04871).
    /// Key insight: traverse THROUGH non-matching nodes to find matching ones,
    /// using 2-hop expansion when selectivity is low (<10%).
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of nearest neighbors to return
    /// * `ef` - Size of dynamic candidate list (must be >= k)
    /// * `filter_fn` - Filter predicate: returns true if node should be considered
    ///
    /// # Returns
    /// Up to k nearest neighbors that match the filter, sorted by distance
    ///
    /// # Performance
    /// - Low selectivity (5-20% match): 3-6x faster than post-filtering
    /// - High selectivity (>60% match): Falls back to standard search + post-filter
    /// - Recall: 93-98% (slightly lower than standard search due to graph sparsity)
    ///
    /// # Reference
    /// ACORN: SIGMOD 2024, arXiv:2403.04871
    #[instrument(skip(self, query, filter_fn), fields(k, ef, dimensions = query.len(), index_size = self.len()))]
    pub(crate) fn search_with_filter<F>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_fn: F,
    ) -> Result<Vec<SearchResult>>
    where
        F: Fn(u32) -> bool,
    {
        if matches!(
            self.validate_search_params(query, k, ef)?,
            SearchValidation::Empty
        ) {
            return Ok(Vec::new());
        }

        // Wrap filter to convert internal node ID to slot
        // After optimize(), id may differ from slot - filter expects slot
        let slot_filter = |id: u32| filter_fn(self.storage.slot(id));

        // Estimate filter selectivity
        let selectivity = self.estimate_selectivity(&slot_filter);

        // Adaptive threshold: bypass ACORN-1 if filter is too permissive
        // Or for small/medium graphs where brute force is fast enough
        // ACORN-1 becomes effective at larger scales (1000+ vectors)
        const SELECTIVITY_THRESHOLD: f32 = 0.4;
        const SMALL_GRAPH_SIZE: usize = 1000;

        if selectivity > SELECTIVITY_THRESHOLD || self.len() <= SMALL_GRAPH_SIZE {
            // Filter is broad (>40% match) or graph is small: use standard search + post-filter
            debug!(selectivity, "Using post-filter path");

            // For very selective filters, we may need to search the entire graph
            // to find all matching items
            let oversample_factor = 1.0 / selectivity.max(0.01);
            let mut oversample_k = ((k as f32 * oversample_factor).ceil() as usize)
                .max(k * 10) // At least 10x k
                .min(self.len());

            // Ensure ef >= oversample_k (required by HNSW)
            let mut search_ef = ef.max(oversample_k).max(self.len().min(500));

            let mut all_results = self.search(query, oversample_k, search_ef)?;
            all_results.retain(|r| filter_fn(r.id));

            // If we didn't find enough, progressively expand search
            // This handles the case where matching items aren't in the nearest neighbors
            while all_results.len() < k && oversample_k < self.len() {
                debug!(found = all_results.len(), wanted = k, "Expanding search");
                oversample_k = (oversample_k * 2).min(self.len());
                search_ef = oversample_k;
                all_results = self.search(query, oversample_k, search_ef)?;
                all_results.retain(|r| filter_fn(r.id));
            }

            all_results.truncate(k);

            debug!(num_results = all_results.len(), "Post-filter complete");

            return Ok(all_results);
        }

        // Filter is selective (<40% match): use ACORN-1
        debug!(selectivity, "Using ACORN-1 filtered search");

        let entry_point = self.entry_point.ok_or(HNSWError::EmptyIndex)?;
        let entry_level = self.storage.level(entry_point);

        // Beam search at layer 0 (find ef nearest that match filter)
        let search_ef = ef.max(k);

        // Start from entry point, descend to layer 0.
        // When entry_level == 0 (95% of nodes), skip upper traversal entirely — no Vec allocation.
        let candidates = if entry_level == 0 {
            self.search_layer_with_filter(
                query,
                std::slice::from_ref(&entry_point),
                search_ef,
                0,
                &slot_filter,
            )?
        } else {
            let mut nearest = vec![entry_point];
            for level in (1..=entry_level).rev() {
                let pairs =
                    self.search_layer_with_filter(query, &nearest, 1, level, &slot_filter)?;
                if pairs.is_empty() {
                    debug!(level, "No matches at this level, falling back");
                    // Keep nearest as-is (still points to entry_point or last good result)
                } else {
                    nearest = pairs.into_iter().map(|(id, _)| id).collect();
                }
            }
            self.search_layer_with_filter(query, &nearest, search_ef, 0, &slot_filter)?
        };

        // Already sorted: comparison_to_actual() is monotonic
        let results: Vec<SearchResult> = candidates
            .iter()
            .take(k)
            .map(|&(id, dist)| {
                let slot = self.storage.slot(id);
                SearchResult::new(slot, self.distance_fn.comparison_to_actual(dist))
            })
            .collect();

        debug!(
            num_results = results.len(),
            closest_distance = results.first().map(|r| r.distance),
            "ACORN-1 search completed"
        );

        // Fallback: if ACORN-1 found fewer than k results, try brute-force post-filter
        // This can happen when the graph structure doesn't connect to matching nodes
        // (especially for rare filters where matching nodes are sparse)
        if results.len() < k {
            debug!(
                found = results.len(),
                wanted = k,
                "ACORN-1 insufficient, falling back to post-filter"
            );

            // Full post-filter search as last resort
            // Use large oversample to find all matching items
            let oversample_k = self.len(); // Search all nodes
            let search_ef = self.len(); // Maximum ef

            let mut all_results = self.search(query, oversample_k, search_ef)?;
            all_results.retain(|r| filter_fn(r.id));
            all_results.truncate(k);

            debug!(
                num_results = all_results.len(),
                "Post-filter fallback complete"
            );

            return Ok(all_results);
        }

        Ok(results)
    }

    /// Estimate filter selectivity by sampling nodes
    ///
    /// Samples up to 100 random nodes to estimate what fraction matches the filter.
    /// Returns value in [0.0, 1.0] where 1.0 means all nodes match.
    pub(super) fn estimate_selectivity<F>(&self, filter_fn: &F) -> f32
    where
        F: Fn(u32) -> bool,
    {
        const SAMPLE_SIZE: usize = 100;

        if self.is_empty() {
            return 1.0;
        }

        let sample_size = SAMPLE_SIZE.min(self.len());
        let step = (self.len() / sample_size).max(1);

        let mut matches = 0;
        for i in 0..sample_size {
            let node_id = (i * step) as u32;
            if filter_fn(node_id) {
                matches += 1;
            }
        }

        matches as f32 / sample_size as f32
    }

    /// Search for nearest neighbors at a specific level with metadata filtering (ACORN-1)
    ///
    /// Key differences from standard `search_layer`:
    /// 1. Only calculates distance for nodes matching the filter
    /// 2. Uses 2-hop exploration when filter is very selective (<10% match rate)
    /// 3. Expands search more aggressively to compensate for graph sparsity
    ///
    /// Optimized (Nov 25, 2025):
    /// - Uses `VisitedList` with O(1) clear (generation-based, like hnswlib)
    /// - Reuses pre-allocated unvisited buffer to avoid per-iteration allocation
    /// - Monomorphized distance dispatch (Dec 12, 2025)
    pub(super) fn search_layer_with_filter<F>(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
        filter_fn: &F,
    ) -> Result<Vec<(u32, f32)>>
    where
        F: Fn(u32) -> bool,
    {
        dispatch_distance!(self.distance_fn, D => {
            self.search_layer_with_filter_mono::<D, F>(
                query,
                entry_points,
                ef,
                level,
                filter_fn,
            )
        })
    }

    /// Monomorphized filtered search layer (static dispatch, no match in hot loop)
    ///
    /// Implements ACORN-1 algorithm from arXiv:2403.04871 with Weaviate optimization:
    /// - 2-hop expansion when neighbor doesn't match filter (adaptive, per-neighbor)
    /// - Truncation to M to bound neighbor list size
    /// - Distance computation only for matching nodes
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    pub(super) fn search_layer_with_filter_mono<D: Distance, F>(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
        filter_fn: &F,
    ) -> Result<Vec<(u32, f32)>>
    where
        F: Fn(u32) -> bool,
    {
        let ctx = DistanceContext::new(query, self, false);
        let collector = AcornCollector {
            storage: &self.storage,
            filter_fn,
            m: self.params.m,
        };

        self.search_layer_internal::<D, _>(entry_points, &ctx, &collector, ef, level)
    }

    /// Search for nearest neighbors at a specific level
    ///
    /// Search layer returning (node_id, comparison_distance) pairs.
    ///
    /// Returns up to `ef` nearest neighbors sorted by distance (closest first).
    /// Distances are comparison distances (L2 squared, raw cosine/dot) — callers
    /// use `distance_fn.comparison_to_actual()` to convert for user-facing results.
    ///
    /// Uses SQ8 quantized distances when available, full precision otherwise.
    pub(super) fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
    ) -> Result<Vec<(u32, f32)>> {
        dispatch_distance!(self.distance_fn, D => {
            self.search_layer_mono::<D>(query, entry_points, ef, level, false)
        })
    }

    /// Search layer using full precision (f32) distances.
    ///
    /// Used during graph construction where quantization noise hurts graph quality.
    pub(super) fn search_layer_full_precision(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
    ) -> Result<Vec<(u32, f32)>> {
        dispatch_distance!(self.distance_fn, D => {
            self.search_layer_mono::<D>(query, entry_points, ef, level, true)
        })
    }

    /// Search layer using the configured construction-distance mode.
    ///
    /// Full precision remains the default because it yields the best graph quality.
    /// SQ8 indexes can opt into quantized construction for faster builds once the
    /// quantizer has been trained.
    pub(super) fn search_layer_for_construction(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
    ) -> Result<Vec<(u32, f32)>> {
        if self.storage.is_sq8() && self.params.use_quantized_construction {
            self.search_layer(query, entry_points, ef, level)
        } else {
            self.search_layer_full_precision(query, entry_points, ef, level)
        }
    }

    #[inline(never)]
    fn search_layer_mono<D: Distance>(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
        full_precision: bool,
    ) -> Result<Vec<(u32, f32)>> {
        let ctx = DistanceContext::new(query, self, full_precision);
        let collector = StandardCollector {
            storage: &self.storage,
        };

        self.search_layer_internal::<D, _>(entry_points, &ctx, &collector, ef, level)
    }
}
