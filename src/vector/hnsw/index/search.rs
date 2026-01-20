//! HNSW search operations
//!
//! Implements k-NN search, filtered search (ACORN-1), and layer-level search.

use super::HNSWIndex;
use crate::vector::hnsw::error::{HNSWError, Result};
use crate::vector::hnsw::node_storage::{NodeStorage, QueryPrep};
use crate::vector::hnsw::types::{Candidate, Distance, SearchResult};
use ordered_float::OrderedFloat;
use tracing::{debug, error, instrument};

/// Context for distance computation during search
///
/// Encapsulates all data needed for optimized distance computation (SQ8, L2 decomposition, etc.)
/// to avoid repeated branching and parameter passing in the hot loop.
struct DistanceContext<'a> {
    query: &'a [f32],
    sq8_prep: Option<QueryPrep>,
    force_full_precision: bool,
    storage: &'a NodeStorage,
}

impl<'a> DistanceContext<'a> {
    /// Create a new distance context for the current search
    fn new(query: &'a [f32], index: &'a HNSWIndex, force_full_precision: bool) -> Self {
        let sq8_prep = if force_full_precision {
            None
        } else {
            index.storage.prepare_query(query)
        };

        Self {
            query,
            sq8_prep,
            force_full_precision,
            storage: &index.storage,
        }
    }

    /// Compute distance to a node using the best available method
    #[inline(always)]
    fn compute<D: Distance>(&self, node_id: u32) -> Result<f32> {
        // SQ8 fast path (skip if force_full_precision)
        if !self.force_full_precision {
            if let Some(ref prep) = self.sq8_prep {
                if let Some(dist) = self.storage.distance_sq8(prep, node_id) {
                    return Ok(dist);
                }
            }
        }

        // Full precision fallback
        if self.storage.is_sq8() {
            // Dequantize for SQ8 mode
            let vec = self
                .storage
                .get_dequantized(node_id)
                .ok_or(HNSWError::VectorNotFound(node_id))?;
            Ok(D::distance(self.query, &vec))
        } else {
            // Direct access for full precision
            let vec = self.storage.vector(node_id);
            Ok(D::distance(self.query, vec))
        }
    }

    /// Check if batch distance computation is available (SQ8 mode)
    #[inline(always)]
    fn has_batch(&self) -> bool {
        !self.force_full_precision && self.sq8_prep.is_some()
    }

    /// Batch compute distances to multiple nodes (SQ8 fast path)
    ///
    /// Returns the number of distances computed. Caller must provide output buffer
    /// large enough to hold distances for all IDs.
    #[inline]
    fn compute_batch(&self, ids: &[u32], distances: &mut [f32]) -> usize {
        if let Some(ref prep) = self.sq8_prep {
            self.storage.distance_sq8_batch(prep, ids, distances)
        } else {
            0
        }
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
            // Upper levels: sparse storage
            for id in self.storage.neighbors_at_level(node_id, level) {
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
struct AcornCollector<'a, F>
where
    F: Fn(u32) -> bool,
{
    index: &'a HNSWIndex,
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
        self.index.collect_matching_neighbors_acorn1(
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
                self.index.collect_matching_neighbors_acorn1(
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
    ) -> Result<Vec<u32>>
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
                if let Some(&farthest) = working.peek() {
                    if current.distance > farthest.distance {
                        break;
                    }
                }

                // Collect neighbors using specialized collector
                collector.collect(current.node_id, level, visited, unvisited);

                let neighbors_slice = unvisited.as_slice();
                let num_neighbors = neighbors_slice.len();

                if num_neighbors == 0 {
                    continue;
                }

                if use_batch && num_neighbors > 1 {
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
                        }
                    }

                    for (i, &neighbor_id) in neighbors_slice.iter().enumerate() {
                        if PREFETCH_ENABLED && i + PREFETCH_DISTANCE < num_neighbors {
                            let prefetch_id = neighbors_slice[i + PREFETCH_DISTANCE];
                            self.storage.prefetch(prefetch_id);
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

            // Return node IDs sorted by distance
            results_buf.extend(working.drain());
            results_buf.sort_unstable_by_key(|c| c.distance);
            let mut output = Vec::with_capacity(results_buf.len());
            output.extend(results_buf.iter().map(|c| c.node_id));
            Ok(output)
        })
    }

    /// Validate search parameters (k, ef, query dimensions, NaN/Inf)
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

        // Start from entry point, descend to layer 0
        let mut nearest = vec![entry_point];

        // Greedy search at each layer (find 1 nearest)
        for level in (1..=entry_level).rev() {
            nearest = self.search_layer(query, &nearest, 1, level)?;
        }

        // Beam search at layer 0 (find ef nearest)
        let candidates = self.search_layer(query, &nearest, ef.max(k), 0)?;

        // Convert to SearchResult and return k nearest
        // Pre-allocate with exact capacity to avoid reallocations
        let mut results = Vec::with_capacity(candidates.len());
        for &id in &candidates {
            let distance = self.distance_exact(query, id)?;
            // Return slot (original RecordStore index) not internal node id
            // After optimize(), id may differ from slot
            let slot = self.storage.slot(id);
            results.push(SearchResult::new(slot, distance));
        }

        // Sort by distance (closest first) - unstable is faster
        results.sort_unstable_by_key(|r| OrderedFloat(r.distance));

        // Return top k
        results.truncate(k);

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
    pub fn search_with_filter<F>(
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
        const SELECTIVITY_THRESHOLD: f32 = 0.6;
        const SMALL_GRAPH_SIZE: usize = 1000;

        if selectivity > SELECTIVITY_THRESHOLD || self.len() <= SMALL_GRAPH_SIZE {
            // Filter is broad (>60% match) or graph is small: use standard search + post-filter
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

        // Filter is selective (<60% match): use ACORN-1
        debug!(selectivity, "Using ACORN-1 filtered search");

        let entry_point = self.entry_point.ok_or(HNSWError::EmptyIndex)?;
        let entry_level = self.storage.level(entry_point);

        // Start from entry point, descend to layer 0
        let mut nearest = vec![entry_point];

        // Greedy search at each layer (find 1 nearest that matches filter)
        for level in (1..=entry_level).rev() {
            nearest = self.search_layer_with_filter(query, &nearest, 1, level, &slot_filter)?;
            if nearest.is_empty() {
                // No matching nodes found at this level, try standard search
                debug!(level, "No matches at this level, falling back");
                nearest = vec![entry_point];
            }
        }

        // Beam search at layer 0 (find ef nearest that match filter)
        let candidates =
            self.search_layer_with_filter(query, &nearest, ef.max(k), 0, &slot_filter)?;

        // Convert to SearchResult and return k nearest
        // Pre-allocate with exact capacity to avoid reallocations
        let mut results = Vec::with_capacity(candidates.len());
        for &id in &candidates {
            let distance = self.distance_exact(query, id)?;
            // Return slot (original RecordStore index) not internal node id
            let slot = self.storage.slot(id);
            results.push(SearchResult::new(slot, distance));
        }

        results.sort_unstable_by_key(|r| OrderedFloat(r.distance));
        results.truncate(k);

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
        let step = self.len() / sample_size;

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
    ) -> Result<Vec<u32>>
    where
        F: Fn(u32) -> bool,
    {
        // Dispatch once at the top level to get full monomorphization benefits
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

    /// ACORN-1 GET-NEIGHBORS: Collect matching neighbors with 2-hop expansion.
    ///
    /// Implements the per-neighbor adaptive expansion from ACORN-1 (arXiv:2403.04871):
    /// - If a 1-hop neighbor matches the filter, add it directly
    /// - If a 1-hop neighbor doesn't match, expand to its neighbors (2-hop)
    /// - Stop early once M matching neighbors are found (truncation)
    #[inline]
    fn collect_matching_neighbors_acorn1<F>(
        &self,
        source_node: u32,
        level: u8,
        visited: &super::super::query_buffers::VisitedList,
        filter_fn: &F,
        m: usize,
        output: &mut Vec<u32>,
    ) where
        F: Fn(u32) -> bool,
    {
        output.clear();
        let neighbors_1hop = self.storage.neighbors_at_level(source_node, level);

        for neighbor_id in neighbors_1hop {
            if visited.contains(neighbor_id) {
                continue;
            }
            if filter_fn(neighbor_id) {
                output.push(neighbor_id);
                if output.len() >= m {
                    return;
                }
            } else {
                // 2-hop expansion for non-matching neighbors
                let second_hop = self.storage.neighbors_at_level(neighbor_id, level);
                for second_hop_id in second_hop {
                    if !visited.contains(second_hop_id) && filter_fn(second_hop_id) {
                        output.push(second_hop_id);
                        if output.len() >= m {
                            return;
                        }
                    }
                }
            }
        }
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
    ) -> Result<Vec<u32>>
    where
        F: Fn(u32) -> bool,
    {
        let ctx = DistanceContext::new(query, self, false);
        let collector = AcornCollector {
            index: self,
            filter_fn,
            m: self.params.m,
        };

        self.search_layer_internal::<D, _>(entry_points, &ctx, &collector, ef, level)
    }

    /// Search for nearest neighbors at a specific level
    ///
    /// Returns node IDs of up to ef nearest neighbors.
    ///
    /// Optimized (Nov 25, 2025):
    /// - Uses `VisitedList` with O(1) clear (generation-based, like hnswlib)
    /// - Reuses pre-allocated unvisited buffer to avoid per-iteration allocation
    pub(super) fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
    ) -> Result<Vec<u32>> {
        // Dispatch once at the top level to get full monomorphization benefits
        // inside the hot loop. Critical for x86/ARM servers.
        dispatch_distance!(self.distance_fn, D => {
            self.search_layer_mono::<D>(query, entry_points, ef, level)
        })
    }

    /// Monomorphized search layer (static dispatch, no match in hot loop)
    ///
    /// The Distance trait enables compile-time specialization. The compiler
    /// generates separate versions for L2, Cosine, and NegDot with the
    /// distance function fully inlined.
    #[inline(never)] // Prevent inlining dispatcher - we want separate code paths
    pub(super) fn search_layer_mono<D: Distance>(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
    ) -> Result<Vec<u32>> {
        let ctx = DistanceContext::new(query, self, false);
        let collector = StandardCollector {
            storage: &self.storage,
        };

        self.search_layer_internal::<D, _>(entry_points, &ctx, &collector, ef, level)
    }

    /// Search layer using full precision (f32) distances
    ///
    /// Used during graph construction where quantization noise hurts graph quality.
    /// Same algorithm as search_layer but uses distance_cmp_full_precision.
    pub(super) fn search_layer_full_precision(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
    ) -> Result<Vec<u32>> {
        dispatch_distance!(self.distance_fn, D => {
            let ctx = DistanceContext::new(query, self, true);
            let collector = StandardCollector {
                storage: &self.storage,
            };

            self.search_layer_internal::<D, _>(entry_points, &ctx, &collector, ef, level)
        })
    }

    /// Search layer returning (node_id, distance) pairs
    ///
    /// Used during graph construction to avoid recomputing distances in select_neighbors_heuristic.
    /// Returns candidates sorted by distance (closest first).
    pub(super) fn search_layer_with_distances(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
    ) -> Result<Vec<(u32, f32)>> {
        dispatch_distance!(self.distance_fn, D => {
            let ctx = DistanceContext::new(query, self, true);
            let collector = StandardCollector {
                storage: &self.storage,
            };

            self.search_layer_internal_with_distances::<D, _>(entry_points, &ctx, &collector, ef, level)
        })
    }

    /// Internal search returning (node_id, distance) pairs instead of just IDs
    #[inline(always)]
    fn search_layer_internal_with_distances<D, C>(
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

            let use_batch = ctx.has_batch();

            // Greedy search (same as search_layer_internal)
            while let Some(Reverse(current)) = candidates.pop() {
                if let Some(&farthest) = working.peek() {
                    if current.distance > farthest.distance {
                        break;
                    }
                }

                collector.collect(current.node_id, level, visited, unvisited);
                let neighbors_slice = unvisited.as_slice();
                let num_neighbors = neighbors_slice.len();

                if num_neighbors == 0 {
                    continue;
                }

                if use_batch && num_neighbors > 1 {
                    if batch_distances.len() < num_neighbors {
                        batch_distances.resize(num_neighbors, 0.0);
                    }

                    let computed = ctx.compute_batch(neighbors_slice, batch_distances);
                    debug_assert_eq!(computed, num_neighbors, "batch distance count mismatch");

                    for (i, &neighbor_id) in neighbors_slice.iter().enumerate() {
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
                    use crate::vector::hnsw::prefetch::PrefetchConfig;
                    const PREFETCH_ENABLED: bool = PrefetchConfig::enabled();
                    const PREFETCH_DISTANCE: usize = PrefetchConfig::stride();

                    if PREFETCH_ENABLED {
                        for &id in neighbors_slice.iter().take(PREFETCH_DISTANCE) {
                            self.storage.prefetch(id);
                        }
                    }

                    for (i, &neighbor_id) in neighbors_slice.iter().enumerate() {
                        if PREFETCH_ENABLED && i + PREFETCH_DISTANCE < num_neighbors {
                            let prefetch_id = neighbors_slice[i + PREFETCH_DISTANCE];
                            self.storage.prefetch(prefetch_id);
                        }

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
}
