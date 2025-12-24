//! HNSW search operations
//!
//! Implements k-NN search, filtered search (ACORN-1), and layer-level search.

use super::HNSWIndex;
use crate::distance::norm_squared;
use crate::vector::hnsw::error::{HNSWError, Result};
use crate::vector::hnsw::storage::UnifiedADC;
use crate::vector::hnsw::types::{
    Candidate, Cosine, Distance, DistanceFunction, NegDot, SearchResult, L2,
};
use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use tracing::{debug, error, instrument, warn};

impl HNSWIndex {
    /// Search for k nearest neighbors
    ///
    /// Returns up to k nearest neighbors sorted by distance (closest first).
    #[instrument(skip(self, query), fields(k, ef, dimensions = query.len(), index_size = self.len()))]
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        // Validate k > 0
        if k == 0 {
            error!(k, ef, "Invalid search parameters: k must be > 0");
            return Err(HNSWError::InvalidSearchParams { k, ef });
        }

        // Validate ef >= k
        if ef < k {
            error!(k, ef, "Invalid search parameters: ef must be >= k");
            return Err(HNSWError::InvalidSearchParams { k, ef });
        }

        // Validate dimensions
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

        // Check for NaN/Inf in query
        if query.iter().any(|x| !x.is_finite()) {
            error!("Invalid query vector: contains NaN or Inf values");
            return Err(HNSWError::InvalidVector);
        }

        // Handle empty index
        if self.is_empty() {
            debug!("Search on empty index, returning empty results");
            return Ok(Vec::new());
        }

        let entry_point = self.entry_point.ok_or(HNSWError::EmptyIndex)?;
        let entry_level = self.nodes[entry_point as usize].level;

        // Start from entry point, descend to layer 0
        let mut nearest = vec![entry_point];

        // Use asymmetric search for RaBitQ storage (CLOUD MOAT - 2-3x speedup)
        let use_asymmetric = self.is_asymmetric();

        // Greedy search at each layer (find 1 nearest)
        for level in (1..=entry_level).rev() {
            nearest = if use_asymmetric {
                self.search_layer_asymmetric(query, &nearest, 1, level)?
            } else {
                self.search_layer(query, &nearest, 1, level)?
            };
        }

        // Beam search at layer 0 (find ef nearest)
        let candidates = if use_asymmetric {
            self.search_layer_asymmetric(query, &nearest, ef.max(k), 0)?
        } else {
            self.search_layer(query, &nearest, ef.max(k), 0)?
        };

        // Convert to SearchResult and return k nearest
        // Pre-allocate with exact capacity to avoid reallocations
        let mut results = Vec::with_capacity(candidates.len());
        for &id in &candidates {
            let distance = self.distance_exact(query, id)?;
            results.push(SearchResult::new(id, distance));
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

    /// Search using quantized (ADC) distances only - no exact distance calculation.
    ///
    /// Returns candidates with approximate distances computed via ADC tables.
    /// Use this for fast search when rescore=False (accept quantization error).
    ///
    /// Falls back to regular search if not in asymmetric mode.
    #[instrument(skip(self, query), fields(k, ef, dimensions = query.len(), index_size = self.len()))]
    pub fn search_asymmetric(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<SearchResult>> {
        // Validate inputs (same as search())
        if k == 0 {
            return Err(HNSWError::InvalidSearchParams { k, ef });
        }
        if ef < k {
            return Err(HNSWError::InvalidSearchParams { k, ef });
        }
        if query.len() != self.dimensions() {
            return Err(HNSWError::DimensionMismatch {
                expected: self.dimensions(),
                actual: query.len(),
            });
        }
        if query.iter().any(|x| !x.is_finite()) {
            return Err(HNSWError::InvalidVector);
        }
        if self.is_empty() {
            return Ok(Vec::new());
        }

        // If not asymmetric, fall back to regular search
        if !self.is_asymmetric() {
            return self.search(query, k, ef);
        }

        let entry_point = self.entry_point.ok_or(HNSWError::EmptyIndex)?;
        let entry_level = self.nodes[entry_point as usize].level;

        // Build ADC lookup table once for this query
        let adc_table = self.vectors.build_adc_table(query);

        // Greedy search at each layer using ADC distances
        let mut nearest = vec![entry_point];
        for level in (1..=entry_level).rev() {
            nearest = self.search_layer_asymmetric(query, &nearest, 1, level)?;
        }

        // Beam search at layer 0 with ADC distances
        let candidates = self.search_layer_asymmetric_with_distances(
            query,
            &nearest,
            ef.max(k),
            0,
            adc_table.as_ref(),
        )?;

        // Return top k with ADC distances (no recomputation)
        let mut results: Vec<SearchResult> = candidates
            .into_iter()
            .map(|(id, dist)| SearchResult::new(id, dist))
            .collect();

        results.truncate(k);

        debug!(
            num_results = results.len(),
            closest_distance = results.first().map(|r| r.distance),
            "Asymmetric search completed (ADC distances)"
        );

        Ok(results)
    }

    /// Search layer returning (id, distance) tuples for asymmetric mode.
    pub(super) fn search_layer_asymmetric_with_distances(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
        adc_table: Option<&UnifiedADC>,
    ) -> Result<Vec<(u32, f32)>> {
        use super::super::query_buffers;

        query_buffers::with_buffers(|buffers| {
            let visited = &mut buffers.visited;
            let candidates = &mut buffers.candidates;
            let working = &mut buffers.working;
            let unvisited = &mut buffers.unvisited;

            for &ep in entry_points {
                let dist = self.distance_with_adc(query, ep, adc_table)?;
                let candidate = Candidate::new(ep, dist);

                candidates.push(Reverse(candidate));
                working.push(candidate);
                visited.insert(ep);
            }

            while let Some(Reverse(current)) = candidates.pop() {
                if let Some(&farthest) = working.peek() {
                    if current.distance > farthest.distance {
                        break;
                    }
                }

                unvisited.clear();
                self.neighbors
                    .with_neighbors(current.node_id, level, |neighbors| {
                        for &id in neighbors {
                            if !visited.contains(id) {
                                unvisited.push(id);
                            }
                        }
                    });

                // Platform-aware prefetching: disabled on Apple Silicon (DMP handles it)
                use crate::vector::hnsw::prefetch::PrefetchConfig;
                const PREFETCH_ENABLED: bool = PrefetchConfig::enabled();
                const PREFETCH_DISTANCE: usize = PrefetchConfig::stride();

                let unvisited_slice = unvisited.as_slice();

                if PREFETCH_ENABLED {
                    for &id in unvisited_slice.iter().take(PREFETCH_DISTANCE) {
                        self.vectors.prefetch_quantized(id);
                    }
                }

                for (i, &neighbor_id) in unvisited_slice.iter().enumerate() {
                    if PREFETCH_ENABLED && i + PREFETCH_DISTANCE < unvisited_slice.len() {
                        self.vectors
                            .prefetch_quantized(unvisited_slice[i + PREFETCH_DISTANCE]);
                    }

                    visited.insert(neighbor_id);

                    let dist = self.distance_with_adc(query, neighbor_id, adc_table)?;
                    let neighbor = Candidate::new(neighbor_id, dist);

                    if let Some(&farthest) = working.peek() {
                        if dist < farthest.distance.0 || working.len() < ef {
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

            // Return (id, distance) tuples sorted by distance
            let mut results: Vec<_> = working.drain().map(|c| (c.node_id, c.distance.0)).collect();
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            Ok(results)
        })
    }

    /// Search for k nearest neighbors with metadata filtering (ACORN-1)
    ///
    /// Implements ACORN-1 filtered search algorithm for efficient metadata-aware search.
    /// Skips distance calculations for nodes that don't match the filter.
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
        // Validate parameters (same as standard search)
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
        if self.is_empty() {
            debug!("Search on empty index, returning empty results");
            return Ok(Vec::new());
        }

        // Estimate filter selectivity
        let selectivity = self.estimate_selectivity(&filter_fn);

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
        let entry_level = self.nodes[entry_point as usize].level;

        // Start from entry point, descend to layer 0
        let mut nearest = vec![entry_point];

        // Greedy search at each layer (find 1 nearest that matches filter)
        for level in (1..=entry_level).rev() {
            nearest =
                self.search_layer_with_filter(query, &nearest, 1, level, &filter_fn, selectivity)?;
            if nearest.is_empty() {
                // No matching nodes found at this level, try standard search
                debug!(level, "No matches at this level, falling back");
                nearest = vec![entry_point];
            }
        }

        // Beam search at layer 0 (find ef nearest that match filter)
        let candidates =
            self.search_layer_with_filter(query, &nearest, ef.max(k), 0, &filter_fn, selectivity)?;

        // Convert to SearchResult and return k nearest
        // Pre-allocate with exact capacity to avoid reallocations
        let mut results = Vec::with_capacity(candidates.len());
        for &id in &candidates {
            let distance = self.distance_exact(query, id)?;
            results.push(SearchResult::new(id, distance));
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
    #[allow(clippy::too_many_arguments)]
    pub(super) fn search_layer_with_filter<F>(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
        filter_fn: &F,
        selectivity: f32,
    ) -> Result<Vec<u32>>
    where
        F: Fn(u32) -> bool,
    {
        // Dispatch once at the top level to get full monomorphization benefits
        match self.distance_fn {
            DistanceFunction::L2 => self.search_layer_with_filter_mono::<L2, F>(
                query,
                entry_points,
                ef,
                level,
                filter_fn,
                selectivity,
            ),
            DistanceFunction::Cosine => self.search_layer_with_filter_mono::<Cosine, F>(
                query,
                entry_points,
                ef,
                level,
                filter_fn,
                selectivity,
            ),
            DistanceFunction::NegativeDotProduct => self
                .search_layer_with_filter_mono::<NegDot, F>(
                    query,
                    entry_points,
                    ef,
                    level,
                    filter_fn,
                    selectivity,
                ),
        }
    }

    /// Monomorphized filtered search layer (static dispatch, no match in hot loop)
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    pub(super) fn search_layer_with_filter_mono<D: Distance, F>(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
        filter_fn: &F,
        selectivity: f32,
    ) -> Result<Vec<u32>>
    where
        F: Fn(u32) -> bool,
    {
        use super::super::query_buffers;

        // Determine if we need 2-hop exploration (very selective filters)
        const TWO_HOP_THRESHOLD: f32 = 0.1;
        let use_two_hop = selectivity < TWO_HOP_THRESHOLD;

        if use_two_hop {
            debug!(selectivity, "Using 2-hop exploration for sparse filter");
        }

        // L2 decomposition optimization: pre-compute query norm once
        let use_l2_decomposition =
            self.supports_l2_decomposition() && D::as_enum() == DistanceFunction::L2;
        let query_norm = if use_l2_decomposition {
            norm_squared(query)
        } else {
            0.0
        };

        query_buffers::with_buffers(|buffers| {
            let visited = &mut buffers.visited;
            let candidates = &mut buffers.candidates;
            let working = &mut buffers.working;
            let neighbors_to_explore = &mut buffers.unvisited; // Reuse unvisited buffer
            let results_buf = &mut buffers.results;

            // Initialize with entry points (only add if they match filter)
            for &ep in entry_points {
                if !filter_fn(ep) {
                    visited.insert(ep);
                    continue;
                }

                let dist = if use_l2_decomposition {
                    self.distance_l2_decomposed(query, query_norm, ep)
                        .ok_or(HNSWError::VectorNotFound(ep))?
                } else {
                    self.distance_cmp_mono::<D>(query, ep)?
                };
                let candidate = Candidate::new(ep, dist);

                candidates.push(Reverse(candidate));
                working.push(candidate);
                visited.insert(ep);
            }

            // If no entry points match, return empty
            if candidates.is_empty() {
                return Ok(Vec::new());
            }

            // Greedy search with filtered distance calculations
            while let Some(Reverse(current)) = candidates.pop() {
                // If current is farther than farthest in working set, stop
                if let Some(&farthest) = working.peek() {
                    if current.distance > farthest.distance {
                        break;
                    }
                }

                // Collect neighbors into pre-allocated buffer (no allocation!)
                neighbors_to_explore.clear();
                let neighbors = self.neighbors.get_neighbors(current.node_id, level);

                for &neighbor_id in &neighbors {
                    if visited.contains(neighbor_id) {
                        continue;
                    }

                    neighbors_to_explore.push(neighbor_id);

                    // 2-hop exploration: if neighbor doesn't match filter, explore its neighbors
                    if use_two_hop && !filter_fn(neighbor_id) {
                        let second_hop = self.neighbors.get_neighbors(neighbor_id, level);
                        for &second_hop_id in &second_hop {
                            if !visited.contains(second_hop_id) {
                                neighbors_to_explore.push(second_hop_id);
                            }
                        }
                    }
                }

                // Process all neighbors (1-hop and 2-hop) with prefetching
                // Platform-aware prefetching: disabled on Apple Silicon
                use crate::vector::hnsw::prefetch::PrefetchConfig;
                const PREFETCH_ENABLED: bool = PrefetchConfig::enabled();
                const PREFETCH_DISTANCE: usize = PrefetchConfig::stride();

                let neighbors_slice = neighbors_to_explore.as_slice();

                // Initial burst prefetch (skip on Apple Silicon)
                if PREFETCH_ENABLED {
                    for &id in neighbors_slice.iter().take(PREFETCH_DISTANCE) {
                        self.vectors.prefetch(id);
                        self.neighbors.prefetch(id, level); // Graph-aware prefetch
                    }
                }

                for (i, &neighbor_id) in neighbors_slice.iter().enumerate() {
                    // Stride prefetch: vectors and neighbor lists
                    if PREFETCH_ENABLED && i + PREFETCH_DISTANCE < neighbors_slice.len() {
                        let prefetch_id = neighbors_slice[i + PREFETCH_DISTANCE];
                        self.vectors.prefetch(prefetch_id);
                        self.neighbors.prefetch(prefetch_id, level); // Graph-aware prefetch
                    }

                    if visited.contains(neighbor_id) {
                        continue;
                    }
                    visited.insert(neighbor_id);

                    // ACORN-1 key optimization: skip distance calculation if filter doesn't match
                    if !filter_fn(neighbor_id) {
                        continue;
                    }

                    let dist = if use_l2_decomposition {
                        self.distance_l2_decomposed(query, query_norm, neighbor_id)
                            .ok_or(HNSWError::VectorNotFound(neighbor_id))?
                    } else {
                        self.distance_cmp_mono::<D>(query, neighbor_id)?
                    };
                    let neighbor = Candidate::new(neighbor_id, dist);

                    // If neighbor is closer than farthest in working set, or working set not full, add it
                    if let Some(&farthest) = working.peek() {
                        if dist < farthest.distance.0 || working.len() < ef {
                            candidates.push(Reverse(neighbor));
                            working.push(neighbor);

                            // Prune working set to ef size
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

            // Return node IDs sorted by distance (closest first)
            // Use pre-allocated buffer to avoid per-search allocation
            results_buf.extend(working.drain());
            results_buf.sort_unstable_by_key(|c| c.distance);
            let mut output = Vec::with_capacity(results_buf.len());
            output.extend(results_buf.iter().map(|c| c.node_id));
            Ok(output)
        })
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
        match self.distance_fn {
            DistanceFunction::L2 => self.search_layer_mono::<L2>(query, entry_points, ef, level),
            DistanceFunction::Cosine => {
                self.search_layer_mono::<Cosine>(query, entry_points, ef, level)
            }
            DistanceFunction::NegativeDotProduct => {
                self.search_layer_mono::<NegDot>(query, entry_points, ef, level)
            }
        }
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
        use super::super::query_buffers;

        // L2 decomposition optimization: pre-compute query norm once
        // ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩ (~7% faster for L2)
        let use_l2_decomposition =
            self.supports_l2_decomposition() && D::as_enum() == DistanceFunction::L2;
        let query_norm = if use_l2_decomposition {
            norm_squared(query)
        } else {
            0.0 // unused
        };

        query_buffers::with_buffers(|buffers| {
            let visited = &mut buffers.visited;
            let candidates = &mut buffers.candidates;
            let working = &mut buffers.working;
            let unvisited = &mut buffers.unvisited;
            let results_buf = &mut buffers.results;

            // Initialize with entry points
            for &ep in entry_points {
                let dist = if use_l2_decomposition {
                    self.distance_l2_decomposed(query, query_norm, ep)
                        .ok_or(HNSWError::VectorNotFound(ep))?
                } else {
                    self.distance_cmp_mono::<D>(query, ep)?
                };
                let candidate = Candidate::new(ep, dist);

                candidates.push(Reverse(candidate));
                working.push(candidate);
                visited.insert(ep);
            }

            // Greedy search
            while let Some(Reverse(current)) = candidates.pop() {
                // If current is farther than farthest in working set, stop
                if let Some(&farthest) = working.peek() {
                    if current.distance > farthest.distance {
                        break;
                    }
                }

                // Collect unvisited neighbors into pre-allocated buffer (no allocation!)
                unvisited.clear();
                self.neighbors
                    .with_neighbors(current.node_id, level, |neighbors| {
                        for &id in neighbors {
                            if !visited.contains(id) {
                                unvisited.push(id);
                            }
                        }
                    });

                // Platform-aware prefetching: disabled on Apple Silicon (DMP handles it)
                // enabled on x86/ARM servers where it provides 8-50% gains
                use crate::vector::hnsw::prefetch::PrefetchConfig;
                const PREFETCH_ENABLED: bool = PrefetchConfig::enabled();
                const PREFETCH_DISTANCE: usize = PrefetchConfig::stride();

                let unvisited_slice = unvisited.as_slice();

                // Initial burst prefetch (skip on Apple Silicon)
                // Prefetch both vectors AND neighbor lists for upcoming nodes
                if PREFETCH_ENABLED {
                    for &id in unvisited_slice.iter().take(PREFETCH_DISTANCE) {
                        self.vectors.prefetch(id);
                        self.neighbors.prefetch(id, level); // Graph-aware prefetch
                    }
                }

                for (i, &neighbor_id) in unvisited_slice.iter().enumerate() {
                    // Stride prefetch: vectors and neighbor lists
                    if PREFETCH_ENABLED && i + PREFETCH_DISTANCE < unvisited_slice.len() {
                        let prefetch_id = unvisited_slice[i + PREFETCH_DISTANCE];
                        self.vectors.prefetch(prefetch_id);
                        self.neighbors.prefetch(prefetch_id, level); // Graph-aware prefetch
                    }

                    visited.insert(neighbor_id);

                    // Use L2 decomposition when available (~7% faster for L2)
                    // Otherwise use monomorphized distance (static dispatch)
                    let dist = if use_l2_decomposition {
                        self.distance_l2_decomposed(query, query_norm, neighbor_id)
                            .ok_or(HNSWError::VectorNotFound(neighbor_id))?
                    } else {
                        self.distance_cmp_mono::<D>(query, neighbor_id)?
                    };
                    let neighbor = Candidate::new(neighbor_id, dist);

                    // If neighbor is closer than farthest in working set, or working set not full, add it
                    if let Some(&farthest) = working.peek() {
                        if dist < farthest.distance.0 || working.len() < ef {
                            candidates.push(Reverse(neighbor));
                            working.push(neighbor);

                            // Prune working set to ef size
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

            // Return node IDs sorted by distance (closest first)
            // Use pre-allocated buffer to avoid per-search allocation
            results_buf.extend(working.drain());
            results_buf.sort_unstable_by_key(|c| c.distance); // unstable is faster
            let mut output = Vec::with_capacity(results_buf.len());
            output.extend(results_buf.iter().map(|c| c.node_id));
            Ok(output)
        })
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
        use super::super::query_buffers;

        query_buffers::with_buffers(|buffers| {
            let visited = &mut buffers.visited;
            let candidates = &mut buffers.candidates;
            let working = &mut buffers.working;
            let unvisited = &mut buffers.unvisited;
            let results_buf = &mut buffers.results;

            // Initialize with entry points
            for &ep in entry_points {
                let dist = self.distance_cmp_full_precision(query, ep)?;
                let candidate = Candidate::new(ep, dist);

                candidates.push(Reverse(candidate));
                working.push(candidate);
                visited.insert(ep);
            }

            // Greedy search
            while let Some(Reverse(current)) = candidates.pop() {
                if let Some(&farthest) = working.peek() {
                    if current.distance > farthest.distance {
                        break;
                    }
                }

                unvisited.clear();
                self.neighbors
                    .with_neighbors(current.node_id, level, |neighbors| {
                        for &id in neighbors {
                            if !visited.contains(id) {
                                unvisited.push(id);
                            }
                        }
                    });

                // Platform-aware prefetching: disabled on Apple Silicon (DMP handles it)
                use crate::vector::hnsw::prefetch::PrefetchConfig;
                const PREFETCH_ENABLED: bool = PrefetchConfig::enabled();
                const PREFETCH_DISTANCE: usize = PrefetchConfig::stride();

                let unvisited_slice = unvisited.as_slice();

                if PREFETCH_ENABLED {
                    for &id in unvisited_slice.iter().take(PREFETCH_DISTANCE) {
                        self.vectors.prefetch(id);
                    }
                }

                for (i, &neighbor_id) in unvisited_slice.iter().enumerate() {
                    if PREFETCH_ENABLED && i + PREFETCH_DISTANCE < unvisited_slice.len() {
                        self.vectors
                            .prefetch(unvisited_slice[i + PREFETCH_DISTANCE]);
                    }

                    visited.insert(neighbor_id);

                    let dist = self.distance_cmp_full_precision(query, neighbor_id)?;
                    let neighbor = Candidate::new(neighbor_id, dist);

                    if let Some(&farthest) = working.peek() {
                        if dist < farthest.distance.0 || working.len() < ef {
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

            // Use pre-allocated buffer to avoid per-search allocation
            results_buf.extend(working.drain());
            results_buf.sort_unstable_by_key(|c| c.distance);
            let mut output = Vec::with_capacity(results_buf.len());
            output.extend(results_buf.iter().map(|c| c.node_id));
            Ok(output)
        })
    }

    /// Compute distance using ADC table if available, with fallback to asymmetric distance
    #[inline]
    pub(super) fn distance_with_adc(
        &self,
        query: &[f32],
        id: u32,
        adc_table: Option<&UnifiedADC>,
    ) -> Result<f32> {
        if let Some(adc) = adc_table {
            if let Some(dist) = self.vectors.distance_adc(adc, id) {
                return Ok(dist);
            }
            // ADC failed, try asymmetric distance
            if let Ok(dist) = self.distance_asymmetric(query, id) {
                return Ok(dist);
            }
            // Both failed - log and return max distance to push to end of results
            warn!(
                id,
                "ADC and asymmetric distance both failed, using f32::MAX"
            );
            Ok(f32::MAX)
        } else {
            self.distance_asymmetric(query, id)
        }
    }

    /// Asymmetric search layer for quantized storage (`RaBitQ` or SQ8)
    ///
    /// Uses ADC (Asymmetric Distance Computation) lookup tables for fast distance.
    /// Falls back to asymmetric distance if ADC fails.
    pub(super) fn search_layer_asymmetric(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
    ) -> Result<Vec<u32>> {
        use super::super::query_buffers;

        let adc_table = self.vectors.build_adc_table(query);

        query_buffers::with_buffers(|buffers| {
            let visited = &mut buffers.visited;
            let candidates = &mut buffers.candidates;
            let working = &mut buffers.working;
            let unvisited = &mut buffers.unvisited;
            let results_buf = &mut buffers.results;

            for &ep in entry_points {
                let dist = self.distance_with_adc(query, ep, adc_table.as_ref())?;
                let candidate = Candidate::new(ep, dist);

                candidates.push(Reverse(candidate));
                working.push(candidate);
                visited.insert(ep);
            }

            // Greedy search
            while let Some(Reverse(current)) = candidates.pop() {
                // If current is farther than farthest in working set, stop
                if let Some(&farthest) = working.peek() {
                    if current.distance > farthest.distance {
                        break;
                    }
                }

                // Collect unvisited neighbors into pre-allocated buffer
                unvisited.clear();
                self.neighbors
                    .with_neighbors(current.node_id, level, |neighbors| {
                        for &id in neighbors {
                            if !visited.contains(id) {
                                unvisited.push(id);
                            }
                        }
                    });

                // Platform-aware prefetching: disabled on Apple Silicon (DMP handles it)
                use crate::vector::hnsw::prefetch::PrefetchConfig;
                const PREFETCH_ENABLED: bool = PrefetchConfig::enabled();
                const PREFETCH_DISTANCE: usize = PrefetchConfig::stride();

                let unvisited_slice = unvisited.as_slice();

                if PREFETCH_ENABLED {
                    for &id in unvisited_slice.iter().take(PREFETCH_DISTANCE) {
                        self.vectors.prefetch_quantized(id);
                    }
                }

                for (i, &neighbor_id) in unvisited_slice.iter().enumerate() {
                    if PREFETCH_ENABLED && i + PREFETCH_DISTANCE < unvisited_slice.len() {
                        self.vectors
                            .prefetch_quantized(unvisited_slice[i + PREFETCH_DISTANCE]);
                    }

                    visited.insert(neighbor_id);

                    let dist = self.distance_with_adc(query, neighbor_id, adc_table.as_ref())?;
                    let neighbor = Candidate::new(neighbor_id, dist);

                    if let Some(&farthest) = working.peek() {
                        if dist < farthest.distance.0 || working.len() < ef {
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

            // Return node IDs sorted by distance (closest first)
            // Use pre-allocated buffer to avoid per-search allocation
            results_buf.extend(working.drain());
            results_buf.sort_unstable_by_key(|c| c.distance);
            let mut output = Vec::with_capacity(results_buf.len());
            output.extend(results_buf.iter().map(|c| c.node_id));
            Ok(output)
        })
    }
}
