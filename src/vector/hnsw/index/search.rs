//! HNSW search operations
//!
//! Contiguous memory access and speculative prefetching.

use super::HNSWIndex;
use crate::vector::hnsw::error::{HNSWError, Result};
use crate::vector::hnsw::storage::HNSWStorage;
use crate::vector::hnsw::types::{Candidate, Distance, SearchResult};
use crate::vector::hnsw::visited::{VisitedList, with_visited_list};

/// Context for distance computation during search.
struct DistanceContext<'a, D: Distance> {
    query: &'a [f32],
    storage: &'a HNSWStorage,
    _metric: std::marker::PhantomData<D>,
}

impl<'a, D: Distance> DistanceContext<'a, D> {
    fn new(query: &'a [f32], storage: &'a HNSWStorage) -> Self {
        Self {
            query,
            storage,
            _metric: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn compute(&self, node_id: u32) -> f32 {
        let vec = self.storage.vector(node_id);
        D::distance(self.query, vec)
    }
}

impl HNSWIndex {
    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimensions() {
            return Err(HNSWError::DimensionMismatch {
                expected: self.dimensions(),
                actual: query.len(),
            });
        }

        // Validate query for NaN and Infinity
        for &val in query {
            if !val.is_finite() {
                return Err(HNSWError::InvalidVector);
            }
        }

        if self.entry_point.is_none() || k == 0 {
            return Ok(Vec::new());
        }

        if ef < k {
            return Err(HNSWError::InvalidSearchParams { k, ef });
        }

        if ef == 0 {
            return Err(HNSWError::InvalidSearchParams { k, ef });
        }

        dispatch_distance!(self.distance_fn, D => {
            D::validate_query(query)?;
            let ctx = DistanceContext::<D>::new(query, &self.storage);

            Ok(with_visited_list(self.len() + 1, |visited| {
                self.search_internal::<D>(query, k, ef, &ctx, visited)
            }))
        })
    }

    /// Search for k nearest neighbors with a filter.
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_fn: &(dyn Fn(u32) -> bool + Sync + Send),
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimensions() {
            return Err(HNSWError::DimensionMismatch {
                expected: self.dimensions(),
                actual: query.len(),
            });
        }

        if self.entry_point.is_none() || k == 0 {
            return Ok(Vec::new());
        }

        dispatch_distance!(self.distance_fn, D => {
            D::validate_query(query)?;
            let ctx = DistanceContext::<D>::new(query, &self.storage);

            // Boost ef for filtered search to improve recall
            let boosted_ef = ef.max(k * 4).max(100);
            Ok(with_visited_list(self.len() + 1, |visited| {
                self.search_internal_filtered::<D>(query, k, boosted_ef, &ctx, visited, filter_fn)
            }))
        })
    }

    fn search_internal<D: Distance>(
        &self,
        _query: &[f32],
        k: usize,
        ef: usize,
        ctx: &DistanceContext<D>,
        visited: &mut VisitedList,
    ) -> Vec<SearchResult> {
        let entry_point = self.entry_point.unwrap();
        let mut nearest_node = entry_point;
        let mut nearest_dist = ctx.compute(entry_point);

        let entry_level = self.storage.get_node_level(entry_point);

        // 1. Greedy descent to layer 0
        for level in (1..=entry_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                self.storage
                    .with_neighbors(nearest_node, level, |neighbors| {
                        for &neighbor_id in neighbors {
                            let dist = ctx.compute(neighbor_id);
                            if dist < nearest_dist {
                                nearest_dist = dist;
                                nearest_node = neighbor_id;
                                changed = true;
                            }
                        }
                    });
            }
        }

        // 2. Beam search at layer 0
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut frontier = BinaryHeap::with_capacity(ef);
        let mut candidates = BinaryHeap::with_capacity(ef);

        frontier.push(Reverse(Candidate::new(nearest_node, nearest_dist)));
        candidates.push(Candidate::new(nearest_node, nearest_dist));
        visited.mark_visited(nearest_node);

        while let Some(Reverse(current)) = frontier.pop() {
            if !candidates.is_empty()
                && candidates.len() >= ef
                && current.distance > candidates.peek().unwrap().distance
            {
                break;
            }

            self.storage
                .with_neighbors(current.node_id, 0, |neighbors| {
                    for (i, &neighbor_id) in neighbors.iter().enumerate() {
                        if visited.is_visited(neighbor_id) {
                            continue;
                        }
                        visited.mark_visited(neighbor_id);

                        // Speculative Prefetching (VSAG style)
                        let stride = crate::vector::hnsw::prefetch::PrefetchConfig::stride();
                        if i + stride < neighbors.len() {
                            self.storage.neighbors.prefetch(neighbors[i + stride], 0);
                            self.storage.prefetch_vector(neighbors[i + stride]);
                        }

                        let dist = ctx.compute(neighbor_id);
                        if candidates.len() < ef || dist < *candidates.peek().unwrap().distance {
                            let candidate = Candidate::new(neighbor_id, dist);
                            frontier.push(Reverse(candidate));
                            candidates.push(candidate);
                            if candidates.len() > ef {
                                candidates.pop();
                            }
                        }
                    }
                });
        }

        let mut results = Vec::with_capacity(k);
        while let Some(c) = candidates.pop() {
            results.push(SearchResult {
                slot: c.node_id,
                distance: c.distance.into_inner(),
            });
        }
        results.reverse();
        results.truncate(k);
        results
    }

    fn search_internal_filtered<D: Distance>(
        &self,
        _query: &[f32],
        k: usize,
        ef: usize,
        ctx: &DistanceContext<D>,
        visited: &mut VisitedList,
        filter_fn: &(dyn Fn(u32) -> bool + Sync + Send),
    ) -> Vec<SearchResult> {
        let entry_point = self.entry_point.unwrap();
        let mut nearest_node = entry_point;
        let mut nearest_dist = ctx.compute(entry_point);

        let entry_level = self.storage.get_node_level(entry_point);

        // 1. Greedy descent to layer 0
        for level in (1..=entry_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                self.storage
                    .with_neighbors(nearest_node, level, |neighbors| {
                        for &neighbor_id in neighbors {
                            let dist = ctx.compute(neighbor_id);
                            if dist < nearest_dist {
                                nearest_dist = dist;
                                nearest_node = neighbor_id;
                                changed = true;
                            }
                        }
                    });
            }
        }

        // 2. Beam search at layer 0 with filter
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut frontier = BinaryHeap::with_capacity(ef);
        let mut candidates = BinaryHeap::with_capacity(k); // Results (filter-passing)
        let mut top_dists = BinaryHeap::with_capacity(ef); // Search threshold (all-pass)

        let initial_dist = ctx.compute(nearest_node);
        frontier.push(Reverse(Candidate::new(nearest_node, initial_dist)));
        if filter_fn(nearest_node) {
            candidates.push(Candidate::new(nearest_node, initial_dist));
        }
        top_dists.push(Candidate::new(nearest_node, initial_dist));
        visited.mark_visited(nearest_node);

        while let Some(Reverse(current)) = frontier.pop() {
            // Standard HNSW termination: current best is worse than our ef-th best overall
            if top_dists.len() >= ef && current.distance > top_dists.peek().unwrap().distance {
                break;
            }

            self.storage
                .with_neighbors(current.node_id, 0, |neighbors| {
                    for (i, &neighbor_id) in neighbors.iter().enumerate() {
                        if visited.is_visited(neighbor_id) {
                            continue;
                        }
                        visited.mark_visited(neighbor_id);

                        // Speculative Prefetching (VSAG style)
                        let stride = crate::vector::hnsw::prefetch::PrefetchConfig::stride();
                        if i + stride < neighbors.len() {
                            self.storage.neighbors.prefetch(neighbors[i + stride], 0);
                            self.storage.prefetch_vector(neighbors[i + stride]);
                        }

                        let dist = ctx.compute(neighbor_id);
                        let candidate = Candidate::new(neighbor_id, dist);

                        // If this node is better than the global threshold, we add it to frontier
                        // even if it fails the filter (this is the key to ACORN-style jumping)
                        if top_dists.len() < ef || dist < *top_dists.peek().unwrap().distance {
                            frontier.push(Reverse(candidate));

                            top_dists.push(candidate);
                            if top_dists.len() > ef {
                                top_dists.pop();
                            }
                        }

                        // But ONLY add to results if it passes the filter
                        if filter_fn(neighbor_id)
                            && (candidates.len() < k || dist < *candidates.peek().unwrap().distance)
                        {
                            candidates.push(candidate);
                            if candidates.len() > k {
                                candidates.pop();
                            }
                        }
                    }
                });
        }

        let mut results = Vec::with_capacity(k);
        while let Some(c) = candidates.pop() {
            results.push(SearchResult {
                slot: c.node_id,
                distance: c.distance.into_inner(),
            });
        }
        results.reverse();
        results.truncate(k);
        results
    }
}
