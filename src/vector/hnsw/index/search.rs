//! HNSW search operations
//!
//! Contiguous memory access and speculative prefetching.

use super::HNSWIndex;
use crate::vector::hnsw::error::Result;
use crate::vector::hnsw::storage::HNSWStorage;
use crate::vector::hnsw::types::{Candidate, Distance, SearchResult};
use crate::vector::hnsw::visited::VisitedList;
use tracing::instrument;

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
    #[instrument(skip(self, query))]
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        if self.entry_point.is_none() || k == 0 {
            return Ok(Vec::new());
        }

        dispatch_distance!(self.distance_fn, D => {
            let ctx = DistanceContext::<D>::new(query, &self.storage);
            let mut visited = VisitedList::new(self.len() + 1);

            self.search_internal::<D>(query, k, ef, &ctx, &mut visited)
        })
    }

    fn search_internal<D: Distance>(
        &self,
        _query: &[f32],
        k: usize,
        ef: usize,
        ctx: &DistanceContext<D>,
        visited: &mut VisitedList,
    ) -> Result<Vec<SearchResult>> {
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

        let mut frontier = BinaryHeap::new();
        let mut candidates = BinaryHeap::new();

        frontier.push(Reverse(Candidate::new(nearest_node, nearest_dist)));
        candidates.push(Candidate::new(nearest_node, nearest_dist));
        visited.mark_visited(nearest_node);

        while let Some(Reverse(current)) = frontier.pop() {
            if candidates
                .peek()
                .is_some_and(|f| current.distance > f.distance)
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
                        if i + 2 < neighbors.len() {
                            self.storage.neighbors.prefetch(neighbors[i + 2], 0);
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
                id: c.node_id,
                distance: c.distance.into_inner(),
            });
        }
        results.reverse();
        results.truncate(k);
        Ok(results)
    }
}
