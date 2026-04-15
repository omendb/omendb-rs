//! HNSW insertion operations
//!
//! Correct HNSW algorithm with unified flat storage.

use super::HNSWIndex;
use crate::vector::hnsw::error::{HNSWError, Result};
use crate::vector::hnsw::types::{Candidate, Distance};
use tracing::instrument;

impl HNSWIndex {
    /// Insert a vector into the index.
    #[instrument(skip(self, vector))]
    pub fn insert(&mut self, vector: &[f32]) -> Result<u32> {
        if vector.len() != self.dimensions() {
            return Err(HNSWError::DimensionMismatch {
                expected: self.dimensions(),
                actual: vector.len(),
            });
        }

        let level = self.random_level();
        let node_id = self.storage.add_node(vector, level);

        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            return Ok(node_id);
        }

        self.insert_into_graph(node_id, vector, level)?;

        let ep = self.entry_point.unwrap();
        if level > self.storage.get_node_level(ep) {
            self.entry_point = Some(node_id);
        }

        Ok(node_id)
    }

    fn insert_into_graph(&mut self, node_id: u32, vector: &[f32], level: u8) -> Result<()> {
        let entry_point = self.entry_point.unwrap();
        let mut nearest_node = entry_point;

        dispatch_distance!(self.distance_fn, D => {
            let mut nearest_dist = D::distance(vector, self.storage.vector(entry_point));
            let entry_level = self.storage.get_node_level(entry_point);

            // 1. Search for nearest node at levels above the node's level
            for lc in (level + 1..=entry_level).rev() {
                let mut changed = true;
                while changed {
                    changed = false;
                    self.storage.with_neighbors(nearest_node, lc, |neighbors| {
                        for &neighbor_id in neighbors {
                            let dist = D::distance(vector, self.storage.vector(neighbor_id));
                            if dist < nearest_dist {
                                nearest_dist = dist;
                                nearest_node = neighbor_id;
                                changed = true;
                            }
                        }
                    });
                }
            }

            // 2. Insert at levels from node's level down to 0
            let mut ep = vec![nearest_node];
            for lc in (0..=level).rev() {
                let candidates = self.search_layer_for_insertion::<D>(vector, &ep, self.params.ef_construction, lc)?;
                let m = if lc == 0 { self.params.m * 2 } else { self.params.m };
                let neighbors = self.select_neighbors_heuristic::<D>(vector, &candidates, m)?;

                self.storage.neighbors.set_neighbors(node_id, lc, &neighbors);

                // Add reverse links
                for &neighbor_id in &neighbors {
                    let mut neighbor_neighbors = self.storage.neighbors.get_neighbors(neighbor_id, lc);
                    neighbor_neighbors.push(node_id);

                    if neighbor_neighbors.len() > m {
                        let neighbor_vec = self.storage.vector(neighbor_id);
                        let mut cand_vec = Vec::with_capacity(neighbor_neighbors.len());
                        for &nn_id in &neighbor_neighbors {
                            cand_vec.push(Candidate::new(nn_id, D::distance(neighbor_vec, self.storage.vector(nn_id))));
                        }
                        neighbor_neighbors = self.select_neighbors_heuristic::<D>(neighbor_vec, &cand_vec, m)?;
                    }
                    self.storage.neighbors.set_neighbors(neighbor_id, lc, &neighbor_neighbors);
                }
                ep = neighbors;
            }
        });

        Ok(())
    }

    fn search_layer_for_insertion<D: Distance>(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
    ) -> Result<Vec<Candidate>> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut visited = crate::vector::hnsw::visited::VisitedList::new(self.len() + 1);
        let mut frontier = BinaryHeap::new();
        let mut candidates = BinaryHeap::new();

        for &ep in entry_points {
            let dist = D::distance(query, self.storage.vector(ep));
            let c = Candidate::new(ep, dist);
            frontier.push(Reverse(c));
            candidates.push(c);
            visited.mark_visited(ep);
        }

        while let Some(Reverse(current)) = frontier.pop() {
            if candidates
                .peek()
                .is_some_and(|f| current.distance > f.distance)
            {
                break;
            }

            self.storage
                .with_neighbors(current.node_id, level, |neighbors| {
                    for &neighbor_id in neighbors {
                        if visited.is_visited(neighbor_id) {
                            continue;
                        }
                        visited.mark_visited(neighbor_id);

                        let dist = D::distance(query, self.storage.vector(neighbor_id));
                        if candidates.len() < ef
                            || ordered_float::OrderedFloat(dist)
                                < candidates.peek().unwrap().distance
                        {
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

        Ok(candidates.into_iter().collect())
    }

    fn select_neighbors_heuristic<D: Distance>(
        &self,
        _query: &[f32],
        candidates: &[Candidate],
        m: usize,
    ) -> Result<Vec<u32>> {
        let mut top_m = candidates.to_vec();
        top_m.sort_by_key(|c| c.distance);
        Ok(top_m.iter().take(m).map(|c| c.node_id).collect())
    }

    fn random_level(&mut self) -> u8 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut level = 0;
        while level < self.params.max_level && rng.r#gen::<f32>() < self.params.ml {
            level += 1;
        }
        level
    }
}
