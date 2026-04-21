//! HNSW insertion operations
//!
//! Correct HNSW algorithm with unified flat storage.

use super::HNSWIndex;
use crate::vector::hnsw::error::{HNSWError, Result};
use crate::vector::hnsw::types::{Candidate, Distance};

impl HNSWIndex {
    /// Insert a vector into the index.
    pub fn insert(&mut self, vector: &[f32]) -> Result<u32> {
        self.validate_vector(vector)?;

        let level = self.random_level();
        let node_id = self.storage.add_node(vector, level);

        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            return Ok(node_id);
        }

        let entry_point = self.entry_point.unwrap();
        self.insert_into_graph_from_entry(
            node_id,
            vector,
            level,
            entry_point,
            self.params.ef_construction,
        )?;

        if level > self.storage.get_node_level(entry_point) {
            self.entry_point = Some(node_id);
        }

        Ok(node_id)
    }

    pub(super) fn validate_vector(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.storage.vectors.dim {
            return Err(HNSWError::DimensionMismatch {
                expected: self.storage.vectors.dim,
                actual: vector.len(),
            });
        }

        for &val in vector {
            if !val.is_finite() {
                return Err(HNSWError::InvalidVector);
            }
        }

        Ok(())
    }

    pub(super) fn insert_into_graph_from_entry(
        &self,
        node_id: u32,
        vector: &[f32],
        level: u8,
        entry_point: u32,
        ef_construction: usize,
    ) -> Result<()> {
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
                let candidates = self.search_layer_for_insertion::<D>(vector, &ep, ef_construction, lc)?;
                let m = if lc == 0 { self.params.m * 2 } else { self.params.m };
                let neighbors = self.select_neighbors_heuristic::<D>(&candidates, m);

                self.storage.neighbors.set_neighbors(node_id, lc, &neighbors);

                // Add reverse links
                for &neighbor_id in &neighbors {
                    self.storage.neighbors.update_neighbors(neighbor_id, lc, |neighbor_neighbors, len| {
                        if !neighbor_neighbors[..*len].contains(&node_id)
                            && *len < neighbor_neighbors.len()
                        {
                            neighbor_neighbors[*len] = node_id;
                            *len += 1;
                        }

                        if *len <= m {
                            return;
                        }

                        let neighbor_vec = self.storage.vector(neighbor_id);
                        let mut cand_vec = Vec::with_capacity(*len);
                        for &nn_id in &neighbor_neighbors[..*len] {
                            cand_vec.push(Candidate::new(
                                nn_id,
                                D::distance(neighbor_vec, self.storage.vector(nn_id)),
                            ));
                        }
                        let selected = self.select_neighbors_heuristic::<D>(&cand_vec, m);
                        for (dst, selected_id) in
                            neighbor_neighbors.iter_mut().zip(selected.iter())
                        {
                            *dst = *selected_id;
                        }
                        *len = selected.len();
                    });
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

        crate::vector::hnsw::visited::with_visited_list(self.len() + 1, |visited| {
            let mut frontier = BinaryHeap::with_capacity(ef);
            let mut candidates = BinaryHeap::with_capacity(ef);

            for &ep in entry_points {
                let dist = D::distance(query, self.storage.vector(ep));
                let c = Candidate::new(ep, dist);
                frontier.push(Reverse(c));
                candidates.push(c);
                visited.mark_visited(ep);
            }

            while let Some(Reverse(current)) = frontier.pop() {
                if candidates.len() >= ef && current.distance > candidates.peek().unwrap().distance
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
        })
    }

    pub(super) fn select_neighbors_heuristic<D: Distance>(
        &self,
        candidates: &[Candidate],
        m: usize,
    ) -> Vec<u32> {
        if candidates.len() <= m {
            let mut nearest = candidates.to_vec();
            nearest.sort_unstable_by_key(|c| c.distance);
            nearest.dedup_by_key(|c| c.node_id);
            return nearest.into_iter().map(|c| c.node_id).collect();
        }

        let mut nearest = candidates.to_vec();
        nearest.sort_unstable_by_key(|c| c.distance);

        let mut selected = Vec::with_capacity(m);
        for candidate in nearest {
            if selected.contains(&candidate.node_id) {
                continue;
            }

            let candidate_vec = self.storage.vector(candidate.node_id);
            let closer_to_selected = selected.iter().any(|&selected_id| {
                D::distance(candidate_vec, self.storage.vector(selected_id))
                    < candidate.distance.into_inner()
            });

            if !closer_to_selected {
                selected.push(candidate.node_id);
                if selected.len() == m {
                    break;
                }
            }
        }

        selected
    }

    pub fn random_level(&mut self) -> u8 {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(self.rng_state);
        let mut level = 0;
        // Correct HNSW random level: while rand < p, level++
        // ml = 1 / ln(1/p). For p=1/M, ml = 1/ln(M).
        // Standard impl: level = -ln(uniform) * ml
        let f: f32 = rng.random();
        if f > 0.0 {
            let l = (-f.ln() * self.params.ml) as u8;
            level = l.min(self.params.max_level);
        }

        self.rng_state = rng.random();
        level
    }
}
