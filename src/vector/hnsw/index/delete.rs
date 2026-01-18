//! HNSW deletion operations
//!
//! Uses lazy delete pattern: mark as deleted, filter during search, rebuild on compact.
//! This is O(1) per deletion and matches production systems (hnswlib, Qdrant, Milvus).
//!
//! The graph structure is NOT modified on delete. Deleted nodes remain in the graph
//! so search can traverse through them to reach live nodes. Results are filtered
//! at the VectorStore level.
//!
//! For MN-RU graph repair (optional), see `mark_deleted_with_repair`.

use super::HNSWIndex;
use crate::vector::hnsw::error::Result;
use std::collections::HashSet;
use tracing::{debug, instrument};

impl HNSWIndex {
    /// Mark a node as deleted (O(1) lazy delete)
    ///
    /// This method does NOT modify the graph structure. The deleted node's neighbors
    /// remain intact so search can traverse through deleted nodes to reach live ones.
    /// Deleted nodes are filtered at the VectorStore level during search.
    ///
    /// # Arguments
    /// * `node_id` - The node ID to mark as deleted
    ///
    /// # Returns
    /// Always returns 0 (no graph repairs performed)
    ///
    /// # Performance
    /// O(1) - only updates entry point if necessary
    #[instrument(skip(self), fields(node_id = node_id))]
    pub fn mark_deleted(&mut self, node_id: u32) -> Result<usize> {
        if (node_id as usize) >= self.storage.len() {
            debug!(node_id, "Node not found, skipping deletion");
            return Ok(0);
        }

        // Lazy delete: graph stays intact, deleted filtering happens at VectorStore level
        // Only update entry point if we're deleting it
        if self.entry_point == Some(node_id) {
            self.update_entry_point_after_delete(node_id);
        }

        debug!(node_id, "Lazy delete complete (graph unchanged)");
        Ok(0)
    }

    /// Mark a node as deleted WITH graph repair using MN-RU algorithm
    ///
    /// Use this for explicit graph maintenance. Most applications should use
    /// `mark_deleted` (lazy delete) instead and rely on compaction for cleanup.
    ///
    /// # Performance
    /// O(M² · L) per deletion where M = max neighbors, L = max level
    #[instrument(skip(self), fields(node_id = node_id))]
    pub fn mark_deleted_with_repair(&mut self, node_id: u32) -> Result<usize> {
        if (node_id as usize) >= self.storage.len() {
            debug!(node_id, "Node not found, skipping deletion");
            return Ok(0);
        }

        let max_level = self.params.max_level;
        let mut total_repairs = 0;

        // Repair at each level (from top to bottom)
        for lc in (0..max_level).rev() {
            let repairs = self.repair_level_mnru(node_id, lc)?;
            total_repairs += repairs;
        }

        // Update entry point if we're deleting it
        if self.entry_point == Some(node_id) {
            self.update_entry_point_after_delete(node_id);
        }

        debug!(
            node_id,
            max_level,
            repairs = total_repairs,
            "MN-RU deletion repair complete"
        );

        Ok(total_repairs)
    }

    /// Repair graph at a specific level using MN-RU algorithm
    ///
    /// Returns the number of replacement edges added.
    fn repair_level_mnru(&mut self, deleted_id: u32, level: u8) -> Result<usize> {
        // Optimization: Instead of scanning ALL N nodes (O(N)), we check only the neighbors
        // of the deleted node (O(M)). In HNSW, edges are predominantly bidirectional.
        // If node A points to deleted node D, it is highly likely that D also points to A
        // unless pruning occurred. Even if we miss a few unidirectional edges, the
        // search logic handles deleted nodes gracefully, and graph connectivity remains high.
        let deleted_neighbors = self.storage.neighbors_at_level(deleted_id, level);
        let mut nodes_with_edge_to_deleted: Vec<u32> = Vec::new();

        for neighbor_id in &deleted_neighbors {
            let n_neighbors = self.storage.neighbors_at_level(*neighbor_id, level);
            if n_neighbors.contains(&deleted_id) {
                nodes_with_edge_to_deleted.push(*neighbor_id);
            }
        }

        let deleted_neighbor_set: HashSet<u32> = deleted_neighbors.iter().copied().collect();

        let mut repairs = 0;
        let m = self.params.m_for_level(level);

        // For each node that has an edge to the deleted node
        for node_id in &nodes_with_edge_to_deleted {
            // Get current neighbors of this node
            let mut node_edges: Vec<u32> = self.storage.neighbors_at_level(*node_id, level);

            // Remove edge to deleted node
            let original_len = node_edges.len();
            node_edges.retain(|&n| n != deleted_id);

            if node_edges.len() == original_len {
                continue; // Edge was already removed somehow
            }

            // Find mutual neighbors: nodes that are neighbors of deleted AND could be good replacements
            // We look for nodes in deleted's neighbor list that aren't already neighbors of current node
            let node_edge_set: HashSet<u32> = node_edges.iter().copied().collect();
            let candidates: Vec<u32> = deleted_neighbor_set
                .iter()
                .filter(|&&n| n != *node_id && !node_edge_set.contains(&n))
                .copied()
                .collect();

            // Find best replacement from candidates
            if let Some(replacement) = self.find_best_replacement(*node_id, &candidates)? {
                // Only add if we're under the neighbor limit
                if node_edges.len() < m {
                    node_edges.push(replacement);
                    repairs += 1;
                }
            }

            // Update the neighbor list
            self.storage
                .set_neighbors_at_level(*node_id, level, node_edges);
        }

        // Clear the deleted node's neighbor lists at this level
        self.storage
            .set_neighbors_at_level(deleted_id, level, Vec::new());

        Ok(repairs)
    }

    /// Find the best replacement edge from candidates
    ///
    /// Returns the candidate that is closest to the source node.
    fn find_best_replacement(&self, source_id: u32, candidates: &[u32]) -> Result<Option<u32>> {
        if candidates.is_empty() {
            return Ok(None);
        }

        // Get source vector (handle both f32 and quantized storage)
        let source_vec = match self.storage.get_dequantized(source_id) {
            Some(v) => v,
            None => return Ok(None),
        };

        // Find closest candidate
        let mut best: Option<(u32, f32)> = None;

        for &candidate_id in candidates {
            let dist = self.distance_cmp(&source_vec, candidate_id)?;

            match best {
                None => best = Some((candidate_id, dist)),
                Some((_, best_dist)) if dist < best_dist => best = Some((candidate_id, dist)),
                _ => {}
            }
        }

        Ok(best.map(|(id, _)| id))
    }

    /// Update entry point after deleting the current entry point
    fn update_entry_point_after_delete(&mut self, deleted_id: u32) {
        // Find the highest-level node that isn't deleted
        // Prefer nodes with neighbors, but fall back to any remaining node
        let mut best_connected: Option<(u32, u8)> = None;
        let mut best_fallback: Option<(u32, u8)> = None;

        for idx in 0..self.storage.len() {
            let node_id = idx as u32;
            if node_id == deleted_id {
                continue;
            }

            let level = self.storage.level(node_id);

            // Track as fallback (any remaining node)
            match best_fallback {
                None => best_fallback = Some((node_id, level)),
                Some((_, best_level)) if level > best_level => {
                    best_fallback = Some((node_id, level));
                }
                _ => {}
            }

            // Check if this node has any neighbors (connected)
            let has_neighbors =
                (0..=level).any(|lc| !self.storage.neighbors_at_level(node_id, lc).is_empty());

            if has_neighbors {
                match best_connected {
                    None => best_connected = Some((node_id, level)),
                    Some((_, best_level)) if level > best_level => {
                        best_connected = Some((node_id, level));
                    }
                    _ => {}
                }
            }
        }

        // Prefer connected node, fall back to any remaining node
        self.entry_point = best_connected.or(best_fallback).map(|(id, _)| id);
        debug!(new_entry = ?self.entry_point, "Updated entry point after deletion");
    }

    /// Batch mark multiple nodes as deleted (O(n) where n = node_ids.len())
    ///
    /// Uses lazy delete pattern - graph structure unchanged.
    ///
    /// # Arguments
    /// * `node_ids` - Node IDs to delete
    ///
    /// # Returns
    /// Always returns 0 (no graph repairs performed)
    #[instrument(skip(self, node_ids), fields(count = node_ids.len()))]
    pub fn mark_deleted_batch(&mut self, node_ids: &[u32]) -> Result<usize> {
        for &node_id in node_ids {
            self.mark_deleted(node_id)?;
        }

        debug!(count = node_ids.len(), "Batch lazy delete complete");
        Ok(0)
    }

    /// Batch mark multiple nodes as deleted WITH graph repair
    ///
    /// Use this for explicit graph maintenance. Most applications should use
    /// `mark_deleted_batch` (lazy delete) instead.
    #[instrument(skip(self, node_ids), fields(count = node_ids.len()))]
    pub fn mark_deleted_batch_with_repair(&mut self, node_ids: &[u32]) -> Result<usize> {
        let mut total_repairs = 0;

        // Sort by level descending to handle higher-level nodes first
        let mut sorted_ids: Vec<u32> = node_ids.to_vec();
        sorted_ids.sort_unstable_by_key(|&id| {
            let idx = id as usize;
            if idx < self.storage.len() {
                std::cmp::Reverse(self.storage.level(id))
            } else {
                std::cmp::Reverse(0)
            }
        });

        for node_id in sorted_ids {
            let repairs = self.mark_deleted_with_repair(node_id)?;
            total_repairs += repairs;
        }

        debug!(
            count = node_ids.len(),
            repairs = total_repairs,
            "Batch deletion with repair complete"
        );

        Ok(total_repairs)
    }

    /// Check if a node is effectively deleted (has no neighbors)
    #[must_use]
    pub fn is_orphaned(&self, node_id: u32) -> bool {
        if (node_id as usize) >= self.storage.len() {
            return true;
        }

        let level = self.storage.level(node_id);
        (0..=level).all(|lc| self.storage.neighbors_at_level(node_id, lc).is_empty())
    }

    /// Count orphaned nodes (nodes with no neighbors)
    ///
    /// Useful for monitoring graph health after deletions.
    #[must_use]
    pub fn count_orphaned(&self) -> usize {
        (0..self.storage.len() as u32)
            .filter(|&id| self.is_orphaned(id))
            .count()
    }

    /// Validate graph connectivity after deletions
    ///
    /// Returns (reachable_count, orphan_count).
    /// A healthy graph should have reachable_count ≈ total_nodes - deleted_count.
    #[must_use]
    pub fn validate_connectivity(&self) -> (usize, usize) {
        self.validate_connectivity_verbose(false)
    }

    /// Validate connectivity with optional verbose output for debugging
    #[must_use]
    pub fn validate_connectivity_verbose(&self, verbose: bool) -> (usize, usize) {
        use std::collections::VecDeque;

        let entry_point = match self.entry_point {
            Some(ep) => ep,
            None => return (0, self.storage.len()),
        };

        // BFS from entry point
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        visited.insert(entry_point);
        queue.push_back(entry_point);

        while let Some(node_id) = queue.pop_front() {
            let level = self.storage.level(node_id);

            // Visit neighbors at all levels
            for lc in 0..=level {
                for neighbor_id in self.storage.neighbors_at_level(node_id, lc) {
                    if visited.insert(neighbor_id) {
                        if verbose {
                            println!("  BFS: node {node_id} level {lc} -> neighbor {neighbor_id}");
                        }
                        queue.push_back(neighbor_id);
                    }
                }
            }
        }

        let reachable = visited.len();
        let orphans = self.storage.len() - reachable;

        if verbose {
            println!("  BFS visited: {visited:?}");
        }

        (reachable, orphans)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::hnsw::types::{DistanceFunction, HNSWParams};

    const TEST_EF_SEARCH: usize = 50;

    fn create_test_index(num_vectors: usize, dimensions: usize) -> HNSWIndex {
        let params = HNSWParams {
            m: 8,
            ef_construction: 50,
            ..Default::default()
        };
        let mut index = HNSWIndex::new(dimensions, params, DistanceFunction::L2, false).unwrap();

        // Insert random vectors
        for i in 0..num_vectors {
            let vector: Vec<f32> = (0..dimensions)
                .map(|d| ((i * 7 + d * 11) % 100) as f32 / 100.0)
                .collect();
            index.insert(&vector).unwrap();
        }

        index
    }

    #[test]
    fn test_lazy_delete_preserves_graph() {
        let mut index = create_test_index(100, 16);

        // Get initial connectivity
        let (initial_reachable, initial_orphans) = index.validate_connectivity();
        assert_eq!(initial_reachable, 100);
        assert_eq!(initial_orphans, 0);

        // Delete a node - lazy delete keeps graph intact
        let repairs = index.mark_deleted(50).unwrap();
        assert_eq!(repairs, 0, "Lazy delete should not repair");

        // Graph structure unchanged - deleted node still has neighbors
        assert!(
            !index.is_orphaned(50),
            "Lazy delete should NOT orphan nodes"
        );

        // All nodes still reachable (including deleted one)
        let (reachable, orphans) = index.validate_connectivity();
        assert_eq!(reachable, 100, "Graph should be fully connected");
        assert_eq!(orphans, 0);
    }

    #[test]
    fn test_lazy_delete_batch() {
        let mut index = create_test_index(200, 32);

        // Delete 10% of nodes
        let delete_ids: Vec<u32> = (0..200).step_by(10).collect();
        let repairs = index.mark_deleted_batch(&delete_ids).unwrap();
        assert_eq!(repairs, 0, "Lazy delete should not repair");

        // Deleted nodes should NOT be orphaned (graph intact)
        for &id in &delete_ids {
            assert!(
                !index.is_orphaned(id),
                "Node {id} should NOT be orphaned (lazy delete)"
            );
        }

        // Graph should be fully connected
        let (reachable, _) = index.validate_connectivity();
        assert_eq!(reachable, 200, "All nodes should still be reachable");
    }

    #[test]
    fn test_entry_point_update() {
        let mut index = create_test_index(50, 8);

        // Get current entry point
        let entry_point = index.entry_point().unwrap();

        // Delete entry point - should still update entry point
        let repairs = index.mark_deleted(entry_point).unwrap();
        assert_eq!(repairs, 0);

        // Entry point should be updated to a different node
        let new_entry = index.entry_point();
        assert!(new_entry.is_some());
        assert_ne!(new_entry.unwrap(), entry_point);
    }

    #[test]
    fn test_search_returns_deleted_nodes() {
        // With lazy delete, HNSW search returns deleted nodes.
        // Filtering happens at VectorStore level, not HNSW level.
        let mut index = create_test_index(500, 64);

        // Search before deletion
        let query: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        let results_before = index.search(&query, 10, TEST_EF_SEARCH).unwrap();

        // Delete some nodes
        let delete_ids: Vec<u32> = (0..500).step_by(20).collect();
        index.mark_deleted_batch(&delete_ids).unwrap();

        // Search after deletion - results may include deleted nodes
        // (VectorStore is responsible for filtering)
        let results_after = index.search(&query, 10, TEST_EF_SEARCH).unwrap();

        // Should still get results
        assert!(!results_after.is_empty());
        println!(
            "Results before: {}, after: {}",
            results_before.len(),
            results_after.len()
        );
    }

    #[test]
    fn test_repair_variant_orphans_nodes() {
        // The _with_repair variant should still work for those who want it
        let mut index = create_test_index(100, 16);

        // Delete with repair
        let repairs = index.mark_deleted_with_repair(50).unwrap();
        println!("Repairs with MN-RU: {repairs}");

        // Node should be orphaned after repair
        assert!(
            index.is_orphaned(50),
            "Node should be orphaned after repair"
        );

        // Graph should still be mostly connected
        let (reachable, _) = index.validate_connectivity();
        assert!(reachable >= 90, "Most nodes should still be reachable");
    }
}

#[cfg(test)]
mod small_graph_tests {
    use super::*;
    use crate::vector::hnsw::types::{DistanceFunction, HNSWParams};

    #[test]
    fn test_small_graph_lazy_delete() {
        // Test lazy delete behavior on small graph
        let params = HNSWParams {
            m: 16,
            ef_construction: 100,
            ..Default::default()
        };
        let mut index = HNSWIndex::new(128, params, DistanceFunction::L2, false).unwrap();

        // Insert 5 uniform vectors
        for i in 0..5 {
            let val = (i + 1) as f32 * 0.1;
            let vector: Vec<f32> = vec![val; 128];
            let id = index.insert(&vector).unwrap();
            println!("Inserted node {id} with value {val}");
        }

        println!("\n=== Before deletion ===");
        println!("Entry point: {:?}", index.entry_point());

        // Print neighbors for each node BEFORE deletion
        println!("Graph structure:");
        for node_id in 0..5u32 {
            let neighbors = index.get_neighbors_level0(node_id);
            println!("  Node {node_id} -> {neighbors:?}");
        }

        let (reachable_before, _) = index.validate_connectivity();
        assert_eq!(reachable_before, 5);

        // Delete node 0 with lazy delete
        println!("\n=== Deleting node 0 (lazy) ===");
        let repairs = index.mark_deleted(0).unwrap();
        assert_eq!(repairs, 0, "Lazy delete should not repair");

        println!("\n=== After deletion ===");
        println!("Entry point: {:?}", index.entry_point());

        // Graph structure should be UNCHANGED with lazy delete
        println!("Graph structure (should be unchanged):");
        for node_id in 0..5u32 {
            let neighbors = index.get_neighbors_level0(node_id);
            println!("  Node {node_id} -> {neighbors:?}");
        }

        // Node 0 should NOT be orphaned (lazy delete keeps graph intact)
        assert!(!index.is_orphaned(0), "Lazy delete should NOT orphan nodes");

        // All nodes still reachable
        let (reachable_after, orphans) = index.validate_connectivity();
        assert_eq!(reachable_after, 5, "All nodes still reachable");
        assert_eq!(orphans, 0);

        // Search still returns node 0 - filtering happens at VectorStore level
        let query: Vec<f32> = vec![0.1; 128];
        let results = index.search(&query, 5, 100).unwrap();
        assert!(!results.is_empty());

        // At HNSW level, node 0 is still in results (lazy delete)
        let has_node_0 = results.iter().any(|r| r.id == 0);
        assert!(
            has_node_0,
            "HNSW should still return deleted node (filtering happens at VectorStore)"
        );
    }
}
