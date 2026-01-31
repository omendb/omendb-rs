// Unified neighbor storage with atomic operations
//
// - Lock-free reads for search performance
// - Per-node locking for parallel construction
// - Dense level 0, sparse upper levels

use std::collections::{HashSet, VecDeque};

use super::level0::Level0Storage;
use super::upper_levels::UpperLevelStorage;

/// Unified neighbor storage with atomic operations
///
/// - Lock-free reads for search performance
/// - Per-node locking for parallel construction
/// - Dense level 0, sparse upper levels
#[derive(Debug)]
pub struct NeighborStorage {
    level0: Level0Storage,
    upper: UpperLevelStorage,
    max_m: usize,
    max_m0: usize,
    max_levels: usize,
}

impl NeighborStorage {
    pub fn new(max_levels: usize, m: usize) -> Self {
        let max_m = m;
        // Use M*32 for level 0 to handle construction overflow before pruning
        // During parallel construction, nodes can accumulate many neighbors before pruning
        // (the old unbounded Vec allowed this, so we need enough capacity to match)
        // Memory: 32*M*4 bytes/node = 2KB/node at level 0 during construction, pruned to M*2*4=128B after
        let max_m0 = m * 32;
        Self {
            level0: Level0Storage::with_capacity(0, max_m0),
            upper: UpperLevelStorage::new(max_m * 8), // Upper levels also need overflow room
            max_m,
            max_m0,
            max_levels,
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(num_nodes: usize, max_levels: usize, m: usize) -> Self {
        let max_m = m;
        let max_m0 = m * 32;
        Self {
            level0: Level0Storage::with_capacity(num_nodes, max_m0),
            upper: UpperLevelStorage::new(max_m * 8),
            max_m,
            max_m0,
            max_levels,
        }
    }

    /// Allocate storage for a new node (requires &mut self)
    pub fn allocate_node(&mut self, node_id: u32, level: u8) {
        self.level0.ensure_capacity(node_id);
        if level > 0 {
            self.upper.allocate_node(node_id, level);
        }
    }

    /// Execute closure with neighbors - LOCK-FREE
    #[inline(always)]
    pub fn with_neighbors<F, R>(&self, node_id: u32, level: u8, f: F) -> R
    where
        F: FnOnce(&[u32]) -> R,
    {
        if level == 0 {
            self.level0.with_neighbors(node_id, f)
        } else {
            self.upper.with_neighbors(node_id, level, f)
        }
    }

    /// Get neighbors as Vec (API compatibility with current GraphStorage)
    #[must_use]
    pub fn get_neighbors(&self, node_id: u32, level: u8) -> Vec<u32> {
        if level == 0 {
            self.level0.get_neighbors(node_id)
        } else {
            self.upper.get_neighbors(node_id, level)
        }
    }

    /// Set neighbors for a node at a level
    pub fn set_neighbors(&self, node_id: u32, level: u8, neighbors: Vec<u32>) {
        if level == 0 {
            self.level0.set_neighbors(node_id, &neighbors);
        } else {
            self.upper.set_neighbors(node_id, level, &neighbors);
        }
    }

    /// Set neighbors (parallel version, for use during parallel construction)
    pub fn set_neighbors_parallel(&self, node_id: u32, level: u8, neighbors: Vec<u32>) {
        self.set_neighbors(node_id, level, neighbors);
    }

    /// Add a single neighbor with locking - O(1) append
    /// Returns false if at capacity (caller should prune and retry)
    #[inline]
    pub fn add_neighbor(&self, node_id: u32, level: u8, neighbor: u32) -> bool {
        if level == 0 {
            self.level0.add_neighbor(node_id, neighbor)
        } else {
            self.upper.add_neighbor(node_id, level, neighbor)
        }
    }

    /// Add bidirectional link with deadlock prevention
    pub fn add_bidirectional_link(&self, node_a: u32, node_b: u32, level: u8) {
        if node_a == node_b {
            return;
        }

        // Lock in ascending order to prevent deadlock
        let (first, second) = if node_a < node_b {
            (node_a, node_b)
        } else {
            (node_b, node_a)
        };

        if level == 0 {
            let _lock1 = self.level0.get_write_lock(first);
            let _lock2 = self.level0.get_write_lock(second);

            if _lock1.is_some() && _lock2.is_some() {
                self.level0.add_neighbor_unlocked(first, second);
                self.level0.add_neighbor_unlocked(second, first);
            }
        } else {
            let _lock1 = self.upper.get_write_lock(first);
            let _lock2 = self.upper.get_write_lock(second);

            if _lock1.is_some() && _lock2.is_some() {
                self.upper.add_neighbor_unlocked(first, level, second);
                self.upper.add_neighbor_unlocked(second, level, first);
            }
        }
    }

    /// Alias for add_bidirectional_link (API compatibility)
    pub fn add_bidirectional_link_parallel(&self, node_a: u32, node_b: u32, level: u8) {
        self.add_bidirectional_link(node_a, node_b, level);
    }

    /// Remove unidirectional link (thread-safe for parallel construction)
    pub fn remove_link_parallel(&self, node_a: u32, node_b: u32, level: u8) {
        if level == 0 {
            self.level0.remove_neighbor(node_a, node_b);
        } else {
            self.upper.remove_neighbor(node_a, level, node_b);
        }
    }

    /// Prefetch neighbor data
    #[inline]
    pub fn prefetch(&self, node_id: u32, _level: u8) {
        // Only prefetch level 0 (hot path)
        self.level0.prefetch(node_id);
    }

    /// Get M_max (max neighbors per node at level 0 after pruning)
    /// Note: actual slot capacity is larger for construction overflow
    #[must_use]
    pub fn m_max(&self) -> usize {
        self.max_m * 2 // Return pruned M_max, not construction overflow
    }

    /// Get M (max neighbors per node at upper levels)
    #[must_use]
    pub fn m(&self) -> usize {
        self.max_m
    }

    /// Get max levels
    #[must_use]
    pub fn max_levels(&self) -> usize {
        self.max_levels
    }

    /// Check if neighbor exists at level (lock-free)
    #[inline]
    pub fn contains_neighbor(&self, node_id: u32, level: u8, neighbor: u32) -> bool {
        if level == 0 {
            self.level0.contains(node_id, neighbor)
        } else {
            self.upper
                .with_neighbors(node_id, level, |n| n.contains(&neighbor))
        }
    }

    /// Get neighbor count at level (lock-free)
    #[inline]
    pub fn neighbor_count(&self, node_id: u32, level: u8) -> usize {
        if level == 0 {
            self.level0.count(node_id)
        } else {
            self.upper.count(node_id, level)
        }
    }

    /// Get number of nodes
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.level0.num_nodes()
    }

    /// Get total memory usage
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.level0.memory_usage() + self.upper.memory_usage()
    }

    /// Reorder nodes using BFS for cache locality
    ///
    /// Returns mapping from old_id -> new_id
    pub fn reorder_bfs(&mut self, entry_point: u32, start_level: u8) -> Vec<u32> {
        let num_nodes = self.level0.num_nodes();
        if num_nodes == 0 {
            return Vec::new();
        }

        // Reverse Cuthill-McKee (RCM) ordering for better cache locality
        // 1. Start from entry point (typically high-degree hub)
        // 2. Visit neighbors sorted by degree (ascending) - low-degree first
        // 3. Reverse final ordering to place high-degree nodes at start
        //
        // This achieves ~85% of Gorder's benefit with minimal complexity.

        // Pre-compute degrees for sorting (level 0 only, where most work happens)
        let degrees: Vec<usize> = (0..num_nodes)
            .map(|i| self.level0.get_neighbors(i as u32).len())
            .collect();

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut order = Vec::with_capacity(num_nodes);

        queue.push_back(entry_point);
        visited.insert(entry_point);

        while let Some(node_id) = queue.pop_front() {
            order.push(node_id);

            // Collect unvisited neighbors from all levels
            let mut unvisited_neighbors = Vec::new();
            for level in (0..=start_level).rev() {
                let neighbors = self.get_neighbors(node_id, level);
                for neighbor_id in neighbors {
                    if visited.insert(neighbor_id) {
                        unvisited_neighbors.push(neighbor_id);
                    }
                }
            }

            // Sort by degree (ascending) - visit low-degree nodes first
            unvisited_neighbors.sort_by_key(|&id| degrees[id as usize]);

            for neighbor_id in unvisited_neighbors {
                queue.push_back(neighbor_id);
            }
        }

        // Handle disconnected nodes (shouldn't happen in well-formed HNSW)
        for node_id in 0..num_nodes as u32 {
            if !visited.contains(&node_id) {
                order.push(node_id);
            }
        }

        // RCM step 3: Reverse the order for optimal bandwidth reduction
        order.reverse();

        // Build old_to_new mapping from reversed order
        let mut old_to_new = vec![0u32; num_nodes];
        for (new_id, &old_id) in order.iter().enumerate() {
            old_to_new[old_id as usize] = new_id as u32;
        }

        // Rebuild storage with new ordering
        let mut new_level0 = Level0Storage::with_capacity(num_nodes, self.max_m0);
        let mut new_upper = UpperLevelStorage::new(self.max_m);

        // Copy data with remapped IDs
        for old_id in 0..num_nodes {
            let new_node_id = old_to_new[old_id];

            // Level 0
            let neighbors: Vec<u32> = self
                .level0
                .get_neighbors(old_id as u32)
                .into_iter()
                .map(|n| old_to_new[n as usize])
                .collect();
            new_level0.ensure_capacity(new_node_id);
            new_level0.set_neighbors(new_node_id, &neighbors);

            // Upper levels
            let max_level = self.upper.get_max_level(old_id as u32);
            if max_level > 0 {
                new_upper.allocate_node(new_node_id, max_level);
                for level in 1..=max_level {
                    let neighbors: Vec<u32> = self
                        .upper
                        .get_neighbors(old_id as u32, level)
                        .into_iter()
                        .map(|n| old_to_new[n as usize])
                        .collect();
                    new_upper.set_neighbors(new_node_id, level, &neighbors);
                }
            }
        }

        self.level0 = new_level0;
        self.upper = new_upper;

        old_to_new
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neighbor_storage_unified() {
        let mut storage = NeighborStorage::new(8, 16);

        // Allocate nodes
        storage.allocate_node(0, 2);
        storage.allocate_node(1, 0); // Level 0 only

        // Set level 0 neighbors
        storage.set_neighbors(0, 0, vec![1, 2, 3]);
        assert_eq!(storage.get_neighbors(0, 0), vec![1, 2, 3]);

        // Set upper level neighbors
        storage.set_neighbors(0, 1, vec![10]);
        assert_eq!(storage.get_neighbors(0, 1), vec![10]);

        // with_neighbors API
        let count = storage.with_neighbors(0, 0, |n| n.len());
        assert_eq!(count, 3);
    }

    #[test]
    fn test_neighbor_storage_bidirectional_link() {
        let mut storage = NeighborStorage::new(8, 16);

        storage.allocate_node(0, 0);
        storage.allocate_node(1, 0);
        storage.allocate_node(2, 0);

        // Add bidirectional link
        storage.add_bidirectional_link(0, 1, 0);

        // Both nodes should have each other
        assert!(storage.get_neighbors(0, 0).contains(&1));
        assert!(storage.get_neighbors(1, 0).contains(&0));

        // Self-link should be ignored
        storage.add_bidirectional_link(0, 0, 0);
        assert!(!storage.get_neighbors(0, 0).contains(&0));
    }

    #[test]
    fn test_neighbor_storage_remove_link() {
        let mut storage = NeighborStorage::new(8, 16);

        storage.allocate_node(0, 0);
        storage.allocate_node(1, 0);

        storage.add_bidirectional_link(0, 1, 0);
        assert!(storage.get_neighbors(0, 0).contains(&1));

        // Remove link (unidirectional)
        storage.remove_link_parallel(0, 1, 0);
        assert!(!storage.get_neighbors(0, 0).contains(&1));
        // Other direction still exists
        assert!(storage.get_neighbors(1, 0).contains(&0));
    }

    #[test]
    fn test_neighbor_storage_memory_usage() {
        let storage = NeighborStorage::with_capacity(100, 8, 16);
        let usage = storage.memory_usage();

        // Should have allocated level 0 data
        // 100 nodes * 32 neighbors * 4 bytes = 12800 bytes for data
        // Plus counts, locks, etc.
        assert!(usage > 12000);
    }
}
