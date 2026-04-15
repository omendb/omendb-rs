//! Generation-based bitset for fast visited list tracking.
//!
//! Provides O(1) constant-time lookup and generation-based reset.
//! This is a key component for SOTA HNSW performance.

/// A visited list that uses a generation counter to avoid clearing the entire bitset.
#[derive(Debug)]
pub struct VisitedList {
    /// Generation for each node ID
    visited: Vec<u32>,
    /// Current generation
    generation: u32,
}

impl VisitedList {
    /// Create a new visited list for the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            visited: vec![0; capacity],
            generation: 1,
        }
    }

    /// Check if a node has been visited in the current generation.
    #[inline(always)]
    pub fn is_visited(&self, node_id: u32) -> bool {
        let node_id = node_id as usize;
        if node_id >= self.visited.len() {
            return false;
        }
        self.visited[node_id] == self.generation
    }

    /// Mark a node as visited in the current generation.
    #[inline(always)]
    pub fn mark_visited(&mut self, node_id: u32) {
        let node_id = node_id as usize;
        if node_id >= self.visited.len() {
            self.visited.resize(node_id + 1, 0);
        }
        self.visited[node_id] = self.generation;
    }

    /// Reset the visited list by incrementing the generation counter.
    ///
    /// If the generation counter overflows, the entire bitset is cleared.
    pub fn next_generation(&mut self) {
        if self.generation == u32::MAX {
            self.visited.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }

    /// Ensure the list has enough capacity for the given number of nodes.
    pub fn ensure_capacity(&mut self, capacity: usize) {
        if capacity > self.visited.len() {
            self.visited.resize(capacity, 0);
        }
    }
}
