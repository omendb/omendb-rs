// Lock-free neighbor list storage for HNSW
//
// Thread-safety:
// - Reads: Lock-free via `ArcSwap` (just atomic load)
// - Writes: Mutex-protected copy-on-write for thread-safety

use arc_swap::ArcSwap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Empty neighbor list constant (avoid allocation for empty results)
static EMPTY_NEIGHBORS: &[u32] = &[];

/// Storage for neighbor lists (lock-free reads, thread-safe writes)
///
/// Neighbors are stored separately from nodes to improve cache utilization.
/// Only fetch neighbors when traversing the graph.
///
/// Thread-safety:
/// - Reads: Lock-free via `ArcSwap` (just atomic load)
/// - Writes: Mutex-protected copy-on-write for thread-safety
///
/// Performance: Search is read-heavy, construction is write-heavy.
/// Lock-free reads give ~40% speedup on high-dimension searches.
#[derive(Debug)]
pub struct NeighborLists {
    /// Neighbor storage: neighbors[`node_id`][level] = `ArcSwap`<Box<[u32]>>
    ///
    /// `ArcSwap` enables:
    /// - Lock-free reads during search (just atomic load + deref)
    /// - Thread-safe writes via copy-on-write
    neighbors: Vec<Vec<ArcSwap<Box<[u32]>>>>,

    /// Write locks for coordinating concurrent edge additions
    /// One mutex per node-level pair to minimize contention
    write_locks: Vec<Vec<Mutex<()>>>,

    /// Maximum levels supported
    max_levels: usize,

    /// `M_max` (max neighbors = M * 2)
    /// Used for pre-allocating neighbor lists to reduce reallocations
    m_max: usize,
}

impl NeighborLists {
    /// Create empty neighbor lists
    #[must_use]
    pub fn new(max_levels: usize) -> Self {
        Self {
            neighbors: Vec::new(),
            write_locks: Vec::new(),
            max_levels,
            m_max: 32, // Default M*2
        }
    }

    /// Create with pre-allocated capacity and M parameter
    #[must_use]
    pub fn with_capacity(num_nodes: usize, max_levels: usize, m: usize) -> Self {
        Self {
            neighbors: Vec::with_capacity(num_nodes),
            write_locks: Vec::with_capacity(num_nodes),
            max_levels,
            m_max: m * 2,
        }
    }

    /// Get `M_max` (max neighbors)
    #[must_use]
    pub fn m_max(&self) -> usize {
        self.m_max
    }

    /// Get number of nodes with neighbor lists
    #[must_use]
    pub fn len(&self) -> usize {
        self.neighbors.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.neighbors.is_empty()
    }

    /// Get max levels
    #[must_use]
    pub fn max_levels(&self) -> usize {
        self.max_levels
    }

    /// Get neighbors for a node at a specific level (lock-free)
    ///
    /// Returns a cloned Vec. For iteration without allocation, use `with_neighbors`.
    #[must_use]
    pub fn get_neighbors(&self, node_id: u32, level: u8) -> Vec<u32> {
        let node_idx = node_id as usize;
        let level_idx = level as usize;

        if node_idx >= self.neighbors.len() {
            return Vec::new();
        }

        if level_idx >= self.neighbors[node_idx].len() {
            return Vec::new();
        }

        // Lock-free read: just atomic load
        self.neighbors[node_idx][level_idx].load().to_vec()
    }

    /// Execute a closure with read access to neighbors (LOCK-FREE, zero-copy)
    ///
    /// This is the hot path for search - just an atomic load, no locking.
    /// ~40% faster than `RwLock` at high dimensions (1536D+).
    #[inline]
    pub fn with_neighbors<F, R>(&self, node_id: u32, level: u8, f: F) -> R
    where
        F: FnOnce(&[u32]) -> R,
    {
        let node_idx = node_id as usize;
        let level_idx = level as usize;

        if node_idx >= self.neighbors.len() {
            return f(EMPTY_NEIGHBORS);
        }

        if level_idx >= self.neighbors[node_idx].len() {
            return f(EMPTY_NEIGHBORS);
        }

        // LOCK-FREE: ArcSwap.load() is just an atomic load
        // The Guard keeps the Arc alive during the closure
        let guard = self.neighbors[node_idx][level_idx].load();
        f(&guard)
    }

    /// Prefetch neighbor list into CPU cache
    ///
    /// Hints to CPU that we'll need the neighbor data soon. This hides memory
    /// latency by overlapping data fetch with computation. Only beneficial on
    /// x86/ARM servers - Apple Silicon's DMP handles this automatically.
    #[inline]
    pub fn prefetch(&self, node_id: u32, level: u8) {
        use crate::vector::hnsw::prefetch::PrefetchConfig;
        if !PrefetchConfig::enabled() {
            return;
        }

        let node_idx = node_id as usize;
        let level_idx = level as usize;

        if node_idx >= self.neighbors.len() {
            return;
        }
        if level_idx >= self.neighbors[node_idx].len() {
            return;
        }

        // Prefetch the ArcSwap pointer (brings neighbor array address into cache)
        // This is a lightweight hint - the actual neighbor data follows
        let ptr = &self.neighbors[node_idx][level_idx] as *const _ as *const u8;
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::_mm_prefetch;
            use std::arch::x86_64::_MM_HINT_T0;
            _mm_prefetch(ptr.cast(), _MM_HINT_T0);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!(
                "prfm pldl1keep, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, preserves_flags)
            );
        }
    }

    /// Allocate storage for a new node (internal helper)
    fn ensure_node_exists(&mut self, node_idx: usize) {
        while self.neighbors.len() <= node_idx {
            let mut levels = Vec::with_capacity(self.max_levels);
            let mut locks = Vec::with_capacity(self.max_levels);
            for _ in 0..self.max_levels {
                // Start with empty boxed slice (no allocation for empty)
                levels.push(ArcSwap::from_pointee(Vec::new().into_boxed_slice()));
                locks.push(Mutex::new(()));
            }
            self.neighbors.push(levels);
            self.write_locks.push(locks);
        }
    }

    /// Set neighbors for a node at a specific level
    pub fn set_neighbors(&mut self, node_id: u32, level: u8, neighbors_list: Vec<u32>) {
        let node_idx = node_id as usize;
        let level_idx = level as usize;

        self.ensure_node_exists(node_idx);

        // Direct store - no lock needed since we have &mut self
        self.neighbors[node_idx][level_idx].store(Arc::new(neighbors_list.into_boxed_slice()));
    }

    /// Add a bidirectional link between two nodes at a level
    ///
    /// Thread-safe with deadlock prevention via ordered locking.
    /// Uses copy-on-write for lock-free reads during search.
    pub fn add_bidirectional_link(&mut self, node_a: u32, node_b: u32, level: u8) {
        let node_a_idx = node_a as usize;
        let node_b_idx = node_b as usize;
        let level_idx = level as usize;

        if node_a_idx == node_b_idx {
            return; // Same node - skip
        }

        // Ensure we have enough nodes
        let max_idx = node_a_idx.max(node_b_idx);
        self.ensure_node_exists(max_idx);

        // Add node_b to node_a's neighbors (copy-on-write)
        {
            let current = self.neighbors[node_a_idx][level_idx].load();
            if !current.contains(&node_b) {
                let mut new_list = current.to_vec();
                new_list.push(node_b);
                self.neighbors[node_a_idx][level_idx].store(Arc::new(new_list.into_boxed_slice()));
            }
        }

        // Add node_a to node_b's neighbors (copy-on-write)
        {
            let current = self.neighbors[node_b_idx][level_idx].load();
            if !current.contains(&node_a) {
                let mut new_list = current.to_vec();
                new_list.push(node_a);
                self.neighbors[node_b_idx][level_idx].store(Arc::new(new_list.into_boxed_slice()));
            }
        }
    }

    /// Add bidirectional link (thread-safe version for parallel construction)
    ///
    /// Assumes nodes are already allocated. Uses mutex + copy-on-write.
    /// Only for use during parallel graph construction where all nodes pre-exist.
    pub fn add_bidirectional_link_parallel(&self, node_a: u32, node_b: u32, level: u8) {
        let node_a_idx = node_a as usize;
        let node_b_idx = node_b as usize;
        let level_idx = level as usize;

        if node_a_idx == node_b_idx {
            return; // Same node - skip
        }

        // Bounds check
        if node_a_idx >= self.neighbors.len() || node_b_idx >= self.neighbors.len() {
            return; // Skip invalid nodes
        }

        // Deadlock prevention: always lock in ascending node_id order
        let (first_idx, second_idx, first_neighbor, second_neighbor) = if node_a_idx < node_b_idx {
            (node_a_idx, node_b_idx, node_b, node_a)
        } else {
            (node_b_idx, node_a_idx, node_a, node_b)
        };

        // Lock both nodes' write locks in order
        let _lock_first = self.write_locks[first_idx][level_idx].lock();
        let _lock_second = self.write_locks[second_idx][level_idx].lock();

        // Copy-on-write for first node
        {
            let current = self.neighbors[first_idx][level_idx].load();
            if !current.contains(&first_neighbor) {
                let mut new_list = current.to_vec();
                new_list.push(first_neighbor);
                self.neighbors[first_idx][level_idx].store(Arc::new(new_list.into_boxed_slice()));
            }
        }

        // Copy-on-write for second node
        {
            let current = self.neighbors[second_idx][level_idx].load();
            if !current.contains(&second_neighbor) {
                let mut new_list = current.to_vec();
                new_list.push(second_neighbor);
                self.neighbors[second_idx][level_idx].store(Arc::new(new_list.into_boxed_slice()));
            }
        }
    }

    /// Remove unidirectional link (thread-safe version for parallel construction)
    ///
    /// Removes link from `node_a` to `node_b` (NOT bidirectional).
    /// Uses mutex + copy-on-write for thread-safety.
    pub fn remove_link_parallel(&self, node_a: u32, node_b: u32, level: u8) {
        let node_a_idx = node_a as usize;
        let level_idx = level as usize;

        // Bounds check
        if node_a_idx >= self.neighbors.len() {
            return; // Skip invalid node
        }

        // Lock and copy-on-write
        let _lock = self.write_locks[node_a_idx][level_idx].lock();
        let current = self.neighbors[node_a_idx][level_idx].load();
        let new_list: Vec<u32> = current.iter().copied().filter(|&n| n != node_b).collect();
        self.neighbors[node_a_idx][level_idx].store(Arc::new(new_list.into_boxed_slice()));
    }

    /// Set neighbors (thread-safe version for parallel construction)
    ///
    /// Assumes node is already allocated. Uses mutex for thread-safety.
    pub fn set_neighbors_parallel(&self, node_id: u32, level: u8, neighbors_list: Vec<u32>) {
        let node_idx = node_id as usize;
        let level_idx = level as usize;

        // Bounds check
        if node_idx >= self.neighbors.len() {
            return; // Skip invalid node
        }

        // Lock and store
        let _lock = self.write_locks[node_idx][level_idx].lock();
        self.neighbors[node_idx][level_idx].store(Arc::new(neighbors_list.into_boxed_slice()));
    }

    /// Get memory usage in bytes (approximate)
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let mut total = 0;

        // Size of outer Vec
        total += self.neighbors.capacity() * std::mem::size_of::<Vec<ArcSwap<Box<[u32]>>>>();

        // Size of each node's level vecs
        for node in &self.neighbors {
            total += node.capacity() * std::mem::size_of::<ArcSwap<Box<[u32]>>>();

            // Size of actual neighbor data (lock-free read)
            for level in node {
                let guard = level.load();
                total += guard.len() * std::mem::size_of::<u32>();
            }
        }

        // Size of write locks
        total += self.write_locks.capacity() * std::mem::size_of::<Vec<Mutex<()>>>();
        for node in &self.write_locks {
            total += node.capacity() * std::mem::size_of::<Mutex<()>>();
        }

        total
    }

    /// Reorder nodes using BFS for cache locality
    ///
    /// This improves cache performance by placing frequently-accessed neighbors
    /// close together in memory. Uses BFS from the entry point to determine ordering.
    ///
    /// Returns a mapping from `old_id` -> `new_id`
    pub fn reorder_bfs(&mut self, entry_point: u32, start_level: u8) -> Vec<u32> {
        use std::collections::{HashSet, VecDeque};

        let num_nodes = self.neighbors.len();
        if num_nodes == 0 {
            return Vec::new();
        }

        // BFS to determine new ordering
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut old_to_new = vec![u32::MAX; num_nodes]; // u32::MAX = not visited
        let mut new_id = 0u32;

        // Start BFS from entry point
        queue.push_back(entry_point);
        visited.insert(entry_point);

        while let Some(node_id) = queue.pop_front() {
            // Assign new ID
            old_to_new[node_id as usize] = new_id;
            new_id += 1;

            // Visit neighbors at all levels (starting from highest)
            for level in (0..=start_level).rev() {
                let neighbors = self.get_neighbors(node_id, level);
                for &neighbor_id in &neighbors {
                    if visited.insert(neighbor_id) {
                        queue.push_back(neighbor_id);
                    }
                }
            }
        }

        // Handle any unvisited nodes (disconnected components)
        for (_old_id, mapping) in old_to_new.iter_mut().enumerate().take(num_nodes) {
            if *mapping == u32::MAX {
                *mapping = new_id;
                new_id += 1;
            }
        }

        // Create new neighbor lists with remapped IDs (using ArcSwap)
        let mut new_neighbors = Vec::with_capacity(num_nodes);
        let mut new_write_locks = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let mut levels = Vec::with_capacity(self.max_levels);
            let mut locks = Vec::with_capacity(self.max_levels);
            for _ in 0..self.max_levels {
                levels.push(ArcSwap::from_pointee(Vec::new().into_boxed_slice()));
                locks.push(Mutex::new(()));
            }
            new_neighbors.push(levels);
            new_write_locks.push(locks);
        }

        for old_id in 0..num_nodes {
            let new_node_id = old_to_new[old_id] as usize;
            #[allow(clippy::needless_range_loop)]
            for level in 0..self.max_levels {
                // Lock-free read of old neighbor list
                let old_neighbor_list = self.neighbors[old_id][level].load();
                let remapped: Vec<u32> = old_neighbor_list
                    .iter()
                    .map(|&old_neighbor| old_to_new[old_neighbor as usize])
                    .collect();
                // Store new neighbor list
                new_neighbors[new_node_id][level].store(Arc::new(remapped.into_boxed_slice()));
            }
        }

        self.neighbors = new_neighbors;
        self.write_locks = new_write_locks;

        old_to_new
    }

    /// Get number of nodes
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.neighbors.len()
    }
}

// Custom serialization for NeighborLists (ArcSwap can't be serialized directly)
impl Serialize for NeighborLists {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("NeighborLists", 3)?;

        // Extract data from ArcSwap for serialization (lock-free)
        let neighbors_data: Vec<Vec<Vec<u32>>> = self
            .neighbors
            .iter()
            .map(|node| node.iter().map(|level| level.load().to_vec()).collect())
            .collect();

        state.serialize_field("neighbors", &neighbors_data)?;
        state.serialize_field("max_levels", &self.max_levels)?;
        state.serialize_field("m_max", &self.m_max)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for NeighborLists {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct NeighborListsData {
            neighbors: Vec<Vec<Vec<u32>>>,
            max_levels: usize,
            m_max: usize,
        }

        let data = NeighborListsData::deserialize(deserializer)?;

        // Wrap data in ArcSwap
        let neighbors: Vec<Vec<ArcSwap<Box<[u32]>>>> = data
            .neighbors
            .iter()
            .map(|node| {
                node.iter()
                    .map(|level| ArcSwap::from_pointee(level.clone().into_boxed_slice()))
                    .collect()
            })
            .collect();

        // Create write locks for each node-level pair
        let write_locks: Vec<Vec<Mutex<()>>> = data
            .neighbors
            .iter()
            .map(|node| node.iter().map(|_| Mutex::new(())).collect())
            .collect();

        Ok(NeighborLists {
            neighbors,
            write_locks,
            max_levels: data.max_levels,
            m_max: data.m_max,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neighbor_lists_basic() {
        let mut lists = NeighborLists::new(8);

        // Set neighbors for node 0, level 0
        lists.set_neighbors(0, 0, vec![1, 2, 3]);

        let neighbors = lists.get_neighbors(0, 0);
        assert_eq!(neighbors, &[1, 2, 3]);

        // Empty level
        let empty = lists.get_neighbors(0, 1);
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_neighbor_lists_bidirectional() {
        let mut lists = NeighborLists::new(8);

        lists.add_bidirectional_link(0, 1, 0);

        assert_eq!(lists.get_neighbors(0, 0), &[1]);
        assert_eq!(lists.get_neighbors(1, 0), &[0]);
    }
}
