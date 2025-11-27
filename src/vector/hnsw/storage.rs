// Vector and neighbor storage for custom HNSW
//
// Design goals:
// - Separate neighbors from nodes (fetch only when needed)
// - Support quantized and full precision vectors
// - Memory-efficient neighbor list storage
// - Thread-safe for parallel HNSW construction

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Storage for neighbor lists (thread-safe for parallel construction)
///
/// Neighbors are stored separately from nodes to improve cache utilization.
/// Only fetch neighbors when traversing the graph.
///
/// Thread-safety: RwLock allows multiple concurrent readers (search) and
/// exclusive writers (edge addition). Critical for parallel HNSW construction.
#[derive(Debug)]
pub struct NeighborLists {
    /// Neighbor storage: neighbors[node_id][level] = RwLock<Vec<neighbor_ids>>
    ///
    /// RwLock enables:
    /// - Parallel reads during search (multiple threads can read simultaneously)
    /// - Exclusive writes during edge addition (one thread modifies at a time)
    /// - Deadlock prevention via ordered locking (always lock lower node_id first)
    neighbors: Vec<Vec<RwLock<Vec<u32>>>>,

    /// Maximum levels supported
    max_levels: usize,

    /// M_max (max neighbors = M * 2)
    /// Used for pre-allocating neighbor lists to reduce reallocations
    m_max: usize,
}

impl NeighborLists {
    /// Create empty neighbor lists
    pub fn new(max_levels: usize) -> Self {
        Self {
            neighbors: Vec::new(),
            max_levels,
            m_max: 32, // Default M*2
        }
    }

    /// Create with pre-allocated capacity and M parameter
    pub fn with_capacity(num_nodes: usize, max_levels: usize, m: usize) -> Self {
        Self {
            neighbors: Vec::with_capacity(num_nodes),
            max_levels,
            m_max: m * 2,
        }
    }

    /// Get M_max (max neighbors)
    pub fn m_max(&self) -> usize {
        self.m_max
    }

    /// Get neighbors for a node at a specific level
    ///
    /// Returns a cloned Vec to release the read lock quickly.
    /// Small performance cost (clone) for thread-safety benefit.
    pub fn get_neighbors(&self, node_id: u32, level: u8) -> Vec<u32> {
        let node_idx = node_id as usize;
        let level_idx = level as usize;

        if node_idx >= self.neighbors.len() {
            return Vec::new();
        }

        if level_idx >= self.neighbors[node_idx].len() {
            return Vec::new();
        }

        // Acquire read lock, clone data, release lock immediately
        self.neighbors[node_idx][level_idx].read().clone()
    }

    /// Execute a closure with read access to neighbors (zero-copy)
    ///
    /// Avoids cloning by holding the read lock during the closure.
    /// Use this for iteration when you don't need ownership.
    #[inline]
    pub fn with_neighbors<F, R>(&self, node_id: u32, level: u8, f: F) -> R
    where
        F: FnOnce(&[u32]) -> R,
    {
        let node_idx = node_id as usize;
        let level_idx = level as usize;

        if node_idx >= self.neighbors.len() {
            return f(&[]);
        }

        if level_idx >= self.neighbors[node_idx].len() {
            return f(&[]);
        }

        // Hold read lock while executing closure - no clone needed
        let guard = self.neighbors[node_idx][level_idx].read();
        f(&guard)
    }

    /// Set neighbors for a node at a specific level
    pub fn set_neighbors(&mut self, node_id: u32, level: u8, neighbors_list: Vec<u32>) {
        let node_idx = node_id as usize;
        let level_idx = level as usize;

        // Ensure we have enough nodes, pre-allocate with m_max capacity
        while self.neighbors.len() <= node_idx {
            let mut levels = Vec::with_capacity(self.max_levels);
            for _ in 0..self.max_levels {
                levels.push(RwLock::new(Vec::with_capacity(self.m_max)));
            }
            self.neighbors.push(levels);
        }

        // Set the neighbors at this level (acquire write lock)
        *self.neighbors[node_idx][level_idx].write() = neighbors_list;
    }

    /// Add a bidirectional link between two nodes at a level
    ///
    /// Thread-safe with deadlock prevention via ordered locking.
    /// Always locks lower node_id first to prevent circular waits.
    pub fn add_bidirectional_link(&mut self, node_a: u32, node_b: u32, level: u8) {
        let node_a_idx = node_a as usize;
        let node_b_idx = node_b as usize;
        let level_idx = level as usize;

        // Ensure we have enough nodes, pre-allocate with m_max capacity
        let max_idx = node_a_idx.max(node_b_idx);
        while self.neighbors.len() <= max_idx {
            let mut levels = Vec::with_capacity(self.max_levels);
            for _ in 0..self.max_levels {
                levels.push(RwLock::new(Vec::with_capacity(self.m_max)));
            }
            self.neighbors.push(levels);
        }

        // Deadlock prevention: always lock in ascending node_id order
        if node_a_idx < node_b_idx {
            // Lock node_a first, then node_b
            let mut lock_a = self.neighbors[node_a_idx][level_idx].write();
            let mut lock_b = self.neighbors[node_b_idx][level_idx].write();

            if !lock_a.contains(&node_b) {
                lock_a.push(node_b);
            }
            if !lock_b.contains(&node_a) {
                lock_b.push(node_a);
            }
        } else if node_a_idx > node_b_idx {
            // Lock node_b first, then node_a
            let mut lock_b = self.neighbors[node_b_idx][level_idx].write();
            let mut lock_a = self.neighbors[node_a_idx][level_idx].write();

            if !lock_b.contains(&node_a) {
                lock_b.push(node_a);
            }
            if !lock_a.contains(&node_b) {
                lock_a.push(node_b);
            }
        } else {
            // Same node - invalid operation, skip
        }
    }

    /// Add bidirectional link (thread-safe version for parallel construction)
    ///
    /// Assumes nodes are already allocated. Uses RwLock for thread-safety.
    /// Only for use during parallel graph construction where all nodes pre-exist.
    pub fn add_bidirectional_link_parallel(&self, node_a: u32, node_b: u32, level: u8) {
        let node_a_idx = node_a as usize;
        let node_b_idx = node_b as usize;
        let level_idx = level as usize;

        // Bounds check
        if node_a_idx >= self.neighbors.len() || node_b_idx >= self.neighbors.len() {
            return; // Skip invalid nodes
        }

        // Deadlock prevention: always lock in ascending node_id order
        if node_a_idx < node_b_idx {
            let mut lock_a = self.neighbors[node_a_idx][level_idx].write();
            let mut lock_b = self.neighbors[node_b_idx][level_idx].write();

            if !lock_a.contains(&node_b) {
                lock_a.push(node_b);
            }
            if !lock_b.contains(&node_a) {
                lock_b.push(node_a);
            }
        } else if node_a_idx > node_b_idx {
            let mut lock_b = self.neighbors[node_b_idx][level_idx].write();
            let mut lock_a = self.neighbors[node_a_idx][level_idx].write();

            if !lock_b.contains(&node_a) {
                lock_b.push(node_a);
            }
            if !lock_a.contains(&node_b) {
                lock_a.push(node_b);
            }
        }
        // If node_a == node_b, skip (same node)
    }

    /// Remove unidirectional link (thread-safe version for parallel construction)
    ///
    /// Removes link from node_a to node_b (NOT bidirectional).
    /// Assumes nodes are already allocated. Uses RwLock for thread-safety.
    pub fn remove_link_parallel(&self, node_a: u32, node_b: u32, level: u8) {
        let node_a_idx = node_a as usize;
        let level_idx = level as usize;

        // Bounds check
        if node_a_idx >= self.neighbors.len() {
            return; // Skip invalid node
        }

        // Remove node_b from node_a's neighbor list
        let mut lock_a = self.neighbors[node_a_idx][level_idx].write();
        lock_a.retain(|&n| n != node_b);
    }

    /// Set neighbors (thread-safe version for parallel construction)
    ///
    /// Assumes node is already allocated. Uses RwLock for thread-safety.
    pub fn set_neighbors_parallel(&self, node_id: u32, level: u8, neighbors_list: Vec<u32>) {
        let node_idx = node_id as usize;
        let level_idx = level as usize;

        // Bounds check
        if node_idx >= self.neighbors.len() {
            return; // Skip invalid node
        }

        // Set the neighbors at this level (acquire write lock)
        *self.neighbors[node_idx][level_idx].write() = neighbors_list;
    }

    /// Get total number of neighbor entries
    pub fn total_neighbors(&self) -> usize {
        self.neighbors
            .iter()
            .flat_map(|node| node.iter())
            .map(|level| level.read().len())
            .sum()
    }

    /// Get memory usage in bytes (approximate)
    pub fn memory_usage(&self) -> usize {
        let mut total = 0;

        // Size of outer Vec
        total += self.neighbors.capacity() * std::mem::size_of::<Vec<RwLock<Vec<u32>>>>();

        // Size of each node's level vecs
        for node in &self.neighbors {
            total += node.capacity() * std::mem::size_of::<RwLock<Vec<u32>>>();

            // Size of actual neighbor data (acquire read lock for each)
            for level in node {
                let lock = level.read();
                total += lock.len() * std::mem::size_of::<u32>();
            }
        }

        total
    }

    /// Reorder nodes using BFS for cache locality
    ///
    /// This improves cache performance by placing frequently-accessed neighbors
    /// close together in memory. Uses BFS from the entry point to determine ordering.
    ///
    /// Returns a mapping from old_id -> new_id
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
                for &neighbor_id in &neighbors {  // ← Add & to iterate over Vec
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

        // Create new neighbor lists with remapped IDs (wrapped in RwLock)
        let mut new_neighbors = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let mut levels = Vec::with_capacity(self.max_levels);
            for _ in 0..self.max_levels {
                levels.push(RwLock::new(Vec::new()));
            }
            new_neighbors.push(levels);
        }

        for old_id in 0..num_nodes {
            let new_id = old_to_new[old_id] as usize;
            #[allow(clippy::needless_range_loop)]
            for level in 0..self.max_levels {
                // Acquire read lock to access old neighbor list
                let old_neighbor_list = self.neighbors[old_id][level].read();
                let remapped: Vec<u32> = old_neighbor_list
                    .iter()
                    .map(|&old_neighbor| old_to_new[old_neighbor as usize])
                    .collect();
                // Acquire write lock to set new neighbor list
                *new_neighbors[new_id][level].write() = remapped;
            }
        }

        self.neighbors = new_neighbors;

        old_to_new
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.neighbors.len()
    }
}

// Custom serialization for NeighborLists (RwLock can't be serialized directly)
impl Serialize for NeighborLists {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("NeighborLists", 3)?;

        // Extract data from RwLocks for serialization
        let neighbors_data: Vec<Vec<Vec<u32>>> = self
            .neighbors
            .iter()
            .map(|node| {
                node.iter()
                    .map(|level| level.read().clone())
                    .collect()
            })
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

        // Wrap data in RwLocks
        let neighbors: Vec<Vec<RwLock<Vec<u32>>>> = data
            .neighbors
            .into_iter()
            .map(|node| {
                node.into_iter()
                    .map(RwLock::new)
                    .collect()
            })
            .collect();

        Ok(NeighborLists {
            neighbors,
            max_levels: data.max_levels,
            m_max: data.m_max,
        })
    }
}

/// Vector storage (quantized or full precision)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VectorStorage {
    /// Full precision f32 vectors
    ///
    /// Memory: dimensions * 4 bytes per vector
    /// Example: 1536D = 6144 bytes per vector
    FullPrecision {
        vectors: Vec<Vec<f32>>,
        dimensions: usize,
    },

    /// Binary quantized vectors
    ///
    /// Memory: dimensions / 8 bytes per vector (1 bit per dimension)
    /// Example: 1536D = 192 bytes per vector (32x compression)
    BinaryQuantized {
        /// Quantized vectors (1 bit per dimension, packed into bytes)
        quantized: Vec<Vec<u8>>,

        /// Original vectors for reranking (optional)
        ///
        /// If present: Memory = quantized + original
        /// If absent: Faster but lower recall
        original: Option<Vec<Vec<f32>>>,

        /// Quantization thresholds (one per dimension)
        thresholds: Vec<f32>,

        /// Vector dimensions
        dimensions: usize,
    },
}

impl VectorStorage {
    /// Create empty full precision storage
    pub fn new_full_precision(dimensions: usize) -> Self {
        Self::FullPrecision {
            vectors: Vec::new(),
            dimensions,
        }
    }

    /// Create empty binary quantized storage
    pub fn new_binary_quantized(dimensions: usize, keep_original: bool) -> Self {
        Self::BinaryQuantized {
            quantized: Vec::new(),
            original: if keep_original {
                Some(Vec::new())
            } else {
                None
            },
            thresholds: vec![0.0; dimensions], // Will be computed during training
            dimensions,
        }
    }

    /// Get number of vectors stored
    pub fn len(&self) -> usize {
        match self {
            Self::FullPrecision { vectors, .. } => vectors.len(),
            Self::BinaryQuantized { quantized, .. } => quantized.len(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        match self {
            Self::FullPrecision { dimensions, .. } => *dimensions,
            Self::BinaryQuantized { dimensions, .. } => *dimensions,
        }
    }

    /// Insert a full precision vector
    pub fn insert(&mut self, vector: Vec<f32>) -> Result<u32, String> {
        match self {
            Self::FullPrecision { vectors, dimensions } => {
                if vector.len() != *dimensions {
                    return Err(format!(
                        "Vector dimension mismatch: expected {}, got {}",
                        dimensions,
                        vector.len()
                    ));
                }
                let id = vectors.len() as u32;
                vectors.push(vector);
                Ok(id)
            }
            Self::BinaryQuantized {
                quantized,
                original,
                thresholds,
                dimensions,
            } => {
                if vector.len() != *dimensions {
                    return Err(format!(
                        "Vector dimension mismatch: expected {}, got {}",
                        dimensions,
                        vector.len()
                    ));
                }

                // Quantize vector
                let quant = Self::quantize_binary(&vector, thresholds);
                let id = quantized.len() as u32;
                quantized.push(quant);

                // Store original if requested
                if let Some(orig) = original {
                    orig.push(vector);
                }

                Ok(id)
            }
        }
    }

    /// Get a vector by ID (full precision)
    pub fn get(&self, id: u32) -> Option<&[f32]> {
        match self {
            Self::FullPrecision { vectors, .. } => {
                vectors.get(id as usize).map(|v| v.as_slice())
            }
            Self::BinaryQuantized { original, .. } => {
                original.as_ref().and_then(|o| o.get(id as usize).map(|v| v.as_slice()))
            }
        }
    }

    /// Prefetch a vector's data into CPU cache (for HNSW search optimization)
    ///
    /// This hints to the CPU to load the vector data into cache before it's needed.
    /// Call this on neighbor[j+1] while computing distance to neighbor[j].
    /// ~10% search speedup per hnswlib benchmarks.
    ///
    /// NOTE: This gets the pointer directly without loading the data, so the
    /// prefetch hint can be issued before the data is needed.
    /// Prefetch vector data into L1 cache
    ///
    /// Simple single-cache-line prefetch (64 bytes).
    /// Hardware prefetcher handles subsequent cache lines.
    #[inline]
    pub fn prefetch(&self, id: u32) {
        let ptr = match self {
            Self::FullPrecision { vectors, .. } => {
                vectors.get(id as usize).map(|v| v.as_ptr())
            }
            Self::BinaryQuantized { original, .. } => {
                original.as_ref().and_then(|o| o.get(id as usize).map(|v| v.as_ptr()))
            }
        };

        if let Some(ptr) = ptr {
            // SAFETY: ptr is valid and aligned since it comes from a valid Vec
            #[cfg(target_arch = "x86_64")]
            unsafe {
                std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
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
    }

    /// Binary quantize a vector
    ///
    /// Each dimension is quantized to 1 bit based on threshold:
    /// - value >= threshold[dim] => 1
    /// - value < threshold[dim] => 0
    fn quantize_binary(vector: &[f32], thresholds: &[f32]) -> Vec<u8> {
        debug_assert_eq!(vector.len(), thresholds.len());

        let num_bytes = vector.len().div_ceil(8); // Round up
        let mut quantized = vec![0u8; num_bytes];

        for (i, (&value, &threshold)) in vector.iter().zip(thresholds.iter()).enumerate() {
            if value >= threshold {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                quantized[byte_idx] |= 1 << bit_idx;
            }
        }

        quantized
    }

    /// Compute quantization thresholds from sample vectors
    ///
    /// Uses median of each dimension as threshold
    pub fn train_quantization(&mut self, sample_vectors: &[Vec<f32>]) -> Result<(), String> {
        match self {
            Self::BinaryQuantized {
                thresholds,
                dimensions,
                ..
            } => {
                if sample_vectors.is_empty() {
                    return Err("Cannot train on empty sample".to_string());
                }

                // Verify all vectors have correct dimensions
                for vec in sample_vectors {
                    if vec.len() != *dimensions {
                        return Err("Sample vector dimension mismatch".to_string());
                    }
                }

                // Compute median for each dimension
                for dim in 0..*dimensions {
                    let mut values: Vec<f32> = sample_vectors.iter().map(|v| v[dim]).collect();
                    values.sort_by_key(|&x| OrderedFloat(x));

                    let median = if values.len().is_multiple_of(2) {
                        let mid = values.len() / 2;
                        (values[mid - 1] + values[mid]) / 2.0
                    } else {
                        values[values.len() / 2]
                    };

                    thresholds[dim] = median;
                }

                Ok(())
            }
            Self::FullPrecision { .. } => {
                Err("Cannot train quantization on full precision storage".to_string())
            }
        }
    }

    /// Get memory usage in bytes (approximate)
    pub fn memory_usage(&self) -> usize {
        match self {
            Self::FullPrecision { vectors, dimensions } => {
                vectors.len() * dimensions * std::mem::size_of::<f32>()
            }
            Self::BinaryQuantized {
                quantized,
                original,
                thresholds,
                dimensions,
            } => {
                let quantized_size = quantized.len() * (dimensions + 7) / 8;
                let original_size = original
                    .as_ref()
                    .map(|o| o.len() * dimensions * std::mem::size_of::<f32>())
                    .unwrap_or(0);
                let thresholds_size = thresholds.len() * std::mem::size_of::<f32>();
                quantized_size + original_size + thresholds_size
            }
        }
    }

    /// Reorder vectors based on node ID mapping
    ///
    /// old_to_new[old_id] = new_id
    /// This reorders vectors to match the BFS-reordered neighbor lists.
    pub fn reorder(&mut self, old_to_new: &[u32]) {
        match self {
            Self::FullPrecision { vectors, .. } => {
                let mut new_vectors = vec![Vec::new(); vectors.len()];
                for (old_id, &new_id) in old_to_new.iter().enumerate() {
                    new_vectors[new_id as usize] = std::mem::take(&mut vectors[old_id]);
                }
                *vectors = new_vectors;
            }
            Self::BinaryQuantized {
                quantized,
                original,
                ..
            } => {
                // Reorder quantized vectors
                let mut new_quantized = vec![Vec::new(); quantized.len()];
                for (old_id, &new_id) in old_to_new.iter().enumerate() {
                    new_quantized[new_id as usize] = std::mem::take(&mut quantized[old_id]);
                }
                *quantized = new_quantized;

                // Reorder original vectors if present
                if let Some(orig) = original {
                    let mut new_original = vec![Vec::new(); orig.len()];
                    for (old_id, &new_id) in old_to_new.iter().enumerate() {
                        new_original[new_id as usize] = std::mem::take(&mut orig[old_id]);
                    }
                    *orig = new_original;
                }
            }
        }
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

    #[test]
    fn test_vector_storage_full_precision() {
        let mut storage = VectorStorage::new_full_precision(3);

        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];

        let id1 = storage.insert(vec1.clone()).unwrap();
        let id2 = storage.insert(vec2.clone()).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(storage.len(), 2);

        assert_eq!(storage.get(0), Some(vec1.as_slice()));
        assert_eq!(storage.get(1), Some(vec2.as_slice()));
    }

    #[test]
    fn test_vector_storage_dimension_check() {
        let mut storage = VectorStorage::new_full_precision(3);

        let wrong_dim = vec![1.0, 2.0]; // Only 2 dimensions
        assert!(storage.insert(wrong_dim).is_err());
    }

    #[test]
    fn test_binary_quantization() {
        let vector = vec![0.5, -0.3, 0.8, -0.1];
        let thresholds = vec![0.0, 0.0, 0.0, 0.0];

        let quantized = VectorStorage::quantize_binary(&vector, &thresholds);

        // First 4 bits should be: 1, 0, 1, 0 (based on >= 0.0)
        // Packed as: bit0=1, bit1=0, bit2=1, bit3=0 => 0b00000101 = 5
        assert_eq!(quantized[0], 5);
    }

    #[test]
    fn test_quantization_training() {
        let mut storage = VectorStorage::new_binary_quantized(2, true);

        let samples = vec![
            vec![1.0, 5.0],
            vec![2.0, 6.0],
            vec![3.0, 7.0],
        ];

        storage.train_quantization(&samples).unwrap();

        // Thresholds should be medians: [2.0, 6.0]
        match storage {
            VectorStorage::BinaryQuantized { thresholds, .. } => {
                assert_eq!(thresholds, vec![2.0, 6.0]);
            }
            _ => panic!("Expected BinaryQuantized storage"),
        }
    }
}
