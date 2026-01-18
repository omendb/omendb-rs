// Vector and neighbor storage for custom HNSW
//
// Design goals:
// - Separate neighbors from nodes (fetch only when needed)
// - Support quantized and full precision vectors
// - Memory-efficient neighbor list storage
// - Thread-safe for parallel HNSW construction
// - LOCK-FREE READS for search performance (ArcSwap)

use arc_swap::ArcSwap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

use crate::compression::scalar::QueryPrep;
use crate::compression::ScalarParams;
use crate::distance::dot_product;

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
        use super::prefetch::PrefetchConfig;
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

/// Vector storage (quantized or full precision)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VectorStorage {
    /// Full precision f32 vectors - FLAT CONTIGUOUS STORAGE
    ///
    /// Memory: dimensions * 4 bytes per vector + 4 bytes for norm
    /// Example: 1536D = 6148 bytes per vector
    ///
    /// Vectors stored in single contiguous array for cache efficiency.
    /// Access: vectors[id * dimensions..(id + 1) * dimensions]
    ///
    /// Norms (||v||^2) are stored separately for L2 decomposition optimization:
    /// ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    /// This reduces L2 distance from 3N FLOPs to 2N+3 FLOPs (~7% faster).
    FullPrecision {
        /// Flat contiguous vector data (all vectors concatenated)
        vectors: Vec<f32>,
        /// Pre-computed squared norms (||v||^2) for L2 decomposition
        norms: Vec<f32>,
        /// Number of vectors stored
        count: usize,
        /// Dimensions per vector
        dimensions: usize,
    },

    /// Scalar quantized vectors (SQ8) - 4x compression, ~97% recall, 2-3x faster
    ///
    /// Memory: 1x (quantized only, no originals stored)
    /// Trade-off: 4x RAM savings for ~3% recall loss
    ///
    /// Uses uniform min/max scaling with integer SIMD distance computation.
    /// Lazy training: Buffers first 256 vectors, then trains and quantizes.
    ///
    /// Note: No rescore support - originals not stored to save memory.
    ScalarQuantized {
        /// Trained quantization parameters (global scale/offset)
        params: ScalarParams,

        /// Quantized vectors as flat contiguous u8 array
        /// Empty until training completes (after 256 vectors)
        /// Access: quantized[id * dimensions..(id + 1) * dimensions]
        quantized: Vec<u8>,

        /// Pre-computed squared norms of dequantized vectors for L2 decomposition
        /// ||dequant(q)||^2 = sum((code[d] * scale + offset)^2)
        /// Enables fast distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        norms: Vec<f32>,

        /// Pre-computed sums of quantized values for fast integer dot product
        /// sum = sum(quantized[d])
        sums: Vec<i32>,

        /// Buffer for training vectors (cleared after training)
        /// During training phase, stores f32 vectors until we have enough to train
        training_buffer: Vec<f32>,

        /// Number of vectors stored
        count: usize,

        /// Vector dimensions
        dimensions: usize,

        /// Whether quantization parameters have been trained
        /// Training happens automatically after 256 vectors are inserted
        trained: bool,
    },
}

impl VectorStorage {
    /// Create empty full precision storage
    #[must_use]
    pub fn new_full_precision(dimensions: usize) -> Self {
        Self::FullPrecision {
            vectors: Vec::new(),
            norms: Vec::new(),
            count: 0,
            dimensions,
        }
    }

    /// Create empty SQ8 (Scalar Quantized) storage
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality
    ///
    /// # Performance
    /// - Search: 2-3x faster than f32 (integer SIMD)
    /// - Memory: 4x smaller (quantized only, no originals)
    /// - Recall: ~97% (no rescore support)
    ///
    /// # Lazy Training
    /// Quantization parameters are trained automatically after 256 vectors.
    /// Before training completes, search falls back to f32 distance on
    /// the training buffer.
    #[must_use]
    pub fn new_sq8_quantized(dimensions: usize) -> Self {
        Self::ScalarQuantized {
            params: ScalarParams::uninitialized(dimensions),
            quantized: Vec::new(),
            norms: Vec::new(),
            sums: Vec::new(),
            training_buffer: Vec::new(),
            count: 0,
            dimensions,
            trained: false,
        }
    }

    /// Check if this storage uses asymmetric search (SQ8)
    ///
    /// SQ8 uses direct asymmetric L2 distance for search.
    /// This gives ~99.9% recall on SIFT-50K.
    #[must_use]
    pub fn is_asymmetric(&self) -> bool {
        matches!(self, Self::ScalarQuantized { .. })
    }

    /// Check if this storage uses SQ8 quantization
    #[must_use]
    pub fn is_sq8(&self) -> bool {
        matches!(self, Self::ScalarQuantized { .. })
    }

    /// Get number of vectors stored
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::FullPrecision { count, .. } | Self::ScalarQuantized { count, .. } => *count,
        }
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get dimensions
    #[must_use]
    pub fn dimensions(&self) -> usize {
        match self {
            Self::FullPrecision { dimensions, .. } | Self::ScalarQuantized { dimensions, .. } => {
                *dimensions
            }
        }
    }

    /// Insert a full precision vector
    pub fn insert(&mut self, vector: Vec<f32>) -> Result<u32, String> {
        match self {
            Self::FullPrecision {
                vectors,
                norms,
                count,
                dimensions,
            } => {
                if vector.len() != *dimensions {
                    return Err(format!(
                        "Vector dimension mismatch: expected {}, got {}",
                        dimensions,
                        vector.len()
                    ));
                }
                let id = *count as u32;
                // Compute and store squared norm for L2 decomposition
                let norm_sq: f32 = vector.iter().map(|&x| x * x).sum();
                norms.push(norm_sq);
                vectors.extend(vector);
                *count += 1;
                Ok(id)
            }
            Self::ScalarQuantized {
                params,
                quantized,
                norms,
                sums,
                training_buffer,
                count,
                dimensions,
                trained,
            } => {
                if vector.len() != *dimensions {
                    return Err(format!(
                        "Vector dimension mismatch: expected {}, got {}",
                        dimensions,
                        vector.len()
                    ));
                }

                let id = *count as u32;
                let dim = *dimensions;

                if *trained {
                    // Already trained - quantize directly, don't store original
                    let quant = params.quantize(&vector);
                    norms.push(quant.norm_sq);
                    sums.push(quant.sum);
                    quantized.extend(quant.data);
                    *count += 1;
                } else {
                    // Still in training phase - buffer the vector
                    training_buffer.extend(vector);
                    *count += 1;

                    if *count >= 256 {
                        // Time to train! Use buffered vectors as training sample
                        let training_refs: Vec<&[f32]> = (0..256)
                            .map(|i| &training_buffer[i * dim..(i + 1) * dim])
                            .collect();
                        *params =
                            ScalarParams::train(&training_refs).map_err(ToString::to_string)?;
                        *trained = true;

                        // Quantize all buffered vectors and store norms/sums
                        quantized.reserve(*count * dim);
                        norms.reserve(*count);
                        sums.reserve(*count);
                        for i in 0..*count {
                            let vec_slice = &training_buffer[i * dim..(i + 1) * dim];
                            let quant = params.quantize(vec_slice);
                            norms.push(quant.norm_sq);
                            sums.push(quant.sum);
                            quantized.extend(quant.data);
                        }

                        // Clear training buffer to free memory
                        training_buffer.clear();
                        training_buffer.shrink_to_fit();
                    }
                }
                // If not trained and count < 256, vectors stay in training_buffer
                // Search will fall back to f32 distance on training_buffer

                Ok(id)
            }
        }
    }

    /// Get a vector by ID (full precision)
    ///
    /// Returns slice directly into contiguous storage - zero-copy, cache-friendly.
    #[inline]
    #[must_use]
    pub fn get(&self, id: u32) -> Option<&[f32]> {
        match self {
            Self::FullPrecision {
                vectors,
                count,
                dimensions,
                ..
            } => {
                let idx = id as usize;
                if idx >= *count {
                    return None;
                }
                let start = idx * *dimensions;
                let end = start + *dimensions;
                Some(&vectors[start..end])
            }
            Self::ScalarQuantized {
                training_buffer,
                count,
                dimensions,
                trained,
                ..
            } => {
                // SQ8 doesn't store originals after training - no rescore support
                // During training phase, return from training buffer
                if *trained {
                    return None; // No originals stored
                }
                let idx = id as usize;
                if idx >= *count {
                    return None;
                }
                let start = idx * *dimensions;
                let end = start + *dimensions;
                Some(&training_buffer[start..end])
            }
        }
    }

    /// Get a vector by ID, dequantizing if necessary (returns owned Vec)
    ///
    /// For full precision storage, clones the slice.
    /// For quantized storage (SQ8), dequantizes the quantized bytes to f32.
    /// Used for neighbor-to-neighbor distance calculations during graph construction.
    #[must_use]
    pub fn get_dequantized(&self, id: u32) -> Option<Vec<f32>> {
        match self {
            Self::FullPrecision {
                vectors,
                count,
                dimensions,
                ..
            } => {
                let idx = id as usize;
                if idx >= *count {
                    return None;
                }
                let start = idx * *dimensions;
                let end = start + *dimensions;
                Some(vectors[start..end].to_vec())
            }
            Self::ScalarQuantized {
                params,
                quantized,
                training_buffer,
                count,
                dimensions,
                trained,
                ..
            } => {
                let idx = id as usize;
                if idx >= *count {
                    return None;
                }
                let dim = *dimensions;
                if *trained {
                    // Dequantize from quantized storage
                    let start = idx * dim;
                    let end = start + dim;
                    Some(params.dequantize(&quantized[start..end]))
                } else {
                    // Still in training phase, return from buffer
                    let start = idx * dim;
                    let end = start + dim;
                    Some(training_buffer[start..end].to_vec())
                }
            }
        }
    }

    /// Compute asymmetric L2 distance (query full precision, candidate quantized)
    ///
    /// This is the HOT PATH for asymmetric search. Works with `ScalarQuantized`
    /// storage. Returns None if storage is not quantized, not trained,
    /// or if id is out of bounds.
    ///
    /// # Performance (Apple Silicon M3 Max, 768D)
    /// - SQ8: Similar speed to full precision (1.07x)
    #[inline(always)]
    #[must_use]
    pub fn distance_asymmetric_l2(&self, query: &[f32], id: u32) -> Option<f32> {
        match self {
            Self::ScalarQuantized {
                params,
                quantized,
                norms,
                sums,
                count,
                dimensions,
                trained,
                ..
            } => {
                // Only use asymmetric distance if trained
                if !*trained {
                    return None;
                }

                let idx = id as usize;
                if idx >= *count {
                    return None;
                }

                let start = idx * *dimensions;
                let end = start + *dimensions;
                // NOTE: This path is inefficient - prepare_query is called per-vector!
                // Use distance_sq8_with_prep() instead for batch operations.
                let query_prep = params.prepare_query(query);
                Some(params.distance_l2_squared_raw(
                    &query_prep,
                    &quantized[start..end],
                    sums[idx],
                    norms[idx],
                ))
            }
            // FullPrecision uses regular L2 distance, not asymmetric
            Self::FullPrecision { .. } => None,
        }
    }

    /// Get ScalarParams reference for SQ8 storage (to prepare query once per search)
    #[inline]
    #[must_use]
    pub fn get_sq8_params(&self) -> Option<&ScalarParams> {
        if let Self::ScalarQuantized {
            params, trained, ..
        } = self
        {
            if *trained {
                return Some(params);
            }
        }
        None
    }

    /// Prepare query for efficient SQ8 distance computation
    ///
    /// Call this ONCE at the start of search, then use `distance_sq8_with_prep()`
    /// for each distance calculation. This avoids the O(D) query preparation
    /// overhead on each distance call.
    ///
    /// Returns None if storage is not SQ8 or not trained.
    #[inline]
    #[must_use]
    pub fn prepare_sq8_query(&self, query: &[f32]) -> Option<QueryPrep> {
        self.get_sq8_params()
            .map(|params| params.prepare_query(query))
    }

    /// Compute SQ8 distance with a pre-prepared query (efficient batch path)
    ///
    /// Use this instead of `distance_asymmetric_l2` when doing multiple distance
    /// computations with the same query. Call `params.prepare_query()` once,
    /// then use this method for each vector.
    #[inline(always)]
    #[must_use]
    pub fn distance_sq8_with_prep(&self, prep: &QueryPrep, id: u32) -> Option<f32> {
        if let Self::ScalarQuantized {
            params,
            quantized,
            norms,
            sums,
            count,
            dimensions,
            trained,
            ..
        } = self
        {
            if !*trained {
                return None;
            }
            let idx = id as usize;
            if idx >= *count {
                return None;
            }
            let start = idx * *dimensions;
            let end = start + *dimensions;
            Some(params.distance_l2_squared_raw(
                prep,
                &quantized[start..end],
                sums[idx],
                norms[idx],
            ))
        } else {
            None
        }
    }

    /// Batch compute SQ8 distances for multiple vectors
    ///
    /// More efficient than calling `distance_sq8_with_prep` in a loop:
    /// - Common terms computed once
    /// - Better cache utilization
    /// - Better instruction-level parallelism
    ///
    /// Returns the number of distances computed (may be less than ids.len() if some IDs are invalid).
    #[inline]
    pub fn distance_sq8_batch(
        &self,
        prep: &QueryPrep,
        ids: &[u32],
        distances: &mut [f32],
    ) -> usize {
        if let Self::ScalarQuantized {
            params,
            quantized,
            norms,
            sums,
            count,
            dimensions,
            trained,
            ..
        } = self
        {
            if !*trained {
                return 0;
            }

            let dim = *dimensions;
            let n = *count;
            let mut computed = 0;

            // Pre-compute common terms
            let scale_sq = params.scale * params.scale;
            let offset_term = params.offset * params.offset * dim as f32;
            let query_norm = prep.norm_sq;

            for (i, &id) in ids.iter().enumerate() {
                let idx = id as usize;
                if idx >= n {
                    continue;
                }

                let start = idx * dim;
                let vec_data = &quantized[start..start + dim];
                let vec_sum = sums[idx];
                let vec_norm_sq = norms[idx];

                let int_dot = params.int_dot_product_pub(&prep.quantized, vec_data);

                let dot = scale_sq * int_dot as f32
                    + params.scale * params.offset * (prep.sum + vec_sum) as f32
                    + offset_term;

                distances[i] = query_norm + vec_norm_sq - 2.0 * dot;
                computed += 1;
            }

            computed
        } else {
            0
        }
    }

    /// Get the pre-computed squared norm (||v||^2) for a vector
    ///
    /// Only available for FullPrecision storage. Used for L2 decomposition optimization.
    #[inline]
    #[must_use]
    pub fn get_norm(&self, id: u32) -> Option<f32> {
        match self {
            Self::FullPrecision { norms, count, .. } => {
                let idx = id as usize;
                if idx >= *count {
                    return None;
                }
                Some(norms[idx])
            }
            Self::ScalarQuantized { .. } => None,
        }
    }

    /// Check if L2 decomposition is available for this storage
    ///
    /// Returns true for:
    /// - FullPrecision storage (always has pre-computed norms)
    /// - ScalarQuantized storage when trained (uses multiversion dot_product)
    ///
    /// The decomposition path uses `dot_product` with `#[multiversion]` which
    /// provides better cross-compilation compatibility than raw NEON intrinsics.
    #[inline]
    #[must_use]
    pub fn supports_l2_decomposition(&self) -> bool {
        // SQ8 is excluded - L2 decomposition causes ~10% recall regression
        // due to numerical precision issues (catastrophic cancellation).
        // SQ8 uses the asymmetric path via distance_asymmetric_l2 for 99%+ recall.
        matches!(self, Self::FullPrecision { .. })
    }

    /// Compute L2 squared distance using decomposition: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    ///
    /// This is ~7-15% faster than direct L2/asymmetric computation because:
    /// - Vector norms are pre-computed during insert
    /// - Query norm is computed once per search (passed in)
    /// - Only dot product is computed per-vector (2N FLOPs vs 3N)
    ///
    /// Works for both FullPrecision and trained ScalarQuantized storage.
    /// Returns None if decomposition is not available.
    #[inline(always)]
    #[must_use]
    pub fn distance_l2_decomposed(&self, query: &[f32], query_norm: f32, id: u32) -> Option<f32> {
        match self {
            Self::FullPrecision {
                vectors,
                norms,
                count,
                dimensions,
            } => {
                let idx = id as usize;
                if idx >= *count {
                    return None;
                }
                let start = idx * *dimensions;
                let end = start + *dimensions;
                let vec = &vectors[start..end];
                let vec_norm = norms[idx];

                // ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
                // Uses SIMD-accelerated dot product for performance
                let dot = dot_product(query, vec);
                Some(query_norm + vec_norm - 2.0 * dot)
            }
            Self::ScalarQuantized {
                params,
                quantized,
                norms,
                sums,
                count,
                dimensions,
                trained,
                ..
            } => {
                if !*trained {
                    return None;
                }
                let idx = id as usize;
                if idx >= *count {
                    return None;
                }
                let start = idx * *dimensions;
                let end = start + *dimensions;
                let vec_norm = norms[idx];
                let vec_sum = sums[idx];

                // Use integer SIMD distance with precomputed sums
                let query_prep = params.prepare_query(query);
                let quantized_slice = &quantized[start..end];
                Some(params.distance_l2_squared_raw(
                    &query_prep,
                    quantized_slice,
                    vec_sum,
                    vec_norm,
                ))
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
        let ptr: Option<*const u8> = match self {
            Self::FullPrecision {
                vectors,
                count,
                dimensions,
                ..
            } => {
                let idx = id as usize;
                if idx >= *count {
                    None
                } else {
                    let start = idx * *dimensions;
                    Some(vectors[start..].as_ptr().cast())
                }
            }
            Self::ScalarQuantized {
                quantized,
                training_buffer,
                count,
                dimensions,
                trained,
                ..
            } => {
                let idx = id as usize;
                if idx >= *count {
                    None
                } else if *trained {
                    // Prefetch quantized data for asymmetric search
                    let start = idx * *dimensions;
                    Some(quantized[start..].as_ptr())
                } else {
                    // Not trained yet - prefetch training buffer f32 data
                    let start = idx * *dimensions;
                    Some(training_buffer[start..].as_ptr().cast())
                }
            }
        };

        if let Some(ptr) = ptr {
            // SAFETY: ptr is valid and aligned since it comes from a valid Vec
            #[cfg(target_arch = "x86_64")]
            unsafe {
                std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
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

    /// Compute quantization thresholds from sample vectors
    ///
    /// Only relevant for SQ8 (scalar quantization).
    pub fn train_quantization(&mut self, sample_vectors: &[Vec<f32>]) -> Result<(), String> {
        match self {
            Self::FullPrecision { .. } => {
                Err("Cannot train quantization on full precision storage".to_string())
            }
            Self::ScalarQuantized {
                params,
                quantized,
                norms,
                sums,
                training_buffer,
                count,
                dimensions,
                trained,
            } => {
                if sample_vectors.is_empty() {
                    return Err("Cannot train on empty sample".to_string());
                }

                // Train params from sample vectors
                let refs: Vec<&[f32]> =
                    sample_vectors.iter().map(std::vec::Vec::as_slice).collect();
                *params = ScalarParams::train(&refs).map_err(ToString::to_string)?;
                *trained = true;

                // If there are vectors in training buffer, quantize them now
                if *count > 0 && quantized.is_empty() && !training_buffer.is_empty() {
                    let dim = *dimensions;
                    quantized.reserve(*count * dim);
                    norms.reserve(*count);
                    sums.reserve(*count);
                    for i in 0..*count {
                        let vec_slice = &training_buffer[i * dim..(i + 1) * dim];
                        let quant = params.quantize(vec_slice);
                        norms.push(quant.norm_sq);
                        sums.push(quant.sum);
                        quantized.extend(quant.data);
                    }
                    // Clear training buffer to free memory
                    training_buffer.clear();
                    training_buffer.shrink_to_fit();
                }

                Ok(())
            }
        }
    }

    /// Get memory usage in bytes (approximate)
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        match self {
            Self::FullPrecision { vectors, norms, .. } => {
                vectors.len() * std::mem::size_of::<f32>()
                    + norms.len() * std::mem::size_of::<f32>()
            }
            Self::ScalarQuantized {
                quantized,
                norms,
                sums,
                training_buffer,
                ..
            } => {
                // Quantized u8 vectors + norms + sums + training buffer (usually empty after training) + params
                let quantized_size = quantized.len();
                let norms_size = norms.len() * std::mem::size_of::<f32>();
                let sums_size = sums.len() * std::mem::size_of::<i32>();
                let buffer_size = training_buffer.len() * std::mem::size_of::<f32>();
                // Uniform params: scale + offset + dimensions = 2 * f32 + usize
                let params_size = 2 * std::mem::size_of::<f32>() + std::mem::size_of::<usize>();
                quantized_size + norms_size + sums_size + buffer_size + params_size
            }
        }
    }

    /// Reorder vectors based on node ID mapping
    ///
    /// `old_to_new`[`old_id`] = `new_id`
    /// This reorders vectors to match the BFS-reordered neighbor lists.
    pub fn reorder(&mut self, old_to_new: &[u32]) {
        match self {
            Self::FullPrecision {
                vectors,
                norms,
                count,
                dimensions,
            } => {
                let dim = *dimensions;
                let n = *count;
                let mut new_vectors = vec![0.0f32; vectors.len()];
                let mut new_norms = vec![0.0f32; norms.len()];
                for (old_id, &new_id) in old_to_new.iter().enumerate() {
                    if old_id < n {
                        let old_start = old_id * dim;
                        let new_start = new_id as usize * dim;
                        new_vectors[new_start..new_start + dim]
                            .copy_from_slice(&vectors[old_start..old_start + dim]);
                        new_norms[new_id as usize] = norms[old_id];
                    }
                }
                *vectors = new_vectors;
                *norms = new_norms;
            }
            Self::ScalarQuantized {
                quantized,
                norms,
                sums,
                count,
                dimensions,
                ..
            } => {
                let dim = *dimensions;
                let n = *count;

                // Reorder quantized vectors, norms, and sums
                let mut new_quantized = vec![0u8; quantized.len()];
                let mut new_norms = vec![0.0f32; norms.len()];
                let mut new_sums = vec![0i32; sums.len()];
                for (old_id, &new_id) in old_to_new.iter().enumerate() {
                    if old_id < n {
                        let old_start = old_id * dim;
                        let new_start = new_id as usize * dim;
                        new_quantized[new_start..new_start + dim]
                            .copy_from_slice(&quantized[old_start..old_start + dim]);
                        if old_id < norms.len() {
                            new_norms[new_id as usize] = norms[old_id];
                        }
                        if old_id < sums.len() {
                            new_sums[new_id as usize] = sums[old_id];
                        }
                    }
                }
                *quantized = new_quantized;
                *norms = new_norms;
                *sums = new_sums;
            }
        }
    }

    /// Check if storage is SQ8 quantized and trained
    #[must_use]
    pub fn is_quantized_and_trained(&self) -> bool {
        matches!(self, Self::ScalarQuantized { trained: true, .. })
    }
}

// ============================================================================
// Atomic Slot Storage (Phase 7.1 - High-Performance Graph Storage)
// ============================================================================
//
// Lock-free reads via atomic operations. O(1) inserts without copy-on-write.
// Per-node mutex for write coordination (NOT global lock).
//
// Level 0: 95%+ of accesses during search - dense atomic array
// Upper levels: ~5-10% of nodes - sparse per-node allocation

/// Level 0 neighbors with atomic operations
///
/// Memory layout: [n0_slot0..n0_slot_max_m0][n1_slot0..n1_slot_max_m0]...
/// Each slot is AtomicU32 for lock-free read/write.
///
/// Key insight: `AtomicU32::load(Ordering::Relaxed)` compiles to a plain load
/// instruction on x86 and ARM. Zero overhead compared to non-atomic access.
#[derive(Debug)]
pub struct Level0Storage {
    /// Dense neighbor data: node_id * max_m0 + slot_idx
    data: Vec<AtomicU32>,

    /// Atomic neighbor counts
    counts: Vec<AtomicU16>,

    /// Per-node write coordination (NOT used for reads)
    write_locks: Vec<Mutex<()>>,

    max_m0: usize,
}

impl Level0Storage {
    /// Create with pre-allocated capacity
    pub fn with_capacity(num_nodes: usize, max_m0: usize) -> Self {
        Self {
            data: (0..num_nodes * max_m0).map(|_| AtomicU32::new(0)).collect(),
            counts: (0..num_nodes).map(|_| AtomicU16::new(0)).collect(),
            write_locks: (0..num_nodes).map(|_| Mutex::new(())).collect(),
            max_m0,
        }
    }

    /// Ensure capacity for node_id (requires &mut self)
    pub fn ensure_capacity(&mut self, node_id: u32) {
        let needed_nodes = node_id as usize + 1;
        if self.counts.len() < needed_nodes {
            let old_len = self.counts.len();
            self.data
                .extend((0..(needed_nodes - old_len) * self.max_m0).map(|_| AtomicU32::new(0)));
            self.counts
                .extend((0..needed_nodes - old_len).map(|_| AtomicU16::new(0)));
            self.write_locks
                .extend((0..needed_nodes - old_len).map(|_| Mutex::new(())));
        }
    }

    /// Execute closure with neighbors
    #[inline(always)]
    #[allow(clippy::needless_range_loop)] // Intentional: reading from one array, writing to another
    pub fn with_neighbors<F, R>(&self, node_id: u32, f: F) -> R
    where
        F: FnOnce(&[u32]) -> R,
    {
        let idx = node_id as usize;
        if idx >= self.counts.len() {
            return f(&[]);
        }
        let base = idx * self.max_m0;
        let count = self.counts[idx].load(Ordering::Acquire) as usize;

        let mut buf = [0u32; 64];
        let n = count.min(64).min(self.max_m0);
        for i in 0..n {
            buf[i] = self.data[base + i].load(Ordering::Relaxed);
        }
        f(&buf[..n])
    }

    /// Get neighbors as Vec (API compatibility, allocates)
    #[must_use]
    pub fn get_neighbors(&self, node_id: u32) -> Vec<u32> {
        let idx = node_id as usize;
        if idx >= self.counts.len() {
            return Vec::new();
        }
        let base = idx * self.max_m0;
        let count = self.counts[idx].load(Ordering::Acquire) as usize;
        let n = count.min(self.max_m0);
        (0..n)
            .map(|i| self.data[base + i].load(Ordering::Relaxed))
            .collect()
    }

    /// Set neighbors for a node (acquires write lock)
    pub fn set_neighbors(&self, node_id: u32, neighbors: &[u32]) {
        let idx = node_id as usize;
        if idx >= self.write_locks.len() {
            return;
        }
        let _lock = self.write_locks[idx].lock();

        let base = idx * self.max_m0;
        let count = neighbors.len().min(self.max_m0);

        for (i, &neighbor) in neighbors[..count].iter().enumerate() {
            self.data[base + i].store(neighbor, Ordering::Relaxed);
        }
        self.counts[idx].store(count as u16, Ordering::Release);
    }

    /// Add a single neighbor - O(1) per neighbor (used in tests)
    #[cfg(test)]
    pub fn add_neighbor(&self, node_id: u32, neighbor: u32) -> bool {
        let idx = node_id as usize;
        if idx >= self.write_locks.len() {
            return false;
        }
        let _lock = self.write_locks[idx].lock();

        let count = self.counts[idx].load(Ordering::Relaxed) as usize;
        if count >= self.max_m0 {
            return false;
        }

        let base = idx * self.max_m0;
        self.data[base + count].store(neighbor, Ordering::Relaxed);
        self.counts[idx].store((count + 1) as u16, Ordering::Release);
        true
    }

    /// Add neighbor without acquiring lock (caller must hold lock)
    #[inline]
    pub fn add_neighbor_unlocked(&self, node_id: u32, neighbor: u32) -> bool {
        let idx = node_id as usize;
        let count = self.counts[idx].load(Ordering::Relaxed) as usize;
        if count >= self.max_m0 {
            return false;
        }
        let base = idx * self.max_m0;
        self.data[base + count].store(neighbor, Ordering::Relaxed);
        self.counts[idx].store((count + 1) as u16, Ordering::Release);
        true
    }

    /// Remove a neighbor (swap-remove)
    pub fn remove_neighbor(&self, node_id: u32, neighbor: u32) {
        let idx = node_id as usize;
        if idx >= self.write_locks.len() {
            return;
        }
        let _lock = self.write_locks[idx].lock();

        let count = self.counts[idx].load(Ordering::Relaxed) as usize;
        let base = idx * self.max_m0;

        for i in 0..count {
            if self.data[base + i].load(Ordering::Relaxed) == neighbor {
                let last = self.data[base + count - 1].load(Ordering::Relaxed);
                self.data[base + i].store(last, Ordering::Relaxed);
                self.counts[idx].store((count - 1) as u16, Ordering::Release);
                break;
            }
        }
    }

    /// Prefetch neighbor data into L1 cache
    #[inline]
    pub fn prefetch(&self, node_id: u32) {
        let base = node_id as usize * self.max_m0;
        if base < self.data.len() {
            let ptr = &self.data[base] as *const AtomicU32 as *const u8;

            #[cfg(target_arch = "x86_64")]
            unsafe {
                std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
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

    /// Get write lock for a node (for bidirectional link operations)
    #[inline]
    pub fn get_write_lock(&self, node_id: u32) -> Option<parking_lot::MutexGuard<'_, ()>> {
        let idx = node_id as usize;
        if idx >= self.write_locks.len() {
            return None;
        }
        Some(self.write_locks[idx].lock())
    }

    pub fn num_nodes(&self) -> usize {
        self.counts.len()
    }

    pub fn memory_usage(&self) -> usize {
        self.data.len() * 4 + self.counts.len() * 2 + self.write_locks.len() * 8
    }
}

/// Upper level links for a single node (levels 1..=max_level)
#[derive(Debug)]
pub struct UpperNodeLinks {
    /// All levels in one allocation: [l1_neighbors][l2_neighbors]...
    data: Vec<AtomicU32>,

    /// Count per level
    counts: Vec<AtomicU8>,

    max_level: u8,
}

/// Upper level neighbors - sparse per-node allocation
///
/// Only ~5-10% of nodes have upper levels. Uses per-node allocation
/// with atomic operations.
#[derive(Debug)]
pub struct UpperLevelStorage {
    /// Per-node upper level data (None for level-0-only nodes)
    nodes: Vec<Option<UpperNodeLinks>>,

    /// Per-node locks for upper level writes
    locks: Vec<Mutex<()>>,

    max_m: usize,
}

impl UpperLevelStorage {
    pub fn new(max_m: usize) -> Self {
        Self {
            nodes: Vec::new(),
            locks: Vec::new(),
            max_m,
        }
    }

    pub fn ensure_capacity(&mut self, node_id: u32) {
        let needed = node_id as usize + 1;
        if self.nodes.len() < needed {
            self.nodes.resize_with(needed, || None);
            self.locks
                .extend((0..needed - self.locks.len()).map(|_| Mutex::new(())));
        }
    }

    /// Allocate upper level storage for a node
    pub fn allocate_node(&mut self, node_id: u32, max_level: u8) {
        self.ensure_capacity(node_id);
        if max_level == 0 {
            return;
        }
        let levels = max_level as usize;
        self.nodes[node_id as usize] = Some(UpperNodeLinks {
            data: (0..levels * self.max_m)
                .map(|_| AtomicU32::new(0))
                .collect(),
            counts: (0..levels).map(|_| AtomicU8::new(0)).collect(),
            max_level,
        });
    }

    /// Get neighbors at upper level (level >= 1) - LOCK-FREE
    #[inline]
    #[allow(clippy::needless_range_loop)] // Intentional: reading from one array, writing to another
    pub fn with_neighbors<F, R>(&self, node_id: u32, level: u8, f: F) -> R
    where
        F: FnOnce(&[u32]) -> R,
    {
        debug_assert!(level >= 1);
        let idx = node_id as usize;

        if idx >= self.nodes.len() {
            return f(&[]);
        }

        match &self.nodes[idx] {
            Some(links) if level <= links.max_level => {
                let level_idx = (level - 1) as usize;
                let base = level_idx * self.max_m;
                let count = links.counts[level_idx].load(Ordering::Acquire) as usize;

                let mut buf = [0u32; 32];
                let n = count.min(32).min(self.max_m);
                for i in 0..n {
                    buf[i] = links.data[base + i].load(Ordering::Relaxed);
                }
                f(&buf[..n])
            }
            _ => f(&[]),
        }
    }

    /// Get neighbors as Vec (API compatibility)
    #[must_use]
    pub fn get_neighbors(&self, node_id: u32, level: u8) -> Vec<u32> {
        let idx = node_id as usize;
        if idx >= self.nodes.len() {
            return Vec::new();
        }
        match &self.nodes[idx] {
            Some(links) if level <= links.max_level => {
                let level_idx = (level - 1) as usize;
                let base = level_idx * self.max_m;
                let count = links.counts[level_idx].load(Ordering::Acquire) as usize;
                let n = count.min(self.max_m);
                (0..n)
                    .map(|i| links.data[base + i].load(Ordering::Relaxed))
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    /// Add neighbor at upper level (used in tests)
    #[cfg(test)]
    pub fn add_neighbor(&self, node_id: u32, level: u8, neighbor: u32) -> bool {
        let idx = node_id as usize;
        if idx >= self.locks.len() {
            return false;
        }
        let _lock = self.locks[idx].lock();

        if let Some(Some(links)) = self.nodes.get(idx) {
            if level <= links.max_level {
                let level_idx = (level - 1) as usize;
                let count = links.counts[level_idx].load(Ordering::Relaxed) as usize;
                if count >= self.max_m {
                    return false;
                }
                let base = level_idx * self.max_m;
                links.data[base + count].store(neighbor, Ordering::Relaxed);
                links.counts[level_idx].store((count + 1) as u8, Ordering::Release);
                return true;
            }
        }
        false
    }

    /// Add neighbor without acquiring lock (caller must hold lock)
    #[inline]
    pub fn add_neighbor_unlocked(&self, node_id: u32, level: u8, neighbor: u32) -> bool {
        let idx = node_id as usize;
        if let Some(Some(links)) = self.nodes.get(idx) {
            if level <= links.max_level {
                let level_idx = (level - 1) as usize;
                let count = links.counts[level_idx].load(Ordering::Relaxed) as usize;
                if count >= self.max_m {
                    return false;
                }
                let base = level_idx * self.max_m;
                links.data[base + count].store(neighbor, Ordering::Relaxed);
                links.counts[level_idx].store((count + 1) as u8, Ordering::Release);
                return true;
            }
        }
        false
    }

    /// Set neighbors at upper level
    pub fn set_neighbors(&self, node_id: u32, level: u8, neighbors: &[u32]) {
        let idx = node_id as usize;
        if idx >= self.locks.len() {
            return;
        }
        let _lock = self.locks[idx].lock();
        if let Some(Some(links)) = self.nodes.get(idx) {
            if level <= links.max_level {
                let level_idx = (level - 1) as usize;
                let base = level_idx * self.max_m;
                let count = neighbors.len().min(self.max_m);
                for (i, &neighbor) in neighbors[..count].iter().enumerate() {
                    links.data[base + i].store(neighbor, Ordering::Relaxed);
                }
                links.counts[level_idx].store(count as u8, Ordering::Release);
            }
        }
    }

    /// Remove a neighbor at upper level
    pub fn remove_neighbor(&self, node_id: u32, level: u8, neighbor: u32) {
        let idx = node_id as usize;
        if idx >= self.locks.len() {
            return;
        }
        let _lock = self.locks[idx].lock();
        if let Some(Some(links)) = self.nodes.get(idx) {
            if level <= links.max_level {
                let level_idx = (level - 1) as usize;
                let base = level_idx * self.max_m;
                let count = links.counts[level_idx].load(Ordering::Relaxed) as usize;
                for i in 0..count {
                    if links.data[base + i].load(Ordering::Relaxed) == neighbor {
                        let last = links.data[base + count - 1].load(Ordering::Relaxed);
                        links.data[base + i].store(last, Ordering::Relaxed);
                        links.counts[level_idx].store((count - 1) as u8, Ordering::Release);
                        break;
                    }
                }
            }
        }
    }

    /// Get write lock for a node
    #[inline]
    pub fn get_write_lock(&self, node_id: u32) -> Option<parking_lot::MutexGuard<'_, ()>> {
        let idx = node_id as usize;
        if idx >= self.locks.len() {
            return None;
        }
        Some(self.locks[idx].lock())
    }

    pub fn get_max_level(&self, node_id: u32) -> u8 {
        self.nodes
            .get(node_id as usize)
            .and_then(|opt| opt.as_ref())
            .map_or(0, |links| links.max_level)
    }

    pub fn memory_usage(&self) -> usize {
        let mut total = self.nodes.len() * std::mem::size_of::<Option<UpperNodeLinks>>();
        total += self.locks.len() * std::mem::size_of::<Mutex<()>>();
        for links in self.nodes.iter().flatten() {
            total += links.data.len() * 4; // AtomicU32
            total += links.counts.len(); // AtomicU8
        }
        total
    }
}

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
        use std::collections::{HashSet, VecDeque};

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
    fn test_sq8_train_empty_sample_rejected() {
        let mut storage = VectorStorage::new_sq8_quantized(4);
        let empty_samples: Vec<Vec<f32>> = vec![];
        let result = storage.train_quantization(&empty_samples);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty sample"));
    }

    // ========================================================================
    // Atomic Slot Storage Tests (Phase 7.1)
    // ========================================================================

    #[test]
    fn test_level0_storage_basic() {
        let mut storage = Level0Storage::with_capacity(0, 32);

        // Allocate space for nodes
        storage.ensure_capacity(2);
        assert_eq!(storage.num_nodes(), 3);

        // Set neighbors for node 0
        storage.set_neighbors(0, &[1, 2, 3]);
        assert_eq!(storage.get_neighbors(0), vec![1, 2, 3]);

        // Set neighbors for node 1
        storage.set_neighbors(1, &[0, 2]);
        assert_eq!(storage.get_neighbors(1), vec![0, 2]);

        // Empty node
        assert!(storage.get_neighbors(2).is_empty());
    }

    #[test]
    fn test_level0_storage_add_remove() {
        let storage = Level0Storage::with_capacity(3, 32);

        // Add neighbors one by one
        assert!(storage.add_neighbor(0, 1));
        assert!(storage.add_neighbor(0, 2));
        assert!(storage.add_neighbor(0, 3));
        assert_eq!(storage.get_neighbors(0), vec![1, 2, 3]);

        // Remove middle neighbor (swap-remove)
        storage.remove_neighbor(0, 2);
        let neighbors = storage.get_neighbors(0);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&3));
    }

    #[test]
    fn test_level0_storage_with_neighbors() {
        let storage = Level0Storage::with_capacity(2, 32);
        storage.set_neighbors(0, &[10, 20, 30]);

        // Use closure-based access
        let sum = storage.with_neighbors(0, |neighbors| neighbors.iter().sum::<u32>());
        assert_eq!(sum, 60);

        // Empty node
        let count = storage.with_neighbors(1, |neighbors| neighbors.len());
        assert_eq!(count, 0);
    }

    #[test]
    fn test_level0_storage_capacity_limit() {
        let storage = Level0Storage::with_capacity(1, 4); // max 4 neighbors

        assert!(storage.add_neighbor(0, 1));
        assert!(storage.add_neighbor(0, 2));
        assert!(storage.add_neighbor(0, 3));
        assert!(storage.add_neighbor(0, 4));
        // Should fail - at capacity
        assert!(!storage.add_neighbor(0, 5));

        assert_eq!(storage.get_neighbors(0).len(), 4);
    }

    #[test]
    fn test_upper_level_storage_basic() {
        let mut storage = UpperLevelStorage::new(16);

        // Allocate node with 3 levels (levels 1, 2, 3)
        storage.allocate_node(0, 3);
        assert_eq!(storage.get_max_level(0), 3);

        // Set neighbors at level 1
        storage.set_neighbors(0, 1, &[10, 20]);
        assert_eq!(storage.get_neighbors(0, 1), vec![10, 20]);

        // Set neighbors at level 2
        storage.set_neighbors(0, 2, &[30]);
        assert_eq!(storage.get_neighbors(0, 2), vec![30]);

        // Level 3 is empty
        assert!(storage.get_neighbors(0, 3).is_empty());

        // Non-existent node
        assert!(storage.get_neighbors(99, 1).is_empty());
    }

    #[test]
    fn test_upper_level_storage_add_remove() {
        let mut storage = UpperLevelStorage::new(16);
        storage.allocate_node(0, 2);

        // Add neighbors
        assert!(storage.add_neighbor(0, 1, 100));
        assert!(storage.add_neighbor(0, 1, 200));
        assert_eq!(storage.get_neighbors(0, 1), vec![100, 200]);

        // Remove neighbor
        storage.remove_neighbor(0, 1, 100);
        assert_eq!(storage.get_neighbors(0, 1), vec![200]);
    }

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
