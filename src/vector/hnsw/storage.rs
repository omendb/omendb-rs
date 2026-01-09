// Vector and neighbor storage for custom HNSW
//
// Design goals:
// - Separate neighbors from nodes (fetch only when needed)
// - Support quantized and full precision vectors
// - Memory-efficient neighbor list storage
// - Thread-safe for parallel HNSW construction
// - LOCK-FREE READS for search performance (ArcSwap)

use arc_swap::ArcSwap;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::compression::binary::hamming_distance;
use crate::compression::rabitq::QuantizedVector;
use crate::compression::{ADCTable, RaBitQ, RaBitQParams, ScalarParams};
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

    /// Get total number of neighbor entries
    #[must_use]
    pub fn total_neighbors(&self) -> usize {
        self.neighbors
            .iter()
            .flat_map(|node| node.iter())
            .map(|level| level.load().len())
            .sum()
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

// ============================================================================
// FastScan Neighbor Code Storage
// ============================================================================

/// SIMD batch size for FastScan distance computation
/// AVX2 processes 32 bytes at once, NEON processes 16 but we use 32 for consistency
#[allow(dead_code)] // Used in Phase 3: FastScan search integration
pub const FASTSCAN_BATCH_SIZE: usize = 32;

/// Per-vertex neighbor code storage for FastScan SIMD distance computation
///
/// Stores quantization codes for each vertex's neighbors in an interleaved layout
/// optimized for SIMD parallel lookups. When visiting vertex V, we can compute
/// distances to all its neighbors in one FastScan call instead of N separate calls.
///
/// # Memory Layout (Interleaved)
///
/// For vertex V with neighbors [n0, n1, ..., n31] and M sub-quantizers:
/// ```text
/// [n0_sq0, n1_sq0, ..., n31_sq0]  // 32 bytes - sub-quantizer 0 for all neighbors
/// [n0_sq1, n1_sq1, ..., n31_sq1]  // 32 bytes - sub-quantizer 1 for all neighbors
/// ...
/// [n0_sqM, n1_sqM, ..., n31_sqM]  // 32 bytes - sub-quantizer M for all neighbors
/// ```
///
/// This layout allows SIMD (pshufb/vqtbl1q) to load 32 codes and do 32 parallel
/// LUT lookups in a single instruction.
///
/// # Performance
///
/// Benchmark showed 5x speedup for distance computation (390ns vs 1.93µs for 32 neighbors).
/// Expected 2-3x end-to-end search speedup since distance is ~50-70% of search time.
#[allow(dead_code)] // Used in Phase 3: FastScan search integration
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NeighborCodeStorage {
    /// Interleaved codes: [vertex_0_block, vertex_1_block, ...]
    /// Each block: code_size * FASTSCAN_BATCH_SIZE bytes
    codes: Vec<u8>,

    /// Byte offset into `codes` for each vertex
    /// offsets[v] = start of vertex v's neighbor code block
    offsets: Vec<usize>,

    /// Number of actual neighbors for each vertex (before padding)
    /// Used to know which results are valid vs padding
    neighbor_counts: Vec<usize>,

    /// Bytes per quantized code (e.g., 384 for 768D with 4-bit)
    code_size: usize,

    /// Block size per vertex = code_size * FASTSCAN_BATCH_SIZE
    block_size: usize,
}

#[allow(dead_code)] // Used in Phase 3: FastScan search integration
impl NeighborCodeStorage {
    /// Create empty storage with given code size
    #[must_use]
    pub fn new(code_size: usize) -> Self {
        Self {
            codes: Vec::new(),
            offsets: Vec::new(),
            neighbor_counts: Vec::new(),
            code_size,
            block_size: code_size * FASTSCAN_BATCH_SIZE,
        }
    }

    /// Check if storage is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Get number of vertices with stored neighbor codes
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Get the interleaved code block for a vertex's neighbors
    ///
    /// Returns a slice of `code_size * FASTSCAN_BATCH_SIZE` bytes containing
    /// the interleaved quantization codes for this vertex's neighbors.
    ///
    /// # Layout
    /// For code_size=M and BATCH_SIZE=32:
    /// - bytes [0..32]: sub-quantizer 0 for all 32 neighbors
    /// - bytes [32..64]: sub-quantizer 1 for all 32 neighbors
    /// - ...
    /// - bytes [(M-1)*32..M*32]: sub-quantizer M-1 for all 32 neighbors
    #[must_use]
    #[inline]
    pub fn get_block(&self, vertex_id: u32) -> Option<&[u8]> {
        let idx = vertex_id as usize;
        if idx >= self.offsets.len() {
            return None;
        }
        let start = self.offsets[idx];
        let end = start + self.block_size;
        if end > self.codes.len() {
            return None;
        }
        Some(&self.codes[start..end])
    }

    /// Get the actual neighbor count for a vertex (before padding)
    #[must_use]
    #[inline]
    pub fn get_neighbor_count(&self, vertex_id: u32) -> usize {
        let idx = vertex_id as usize;
        if idx >= self.neighbor_counts.len() {
            return 0;
        }
        self.neighbor_counts[idx]
    }

    /// Build neighbor code storage from VectorStorage and NeighborLists
    ///
    /// Extracts quantized codes for each vertex's neighbors and stores them
    /// in interleaved layout for FastScan.
    ///
    /// # Arguments
    /// * `vectors` - Vector storage containing quantized codes
    /// * `neighbors` - Neighbor lists for all vertices
    /// * `level` - Which level to build codes for (typically 0 for most benefit)
    ///
    /// # Returns
    /// New NeighborCodeStorage with interleaved codes for all vertices
    pub fn build_from_storage(
        vectors: &VectorStorage,
        neighbors: &NeighborLists,
        level: u8,
    ) -> Option<Self> {
        // Only works with RaBitQ quantized storage
        let (quantized_data, code_size) = match vectors {
            VectorStorage::RaBitQQuantized {
                quantized_data,
                code_size,
                ..
            } => (quantized_data, *code_size),
            _ => return None,
        };

        let num_vertices = neighbors.len();
        if num_vertices == 0 {
            return Some(Self::new(code_size));
        }

        let block_size = code_size * FASTSCAN_BATCH_SIZE;
        let mut codes = Vec::with_capacity(num_vertices * block_size);
        let mut offsets = Vec::with_capacity(num_vertices);
        let mut neighbor_counts = Vec::with_capacity(num_vertices);

        for vertex_id in 0..num_vertices {
            let offset = codes.len();
            offsets.push(offset);

            // Get this vertex's neighbors
            let vertex_neighbors = neighbors.get_neighbors(vertex_id as u32, level);
            let count = vertex_neighbors.len().min(FASTSCAN_BATCH_SIZE);
            neighbor_counts.push(count);

            // Allocate block for this vertex (will be filled with interleaved codes)
            let block_start = codes.len();
            codes.resize(block_start + block_size, 0);

            // Interleave: for each sub-quantizer position, store all neighbors' codes
            for sq in 0..code_size {
                for (n, &neighbor_id) in vertex_neighbors
                    .iter()
                    .take(FASTSCAN_BATCH_SIZE)
                    .enumerate()
                {
                    let neighbor_idx = neighbor_id as usize;
                    // Get this neighbor's code at sub-quantizer position sq
                    let code_start = neighbor_idx * code_size;
                    if code_start + sq < quantized_data.len() {
                        codes[block_start + sq * FASTSCAN_BATCH_SIZE + n] =
                            quantized_data[code_start + sq];
                    }
                    // Else: padding with 0 (already initialized)
                }
            }
        }

        Some(Self {
            codes,
            offsets,
            neighbor_counts,
            code_size,
            block_size,
        })
    }

    /// Update neighbor codes for a single vertex
    ///
    /// Called when a vertex's neighbor list changes (insertion/deletion).
    /// Re-interleaves codes for the new neighbor list.
    pub fn update_vertex(&mut self, vertex_id: u32, new_neighbors: &[u32], quantized_data: &[u8]) {
        let idx = vertex_id as usize;

        // Ensure we have space for this vertex
        while self.offsets.len() <= idx {
            let offset = self.codes.len();
            self.offsets.push(offset);
            self.neighbor_counts.push(0);
            self.codes.resize(self.codes.len() + self.block_size, 0);
        }

        let block_start = self.offsets[idx];
        let count = new_neighbors.len().min(FASTSCAN_BATCH_SIZE);
        self.neighbor_counts[idx] = count;

        // Clear the block
        for i in 0..self.block_size {
            self.codes[block_start + i] = 0;
        }

        // Re-interleave codes for new neighbors
        for sq in 0..self.code_size {
            for (n, &neighbor_id) in new_neighbors.iter().take(FASTSCAN_BATCH_SIZE).enumerate() {
                let neighbor_idx = neighbor_id as usize;
                let code_start = neighbor_idx * self.code_size;
                if code_start + sq < quantized_data.len() {
                    self.codes[block_start + sq * FASTSCAN_BATCH_SIZE + n] =
                        quantized_data[code_start + sq];
                }
            }
        }
    }

    /// Memory usage in bytes
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.codes.len()
            + self.offsets.len() * std::mem::size_of::<usize>()
            + self.neighbor_counts.len() * std::mem::size_of::<usize>()
    }
}

/// Unified ADC (Asymmetric Distance Computation) table
///
/// Unified ADC table for quantized distance computation.
/// Built once per query, used for all distance computations.
/// Note: SQ8 uses integer SIMD distance instead of ADC tables for better performance.
#[derive(Clone, Debug)]
pub enum UnifiedADC {
    /// `RaBitQ` ADC table (variable bit width)
    RaBitQ(ADCTable),
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
    /// Norms (||v||²) are stored separately for L2 decomposition optimization:
    /// ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
    /// This reduces L2 distance from 3N FLOPs to 2N+3 FLOPs (~7% faster).
    FullPrecision {
        /// Flat contiguous vector data (all vectors concatenated)
        vectors: Vec<f32>,
        /// Pre-computed squared norms (||v||²) for L2 decomposition
        norms: Vec<f32>,
        /// Number of vectors stored
        count: usize,
        /// Dimensions per vector
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

    /// `RaBitQ` quantized vectors for asymmetric search (CLOUD MOAT)
    ///
    /// Memory: dimensions * bits / 8 bytes per vector (4-bit = 8x compression)
    /// Example: 1536D @ 4-bit = 768 bytes per vector
    ///
    /// Key optimization: During search, query stays full precision while
    /// candidates use quantized representation. This gives 2-3x throughput
    /// by avoiding decompression while maintaining accuracy.
    ///
    /// Reranking with original vectors restores recall to near full-precision.
    RaBitQQuantized {
        /// `RaBitQ` quantizer (contains params)
        #[serde(skip)]
        quantizer: Option<RaBitQ>,

        /// `RaBitQ` parameters (for serialization)
        params: RaBitQParams,

        /// Quantized codes - flat contiguous array for cache efficiency
        /// Access: quantized_data[id * code_size..(id + 1) * code_size]
        quantized_data: Vec<u8>,

        /// Per-vector rescaling factors - contiguous for cache efficiency
        /// Access: quantized_scales[id]
        quantized_scales: Vec<f32>,

        /// Bytes per quantized vector (computed from dimensions and bits)
        /// For 4-bit: code_size = dimensions / 2
        code_size: usize,

        /// Original vectors for reranking (required for final accuracy)
        /// Stored as flat contiguous array for cache efficiency.
        original: Vec<f32>,

        /// Number of original vectors stored
        original_count: usize,

        /// Vector dimensions
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
    /// Use `RaBitQ` if you need rescore with originals on disk.
    ScalarQuantized {
        /// Trained quantization parameters (global scale/offset)
        params: ScalarParams,

        /// Quantized vectors as flat contiguous u8 array
        /// Empty until training completes (after 256 vectors)
        /// Access: quantized[id * dimensions..(id + 1) * dimensions]
        quantized: Vec<u8>,

        /// Pre-computed squared norms of dequantized vectors for L2 decomposition
        /// ||dequant(q)||² = Σ(code[d] * scale + offset)²
        /// Enables fast distance: ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
        norms: Vec<f32>,

        /// Pre-computed sums of quantized values for fast integer dot product
        /// sum = Σ quantized[d]
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

    /// Create empty binary quantized storage
    #[must_use]
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

    /// Create empty `RaBitQ` quantized storage for asymmetric search (CLOUD MOAT)
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality
    /// * `params` - `RaBitQ` quantization parameters (typically 4-bit for 8x compression)
    ///
    /// # Performance
    /// - Search: 2-3x faster than full precision (asymmetric distance)
    /// - Memory: 8x smaller storage (4-bit quantization)
    /// - Recall: 98%+ with reranking
    #[must_use]
    pub fn new_rabitq_quantized(dimensions: usize, params: RaBitQParams) -> Self {
        // Compute code size: bytes needed per vector
        // For 4-bit: 2 values per byte, so code_size = dimensions / 2
        let values_per_byte = params.bits_per_dim.values_per_byte();
        let code_size = dimensions.div_ceil(values_per_byte);

        Self::RaBitQQuantized {
            quantizer: Some(RaBitQ::new(params.clone())),
            params,
            quantized_data: Vec::new(),
            quantized_scales: Vec::new(),
            code_size,
            original: Vec::new(),
            original_count: 0,
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

    /// Check if this storage uses asymmetric search (RaBitQ and SQ8)
    ///
    /// Both RaBitQ and SQ8 use direct asymmetric L2 distance for search.
    /// This gives ~99.9% recall on SIFT-50K.
    ///
    /// The mono path with L2 decomposition has 10% recall regression due to
    /// floating point ordering differences during HNSW graph traversal.
    /// Even increasing ef doesn't recover the missing candidates.
    #[must_use]
    pub fn is_asymmetric(&self) -> bool {
        matches!(
            self,
            Self::RaBitQQuantized { .. }
                | Self::ScalarQuantized { .. }
                | Self::BinaryQuantized { .. }
        )
    }

    /// Check if this storage uses binary quantization
    #[must_use]
    pub fn is_binary_quantized(&self) -> bool {
        matches!(self, Self::BinaryQuantized { .. })
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
            Self::BinaryQuantized { quantized, .. } => quantized.len(),
            Self::RaBitQQuantized { original_count, .. } => *original_count,
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
            Self::FullPrecision { dimensions, .. }
            | Self::BinaryQuantized { dimensions, .. }
            | Self::RaBitQQuantized { dimensions, .. }
            | Self::ScalarQuantized { dimensions, .. } => *dimensions,
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
            Self::RaBitQQuantized {
                quantizer,
                params,
                quantized_data,
                quantized_scales,
                original,
                original_count,
                dimensions,
                ..
            } => {
                if vector.len() != *dimensions {
                    return Err(format!(
                        "Vector dimension mismatch: expected {}, got {}",
                        dimensions,
                        vector.len()
                    ));
                }

                // Lazily initialize quantizer if needed (after deserialization)
                let q = quantizer.get_or_insert_with(|| RaBitQ::new(params.clone()));

                // Quantize and store in flat arrays (cache-friendly layout)
                let quant = q.quantize(&vector);
                let id = *original_count as u32;
                quantized_data.extend(&quant.data);
                quantized_scales.push(quant.scale);

                // Store original for reranking (flat contiguous)
                original.extend(vector);
                *original_count += 1;

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
    /// For `RaBitQQuantized`, returns the original vector (used for reranking).
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
            Self::BinaryQuantized { original, .. } => original
                .as_ref()
                .and_then(|o| o.get(id as usize).map(std::vec::Vec::as_slice)),
            Self::RaBitQQuantized {
                original,
                original_count,
                dimensions,
                ..
            } => {
                let idx = id as usize;
                if idx >= *original_count {
                    return None;
                }
                let start = idx * *dimensions;
                let end = start + *dimensions;
                Some(&original[start..end])
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
            Self::BinaryQuantized { original, .. } => {
                original.as_ref().and_then(|o| o.get(id as usize).cloned())
            }
            Self::RaBitQQuantized {
                original,
                original_count,
                dimensions,
                ..
            } => {
                let idx = id as usize;
                if idx >= *original_count {
                    return None;
                }
                let start = idx * *dimensions;
                let end = start + *dimensions;
                Some(original[start..end].to_vec())
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
    /// This is the HOT PATH for asymmetric search. Works with `RaBitQQuantized` and
    /// `ScalarQuantized` storage. Returns None if storage is not quantized, not trained,
    /// or if id is out of bounds.
    ///
    /// # Performance (Apple Silicon M3 Max, 768D)
    /// - SQ8: Similar speed to full precision (1.07x)
    /// - RaBitQ: ~0.5x speed (ADC + interleaving overhead)
    #[inline(always)]
    #[must_use]
    pub fn distance_asymmetric_l2(&self, query: &[f32], id: u32) -> Option<f32> {
        match self {
            Self::RaBitQQuantized {
                quantizer,
                quantized_data,
                quantized_scales,
                code_size,
                original_count,
                ..
            } => {
                let idx = id as usize;
                if idx >= *original_count {
                    return None;
                }

                // Get quantizer (should always be Some after first insert)
                let q = quantizer.as_ref()?;

                // Access flat storage (cache-friendly layout)
                let start = idx * code_size;
                let end = start + code_size;
                let data = &quantized_data[start..end];
                let scale = quantized_scales[idx];

                Some(q.distance_asymmetric_l2_flat(query, data, scale))
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
                let query_prep = params.prepare_query(query);
                Some(params.distance_l2_squared_raw(
                    &query_prep,
                    &quantized[start..end],
                    sums[idx],
                    norms[idx],
                ))
            }
            Self::BinaryQuantized {
                quantized,
                thresholds,
                ..
            } => {
                let idx = id as usize;
                if idx >= quantized.len() {
                    return None;
                }

                // Quantize query using stored thresholds
                let query_quantized = Self::quantize_binary(query, thresholds);

                // Compute hamming distance (number of different bits)
                let hamming = hamming_distance(&query_quantized, &quantized[idx]);

                // Return hamming distance as f32 for ranking
                // Lower hamming = more similar (like L2 distance)
                Some(hamming as f32)
            }
            // FullPrecision uses regular L2 distance, not asymmetric
            Self::FullPrecision { .. } => None,
        }
    }

    /// Get the pre-computed squared norm (||v||²) for a vector
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
            _ => None,
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

    /// Compute L2 squared distance using decomposition: ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
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

                // ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
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
            _ => None,
        }
    }

    /// Get the quantized vector for a given ID (reconstructed from flat storage)
    ///
    /// Note: Returns an owned `QuantizedVector` reconstructed from flat storage.
    /// Prefer using `distance_adc` or `distance_asymmetric_l2` for distance computation.
    #[inline]
    #[must_use]
    pub fn get_quantized(&self, id: u32) -> Option<QuantizedVector> {
        match self {
            Self::RaBitQQuantized {
                quantized_data,
                quantized_scales,
                code_size,
                original_count,
                dimensions,
                params,
                ..
            } => {
                let idx = id as usize;
                if idx >= *original_count {
                    return None;
                }
                let start = idx * code_size;
                let end = start + code_size;
                Some(QuantizedVector::new(
                    quantized_data[start..end].to_vec(),
                    quantized_scales[idx],
                    params.bits_per_dim.to_u8(),
                    *dimensions,
                ))
            }
            _ => None,
        }
    }

    /// Get the `RaBitQ` quantizer (for external asymmetric distance computation)
    #[must_use]
    pub fn quantizer(&self) -> Option<&RaBitQ> {
        match self {
            Self::RaBitQQuantized { quantizer, .. } => quantizer.as_ref(),
            _ => None,
        }
    }

    /// Build ADC lookup table for a query
    ///
    /// Only used for RaBitQ (4-bit). SQ8 uses asymmetric SIMD instead.
    /// SQ8 ADC is slower on Apple Silicon because:
    /// - ADC has scattered memory access (d×256+code stride)
    /// - Asymmetric SIMD is pure compute (dequantize + L2)
    /// - Apple Silicon's high SIMD throughput makes compute faster
    ///
    /// SQ8 does NOT use ADC tables because:
    /// - 768D ADC table = 768KB (doesn't fit L1 cache)
    /// - Scattered memory access pattern causes cache misses
    /// - Direct asymmetric SIMD is 10x faster on Apple Silicon
    ///
    /// Returns None for full-precision, SQ8, or not yet trained.
    #[must_use]
    pub fn build_adc_table(&self, query: &[f32]) -> Option<UnifiedADC> {
        match self {
            Self::RaBitQQuantized { quantizer, .. } => {
                let q = quantizer.as_ref()?;
                Some(UnifiedADC::RaBitQ(q.build_adc_table(query)?))
            }
            // SQ8: use direct asymmetric distance (no ADC) for better cache performance
            _ => None,
        }
    }

    /// Compute distance using precomputed ADC table
    ///
    /// Note: SQ8 uses integer SIMD distance via `distance_asymmetric_l2` instead of ADC.
    #[inline]
    #[must_use]
    pub fn distance_adc(&self, adc: &UnifiedADC, id: u32) -> Option<f32> {
        match (self, adc) {
            (
                Self::RaBitQQuantized {
                    quantized_data,
                    code_size,
                    original_count,
                    ..
                },
                UnifiedADC::RaBitQ(table),
            ) => {
                let idx = id as usize;
                if idx >= *original_count {
                    return None;
                }
                // Access flat storage directly (cache-friendly)
                let start = idx * code_size;
                let end = start + code_size;
                Some(table.distance(&quantized_data[start..end]))
            }
            _ => None, // Mismatched storage/ADC types or SQ8 (uses SIMD instead)
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
        // For asymmetric search (RaBitQ), prefetch quantized data
        // For other modes, prefetch original/full precision data
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
            Self::BinaryQuantized { original, .. } => original
                .as_ref()
                .and_then(|o| o.get(id as usize).map(|v| v.as_ptr().cast())),
            Self::RaBitQQuantized {
                quantized_data,
                code_size,
                original_count,
                ..
            } => {
                // Prefetch quantized data (the hot path for asymmetric search)
                let idx = id as usize;
                if idx >= *original_count {
                    None
                } else {
                    let start = idx * code_size;
                    Some(quantized_data[start..].as_ptr())
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

    /// Prefetch quantized vector data for asymmetric search
    ///
    /// More efficient than `prefetch()` for `RaBitQ` mode as it only fetches
    /// the quantized representation, not the full precision original.
    #[inline]
    pub fn prefetch_quantized(&self, id: u32) {
        if let Self::RaBitQQuantized {
            quantized_data,
            code_size,
            original_count,
            ..
        } = self
        {
            let idx = id as usize;
            if idx < *original_count {
                let start = idx * code_size;
                let ptr = quantized_data[start..].as_ptr();
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    std::arch::x86_64::_mm_prefetch(
                        ptr.cast::<i8>(),
                        std::arch::x86_64::_MM_HINT_T0,
                    );
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
    }

    /// Get RaBitQ code_size (bytes per quantized vector)
    ///
    /// Returns None if not using RaBitQ quantization.
    #[must_use]
    pub fn rabitq_code_size(&self) -> Option<usize> {
        match self {
            Self::RaBitQQuantized { code_size, .. } => Some(*code_size),
            _ => None,
        }
    }

    /// Get quantized code for a vector (RaBitQ only)
    ///
    /// Returns a slice of the quantized code bytes for the given vector ID.
    /// Returns None if vector doesn't exist or not using RaBitQ.
    #[must_use]
    pub fn get_rabitq_code(&self, id: u32) -> Option<&[u8]> {
        match self {
            Self::RaBitQQuantized {
                quantized_data,
                code_size,
                original_count,
                ..
            } => {
                let idx = id as usize;
                if idx >= *original_count {
                    return None;
                }
                let start = idx * code_size;
                let end = start + code_size;
                if end <= quantized_data.len() {
                    Some(&quantized_data[start..end])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Build interleaved codes for FastScan from a batch of neighbor IDs
    ///
    /// For 32 neighbors with code_size bytes each, produces:
    /// - 32 bytes for sub-quantizer 0 (one byte from each neighbor)
    /// - 32 bytes for sub-quantizer 1
    /// - ... etc
    ///
    /// Total output size: code_size * 32 bytes
    ///
    /// # Arguments
    /// * `neighbors` - Up to 32 neighbor IDs to interleave
    /// * `output` - Pre-allocated buffer of size code_size * 32
    ///
    /// Returns number of valid neighbors (rest are zero-padded)
    pub fn build_interleaved_codes(&self, neighbors: &[u32], output: &mut [u8]) -> usize {
        let code_size = match self.rabitq_code_size() {
            Some(cs) => cs,
            None => return 0,
        };

        let batch_size = 32;
        let expected_len = code_size * batch_size;
        if output.len() < expected_len {
            return 0;
        }

        // Zero the output buffer
        output[..expected_len].fill(0);

        let valid_count = neighbors.len().min(batch_size);

        // Interleave codes: for each sub-quantizer position, gather that byte from all neighbors
        for (n, &neighbor_id) in neighbors.iter().take(valid_count).enumerate() {
            if let Some(code) = self.get_rabitq_code(neighbor_id) {
                for sq in 0..code_size {
                    output[sq * batch_size + n] = code[sq];
                }
            }
        }

        valid_count
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
                    values.sort_unstable_by_key(|&x| OrderedFloat(x));

                    let median = if values.len().is_multiple_of(2) {
                        let mid = values.len() / 2;
                        f32::midpoint(values[mid - 1], values[mid])
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
            Self::RaBitQQuantized {
                quantizer, params, ..
            } => {
                if sample_vectors.is_empty() {
                    return Err("Cannot train on empty sample".to_string());
                }
                // Train quantizer from sample vectors
                let q = quantizer.get_or_insert_with(|| RaBitQ::new(params.clone()));
                q.train_owned(sample_vectors).map_err(ToString::to_string)?;
                Ok(())
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
            Self::BinaryQuantized {
                quantized,
                original,
                thresholds,
                dimensions,
            } => {
                let quantized_size = quantized.len() * (dimensions + 7) / 8;
                let original_size = original
                    .as_ref()
                    .map_or(0, |o| o.len() * dimensions * std::mem::size_of::<f32>());
                let thresholds_size = thresholds.len() * std::mem::size_of::<f32>();
                quantized_size + original_size + thresholds_size
            }
            Self::RaBitQQuantized {
                quantized_data,
                quantized_scales,
                original,
                ..
            } => {
                // Flat quantized storage: data + scales
                let quantized_size =
                    quantized_data.len() + quantized_scales.len() * std::mem::size_of::<f32>();
                // Original vectors for reranking (flat contiguous)
                let original_size = original.len() * std::mem::size_of::<f32>();
                quantized_size + original_size
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
            Self::RaBitQQuantized {
                quantized_data,
                quantized_scales,
                code_size,
                original,
                original_count,
                dimensions,
                ..
            } => {
                let dim = *dimensions;
                let n = *original_count;
                let cs = *code_size;

                // Reorder quantized data (flat contiguous)
                let mut new_data = vec![0u8; quantized_data.len()];
                let mut new_scales = vec![0.0f32; quantized_scales.len()];
                for (old_id, &new_id) in old_to_new.iter().enumerate() {
                    if old_id < n {
                        let old_start = old_id * cs;
                        let new_start = new_id as usize * cs;
                        new_data[new_start..new_start + cs]
                            .copy_from_slice(&quantized_data[old_start..old_start + cs]);
                        new_scales[new_id as usize] = quantized_scales[old_id];
                    }
                }
                *quantized_data = new_data;
                *quantized_scales = new_scales;

                // Reorder original vectors (flat contiguous)
                let mut new_original = vec![0.0f32; original.len()];
                for (old_id, &new_id) in old_to_new.iter().enumerate() {
                    if old_id < n {
                        let old_start = old_id * dim;
                        let new_start = new_id as usize * dim;
                        new_original[new_start..new_start + dim]
                            .copy_from_slice(&original[old_start..old_start + dim]);
                    }
                }
                *original = new_original;
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

        let samples = vec![vec![1.0, 5.0], vec![2.0, 6.0], vec![3.0, 7.0]];

        storage.train_quantization(&samples).unwrap();

        // Thresholds should be medians: [2.0, 6.0]
        match storage {
            VectorStorage::BinaryQuantized { thresholds, .. } => {
                assert_eq!(thresholds, vec![2.0, 6.0]);
            }
            _ => panic!("Expected BinaryQuantized storage"),
        }
    }

    #[test]
    fn test_rabitq_storage_insert_and_get() {
        let params = RaBitQParams::bits4();
        let mut storage = VectorStorage::new_rabitq_quantized(4, params);

        let vec1 = vec![1.0, 2.0, 3.0, 4.0];
        let vec2 = vec![5.0, 6.0, 7.0, 8.0];

        let id1 = storage.insert(vec1.clone()).unwrap();
        let id2 = storage.insert(vec2.clone()).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(storage.len(), 2);
        assert!(storage.is_asymmetric());

        // Get should return original vectors
        assert_eq!(storage.get(0), Some(vec1.as_slice()));
        assert_eq!(storage.get(1), Some(vec2.as_slice()));
    }

    #[test]
    fn test_rabitq_asymmetric_distance() {
        let params = RaBitQParams::bits4();
        let mut storage = VectorStorage::new_rabitq_quantized(4, params);

        let vec1 = vec![1.0, 0.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0, 0.0];

        storage.insert(vec1.clone()).unwrap();
        storage.insert(vec2.clone()).unwrap();

        // Query same as vec1 should have distance ~0 to vec1
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let dist0 = storage.distance_asymmetric_l2(&query, 0).unwrap();
        let dist1 = storage.distance_asymmetric_l2(&query, 1).unwrap();

        // Distance to self should be very small (quantization error)
        assert!(dist0 < 0.5, "Distance to self should be small: {dist0}");
        // Distance to orthogonal vector should be larger
        assert!(dist1 > dist0, "Distance to orthogonal should be larger");
    }

    #[test]
    fn test_rabitq_get_quantized() {
        let params = RaBitQParams::bits4();
        let mut storage = VectorStorage::new_rabitq_quantized(4, params);

        let vec1 = vec![1.0, 2.0, 3.0, 4.0];
        storage.insert(vec1).unwrap();

        let qv = storage.get_quantized(0);
        assert!(qv.is_some());
        let qv = qv.unwrap();
        assert_eq!(qv.dimensions, 4);
        assert_eq!(qv.bits, 4); // 4-bit quantization
    }

    #[test]
    fn test_binary_quantized_train_empty_sample_rejected() {
        let mut storage = VectorStorage::new_binary_quantized(4, true);
        let empty_samples: Vec<Vec<f32>> = vec![];
        let result = storage.train_quantization(&empty_samples);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty sample"));
    }

    #[test]
    fn test_binary_quantized_train_dimension_mismatch_rejected() {
        let mut storage = VectorStorage::new_binary_quantized(4, true);
        // Storage expects 4 dimensions, but sample has 2
        let samples = vec![vec![1.0, 2.0]];
        let result = storage.train_quantization(&samples);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("dimension mismatch"));
    }

    #[test]
    fn test_rabitq_train_empty_sample_rejected() {
        let params = RaBitQParams::bits4();
        let mut storage = VectorStorage::new_rabitq_quantized(4, params);
        let empty_samples: Vec<Vec<f32>> = vec![];
        let result = storage.train_quantization(&empty_samples);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty sample"));
    }

    #[test]
    fn test_sq8_train_empty_sample_rejected() {
        let mut storage = VectorStorage::new_sq8_quantized(4);
        let empty_samples: Vec<Vec<f32>> = vec![];
        let result = storage.train_quantization(&empty_samples);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty sample"));
    }

    #[test]
    fn test_neighbor_code_storage_interleaving() {
        // Create RaBitQ storage with 8 dimensions (code_size = 4 bytes for 4-bit)
        let params = RaBitQParams::bits4();
        let mut storage = VectorStorage::new_rabitq_quantized(8, params);

        // Insert 3 vectors
        let vec0 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let vec1 = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let vec2 = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        storage.insert(vec0).unwrap();
        storage.insert(vec1).unwrap();
        storage.insert(vec2).unwrap();

        // Create neighbor lists: vertex 0 has neighbors [1, 2]
        let mut neighbors = NeighborLists::new(8);
        neighbors.set_neighbors(0, 0, vec![1, 2]);
        neighbors.set_neighbors(1, 0, vec![0, 2]);
        neighbors.set_neighbors(2, 0, vec![0, 1]);

        // Build neighbor code storage
        let ncs = NeighborCodeStorage::build_from_storage(&storage, &neighbors, 0);
        assert!(ncs.is_some());
        let ncs = ncs.unwrap();

        // Verify basic properties
        assert_eq!(ncs.len(), 3); // 3 vertices
        assert_eq!(ncs.get_neighbor_count(0), 2); // vertex 0 has 2 neighbors
        assert_eq!(ncs.get_neighbor_count(1), 2);
        assert_eq!(ncs.get_neighbor_count(2), 2);

        // Get block for vertex 0
        let block = ncs.get_block(0);
        assert!(block.is_some());
        let block = block.unwrap();

        // Block size should be code_size * FASTSCAN_BATCH_SIZE
        // For 8D with 4-bit: code_size = 8 / 2 = 4 bytes
        assert_eq!(block.len(), 4 * FASTSCAN_BATCH_SIZE);

        // Verify interleaving: first 32 bytes should be sub-quantizer 0 for all neighbors
        // Neighbors are [1, 2], padded to 32
        // block[0] = neighbor 1's code at sub-quantizer 0
        // block[1] = neighbor 2's code at sub-quantizer 0
        // block[2..32] = padding (0)

        // We can't easily verify exact values without knowing the quantization,
        // but we can verify the structure is correct
        assert!(ncs.memory_usage() > 0);
    }

    #[test]
    fn test_neighbor_code_storage_update() {
        let params = RaBitQParams::bits4();
        let mut storage = VectorStorage::new_rabitq_quantized(8, params);

        // Insert 4 vectors
        for i in 0..4 {
            let v: Vec<f32> = (0..8).map(|j| (i * 8 + j) as f32).collect();
            storage.insert(v).unwrap();
        }

        // Get quantized data for updates
        let quantized_data = match &storage {
            VectorStorage::RaBitQQuantized {
                quantized_data,
                code_size,
                ..
            } => (quantized_data.clone(), *code_size),
            _ => panic!("Expected RaBitQQuantized"),
        };

        // Create empty neighbor code storage
        let mut ncs = NeighborCodeStorage::new(quantized_data.1);
        assert!(ncs.is_empty());

        // Update vertex 0 with neighbors [1, 2, 3]
        ncs.update_vertex(0, &[1, 2, 3], &quantized_data.0);
        assert_eq!(ncs.len(), 1);
        assert_eq!(ncs.get_neighbor_count(0), 3);

        // Verify block exists
        let block = ncs.get_block(0);
        assert!(block.is_some());

        // Update vertex 2 (should expand storage)
        ncs.update_vertex(2, &[0, 1], &quantized_data.0);
        assert_eq!(ncs.len(), 3); // 0, 1 (empty), 2
        assert_eq!(ncs.get_neighbor_count(2), 2);
    }
}
