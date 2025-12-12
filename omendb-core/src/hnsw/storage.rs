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

use crate::compression::{ADCTable, QuantizedVector, RaBitQ, RaBitQParams, ScalarParams};

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

/// Vector storage (quantized or full precision)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VectorStorage {
    /// Full precision f32 vectors - FLAT CONTIGUOUS STORAGE
    ///
    /// Memory: dimensions * 4 bytes per vector + 4 bytes for norm
    /// Example: 1536D = 6144 bytes per vector + 4 bytes norm
    ///
    /// Vectors stored in single contiguous array for cache efficiency.
    /// Access: vectors[id * dimensions..(id + 1) * dimensions]
    ///
    /// L2 distance decomposition: ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
    /// Precomputed norms give ~8% speedup in distance calculations.
    FullPrecision {
        /// Flat contiguous vector data (all vectors concatenated)
        vectors: Vec<f32>,
        /// Precomputed squared norms (||v||²) for L2 decomposition
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

        /// Quantized vectors (`RaBitQ` format)
        quantized: Vec<QuantizedVector>,

        /// Original vectors for reranking (required for final accuracy)
        /// Stored as flat contiguous array for cache efficiency.
        original: Vec<f32>,

        /// Number of original vectors stored
        original_count: usize,

        /// Vector dimensions
        dimensions: usize,
    },

    /// SQ8 (Scalar Quantization 8-bit) - fast SIMD-accelerated search
    ///
    /// Memory: dimensions bytes per vector (4x compression vs f32)
    /// Example: 768D = 768 bytes per vector
    ///
    /// Performance: 2x faster than f32 due to:
    /// - Direct SIMD int8 operations (not ADC lookup tables)
    /// - 4x less memory bandwidth
    /// - Better cache utilization
    ///
    /// Recall: ~99% with rescoring
    SQ8Quantized {
        /// Scalar quantization parameters (trained min/scale per dimension)
        #[serde(skip)]
        params: Option<ScalarParams>,

        /// Quantized vectors (u8 per dimension, flat contiguous)
        quantized: Vec<u8>,

        /// Original vectors for reranking (flat contiguous)
        original: Vec<f32>,

        /// Number of vectors stored
        count: usize,

        /// Vector dimensions
        dimensions: usize,

        /// Training sample buffer (first N vectors before training)
        #[serde(skip)]
        training_buffer: Vec<Vec<f32>>,

        /// Whether quantizer has been trained
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
        Self::RaBitQQuantized {
            quantizer: Some(RaBitQ::new(params.clone())),
            params,
            quantized: Vec::new(),
            original: Vec::new(),
            original_count: 0,
            dimensions,
        }
    }

    /// Create empty SQ8 quantized storage
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality
    ///
    /// # Performance
    /// - Search: 2x faster than full precision (direct SIMD int8)
    /// - Memory: 4x smaller storage
    /// - Recall: ~99% with rescoring
    #[must_use]
    pub fn new_sq8_quantized(dimensions: usize) -> Self {
        Self::SQ8Quantized {
            params: None,
            quantized: Vec::new(),
            original: Vec::new(),
            count: 0,
            dimensions,
            training_buffer: Vec::new(),
            trained: false,
        }
    }

    /// Check if this storage uses asymmetric search (`RaBitQ` or `SQ8`)
    #[must_use]
    pub fn is_asymmetric(&self) -> bool {
        matches!(
            self,
            Self::RaBitQQuantized { .. } | Self::SQ8Quantized { .. }
        )
    }

    /// Check if this storage uses SQ8 quantization
    #[must_use]
    pub fn is_sq8(&self) -> bool {
        matches!(self, Self::SQ8Quantized { .. })
    }

    /// Get number of vectors stored
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::FullPrecision { count, .. } => *count,
            Self::BinaryQuantized { quantized, .. } => quantized.len(),
            Self::RaBitQQuantized { original_count, .. } => *original_count,
            Self::SQ8Quantized {
                count,
                training_buffer,
                trained,
                ..
            } => {
                if *trained {
                    *count
                } else {
                    training_buffer.len()
                }
            }
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
            | Self::SQ8Quantized { dimensions, .. } => *dimensions,
        }
    }

    /// Insert a full precision vector
    #[allow(clippy::items_after_statements)]
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
                // Compute squared norm for L2 decomposition: ||v||²
                let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
                let id = *count as u32;
                vectors.extend(vector);
                norms.push(norm_sq);
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
                quantized,
                original,
                original_count,
                dimensions,
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

                // Quantize and store
                let quant = q.quantize(&vector);
                let id = *original_count as u32;
                quantized.push(quant);

                // Store original for reranking (flat contiguous)
                original.extend(vector);
                *original_count += 1;

                Ok(id)
            }
            Self::SQ8Quantized {
                params,
                quantized,
                original,
                count,
                dimensions,
                training_buffer,
                trained,
            } => {
                if vector.len() != *dimensions {
                    return Err(format!(
                        "Vector dimension mismatch: expected {}, got {}",
                        dimensions,
                        vector.len()
                    ));
                }

                const TRAINING_SIZE: usize = 256;

                // Train on first batch if not yet trained
                if *trained {
                    // Already trained, quantize and store
                    let p = params.as_ref().unwrap();
                    let id = *count as u32;
                    let q = p.quantize(&vector);
                    quantized.extend(q);
                    original.extend(vector);
                    *count += 1;
                    Ok(id)
                } else {
                    training_buffer.push(vector);

                    if training_buffer.len() >= TRAINING_SIZE {
                        // Train quantizer on collected samples
                        let refs: Vec<&[f32]> = training_buffer
                            .iter()
                            .map(std::vec::Vec::as_slice)
                            .collect();
                        let trained_params = ScalarParams::train(&refs);
                        *params = Some(trained_params);
                        *trained = true;

                        // Quantize all buffered vectors
                        let p = params.as_ref().unwrap();
                        for buffered_vec in training_buffer.drain(..) {
                            let q = p.quantize(&buffered_vec);
                            quantized.extend(q);
                            original.extend(buffered_vec);
                            *count += 1;
                        }

                        return Ok((*count - 1) as u32);
                    }

                    // Still collecting - just buffer, don't quantize yet
                    // Return sequential ID based on buffer position
                    Ok((training_buffer.len() - 1) as u32)
                }
            }
        }
    }

    /// Get a vector by ID (full precision)
    ///
    /// Returns slice directly into contiguous storage - zero-copy, cache-friendly.
    /// For `RaBitQQuantized` and `SQ8Quantized`, returns the original vector (used for reranking).
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
            Self::SQ8Quantized {
                original,
                count,
                dimensions,
                training_buffer,
                trained,
                ..
            } => {
                let idx = id as usize;
                // During pre-training, vectors are in training_buffer
                if !*trained {
                    training_buffer.get(idx).map(std::vec::Vec::as_slice)
                } else if idx >= *count {
                    None
                } else {
                    let start = idx * *dimensions;
                    let end = start + *dimensions;
                    Some(&original[start..end])
                }
            }
        }
    }

    /// Get precomputed squared norm (||v||²) for a vector by ID
    ///
    /// Used for L2 distance decomposition: ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
    /// Returns None if id is out of bounds or storage doesn't support norms.
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
            // Quantized variants don't store norms (they use asymmetric distance)
            _ => None,
        }
    }

    /// Compute asymmetric L2 distance (query full precision, candidate quantized)
    ///
    /// This is the HOT PATH for asymmetric search. Works with `RaBitQQuantized` and `SQ8Quantized`.
    /// Returns None if storage doesn't support asymmetric search or if id is out of bounds.
    ///
    /// # Performance
    /// - SQ8: 2x faster than full precision (direct SIMD int8)
    /// - `RaBitQ`: Uses ADC lookup tables
    #[inline]
    #[must_use]
    pub fn distance_asymmetric_l2(&self, query: &[f32], id: u32) -> Option<f32> {
        match self {
            Self::RaBitQQuantized {
                quantizer,
                quantized,
                ..
            } => {
                let idx = id as usize;
                if idx >= quantized.len() {
                    return None;
                }

                // Get quantizer (should always be Some after first insert)
                let q = quantizer.as_ref()?;

                Some(q.distance_asymmetric_l2(query, &quantized[idx]))
            }
            Self::SQ8Quantized {
                params,
                quantized,
                count,
                dimensions,
                training_buffer,
                trained,
                ..
            } => {
                let idx = id as usize;

                // During pre-training, compute exact L2 from buffer
                if !*trained {
                    let vec = training_buffer.get(idx)?;
                    let dist_sq: f32 = query
                        .iter()
                        .zip(vec.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    return Some(dist_sq.sqrt());
                }

                if idx >= *count {
                    return None;
                }

                // Get params (should always be Some after training)
                let p = params.as_ref()?;

                // Get quantized vector slice
                let start = idx * *dimensions;
                let end = start + *dimensions;
                let quantized_vec = &quantized[start..end];

                // Compute asymmetric L2 distance (SIMD accelerated)
                Some(p.asymmetric_l2_squared(query, quantized_vec).sqrt())
            }
            // For non-quantized storage, return None (caller should use regular distance)
            _ => None,
        }
    }

    /// Get the quantized vector for a given ID (for asymmetric distance in external code)
    #[inline]
    #[must_use]
    pub fn get_quantized(&self, id: u32) -> Option<&QuantizedVector> {
        match self {
            Self::RaBitQQuantized { quantized, .. } => quantized.get(id as usize),
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

    /// Build ADC lookup table for a query (5-10x faster than per-candidate decompression)
    ///
    /// Returns None if:
    /// - Storage is not `RaBitQ` quantized, or
    /// - Quantizer has not been trained
    #[must_use]
    pub fn build_adc_table(&self, query: &[f32]) -> Option<ADCTable> {
        match self {
            Self::RaBitQQuantized { quantizer, .. } => {
                let q = quantizer.as_ref()?;
                // Uses trained per-dimension min/max for correct distances
                q.build_adc_table(query)
            }
            _ => None,
        }
    }

    /// Compute distance using precomputed ADC table
    #[inline]
    #[must_use]
    pub fn distance_adc(&self, adc: &ADCTable, id: u32) -> Option<f32> {
        match self {
            Self::RaBitQQuantized { quantized, .. } => {
                let qv = quantized.get(id as usize)?;
                Some(adc.distance(&qv.data))
            }
            _ => None,
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
    /// Stride prefetch - prefetches multiple cache lines to cover vector data.
    /// VSAG-style optimization: reduces L3 cache miss from ~93% to ~39%.
    ///
    /// For 768D f32 vectors: 3072 bytes = 48 cache lines
    /// We prefetch up to 8 cache lines (512 bytes) per call - enough for SQ8
    /// and the initial portion of f32 vectors. Hardware prefetcher handles the rest.
    #[inline]
    pub fn prefetch(&self, id: u32) {
        let (ptr, bytes) = match self {
            Self::FullPrecision {
                vectors,
                count,
                dimensions,
                ..
            } => {
                let idx = id as usize;
                if idx >= *count {
                    (None, 0)
                } else {
                    let start = idx * *dimensions;
                    let bytes = *dimensions * 4; // f32 = 4 bytes
                    (Some(vectors[start..].as_ptr().cast::<u8>()), bytes)
                }
            }
            Self::BinaryQuantized { original, .. } => {
                let ptr = original
                    .as_ref()
                    .and_then(|o| o.get(id as usize).map(|v| v.as_ptr().cast::<u8>()));
                let bytes = original
                    .as_ref()
                    .and_then(|o| o.first().map(|v| v.len() * 4))
                    .unwrap_or(0);
                (ptr, bytes)
            }
            Self::RaBitQQuantized { quantized, .. } => {
                let entry = quantized.get(id as usize);
                let ptr = entry.map(|q| q.data.as_ptr());
                let bytes = entry.map_or(0, |q| q.data.len());
                (ptr, bytes)
            }
            Self::SQ8Quantized {
                quantized,
                count,
                dimensions,
                ..
            } => {
                let idx = id as usize;
                if idx >= *count {
                    (None, 0)
                } else {
                    let start = idx * *dimensions;
                    (Some(quantized[start..].as_ptr()), *dimensions) // u8 = 1 byte
                }
            }
        };

        if let Some(ptr) = ptr {
            // Calculate cache lines to prefetch (64 bytes each)
            // Limit to 8 cache lines (512 bytes) to avoid cache pollution
            let cache_lines = bytes.div_ceil(64).min(8);

            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                for i in 0..cache_lines {
                    _mm_prefetch(ptr.add(i * 64).cast::<i8>(), _MM_HINT_T0);
                }
            }
            #[cfg(target_arch = "aarch64")]
            unsafe {
                for i in 0..cache_lines {
                    let p = ptr.add(i * 64);
                    std::arch::asm!(
                        "prfm pldl1keep, [{ptr}]",
                        ptr = in(reg) p,
                        options(nostack, preserves_flags)
                    );
                }
            }
        }
    }

    /// Prefetch quantized vector data for asymmetric search (stride version)
    ///
    /// More efficient than `prefetch()` for `RaBitQ`/`SQ8` mode as it only fetches
    /// the quantized representation, not the full precision original.
    /// Uses stride prefetching to cover the full quantized vector.
    #[inline]
    pub fn prefetch_quantized(&self, id: u32) {
        let (ptr, bytes) = match self {
            Self::RaBitQQuantized { quantized, .. } => {
                let entry = quantized.get(id as usize);
                let ptr = entry.map(|q| q.data.as_ptr());
                let bytes = entry.map_or(0, |q| q.data.len());
                (ptr, bytes)
            }
            Self::SQ8Quantized {
                quantized,
                count,
                dimensions,
                ..
            } => {
                let idx = id as usize;
                if idx >= *count {
                    (None, 0)
                } else {
                    let start = idx * *dimensions;
                    (Some(quantized[start..].as_ptr()), *dimensions)
                }
            }
            _ => (None, 0),
        };

        if let Some(ptr) = ptr {
            // Quantized vectors are smaller - prefetch up to 4 cache lines
            let cache_lines = bytes.div_ceil(64).min(4);

            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                for i in 0..cache_lines {
                    _mm_prefetch(ptr.add(i * 64).cast::<i8>(), _MM_HINT_T0);
                }
            }
            #[cfg(target_arch = "aarch64")]
            unsafe {
                for i in 0..cache_lines {
                    let p = ptr.add(i * 64);
                    std::arch::asm!(
                        "prfm pldl1keep, [{ptr}]",
                        ptr = in(reg) p,
                        options(nostack, preserves_flags)
                    );
                }
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
                q.train_owned(sample_vectors);
                Ok(())
            }
            Self::SQ8Quantized {
                params,
                dimensions,
                trained,
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
                // Train scalar quantizer from sample vectors
                let refs: Vec<&[f32]> =
                    sample_vectors.iter().map(std::vec::Vec::as_slice).collect();
                let trained_params = ScalarParams::train(&refs);
                *params = Some(trained_params);
                *trained = true;
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
                quantized,
                original,
                ..
            } => {
                // Quantized vectors: data + scale + bits fields
                let quantized_size: usize = quantized
                    .iter()
                    .map(|q| q.data.len() + std::mem::size_of::<f32>() + 1) // data + scale + bits
                    .sum();
                // Original vectors for reranking (flat contiguous)
                let original_size = original.len() * std::mem::size_of::<f32>();
                quantized_size + original_size
            }
            Self::SQ8Quantized {
                quantized,
                original,
                params,
                ..
            } => {
                // Quantized vectors: u8 per dimension
                let quantized_size = quantized.len();
                // Original vectors for reranking (flat contiguous)
                let original_size = original.len() * std::mem::size_of::<f32>();
                // Params: mins and scales vectors
                let params_size = params
                    .as_ref()
                    .map_or(0, |p| p.mins.len() * std::mem::size_of::<f32>() * 2);
                quantized_size + original_size + params_size
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
                quantized,
                original,
                original_count,
                dimensions,
                ..
            } => {
                let dim = *dimensions;
                let n = *original_count;

                // Reorder quantized vectors
                let mut new_quantized: Vec<QuantizedVector> = Vec::with_capacity(quantized.len());
                for _ in 0..quantized.len() {
                    // Placeholder - will be replaced
                    new_quantized.push(QuantizedVector::new(Vec::new(), 1.0, 4, dim));
                }
                for (old_id, &new_id) in old_to_new.iter().enumerate() {
                    if old_id < quantized.len() {
                        new_quantized[new_id as usize] = std::mem::replace(
                            &mut quantized[old_id],
                            QuantizedVector::new(Vec::new(), 1.0, 4, dim),
                        );
                    }
                }
                *quantized = new_quantized;

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
            Self::SQ8Quantized {
                quantized,
                original,
                count,
                dimensions,
                ..
            } => {
                let dim = *dimensions;
                let n = *count;

                // Reorder quantized vectors (flat contiguous u8)
                let mut new_quantized = vec![0u8; quantized.len()];
                for (old_id, &new_id) in old_to_new.iter().enumerate() {
                    if old_id < n {
                        let old_start = old_id * dim;
                        let new_start = new_id as usize * dim;
                        new_quantized[new_start..new_start + dim]
                            .copy_from_slice(&quantized[old_start..old_start + dim]);
                    }
                }
                *quantized = new_quantized;

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
        assert_eq!(qv.unwrap().dimensions, 4);
        assert_eq!(qv.unwrap().bits, 4); // 4-bit quantization
    }

    #[test]
    fn test_sq8_storage_insert_and_get() {
        let mut storage = VectorStorage::new_sq8_quantized(4);

        let vec1 = vec![1.0, 2.0, 3.0, 4.0];
        let vec2 = vec![5.0, 6.0, 7.0, 8.0];

        let id1 = storage.insert(vec1.clone()).unwrap();
        let id2 = storage.insert(vec2.clone()).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(storage.len(), 2);
        assert!(storage.is_asymmetric());
        assert!(storage.is_sq8());

        // Get should return original vectors
        assert_eq!(storage.get(0), Some(vec1.as_slice()));
        assert_eq!(storage.get(1), Some(vec2.as_slice()));
    }

    #[test]
    fn test_sq8_asymmetric_distance() {
        let mut storage = VectorStorage::new_sq8_quantized(4);

        // Insert enough vectors to trigger training (normalized [0,1] range)
        for i in 0..300 {
            let t = i as f32 / 300.0;
            let vec = vec![t, t + 0.1, t + 0.2, t + 0.3];
            storage.insert(vec).unwrap();
        }

        // Insert test vectors in middle of trained distribution
        let vec1 = vec![0.5, 0.5, 0.5, 0.5];
        let vec2 = vec![0.8, 0.8, 0.8, 0.8];

        let id1 = storage.insert(vec1.clone()).unwrap();
        let id2 = storage.insert(vec2.clone()).unwrap();

        // Query same as vec1 should have small distance
        let query = vec![0.5, 0.5, 0.5, 0.5];
        let dist1 = storage.distance_asymmetric_l2(&query, id1).unwrap();
        let dist2 = storage.distance_asymmetric_l2(&query, id2).unwrap();

        // Distance to similar vector should be small (quantization error)
        assert!(dist1 < 0.1, "Distance to similar should be small: {dist1}");
        // Distance to different vector should be larger
        assert!(
            dist2 > dist1,
            "Distance to different should be larger: {dist2} vs {dist1}"
        );
    }

    #[test]
    fn test_sq8_memory_compression() {
        let mut storage = VectorStorage::new_sq8_quantized(768);

        // Insert >= 256 vectors to trigger training
        for i in 0..300 {
            let vec: Vec<f32> = (0..768).map(|j| (i + j) as f32 / 768.0).collect();
            storage.insert(vec).unwrap();
        }

        // SQ8 stores: quantized (768 bytes) + original (768*4 bytes) per vector
        // Total: 300 * (768 + 768*4) = 300 * 3840 = 1,152,000 bytes
        let mem = storage.memory_usage();
        assert!(mem > 0, "Memory usage should be positive");

        // Verify compression: quantized should be 4x smaller than f32
        // quantized: 300 * 768 = 230,400 bytes
        // original: 300 * 768 * 4 = 921,600 bytes
        // Total: ~1.15 MB (vs 1.84 MB for full f32 storage of 2 copies)
        let quantized_size = 300 * 768;
        let original_size = 300 * 768 * 4;
        assert!(
            mem >= quantized_size + original_size - 1000,
            "Memory should include both quantized and original: {mem}"
        );
    }

    #[test]
    fn test_sq8_pretraining_access() {
        let mut storage = VectorStorage::new_sq8_quantized(4);

        // Insert < 256 vectors (pre-training phase)
        for i in 0..100 {
            let vec = vec![i as f32, i as f32, i as f32, i as f32];
            let id = storage.insert(vec).unwrap();
            assert_eq!(id, i as u32, "ID should be sequential");
        }

        // Should be able to get vectors during pre-training
        let v0 = storage.get(0).unwrap();
        assert_eq!(v0, &[0.0, 0.0, 0.0, 0.0]);

        let v50 = storage.get(50).unwrap();
        assert_eq!(v50, &[50.0, 50.0, 50.0, 50.0]);

        // Distance should work (exact L2 during pre-training)
        let query = vec![0.0, 0.0, 0.0, 0.0];
        let dist_to_0 = storage.distance_asymmetric_l2(&query, 0).unwrap();
        let dist_to_50 = storage.distance_asymmetric_l2(&query, 50).unwrap();
        assert!(dist_to_0 < 0.01, "Distance to same vector: {dist_to_0}");
        assert!(dist_to_50 > dist_to_0, "Distance ordering");

        // len() should report correct count
        assert_eq!(storage.len(), 100);
    }

    #[test]
    fn test_sq8_recall() {
        let mut storage = VectorStorage::new_sq8_quantized(128);

        // Insert training vectors
        let mut vectors = Vec::new();
        for i in 0..500 {
            let vec: Vec<f32> = (0..128)
                .map(|j| ((i * 7 + j) % 100) as f32 / 100.0)
                .collect();
            vectors.push(vec.clone());
            storage.insert(vec).unwrap();
        }

        // Test recall: for each vector, distance to itself should be small
        let mut recall_errors = 0;
        for (id, orig) in vectors.iter().enumerate() {
            let dist = storage.distance_asymmetric_l2(orig, id as u32).unwrap();
            // Quantization error should be small for vectors within training distribution
            if dist > 0.5 {
                recall_errors += 1;
            }
        }

        // Expect < 5% recall errors (quantization distortion)
        let error_rate = recall_errors as f32 / vectors.len() as f32;
        assert!(
            error_rate < 0.05,
            "Too many recall errors: {recall_errors}/{} ({:.1}%)",
            vectors.len(),
            error_rate * 100.0
        );
    }

    #[test]
    #[ignore] // Run with: cargo test --release -p omendb-core -- --ignored sq8_speed
    fn test_sq8_speed_vs_f32() {
        use std::time::Instant;

        let dims = 768;
        let n_vectors = 1000;
        let n_queries = 100;

        // Generate vectors (normalized [0,1] range)
        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|i| {
                (0..dims)
                    .map(|j| ((i * 7 + j) % 1000) as f32 / 1000.0)
                    .collect()
            })
            .collect();
        let queries: Vec<Vec<f32>> = (0..n_queries)
            .map(|i| {
                (0..dims)
                    .map(|j| ((i * 13 + j) % 1000) as f32 / 1000.0)
                    .collect()
            })
            .collect();

        // Benchmark f32 L2 distance
        let start = Instant::now();
        let mut sum = 0.0f32;
        for q in &queries {
            for v in &vectors {
                let dist: f32 = q.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                sum += dist.sqrt();
            }
        }
        let f32_time = start.elapsed();
        eprintln!("f32 L2: {:?} (checksum={:.2})", f32_time, sum);

        // Setup SQ8 storage
        let mut storage = VectorStorage::new_sq8_quantized(dims);
        for v in &vectors {
            storage.insert(v.clone()).unwrap();
        }

        // Benchmark SQ8 asymmetric distance
        let start = Instant::now();
        sum = 0.0;
        for q in &queries {
            for id in 0..n_vectors as u32 {
                if let Some(dist) = storage.distance_asymmetric_l2(q, id) {
                    sum += dist;
                }
            }
        }
        let sq8_time = start.elapsed();
        eprintln!("SQ8 L2: {:?} (checksum={:.2})", sq8_time, sum);

        let speedup = f32_time.as_secs_f64() / sq8_time.as_secs_f64();
        eprintln!("\nSpeedup: {:.2}x", speedup);

        // We expect SQ8 to be faster than f32 (target: 1.5-2x)
        assert!(
            speedup > 1.0,
            "SQ8 should be faster than f32, got {:.2}x",
            speedup
        );
    }
}
