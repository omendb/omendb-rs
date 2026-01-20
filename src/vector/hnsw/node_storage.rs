//! Unified node storage for HNSW - THE storage layer
//!
//! Architecture:
//! - Level 0: Colocated vectors + neighbors, cache-line aligned (hot path, 95% of operations)
//! - Upper levels: Sparse storage, only allocated for nodes with level > 0 (cold path)
//! - Supports both full precision (f32) and SQ8 quantization (u8)
//!
//! Level 0 node layout in memory (fixed size per index):
//!
//! Full precision:
//! ```text
//! [neighbor_count: u16][pad: u16][neighbors: [u32; M*2]][vector: [f32; D]][slot: u32][level: u8][padding]
//! ```
//! Total: 4 + 4*(M*2) + 4*D + 5 + padding bytes (rounded to cache line)
//!
//! SQ8 quantized:
//! ```text
//! [neighbor_count: u16][pad: u16][neighbors: [u32; M*2]][quantized: [u8; D]][slot: u32][level: u8][padding]
//! ```
//! Total: 4 + 4*(M*2) + D + 5 + padding bytes (4x smaller vectors)
//! Plus separate norms and sums arrays for L2 decomposition.
//!
//! Benefits:
//! - Single prefetch covers both neighbors and vector (level 0)
//! - Zero-copy neighbor access (no buffer copy)
//! - Cache-line aligned node access
//! - Sparse upper levels (memory efficient, only 5% of nodes)
//! - 2-3x faster search at high dimensions (768D+)
//! - SQ8: 4x memory reduction, 2-3x faster (integer SIMD)
//!
//! All fields after the count are 4-byte aligned, ensured by the 2-byte padding after count.

// Allow pointer casts - we ensure alignment via layout design (all offsets are 4-byte aligned)
#![allow(clippy::cast_ptr_alignment)]

use crate::compression::scalar::ScalarParams;
use rustc_hash::FxHashMap;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fmt;
use std::ptr::NonNull;

// Re-export QueryPrep for use by callers
pub use crate::compression::scalar::QueryPrep;

/// Storage mode for vectors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StorageMode {
    /// Full precision f32 vectors (D * 4 bytes per vector)
    #[default]
    FullPrecision,
    /// SQ8 quantized vectors (D bytes per vector, 4x compression)
    SQ8,
}

/// Cache-line alignment for optimal prefetch
const CACHE_LINE: usize = 64;

/// Storage backing type
enum StorageBacking {
    /// Owned heap allocation
    Owned {
        data: NonNull<u8>,
        layout: Layout,
        capacity: usize,
    },
    /// Memory-mapped file (read-only)
    #[cfg(feature = "mmap")]
    Mmap(memmap2::Mmap),
}

impl Default for StorageBacking {
    fn default() -> Self {
        StorageBacking::Owned {
            data: NonNull::dangling(),
            layout: Layout::from_size_align(0, 1).unwrap(),
            capacity: 0,
        }
    }
}

impl fmt::Debug for StorageBacking {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageBacking::Owned { capacity, .. } => {
                write!(f, "Owned {{ capacity: {capacity} }}")
            }
            #[cfg(feature = "mmap")]
            StorageBacking::Mmap(mmap) => write!(f, "Mmap {{ len: {} }}", mmap.len()),
        }
    }
}

/// Unified storage with colocated vectors and neighbors
///
/// This storage format places vectors and neighbors together in memory
/// so that a single cache prefetch covers both. This significantly improves
/// search performance by reducing cache misses during graph traversal.
///
/// Level 0 is stored colocated (hot path). Upper levels are stored sparsely
/// (only ~5% of nodes have upper levels).
///
/// Supports both full precision (f32) and SQ8 quantized (u8) vectors.
pub struct NodeStorage {
    /// Level 0 storage backing (owned or mmap) - colocated vectors + neighbors
    backing: StorageBacking,
    /// Number of nodes in use
    len: usize,
    /// Size of each node in bytes (cache-line aligned)
    node_size: usize,
    /// Offset to neighbors array (after u16 count)
    neighbors_offset: usize,
    /// Offset to vector data (after neighbors)
    vector_offset: usize,
    /// Offset to metadata (slot, level)
    metadata_offset: usize,
    /// Vector dimensions
    dimensions: usize,
    /// Max neighbors at level 0 (M * 2)
    max_neighbors: usize,
    /// Max neighbors at upper levels (M)
    max_neighbors_upper: usize,
    /// Max level supported
    max_level: usize,
    /// Upper level neighbors: node_id -> [level-1] -> neighbors
    /// Only populated for nodes with level > 0 (~5% of nodes).
    /// Using HashMap instead of Vec<Option<...>> saves ~7MB at 1M vectors.
    /// Using Vec<Vec<u32>> instead of Box<[Vec<u32>]> allows in-place mutation.
    upper_neighbors: FxHashMap<u32, Vec<Vec<u32>>>,

    // SQ8 quantization support
    /// Storage mode (full precision or SQ8)
    mode: StorageMode,
    /// Scalar quantization parameters (only for SQ8 mode)
    sq8_params: Option<ScalarParams>,
    /// Squared norms for each vector (used in L2 decomposition)
    norms: Vec<f32>,
    /// Sum of quantized codes (only for SQ8 mode, used in L2 decomposition)
    sq8_sums: Vec<i32>,
    /// Training buffer for lazy SQ8 training (first 256 vectors)
    training_buffer: Vec<f32>,
    /// Whether SQ8 quantization has been trained
    sq8_trained: bool,
}

impl fmt::Debug for NodeStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeStorage")
            .field("len", &self.len)
            .field("dimensions", &self.dimensions)
            .field("max_neighbors", &self.max_neighbors)
            .field("mode", &self.mode)
            .field("sq8_trained", &self.sq8_trained)
            .finish_non_exhaustive()
    }
}

impl NodeStorage {
    /// Create new full precision storage
    ///
    /// # Arguments
    /// - `dimensions`: Vector dimensionality
    /// - `m`: HNSW M parameter (level 0 gets M*2 neighbors, upper levels get M)
    /// - `max_levels`: Maximum number of levels in the HNSW graph
    #[must_use]
    pub fn new(dimensions: usize, m: usize, max_levels: usize) -> Self {
        Self::with_mode(dimensions, m, max_levels, StorageMode::FullPrecision)
    }

    /// Create new SQ8 quantized storage
    ///
    /// SQ8 provides:
    /// - 4x memory reduction (u8 instead of f32)
    /// - 2-3x faster search (integer SIMD)
    /// - ~99% recall with L2 decomposition
    ///
    /// Quantization is trained lazily after 256 vectors are inserted.
    #[must_use]
    pub fn new_sq8(dimensions: usize, m: usize, max_levels: usize) -> Self {
        Self::with_mode(dimensions, m, max_levels, StorageMode::SQ8)
    }

    /// Create storage with specified mode
    #[must_use]
    fn with_mode(dimensions: usize, m: usize, max_levels: usize, mode: StorageMode) -> Self {
        let max_neighbors = m * 2; // Level 0 gets M*2
        let max_neighbors_upper = m; // Upper levels get M

        // Layout: [count:2][pad:2][neighbors:M*2*4][vector][slot:4][level:1]
        // Vector size depends on mode: f32 (4 bytes) vs u8 (1 byte)
        let neighbors_offset = 4; // 2 (count) + 2 (padding) = 4
        let vector_offset = neighbors_offset + max_neighbors * 4;
        let vector_size = match mode {
            StorageMode::FullPrecision => dimensions * 4, // f32
            StorageMode::SQ8 => dimensions,               // u8
        };
        let metadata_offset = vector_offset + vector_size;
        let raw_size = metadata_offset + 4 + 1; // slot (4) + level (1)

        // Round up to cache line boundary for alignment
        let node_size = raw_size.div_ceil(CACHE_LINE) * CACHE_LINE;

        Self {
            backing: StorageBacking::default(),
            len: 0,
            node_size,
            neighbors_offset,
            vector_offset,
            metadata_offset,
            dimensions,
            max_neighbors,
            max_neighbors_upper,
            max_level: max_levels,
            upper_neighbors: FxHashMap::default(),
            mode,
            sq8_params: None,
            norms: Vec::new(),
            sq8_sums: Vec::new(),
            training_buffer: Vec::new(),
            sq8_trained: false,
        }
    }

    /// Node size in bytes
    #[inline]
    #[must_use]
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// Vector dimensions
    #[inline]
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Max neighbors per node at level 0
    #[inline]
    #[must_use]
    pub fn max_neighbors(&self) -> usize {
        self.max_neighbors
    }

    /// Max neighbors per node at upper levels
    #[inline]
    #[must_use]
    pub fn max_neighbors_upper(&self) -> usize {
        self.max_neighbors_upper
    }

    /// Maximum level supported
    #[inline]
    #[must_use]
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Storage mode (full precision or SQ8)
    #[inline]
    #[must_use]
    pub fn mode(&self) -> StorageMode {
        self.mode
    }

    /// Check if this storage uses SQ8 quantization
    #[inline]
    #[must_use]
    pub fn is_sq8(&self) -> bool {
        self.mode == StorageMode::SQ8
    }

    /// Check if SQ8 quantization is trained (only relevant for SQ8 mode)
    #[inline]
    #[must_use]
    pub fn is_trained(&self) -> bool {
        self.mode == StorageMode::FullPrecision || self.sq8_trained
    }

    /// Number of nodes
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get current capacity
    #[inline]
    fn capacity(&self) -> usize {
        match &self.backing {
            StorageBacking::Owned { capacity, .. } => *capacity,
            #[cfg(feature = "mmap")]
            StorageBacking::Mmap(mmap) => mmap.len() / self.node_size,
        }
    }

    /// Allocate a new node, returns node ID
    pub fn allocate_node(&mut self) -> u32 {
        if self.len >= self.capacity() {
            self.grow();
        }
        let node_id = self.len as u32;
        self.len += 1;
        // Upper neighbors allocated on-demand via allocate_upper_levels()
        node_id
    }

    /// Allocate upper level storage for a node
    ///
    /// Called when a node is assigned a level > 0. If the node already has
    /// upper levels allocated but needs more, extends the storage.
    pub fn allocate_upper_levels(&mut self, id: u32, level: u8) {
        if level == 0 {
            return;
        }

        let needed_levels = level as usize;

        match self.upper_neighbors.entry(id) {
            std::collections::hash_map::Entry::Vacant(e) => {
                // Allocate empty Vec for each upper level (levels 1..=level)
                let levels: Vec<Vec<u32>> = (0..needed_levels).map(|_| Vec::new()).collect();
                e.insert(levels);
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                if e.get().len() < needed_levels {
                    // Extend existing allocation in place
                    e.get_mut().resize(needed_levels, Vec::new());
                }
            }
        }
    }

    /// Grow capacity (double or initial 64)
    fn grow(&mut self) {
        let (old_data, old_layout, old_capacity) = match &self.backing {
            StorageBacking::Owned {
                data,
                layout,
                capacity,
            } => (*data, *layout, *capacity),
            #[cfg(feature = "mmap")]
            StorageBacking::Mmap(_) => panic!("Cannot grow mmap-backed storage"),
        };

        let new_capacity = if old_capacity == 0 {
            64
        } else {
            old_capacity * 2
        };
        let new_size = new_capacity * self.node_size;
        let new_layout = Layout::from_size_align(new_size, CACHE_LINE).expect("Invalid layout");

        // SAFETY: We're allocating zeroed memory with valid layout
        let new_ptr = unsafe {
            let ptr = alloc_zeroed(new_layout);
            NonNull::new(ptr).expect("Allocation failed")
        };

        // Copy old data if any
        if old_capacity > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    old_data.as_ptr(),
                    new_ptr.as_ptr(),
                    self.len * self.node_size,
                );
                dealloc(old_data.as_ptr(), old_layout);
            }
        }

        self.backing = StorageBacking::Owned {
            data: new_ptr,
            layout: new_layout,
            capacity: new_capacity,
        };
    }

    /// Get pointer to node data
    #[inline]
    fn node_ptr(&self, id: u32) -> *const u8 {
        debug_assert!(
            (id as usize) < self.len,
            "Node ID {} out of bounds (len={})",
            id,
            self.len
        );
        let base = match &self.backing {
            StorageBacking::Owned { data, .. } => data.as_ptr(),
            #[cfg(feature = "mmap")]
            StorageBacking::Mmap(mmap) => mmap.as_ptr(),
        };
        unsafe { base.add(id as usize * self.node_size) }
    }

    /// Get mutable pointer to node data
    #[inline]
    fn node_ptr_mut(&mut self, id: u32) -> *mut u8 {
        debug_assert!(
            (id as usize) < self.len,
            "Node ID {} out of bounds (len={})",
            id,
            self.len
        );
        let data = match &self.backing {
            StorageBacking::Owned { data, .. } => *data,
            #[cfg(feature = "mmap")]
            StorageBacking::Mmap(_) => panic!("Cannot mutate mmap-backed storage"),
        };
        unsafe { data.as_ptr().add(id as usize * self.node_size) }
    }

    /// Zero-copy access to vector (full precision mode only)
    ///
    /// Panics in SQ8 mode - use `get_dequantized()` instead.
    #[inline]
    #[must_use]
    pub fn vector(&self, id: u32) -> &[f32] {
        debug_assert!(
            self.mode == StorageMode::FullPrecision,
            "vector() not available in SQ8 mode, use get_dequantized()"
        );
        let ptr = self.node_ptr(id);
        unsafe {
            let vec_ptr = ptr.add(self.vector_offset) as *const f32;
            std::slice::from_raw_parts(vec_ptr, self.dimensions)
        }
    }

    /// Zero-copy access to quantized vector (SQ8 mode only)
    #[inline]
    #[must_use]
    pub fn quantized_vector(&self, id: u32) -> &[u8] {
        debug_assert!(
            self.mode == StorageMode::SQ8,
            "quantized_vector() only available in SQ8 mode"
        );
        let ptr = self.node_ptr(id);
        unsafe {
            let vec_ptr = ptr.add(self.vector_offset);
            std::slice::from_raw_parts(vec_ptr, self.dimensions)
        }
    }

    /// Get dequantized vector (handles both trained and untrained SQ8 mode)
    ///
    /// In full precision mode, returns the vector directly.
    /// In SQ8 mode before training, returns the vector from the training buffer.
    /// In SQ8 mode after training, returns the dequantized vector.
    #[must_use]
    pub fn get_dequantized(&self, id: u32) -> Option<Vec<f32>> {
        if self.mode != StorageMode::SQ8 {
            return Some(self.vector(id).to_vec());
        }

        let id_usize = id as usize;

        // If SQ8 trained, dequantize from storage
        if self.sq8_trained {
            let params = self.sq8_params.as_ref()?;
            let quantized = self.quantized_vector(id);
            return Some(params.dequantize(quantized));
        }

        // Not trained yet - get from training buffer
        let dim = self.dimensions;
        let start = id_usize * dim;
        let end = start + dim;
        if end <= self.training_buffer.len() {
            Some(self.training_buffer[start..end].to_vec())
        } else {
            None
        }
    }

    /// Get squared norm for a vector (used in L2 decomposition)
    #[inline]
    #[must_use]
    pub fn get_norm(&self, id: u32) -> Option<f32> {
        self.norms.get(id as usize).copied()
    }

    /// Set vector data (handles both full precision and SQ8 modes)
    pub fn set_vector(&mut self, id: u32, vector: &[f32]) {
        debug_assert_eq!(
            vector.len(),
            self.dimensions,
            "Vector length {} doesn't match dimensions {}",
            vector.len(),
            self.dimensions
        );

        match self.mode {
            StorageMode::FullPrecision => {
                // Store vector directly and compute norm
                let ptr = self.node_ptr_mut(id);
                unsafe {
                    let vec_ptr = ptr.add(self.vector_offset) as *mut f32;
                    std::ptr::copy_nonoverlapping(vector.as_ptr(), vec_ptr, self.dimensions);
                }
                // Compute and store squared norm
                let norm_sq: f32 = vector.iter().map(|&x| x * x).sum();
                let id_usize = id as usize;
                if id_usize >= self.norms.len() {
                    self.norms.resize(id_usize + 1, 0.0);
                }
                self.norms[id_usize] = norm_sq;
            }
            StorageMode::SQ8 => {
                self.set_vector_sq8(id, vector);
            }
        }
    }

    /// Set vector in SQ8 mode with lazy training
    fn set_vector_sq8(&mut self, id: u32, vector: &[f32]) {
        let id_usize = id as usize;

        if self.sq8_trained {
            // Already trained - quantize directly
            let params = self.sq8_params.as_ref().expect("SQ8 params should exist");
            let quant = params.quantize(vector);

            // Store quantized vector
            let ptr = self.node_ptr_mut(id);
            unsafe {
                let vec_ptr = ptr.add(self.vector_offset);
                std::ptr::copy_nonoverlapping(quant.data.as_ptr(), vec_ptr, self.dimensions);
            }

            // Store norm and sum
            if id_usize >= self.norms.len() {
                self.norms.resize(id_usize + 1, 0.0);
            }
            if id_usize >= self.sq8_sums.len() {
                self.sq8_sums.resize(id_usize + 1, 0);
            }
            self.norms[id_usize] = quant.norm_sq;
            self.sq8_sums[id_usize] = quant.sum;
        } else {
            // Still in training phase - buffer the vector
            self.training_buffer.extend_from_slice(vector);

            // Store zeros in the colocated storage for now (will be filled after training)
            let ptr = self.node_ptr_mut(id);
            unsafe {
                let vec_ptr = ptr.add(self.vector_offset);
                std::ptr::write_bytes(vec_ptr, 0, self.dimensions);
            }

            // Check if we have enough vectors to train (256 threshold)
            let num_vectors = self.training_buffer.len() / self.dimensions;
            if num_vectors >= 256 {
                self.train_quantization();
            }
        }
    }

    /// Train SQ8 quantization from buffered vectors
    fn train_quantization(&mut self) {
        let dim = self.dimensions;
        let num_vectors = self.training_buffer.len() / dim;

        // Build training sample (refs to slices)
        let training_refs: Vec<&[f32]> = (0..num_vectors)
            .map(|i| &self.training_buffer[i * dim..(i + 1) * dim])
            .collect();

        // Train quantization parameters
        let params = ScalarParams::train(&training_refs).expect("Failed to train SQ8 params");
        self.sq8_params = Some(params);
        self.sq8_trained = true;

        // Quantize all buffered vectors and store them
        self.norms.reserve(num_vectors);
        self.sq8_sums.reserve(num_vectors);

        for i in 0..num_vectors {
            let vec_slice = &self.training_buffer[i * dim..(i + 1) * dim];
            let quant = params.quantize(vec_slice);

            // Store quantized vector in colocated storage
            let ptr = self.node_ptr_mut(i as u32);
            unsafe {
                let vec_ptr = ptr.add(self.vector_offset);
                std::ptr::copy_nonoverlapping(quant.data.as_ptr(), vec_ptr, dim);
            }

            // Store norm and sum
            if i >= self.norms.len() {
                self.norms.push(quant.norm_sq);
            } else {
                self.norms[i] = quant.norm_sq;
            }
            if i >= self.sq8_sums.len() {
                self.sq8_sums.push(quant.sum);
            } else {
                self.sq8_sums[i] = quant.sum;
            }
        }

        // Clear training buffer
        self.training_buffer.clear();
        self.training_buffer.shrink_to_fit();
    }

    /// Prepare query for SQ8 distance calculation
    #[must_use]
    pub fn prepare_query(&self, query: &[f32]) -> Option<QueryPrep> {
        self.sq8_params
            .as_ref()
            .map(|params| params.prepare_query(query))
    }

    /// Compute SQ8 L2 distance (requires trained quantization)
    ///
    /// Uses integer SIMD for fast distance calculation.
    #[inline]
    #[must_use]
    pub fn distance_sq8(&self, prep: &QueryPrep, id: u32) -> Option<f32> {
        let params = self.sq8_params.as_ref()?;
        if !self.sq8_trained {
            return None;
        }

        let idx = id as usize;
        if idx >= self.len {
            return None;
        }

        let quantized = self.quantized_vector(id);
        let vec_norm_sq = self.norms.get(idx)?;
        let vec_sum = *self.sq8_sums.get(idx)?;

        // L2 decomposition: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        let scale_sq = params.scale * params.scale;
        let offset_term = params.offset * params.offset * self.dimensions as f32;

        let int_dot = params.int_dot_product_pub(&prep.quantized, quantized);

        let dot = scale_sq * int_dot as f32
            + params.scale * params.offset * (prep.sum + vec_sum) as f32
            + offset_term;

        Some(prep.norm_sq + vec_norm_sq - 2.0 * dot)
    }

    /// Batch compute SQ8 L2 distances
    ///
    /// Fills distances buffer with SQ8 distances for the given IDs.
    /// Returns the number of distances computed (some IDs may be out of range).
    #[inline]
    pub fn distance_sq8_batch(
        &self,
        prep: &QueryPrep,
        ids: &[u32],
        distances: &mut [f32],
    ) -> usize {
        let mut count = 0;
        for (&id, dist) in ids.iter().zip(distances.iter_mut()) {
            if let Some(d) = self.distance_sq8(prep, id) {
                *dist = d;
                count += 1;
            }
        }
        count
    }

    /// Get neighbor count at level 0 (hot path, colocated storage)
    #[inline]
    #[must_use]
    pub fn neighbor_count(&self, id: u32) -> usize {
        let ptr = self.node_ptr(id);
        unsafe { u16::from_le_bytes([*ptr, *ptr.add(1)]) as usize }
    }

    /// Get neighbor count at any level
    #[inline]
    #[must_use]
    pub fn neighbor_count_at_level(&self, id: u32, level: u8) -> usize {
        if level == 0 {
            return self.neighbor_count(id);
        }
        match self.upper_neighbors.get(&id) {
            Some(levels) => {
                let level_idx = level as usize - 1;
                if level_idx < levels.len() {
                    levels[level_idx].len()
                } else {
                    0
                }
            }
            None => 0,
        }
    }

    /// Zero-copy access to level 0 neighbors (hot path, colocated storage)
    #[inline]
    #[must_use]
    pub fn neighbors(&self, id: u32) -> &[u32] {
        let count = self.neighbor_count(id);
        if count == 0 {
            return &[];
        }
        // Clamp to max_neighbors to prevent buffer overread on corrupt data
        let count = count.min(self.max_neighbors);
        let ptr = self.node_ptr(id);
        unsafe {
            let neighbors_ptr = ptr.add(self.neighbors_offset) as *const u32;
            std::slice::from_raw_parts(neighbors_ptr, count)
        }
    }

    /// Get neighbors at any level
    ///
    /// Level 0 uses the colocated storage (zero-copy).
    /// Upper levels use the sparse storage.
    #[inline]
    #[must_use]
    pub fn neighbors_at_level(&self, id: u32, level: u8) -> Vec<u32> {
        if level == 0 {
            return self.neighbors(id).to_vec();
        }
        match self.upper_neighbors.get(&id) {
            Some(levels) => {
                let level_idx = level as usize - 1;
                if level_idx < levels.len() {
                    levels[level_idx].clone()
                } else {
                    Vec::new()
                }
            }
            None => Vec::new(),
        }
    }

    /// Set level 0 neighbors (overwrites all, colocated storage)
    pub fn set_neighbors(&mut self, id: u32, neighbors: &[u32]) {
        debug_assert!(
            neighbors.len() <= self.max_neighbors,
            "Too many neighbors: {} > {}",
            neighbors.len(),
            self.max_neighbors
        );
        let ptr = self.node_ptr_mut(id);
        unsafe {
            // Write count
            let count = neighbors.len() as u16;
            let count_bytes = count.to_le_bytes();
            *ptr = count_bytes[0];
            *ptr.add(1) = count_bytes[1];

            // Write neighbors
            if !neighbors.is_empty() {
                let neighbors_ptr = ptr.add(self.neighbors_offset) as *mut u32;
                std::ptr::copy_nonoverlapping(neighbors.as_ptr(), neighbors_ptr, neighbors.len());
            }
        }
    }

    /// Set neighbors at any level
    ///
    /// Level 0 uses colocated storage. Upper levels use sparse storage.
    pub fn set_neighbors_at_level(&mut self, id: u32, level: u8, neighbors: Vec<u32>) {
        if level == 0 {
            self.set_neighbors(id, &neighbors);
            return;
        }

        // Ensure upper levels are allocated
        self.allocate_upper_levels(id, level);

        if let Some(levels) = self.upper_neighbors.get_mut(&id) {
            let level_idx = level as usize - 1;
            if level_idx < levels.len() {
                levels[level_idx] = neighbors;
            }
        }
    }

    /// Add a neighbor at a specific level (for incremental construction)
    pub fn add_neighbor(&mut self, id: u32, level: u8, neighbor: u32) {
        if level == 0 {
            // Level 0: append to colocated storage
            let count = self.neighbor_count(id);
            if count >= self.max_neighbors {
                return; // At capacity
            }
            let ptr = self.node_ptr_mut(id);
            unsafe {
                // Update count
                let new_count = (count + 1) as u16;
                let count_bytes = new_count.to_le_bytes();
                *ptr = count_bytes[0];
                *ptr.add(1) = count_bytes[1];

                // Write new neighbor
                let neighbors_ptr = ptr.add(self.neighbors_offset) as *mut u32;
                *neighbors_ptr.add(count) = neighbor;
            }
        } else {
            // Upper level: append to sparse storage
            self.allocate_upper_levels(id, level);
            if let Some(levels) = self.upper_neighbors.get_mut(&id) {
                let level_idx = level as usize - 1;
                if level_idx < levels.len() && levels[level_idx].len() < self.max_neighbors_upper {
                    levels[level_idx].push(neighbor);
                }
            }
        }
    }

    /// Check if a neighbor exists at a specific level
    ///
    /// Used during parallel construction to avoid duplicate links.
    #[inline]
    #[must_use]
    pub fn contains_neighbor(&self, id: u32, level: u8, neighbor: u32) -> bool {
        if level == 0 {
            self.neighbors(id).contains(&neighbor)
        } else {
            self.neighbors_at_level(id, level).contains(&neighbor)
        }
    }

    /// Try to add a neighbor at a specific level, returns true if added
    ///
    /// Returns false if:
    /// - The neighbor already exists (duplicate)
    /// - The neighbor list is at capacity
    ///
    /// Used during parallel construction for atomic-style neighbor updates.
    pub fn try_add_neighbor(&mut self, id: u32, level: u8, neighbor: u32) -> bool {
        if self.contains_neighbor(id, level, neighbor) {
            return false; // Already exists
        }

        let max = if level == 0 {
            self.max_neighbors
        } else {
            self.max_neighbors_upper
        };

        if self.neighbor_count_at_level(id, level) >= max {
            return false; // At capacity
        }

        self.add_neighbor(id, level, neighbor);
        true
    }

    /// Get slot ID (original RecordStore slot)
    #[inline]
    #[must_use]
    pub fn slot(&self, id: u32) -> u32 {
        let ptr = self.node_ptr(id);
        unsafe {
            let slot_ptr = ptr.add(self.metadata_offset) as *const u32;
            u32::from_le(*slot_ptr)
        }
    }

    /// Set slot ID
    pub fn set_slot(&mut self, id: u32, slot: u32) {
        let ptr = self.node_ptr_mut(id);
        unsafe {
            let slot_ptr = ptr.add(self.metadata_offset) as *mut u32;
            *slot_ptr = slot.to_le();
        }
    }

    /// Get node level
    #[inline]
    #[must_use]
    pub fn level(&self, id: u32) -> u8 {
        let ptr = self.node_ptr(id);
        unsafe { *ptr.add(self.metadata_offset + 4) }
    }

    /// Set node level
    pub fn set_level(&mut self, id: u32, level: u8) {
        let ptr = self.node_ptr_mut(id);
        unsafe {
            *ptr.add(self.metadata_offset + 4) = level;
        }
    }

    /// Prefetch node data into cache
    ///
    /// Call this on nodes you're about to access to hide memory latency.
    /// Uses platform-aware prefetch (disabled on Apple Silicon where DMP handles it).
    #[inline]
    pub fn prefetch(&self, id: u32) {
        use super::prefetch::PrefetchConfig;

        if !PrefetchConfig::enabled() || (id as usize) >= self.len {
            return;
        }

        let ptr = self.node_ptr(id);

        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
        }

        // aarch64 prefetch requires nightly feature, skip for now
        // Apple Silicon's DMP handles prefetching automatically anyway
        #[cfg(not(target_arch = "x86_64"))]
        let _ = ptr;
    }

    /// Memory usage in bytes
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let level0_usage = match &self.backing {
            StorageBacking::Owned { capacity, .. } => capacity * self.node_size,
            #[cfg(feature = "mmap")]
            StorageBacking::Mmap(mmap) => mmap.len(),
        };

        // Calculate upper level storage (HashMap values only)
        let upper_usage: usize = self
            .upper_neighbors
            .values()
            .map(|levels: &Vec<Vec<u32>>| {
                levels.iter().map(|v| v.len() * 4).sum::<usize>()
                    + levels.len() * std::mem::size_of::<Vec<u32>>()
            })
            .sum();

        // SQ8 auxiliary storage
        let sq8_usage = self.norms.len() * 4  // f32 norms
            + self.sq8_sums.len() * 4  // i32 sums
            + self.training_buffer.len() * 4; // f32 training buffer

        level0_usage + upper_usage + sq8_usage
    }

    // =========================================================================
    // Layout accessors (for persistence)
    // =========================================================================

    /// Offset to neighbors array in node layout
    #[inline]
    #[must_use]
    pub fn neighbors_offset(&self) -> usize {
        self.neighbors_offset
    }

    /// Offset to vector data in node layout
    #[inline]
    #[must_use]
    pub fn vector_offset(&self) -> usize {
        self.vector_offset
    }

    /// Offset to metadata in node layout
    #[inline]
    #[must_use]
    pub fn metadata_offset(&self) -> usize {
        self.metadata_offset
    }

    /// Get raw bytes of storage data (for persistence)
    ///
    /// Returns a slice of all node data (len * node_size bytes).
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match &self.backing {
            StorageBacking::Owned { data, .. } => {
                if self.len == 0 {
                    &[]
                } else {
                    unsafe { std::slice::from_raw_parts(data.as_ptr(), self.len * self.node_size) }
                }
            }
            #[cfg(feature = "mmap")]
            StorageBacking::Mmap(mmap) => &mmap[..self.len * self.node_size],
        }
    }

    /// Construct storage from raw bytes (for loading)
    ///
    /// Takes ownership of the data vector.
    ///
    /// # Panics
    /// Panics if parameters are inconsistent with data size.
    #[allow(clippy::too_many_arguments)]
    pub fn from_bytes(
        data: Vec<u8>,
        len: usize,
        node_size: usize,
        neighbors_offset: usize,
        vector_offset: usize,
        metadata_offset: usize,
        dimensions: usize,
        max_neighbors: usize,
    ) -> Self {
        use std::alloc::Layout;

        // Validate parameters to prevent memory safety issues
        let expected_size = len.checked_mul(node_size);
        assert!(
            expected_size.is_some() && expected_size.unwrap() <= data.len(),
            "Invalid segment: len={} * node_size={} exceeds data.len()={}",
            len,
            node_size,
            data.len()
        );
        assert!(
            node_size == 0 || neighbors_offset < node_size,
            "Invalid segment: neighbors_offset {neighbors_offset} >= node_size {node_size}",
        );
        assert!(
            node_size == 0 || vector_offset < node_size,
            "Invalid segment: vector_offset {vector_offset} >= node_size {node_size}",
        );

        let capacity = if node_size > 0 && !data.is_empty() {
            data.len() / node_size
        } else {
            0
        };

        // Convert Vec<u8> to owned allocation with proper alignment
        let backing = if data.is_empty() {
            StorageBacking::default()
        } else {
            // Allocate with CACHE_LINE alignment for optimal performance
            let layout = Layout::from_size_align(data.len(), CACHE_LINE).expect("Invalid layout");
            // SAFETY: We allocate with proper alignment and copy data
            let ptr = unsafe {
                use std::alloc::alloc;
                let raw = alloc(layout);
                if raw.is_null() {
                    std::alloc::handle_alloc_error(layout);
                }
                // Copy data to properly aligned allocation
                std::ptr::copy_nonoverlapping(data.as_ptr(), raw, data.len());
                NonNull::new(raw).expect("Allocation should not return null")
            };
            // data is dropped here, freeing the original unaligned allocation
            StorageBacking::Owned {
                data: ptr,
                layout,
                capacity,
            }
        };

        // M = max_neighbors / 2 (level 0 has M*2)
        let max_neighbors_upper = max_neighbors / 2;

        Self {
            backing,
            len,
            node_size,
            neighbors_offset,
            vector_offset,
            metadata_offset,
            dimensions,
            max_neighbors,
            max_neighbors_upper,
            max_level: 8, // Default max level
            upper_neighbors: FxHashMap::default(),
            mode: StorageMode::FullPrecision, // Default to full precision for loaded data
            sq8_params: None,
            norms: Vec::new(),
            sq8_sums: Vec::new(),
            training_buffer: Vec::new(),
            sq8_trained: false,
        }
    }

    /// Construct storage from memory-mapped file (for mmap loading)
    #[cfg(feature = "mmap")]
    #[allow(clippy::too_many_arguments)]
    pub fn from_mmap(
        mmap: memmap2::Mmap,
        len: usize,
        node_size: usize,
        neighbors_offset: usize,
        vector_offset: usize,
        metadata_offset: usize,
        dimensions: usize,
        max_neighbors: usize,
    ) -> Self {
        let max_neighbors_upper = max_neighbors / 2;

        Self {
            backing: StorageBacking::Mmap(mmap),
            len,
            node_size,
            neighbors_offset,
            vector_offset,
            metadata_offset,
            dimensions,
            max_neighbors,
            max_neighbors_upper,
            max_level: 8,
            upper_neighbors: FxHashMap::default(),
            mode: StorageMode::FullPrecision,
            sq8_params: None,
            norms: Vec::new(),
            sq8_sums: Vec::new(),
            training_buffer: Vec::new(),
            sq8_trained: false,
        }
    }

    /// Serialize complete storage state to bytes
    ///
    /// Format:
    /// - Header: len, node_size, offsets, dimensions, max_neighbors (7 * u64)
    /// - Mode: u8 (0 = FullPrecision, 1 = SQ8)
    /// - SQ8 trained: u8
    /// - Raw node data: len * node_size bytes
    /// - If SQ8: scale, offset (2 * f32), norms (len * f32), sq8_sums (len * i32)
    /// - Upper neighbors count: u64
    /// - For each node with upper neighbors: node_id (u32), num_levels (u8), then for each level: count (u16), neighbors ([u32])
    pub fn serialize_full(&self) -> Vec<u8> {
        let mut out = Vec::new();

        // Header
        out.extend_from_slice(&(self.len as u64).to_le_bytes());
        out.extend_from_slice(&(self.node_size as u64).to_le_bytes());
        out.extend_from_slice(&(self.neighbors_offset as u64).to_le_bytes());
        out.extend_from_slice(&(self.vector_offset as u64).to_le_bytes());
        out.extend_from_slice(&(self.metadata_offset as u64).to_le_bytes());
        out.extend_from_slice(&(self.dimensions as u64).to_le_bytes());
        out.extend_from_slice(&(self.max_neighbors as u64).to_le_bytes());

        // Mode and trained flag
        let mode_byte: u8 = match self.mode {
            StorageMode::FullPrecision => 0,
            StorageMode::SQ8 => 1,
        };
        out.push(mode_byte);
        out.push(u8::from(self.sq8_trained));

        // Raw node data
        let raw_data = self.as_bytes();
        out.extend_from_slice(&(raw_data.len() as u64).to_le_bytes());
        out.extend_from_slice(raw_data);

        // SQ8 params if present
        if let Some(ref params) = self.sq8_params {
            out.push(1); // has params
            out.extend_from_slice(&params.scale.to_le_bytes());
            out.extend_from_slice(&params.offset.to_le_bytes());
        } else {
            out.push(0); // no params
        }

        // Norms (only if SQ8 trained)
        out.extend_from_slice(&(self.norms.len() as u64).to_le_bytes());
        for &norm in &self.norms {
            out.extend_from_slice(&norm.to_le_bytes());
        }

        // SQ8 sums (only if SQ8 trained)
        out.extend_from_slice(&(self.sq8_sums.len() as u64).to_le_bytes());
        for &sum in &self.sq8_sums {
            out.extend_from_slice(&sum.to_le_bytes());
        }

        // Upper neighbors (HashMap - only stores nodes with upper levels)
        out.extend_from_slice(&(self.upper_neighbors.len() as u64).to_le_bytes());

        for (&node_id, levels) in &self.upper_neighbors {
            out.extend_from_slice(&node_id.to_le_bytes());
            out.push(levels.len() as u8); // num levels (excluding level 0)
            for level_neighbors in levels {
                out.extend_from_slice(&(level_neighbors.len() as u16).to_le_bytes());
                for &neighbor in level_neighbors {
                    out.extend_from_slice(&neighbor.to_le_bytes());
                }
            }
        }

        out
    }

    /// Deserialize complete storage state from bytes
    ///
    /// Returns the deserialized storage and the number of bytes consumed.
    pub fn deserialize_full(data: &[u8]) -> Result<Self, String> {
        // Helper to read bytes safely
        fn read_bytes<'a>(data: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], String> {
            if *pos + n > data.len() {
                return Err(format!(
                    "Data too short: need {} bytes at position {}, have {}",
                    n,
                    *pos,
                    data.len()
                ));
            }
            let result = &data[*pos..*pos + n];
            *pos += n;
            Ok(result)
        }

        fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, String> {
            let bytes = read_bytes(data, pos, 8)?;
            Ok(u64::from_le_bytes(
                bytes.try_into().map_err(|_| "Invalid u64")?,
            ))
        }

        fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
            let bytes = read_bytes(data, pos, 4)?;
            Ok(u32::from_le_bytes(
                bytes.try_into().map_err(|_| "Invalid u32")?,
            ))
        }

        fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16, String> {
            let bytes = read_bytes(data, pos, 2)?;
            Ok(u16::from_le_bytes(
                bytes.try_into().map_err(|_| "Invalid u16")?,
            ))
        }

        fn read_f32(data: &[u8], pos: &mut usize) -> Result<f32, String> {
            let bytes = read_bytes(data, pos, 4)?;
            Ok(f32::from_le_bytes(
                bytes.try_into().map_err(|_| "Invalid f32")?,
            ))
        }

        fn read_i32(data: &[u8], pos: &mut usize) -> Result<i32, String> {
            let bytes = read_bytes(data, pos, 4)?;
            Ok(i32::from_le_bytes(
                bytes.try_into().map_err(|_| "Invalid i32")?,
            ))
        }

        fn read_u8(data: &[u8], pos: &mut usize) -> Result<u8, String> {
            let bytes = read_bytes(data, pos, 1)?;
            Ok(bytes[0])
        }

        if data.len() < 58 {
            return Err("Data too short for header".to_string());
        }

        let mut pos = 0;

        // Read header
        let len = read_u64(data, &mut pos)? as usize;
        let node_size = read_u64(data, &mut pos)? as usize;
        let neighbors_offset = read_u64(data, &mut pos)? as usize;
        let vector_offset = read_u64(data, &mut pos)? as usize;
        let metadata_offset = read_u64(data, &mut pos)? as usize;
        let dimensions = read_u64(data, &mut pos)? as usize;
        let max_neighbors = read_u64(data, &mut pos)? as usize;

        // Mode and trained flag
        let mode_byte = read_u8(data, &mut pos)?;
        let mode = match mode_byte {
            0 => StorageMode::FullPrecision,
            1 => StorageMode::SQ8,
            _ => return Err(format!("Invalid storage mode: {mode_byte}")),
        };
        let sq8_trained = read_u8(data, &mut pos)? != 0;

        // Raw node data
        let raw_len = read_u64(data, &mut pos)? as usize;
        let raw_data = read_bytes(data, &mut pos, raw_len)?.to_vec();

        // SQ8 params
        let has_params = read_u8(data, &mut pos)? != 0;
        let sq8_params = if has_params {
            let scale = read_f32(data, &mut pos)?;
            let offset = read_f32(data, &mut pos)?;
            Some(ScalarParams {
                scale,
                offset,
                dimensions,
            })
        } else {
            None
        };

        // Norms
        let norms_len = read_u64(data, &mut pos)? as usize;
        let mut norms = Vec::with_capacity(norms_len);
        for _ in 0..norms_len {
            norms.push(read_f32(data, &mut pos)?);
        }

        // SQ8 sums
        let sums_len = read_u64(data, &mut pos)? as usize;
        let mut sq8_sums = Vec::with_capacity(sums_len);
        for _ in 0..sums_len {
            sq8_sums.push(read_i32(data, &mut pos)?);
        }

        // Upper neighbors (HashMap - only nodes with upper levels)
        let upper_count = read_u64(data, &mut pos)? as usize;
        let mut upper_neighbors: FxHashMap<u32, Vec<Vec<u32>>> =
            FxHashMap::with_capacity_and_hasher(upper_count, rustc_hash::FxBuildHasher);

        for _ in 0..upper_count {
            let node_id = read_u32(data, &mut pos)?;
            let num_levels = read_u8(data, &mut pos)? as usize;

            let mut levels = Vec::with_capacity(num_levels);
            for _ in 0..num_levels {
                let count = read_u16(data, &mut pos)? as usize;
                let mut neighbors = Vec::with_capacity(count);
                for _ in 0..count {
                    neighbors.push(read_u32(data, &mut pos)?);
                }
                levels.push(neighbors);
            }
            upper_neighbors.insert(node_id, levels);
        }

        // Construct storage from raw bytes
        let mut storage = Self::from_bytes(
            raw_data,
            len,
            node_size,
            neighbors_offset,
            vector_offset,
            metadata_offset,
            dimensions,
            max_neighbors,
        );

        // Restore SQ8 state
        storage.mode = mode;
        storage.sq8_params = sq8_params;
        storage.norms = norms;
        storage.sq8_sums = sq8_sums;
        storage.sq8_trained = sq8_trained;
        storage.upper_neighbors = upper_neighbors;

        Ok(storage)
    }

    /// Reorder nodes using BFS traversal for cache-friendly layout
    ///
    /// Returns the old-to-new ID mapping.
    pub fn reorder_bfs(&mut self, entry_point: u32, _max_level: u8) -> Vec<u32> {
        use std::collections::{HashSet, VecDeque};

        let n = self.len;
        if n == 0 {
            return Vec::new();
        }

        // BFS from entry point to determine optimal ordering
        let mut visited = HashSet::new();
        let mut bfs_order = Vec::with_capacity(n);
        let mut queue = VecDeque::new();

        // Start from entry point at highest level, then traverse down
        queue.push_back(entry_point);
        visited.insert(entry_point);

        while let Some(node_id) = queue.pop_front() {
            bfs_order.push(node_id);

            // Visit neighbors at level 0 (most important for cache locality)
            for &neighbor in self.neighbors(node_id) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        // Add any nodes not reachable from entry point
        for node_id in 0..n as u32 {
            if visited.insert(node_id) {
                bfs_order.push(node_id);
            }
        }

        // Create mapping: old_to_new[old_id] = new_id
        let mut old_to_new = vec![0u32; n];
        for (new_id, &old_id) in bfs_order.iter().enumerate() {
            old_to_new[old_id as usize] = new_id as u32;
        }

        // Reorder the storage (creates new backing, copies data in BFS order)
        if let Err(e) = self.apply_reorder(&bfs_order, &old_to_new) {
            tracing::error!("Failed to apply reorder: {}", e);
            return old_to_new;
        }

        old_to_new
    }

    /// Apply a reordering to the storage
    fn apply_reorder(&mut self, bfs_order: &[u32], old_to_new: &[u32]) -> Result<(), String> {
        let n = self.len;
        if n == 0 {
            return Ok(());
        }

        // Allocate new storage
        let new_size = n * self.node_size;
        let layout = Layout::from_size_align(new_size, CACHE_LINE)
            .map_err(|e| format!("Invalid layout for reorder: {e}"))?;
        let new_ptr = unsafe { alloc_zeroed(layout) };
        if new_ptr.is_null() {
            return Err(format!("Allocation failed for reorder: {new_size} bytes"));
        }

        // Copy nodes in BFS order
        for (new_idx, &old_id) in bfs_order.iter().enumerate() {
            let old_ptr = self.node_ptr(old_id);
            let new_node_ptr = unsafe { new_ptr.add(new_idx * self.node_size) };

            // Copy the node data
            unsafe {
                std::ptr::copy_nonoverlapping(old_ptr, new_node_ptr, self.node_size);
            }

            // Update neighbor IDs to use new indices
            let count_ptr = new_node_ptr as *mut u16;
            let count = unsafe { *count_ptr } as usize;
            let neighbors_ptr = unsafe { new_node_ptr.add(self.neighbors_offset) as *mut u32 };

            for i in 0..count.min(self.max_neighbors) {
                let old_neighbor = unsafe { *neighbors_ptr.add(i) };
                if (old_neighbor as usize) < old_to_new.len() {
                    unsafe {
                        *neighbors_ptr.add(i) = old_to_new[old_neighbor as usize];
                    }
                }
            }
        }

        // Reorder norms and sq8_sums
        if !self.norms.is_empty() {
            let old_norms = std::mem::take(&mut self.norms);
            self.norms = vec![0.0; n];
            for (new_idx, &old_id) in bfs_order.iter().enumerate() {
                if (old_id as usize) < old_norms.len() {
                    self.norms[new_idx] = old_norms[old_id as usize];
                }
            }
        }

        if !self.sq8_sums.is_empty() {
            let old_sums = std::mem::take(&mut self.sq8_sums);
            self.sq8_sums = vec![0; n];
            for (new_idx, &old_id) in bfs_order.iter().enumerate() {
                if (old_id as usize) < old_sums.len() {
                    self.sq8_sums[new_idx] = old_sums[old_id as usize];
                }
            }
        }

        // Reorder upper neighbors (HashMap: old_id -> new_id mapping)
        if !self.upper_neighbors.is_empty() {
            let old_upper = std::mem::take(&mut self.upper_neighbors);
            for (new_idx, &old_id) in bfs_order.iter().enumerate() {
                if let Some(levels) = old_upper.get(&old_id) {
                    // Remap neighbor IDs in upper levels
                    let new_levels: Vec<Vec<u32>> = levels
                        .iter()
                        .map(|neighbors| {
                            neighbors
                                .iter()
                                .filter_map(|&old_n| {
                                    if (old_n as usize) < old_to_new.len() {
                                        Some(old_to_new[old_n as usize])
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        })
                        .collect();
                    self.upper_neighbors.insert(new_idx as u32, new_levels);
                }
            }
        }

        // Swap in new backing
        let old_backing = std::mem::replace(
            &mut self.backing,
            StorageBacking::Owned {
                data: NonNull::new(new_ptr).expect("Allocation should not return null"),
                layout,
                capacity: n,
            },
        );

        // Deallocate old backing
        match old_backing {
            StorageBacking::Owned {
                data,
                layout: old_layout,
                ..
            } => unsafe { dealloc(data.as_ptr(), old_layout) },
            #[cfg(feature = "mmap")]
            StorageBacking::Mmap(_) => {} // Mmap dropped automatically
        }

        Ok(())
    }
}

impl Drop for NodeStorage {
    fn drop(&mut self) {
        match &self.backing {
            StorageBacking::Owned {
                data,
                layout,
                capacity,
            } => {
                if *capacity > 0 {
                    unsafe {
                        dealloc(data.as_ptr(), *layout);
                    }
                }
            }
            #[cfg(feature = "mmap")]
            StorageBacking::Mmap(_) => {
                // Mmap is dropped automatically
            }
        }
    }
}

// SAFETY: The raw pointer is only accessed through &self or &mut self,
// ensuring exclusive access for mutations.
unsafe impl Send for NodeStorage {}
unsafe impl Sync for NodeStorage {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_layout_size() {
        // M=16, D=128:
        // count(2) + neighbors(16*2*4=128) + vector(128*4=512) + slot(4) + level(1) = 647
        // Rounded to cache line (64): 704
        let storage = NodeStorage::new(128, 16, 8);
        assert_eq!(storage.node_size(), 704);

        // M=32, D=768:
        // count(2) + neighbors(32*2*4=256) + vector(768*4=3072) + slot(4) + level(1) = 3335
        // Rounded to cache line (64): 3392
        let storage = NodeStorage::new(768, 32, 8);
        assert_eq!(storage.node_size(), 3392);
    }

    #[test]
    fn test_store_and_retrieve_vector() {
        let mut storage = NodeStorage::new(4, 2, 8);
        let vector = vec![1.0f32, 2.0, 3.0, 4.0];
        storage.allocate_node();
        storage.set_vector(0, &vector);

        let retrieved = storage.vector(0);
        assert_eq!(retrieved, &vector[..]);
    }

    #[test]
    fn test_store_and_retrieve_neighbors() {
        let mut storage = NodeStorage::new(4, 2, 8);
        storage.allocate_node();
        storage.allocate_node();
        storage.allocate_node();

        // Set neighbors for node 0
        storage.set_neighbors(0, &[1, 2]);

        let neighbors = storage.neighbors(0);
        assert_eq!(neighbors, &[1, 2]);

        // Empty neighbors
        assert_eq!(storage.neighbors(1), &[] as &[u32]);
    }

    #[test]
    fn test_metadata_slot_mapping() {
        let mut storage = NodeStorage::new(4, 2, 8);
        storage.allocate_node();
        storage.set_slot(0, 42);
        assert_eq!(storage.slot(0), 42);
    }

    #[test]
    fn test_metadata_level() {
        let mut storage = NodeStorage::new(4, 2, 8);
        storage.allocate_node();
        storage.set_level(0, 5);
        assert_eq!(storage.level(0), 5);
    }

    #[test]
    fn test_prefetch_does_not_crash() {
        let mut storage = NodeStorage::new(128, 16, 8);
        for _ in 0..10 {
            storage.allocate_node();
        }
        // Prefetch should not crash even for boundary nodes
        storage.prefetch(0);
        storage.prefetch(9);
        // Out of bounds prefetch should be a no-op
        storage.prefetch(100);
    }

    #[test]
    fn test_multiple_nodes() {
        let mut storage = NodeStorage::new(4, 2, 8);

        // Allocate and populate 100 nodes
        for i in 0..100 {
            let id = storage.allocate_node();
            assert_eq!(id, i as u32);

            let vector: Vec<f32> = (0..4).map(|j| (i * 4 + j) as f32).collect();
            storage.set_vector(id, &vector);
            storage.set_slot(id, i as u32 * 10);
            storage.set_level(id, (i % 8) as u8);

            if i > 0 {
                storage.set_neighbors(id, &[(i - 1) as u32]);
            }
        }

        // Verify all data
        for i in 0..100 {
            let id = i as u32;
            let expected_vector: Vec<f32> = (0..4).map(|j| (i * 4 + j) as f32).collect();

            assert_eq!(storage.vector(id), &expected_vector[..]);
            assert_eq!(storage.slot(id), i as u32 * 10);
            assert_eq!(storage.level(id), (i % 8) as u8);

            if i > 0 {
                assert_eq!(storage.neighbors(id), &[(i - 1) as u32]);
            }
        }
    }

    #[test]
    fn test_memory_usage() {
        let mut storage = NodeStorage::new(4, 2, 8);
        assert_eq!(storage.memory_usage(), 0);

        storage.allocate_node();
        // After first allocation, capacity should be 64
        assert!(storage.memory_usage() > 0);
    }

    #[test]
    fn test_grow_capacity() {
        let mut storage = NodeStorage::new(4, 2, 8);

        // Allocate more than initial capacity
        for i in 0..100 {
            let id = storage.allocate_node();
            assert_eq!(id, i as u32);
        }

        assert_eq!(storage.len(), 100);
        assert!(storage.capacity() >= 100);
    }

    #[test]
    fn test_upper_level_allocation() {
        let mut storage = NodeStorage::new(4, 16, 8); // M=16, max_level=8
        let id = storage.allocate_node();

        // No upper levels initially
        assert_eq!(storage.neighbor_count_at_level(id, 1), 0);
        assert_eq!(storage.neighbor_count_at_level(id, 2), 0);

        // Allocate levels 1-3
        storage.allocate_upper_levels(id, 3);

        // Still empty but allocated
        assert_eq!(storage.neighbor_count_at_level(id, 1), 0);
        assert_eq!(storage.neighbor_count_at_level(id, 2), 0);
        assert_eq!(storage.neighbor_count_at_level(id, 3), 0);
    }

    #[test]
    fn test_upper_level_neighbors() {
        let mut storage = NodeStorage::new(4, 16, 8);
        let id = storage.allocate_node();
        storage.allocate_node(); // node 1
        storage.allocate_node(); // node 2

        // Set level 0 neighbors (uses colocated storage)
        storage.set_neighbors_at_level(id, 0, vec![1, 2]);
        assert_eq!(storage.neighbors_at_level(id, 0), vec![1, 2]);
        assert_eq!(storage.neighbors(id), &[1, 2]); // Same data

        // Set upper level neighbors
        storage.set_neighbors_at_level(id, 1, vec![1]);
        storage.set_neighbors_at_level(id, 2, vec![2]);

        assert_eq!(storage.neighbors_at_level(id, 1), vec![1]);
        assert_eq!(storage.neighbors_at_level(id, 2), vec![2]);

        // Level 0 unchanged
        assert_eq!(storage.neighbors_at_level(id, 0), vec![1, 2]);
    }

    #[test]
    fn test_add_neighbor() {
        let mut storage = NodeStorage::new(4, 4, 8); // M=4 -> level0 max=8, upper max=4
        let id = storage.allocate_node();
        for _ in 0..10 {
            storage.allocate_node();
        }

        // Add level 0 neighbors one by one
        storage.add_neighbor(id, 0, 1);
        storage.add_neighbor(id, 0, 2);
        assert_eq!(storage.neighbors(id), &[1, 2]);

        // Add upper level neighbors
        storage.allocate_upper_levels(id, 2);
        storage.add_neighbor(id, 1, 3);
        storage.add_neighbor(id, 1, 4);
        storage.add_neighbor(id, 2, 5);

        assert_eq!(storage.neighbors_at_level(id, 1), vec![3, 4]);
        assert_eq!(storage.neighbors_at_level(id, 2), vec![5]);

        // Level 0 unchanged
        assert_eq!(storage.neighbors(id), &[1, 2]);
    }

    #[test]
    fn test_upper_level_memory_usage() {
        let mut storage = NodeStorage::new(4, 16, 8);

        // Allocate 100 nodes, only 10% have upper levels (realistic HNSW)
        for i in 0..100u32 {
            storage.allocate_node();
            if i % 10 == 0 {
                // ~10% have upper levels
                storage.allocate_upper_levels(i, 2);
                storage.set_neighbors_at_level(i, 1, vec![0, 1, 2]);
            }
        }

        let mem = storage.memory_usage();
        assert!(mem > 0);
        // Upper level storage is sparse, should be much smaller than level 0
    }

    #[test]
    fn test_max_neighbors_enforcement() {
        let mut storage = NodeStorage::new(4, 2, 8); // M=2 -> level0 max=4, upper max=2
        let id = storage.allocate_node();
        for _ in 0..10 {
            storage.allocate_node();
        }

        // Fill level 0 to capacity (M*2 = 4)
        for i in 1..=4 {
            storage.add_neighbor(id, 0, i);
        }
        assert_eq!(storage.neighbor_count(id), 4);

        // Try to add more - should be ignored (at capacity)
        storage.add_neighbor(id, 0, 5);
        assert_eq!(storage.neighbor_count(id), 4);
        assert!(!storage.neighbors(id).contains(&5));

        // Upper level has max M=2
        storage.allocate_upper_levels(id, 1);
        storage.add_neighbor(id, 1, 1);
        storage.add_neighbor(id, 1, 2);
        assert_eq!(storage.neighbor_count_at_level(id, 1), 2);

        // Try to exceed - should be ignored
        storage.add_neighbor(id, 1, 3);
        assert_eq!(storage.neighbor_count_at_level(id, 1), 2);
    }

    // =========================================================================
    // SQ8 Quantization Tests
    // =========================================================================

    #[test]
    fn test_sq8_node_layout_size() {
        // SQ8 uses u8 per dimension instead of f32 (4x smaller)
        // M=16, D=128:
        // Full precision: 4 + 128 + 512 + 5 = 649 -> 704 (cache aligned)
        // SQ8: 4 + 128 + 128 + 5 = 265 -> 320 (cache aligned)
        let fp_storage = NodeStorage::new(128, 16, 8);
        let sq8_storage = NodeStorage::new_sq8(128, 16, 8);

        assert_eq!(fp_storage.node_size(), 704);
        assert_eq!(sq8_storage.node_size(), 320);
        assert!(sq8_storage.node_size() < fp_storage.node_size());

        // Verify mode
        assert_eq!(fp_storage.mode(), StorageMode::FullPrecision);
        assert_eq!(sq8_storage.mode(), StorageMode::SQ8);
    }

    #[test]
    fn test_sq8_lazy_training() {
        let mut storage = NodeStorage::new_sq8(4, 2, 8);
        assert!(!storage.is_trained());

        // Insert 255 vectors (not enough to train)
        for i in 0..255 {
            storage.allocate_node();
            let vector: Vec<f32> = (0..4).map(|j| (i * 4 + j) as f32).collect();
            storage.set_vector(i as u32, &vector);
        }
        assert!(!storage.is_trained());

        // Insert 256th vector - should trigger training
        storage.allocate_node();
        let vector: Vec<f32> = (0..4).map(|j| (255 * 4 + j) as f32).collect();
        storage.set_vector(255, &vector);
        assert!(storage.is_trained());

        // New vectors should be quantized directly
        storage.allocate_node();
        let vector: Vec<f32> = (0..4).map(|j| (256 * 4 + j) as f32).collect();
        storage.set_vector(256, &vector);
        assert_eq!(storage.len(), 257);
    }

    #[test]
    fn test_sq8_dequantization() {
        let mut storage = NodeStorage::new_sq8(4, 2, 8);

        // Insert enough vectors to trigger training
        for i in 0..256 {
            storage.allocate_node();
            let vector: Vec<f32> = (0..4).map(|j| (i + j) as f32 / 255.0).collect();
            storage.set_vector(i as u32, &vector);
        }
        assert!(storage.is_trained());

        // Dequantized should be approximately equal to original
        let original: Vec<f32> = (0..4).map(|j| (100 + j) as f32 / 255.0).collect();
        let dequantized = storage.get_dequantized(100).unwrap();

        // Check approximate equality (quantization introduces small errors)
        for (o, d) in original.iter().zip(dequantized.iter()) {
            assert!((o - d).abs() < 0.02, "Dequantization error too large");
        }
    }

    #[test]
    fn test_sq8_distance_calculation() {
        let mut storage = NodeStorage::new_sq8(128, 2, 8);

        // Insert vectors with known values (realistic high-dimensional data)
        for i in 0..256 {
            storage.allocate_node();
            // Random-ish distribution with meaningful variance
            let vector: Vec<f32> = (0..128)
                .map(|j| ((i * 128 + j) % 255) as f32 / 255.0)
                .collect();
            storage.set_vector(i as u32, &vector);
        }
        assert!(storage.is_trained());

        // Query vector (middle of range)
        let query: Vec<f32> = (0..128).map(|j| (j % 255) as f32 / 255.0).collect();
        let prep = storage.prepare_query(&query).expect("Should have params");

        // Calculate distance to vectors
        for id in [0, 50, 100, 150, 200, 250] {
            let dist = storage.distance_sq8(&prep, id);
            assert!(
                dist.is_some(),
                "Distance should be computable for vector {id}"
            );
            // Allow small negative values due to floating point precision
            let dist_val = dist.unwrap();
            assert!(
                dist_val >= -0.01,
                "Distance {} for vector {} is too negative",
                dist_val,
                id
            );
        }

        // Distance to self should be near zero
        storage.allocate_node();
        storage.set_vector(256, &query);
        let self_dist = storage.distance_sq8(&prep, 256).unwrap();
        assert!(
            self_dist.abs() < 0.1,
            "Self-distance should be near zero, got {}",
            self_dist
        );
    }

    #[test]
    fn test_sq8_norms_stored() {
        let mut storage = NodeStorage::new_sq8(4, 2, 8);

        // Insert enough vectors to trigger training
        for i in 0..256 {
            storage.allocate_node();
            let vector: Vec<f32> = (0..4).map(|j| (i + j) as f32).collect();
            storage.set_vector(i as u32, &vector);
        }

        // After training, norms should be stored
        for i in 0..256 {
            let norm = storage.get_norm(i as u32);
            assert!(norm.is_some(), "Norm should be stored for vector {i}");
            assert!(norm.unwrap() >= 0.0, "Norm should be non-negative");
        }
    }

    #[test]
    fn test_full_precision_norms_stored() {
        let mut storage = NodeStorage::new(4, 2, 8);

        // Insert some vectors
        for i in 0..10 {
            storage.allocate_node();
            let vector: Vec<f32> = (0..4).map(|j| (i + j) as f32).collect();
            storage.set_vector(i as u32, &vector);
        }

        // Norms should be stored for full precision too
        for i in 0..10 {
            let norm = storage.get_norm(i as u32);
            assert!(norm.is_some(), "Norm should be stored for vector {i}");

            // Verify norm is correct: sum of squares
            let vector: Vec<f32> = (0..4).map(|j| (i + j) as f32).collect();
            let expected_norm: f32 = vector.iter().map(|x| x * x).sum();
            assert!(
                (norm.unwrap() - expected_norm).abs() < 0.01,
                "Norm should equal sum of squares"
            );
        }
    }
}
