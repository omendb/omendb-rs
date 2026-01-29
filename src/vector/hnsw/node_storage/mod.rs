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

mod quantization;
mod reorder;
mod serialization;

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
    pub(crate) len: usize,
    /// Size of each node in bytes (cache-line aligned)
    pub(crate) node_size: usize,
    /// Offset to neighbors array (after u16 count)
    pub(crate) neighbors_offset: usize,
    /// Offset to vector data (after neighbors)
    pub(crate) vector_offset: usize,
    /// Offset to metadata (slot, level)
    pub(crate) metadata_offset: usize,
    /// Vector dimensions
    pub(crate) dimensions: usize,
    /// Max neighbors at level 0 (M * 2)
    pub(crate) max_neighbors: usize,
    /// Max neighbors at upper levels (M)
    max_neighbors_upper: usize,
    /// Max level supported
    max_level: usize,
    /// Upper level neighbors: node_id -> [level-1] -> neighbors
    /// Only populated for nodes with level > 0 (~5% of nodes).
    /// Using HashMap instead of Vec<Option<...>> saves ~7MB at 1M vectors.
    /// Using Vec<Vec<u32>> instead of Box<[Vec<u32>]> allows in-place mutation.
    pub(crate) upper_neighbors: FxHashMap<u32, Vec<Vec<u32>>>,

    // SQ8 quantization support
    /// Storage mode (full precision or SQ8)
    pub(crate) mode: StorageMode,
    /// Scalar quantization parameters (only for SQ8 mode)
    pub(crate) sq8_params: Option<ScalarParams>,
    /// Squared norms for each vector (used in L2 decomposition)
    pub(crate) norms: Vec<f32>,
    /// Sum of quantized codes (only for SQ8 mode, used in L2 decomposition)
    pub(crate) sq8_sums: Vec<i32>,
    /// Training buffer for lazy SQ8 training (first 256 vectors)
    pub(crate) training_buffer: Vec<f32>,
    /// Whether SQ8 quantization has been trained
    pub(crate) sq8_trained: bool,
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
    pub(crate) fn node_ptr(&self, id: u32) -> *const u8 {
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
    pub(crate) fn node_ptr_mut(&mut self, id: u32) -> *mut u8 {
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
        #[cfg(debug_assertions)]
        if count > self.max_neighbors {
            tracing::warn!(
                node_id = id,
                stored_count = count,
                max_neighbors = self.max_neighbors,
                "Neighbor count exceeds max_neighbors, clamping"
            );
        }
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

    /// Get neighbors at any level as Cow (zero-copy for all levels)
    ///
    /// Use this in performance-critical paths to avoid allocation.
    /// Returns borrowed slice for both level 0 (colocated) and upper levels (sparse).
    #[inline]
    #[must_use]
    pub fn neighbors_at_level_cow(&self, id: u32, level: u8) -> std::borrow::Cow<'_, [u32]> {
        use std::borrow::Cow;
        if level == 0 {
            Cow::Borrowed(self.neighbors(id))
        } else {
            match self.upper_neighbors.get(&id) {
                Some(levels) => {
                    let level_idx = level as usize - 1;
                    if level_idx < levels.len() {
                        Cow::Borrowed(&levels[level_idx])
                    } else {
                        Cow::Borrowed(&[])
                    }
                }
                None => Cow::Borrowed(&[]),
            }
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

// Implement NeighborStorage trait for ACORN-1 shared algorithm
impl super::acorn::NeighborStorage for NodeStorage {
    #[inline]
    fn neighbors_at_level_cow(&self, node: u32, level: u8) -> std::borrow::Cow<'_, [u32]> {
        self.neighbors_at_level_cow(node, level)
    }
}

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
