//! Node storage - colocates vectors and neighbors for cache efficiency
//!
//! Node layout in memory (fixed size per index):
//! ```text
//! [neighbor_count: u16][pad: u16][neighbors: [u32; M*2]][vector: [f32; D]][slot: u32][level: u8][padding]
//! ```
//!
//! Total size: 4 + 4*(M*2) + 4*D + 4 + 1 + padding bytes (rounded to cache line)
//! Example (M=16, D=128): 4 + 128 + 512 + 4 + 1 + padding = 704 bytes/node (rounded to 704)
//!
//! Benefits:
//! - Single prefetch covers both neighbors and vector
//! - Zero-copy neighbor access (no buffer copy)
//! - Cache-line aligned node access
//! - 2-3x faster search at high dimensions (768D+)
//!
//! All fields after the count are 4-byte aligned, ensured by the 2-byte padding after count.

// Allow pointer casts - we ensure alignment via layout design (all offsets are 4-byte aligned)
#![allow(clippy::cast_ptr_alignment)]

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;

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

/// Unified storage with colocated vectors and neighbors
///
/// This storage format places vectors and neighbors together in memory
/// so that a single cache prefetch covers both. This significantly improves
/// search performance by reducing cache misses during graph traversal.
pub struct NodeStorage {
    /// Storage backing (owned or mmap)
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
}

impl NodeStorage {
    /// Create new unified storage
    ///
    /// # Arguments
    /// - `dimensions`: Vector dimensionality
    /// - `m`: HNSW M parameter (level 0 gets M*2 neighbors)
    /// - `_max_levels`: Max levels (reserved for future upper level storage)
    #[must_use]
    pub fn new(dimensions: usize, m: usize, _max_levels: usize) -> Self {
        let max_neighbors = m * 2;

        // Layout: [count:2][pad:2][neighbors:M*2*4][vector:D*4][slot:4][level:1]
        // We add 2 bytes of padding after count to ensure neighbors are 4-byte aligned
        let neighbors_offset = 4; // 2 (count) + 2 (padding) = 4
        let vector_offset = neighbors_offset + max_neighbors * 4;
        let metadata_offset = vector_offset + dimensions * 4;
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

    /// Max neighbors per node
    #[inline]
    #[must_use]
    pub fn max_neighbors(&self) -> usize {
        self.max_neighbors
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
        node_id
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

    /// Zero-copy access to vector
    #[inline]
    #[must_use]
    pub fn vector(&self, id: u32) -> &[f32] {
        let ptr = self.node_ptr(id);
        unsafe {
            let vec_ptr = ptr.add(self.vector_offset) as *const f32;
            std::slice::from_raw_parts(vec_ptr, self.dimensions)
        }
    }

    /// Set vector data
    pub fn set_vector(&mut self, id: u32, vector: &[f32]) {
        debug_assert_eq!(
            vector.len(),
            self.dimensions,
            "Vector length {} doesn't match dimensions {}",
            vector.len(),
            self.dimensions
        );
        let ptr = self.node_ptr_mut(id);
        unsafe {
            let vec_ptr = ptr.add(self.vector_offset) as *mut f32;
            std::ptr::copy_nonoverlapping(vector.as_ptr(), vec_ptr, self.dimensions);
        }
    }

    /// Get neighbor count
    #[inline]
    #[must_use]
    pub fn neighbor_count(&self, id: u32) -> usize {
        let ptr = self.node_ptr(id);
        unsafe { u16::from_le_bytes([*ptr, *ptr.add(1)]) as usize }
    }

    /// Zero-copy access to neighbors
    #[inline]
    #[must_use]
    pub fn neighbors(&self, id: u32) -> &[u32] {
        let count = self.neighbor_count(id);
        if count == 0 {
            return &[];
        }
        let ptr = self.node_ptr(id);
        unsafe {
            let neighbors_ptr = ptr.add(self.neighbors_offset) as *const u32;
            std::slice::from_raw_parts(neighbors_ptr, count)
        }
    }

    /// Set neighbors (overwrites all)
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
        match &self.backing {
            StorageBacking::Owned { capacity, .. } => capacity * self.node_size,
            #[cfg(feature = "mmap")]
            StorageBacking::Mmap(mmap) => mmap.len(),
        }
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

        let capacity = if node_size > 0 && !data.is_empty() {
            data.len() / node_size
        } else {
            0
        };

        // Convert Vec<u8> to owned allocation
        let backing = if data.is_empty() {
            StorageBacking::default()
        } else {
            let layout = Layout::from_size_align(data.len(), CACHE_LINE).expect("Invalid layout");
            // SAFETY: We're taking ownership of the data and converting to NonNull
            let ptr = {
                let boxed = data.into_boxed_slice();
                let raw = Box::into_raw(boxed) as *mut u8;
                NonNull::new(raw).expect("Box should not be null")
            };
            StorageBacking::Owned {
                data: ptr,
                layout,
                capacity,
            }
        };

        Self {
            backing,
            len,
            node_size,
            neighbors_offset,
            vector_offset,
            metadata_offset,
            dimensions,
            max_neighbors,
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
        Self {
            backing: StorageBacking::Mmap(mmap),
            len,
            node_size,
            neighbors_offset,
            vector_offset,
            metadata_offset,
            dimensions,
            max_neighbors,
        }
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
}
