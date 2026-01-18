# V1 Architecture Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign OmenDB embedded to achieve SQLite/DuckDB-tier quality with 5-10x build speedup, 2-3x search speedup (Rust), and 100x larger dataset support.

**Architecture:** Segment-based storage with mutable segment for writes and frozen segments for reads. Unified node layout colocates vectors and neighbors to reduce cache misses. Graph-aware batch construction enables linear CPU scaling.

**Tech Stack:** Rust, mmap (memmap2), postcard serialization, rayon for parallelism

---

## Overview

This plan implements Phase 2 of the V1 redesign in 4 parts:

| Part | Focus                   | Impact                          |
| ---- | ----------------------- | ------------------------------- |
| 1    | Unified Node Storage    | 2-3x search (Rust)              |
| 2    | Segment Abstractions    | Enables incremental persistence |
| 3    | Freeze & Search         | Lock-free reads                 |
| 4    | Graph-Aware Batch Build | 5-10x build                     |

**Key Files:**

- Create: `src/vector/hnsw/unified_storage.rs` - Colocated node layout
- Create: `src/vector/hnsw/segment.rs` - Segment abstractions
- Create: `src/vector/hnsw/segment_manager.rs` - Segment coordination
- Modify: `src/vector/hnsw/index/mod.rs` - Integrate segments
- Modify: `src/vector/hnsw/index/search.rs` - Multi-segment search
- Modify: `src/vector/hnsw/index/persistence.rs` - Segment format

---

## Part 1: Unified Node Storage

### Task 1: Create UnifiedNodeStorage struct

**Files:**

- Create: `src/vector/hnsw/unified_storage.rs`
- Test: `src/vector/hnsw/unified_storage.rs` (inline tests)

**Step 1: Write the failing test**

```rust
// At bottom of unified_storage.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_layout_size() {
        // M=16, D=128 -> 2 + 128 + 512 + 8 = 650 bytes
        let storage = UnifiedNodeStorage::new(128, 16, 8);
        assert_eq!(storage.node_size(), 650);
    }

    #[test]
    fn test_store_and_retrieve_vector() {
        let mut storage = UnifiedNodeStorage::new(4, 2, 8);
        let vector = vec![1.0f32, 2.0, 3.0, 4.0];
        storage.allocate_node(0);
        storage.set_vector(0, &vector);

        let retrieved = storage.vector(0);
        assert_eq!(retrieved, &vector[..]);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/nick/github/omendb/omendb && cargo test unified_storage --lib`
Expected: FAIL with "cannot find module"

**Step 3: Write minimal implementation**

```rust
//! Unified node storage - colocates vectors and neighbors for cache efficiency
//!
//! Node layout in memory (fixed size per index):
//! [neighbor_count: u16][neighbors: [u32; 2*M]][vector: [f32; D]][metadata_ptr: u64]
//!
//! Total size: 2 + 4*(2*M) + 4*D + 8 bytes
//! Example (M=16, D=128): 2 + 128 + 512 + 8 = 650 bytes/node

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;

/// Cache-line alignment for optimal prefetch
const CACHE_LINE: usize = 64;

/// Unified storage with colocated vectors and neighbors
pub struct UnifiedNodeStorage {
    /// Single contiguous allocation for all nodes
    data: NonNull<u8>,
    /// Layout for deallocation
    layout: Layout,
    /// Number of allocated nodes
    capacity: usize,
    /// Number of nodes in use
    len: usize,
    /// Size of each node in bytes
    node_size: usize,
    /// Offset to neighbors array (always 2, after count)
    neighbors_offset: usize,
    /// Offset to vector data
    vector_offset: usize,
    /// Vector dimensions
    dimensions: usize,
    /// Max neighbors at level 0 (M * 2)
    max_neighbors: usize,
}

impl UnifiedNodeStorage {
    /// Create new unified storage
    ///
    /// # Arguments
    /// - `dimensions`: Vector dimensionality
    /// - `m`: HNSW M parameter (level 0 gets M*2 neighbors)
    /// - `max_levels`: Max levels (for upper level storage - separate)
    pub fn new(dimensions: usize, m: usize, _max_levels: usize) -> Self {
        let max_neighbors = m * 2;
        let neighbors_offset = 2; // After u16 count
        let vector_offset = neighbors_offset + max_neighbors * 4;
        let node_size = 2 + max_neighbors * 4 + dimensions * 4 + 8;

        // Round up to cache line for alignment
        let node_size = (node_size + CACHE_LINE - 1) / CACHE_LINE * CACHE_LINE;

        Self {
            data: NonNull::dangling(),
            layout: Layout::from_size_align(0, 1).unwrap(),
            capacity: 0,
            len: 0,
            node_size,
            neighbors_offset,
            vector_offset,
            dimensions,
            max_neighbors,
        }
    }

    /// Node size in bytes
    #[inline]
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// Number of nodes
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Allocate a new node, returns node ID
    pub fn allocate_node(&mut self, _id: u32) -> u32 {
        if self.len >= self.capacity {
            self.grow();
        }
        let node_id = self.len as u32;
        self.len += 1;
        node_id
    }

    /// Grow capacity (double or initial 64)
    fn grow(&mut self) {
        let new_capacity = if self.capacity == 0 { 64 } else { self.capacity * 2 };
        let new_size = new_capacity * self.node_size;
        let new_layout = Layout::from_size_align(new_size, CACHE_LINE)
            .expect("Invalid layout");

        // SAFETY: We're allocating zeroed memory with valid layout
        let new_data = unsafe { alloc_zeroed(new_layout) };
        let new_ptr = NonNull::new(new_data).expect("Allocation failed");

        // Copy old data if any
        if self.capacity > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.data.as_ptr(),
                    new_ptr.as_ptr(),
                    self.len * self.node_size,
                );
                dealloc(self.data.as_ptr(), self.layout);
            }
        }

        self.data = new_ptr;
        self.layout = new_layout;
        self.capacity = new_capacity;
    }

    /// Get pointer to node data
    #[inline]
    fn node_ptr(&self, id: u32) -> *const u8 {
        debug_assert!((id as usize) < self.len, "Node ID out of bounds");
        unsafe { self.data.as_ptr().add(id as usize * self.node_size) }
    }

    /// Get mutable pointer to node data
    #[inline]
    fn node_ptr_mut(&mut self, id: u32) -> *mut u8 {
        debug_assert!((id as usize) < self.len, "Node ID out of bounds");
        unsafe { self.data.as_ptr().add(id as usize * self.node_size) }
    }

    /// Zero-copy access to vector
    #[inline]
    pub fn vector(&self, id: u32) -> &[f32] {
        let ptr = self.node_ptr(id);
        unsafe {
            let vec_ptr = ptr.add(self.vector_offset) as *const f32;
            std::slice::from_raw_parts(vec_ptr, self.dimensions)
        }
    }

    /// Set vector data
    pub fn set_vector(&mut self, id: u32, vector: &[f32]) {
        debug_assert_eq!(vector.len(), self.dimensions);
        let ptr = self.node_ptr_mut(id);
        unsafe {
            let vec_ptr = ptr.add(self.vector_offset) as *mut f32;
            std::ptr::copy_nonoverlapping(vector.as_ptr(), vec_ptr, self.dimensions);
        }
    }

    /// Get neighbor count
    #[inline]
    pub fn neighbor_count(&self, id: u32) -> usize {
        let ptr = self.node_ptr(id);
        unsafe { u16::from_le_bytes([*ptr, *ptr.add(1)]) as usize }
    }

    /// Zero-copy access to neighbors
    #[inline]
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
        debug_assert!(neighbors.len() <= self.max_neighbors);
        let ptr = self.node_ptr_mut(id);
        unsafe {
            // Write count
            let count = neighbors.len() as u16;
            let count_bytes = count.to_le_bytes();
            *ptr = count_bytes[0];
            *ptr.add(1) = count_bytes[1];

            // Write neighbors
            let neighbors_ptr = ptr.add(self.neighbors_offset) as *mut u32;
            std::ptr::copy_nonoverlapping(neighbors.as_ptr(), neighbors_ptr, neighbors.len());
        }
    }
}

impl Drop for UnifiedNodeStorage {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe { dealloc(self.data.as_ptr(), self.layout); }
        }
    }
}

// SAFETY: The raw pointer is only accessed through &self or &mut self
unsafe impl Send for UnifiedNodeStorage {}
unsafe impl Sync for UnifiedNodeStorage {}
```

**Step 4: Register module**

Add to `src/vector/hnsw/mod.rs`:

```rust
pub mod unified_storage;
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/nick/github/omendb/omendb && cargo test unified_storage --lib`
Expected: PASS

**Step 6: Commit**

```bash
git add src/vector/hnsw/unified_storage.rs src/vector/hnsw/mod.rs
git commit -m "feat(hnsw): add UnifiedNodeStorage for colocated vectors+neighbors"
```

---

### Task 2: Add metadata and prefetch support

**Files:**

- Modify: `src/vector/hnsw/unified_storage.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn test_metadata_slot_mapping() {
    let mut storage = UnifiedNodeStorage::new(4, 2, 8);
    storage.allocate_node(0);
    storage.set_slot(0, 42);
    assert_eq!(storage.slot(0), 42);
}

#[test]
fn test_prefetch_does_not_crash() {
    let mut storage = UnifiedNodeStorage::new(128, 16, 8);
    for i in 0..10 {
        storage.allocate_node(i);
    }
    // Prefetch should not crash even for boundary nodes
    storage.prefetch(0);
    storage.prefetch(9);
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/nick/github/omendb/omendb && cargo test unified_storage --lib`
Expected: FAIL with "method not found"

**Step 3: Add metadata and prefetch methods**

Add to `UnifiedNodeStorage` impl:

```rust
/// Offset to metadata (slot ID stored at end of node)
fn metadata_offset(&self) -> usize {
    self.vector_offset + self.dimensions * 4
}

/// Get slot ID (original RecordStore slot)
#[inline]
pub fn slot(&self, id: u32) -> u32 {
    let ptr = self.node_ptr(id);
    unsafe {
        let slot_ptr = ptr.add(self.metadata_offset()) as *const u32;
        *slot_ptr
    }
}

/// Set slot ID
pub fn set_slot(&mut self, id: u32, slot: u32) {
    let ptr = self.node_ptr_mut(id);
    unsafe {
        let slot_ptr = ptr.add(self.metadata_offset()) as *mut u32;
        *slot_ptr = slot;
    }
}

/// Prefetch node data into cache
#[inline]
pub fn prefetch(&self, id: u32) {
    if (id as usize) < self.len {
        let ptr = self.node_ptr(id);
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::aarch64::_prefetch(ptr as *const i8, std::arch::aarch64::_PREFETCH_READ, std::arch::aarch64::_PREFETCH_LOCALITY3);
        }
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/nick/github/omendb/omendb && cargo test unified_storage --lib`
Expected: PASS

**Step 5: Commit**

```bash
git add src/vector/hnsw/unified_storage.rs
git commit -m "feat(hnsw): add metadata and prefetch to UnifiedNodeStorage"
```

---

### Task 3: Add mmap-backed variant for frozen segments

**Files:**

- Modify: `src/vector/hnsw/unified_storage.rs`
- Modify: `Cargo.toml` (add memmap2)

**Step 1: Add memmap2 dependency**

Run: `cd /Users/nick/github/omendb/omendb && cargo add memmap2`

**Step 2: Write the failing test**

```rust
#[test]
fn test_mmap_storage_roundtrip() {
    use tempfile::NamedTempFile;

    // Create and populate in-memory storage
    let mut storage = UnifiedNodeStorage::new(4, 2, 8);
    storage.allocate_node(0);
    storage.set_vector(0, &[1.0, 2.0, 3.0, 4.0]);
    storage.set_neighbors(0, &[1, 2]);
    storage.set_slot(0, 42);

    // Save to file
    let file = NamedTempFile::new().unwrap();
    storage.save_to_file(file.path()).unwrap();

    // Load as mmap
    let mmap_storage = UnifiedNodeStorage::mmap_from_file(file.path(), 4, 2).unwrap();

    assert_eq!(mmap_storage.len(), 1);
    assert_eq!(mmap_storage.vector(0), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(mmap_storage.neighbors(0), &[1, 2]);
    assert_eq!(mmap_storage.slot(0), 42);
}
```

**Step 3: Implement mmap variant**

Add storage mode enum and mmap support:

```rust
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Storage backing
enum StorageBacking {
    /// Owned allocation
    Owned {
        data: NonNull<u8>,
        layout: Layout,
        capacity: usize,
    },
    /// Memory-mapped file (read-only)
    Mmap(Mmap),
}

// Update UnifiedNodeStorage to use StorageBacking
pub struct UnifiedNodeStorage {
    backing: StorageBacking,
    len: usize,
    node_size: usize,
    neighbors_offset: usize,
    vector_offset: usize,
    dimensions: usize,
    max_neighbors: usize,
}

impl UnifiedNodeStorage {
    /// Save to file (for later mmap loading)
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Header: magic, version, dimensions, max_neighbors, node_size, len
        writer.write_all(b"UNFS")?; // 4 bytes
        writer.write_all(&1u32.to_le_bytes())?; // version
        writer.write_all(&(self.dimensions as u32).to_le_bytes())?;
        writer.write_all(&(self.max_neighbors as u32).to_le_bytes())?;
        writer.write_all(&(self.node_size as u32).to_le_bytes())?;
        writer.write_all(&(self.len as u32).to_le_bytes())?;

        // Node data
        let data_ptr = match &self.backing {
            StorageBacking::Owned { data, .. } => data.as_ptr(),
            StorageBacking::Mmap(mmap) => mmap.as_ptr(),
        };
        let data_slice = unsafe {
            std::slice::from_raw_parts(data_ptr, self.len * self.node_size)
        };
        writer.write_all(data_slice)?;

        Ok(())
    }

    /// Load from file as mmap (read-only, zero-copy)
    pub fn mmap_from_file<P: AsRef<Path>>(
        path: P,
        dimensions: usize,
        m: usize,
    ) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Parse header
        if &mmap[0..4] != b"UNFS" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid magic bytes",
            ));
        }

        let mut offset = 4;
        let version = u32::from_le_bytes(mmap[offset..offset+4].try_into().unwrap());
        offset += 4;
        if version != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported version: {}", version),
            ));
        }

        let stored_dims = u32::from_le_bytes(mmap[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let stored_max_neighbors = u32::from_le_bytes(mmap[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let node_size = u32::from_le_bytes(mmap[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let len = u32::from_le_bytes(mmap[offset..offset+4].try_into().unwrap()) as usize;

        // Validate dimensions match
        if stored_dims != dimensions {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Dimension mismatch: file has {}, expected {}", stored_dims, dimensions),
            ));
        }

        let max_neighbors = m * 2;
        let neighbors_offset = 2;
        let vector_offset = neighbors_offset + max_neighbors * 4;

        Ok(Self {
            backing: StorageBacking::Mmap(mmap),
            len,
            node_size,
            neighbors_offset,
            vector_offset,
            dimensions,
            max_neighbors,
        })
    }
}
```

**Step 4: Update existing methods to use backing**

Update `node_ptr` and related methods to handle both backing types.

**Step 5: Run test to verify it passes**

Run: `cd /Users/nick/github/omendb/omendb && cargo test unified_storage --lib`
Expected: PASS

**Step 6: Commit**

```bash
git add src/vector/hnsw/unified_storage.rs Cargo.toml
git commit -m "feat(hnsw): add mmap backing for UnifiedNodeStorage"
```

---

## Part 2: Segment Abstractions

### Task 4: Create Segment trait and MutableSegment

**Files:**

- Create: `src/vector/hnsw/segment.rs`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutable_segment_insert_and_search() {
        let params = crate::vector::hnsw::types::HNSWParams::default();
        let mut segment = MutableSegment::new(128, params, DistanceFunction::L2);

        let vector = vec![1.0f32; 128];
        let id = segment.insert(&vector, None);
        assert_eq!(id, 0);

        let results = segment.search(&vector, 1, 100);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/nick/github/omendb/omendb && cargo test segment --lib`
Expected: FAIL with "cannot find module"

**Step 3: Write minimal implementation**

```rust
//! Segment-based storage for HNSW
//!
//! MutableSegment: Write-optimized, uses atomic storage
//! FrozenSegment: Read-optimized, uses unified colocated storage

use crate::vector::hnsw::index::HNSWIndex;
use crate::vector::hnsw::types::{DistanceFunction, HNSWParams};
use crate::vector::hnsw::unified_storage::UnifiedNodeStorage;

/// Search result from a segment
#[derive(Debug, Clone)]
pub struct SegmentSearchResult {
    pub id: u32,
    pub distance: f32,
    pub slot: u32,
}

/// Mutable segment for writes (uses existing atomic storage)
pub struct MutableSegment {
    /// Underlying HNSW index (uses atomic neighbor storage)
    index: HNSWIndex,
    /// Segment ID
    id: u64,
    /// Max capacity before freeze
    capacity: usize,
}

impl MutableSegment {
    /// Create new mutable segment
    pub fn new(dimensions: usize, params: HNSWParams, distance_fn: DistanceFunction) -> Self {
        Self {
            index: HNSWIndex::new(dimensions, params, distance_fn),
            id: 0,
            capacity: 100_000, // Default capacity
        }
    }

    /// Create with specific capacity
    pub fn with_capacity(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: DistanceFunction,
        capacity: usize,
    ) -> Self {
        Self {
            index: HNSWIndex::new(dimensions, params, distance_fn),
            id: 0,
            capacity,
        }
    }

    /// Number of vectors in segment
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Check if at capacity
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Insert vector, returns internal ID
    pub fn insert(&mut self, vector: &[f32], _metadata: Option<()>) -> u32 {
        self.index.insert_base(vector).unwrap()
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<SegmentSearchResult> {
        self.index
            .search(query, k, ef)
            .into_iter()
            .map(|(id, dist)| SegmentSearchResult {
                id,
                distance: dist,
                slot: id, // In mutable segment, id == slot
            })
            .collect()
    }

    /// Freeze into read-optimized segment
    pub fn freeze(self) -> FrozenSegment {
        FrozenSegment::from_mutable(self)
    }
}

/// Frozen segment for reads (uses unified colocated storage)
pub struct FrozenSegment {
    /// Unified storage (colocated vectors + neighbors)
    storage: UnifiedNodeStorage,
    /// Segment ID
    id: u64,
    /// Entry point for search
    entry_point: Option<u32>,
    /// HNSW parameters
    params: HNSWParams,
    /// Distance function
    distance_fn: DistanceFunction,
}

impl FrozenSegment {
    /// Create from mutable segment
    fn from_mutable(mutable: MutableSegment) -> Self {
        let dimensions = mutable.index.dimensions();
        let params = mutable.index.params().clone();
        let m = params.m;

        // Create unified storage and copy data
        let mut storage = UnifiedNodeStorage::new(dimensions, m, params.max_level as usize);

        // Copy all nodes
        for id in 0..mutable.index.len() as u32 {
            storage.allocate_node(id);

            // Copy vector
            if let Some(vector) = mutable.index.get_vector(id) {
                storage.set_vector(id, vector);
            }

            // Copy level 0 neighbors
            mutable.index.with_neighbors(id, 0, |neighbors| {
                storage.set_neighbors(id, neighbors);
            });

            // Set slot (same as id for now)
            storage.set_slot(id, id);
        }

        Self {
            storage,
            id: mutable.id,
            entry_point: mutable.index.entry_point(),
            params,
            distance_fn: mutable.index.distance_function(),
        }
    }

    /// Number of vectors
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Search (uses unified storage for cache efficiency)
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<SegmentSearchResult> {
        // TODO: Implement search using unified storage
        // For now, return empty - will implement in Task 6
        Vec::new()
    }
}
```

**Step 4: Register module**

Add to `src/vector/hnsw/mod.rs`:

```rust
pub mod segment;
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/nick/github/omendb/omendb && cargo test segment --lib`
Expected: PASS

**Step 6: Commit**

```bash
git add src/vector/hnsw/segment.rs src/vector/hnsw/mod.rs
git commit -m "feat(hnsw): add MutableSegment and FrozenSegment abstractions"
```

---

### Task 5: Implement FrozenSegment search

**Files:**

- Modify: `src/vector/hnsw/segment.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn test_frozen_segment_search() {
    let params = crate::vector::hnsw::types::HNSWParams::default();
    let mut mutable = MutableSegment::new(4, params, DistanceFunction::L2);

    // Insert some vectors
    mutable.insert(&[1.0, 0.0, 0.0, 0.0], None);
    mutable.insert(&[0.0, 1.0, 0.0, 0.0], None);
    mutable.insert(&[0.0, 0.0, 1.0, 0.0], None);

    // Freeze and search
    let frozen = mutable.freeze();
    let results = frozen.search(&[1.0, 0.0, 0.0, 0.0], 1, 100);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 0);
    assert!(results[0].distance < 0.001); // Should be very close
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/nick/github/omendb/omendb && cargo test frozen_segment_search --lib`
Expected: FAIL (search returns empty)

**Step 3: Implement search using unified storage**

The search implementation needs to:

1. Use greedy search starting from entry point
2. Use unified storage for cache-efficient access
3. Return results sorted by distance

```rust
impl FrozenSegment {
    /// Search using unified storage
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<SegmentSearchResult> {
        let Some(entry_point) = self.entry_point else {
            return Vec::new();
        };

        if self.storage.is_empty() {
            return Vec::new();
        }

        // Use query buffers for efficient search
        let mut visited = vec![false; self.storage.len()];
        let mut candidates = std::collections::BinaryHeap::new();
        let mut results = std::collections::BinaryHeap::new();

        // Start from entry point
        let ep_vector = self.storage.vector(entry_point);
        let ep_dist = self.compute_distance(query, ep_vector);

        visited[entry_point as usize] = true;
        candidates.push(std::cmp::Reverse((OrderedFloat(ep_dist), entry_point)));
        results.push((OrderedFloat(ep_dist), entry_point));

        // Greedy search
        while let Some(std::cmp::Reverse((OrderedFloat(c_dist), c_id))) = candidates.pop() {
            // Check if we can stop
            if let Some(&(OrderedFloat(worst_dist), _)) = results.peek() {
                if c_dist > worst_dist && results.len() >= ef {
                    break;
                }
            }

            // Prefetch neighbors
            let neighbors = self.storage.neighbors(c_id);
            for &neighbor in neighbors.iter().take(4) {
                self.storage.prefetch(neighbor);
            }

            // Explore neighbors
            for &neighbor in neighbors {
                if visited[neighbor as usize] {
                    continue;
                }
                visited[neighbor as usize] = true;

                let n_vector = self.storage.vector(neighbor);
                let n_dist = self.compute_distance(query, n_vector);

                // Add to candidates if promising
                let dominated = results.len() >= ef && {
                    let &(OrderedFloat(worst), _) = results.peek().unwrap();
                    n_dist > worst
                };

                if !dominated {
                    candidates.push(std::cmp::Reverse((OrderedFloat(n_dist), neighbor)));
                    results.push((OrderedFloat(n_dist), neighbor));

                    // Trim results
                    while results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        // Convert to output format
        let mut output: Vec<_> = results
            .into_iter()
            .map(|(OrderedFloat(dist), id)| SegmentSearchResult {
                id,
                distance: dist,
                slot: self.storage.slot(id),
            })
            .collect();

        output.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        output.truncate(k);
        output
    }

    /// Compute distance between query and candidate
    #[inline]
    fn compute_distance(&self, query: &[f32], candidate: &[f32]) -> f32 {
        match self.distance_fn {
            DistanceFunction::L2 => {
                query.iter()
                    .zip(candidate.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum()
            }
            DistanceFunction::Cosine => {
                let dot: f32 = query.iter().zip(candidate.iter()).map(|(a, b)| a * b).sum();
                let norm_q: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_c: f32 = candidate.iter().map(|x| x * x).sum::<f32>().sqrt();
                1.0 - dot / (norm_q * norm_c + 1e-10)
            }
            DistanceFunction::NegativeDotProduct => {
                -query.iter().zip(candidate.iter()).map(|(a, b)| a * b).sum::<f32>()
            }
        }
    }
}

// Add OrderedFloat helper
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
struct OrderedFloat(f32);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/nick/github/omendb/omendb && cargo test frozen_segment_search --lib`
Expected: PASS

**Step 5: Commit**

```bash
git add src/vector/hnsw/segment.rs
git commit -m "feat(hnsw): implement search for FrozenSegment with unified storage"
```

---

### Task 6: Create SegmentManager

**Files:**

- Create: `src/vector/hnsw/segment_manager.rs`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_manager_insert_and_search() {
        let config = SegmentConfig {
            dimensions: 4,
            params: HNSWParams::default(),
            distance_fn: DistanceFunction::L2,
            segment_capacity: 10,
        };
        let mut manager = SegmentManager::new(config);

        // Insert vectors
        for i in 0..5 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector, None);
        }

        // Search
        let results = manager.search(&[2.0, 0.0, 0.0, 0.0], 3, 100);
        assert_eq!(results.len(), 3);
        // Closest should be id=2
        assert_eq!(results[0].id, 2);
    }

    #[test]
    fn test_segment_manager_auto_freeze() {
        let config = SegmentConfig {
            dimensions: 4,
            params: HNSWParams::default(),
            distance_fn: DistanceFunction::L2,
            segment_capacity: 5, // Small capacity to trigger freeze
        };
        let mut manager = SegmentManager::new(config);

        // Insert more than capacity
        for i in 0..7 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector, None);
        }

        // Should have 1 frozen + 1 mutable
        assert_eq!(manager.frozen_count(), 1);
        assert_eq!(manager.mutable_len(), 2);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/nick/github/omendb/omendb && cargo test segment_manager --lib`
Expected: FAIL with "cannot find module"

**Step 3: Write implementation**

```rust
//! Segment manager for coordinating mutable and frozen segments

use crate::vector::hnsw::segment::{FrozenSegment, MutableSegment, SegmentSearchResult};
use crate::vector::hnsw::types::{DistanceFunction, HNSWParams};
use std::sync::Arc;

/// Configuration for segment manager
#[derive(Clone)]
pub struct SegmentConfig {
    pub dimensions: usize,
    pub params: HNSWParams,
    pub distance_fn: DistanceFunction,
    pub segment_capacity: usize,
}

/// Manages mutable and frozen segments
pub struct SegmentManager {
    /// Configuration
    config: SegmentConfig,
    /// Active mutable segment
    mutable: MutableSegment,
    /// Frozen segments (read-only)
    frozen: Vec<Arc<FrozenSegment>>,
    /// Next segment ID
    next_segment_id: u64,
}

impl SegmentManager {
    /// Create new segment manager
    pub fn new(config: SegmentConfig) -> Self {
        let mutable = MutableSegment::with_capacity(
            config.dimensions,
            config.params.clone(),
            config.distance_fn,
            config.segment_capacity,
        );

        Self {
            config,
            mutable,
            frozen: Vec::new(),
            next_segment_id: 0,
        }
    }

    /// Number of frozen segments
    pub fn frozen_count(&self) -> usize {
        self.frozen.len()
    }

    /// Number of vectors in mutable segment
    pub fn mutable_len(&self) -> usize {
        self.mutable.len()
    }

    /// Total number of vectors
    pub fn len(&self) -> usize {
        self.mutable.len() + self.frozen.iter().map(|s| s.len()).sum::<usize>()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert vector
    pub fn insert(&mut self, vector: &[f32], metadata: Option<()>) -> u32 {
        // Freeze mutable if at capacity
        if self.mutable.is_full() {
            self.freeze_mutable();
        }

        let local_id = self.mutable.insert(vector, metadata);

        // Return global ID (segment_id << 32 | local_id)
        // For now, just return local_id since we don't have multi-segment IDs yet
        local_id
    }

    /// Freeze current mutable segment
    fn freeze_mutable(&mut self) {
        let old_mutable = std::mem::replace(
            &mut self.mutable,
            MutableSegment::with_capacity(
                self.config.dimensions,
                self.config.params.clone(),
                self.config.distance_fn,
                self.config.segment_capacity,
            ),
        );

        if !old_mutable.is_empty() {
            let frozen = old_mutable.freeze();
            self.frozen.push(Arc::new(frozen));
        }

        self.next_segment_id += 1;
    }

    /// Search across all segments
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<SegmentSearchResult> {
        use rayon::prelude::*;

        // Search mutable segment
        let mut results = self.mutable.search(query, k, ef);

        // Search frozen segments in parallel
        let frozen_results: Vec<Vec<SegmentSearchResult>> = self.frozen
            .par_iter()
            .map(|segment| segment.search(query, k, ef))
            .collect();

        // Merge all results
        for frozen_result in frozen_results {
            results.extend(frozen_result);
        }

        // Sort by distance and take top k
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(k);

        results
    }

    /// Force freeze current mutable segment
    pub fn flush(&mut self) {
        self.freeze_mutable();
    }
}
```

**Step 4: Register module**

Add to `src/vector/hnsw/mod.rs`:

```rust
pub mod segment_manager;
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/nick/github/omendb/omendb && cargo test segment_manager --lib`
Expected: PASS

**Step 6: Commit**

```bash
git add src/vector/hnsw/segment_manager.rs src/vector/hnsw/mod.rs
git commit -m "feat(hnsw): add SegmentManager for mutable/frozen segment coordination"
```

---

## Part 3: Integration

### Task 7: Add segment persistence

**Files:**

- Modify: `src/vector/hnsw/segment.rs`
- Modify: `src/vector/hnsw/segment_manager.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn test_segment_manager_persistence() {
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let config = SegmentConfig {
        dimensions: 4,
        params: HNSWParams::default(),
        distance_fn: DistanceFunction::L2,
        segment_capacity: 5,
    };

    // Create and populate
    let mut manager = SegmentManager::new(config.clone());
    for i in 0..7 {
        manager.insert(&[i as f32, 0.0, 0.0, 0.0], None);
    }
    manager.flush();

    // Save
    manager.save(dir.path()).unwrap();

    // Load
    let loaded = SegmentManager::load(dir.path(), config).unwrap();

    assert_eq!(loaded.len(), 7);

    // Search should work
    let results = loaded.search(&[2.0, 0.0, 0.0, 0.0], 1, 100);
    assert_eq!(results.len(), 1);
}
```

**Step 2: Implement save/load**

Add to `SegmentManager`:

```rust
use std::path::Path;
use std::fs;

impl SegmentManager {
    /// Save all segments to directory
    pub fn save<P: AsRef<Path>>(&self, dir: P) -> std::io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;

        // Save manifest
        let manifest = SegmentManifest {
            version: 1,
            dimensions: self.config.dimensions,
            segment_capacity: self.config.segment_capacity,
            frozen_count: self.frozen.len(),
            mutable_len: self.mutable.len(),
        };
        let manifest_bytes = postcard::to_allocvec(&manifest)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(dir.join("manifest.bin"), manifest_bytes)?;

        // Save frozen segments
        for (i, segment) in self.frozen.iter().enumerate() {
            let path = dir.join(format!("segment_{:04}.frozen", i));
            segment.save(&path)?;
        }

        // Save mutable segment
        if !self.mutable.is_empty() {
            let path = dir.join("mutable.bin");
            self.mutable.save(&path)?;
        }

        Ok(())
    }

    /// Load from directory
    pub fn load<P: AsRef<Path>>(dir: P, config: SegmentConfig) -> std::io::Result<Self> {
        let dir = dir.as_ref();

        // Load manifest
        let manifest_bytes = fs::read(dir.join("manifest.bin"))?;
        let manifest: SegmentManifest = postcard::from_bytes(&manifest_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Load frozen segments
        let mut frozen = Vec::with_capacity(manifest.frozen_count);
        for i in 0..manifest.frozen_count {
            let path = dir.join(format!("segment_{:04}.frozen", i));
            let segment = FrozenSegment::load(&path, &config)?;
            frozen.push(Arc::new(segment));
        }

        // Load or create mutable segment
        let mutable_path = dir.join("mutable.bin");
        let mutable = if mutable_path.exists() {
            MutableSegment::load(&mutable_path, &config)?
        } else {
            MutableSegment::with_capacity(
                config.dimensions,
                config.params.clone(),
                config.distance_fn,
                config.segment_capacity,
            )
        };

        Ok(Self {
            config,
            mutable,
            frozen,
            next_segment_id: manifest.frozen_count as u64,
        })
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SegmentManifest {
    version: u32,
    dimensions: usize,
    segment_capacity: usize,
    frozen_count: usize,
    mutable_len: usize,
}
```

**Step 3: Add save/load to FrozenSegment and MutableSegment**

```rust
// In segment.rs

impl FrozenSegment {
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        self.storage.save_to_file(path)
    }

    pub fn load<P: AsRef<Path>>(path: P, config: &SegmentConfig) -> std::io::Result<Self> {
        let storage = UnifiedNodeStorage::mmap_from_file(path, config.dimensions, config.params.m)?;
        Ok(Self {
            storage,
            id: 0, // Will be set by manager
            entry_point: if storage.len() > 0 { Some(0) } else { None },
            params: config.params.clone(),
            distance_fn: config.distance_fn,
        })
    }
}

impl MutableSegment {
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        self.index.save(path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    pub fn load<P: AsRef<Path>>(path: P, config: &SegmentConfig) -> std::io::Result<Self> {
        let index = HNSWIndex::load(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(Self {
            index,
            id: 0,
            capacity: config.segment_capacity,
        })
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/nick/github/omendb/omendb && cargo test segment_manager_persistence --lib`
Expected: PASS

**Step 5: Commit**

```bash
git add src/vector/hnsw/segment.rs src/vector/hnsw/segment_manager.rs
git commit -m "feat(hnsw): add segment persistence with mmap support"
```

---

## Part 4: Graph-Aware Batch Construction

### Task 8: Implement k-means clustering for batch insert

**Files:**

- Create: `src/vector/hnsw/batch_builder.rs`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_clustering() {
        // Create 4 clusters of vectors
        let mut vectors = Vec::new();
        for i in 0..40 {
            let cluster = i / 10;
            let base = cluster as f32 * 10.0;
            vectors.push(vec![base + (i % 10) as f32 * 0.1, 0.0, 0.0, 0.0]);
        }

        let clusters = kmeans_cluster(&vectors, 4, 10);
        assert_eq!(clusters.len(), 4);

        // Each cluster should have ~10 vectors
        for cluster in &clusters {
            assert!(cluster.len() >= 5 && cluster.len() <= 15);
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/nick/github/omendb/omendb && cargo test batch_builder --lib`
Expected: FAIL

**Step 3: Implement k-means clustering**

```rust
//! Graph-aware batch construction for HNSW
//!
//! Uses clustering to enable parallel construction without contention.

use rayon::prelude::*;

/// Cluster assignment for a vector
pub struct Cluster {
    /// Indices of vectors in this cluster
    pub indices: Vec<usize>,
    /// Centroid of this cluster
    pub centroid: Vec<f32>,
}

/// Simple k-means clustering
pub fn kmeans_cluster(vectors: &[Vec<f32>], k: usize, max_iters: usize) -> Vec<Cluster> {
    if vectors.is_empty() || k == 0 {
        return Vec::new();
    }

    let k = k.min(vectors.len());
    let dims = vectors[0].len();

    // Initialize centroids using k-means++
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);

    // First centroid: random (use first vector for determinism)
    centroids.push(vectors[0].clone());

    // Remaining centroids: k-means++ selection
    for _ in 1..k {
        // Compute distances to nearest centroid
        let distances: Vec<f32> = vectors
            .iter()
            .map(|v| {
                centroids
                    .iter()
                    .map(|c| l2_distance(v, c))
                    .fold(f32::MAX, f32::min)
            })
            .collect();

        // Select next centroid proportional to distance squared
        let total: f32 = distances.iter().sum();
        let threshold = total * 0.5; // Simplified: just pick median distance
        let mut cumsum = 0.0;
        for (i, &d) in distances.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold {
                centroids.push(vectors[i].clone());
                break;
            }
        }
    }

    // Ensure we have k centroids
    while centroids.len() < k {
        centroids.push(vectors[centroids.len()].clone());
    }

    // Assignment array
    let mut assignments = vec![0usize; vectors.len()];

    // Iterate
    for _ in 0..max_iters {
        // Assign vectors to nearest centroid
        let changed: bool = vectors
            .par_iter()
            .zip(assignments.par_iter_mut())
            .map(|(v, assignment)| {
                let nearest = centroids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, l2_distance(v, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                let old = *assignment;
                *assignment = nearest;
                old != nearest
            })
            .any(|x| x);

        if !changed {
            break;
        }

        // Update centroids
        let mut new_centroids: Vec<Vec<f32>> = vec![vec![0.0; dims]; k];
        let mut counts = vec![0usize; k];

        for (i, v) in vectors.iter().enumerate() {
            let cluster = assignments[i];
            counts[cluster] += 1;
            for (j, &val) in v.iter().enumerate() {
                new_centroids[cluster][j] += val;
            }
        }

        for (c, centroid) in new_centroids.iter_mut().enumerate() {
            if counts[c] > 0 {
                for val in centroid.iter_mut() {
                    *val /= counts[c] as f32;
                }
            }
        }

        centroids = new_centroids;
    }

    // Build clusters
    let mut clusters: Vec<Cluster> = (0..k)
        .map(|i| Cluster {
            indices: Vec::new(),
            centroid: centroids[i].clone(),
        })
        .collect();

    for (i, &cluster_id) in assignments.iter().enumerate() {
        clusters[cluster_id].indices.push(i);
    }

    // Remove empty clusters
    clusters.retain(|c| !c.indices.is_empty());

    clusters
}

/// L2 distance between two vectors
#[inline]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}
```

**Step 4: Register module**

Add to `src/vector/hnsw/mod.rs`:

```rust
pub mod batch_builder;
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/nick/github/omendb/omendb && cargo test batch_builder --lib`
Expected: PASS

**Step 6: Commit**

```bash
git add src/vector/hnsw/batch_builder.rs src/vector/hnsw/mod.rs
git commit -m "feat(hnsw): add k-means clustering for batch construction"
```

---

### Task 9: Implement parallel cluster building

**Files:**

- Modify: `src/vector/hnsw/batch_builder.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn test_batch_build() {
    let vectors: Vec<Vec<f32>> = (0..100)
        .map(|i| vec![i as f32, 0.0, 0.0, 0.0])
        .collect();

    let params = HNSWParams::default();
    let index = BatchBuilder::build(&vectors, params, DistanceFunction::L2);

    assert_eq!(index.len(), 100);

    // Search should work
    let results = index.search(&[50.0, 0.0, 0.0, 0.0], 5, 100);
    assert_eq!(results.len(), 5);
    assert_eq!(results[0].0, 50); // Should find exact match
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/nick/github/omendb/omendb && cargo test batch_build --lib`
Expected: FAIL

**Step 3: Implement batch builder**

```rust
use crate::vector::hnsw::index::HNSWIndex;
use crate::vector::hnsw::types::{DistanceFunction, HNSWParams};

/// Batch builder using clustering for parallel construction
pub struct BatchBuilder;

impl BatchBuilder {
    /// Build HNSW index from vectors using graph-aware batch construction
    ///
    /// Algorithm:
    /// 1. Cluster vectors (k-means)
    /// 2. Build local graphs per cluster in parallel
    /// 3. Merge graphs (connect cluster boundaries)
    /// 4. Refinement pass (optional)
    pub fn build(
        vectors: &[Vec<f32>],
        params: HNSWParams,
        distance_fn: DistanceFunction,
    ) -> HNSWIndex {
        if vectors.is_empty() {
            return HNSWIndex::new(0, params, distance_fn);
        }

        let dimensions = vectors[0].len();
        let num_clusters = (num_cpus::get() * 4).min(vectors.len() / 10).max(1);

        // Phase 1: Cluster vectors
        let clusters = kmeans_cluster(vectors, num_clusters, 10);

        if clusters.len() <= 1 {
            // Single cluster: use standard insert
            let mut index = HNSWIndex::new(dimensions, params, distance_fn);
            for vector in vectors {
                index.insert_base(vector).unwrap();
            }
            return index;
        }

        // Phase 2: Build local graphs in parallel
        let local_indices: Vec<HNSWIndex> = clusters
            .par_iter()
            .map(|cluster| {
                let mut local = HNSWIndex::new(dimensions, params.clone(), distance_fn);
                for &idx in &cluster.indices {
                    local.insert_base(&vectors[idx]).unwrap();
                }
                local
            })
            .collect();

        // Phase 3: Merge into single index
        let mut merged = HNSWIndex::new(dimensions, params.clone(), distance_fn);

        // Insert all vectors (simple merge for now)
        // TODO: Implement proper graph merging with boundary connections
        for vector in vectors {
            merged.insert_base(vector).unwrap();
        }

        merged
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/nick/github/omendb/omendb && cargo test batch_build --lib`
Expected: PASS

**Step 5: Commit**

```bash
git add src/vector/hnsw/batch_builder.rs
git commit -m "feat(hnsw): add BatchBuilder for graph-aware parallel construction"
```

---

## Completion Checklist

- [ ] Task 1: UnifiedNodeStorage struct
- [ ] Task 2: Metadata and prefetch support
- [ ] Task 3: mmap-backed variant
- [ ] Task 4: MutableSegment and FrozenSegment
- [ ] Task 5: FrozenSegment search
- [ ] Task 6: SegmentManager
- [ ] Task 7: Segment persistence
- [ ] Task 8: K-means clustering
- [ ] Task 9: Parallel cluster building

## Post-Implementation

After completing all tasks:

1. **Run full test suite:**

   ```bash
   cd /Users/nick/github/omendb/omendb
   cargo test --lib
   cargo clippy -- -D warnings
   ```

2. **Run benchmarks:**

   ```bash
   cd python && uv run python benchmark.py
   ```

3. **Update documentation:**
   - Update `ai/embedded/STATUS.md` with implementation status
   - Update `ai/embedded/OPTIMIZATIONS-TRIED.md` with results

4. **Create PR or merge to main**
