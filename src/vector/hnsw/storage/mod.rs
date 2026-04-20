//! Unified flat storage for HNSW.
//!
//! Provides two contiguous matrices:
//! 1. VectorMatrix: [N, Dim] f32
//! 2. NeighborMatrix: [N, stride] u32 (where stride = (max_levels + 1) * CHUNK_SIZE)
//!
//! This layout eliminates all pointer chasing and branching during search.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};

const MAX_LEVEL_CAPACITY: usize = 256; // Count slot + up to 255 neighbors

/// High-performance flat vector matrix.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct VectorMatrix {
    pub data: Vec<f32>,
    pub dim: usize,
}

impl VectorMatrix {
    pub fn new(dim: usize) -> Self {
        Self {
            data: Vec::new(),
            dim,
        }
    }

    pub fn with_capacity(num_nodes: usize, dim: usize) -> Self {
        Self {
            data: Vec::with_capacity(num_nodes * dim),
            dim,
        }
    }

    #[inline(always)]
    pub fn get(&self, id: u32) -> &[f32] {
        let base = id as usize * self.dim;
        &self.data[base..base + self.dim]
    }

    #[inline(always)]
    pub fn prefetch(&self, id: u32) {
        let base = id as usize * self.dim;
        if base < self.data.len() {
            let ptr = &self.data[base] as *const f32 as *const u8;
            #[cfg(target_arch = "x86_64")]
            unsafe {
                std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
            }
            #[cfg(target_arch = "aarch64")]
            unsafe {
                std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
            }
        }
    }

    pub fn add(&mut self, vector: &[f32]) -> u32 {
        debug_assert_eq!(vector.len(), self.dim);
        let id = (self.data.len() / self.dim) as u32;
        self.data.extend_from_slice(vector);
        id
    }
}

/// High-performance flat neighbor matrix.
#[derive(Debug, Serialize, Deserialize)]
pub struct NeighborMatrix {
    /// Contiguous data with a wider level-0 row and compact upper levels.
    pub data: Vec<AtomicU32>,
    pub stride: usize,
    pub m: usize,
    pub m0: usize,
    pub max_levels: usize,
    #[serde(skip)]
    pub locks: Vec<Mutex<()>>,
}

impl NeighborMatrix {
    pub fn new(max_levels: usize, m: usize) -> Self {
        let m0 = m * 2;
        let stride = (m0 + 1) + max_levels * (m + 1);
        Self {
            data: Vec::new(),
            stride,
            m,
            m0,
            max_levels,
            locks: Vec::new(),
        }
    }

    #[inline(always)]
    fn level_capacity(&self, level: u8) -> usize {
        if level == 0 { self.m0 + 1 } else { self.m + 1 }
    }

    #[inline(always)]
    fn level_offset(&self, level: u8) -> usize {
        if level == 0 {
            0
        } else {
            (self.m0 + 1) + (level as usize - 1) * (self.m + 1)
        }
    }

    pub fn ensure_capacity(&mut self, num_nodes: usize) {
        if self.locks.len() < num_nodes {
            let old_nodes = self.locks.len();
            let new_nodes = num_nodes - old_nodes;
            self.data
                .extend((0..new_nodes * self.stride).map(|_| AtomicU32::new(0)));
            self.locks.extend((0..new_nodes).map(|_| Mutex::new(())));
        }
    }

    #[inline(always)]
    pub fn with_neighbors<F, R>(&self, node_id: u32, level: u8, f: F) -> R
    where
        F: FnOnce(&[u32]) -> R,
    {
        let node_idx = node_id as usize;
        let level_idx = level as usize;

        if node_idx >= self.locks.len() || level_idx > self.max_levels {
            return f(&[]);
        }

        let slot_capacity = self.level_capacity(level);
        debug_assert!(slot_capacity <= MAX_LEVEL_CAPACITY);
        let base = node_idx * self.stride + self.level_offset(level);
        let count = self.data[base].load(Ordering::Acquire) as usize;
        let n = count.min(slot_capacity - 1);

        let mut buf = [0u32; MAX_LEVEL_CAPACITY];
        for i in 0..n {
            buf[i] = self.data[base + 1 + i].load(Ordering::Relaxed);
        }
        f(&buf[..n])
    }

    pub fn set_neighbors(&self, node_id: u32, level: u8, neighbors: &[u32]) {
        let _lock = self.locks[node_id as usize].lock();
        let slot_capacity = self.level_capacity(level);
        let base = node_id as usize * self.stride + self.level_offset(level);
        let n = neighbors.len().min(slot_capacity - 1);

        for i in 0..n {
            self.data[base + 1 + i].store(neighbors[i], Ordering::Relaxed);
        }
        self.data[base].store(n as u32, Ordering::Release);
    }

    #[inline(always)]
    pub fn prefetch(&self, node_id: u32, level: u8) {
        if !crate::vector::hnsw::prefetch::PrefetchConfig::enabled() {
            return;
        }
        let base = node_id as usize * self.stride + self.level_offset(level);
        if base < self.data.len() {
            let ptr = &self.data[base] as *const AtomicU32 as *const u8;
            #[cfg(target_arch = "x86_64")]
            unsafe {
                std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
            }
            #[cfg(target_arch = "aarch64")]
            unsafe {
                std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
            }
        }
    }
}

/// Proper unified storage for HNSWIndex.
#[derive(Debug, Serialize, Deserialize)]
pub struct HNSWStorage {
    pub vectors: VectorMatrix,
    pub neighbors: NeighborMatrix,
    /// Max level assigned to each node
    pub max_levels: Vec<AtomicU8>,
}

impl HNSWStorage {
    pub fn new(dim: usize, max_levels: usize, m: usize) -> Self {
        Self {
            vectors: VectorMatrix::new(dim),
            neighbors: NeighborMatrix::new(max_levels, m),
            max_levels: Vec::new(),
        }
    }

    pub fn with_capacity(dim: usize, max_levels: usize, m: usize, num_nodes: usize) -> Self {
        let mut neighbors = NeighborMatrix::new(max_levels, m);
        neighbors.ensure_capacity(num_nodes);
        Self {
            vectors: VectorMatrix::with_capacity(num_nodes, dim),
            neighbors,
            max_levels: Vec::with_capacity(num_nodes),
        }
    }

    #[inline(always)]
    pub fn vector(&self, id: u32) -> &[f32] {
        self.vectors.get(id)
    }

    #[inline(always)]
    pub fn prefetch_vector(&self, id: u32) {
        if crate::vector::hnsw::prefetch::PrefetchConfig::enabled() {
            self.vectors.prefetch(id);
        }
    }

    #[inline(always)]
    pub fn with_neighbors<F, R>(&self, id: u32, level: u8, f: F) -> R
    where
        F: FnOnce(&[u32]) -> R,
    {
        self.neighbors.with_neighbors(id, level, f)
    }

    pub fn add_node(&mut self, vector: &[f32], max_level: u8) -> u32 {
        let id = self.vectors.add(vector);
        self.neighbors.ensure_capacity(id as usize + 1);
        self.max_levels.push(AtomicU8::new(max_level));
        id
    }

    #[inline(always)]
    pub fn get_node_level(&self, id: u32) -> u8 {
        self.max_levels[id as usize].load(Ordering::Acquire)
    }

    pub fn reorder_bfs(&mut self, _entry: u32, _max_level: u8) -> Vec<u32> {
        (0..self.len() as u32).collect()
    }

    /// Compatibility alias for get_node_level
    #[inline(always)]
    pub fn level(&self, id: u32) -> u8 {
        self.get_node_level(id)
    }

    pub fn memory_usage(&self) -> usize {
        self.vectors.data.len() * 4
            + self.neighbors.data.len() * 4
            + self.max_levels.len()
            + self.neighbors.locks.len() * 64 // Rough estimate for Mutex
    }

    pub fn len(&self) -> usize {
        self.max_levels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn restore_locks(&mut self) {
        let num_nodes = self.len();
        self.neighbors.locks = (0..num_nodes).map(|_| Mutex::new(())).collect();
    }
}

impl NeighborMatrix {
    pub fn get_neighbors(&self, node_id: u32, level: u8) -> Vec<u32> {
        self.with_neighbors(node_id, level, <[u32]>::to_vec)
    }

    pub fn update_neighbors<F>(&self, node_id: u32, level: u8, f: F)
    where
        F: FnOnce(&mut [u32; MAX_LEVEL_CAPACITY], &mut usize),
    {
        let node_idx = node_id as usize;
        let level_idx = level as usize;

        if node_idx >= self.locks.len() || level_idx > self.max_levels {
            return;
        }

        let _lock = self.locks[node_idx].lock();
        let slot_capacity = self.level_capacity(level);
        debug_assert!(slot_capacity <= MAX_LEVEL_CAPACITY);
        let base = node_idx * self.stride + self.level_offset(level);
        let mut count = (self.data[base].load(Ordering::Acquire) as usize).min(slot_capacity - 1);

        let mut buf = [0u32; MAX_LEVEL_CAPACITY];
        for (dst, src) in buf
            .iter_mut()
            .zip((0..count).map(|i| self.data[base + 1 + i].load(Ordering::Relaxed)))
        {
            *dst = src;
        }

        f(&mut buf, &mut count);

        let n = count.min(slot_capacity - 1);
        for (i, neighbor) in buf.iter().take(n).enumerate() {
            self.data[base + 1 + i].store(*neighbor, Ordering::Relaxed);
        }
        self.data[base].store(n as u32, Ordering::Release);
    }

    pub fn contains_neighbor(&self, node_id: u32, level: u8, neighbor: u32) -> bool {
        self.with_neighbors(node_id, level, |n| n.contains(&neighbor))
    }
}

#[cfg(test)]
mod tests_stride;
