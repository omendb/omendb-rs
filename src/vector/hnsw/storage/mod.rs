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

const CHUNK_SIZE: usize = 16; // 64 bytes (1 count + 15 neighbors)

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
        let start = id as usize * self.dim;
        &self.data[start..start + self.dim]
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
    /// Contiguous data: [node_id * stride + level * CHUNK_SIZE]
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
        let stride = (max_levels + 1) * CHUNK_SIZE;
        Self {
            data: Vec::new(),
            stride,
            m,
            m0: m * 2,
            max_levels,
            locks: Vec::new(),
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

        let base = node_idx * self.stride + level_idx * CHUNK_SIZE;
        let count = self.data[base].load(Ordering::Acquire) as usize;
        let n = count.min(CHUNK_SIZE - 1);

        let mut buf = [0u32; CHUNK_SIZE];
        for i in 0..n {
            buf[i] = self.data[base + 1 + i].load(Ordering::Relaxed);
        }
        f(&buf[..n])
    }

    pub fn set_neighbors(&self, node_id: u32, level: u8, neighbors: &[u32]) {
        let _lock = self.locks[node_id as usize].lock();
        let base = node_id as usize * self.stride + level as usize * CHUNK_SIZE;
        let n = neighbors.len().min(CHUNK_SIZE - 1);

        for i in 0..n {
            self.data[base + 1 + i].store(neighbors[i], Ordering::Relaxed);
        }
        self.data[base].store(n as u32, Ordering::Release);
    }

    #[inline(always)]
    pub fn prefetch(&self, node_id: u32, level: u8) {
        let base = node_id as usize * self.stride + level as usize * CHUNK_SIZE;
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

    #[inline(always)]
    pub fn vector(&self, id: u32) -> &[f32] {
        self.vectors.get(id)
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
}

impl NeighborMatrix {
    pub fn get_neighbors(&self, node_id: u32, level: u8) -> Vec<u32> {
        self.with_neighbors(node_id, level, |n| n.to_vec())
    }

    pub fn contains_neighbor(&self, node_id: u32, level: u8, neighbor: u32) -> bool {
        self.with_neighbors(node_id, level, |n| n.contains(&neighbor))
    }
}
