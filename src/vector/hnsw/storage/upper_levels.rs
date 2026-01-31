// Upper level neighbor storage for HNSW
//
// Sparse per-node allocation for levels 1+.
// Only ~5-10% of nodes have upper levels.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};

/// Upper level links for a single node (levels 1..=max_level)
#[derive(Debug)]
pub struct UpperNodeLinks {
    /// All levels in one allocation: [l1_neighbors][l2_neighbors]...
    pub(crate) data: Vec<AtomicU32>,

    /// Count per level
    pub(crate) counts: Vec<AtomicU8>,

    pub(crate) max_level: u8,
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

    /// Add neighbor at upper level with locking - O(1) append
    /// Returns false if at capacity
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

    /// Get neighbor count at level (lock-free)
    #[inline]
    pub fn count(&self, node_id: u32, level: u8) -> usize {
        debug_assert!(level >= 1);
        let idx = node_id as usize;
        if idx >= self.nodes.len() {
            return 0;
        }
        match &self.nodes[idx] {
            Some(links) if level <= links.max_level => {
                let level_idx = (level - 1) as usize;
                links.counts[level_idx].load(Ordering::Acquire) as usize
            }
            _ => 0,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
