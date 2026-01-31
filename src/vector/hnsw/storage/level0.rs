// Level 0 neighbor storage with atomic operations
//
// Lock-free reads via atomic operations. O(1) inserts without copy-on-write.
// Per-node mutex for write coordination (NOT global lock).
//
// Level 0: 95%+ of accesses during search - dense atomic array

use parking_lot::Mutex;
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering};

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

    /// Add a single neighbor with locking - O(1) append
    /// Returns false if at capacity
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

    /// Check if neighbor exists (lock-free)
    #[inline]
    pub fn contains(&self, node_id: u32, neighbor: u32) -> bool {
        let idx = node_id as usize;
        if idx >= self.counts.len() {
            return false;
        }
        let base = idx * self.max_m0;
        let count = self.counts[idx].load(Ordering::Acquire) as usize;
        for i in 0..count.min(self.max_m0) {
            if self.data[base + i].load(Ordering::Relaxed) == neighbor {
                return true;
            }
        }
        false
    }

    /// Get neighbor count (lock-free)
    #[inline]
    pub fn count(&self, node_id: u32) -> usize {
        let idx = node_id as usize;
        if idx >= self.counts.len() {
            return 0;
        }
        self.counts[idx].load(Ordering::Acquire) as usize
    }

    pub fn num_nodes(&self) -> usize {
        self.counts.len()
    }

    pub fn memory_usage(&self) -> usize {
        self.data.len() * 4 + self.counts.len() * 2 + self.write_locks.len() * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
