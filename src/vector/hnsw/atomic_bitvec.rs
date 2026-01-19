//! Atomic bitmap for tracking ready nodes during parallel HNSW construction
//!
//! Used to mark nodes as "ready" (fully connected) so parallel searches
//! only consider nodes that have complete neighbor lists.

use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic bitmap for tracking ready nodes
///
/// Thread-safe bitmap using atomic operations. Supports concurrent
/// set operations and reads without external synchronization.
#[derive(Debug)]
pub struct AtomicBitVec {
    /// Atomic words, each holding 64 bits
    words: Vec<AtomicU64>,
    /// Number of bits (nodes) the bitmap can track
    capacity: usize,
}

impl AtomicBitVec {
    /// Create a new bitmap with the given capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let num_words = capacity.div_ceil(64);
        let words = (0..num_words).map(|_| AtomicU64::new(0)).collect();
        Self { words, capacity }
    }

    /// Create an empty bitmap
    #[must_use]
    pub fn empty() -> Self {
        Self {
            words: Vec::new(),
            capacity: 0,
        }
    }

    /// Current capacity in bits
    #[inline]
    #[must_use]
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Grow the bitmap to accommodate at least `new_capacity` bits
    ///
    /// This is NOT thread-safe - must be called with exclusive access.
    #[allow(dead_code)]
    pub fn grow(&mut self, new_capacity: usize) {
        if new_capacity <= self.capacity {
            return;
        }

        let new_num_words = new_capacity.div_ceil(64);
        let old_num_words = self.words.len();

        // Add new zeroed words
        for _ in old_num_words..new_num_words {
            self.words.push(AtomicU64::new(0));
        }

        self.capacity = new_capacity;
    }

    /// Set a bit (mark node as ready)
    ///
    /// Thread-safe: uses atomic OR operation.
    /// Debug builds assert on out-of-bounds; release builds silently no-op.
    #[inline]
    pub fn set(&self, idx: usize) {
        debug_assert!(
            idx < self.capacity,
            "AtomicBitVec::set out of bounds: idx={}, capacity={}",
            idx,
            self.capacity
        );
        if idx >= self.capacity {
            return;
        }
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        let mask = 1u64 << bit_idx;
        self.words[word_idx].fetch_or(mask, Ordering::Release);
    }

    /// Check if a bit is set (node is ready)
    ///
    /// Thread-safe: uses atomic load.
    /// Debug builds assert on out-of-bounds; release builds return false.
    #[inline]
    #[must_use]
    pub fn is_ready(&self, idx: usize) -> bool {
        debug_assert!(
            idx < self.capacity,
            "AtomicBitVec::is_ready out of bounds: idx={}, capacity={}",
            idx,
            self.capacity
        );
        if idx >= self.capacity {
            return false;
        }
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        let mask = 1u64 << bit_idx;
        (self.words[word_idx].load(Ordering::Acquire) & mask) != 0
    }

    /// Clear all bits
    ///
    /// This is NOT thread-safe - must be called with exclusive access.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        for word in &self.words {
            word.store(0, Ordering::Relaxed);
        }
    }

    /// Count the number of set bits
    #[must_use]
    #[allow(dead_code)]
    pub fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as usize)
            .sum()
    }
}

impl Default for AtomicBitVec {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_basic_operations() {
        let bv = AtomicBitVec::new(128);

        assert!(!bv.is_ready(0));
        assert!(!bv.is_ready(63));
        assert!(!bv.is_ready(64));
        assert!(!bv.is_ready(127));

        bv.set(0);
        bv.set(63);
        bv.set(64);
        bv.set(127);

        assert!(bv.is_ready(0));
        assert!(bv.is_ready(63));
        assert!(bv.is_ready(64));
        assert!(bv.is_ready(127));
        assert!(!bv.is_ready(1));
        assert!(!bv.is_ready(65));

        assert_eq!(bv.count_ones(), 4);
    }

    #[test]
    fn test_grow() {
        let mut bv = AtomicBitVec::new(64);
        assert_eq!(bv.capacity(), 64);

        bv.set(63);
        bv.grow(256);

        assert_eq!(bv.capacity(), 256);
        assert!(bv.is_ready(63)); // Old data preserved
        assert!(!bv.is_ready(128));

        bv.set(200);
        assert!(bv.is_ready(200));
    }

    #[test]
    fn test_concurrent_set() {
        let bv = Arc::new(AtomicBitVec::new(1000));
        let mut handles = Vec::new();

        // Spawn threads that each set different bits
        for t in 0..10 {
            let bv = Arc::clone(&bv);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    bv.set(t * 100 + i);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // All bits should be set
        assert_eq!(bv.count_ones(), 1000);
        for i in 0..1000 {
            assert!(bv.is_ready(i), "Bit {} should be set", i);
        }
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_out_of_bounds() {
        // This test only runs in release mode since debug builds assert on OOB
        let bv = AtomicBitVec::new(64);

        // Out of bounds set should be no-op
        bv.set(100);
        assert!(!bv.is_ready(100));

        // Out of bounds read should return false
        assert!(!bv.is_ready(1000));
    }

    #[test]
    fn test_clear() {
        let mut bv = AtomicBitVec::new(128);

        for i in 0..128 {
            bv.set(i);
        }
        assert_eq!(bv.count_ones(), 128);

        bv.clear();
        assert_eq!(bv.count_ones(), 0);

        for i in 0..128 {
            assert!(!bv.is_ready(i));
        }
    }
}
