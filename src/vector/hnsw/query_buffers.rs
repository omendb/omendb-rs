// Thread-local query buffers for allocation-free search
//
// Reuses temporary buffers across queries to reduce allocations.
// From profiling: 7.3M allocations identified (76% in search operations).
//
// Thread-local storage ensures:
// - No contention between threads
// - Amortizes allocation cost across queries
// - 10-15% performance improvement expected
//
// Optimization (Nov 25, 2025):
// - Replaced HashSet with VisitedList (generation-based, O(1) clear)
// - This is how hnswlib achieves fast visited tracking

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::types::Candidate;

/// Fast visited list using generation markers (like hnswlib)
///
/// O(1) insert, O(1) contains, O(1) clear (just increment generation)
/// Much faster than `HashSet` for HNSW traversal.
pub struct VisitedList {
    /// visited[i] = generation when node i was last visited
    visited: Vec<u32>,
    /// Current generation (incremented on clear)
    generation: u32,
}

impl Default for VisitedList {
    fn default() -> Self {
        Self::new()
    }
}

impl VisitedList {
    /// Create new empty visited list
    pub fn new() -> Self {
        Self {
            visited: Vec::new(),
            generation: 1, // Start at 1 so 0 means "never visited"
        }
    }

    /// O(1) clear - just increment generation
    #[inline]
    pub fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Rare wraparound case: reset everything
            self.visited.fill(0);
            self.generation = 1;
        }
    }

    /// Check if node was visited this generation
    #[inline]
    pub fn contains(&self, id: u32) -> bool {
        self.visited.get(id as usize).copied() == Some(self.generation)
    }

    /// Mark node as visited
    #[inline]
    pub fn insert(&mut self, id: u32) {
        let idx = id as usize;
        if idx >= self.visited.len() {
            // Grow to accommodate new node (amortized O(1))
            self.visited.resize(idx + 1, 0);
        }
        self.visited[idx] = self.generation;
    }

    /// Prefetch visited array entry for a node (hides memory latency)
    ///
    /// Call this 1-2 iterations ahead to ensure data is in L1 cache.
    /// Like hnswlib, we prefetch the visited array alongside vector data.
    #[inline]
    #[allow(dead_code)] // Future optimization
    pub fn prefetch(&self, id: u32) {
        let idx = id as usize;
        if idx < self.visited.len() {
            let ptr = self.visited.as_ptr().wrapping_add(idx);
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

    /// Check if empty (no nodes visited this generation)
    #[inline]
    #[allow(dead_code)] // Standard API
    pub fn is_empty(&self) -> bool {
        !self.visited.contains(&self.generation)
    }
}

/// Reusable buffers for search operations
///
/// These are cleared and reused across queries to avoid allocations.
pub struct QueryBuffers {
    /// Visited nodes during graph traversal (fast generation-based)
    pub visited: VisitedList,

    /// Candidate queue (min-heap)
    pub candidates: BinaryHeap<Reverse<Candidate>>,

    /// Working set (max-heap)
    pub working: BinaryHeap<Candidate>,

    /// Entry points for layer traversal
    pub entry_points: Vec<u32>,

    /// Pre-allocated buffer for unvisited neighbors
    pub unvisited: Vec<u32>,

    /// Pre-allocated buffer for search results (avoids allocation in return path)
    pub results: Vec<Candidate>,

    /// Pre-allocated buffer for batch distance computation (SQ8 mode)
    pub batch_distances: Vec<f32>,
}

impl Default for QueryBuffers {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryBuffers {
    /// Create new empty buffers
    pub fn new() -> Self {
        Self {
            visited: VisitedList::new(),
            candidates: BinaryHeap::new(),
            working: BinaryHeap::new(),
            entry_points: Vec::new(),
            unvisited: Vec::new(),
            results: Vec::new(),
            batch_distances: Vec::new(),
        }
    }

    /// Clear all buffers for reuse
    pub fn clear(&mut self) {
        self.visited.clear(); // O(1) now!
        self.candidates.clear();
        self.working.clear();
        self.entry_points.clear();
        self.unvisited.clear();
        self.results.clear();
        // batch_distances doesn't need clearing - overwritten each use
    }
}

thread_local! {
    /// Thread-local query buffers
    ///
    /// Each thread gets its own buffers, avoiding contention and allocations.
    static QUERY_BUFFERS: RefCell<QueryBuffers> = RefCell::new(QueryBuffers::new());
}

/// Use thread-local buffers for a query
///
/// Clears buffers before use. Buffers retain capacity across queries
/// for amortized allocation.
pub fn with_buffers<F, R>(f: F) -> R
where
    F: FnOnce(&mut QueryBuffers) -> R,
{
    QUERY_BUFFERS.with(|buffers| {
        let mut buffers = buffers.borrow_mut();
        buffers.clear();
        f(&mut buffers)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visited_list_basic() {
        let mut visited = VisitedList::new();

        assert!(!visited.contains(0));
        assert!(!visited.contains(100));

        visited.insert(42);
        assert!(visited.contains(42));
        assert!(!visited.contains(0));

        visited.insert(100);
        assert!(visited.contains(42));
        assert!(visited.contains(100));
    }

    #[test]
    fn test_visited_list_clear() {
        let mut visited = VisitedList::new();

        visited.insert(1);
        visited.insert(2);
        visited.insert(3);

        assert!(visited.contains(1));
        assert!(visited.contains(2));
        assert!(visited.contains(3));

        // Clear should reset in O(1)
        visited.clear();

        assert!(!visited.contains(1));
        assert!(!visited.contains(2));
        assert!(!visited.contains(3));

        // Should be able to reuse
        visited.insert(1);
        assert!(visited.contains(1));
        assert!(!visited.contains(2));
    }

    #[test]
    fn test_visited_list_generation_reuse() {
        let mut visited = VisitedList::new();

        // Multiple clear cycles should work correctly
        for _ in 0..10 {
            visited.insert(42);
            assert!(visited.contains(42));
            visited.clear();
            assert!(!visited.contains(42));
        }
    }

    #[test]
    fn test_query_buffers_creation() {
        let buffers = QueryBuffers::new();
        assert!(buffers.visited.is_empty());
        assert!(buffers.candidates.is_empty());
        assert!(buffers.working.is_empty());
        assert!(buffers.entry_points.is_empty());
    }

    #[test]
    fn test_query_buffers_clear() {
        let mut buffers = QueryBuffers::new();

        // Add some data
        buffers.visited.insert(1);
        buffers.entry_points.push(0);

        // Clear
        buffers.clear();

        assert!(!buffers.visited.contains(1));
        assert!(buffers.entry_points.is_empty());
    }

    #[test]
    fn test_with_buffers() {
        // Use buffers
        with_buffers(|buffers| {
            buffers.visited.insert(42);
            assert!(buffers.visited.contains(42));
        });

        // Buffers should be cleared after use
        with_buffers(|buffers| {
            assert!(!buffers.visited.contains(42));
        });
    }

    #[test]
    fn test_thread_local_isolation() {
        use std::thread;

        // Main thread
        with_buffers(|buffers| {
            buffers.visited.insert(1);
        });

        // Spawn new thread
        let handle = thread::spawn(|| {
            with_buffers(|buffers| {
                // Should not see main thread's data
                assert!(!buffers.visited.contains(1));
                buffers.visited.insert(2);
            });
        });

        handle.join().unwrap();

        // Main thread should not see spawned thread's data
        with_buffers(|buffers| {
            assert!(!buffers.visited.contains(2));
        });
    }
}
