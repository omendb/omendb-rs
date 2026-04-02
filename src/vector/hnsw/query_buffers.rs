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

const SEARCH_RESULT_PARTIAL_SORT_THRESHOLD: usize = 4;

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
    pub fn prefetch(&self, id: u32) {
        let idx = id as usize;
        if idx < self.visited.len() {
            let ptr = self.visited.as_ptr().wrapping_add(idx);
            #[cfg(target_arch = "x86_64")]
            // SAFETY: Pointer within bounds-checked array/slice, prefetch is a non-faulting read hint
            unsafe {
                std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
            }
            #[cfg(target_arch = "aarch64")]
            // SAFETY: Pointer within bounds-checked array/slice, prefetch is a non-faulting read hint
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

/// Reusable workset for query-time HNSW traversal.
///
/// This owns the full mutable search state so the hot path does not have to
/// manually coordinate multiple buffers and heaps.
pub struct SearchWorkset {
    visited: VisitedList,
    frontier: BinaryHeap<Reverse<Candidate>>,
    accepted: BinaryHeap<Candidate>,
    neighbors: Vec<u32>,
    finalized: Vec<Candidate>,
    batch_distances: Vec<f32>,
}

impl Default for SearchWorkset {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchWorkset {
    /// Create new empty workset
    pub fn new() -> Self {
        Self {
            visited: VisitedList::new(),
            frontier: BinaryHeap::new(),
            accepted: BinaryHeap::new(),
            neighbors: Vec::new(),
            finalized: Vec::new(),
            batch_distances: Vec::new(),
        }
    }

    /// Clear all state for reuse
    pub fn clear(&mut self) {
        self.visited.clear();
        self.frontier.clear();
        self.accepted.clear();
        self.neighbors.clear();
        self.finalized.clear();
        // batch_distances doesn't need clearing - overwritten each use
    }

    pub fn entry_buffers(&mut self) -> (&mut VisitedList, &mut Vec<u32>) {
        (&mut self.visited, &mut self.neighbors)
    }

    #[inline(always)]
    pub fn collector_buffers(&mut self) -> (&VisitedList, &mut Vec<u32>) {
        (&self.visited, &mut self.neighbors)
    }

    #[inline(always)]
    pub fn take_neighbors(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.neighbors)
    }

    #[inline(always)]
    pub fn restore_neighbors(&mut self, neighbors: Vec<u32>) {
        self.neighbors = neighbors;
    }

    #[inline(always)]
    pub fn take_batch_distances(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.batch_distances)
    }

    #[inline(always)]
    pub fn restore_batch_distances(&mut self, batch_distances: Vec<f32>) {
        self.batch_distances = batch_distances;
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.frontier.is_empty()
    }

    #[inline(always)]
    pub fn pop_frontier(&mut self) -> Option<Candidate> {
        self.frontier.pop().map(|Reverse(candidate)| candidate)
    }

    #[inline(always)]
    pub fn should_stop(&self, current: Candidate) -> bool {
        self.accepted
            .peek()
            .is_some_and(|&farthest| current.distance > farthest.distance)
    }

    #[inline(always)]
    pub fn seed(&mut self, candidate: Candidate) {
        self.frontier.push(Reverse(candidate));
        self.accepted.push(candidate);
    }

    #[inline(always)]
    pub fn record_visited(&mut self, node_id: u32) -> bool {
        if self.visited.contains(node_id) {
            return false;
        }
        self.visited.insert(node_id);
        true
    }

    #[inline(always)]
    pub fn prefetch_visited(&self, node_id: u32) {
        self.visited.prefetch(node_id);
    }

    #[inline(always)]
    pub fn consider(&mut self, neighbor: Candidate, ef: usize) {
        let admit = self
            .accepted
            .peek()
            .is_none_or(|&farthest| neighbor.distance < farthest.distance || self.accepted.len() < ef);
        if !admit {
            return;
        }

        self.frontier.push(Reverse(neighbor));
        self.accepted.push(neighbor);
        if self.accepted.len() > ef {
            self.accepted.pop();
        }
    }

    pub fn finalize(&mut self, result_limit: usize) -> Vec<(u32, f32)> {
        if result_limit == 0 || self.accepted.is_empty() {
            return Vec::new();
        }

        self.finalized.extend(self.accepted.drain());
        let result_limit = result_limit.min(self.finalized.len());

        if self.finalized.len() > result_limit.saturating_mul(SEARCH_RESULT_PARTIAL_SORT_THRESHOLD)
        {
            self.finalized
                .select_nth_unstable_by(result_limit, |a, b| a.distance.cmp(&b.distance));
            self.finalized.truncate(result_limit);
            self.finalized
                .sort_unstable_by(|a, b| a.distance.cmp(&b.distance));
        } else {
            self.finalized
                .sort_unstable_by(|a, b| a.distance.cmp(&b.distance));
            self.finalized.truncate(result_limit);
        }

        self.finalized
            .iter()
            .map(|candidate| (candidate.node_id, candidate.distance.into_inner()))
            .collect()
    }
}

thread_local! {
    /// Thread-local query buffers
    ///
    /// Each thread gets its own buffers, avoiding contention and allocations.
    static QUERY_WORKSET: RefCell<SearchWorkset> = RefCell::new(SearchWorkset::new());
}

/// Use thread-local search workset for a query
///
/// Clears state before use. Buffers retain capacity across queries
/// for amortized allocation.
pub fn with_workset<F, R>(f: F) -> R
where
    F: FnOnce(&mut SearchWorkset) -> R,
{
    QUERY_WORKSET.with(|workset| {
        let mut workset = workset.borrow_mut();
        workset.clear();
        f(&mut workset)
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
    fn test_search_workset_creation() {
        let workset = SearchWorkset::new();
        assert!(workset.visited.is_empty());
        assert!(workset.is_empty());
        assert!(workset.neighbors.is_empty());
    }

    #[test]
    fn test_search_workset_clear() {
        let mut workset = SearchWorkset::new();

        // Add some data
        workset.record_visited(1);
        workset.neighbors.push(0);
        workset.seed(Candidate::new(42, 1.0));

        // Clear
        workset.clear();

        assert!(!workset.visited.contains(1));
        assert!(workset.neighbors.is_empty());
        assert!(workset.is_empty());
    }

    #[test]
    fn test_with_workset() {
        // Use workset
        with_workset(|workset| {
            workset.record_visited(42);
            assert!(workset.visited.contains(42));
        });

        // Workset should be cleared after use
        with_workset(|workset| {
            assert!(!workset.visited.contains(42));
        });
    }

    #[test]
    fn test_thread_local_isolation() {
        use std::thread;

        // Main thread
        with_workset(|workset| {
            workset.record_visited(1);
        });

        // Spawn new thread
        let handle = thread::spawn(|| {
            with_workset(|workset| {
                // Should not see main thread's data
                assert!(!workset.visited.contains(1));
                workset.record_visited(2);
            });
        });

        handle.join().unwrap();

        // Main thread should not see spawned thread's data
        with_workset(|workset| {
            assert!(!workset.visited.contains(2));
        });
    }

    #[test]
    fn finalize_uses_partial_sort_for_top_k() {
        let mut workset = SearchWorkset::new();
        for candidate in [
            Candidate::new(10, 9.0),
            Candidate::new(11, 1.0),
            Candidate::new(12, 8.0),
            Candidate::new(13, 2.0),
            Candidate::new(14, 7.0),
            Candidate::new(15, 3.0),
            Candidate::new(16, 6.0),
            Candidate::new(17, 4.0),
            Candidate::new(18, 5.0),
        ] {
            workset.seed(candidate);
        }

        let output = workset.finalize(2);

        assert_eq!(output, vec![(11, 1.0), (13, 2.0)]);
    }

    #[test]
    fn finalize_keeps_full_order_when_small() {
        let mut workset = SearchWorkset::new();
        for candidate in [
            Candidate::new(10, 3.0),
            Candidate::new(11, 1.0),
            Candidate::new(12, 2.0),
        ] {
            workset.seed(candidate);
        }

        let output = workset.finalize(3);

        assert_eq!(output, vec![(11, 1.0), (12, 2.0), (10, 3.0)]);
    }
}
