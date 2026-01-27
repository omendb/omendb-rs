//! ACORN-1 Filtered Search Algorithm (arXiv:2403.04871)
//!
//! This module provides a shared implementation of the ACORN-1 2-hop neighbor
//! expansion algorithm used by both HNSWIndex and FrozenSegment.
//!
//! The key insight is that both storage types can provide neighbors via
//! `Cow<'_, [u32]>`, allowing zero-copy access where possible.

use super::query_buffers::VisitedList;
use std::borrow::Cow;

/// Trait for accessing neighbors in HNSW graph structures.
///
/// Implemented by both `NodeStorage` (HNSWIndex) and `FrozenStorage` (FrozenSegment).
pub trait NeighborStorage {
    /// Get neighbors at a specific level, zero-copy where possible.
    fn neighbors_at_level_cow(&self, node: u32, level: u8) -> Cow<'_, [u32]>;
}

/// ACORN-1 GET-NEIGHBORS: Collect matching neighbors with 2-hop expansion.
///
/// From arXiv:2403.04871:
/// - If a 1-hop neighbor matches the filter, add it directly
/// - If a 1-hop neighbor doesn't match, expand to its neighbors (2-hop)
/// - Stop early once M matching neighbors are found (truncation)
///
/// This is the core ACORN-1 algorithm step, shared by HNSWIndex and FrozenSegment.
#[inline]
pub fn collect_matching_neighbors<S, F>(
    storage: &S,
    source_node: u32,
    level: u8,
    visited: &VisitedList,
    filter_fn: &F,
    m: usize,
    output: &mut Vec<u32>,
) where
    S: NeighborStorage,
    F: Fn(u32) -> bool,
{
    output.clear();
    let neighbors_1hop = storage.neighbors_at_level_cow(source_node, level);

    for &neighbor_id in &*neighbors_1hop {
        if visited.contains(neighbor_id) {
            continue;
        }
        if filter_fn(neighbor_id) {
            output.push(neighbor_id);
            if output.len() >= m {
                return;
            }
        } else {
            // 2-hop expansion for non-matching neighbors (zero-copy via Cow)
            let second_hop = storage.neighbors_at_level_cow(neighbor_id, level);
            for &second_hop_id in &*second_hop {
                if !visited.contains(second_hop_id) && filter_fn(second_hop_id) {
                    output.push(second_hop_id);
                    if output.len() >= m {
                        return;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestStorage {
        neighbors: Vec<Vec<u32>>,
    }

    impl NeighborStorage for TestStorage {
        fn neighbors_at_level_cow(&self, node: u32, _level: u8) -> Cow<'_, [u32]> {
            if (node as usize) < self.neighbors.len() {
                Cow::Borrowed(&self.neighbors[node as usize])
            } else {
                Cow::Borrowed(&[])
            }
        }
    }

    #[test]
    fn test_collect_matching_direct() {
        // Simple graph: 0 -> [1, 2, 3], all match filter
        let storage = TestStorage {
            neighbors: vec![
                vec![1, 2, 3], // node 0's neighbors
                vec![0],       // node 1
                vec![0],       // node 2
                vec![0],       // node 3
            ],
        };

        let visited = VisitedList::new();
        let filter = |_id: u32| true; // all match
        let mut output = Vec::new();

        collect_matching_neighbors(&storage, 0, 0, &visited, &filter, 10, &mut output);

        assert_eq!(output, vec![1, 2, 3]);
    }

    #[test]
    fn test_collect_matching_2hop() {
        // Graph: 0 -> [1], 1 -> [2, 3]
        // Filter: only even nodes match
        // Should find 2 via 2-hop expansion through 1
        let storage = TestStorage {
            neighbors: vec![
                vec![1],    // node 0 -> 1 (doesn't match)
                vec![2, 3], // node 1 -> 2, 3 (2 matches)
                vec![],     // node 2
                vec![],     // node 3
            ],
        };

        let visited = VisitedList::new();
        let filter = |id: u32| id % 2 == 0; // even nodes match
        let mut output = Vec::new();

        collect_matching_neighbors(&storage, 0, 0, &visited, &filter, 10, &mut output);

        assert_eq!(output, vec![2]); // Found via 2-hop
    }

    #[test]
    fn test_collect_truncation() {
        // Many neighbors, but limit to M=2
        let storage = TestStorage {
            neighbors: vec![vec![1, 2, 3, 4, 5]],
        };

        let visited = VisitedList::new();
        let filter = |_id: u32| true;
        let mut output = Vec::new();

        collect_matching_neighbors(&storage, 0, 0, &visited, &filter, 2, &mut output);

        assert_eq!(output.len(), 2);
    }
}
