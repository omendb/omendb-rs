/*
 * Greedy Search Algorithm for IP-DiskANN
 *
 * Based on:
 * - DiskANN paper (NeurIPS 2019), Algorithm 1
 * - Microsoft's Rust implementation (~/github/microsoft/DiskANN/rust/diskann/src/algorithm/search/search.rs)
 * - Vamana algorithm study (ai/research/vamana_algorithm_notes.md)
 *
 * Greedy search traverses the graph to find k nearest neighbors.
 * Maintains a priority queue of best candidates and expands them greedily.
 */

use super::graph::BiDirectionalGraph;
use super::types::{Neighbor, NodeId, Distance};
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;

/// Search for k nearest neighbors using greedy graph traversal
///
/// # Algorithm (based on DiskANN paper + Microsoft implementation)
/// 1. Start from entry_node
/// 2. Maintain a priority queue of candidates (best L candidates)
/// 3. While there are better unvisited candidates:
///    - Pop the closest unvisited candidate
///    - Explore its neighbors
///    - Add new neighbors to the candidate pool
///    - Keep only best L candidates
/// 4. Return top-k from the candidate pool
///
/// # Parameters
/// - `query`: Query vector
/// - `graph`: The bi-directional graph to search
/// - `vectors`: Vector storage (for distance computation)
/// - `k`: Number of nearest neighbors to return
/// - `search_list_size`: Size of search list (L in paper, controls quality/speed tradeoff)
///
/// # Returns
/// Vector of k nearest neighbors (sorted by distance, ascending)
pub fn greedy_search(
    query: &[f32],
    graph: &BiDirectionalGraph,
    vectors: &std::collections::HashMap<NodeId, Vec<f32>>,
    k: usize,
    search_list_size: usize,
) -> Vec<Neighbor> {
    // Get entry node (start point for search)
    let entry_node = match graph.entry_node() {
        Some(node) => node,
        None => return Vec::new(), // Empty graph
    };

    // Best candidates pool (sorted by distance, best first)
    let mut best_candidates = BinaryHeap::new();

    // Track which nodes we've visited (explored their neighbors)
    let mut visited = HashSet::new();

    // Track which nodes we've seen (in best_candidates)
    let mut seen = HashSet::new();

    // Initialize with entry node
    if let Some(entry_vector) = vectors.get(&entry_node) {
        let distance = l2_distance(query, entry_vector);
        let neighbor = Neighbor::new(entry_node, distance);
        best_candidates.push(Reverse(neighbor));
        seen.insert(entry_node);
    }

    // Greedy traversal: keep exploring until no better candidates
    loop {
        // Find the closest unvisited candidate
        let mut closest_unvisited = None;
        let mut temp_candidates = Vec::new();

        while let Some(Reverse(candidate)) = best_candidates.pop() {
            temp_candidates.push(candidate);
            if !visited.contains(&candidate.id) {
                closest_unvisited = Some(candidate);
                break;
            }
        }

        // Restore candidates we popped
        for candidate in temp_candidates {
            best_candidates.push(Reverse(candidate));
        }

        // If no unvisited candidates, we're done
        let current = match closest_unvisited {
            Some(c) => c,
            None => break,
        };

        // Mark as visited
        visited.insert(current.id);

        // Explore neighbors
        if let Some(node) = graph.get_node(current.id) {
            for &neighbor_id in &node.out_neighbors {
                // Skip if already seen
                if seen.contains(&neighbor_id) {
                    continue;
                }

                // Compute distance to query
                if let Some(neighbor_vector) = vectors.get(&neighbor_id) {
                    let distance = l2_distance(query, neighbor_vector);
                    let neighbor = Neighbor::new(neighbor_id, distance);

                    // Add to candidate pool
                    best_candidates.push(Reverse(neighbor));
                    seen.insert(neighbor_id);
                }
            }
        }

        // Prune to top-L candidates
        if best_candidates.len() > search_list_size {
            let mut sorted_candidates: Vec<_> = best_candidates
                .into_iter()
                .map(|Reverse(n)| n)
                .collect();
            sorted_candidates.sort();
            sorted_candidates.truncate(search_list_size);

            best_candidates = sorted_candidates
                .into_iter()
                .map(Reverse)
                .collect();
        }
    }

    // Return top-k candidates
    let mut result: Vec<_> = best_candidates
        .into_iter()
        .map(|Reverse(n)| n)
        .collect();
    result.sort();
    result.truncate(k);
    result
}

/// Pop the closest unvisited candidate from the heap
///
/// We track which candidates have been "visited" (expanded) separately
/// because the heap contains both visited and unvisited.
fn pop_closest_unvisited(heap: &mut BinaryHeap<Reverse<Neighbor>>) -> Option<Neighbor> {
    // In a real implementation, we'd track visited status more efficiently
    // For now, just pop the minimum (closest)
    heap.pop().map(|Reverse(n)| n)
}

/// L2 (Euclidean) distance between two vectors
#[inline]
fn l2_distance(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::graph::BiDirectionalGraph;

    #[test]
    fn test_greedy_search_finds_nearest() {
        // Create a simple graph: 0 -> 1 -> 2
        let mut graph = BiDirectionalGraph::new();
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.set_entry_node(0);

        // Vectors
        let mut vectors = std::collections::HashMap::new();
        vectors.insert(0, vec![0.0, 0.0]);
        vectors.insert(1, vec![1.0, 0.0]);
        vectors.insert(2, vec![2.0, 0.0]);

        // Query close to node 1
        let query = vec![0.9, 0.0];

        let results = greedy_search(&query, &graph, &vectors, 1, 10);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1); // Node 1 is closest
    }

    #[test]
    fn test_greedy_search_returns_k_neighbors() {
        // Create a graph with multiple nodes
        let mut graph = BiDirectionalGraph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(1, 3);
        graph.set_entry_node(0);

        let mut vectors = std::collections::HashMap::new();
        vectors.insert(0, vec![0.0]);
        vectors.insert(1, vec![1.0]);
        vectors.insert(2, vec![2.0]);
        vectors.insert(3, vec![3.0]);

        let query = vec![0.5];

        let results = greedy_search(&query, &graph, &vectors, 2, 10);

        assert_eq!(results.len(), 2);
        // Should return the 2 closest (0 and 1)
        assert!(results[0].id == 0 || results[0].id == 1);
        assert!(results[1].id == 0 || results[1].id == 1);
    }

    #[test]
    fn test_greedy_search_empty_graph() {
        let graph = BiDirectionalGraph::new();
        let vectors = std::collections::HashMap::new();
        let query = vec![1.0];

        let results = greedy_search(&query, &graph, &vectors, 5, 10);

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_l2_distance() {
        let v1 = vec![0.0, 0.0];
        let v2 = vec![3.0, 4.0];

        let dist = l2_distance(&v1, &v2);
        assert!((dist - 5.0).abs() < 1e-6); // 3-4-5 triangle
    }
}
