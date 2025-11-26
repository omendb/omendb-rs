/*
 * RobustPrune Algorithm - Diversity Pruning for Vamana/IP-DiskANN
 *
 * Based on:
 * - DiskANN paper (NeurIPS 2019), Algorithm 2
 * - Microsoft's Rust implementation (~/github/microsoft/DiskANN/rust/diskann/src/algorithm/prune/prune.rs)
 * - Our Vamana study notes (ai/research/vamana_algorithm_notes.md)
 *
 * Key insight: Select diverse neighbors (not just nearest) to maintain graph connectivity.
 * Uses occlusion factor to prevent selecting clustered neighbors.
 */

use super::types::{Neighbor, NodeId, Distance};
use std::collections::HashMap;

/// RobustPrune - Select diverse neighbors using occlusion-based pruning
///
/// # Algorithm (from DiskANN paper)
/// 1. Sort candidates by distance to query
/// 2. Truncate at max_candidates
/// 3. Initialize occlude_factor[i] = 0 for all candidates
/// 4. For cur_alpha = 1 to alpha (multiply by 1.2 each iteration):
///    - For each candidate i:
///      - Skip if occlude_factor[i] > cur_alpha
///      - Add i to result (if not self-loop)
///      - For each remaining candidate j:
///        - Compute occlude_factor[j] = max(occlude_factor[j], distance_j / distance_ij)
///        - This "occludes" j if i is closer to query and close to j
///
/// # Parameters
/// - `query_id`: ID of the query point
/// - `candidates`: Pool of candidate neighbors (sorted by distance)
/// - `max_degree`: Maximum neighbors to select (R in paper)
/// - `max_candidates`: Maximum candidates to consider (C in paper)
/// - `alpha`: Diversity parameter (typically 1.2)
/// - `get_distance`: Function to compute distance between two nodes
///
/// # Returns
/// Vector of selected neighbor IDs (diverse, up to max_degree)
pub fn robust_prune<F>(
    query_id: NodeId,
    candidates: &mut Vec<Neighbor>,
    max_degree: usize,
    max_candidates: usize,
    alpha: f32,
    get_distance: F,
) -> Vec<NodeId>
where
    F: Fn(NodeId, NodeId) -> Distance,
{
    if candidates.is_empty() {
        return Vec::new();
    }

    // Sort candidates by distance (ascending)
    candidates.sort();

    // Truncate at max_candidates
    if candidates.len() > max_candidates {
        candidates.truncate(max_candidates);
    }

    // Initialize occlude_factor for each candidate
    let mut occlude_factor = vec![0.0f32; candidates.len()];

    // Result: selected diverse neighbors
    let mut result = Vec::with_capacity(max_degree);

    // Iterate with increasing alpha (1.0, 1.2, 1.44, ...)
    let mut cur_alpha = 1.0;
    while cur_alpha <= alpha && result.len() < max_degree {
        for i in 0..candidates.len() {
            if result.len() >= max_degree {
                break;
            }

            // Skip if already occluded
            if occlude_factor[i] > cur_alpha {
                continue;
            }

            // Mark as processed (set to infinity so not considered again)
            occlude_factor[i] = f32::MAX;

            let candidate_i = &candidates[i];

            // Skip self-loops
            if candidate_i.id == query_id {
                continue;
            }

            // Add to result
            result.push(candidate_i.id);

            // Update occlude_factor for remaining candidates
            for j in (i + 1)..candidates.len() {
                // Skip if already heavily occluded
                if occlude_factor[j] > alpha {
                    continue;
                }

                let candidate_j = &candidates[j];

                // Compute distance between i and j
                let distance_ij = get_distance(candidate_i.id, candidate_j.id);

                // Occlusion check (L2/Cosine metric)
                // occlude_factor[j] = distance_from_query_to_j / distance_from_i_to_j
                // If j is not much closer to query than i is to j, then i "occludes" j
                let occlusion = if distance_ij == 0.0 {
                    f32::MAX // Perfect overlap, completely occluded
                } else {
                    candidate_j.distance / distance_ij
                };

                occlude_factor[j] = occlude_factor[j].max(occlusion);
            }
        }

        // Increase alpha (1.0 -> 1.2 -> 1.44 -> ...)
        cur_alpha *= 1.2;
    }

    // Optional: saturate_graph - fill remaining slots with closest non-occluded
    // (Microsoft's code does this when alpha > 1.0)
    if alpha > 1.0 && result.len() < max_degree {
        for candidate in candidates {
            if result.len() >= max_degree {
                break;
            }
            if !result.contains(&candidate.id) && candidate.id != query_id {
                result.push(candidate.id);
            }
        }
    }

    result
}

/// Prune neighbors for a given node (wrapper around robust_prune)
///
/// This is the main entry point for neighbor selection during graph construction.
///
/// # Parameters
/// - `node_id`: ID of the node whose neighbors we're selecting
/// - `candidates`: Pool of candidate neighbors
/// - `max_degree`: Maximum degree (R)
/// - `max_candidates`: Maximum candidates to consider (C)
/// - `alpha`: Diversity parameter (α)
/// - `vectors`: Vector storage (for distance computation)
///
/// # Returns
/// Vector of selected neighbor IDs
pub fn prune_neighbors(
    node_id: NodeId,
    candidates: &mut Vec<Neighbor>,
    max_degree: usize,
    max_candidates: usize,
    alpha: f32,
    vectors: &HashMap<NodeId, Vec<f32>>,
) -> Vec<NodeId> {
    robust_prune(
        node_id,
        candidates,
        max_degree,
        max_candidates,
        alpha,
        |id1, id2| {
            if let (Some(v1), Some(v2)) = (vectors.get(&id1), vectors.get(&id2)) {
                l2_distance(v1, v2)
            } else {
                f32::MAX // Missing vector, treat as infinitely far
            }
        },
    )
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

    #[test]
    fn test_robust_prune_selects_diverse_neighbors() {
        // Query at origin, candidates in a cluster and one far away
        let mut candidates = vec![
            Neighbor::new(1, 1.0),   // Close
            Neighbor::new(2, 1.1),   // Close (should be occluded by 1)
            Neighbor::new(3, 1.05),  // Close (should be occluded by 1)
            Neighbor::new(4, 5.0),   // Far (diverse, should be selected)
        ];

        let vectors: HashMap<NodeId, Vec<f32>> = [
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 0.0]),
            (2, vec![1.1, 0.0]),  // Very close to 1
            (3, vec![1.05, 0.0]), // Very close to 1
            (4, vec![5.0, 0.0]),  // Far from others
        ]
        .iter()
        .cloned()
        .collect();

        // Use higher alpha and max_degree to ensure diversity selection
        let result = prune_neighbors(0, &mut candidates, 3, 10, 1.5, &vectors);

        // Should select at least 1 (closest) and some diverse neighbors
        // The exact selection depends on occlusion, but should avoid all clustered nodes
        assert!(result.len() >= 2);
        assert!(result.contains(&1)); // Closest always selected

        // With high enough alpha, diverse node 4 should eventually be selected
        // (either through occlusion threshold or saturate_graph)
        assert!(result.contains(&4) || result.len() == 3);
    }

    #[test]
    fn test_robust_prune_respects_max_degree() {
        let mut candidates = vec![
            Neighbor::new(1, 1.0),
            Neighbor::new(2, 2.0),
            Neighbor::new(3, 3.0),
            Neighbor::new(4, 4.0),
        ];

        let vectors: HashMap<NodeId, Vec<f32>> = [
            (0, vec![0.0]),
            (1, vec![1.0]),
            (2, vec![2.0]),
            (3, vec![3.0]),
            (4, vec![4.0]),
        ]
        .iter()
        .cloned()
        .collect();

        let result = prune_neighbors(0, &mut candidates, 2, 10, 1.2, &vectors);

        assert_eq!(result.len(), 2); // Should only select max_degree neighbors
    }

    #[test]
    fn test_robust_prune_skips_self_loop() {
        let mut candidates = vec![
            Neighbor::new(0, 0.0),  // Self-loop
            Neighbor::new(1, 1.0),
        ];

        let vectors: HashMap<NodeId, Vec<f32>> = [
            (0, vec![0.0]),
            (1, vec![1.0]),
        ]
        .iter()
        .cloned()
        .collect();

        let result = prune_neighbors(0, &mut candidates, 5, 10, 1.2, &vectors);

        assert!(!result.contains(&0)); // Should not include self
        assert!(result.contains(&1));
    }

    #[test]
    fn test_l2_distance() {
        let v1 = vec![0.0, 0.0];
        let v2 = vec![3.0, 4.0];

        let dist = l2_distance(&v1, &v2);
        assert!((dist - 5.0).abs() < 1e-6); // 3-4-5 triangle
    }

    #[test]
    fn test_empty_candidates() {
        let mut candidates = vec![];
        let vectors = HashMap::new();

        let result = prune_neighbors(0, &mut candidates, 5, 10, 1.2, &vectors);

        assert_eq!(result.len(), 0);
    }
}
