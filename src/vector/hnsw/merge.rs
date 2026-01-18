// HNSW Graph Merging - IGTM Algorithm
//
// Implements Iterative Greedy Tree Merging from:
// - Elasticsearch Labs: "Speeding up merging of HNSW graphs" (2024-2025)
// - arXiv:2505.16064 (May 2025) - MERGE-HNSW algorithms
//
// Expected speedup: 1.28-1.72x for batch inserts (validated in Lucene 10.2)

use super::error::{HNSWError, Result};
use super::index::HNSWIndex;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tracing::{debug, info, instrument, warn};

/// Configuration for graph merging
#[derive(Clone, Debug)]
pub struct MergeConfig {
    /// Minimum number of neighbors each vertex must have in the join set
    /// Default: 2 (from IGTM paper)
    pub min_coverage: usize,

    /// ef parameter for fast search during merge (lower than `ef_construction` for speed)
    /// Default: `ef_construction` / 2
    pub fast_ef: Option<usize>,

    /// Whether to use parallel join set computation
    pub parallel_join_set: bool,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            min_coverage: 2,
            fast_ef: None,
            parallel_join_set: true,
        }
    }
}

/// Statistics from a merge operation
#[derive(Clone, Debug)]
pub struct MergeStats {
    /// Total vectors merged from small graph
    pub vectors_merged: usize,

    /// Size of the join set (strategic vertices inserted first)
    pub join_set_size: usize,

    /// Time spent computing join set
    pub join_set_duration: Duration,

    /// Time spent inserting join set
    pub join_set_insert_duration: Duration,

    /// Time spent inserting remaining vectors
    pub remaining_insert_duration: Duration,

    /// Total merge duration
    pub total_duration: Duration,

    /// Vectors inserted using fast path (entry points from join set)
    pub fast_path_inserts: usize,

    /// Vectors inserted using fallback (standard insert)
    pub fallback_inserts: usize,
}

impl MergeStats {
    /// Calculate speedup vs naive approach (estimated)
    #[must_use]
    pub fn estimated_speedup(&self) -> f64 {
        // Naive: all vectors go through full search
        // IGTM: join_set gets full search, remaining get fast search
        // Fast search is ~5x faster per vector

        // Assuming fast path is 5x faster
        // Speedup = 1 / (join_set_ratio + remaining_ratio * 0.2)
        let join_set_ratio = self.join_set_size as f64 / self.vectors_merged.max(1) as f64;
        let remaining_ratio = 1.0 - join_set_ratio;

        1.0 / (join_set_ratio + remaining_ratio * 0.2)
    }
}

/// HNSW Graph Merger using IGTM algorithm
///
/// Merges a small graph into a large graph using strategic vertex selection.
/// Expected 1.3-1.7x speedup over naive insertion.
pub struct GraphMerger {
    config: MergeConfig,
}

impl GraphMerger {
    /// Create a new graph merger with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: MergeConfig::default(),
        }
    }

    /// Create a new graph merger with custom configuration
    #[must_use]
    pub fn with_config(config: MergeConfig) -> Self {
        Self { config }
    }

    /// Merge a small graph into a large graph using IGTM algorithm
    ///
    /// # Algorithm
    /// 1. Compute join set: Find minimal vertex subset that covers all vertices
    ///    (every vertex has ≥`min_coverage` neighbors in the join set)
    /// 2. Insert join set into large graph using standard insertion
    /// 3. For remaining vertices, use join set neighbors as entry points for fast insertion
    ///
    /// # Arguments
    /// * `large` - Target graph (will be modified)
    /// * `small` - Source graph (vectors will be moved)
    ///
    /// # Returns
    /// Merge statistics including timing breakdown
    #[instrument(skip(self, large, small), fields(large_size = large.len(), small_size = small.len()))]
    pub fn merge_graphs(&self, large: &mut HNSWIndex, small: &HNSWIndex) -> Result<MergeStats> {
        let total_start = Instant::now();
        let small_size = small.len();

        if small_size == 0 {
            return Ok(MergeStats {
                vectors_merged: 0,
                join_set_size: 0,
                join_set_duration: Duration::ZERO,
                join_set_insert_duration: Duration::ZERO,
                remaining_insert_duration: Duration::ZERO,
                total_duration: total_start.elapsed(),
                fast_path_inserts: 0,
                fallback_inserts: 0,
            });
        }

        info!(
            large_size = large.len(),
            small_size = small_size,
            "Starting IGTM graph merge"
        );

        // Phase 1: Compute join set
        let join_set_start = Instant::now();
        let join_set = self.compute_join_set(small);
        let join_set_duration = join_set_start.elapsed();

        debug!(
            join_set_size = join_set.len(),
            coverage_target = self.config.min_coverage,
            duration_ms = join_set_duration.as_millis(),
            "Join set computed"
        );

        // Phase 2: Insert join set vectors and build ID mapping
        let join_insert_start = Instant::now();
        let mut small_to_large: HashMap<u32, u32> = HashMap::with_capacity(join_set.len());

        for &small_id in &join_set {
            let vector = small
                .get_vector_dequantized(small_id)
                .ok_or(HNSWError::VectorNotFound(small_id))?;
            let large_id = large.insert(&vector)?;
            small_to_large.insert(small_id, large_id);
        }
        let join_set_insert_duration = join_insert_start.elapsed();

        debug!(
            inserted = join_set.len(),
            duration_ms = join_set_insert_duration.as_millis(),
            "Join set inserted"
        );

        // Phase 3: Insert remaining vectors using fast path
        let remaining_start = Instant::now();
        let mut fast_path_inserts = 0;
        let mut fallback_inserts = 0;

        let fast_ef = self
            .config
            .fast_ef
            .unwrap_or(large.params().ef_construction / 2);

        for node_id in 0..small.len() as u32 {
            if join_set.contains(&node_id) {
                continue;
            }

            let vector = small
                .get_vector_dequantized(node_id)
                .ok_or(HNSWError::VectorNotFound(node_id))?;

            // Find neighbors of this node that are in the join set
            // Map small graph IDs to large graph IDs
            let small_neighbors = small.get_neighbors_level0(node_id);
            let entry_points: Vec<u32> = small_neighbors
                .iter()
                .filter_map(|&small_neighbor_id| small_to_large.get(&small_neighbor_id).copied())
                .collect();

            if entry_points.is_empty() {
                // Fallback: no join set neighbors, use standard insert
                large.insert(&vector)?;
                fallback_inserts += 1;
            } else {
                // Fast path: use mapped join set neighbors as entry points in large graph
                large.insert_with_hints(&vector, &entry_points, fast_ef)?;
                fast_path_inserts += 1;
            }
        }
        let remaining_insert_duration = remaining_start.elapsed();

        let total_duration = total_start.elapsed();

        let stats = MergeStats {
            vectors_merged: small_size,
            join_set_size: join_set.len(),
            join_set_duration,
            join_set_insert_duration,
            remaining_insert_duration,
            total_duration,
            fast_path_inserts,
            fallback_inserts,
        };

        info!(
            vectors_merged = stats.vectors_merged,
            join_set_size = stats.join_set_size,
            fast_path_ratio = format!(
                "{:.1}%",
                (stats.fast_path_inserts as f64 / stats.vectors_merged.max(1) as f64) * 100.0
            ),
            total_ms = stats.total_duration.as_millis(),
            estimated_speedup = format!("{:.2}x", stats.estimated_speedup()),
            "IGTM merge complete"
        );

        Ok(stats)
    }

    /// Compute join set using greedy covering algorithm
    ///
    /// Finds minimal subset J such that every vertex v has ≥`min_coverage` neighbors in J.
    /// Uses greedy selection: pick vertex maximizing coverage gain at each step.
    fn compute_join_set(&self, graph: &HNSWIndex) -> HashSet<u32> {
        let mut join_set = HashSet::new();
        let mut coverage: HashMap<u32, usize> = HashMap::new();

        let num_vectors = graph.len();
        if num_vectors == 0 {
            return join_set;
        }

        // Greedy selection until all vertices are covered
        while !self.is_fully_covered(&coverage, graph) {
            // Find vertex with maximum gain
            let best = (0..num_vectors as u32)
                .filter(|id| !join_set.contains(id))
                .max_by_key(|&id| {
                    self.calculate_gain(id, &join_set, &coverage, graph)
                        .unwrap_or(0)
                });

            if let Some(best_id) = best {
                join_set.insert(best_id);

                // Update coverage: all neighbors of best_id gain a neighbor in J
                let neighbors = graph.get_neighbors_level0(best_id);
                for &neighbor in &neighbors {
                    *coverage.entry(neighbor).or_insert(0) += 1;
                }

                // Also update coverage for best_id itself (it's now covered)
                *coverage.entry(best_id).or_insert(0) += self.config.min_coverage;
            } else {
                // No more vertices to add, but not fully covered
                // This can happen with disconnected components
                warn!("Join set computation terminated early - graph may have disconnected components");
                break;
            }
        }

        join_set
    }

    /// Calculate gain for adding vertex to join set
    ///
    /// Gain = number of vertices that would increase their coverage
    #[allow(clippy::unnecessary_wraps)]
    fn calculate_gain(
        &self,
        vertex_id: u32,
        join_set: &HashSet<u32>,
        coverage: &HashMap<u32, usize>,
        graph: &HNSWIndex,
    ) -> Result<usize> {
        // Skip if already in join set
        if join_set.contains(&vertex_id) {
            return Ok(0);
        }

        let neighbors = graph.get_neighbors_level0(vertex_id);
        let mut gain = 0;

        // Gain for self (if not yet covered)
        let self_coverage = coverage.get(&vertex_id).copied().unwrap_or(0);
        if self_coverage < self.config.min_coverage {
            gain += 1;
        }

        // Gain for each neighbor that would benefit
        for &neighbor in &neighbors {
            let neighbor_coverage = coverage.get(&neighbor).copied().unwrap_or(0);
            if neighbor_coverage < self.config.min_coverage {
                gain += 1;
            }
        }

        Ok(gain)
    }

    /// Check if all vertices have sufficient coverage
    fn is_fully_covered(&self, coverage: &HashMap<u32, usize>, graph: &HNSWIndex) -> bool {
        for node_id in 0..graph.len() as u32 {
            let c = coverage.get(&node_id).copied().unwrap_or(0);
            if c < self.config.min_coverage {
                return false;
            }
        }
        true
    }
}

impl Default for GraphMerger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::hnsw::{DistanceFunction, HNSWParams};

    fn create_test_index(num_vectors: usize, dim: usize) -> HNSWIndex {
        let params = HNSWParams {
            m: 16,
            ef_construction: 100,
            ..Default::default()
        };
        let mut index = HNSWIndex::new(dim, params, DistanceFunction::L2, false).unwrap();

        for i in 0..num_vectors {
            let vector: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 / 100.0).collect();
            index.insert(&vector).unwrap();
        }

        index
    }

    #[test]
    fn test_merge_empty_small_graph() {
        let mut large = create_test_index(100, 8);
        let small = HNSWIndex::new(8, HNSWParams::default(), DistanceFunction::L2, false).unwrap();

        let merger = GraphMerger::new();
        let stats = merger.merge_graphs(&mut large, &small).unwrap();

        assert_eq!(stats.vectors_merged, 0);
        assert_eq!(stats.join_set_size, 0);
        assert_eq!(large.len(), 100);
    }

    #[test]
    fn test_merge_small_graphs() {
        let mut large = create_test_index(100, 8);
        let small = create_test_index(50, 8);

        let initial_size = large.len();
        let merger = GraphMerger::new();
        let stats = merger.merge_graphs(&mut large, &small).unwrap();

        assert_eq!(stats.vectors_merged, 50);
        assert_eq!(large.len(), initial_size + 50);
        assert!(stats.join_set_size > 0);
        assert!(stats.join_set_size <= 50);
    }

    #[test]
    fn test_join_set_coverage() {
        let small = create_test_index(100, 8);
        let merger = GraphMerger::new();

        let join_set = merger.compute_join_set(&small);

        // Join set should be non-empty
        assert!(!join_set.is_empty());

        // Join set should be smaller than total (typically 10-30%)
        assert!(join_set.len() < small.len());

        // All vertices should have sufficient coverage
        let mut coverage: HashMap<u32, usize> = HashMap::new();
        for &j_id in &join_set {
            let neighbors = small.get_neighbors_level0(j_id);
            for &n in &neighbors {
                *coverage.entry(n).or_insert(0) += 1;
            }
            *coverage.entry(j_id).or_insert(0) += merger.config.min_coverage;
        }

        for node_id in 0..small.len() as u32 {
            let c = coverage.get(&node_id).copied().unwrap_or(0);
            assert!(
                c >= merger.config.min_coverage,
                "Node {} has insufficient coverage: {} < {}",
                node_id,
                c,
                merger.config.min_coverage
            );
        }
    }

    #[test]
    fn test_merge_preserves_searchability() {
        let mut large = create_test_index(100, 8);
        let small = create_test_index(50, 8);

        // Remember a vector from small graph
        let test_vector = small.get_vector_dequantized(25).unwrap();

        let merger = GraphMerger::new();
        merger.merge_graphs(&mut large, &small).unwrap();

        // Should be able to find similar vectors after merge
        let results = large.search(&test_vector, 5, 50).unwrap();
        assert!(!results.is_empty());

        // At least one result should be close
        assert!(results[0].distance < 1.0);
    }
}
