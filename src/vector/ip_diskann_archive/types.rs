/*
 * Type definitions for IP-DiskANN
 *
 * Reference: Microsoft DiskANN Rust implementation
 * Location: ~/github/microsoft/DiskANN/rust/diskann/src/model/neighbor.rs
 */

use std::cmp::Ordering;

/// Node identifier (u32 for efficiency, supports up to 4B nodes)
pub type NodeId = u32;

/// Distance type (f32 for performance)
pub type Distance = f32;

/// A neighbor in the graph with its distance
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub id: NodeId,
    pub distance: Distance,
}

impl Neighbor {
    #[inline]
    pub fn new(id: NodeId, distance: Distance) -> Self {
        Self { id, distance }
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Sort by distance (ascending), then by id for stability
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// Configuration parameters for IP-DiskANN
#[derive(Debug, Clone)]
pub struct IPDiskANNConfig {
    /// Maximum degree (R in Vamana paper)
    pub max_degree: usize,

    /// Maximum candidates to consider during pruning (C in Vamana paper)
    pub max_candidates: usize,

    /// Diversity parameter for RobustPrune (α in Vamana paper, typically 1.2)
    pub alpha: f32,

    /// Search list size (L in Vamana paper)
    pub search_list_size: usize,
}

impl Default for IPDiskANNConfig {
    fn default() -> Self {
        Self {
            max_degree: 64,          // R = 64 (good for most datasets)
            max_candidates: 750,     // C = 750 (from DiskANN paper)
            alpha: 1.2,              // α = 1.2 (diversity parameter)
            search_list_size: 100,   // L = 100 (search quality)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbor_ordering() {
        let n1 = Neighbor::new(1, 0.5);
        let n2 = Neighbor::new(2, 0.3);
        let n3 = Neighbor::new(3, 0.5);

        assert!(n2 < n1); // Smaller distance comes first
        assert!(n1 < n3); // Same distance, smaller id comes first
    }

    #[test]
    fn default_config() {
        let config = IPDiskANNConfig::default();
        assert_eq!(config.max_degree, 64);
        assert_eq!(config.alpha, 1.2);
    }
}
