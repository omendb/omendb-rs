/*
 * IP-DiskANN Implementation
 *
 * Based on:
 * - IP-DiskANN paper (arXiv 2502.13826, February 2025)
 * - Microsoft DiskANN Rust reference (MIT licensed)
 * - Our Vamana algorithm study (ai/research/vamana_algorithm_notes.md)
 *
 * Key differences from standard DiskANN:
 * - Bi-directional edges (in_neighbors + out_neighbors)
 * - In-place insertion (no batch consolidation)
 * - In-place deletion (efficient via in-neighbors)
 */

pub mod types;
pub mod graph;
pub mod prune;
pub mod search;
pub mod index;

#[cfg(test)]
mod integration_tests;

#[cfg(test)]
mod debug_test;

pub use types::{Neighbor, NodeId, IPDiskANNConfig};
pub use graph::{GraphNode, BiDirectionalGraph};
pub use prune::{robust_prune, prune_neighbors};
pub use search::greedy_search;
pub use index::IPDiskANNIndex;
