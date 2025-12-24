// Core data structures for custom HNSW implementation
//
// Design goals:
// - Cache-line aligned hot data (64 bytes)
// - Index-based (u32 node IDs, not pointers)
// - Separate hot/cold data for better cache utilization

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

/// HNSW construction parameters
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct HNSWParams {
    /// Number of bidirectional links per node (M)
    ///
    /// Higher M = better recall, more memory, slower construction
    /// Typical range: 16-64
    pub m: usize,

    /// Size of dynamic candidate list during construction (`ef_construction`)
    ///
    /// Higher ef = better recall, slower construction
    /// Must be >= M
    pub ef_construction: usize,

    /// Normalization factor for level assignment (ml = 1/ln(M))
    ///
    /// Determines probability distribution for level selection
    pub ml: f32,

    /// Random seed for reproducible level assignment
    pub seed: u64,

    /// Maximum allowed level (typically 6-8 for millions of vectors)
    pub max_level: u8,
}

impl Default for HNSWParams {
    fn default() -> Self {
        let m = 16; // Industry standard (ChromaDB, hnswlib, Milvus, pgvector)
        Self {
            m,
            ef_construction: 100, // Good balance of build speed and recall
            ml: 1.0 / (m as f32).ln(),
            seed: 42,
            max_level: 8, // Support up to ~100M vectors
        }
    }
}

impl HNSWParams {
    /// Create parameters optimized for recall
    #[must_use]
    pub fn high_recall() -> Self {
        let m = 64;
        Self {
            m,
            ef_construction: 400,
            ml: 1.0 / (m as f32).ln(),
            seed: 42,
            max_level: 8,
        }
    }

    /// Create parameters optimized for memory
    #[must_use]
    pub fn low_memory() -> Self {
        let m = 16;
        Self {
            m,
            ef_construction: 100,
            ml: 1.0 / (m as f32).ln(),
            seed: 42,
            max_level: 6,
        }
    }

    /// Builder: set M parameter
    #[must_use]
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.ml = 1.0 / (m as f32).ln();
        self
    }

    /// Builder: set `ef_construction` parameter
    #[must_use]
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Builder: set `ef_search` parameter (stored in construction for defaults)
    /// Note: This is primarily used for passing the search ef to the index.
    #[must_use]
    pub fn with_ef_search(self, _ef: usize) -> Self {
        // ef_search is not stored in HNSWParams, it's a runtime parameter
        // This method exists for API consistency
        self
    }

    /// Validate parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.m == 0 {
            return Err("M must be greater than 0".to_string());
        }
        if self.ef_construction < self.m {
            return Err("ef_construction must be >= M".to_string());
        }
        if self.ml <= 0.0 {
            return Err("ml must be greater than 0".to_string());
        }
        if self.max_level == 0 {
            return Err("max_level must be greater than 0".to_string());
        }
        Ok(())
    }
}

/// HNSW node with cache-optimized layout
///
/// Hot data (first 64 bytes = 1 cache line):
/// - Node ID (4 bytes)
/// - Level (1 byte)
/// - Neighbor counts per level (8 bytes for 8 levels max)
/// - Padding to 64 bytes
///
/// Cold data stored separately:
/// - Neighbors (only fetched when traversing)
/// - Vector data (only fetched when computing distances)
#[repr(C, align(64))] // Cache-line aligned
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HNSWNode {
    /// Node ID (u32 = 4 bytes, supports 4 billion vectors)
    pub id: u32,

    /// Current level (0 to `max_level`)
    pub level: u8,

    /// Neighbor counts per level (u8 = 1 byte per level, max 8 levels)
    ///
    /// `neighbor_counts`[i] = number of neighbors at level i
    pub neighbor_counts: [u8; 8],

    /// Reserved for future use (extensions, flags, etc.)
    #[serde(skip, default = "default_reserved")]
    _reserved: [u8; 3],

    /// Padding to complete 64-byte cache line (64 - 4 - 1 - 8 - 3 = 48)
    #[serde(skip, default = "default_padding")]
    _padding: [u8; 48],
}

// Default functions for serde skipped fields
fn default_reserved() -> [u8; 3] {
    [0; 3]
}

fn default_padding() -> [u8; 48] {
    [0; 48]
}

impl HNSWNode {
    /// Create a new node
    #[must_use]
    pub fn new(id: u32, level: u8) -> Self {
        Self {
            id,
            level,
            neighbor_counts: [0; 8],
            _reserved: [0; 3],
            _padding: [0; 48],
        }
    }

    /// Get number of neighbors at a given level
    #[must_use]
    pub fn neighbor_count(&self, level: u8) -> usize {
        if level <= self.level {
            self.neighbor_counts[level as usize] as usize
        } else {
            0
        }
    }

    /// Set number of neighbors at a given level
    ///
    /// Note: Allows setting neighbor counts at ANY level, not just levels <= node.level.
    /// This is necessary because bidirectional edges can create connections at layers
    /// higher than the node's originally assigned level.
    pub fn set_neighbor_count(&mut self, level: u8, count: usize) {
        if (level as usize) < self.neighbor_counts.len() {
            self.neighbor_counts[level as usize] = count.min(255) as u8;
        }
    }
}

impl Default for HNSWNode {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

// Compile-time assertion that HNSWNode is exactly 64 bytes
const _: () = assert!(std::mem::size_of::<HNSWNode>() == 64);

/// Distance function for vector similarity
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceFunction {
    /// L2 / Euclidean distance
    L2,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
    /// Negative inner product (for maximum inner product search)
    NegativeDotProduct,
}

impl DistanceFunction {
    /// Compute distance between two vectors
    ///
    /// Uses SIMD-accelerated implementations when `simd` feature is enabled.
    /// Falls back to optimized scalar implementations otherwise.
    #[must_use]
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::L2 => l2_distance(a, b),
            Self::Cosine => cosine_distance(a, b),
            Self::NegativeDotProduct => -dot_product(a, b),
        }
    }

    /// Compute distance for internal comparisons (optimized, may not be actual distance)
    ///
    /// For L2: returns squared distance (skips sqrt) since relative ordering is preserved
    /// For Cosine/NegativeDotProduct: same as `distance()`
    ///
    /// This is ~10-15% faster for L2 in HNSW `search_layer`.
    #[inline(always)]
    #[must_use]
    pub fn distance_for_comparison(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::L2 => l2_distance_squared(a, b),
            Self::Cosine => cosine_distance(a, b),
            Self::NegativeDotProduct => -dot_product(a, b),
        }
    }

    /// Convert comparison distance back to actual distance
    ///
    /// For L2: applies sqrt
    /// For others: identity
    #[inline]
    #[must_use]
    pub fn comparison_to_actual(&self, d: f32) -> f32 {
        match self {
            Self::L2 => d.sqrt(),
            _ => d,
        }
    }
}

// Re-export SIMD distance functions (single source of truth)
pub use crate::distance::{cosine_distance, dot_product, l2_distance, l2_distance_squared};

// Distance trait for monomorphization (~10-15% faster search via static dispatch)
// Each implementation computes distance differently but all enable compile-time specialization.

/// Trait for distance computation with compile-time dispatch
pub trait Distance: Copy + Clone + Send + Sync + 'static {
    /// Compute distance between two vectors (for comparisons, may skip sqrt for L2)
    fn distance(a: &[f32], b: &[f32]) -> f32;

    /// Convert comparison distance to actual distance (applies sqrt for L2)
    fn to_actual(d: f32) -> f32;

    /// Get the enum variant for runtime dispatch when needed
    fn as_enum() -> DistanceFunction;
}

/// L2 (Euclidean) distance
#[derive(Copy, Clone, Debug)]
pub struct L2;

impl Distance for L2 {
    #[inline(always)]
    fn distance(a: &[f32], b: &[f32]) -> f32 {
        l2_distance_squared(a, b)
    }

    #[inline(always)]
    fn to_actual(d: f32) -> f32 {
        d.sqrt()
    }

    #[inline(always)]
    fn as_enum() -> DistanceFunction {
        DistanceFunction::L2
    }
}

/// Cosine distance (1 - cosine similarity)
#[derive(Copy, Clone, Debug)]
pub struct Cosine;

impl Distance for Cosine {
    #[inline(always)]
    fn distance(a: &[f32], b: &[f32]) -> f32 {
        cosine_distance(a, b)
    }

    #[inline(always)]
    fn to_actual(d: f32) -> f32 {
        d
    }

    #[inline(always)]
    fn as_enum() -> DistanceFunction {
        DistanceFunction::Cosine
    }
}

/// Negative dot product (for maximum inner product search)
#[derive(Copy, Clone, Debug)]
pub struct NegDot;

impl Distance for NegDot {
    #[inline(always)]
    fn distance(a: &[f32], b: &[f32]) -> f32 {
        -dot_product(a, b)
    }

    #[inline(always)]
    fn to_actual(d: f32) -> f32 {
        d
    }

    #[inline(always)]
    fn as_enum() -> DistanceFunction {
        DistanceFunction::NegativeDotProduct
    }
}

/// Candidate during search (node ID + distance)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Candidate {
    /// Distance to query (`OrderedFloat` for Ord)
    pub distance: OrderedFloat<f32>,

    /// Node ID
    pub node_id: u32,
}

impl Candidate {
    #[must_use]
    pub fn new(node_id: u32, distance: f32) -> Self {
        Self {
            distance: OrderedFloat(distance),
            node_id,
        }
    }
}

/// Search result (node ID + distance)
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Node ID
    pub id: u32,

    /// Distance to query
    pub distance: f32,
}

impl SearchResult {
    #[must_use]
    pub fn new(id: u32, distance: f32) -> Self {
        Self { id, distance }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_node_size() {
        // Verify cache-line alignment
        assert_eq!(std::mem::size_of::<HNSWNode>(), 64);
        assert_eq!(std::mem::align_of::<HNSWNode>(), 64);
    }

    #[test]
    fn test_hnsw_node_creation() {
        let node = HNSWNode::new(42, 3);
        assert_eq!(node.id, 42);
        assert_eq!(node.level, 3);
        assert_eq!(node.neighbor_count(0), 0);
        assert_eq!(node.neighbor_count(3), 0);
    }

    #[test]
    fn test_hnsw_node_neighbor_counts() {
        let mut node = HNSWNode::new(0, 5);

        node.set_neighbor_count(0, 48);
        node.set_neighbor_count(2, 24);
        node.set_neighbor_count(5, 12);

        assert_eq!(node.neighbor_count(0), 48);
        assert_eq!(node.neighbor_count(1), 0);
        assert_eq!(node.neighbor_count(2), 24);
        assert_eq!(node.neighbor_count(5), 12);

        // Level 6 > node.level (5), should return 0
        assert_eq!(node.neighbor_count(6), 0);
    }

    #[test]
    fn test_params_validation() {
        let params = HNSWParams::default();
        assert!(params.validate().is_ok());

        let mut invalid_params = HNSWParams::default();
        invalid_params.m = 0;
        assert!(invalid_params.validate().is_err());

        invalid_params = HNSWParams::default();
        invalid_params.ef_construction = 10; // < M (16)
        assert!(invalid_params.validate().is_err());
    }

    #[test]
    fn test_l2_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let dist = l2_distance(&a, &b);
        let expected = (3.0_f32.powi(2) * 3.0).sqrt(); // sqrt(9 + 9 + 9) = sqrt(27)

        assert!((dist - expected).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];

        let dist = cosine_distance(&a, &b);
        assert!((dist - 0.0).abs() < 1e-6); // Identical vectors, distance = 0

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];

        let dist = cosine_distance(&c, &d);
        assert!((dist - 1.0).abs() < 1e-6); // Orthogonal vectors, distance = 1
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let dot = dot_product(&a, &b);
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn test_candidate_ordering() {
        let c1 = Candidate::new(1, 0.5);
        let c2 = Candidate::new(2, 0.3);
        let c3 = Candidate::new(3, 0.7);

        // Candidates are ordered by distance (lower = better)
        assert!(c2 < c1);
        assert!(c1 < c3);
    }
}
