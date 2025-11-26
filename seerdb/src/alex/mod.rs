//! ALEX: Adaptive Learned Index
//!
//! A production-ready implementation of the ALEX learned index structure
//! from Microsoft Research.
//!
//! ALEX (Adaptive Learned indEX) combines machine learning models with
//! traditional indexing techniques to provide fast lookups, inserts, updates,
//! and deletes. Unlike static learned indexes (RMI), ALEX supports dynamic
//! workloads through:
//!
//! 1. **Gapped Arrays**: Leaf nodes maintain spare capacity for future inserts
//! 2. **Local Node Splitting**: When nodes fill, split locally instead of rebuilding entire tree
//! 3. **Adaptive Models**: Models adapt to data distribution changes
//!
//! ## Performance Characteristics
//!
//! - **Lookups**: O(log n) with learned index acceleration → practical O(1) to O(log log n)
//! - **Inserts**: O(log n) amortized (O(1) into gapped array + occasional splits)
//! - **Deletes**: O(log n) with lazy deletion
//! - **Space**: O(n) with tunable expansion factor (1.0 = 50% overhead by default)
//!
//! ## References
//!
//! - ALEX Paper: <https://www.microsoft.com/en-us/research/uploads/prod/2020/04/MSRAlexTechnicalReportV2.pdf>
//! - GitHub: <https://github.com/microsoft/ALEX>
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use seerdb::alex::AlexTree;
//!
//! let mut index = AlexTree::new();
//!
//! // Insert key-value pairs
//! index.insert(10, vec![1, 2, 3])?;
//! index.insert(20, vec![4, 5, 6])?;
//! index.insert(15, vec![7, 8, 9])?;  // Out of order - ALEX handles it
//!
//! // Lookup
//! if let Some(value) = index.get(15)? {
//!     println!("Found: {:?}", value);
//! }
//!
//! // Range query
//! for (key, value) in index.range(10..=20)? {
//!     println!("{}: {:?}", key, value);
//! }
//! ```

pub mod alex_tree;
pub mod gapped_node;
pub mod linear_model;
pub mod multi_level;
#[cfg(feature = "simd")]
mod simd_search;
// Fallback for non-SIMD builds
#[cfg(not(feature = "simd"))]
mod simd_search {
    /// Scalar fallback for simd_search_i64
    pub fn simd_search_i64(keys: &[Option<i64>], key: i64) -> Option<usize> {
        keys.iter().position(|&k| k == Some(key))
    }
}

// Re-exports
pub use alex_tree::AlexTree;
pub use gapped_node::GappedNode;
pub use linear_model::LinearModel;
pub use multi_level::MultiLevelAlexTree;
