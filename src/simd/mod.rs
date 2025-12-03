//! SIMD-accelerated distance calculations.

pub mod distance;

pub use distance::{cosine_distance, dot_product, l2_distance, l2_distance_squared};
