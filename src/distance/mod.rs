//! SIMD-accelerated distance calculations.

mod ops;

pub use ops::{
    cosine_distance, dot_product, l2_distance, l2_distance_squared, l2_squared_decomposed,
    norm_squared,
};
