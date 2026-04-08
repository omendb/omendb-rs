use super::*;

mod crud;
mod edges;
mod golden_data;
mod hybrid;
mod metadata;
mod multivec;
mod persistence;
mod proptest_tests;
mod quantization;
mod search;
mod sparse;

pub(super) fn random_vector(dim: usize, seed: usize) -> Vector {
    let data: Vec<f32> = (0..dim).map(|i| ((seed + i) as f32) * 0.1).collect();
    Vector::new(data)
}
mod explain;
