use super::*;
use crate::catalog::{
    CollectionSchema, DenseSchema, FrozenDenseIndexKind, MutableDenseIndexKind, QuantizationMode,
};

mod crud;
mod edges;
mod golden_data;
mod hybrid;
mod metadata;
mod multivec;
mod persistence;
mod proptest_tests;
mod quantization;
mod schema;
mod search;
mod sparse;

pub(super) fn random_vector(dim: usize, seed: usize) -> Vector {
    let data: Vec<f32> = (0..dim).map(|i| ((seed + i) as f32) * 0.1).collect();
    Vector::new(data)
}

pub(super) fn dense_schema(dim: u32) -> CollectionSchema {
    CollectionSchema {
        name: String::new(),
        metric: Metric::L2,
        dense: Some(DenseSchema {
            dim,
            quantization: QuantizationMode::None,
            mutable_index: MutableDenseIndexKind::Hnsw,
            frozen_index: FrozenDenseIndexKind::Hnsw,
        }),
        sparse: None,
        multi: None,
        text: None,
    }
}
mod explain;
