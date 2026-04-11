//! Collection catalog and schema contracts for the next storage rewrite.

pub mod collection;
pub mod schema;

pub use collection::CollectionDefinition;
pub use schema::{
    CollectionName, CollectionSchema, DenseSchema, FrozenDenseIndexKind, GraphSchema,
    GraphTemporalMode, MultiEncoderKind, MultiSchema, MutableDenseIndexKind, QuantizationMode,
    SlotId, SparseIndexKind, SparseSchema, TextSchema,
};
