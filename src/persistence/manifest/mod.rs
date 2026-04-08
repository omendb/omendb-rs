//! Manifest contracts.

mod schema;

pub use schema::{
    CollectionManifest, DerivedManifest, FileManifest, FrozenDenseSection, GenerationManifest,
    SparseSection, TextSection,
};
