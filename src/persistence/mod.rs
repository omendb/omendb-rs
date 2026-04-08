//! Persistence contracts for the rewrite.

pub mod checkpoint;
pub mod manifest;
pub mod wal;

pub use checkpoint::{
    CollectionCheckpoint, DenseCheckpoint, MultiCheckpoint, SlotMetaCheckpoint, SparseCheckpoint,
    TextCheckpoint,
};
pub use manifest::{
    CollectionManifest, DerivedManifest, FileManifest, FrozenDenseSection, GenerationManifest,
    SparseSection, TextSection,
};
pub use wal::{DeleteRecordOp, PutDenseOp, PutMetaOp, PutMultiOp, PutSparseOp, PutTextOp, WalOp};
