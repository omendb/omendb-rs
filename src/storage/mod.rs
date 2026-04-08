//! Modality-aware storage contracts for the rewrite.

pub mod dense_store;
pub mod multi_store;
pub mod record_store;
pub mod sparse_store;
pub mod text_store;

pub use dense_store::{DenseStore, DenseValue};
pub use multi_store::{MultiRange, MultiStore};
pub use record_store::{DirtySets, RecordFlags, RecordMeta, RecordStore};
pub use sparse_store::SparseStore;
pub use text_store::TextStore;
