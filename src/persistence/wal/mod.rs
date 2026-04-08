//! WAL record contracts.

mod ops;

pub use ops::{DeleteRecordOp, PutDenseOp, PutMetaOp, PutMultiOp, PutSparseOp, PutTextOp, WalOp};
