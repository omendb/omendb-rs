//! Sparse index contracts.

use crate::catalog::SlotId;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseSearchResult {
    pub slot: SlotId,
    pub score: f32,
}
