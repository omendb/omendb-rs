//! Dense index trait boundaries for the rewrite.

use crate::catalog::SlotId;
use anyhow::Result;

#[derive(Debug, Clone, Copy, Default)]
pub struct DenseSearchParams {
    pub ef: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrozenSearchResult {
    pub slot: SlotId,
    pub distance: f32,
}

pub trait MutableDenseIndex: Send + Sync {
    fn insert(&mut self, slot: SlotId, vector: &[f32]) -> Result<()>;
    fn search(
        &self,
        query: &[f32],
        limit: usize,
        params: DenseSearchParams,
    ) -> Result<Vec<FrozenSearchResult>>;
}

pub trait FrozenDenseIndex: Send + Sync {
    fn search(
        &self,
        query: &[f32],
        limit: usize,
        params: DenseSearchParams,
    ) -> Result<Vec<FrozenSearchResult>>;
}
