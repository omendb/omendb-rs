//! Multi-vector storage types.

use crate::catalog::SlotId;
use anyhow::Result;
use parking_lot::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiRange {
    pub start: u32,
    pub len: u32,
}

#[derive(Debug, Default)]
pub struct MultiStore {
    #[allow(dead_code)]
    pub(crate) values: RwLock<Vec<f32>>,
    pub(crate) ranges: RwLock<Vec<Option<MultiRange>>>,
    pub(crate) token_dim: RwLock<Option<u32>>,
}

impl MultiStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_token_dim(&self, token_dim: u32) {
        *self.token_dim.write() = Some(token_dim);
    }

    pub fn clear(&self, slot: SlotId) -> Result<()> {
        let mut ranges = self.ranges.write();
        let index = usize::try_from(slot).expect("slot fits usize");
        if let Some(entry) = ranges.get_mut(index) {
            *entry = None;
        }
        Ok(())
    }
}
