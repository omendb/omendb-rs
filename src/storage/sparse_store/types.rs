//! Sparse storage types.

use crate::catalog::SlotId;
use crate::vector::sparse::SparseVector;
use anyhow::Result;
use parking_lot::RwLock;

#[derive(Debug, Default)]
pub struct SparseStore {
    pub(crate) values: RwLock<Vec<Option<SparseVector>>>,
}

impl SparseStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&self, slot: SlotId, vector: SparseVector) -> Result<()> {
        let mut slots = self.values.write();
        let index = usize::try_from(slot).expect("slot fits usize");
        if slots.len() <= index {
            slots.resize(index + 1, None);
        }
        slots[index] = Some(vector);
        Ok(())
    }

    pub fn clear(&self, slot: SlotId) -> Result<()> {
        let mut slots = self.values.write();
        let index = usize::try_from(slot).expect("slot fits usize");
        if let Some(entry) = slots.get_mut(index) {
            *entry = None;
        }
        Ok(())
    }

    #[must_use]
    pub fn get_cloned(&self, slot: SlotId) -> Option<SparseVector> {
        let slots = self.values.read();
        slots
            .get(usize::try_from(slot).expect("slot fits usize"))
            .and_then(|entry| entry.clone())
    }
}
