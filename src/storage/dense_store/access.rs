//! Dense storage access contracts.

use crate::catalog::SlotId;
use crate::storage::dense_store::{DenseStore, DenseValue};
use anyhow::{Result, anyhow, ensure};
use memmap2::Mmap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

impl DenseStore {
    #[must_use]
    pub fn new(dim: u32) -> Self {
        Self {
            values: parking_lot::RwLock::new(Vec::new()),
            mmaps: parking_lot::RwLock::new(Vec::new()),
            dim: std::sync::atomic::AtomicU32::new(dim),
        }
    }

    #[must_use]
    pub fn dim(&self) -> u32 {
        self.dim.load(Ordering::Relaxed)
    }

    pub fn add_mmap(&self, mmap: Arc<Mmap>) -> u32 {
        let mut mmaps = self.mmaps.write();
        let id = u32::try_from(mmaps.len()).expect("mmap registry exceeds u32::MAX");
        mmaps.push(mmap);
        id
    }

    pub fn set_owned(&self, slot: SlotId, values: Vec<f32>) -> Result<()> {
        ensure!(
            u32::try_from(values.len()).map_err(|_| anyhow!("dense vector too large"))? == self.dim(),
            "dense vector dimension mismatch"
        );
        let mut slots = self.values.write();
        let index = usize::try_from(slot).expect("slot fits usize");
        if slots.len() <= index {
            slots.resize(index + 1, None);
        }
        slots[index] = Some(DenseValue::Owned(values));
        Ok(())
    }

    pub fn set_mmap_ref(&self, slot: SlotId, mmap_id: u32, offset_bytes: u64, len: u32) -> Result<()> {
        ensure!(len == self.dim(), "dense mmap dimension mismatch");
        let mut slots = self.values.write();
        let index = usize::try_from(slot).expect("slot fits usize");
        if slots.len() <= index {
            slots.resize(index + 1, None);
        }
        slots[index] = Some(DenseValue::MmapRef {
            mmap_id,
            offset_bytes,
            len,
        });
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

    pub fn with_slice<T>(&self, slot: SlotId, f: impl FnOnce(Option<&[f32]>) -> T) -> T {
        let slots = self.values.read();
        let index = usize::try_from(slot).expect("slot fits usize");
        let Some(value) = slots.get(index).and_then(Option::as_ref) else {
            return f(None);
        };

        match value {
            DenseValue::Owned(values) => f(Some(values.as_slice())),
            DenseValue::MmapRef {
                mmap_id,
                offset_bytes,
                len,
            } => {
                let mmaps = self.mmaps.read();
                let Some(mmap) = mmaps.get(usize::try_from(*mmap_id).expect("mmap id fits usize")) else {
                    return f(None);
                };
                let offset = usize::try_from(*offset_bytes).expect("offset fits usize");
                let slice = unsafe {
                    std::slice::from_raw_parts(mmap.as_ptr().add(offset).cast::<f32>(), usize::try_from(*len).expect("len fits usize"))
                };
                f(Some(slice))
            }
        }
    }
}
