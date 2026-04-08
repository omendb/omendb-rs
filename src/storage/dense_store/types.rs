//! Dense value storage types.

use memmap2::Mmap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;

#[derive(Debug, Clone)]
pub enum DenseValue {
    Owned(Vec<f32>),
    MmapRef {
        mmap_id: u32,
        offset_bytes: u64,
        len: u32,
    },
}

#[derive(Debug)]
pub struct DenseStore {
    pub(crate) values: RwLock<Vec<Option<DenseValue>>>,
    pub(crate) mmaps: RwLock<Vec<Arc<Mmap>>>,
    pub(crate) dim: AtomicU32,
}
