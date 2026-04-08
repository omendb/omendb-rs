//! Record store type skeleton.

use crate::catalog::SlotId;
use crate::storage::{DenseStore, MultiStore, RecordMeta, SparseStore, TextStore};
use dashmap::DashMap;
use parking_lot::RwLock;
use roaring::RoaringBitmap;
use rustc_hash::FxHasher;
use std::hash::BuildHasherDefault;
use std::sync::atomic::AtomicU32;

#[derive(Debug, Default)]
pub struct DirtySets {
    pub meta: RwLock<RoaringBitmap>,
    pub dense: RwLock<RoaringBitmap>,
    pub sparse: RwLock<RoaringBitmap>,
    pub multi: RwLock<RoaringBitmap>,
    pub text: RwLock<RoaringBitmap>,
}

#[derive(Debug)]
pub struct RecordStore {
    pub meta: RwLock<Vec<Option<RecordMeta>>>,
    pub dense: DenseStore,
    pub sparse: SparseStore,
    pub multi: MultiStore,
    pub text: TextStore,
    pub deleted: RwLock<RoaringBitmap>,
    pub dirty: DirtySets,
    pub id_to_slot: DashMap<String, SlotId, BuildHasherDefault<FxHasher>>,
    pub live_count: AtomicU32,
}
