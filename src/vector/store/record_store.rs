//! RecordStore - Single source of truth for vector records
//!
//! RecordStore owns: vectors, ids, deleted bitmap, metadata.
//! HNSW owns: graph structure only.
//! OmenFile: pure I/O (no state duplication).

use dashmap::DashMap;
use parking_lot::RwLock;
use roaring::RoaringBitmap;
use rustc_hash::{FxHashMap, FxHasher};
use serde_json::Value as JsonValue;
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::vector::sparse::SparseVector;
use memmap2::Mmap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub enum SnapshotMmapSource {
    VecFile,
    MainFile,
}

#[derive(Debug, Clone)]
pub enum SnapshotVectorData {
    Owned(Vec<f32>),
    Mmap {
        source: SnapshotMmapSource,
        offset: usize,
        len: usize,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct SnapshotRecord {
    pub id: String,
    pub vector: SnapshotVectorData,
    pub metadata: Option<JsonValue>,
}

#[derive(Debug, Clone)]
pub enum VectorData {
    Owned(Vec<f32>),
    Mmap(Arc<Mmap>, usize, usize),
}

impl VectorData {
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        match self {
            Self::Owned(v) => v.as_slice(),
            Self::Mmap(mmap, offset, len) => unsafe {
                std::slice::from_raw_parts(mmap.as_ptr().add(*offset).cast::<f32>(), *len)
            },
        }
    }

    #[inline]
    pub fn to_vec(&self) -> Vec<f32> {
        self.as_slice().to_vec()
    }
}

#[derive(Debug, Clone)]
enum StoredVectorData {
    Owned(Vec<f32>),
    Mmap { mmap_id: usize, offset: usize, len: usize },
}

#[derive(Debug, Clone)]
struct StoredRecord {
    id: String,
    vector: StoredVectorData,
    sparse: Option<SparseVector>,
    metadata: Option<JsonValue>,
}

/// A single record in the store
#[derive(Debug, Clone)]
pub struct Record {
    pub id: String,
    pub vector: VectorData,
    pub metadata: Option<JsonValue>,
}

impl serde::Serialize for Record {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Record", 3)?;
        state.serialize_field("id", &self.id)?;
        state.serialize_field("vector", self.vector.as_slice())?;
        state.serialize_field("metadata", &self.metadata)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for Record {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct RecordData {
            id: String,
            vector: Vec<f32>,
            metadata: Option<JsonValue>,
        }
        let data = RecordData::deserialize(deserializer)?;
        Ok(Record::new(data.id, data.vector, data.metadata))
    }
}

impl Record {
    /// Create a new record
    #[inline]
    pub fn new(id: String, vector: Vec<f32>, metadata: Option<JsonValue>) -> Self {
        Self {
            id,
            vector: VectorData::Owned(vector),
            metadata,
        }
    }

    /// Create a new record from an mmap slice
    #[inline]
    pub fn new_mmap(
        id: String,
        mmap: Arc<Mmap>,
        offset: usize,
        len: usize,
        metadata: Option<JsonValue>,
    ) -> Self {
        Self {
            id,
            vector: VectorData::Mmap(mmap, offset, len),
            metadata,
        }
    }
}

/// RecordStore - owns all vector records with O(1) operations
///
/// Slot-based storage where each vector gets a slot index.
/// Deleted vectors are tracked in a RoaringBitmap for O(1) delete/is_live checks.
#[derive(Debug)]
pub struct RecordStore {
    /// Slot-based storage - Some(record) for live, None for deleted/empty
    slots: RwLock<Vec<Option<StoredRecord>>>,

    /// Single deleted bitmap (RoaringBitmap for O(1) operations)
    deleted: RwLock<RoaringBitmap>,

    /// Shared mmap ownership for all mmap-backed vectors in the store.
    mmaps: RwLock<Vec<Arc<Mmap>>>,

    /// Single ID to slot mapping - Thread-safe for concurrent lookups
    id_to_slot: DashMap<String, u32, BuildHasherDefault<FxHasher>>,

    /// Derived live count (cached for O(1) len()) - Atomic for concurrent reads
    live_count: AtomicU32,

    /// Vector dimensions (fixed after first insert)
    dimensions: AtomicU32,

    /// Slots modified since last checkpoint (for incremental persistence)
    dirty_slots: RwLock<RoaringBitmap>,
}

impl RecordStore {
    /// Create a new empty RecordStore
    pub fn new(dimensions: u32) -> Self {
        Self {
            slots: RwLock::new(Vec::new()),
            deleted: RwLock::new(RoaringBitmap::new()),
            mmaps: RwLock::new(Vec::new()),
            id_to_slot: DashMap::default(),
            live_count: AtomicU32::new(0),
            dimensions: AtomicU32::new(dimensions),
            dirty_slots: RwLock::new(RoaringBitmap::new()),
        }
    }

    /// Restore from snapshot (for persistence loading)
    pub fn from_snapshot(
        slots: Vec<Option<Record>>,
        deleted: RoaringBitmap,
        dimensions: u32,
    ) -> Self {
        let store = Self::new(dimensions);
        let stored_slots: Vec<Option<StoredRecord>> = slots
            .into_iter()
            .map(|record| record.map(|record| store.store_record(record)))
            .collect();
        let live_count = store.rebuild_indexes_from_slots(&stored_slots, &deleted);
        *store.slots.write() = stored_slots;
        *store.deleted.write() = deleted;
        store.live_count.store(live_count, Ordering::SeqCst);
        store
    }

    pub(crate) fn restore_snapshot_records(
        &mut self,
        slots: Vec<Option<SnapshotRecord>>,
        deleted: RoaringBitmap,
        live_count: u32,
        vec_mmap: Option<Arc<Mmap>>,
        main_mmap: Option<Arc<Mmap>>,
    ) {
        let stored_slots: Vec<Option<StoredRecord>> = slots
            .into_iter()
            .map(|record| {
                record.map(|record| StoredRecord {
                    id: record.id,
                    vector: self.store_snapshot_vector(record.vector, &vec_mmap, &main_mmap),
                    sparse: None,
                    metadata: record.metadata,
                })
            })
            .collect();
        self.id_to_slot.clear();
        self.rebuild_indexes_from_slots(&stored_slots, &deleted);
        *self.slots.write() = stored_slots;
        *self.deleted.write() = deleted;
        self.live_count.store(live_count, Ordering::SeqCst);
    }

    /// Set a record (insert or update)
    ///
    /// Returns the slot index where the record was stored.
    /// For updates, returns existing slot. For inserts, returns new slot.
    pub fn set(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata: Option<JsonValue>,
    ) -> anyhow::Result<u32> {
        // Validate dimensions
        let current_dims = self.dimensions.load(Ordering::Relaxed);
        if current_dims == 0 {
            let new_dims = vector.len() as u32;
            if self
                .dimensions
                .compare_exchange(0, new_dims, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                // Someone else set it, check if it matches
                let final_dims = self.dimensions.load(Ordering::SeqCst);
                if vector.len() as u32 != final_dims {
                    anyhow::bail!(
                        "Vector dimension mismatch: expected {}, got {}",
                        final_dims,
                        vector.len()
                    );
                }
            }
        } else if vector.len() as u32 != current_dims {
            anyhow::bail!(
                "Vector dimension mismatch: expected {}, got {}",
                current_dims,
                vector.len()
            );
        }

        // Check for existing record (update case)
        let mut carried_sparse = None;
        if let Some(old_slot_ref) = self.id_to_slot.get(&id) {
            let old_slot = *old_slot_ref;
            carried_sparse = self
                .slots
                .read()
                .get(old_slot as usize)
                .and_then(|record| record.as_ref())
                .and_then(|record| record.sparse.clone());
            let mut deleted = self.deleted.write();
            if !deleted.contains(old_slot) {
                deleted.insert(old_slot);
                self.dirty_slots.write().insert(old_slot);
                self.live_count.fetch_sub(1, Ordering::Relaxed);
            }
        }

        // Insert at new slot
        let mut slots = self.slots.write();
        let slot = slots.len() as u32;
        slots.push(Some(StoredRecord {
            id: id.clone(),
            vector: StoredVectorData::Owned(vector),
            sparse: carried_sparse,
            metadata,
        }));
        self.id_to_slot.insert(id, slot);
        self.live_count.fetch_add(1, Ordering::Relaxed);
        self.dirty_slots.write().insert(slot);

        Ok(slot)
    }

    /// Delete a record by ID - O(1)
    pub fn delete(&self, id: &str) -> Option<u32> {
        let slot = *self.id_to_slot.get(id)?;

        let mut deleted = self.deleted.write();
        if deleted.contains(slot) {
            return None;
        }

        deleted.insert(slot);
        self.dirty_slots.write().insert(slot);
        self.live_count.fetch_sub(1, Ordering::Relaxed);
        self.id_to_slot.remove(id);

        Some(slot)
    }

    /// Get a record by ID
    pub fn get(&self, id: &str) -> Option<Record> {
        let slot = *self.id_to_slot.get(id)?;
        if self.deleted.read().contains(slot) {
            return None;
        }
        self.slots
            .read()
            .get(slot as usize)
            .and_then(|record| record.as_ref())
            .map(|record| self.materialize_record(record))
    }

    /// Get a record by slot index
    pub fn get_by_slot(&self, slot: u32) -> Option<Record> {
        if self.deleted.read().contains(slot) {
            return None;
        }
        self.slots
            .read()
            .get(slot as usize)
            .and_then(|record| record.as_ref())
            .map(|record| self.materialize_record(record))
    }

    /// Get just the search result fields for a slot.
    ///
    /// This avoids cloning the stored vector when materializing final search
    /// results, where only the user ID and metadata are needed.
    pub fn get_result_fields_by_slot(&self, slot: u32) -> Option<(String, Option<JsonValue>)> {
        if self.deleted.read().contains(slot) {
            return None;
        }

        self.slots
            .read()
            .get(slot as usize)
            .and_then(|record| record.as_ref())
            .map(|record| (record.id.clone(), record.metadata.clone()))
    }

    /// Check if a slot is live (not deleted)
    #[inline]
    pub fn is_live(&self, slot: u32) -> bool {
        !self.deleted.read().contains(slot) && (slot as usize) < self.slots.read().len()
    }

    /// Get the slot for a string ID
    #[inline]
    pub fn get_slot(&self, id: &str) -> Option<u32> {
        self.id_to_slot.get(id).map(|r| *r)
    }

    /// Get the ID for a slot
    pub fn get_id(&self, slot: u32) -> Option<String> {
        self.slots
            .read()
            .get(slot as usize)
            .and_then(|r| r.as_ref().map(|r| r.id.clone()))
    }

    /// Get live record count - O(1)
    #[inline]
    pub fn len(&self) -> u32 {
        self.live_count.load(Ordering::Relaxed)
    }

    /// Check if store is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get total slot count (including deleted)
    #[inline]
    pub fn slot_count(&self) -> u32 {
        self.slots.read().len() as u32
    }

    /// Get vector dimensions
    #[inline]
    pub fn dimensions(&self) -> u32 {
        self.dimensions.load(Ordering::Relaxed)
    }

    /// Set dimensions (only if currently 0)
    pub fn set_dimensions(&self, dimensions: u32) {
        let _ = self
            .dimensions
            .compare_exchange(0, dimensions, Ordering::SeqCst, Ordering::SeqCst);
    }

    /// Get deleted count
    #[inline]
    pub fn deleted_count(&self) -> u32 {
        self.deleted.read().len() as u32
    }

    /// Iterate over live records (returns clones for thread safety)
    pub fn iter_live(&self) -> impl Iterator<Item = (u32, Record)> {
        let slots = self.slots.read().clone();
        let deleted = self.deleted.read().clone();
        let mmaps = self.mmaps.read().clone();

        slots
            .into_iter()
            .enumerate()
            .filter_map(move |(slot, record_opt)| {
                let slot = slot as u32;
                if deleted.contains(slot) {
                    return None;
                }
                record_opt.map(|record| {
                    (
                        slot,
                        RecordStore::materialize_record_from_parts(&mmaps, &record),
                    )
                })
            })
    }

    /// Get a clone of the deleted bitmap
    pub fn deleted_bitmap(&self) -> RoaringBitmap {
        self.deleted.read().clone()
    }

    /// Update metadata for a record by slot
    pub fn update_metadata(&self, slot: u32, metadata: JsonValue) -> anyhow::Result<()> {
        let mut slots = self.slots.write();
        let record = slots
            .get_mut(slot as usize)
            .and_then(|r| r.as_mut())
            .ok_or_else(|| anyhow::anyhow!("Slot {slot} not found"))?;

        self.dirty_slots.write().insert(slot);
        record.metadata = Some(metadata);
        Ok(())
    }

    /// Update sparse payload for a record by slot.
    pub fn update_sparse(&self, slot: u32, sparse: Option<SparseVector>) -> anyhow::Result<()> {
        let mut slots = self.slots.write();
        let record = slots
            .get_mut(slot as usize)
            .and_then(|r| r.as_mut())
            .ok_or_else(|| anyhow::anyhow!("Slot {slot} not found"))?;

        self.dirty_slots.write().insert(slot);
        record.sparse = sparse;
        Ok(())
    }

    /// Get a cloned sparse payload for a slot.
    pub fn get_sparse(&self, slot: u32) -> Option<SparseVector> {
        self.slots
            .read()
            .get(slot as usize)?
            .as_ref()
            .and_then(|record| record.sparse.clone())
    }

    /// Export vector references for checkpoint (returns owned copy for stability)
    pub fn export_vectors(&self) -> Vec<Option<Vec<f32>>> {
        let slots = self.slots.read();
        slots
            .iter()
            .map(|opt| {
                opt.as_ref()
                    .map(|record| self.materialize_vector(&record.vector).to_vec())
            })
            .collect()
    }

    /// Return a copy of the ID→slot map
    pub fn id_to_slot_ref(&self) -> FxHashMap<String, u32> {
        let mut map = FxHashMap::default();
        for entry in &self.id_to_slot {
            map.insert(entry.key().clone(), *entry.value());
        }
        map
    }

    /// Export deleted slots for checkpoint
    pub fn export_deleted(&self) -> Vec<u32> {
        self.deleted.read().iter().collect()
    }

    /// Iterate (slot, metadata) pairs
    pub fn iter_metadata(&self) -> Vec<(u32, JsonValue)> {
        self.slots
            .read()
            .iter()
            .enumerate()
            .filter_map(|(slot, opt): (usize, &Option<StoredRecord>)| {
                opt.as_ref()
                    .and_then(|r| r.metadata.clone())
                    .map(|m| (slot as u32, m))
            })
            .collect()
    }

    /// Take dirty slots bitmap, resetting it to empty
    pub fn take_dirty_slots(&self) -> RoaringBitmap {
        std::mem::take(&mut *self.dirty_slots.write())
    }

    /// Restore dirty slots
    pub fn restore_dirty_slots(&self, slots: RoaringBitmap) {
        *self.dirty_slots.write() |= slots;
    }

    /// Get vector data for a slot
    pub fn get_vector(&self, slot: u32) -> Option<VectorData> {
        self.slots
            .read()
            .get(slot as usize)?
            .as_ref()
            .map(|record| self.materialize_vector(&record.vector))
    }

    /// Borrow vector data for a slot while holding the slot lock.
    ///
    /// This is for hot write paths that already own the source vector and only
    /// need a temporary borrow into the record store before handing the slice
    /// to the dense engine.
    pub fn with_vector_by_slot<T>(&self, slot: u32, f: impl FnOnce(Option<&[f32]>) -> T) -> T {
        let slots = self.slots.read();
        let record = slots.get(slot as usize).and_then(|record| record.as_ref());
        self.with_record_slice(record, f)
    }

    /// Borrow vector slices for a list of slots while holding the slot lock.
    ///
    /// The slices are only valid for the duration of the callback.
    pub fn with_vectors_by_slots<T>(&self, slots: &[u32], f: impl FnOnce(Vec<&[f32]>) -> T) -> T {
        let records = self.slots.read();
        let mmaps = self.mmaps.read();
        let vectors = slots
            .iter()
            .filter_map(|&slot| {
                records
                    .get(slot as usize)
                    .and_then(|record| record.as_ref())
                    .map(|record| Self::stored_vector_as_slice(&mmaps, &record.vector))
            })
            .collect();
        f(vectors)
    }

    /// Compact the store - removes deleted records and reassigns slots
    pub fn compact(&self) -> FxHashMap<u32, u32> {
        let mut old_to_new: FxHashMap<u32, u32> = FxHashMap::default();
        let mut new_slots: Vec<Option<StoredRecord>> = Vec::with_capacity(self.len() as usize);
        let mut new_id_to_slot: Vec<(String, u32)> = Vec::with_capacity(self.len() as usize);

        let slots = self.slots.read();
        let deleted = self.deleted.read();

        for (old_slot, record_opt) in slots.iter().enumerate() {
            let old_slot = old_slot as u32;
            if deleted.contains(old_slot) {
                continue;
            }

            if let Some(record) = record_opt {
                let new_slot = new_slots.len() as u32;
                old_to_new.insert(old_slot, new_slot);
                new_id_to_slot.push((record.id.clone(), new_slot));
                new_slots.push(Some(record.clone()));
            }
        }

        drop(slots);
        drop(deleted);

        // Update state
        *self.slots.write() = new_slots;
        self.id_to_slot.clear();
        for (id, slot) in new_id_to_slot {
            self.id_to_slot.insert(id, slot);
        }
        self.deleted.write().clear();
        *self.dirty_slots.write() = (0..self.slots.read().len() as u32).collect();

        old_to_new
    }

    fn rebuild_indexes_from_slots(
        &self,
        slots: &[Option<StoredRecord>],
        deleted: &RoaringBitmap,
    ) -> u32 {
        let mut live_count = 0u32;
        for (slot, record_opt) in slots.iter().enumerate() {
            let slot = slot as u32;
            if deleted.contains(slot) {
                continue;
            }
            if let Some(record) = record_opt {
                self.id_to_slot.insert(record.id.clone(), slot);
                live_count += 1;
            }
        }
        live_count
    }

    fn store_record(&self, record: Record) -> StoredRecord {
        StoredRecord {
            id: record.id,
            vector: self.store_vector(record.vector),
            sparse: None,
            metadata: record.metadata,
        }
    }

    fn store_vector(&self, vector: VectorData) -> StoredVectorData {
        match vector {
            VectorData::Owned(vector) => StoredVectorData::Owned(vector),
            VectorData::Mmap(mmap, offset, len) => StoredVectorData::Mmap {
                mmap_id: self.add_mmap(mmap),
                offset,
                len,
            },
        }
    }

    fn add_mmap(&self, mmap: Arc<Mmap>) -> usize {
        let mut mmaps = self.mmaps.write();
        if let Some(idx) = mmaps.iter().position(|existing| Arc::ptr_eq(existing, &mmap)) {
            idx
        } else {
            let idx = mmaps.len();
            mmaps.push(mmap);
            idx
        }
    }

    fn store_snapshot_vector(
        &self,
        vector: SnapshotVectorData,
        vec_mmap: &Option<Arc<Mmap>>,
        main_mmap: &Option<Arc<Mmap>>,
    ) -> StoredVectorData {
        match vector {
            SnapshotVectorData::Owned(vector) => StoredVectorData::Owned(vector),
            SnapshotVectorData::Mmap {
                source,
                offset,
                len,
            } => {
                let mmap = match source {
                    SnapshotMmapSource::VecFile => vec_mmap
                        .as_ref()
                        .unwrap_or_else(|| panic!("missing vec mmap for snapshot vector"))
                        .clone(),
                    SnapshotMmapSource::MainFile => main_mmap
                        .as_ref()
                        .unwrap_or_else(|| panic!("missing main mmap for snapshot vector"))
                        .clone(),
                };
                StoredVectorData::Mmap {
                    mmap_id: self.add_mmap(mmap),
                    offset,
                    len,
                }
            }
        }
    }

    fn materialize_record(&self, record: &StoredRecord) -> Record {
        let mmaps = self.mmaps.read();
        Self::materialize_record_from_parts(&mmaps, record)
    }

    fn materialize_record_from_parts(mmaps: &[Arc<Mmap>], record: &StoredRecord) -> Record {
        Record {
            id: record.id.clone(),
            vector: Self::materialize_vector_from_parts(mmaps, &record.vector),
            metadata: record.metadata.clone(),
        }
    }

    fn materialize_vector(&self, vector: &StoredVectorData) -> VectorData {
        let mmaps = self.mmaps.read();
        Self::materialize_vector_from_parts(&mmaps, vector)
    }

    fn materialize_vector_from_parts(mmaps: &[Arc<Mmap>], vector: &StoredVectorData) -> VectorData {
        match vector {
            StoredVectorData::Owned(vector) => VectorData::Owned(vector.clone()),
            StoredVectorData::Mmap {
                mmap_id,
                offset,
                len,
            } => VectorData::Mmap(mmaps[*mmap_id].clone(), *offset, *len),
        }
    }

    fn with_record_slice<T>(
        &self,
        record: Option<&StoredRecord>,
        f: impl FnOnce(Option<&[f32]>) -> T,
    ) -> T {
        if let Some(record) = record {
            let mmaps = self.mmaps.read();
            f(Some(Self::stored_vector_as_slice(&mmaps, &record.vector)))
        } else {
            f(None)
        }
    }

    fn stored_vector_as_slice<'a>(mmaps: &'a [Arc<Mmap>], vector: &'a StoredVectorData) -> &'a [f32] {
        match vector {
            StoredVectorData::Owned(vector) => vector.as_slice(),
            StoredVectorData::Mmap {
                mmap_id,
                offset,
                len,
            } => unsafe {
                std::slice::from_raw_parts(mmaps[*mmap_id].as_ptr().add(*offset).cast::<f32>(), *len)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::sparse::SparseVector;

    #[test]
    fn test_new_empty() {
        let store = RecordStore::new(128);
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        assert_eq!(store.dimensions(), 128);
    }

    #[test]
    fn test_set_insert() {
        let store = RecordStore::new(3);

        let slot = store
            .set("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();
        assert_eq!(slot, 0);
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());

        let slot2 = store
            .set("vec2".to_string(), vec![4.0, 5.0, 6.0], None)
            .unwrap();
        assert_eq!(slot2, 1);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_set_update() {
        let store = RecordStore::new(3);

        let slot1 = store
            .set("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();
        assert_eq!(slot1, 0);

        // Update same ID - creates new slot (to maintain slot == HNSW node ID)
        let slot2 = store
            .set("vec1".to_string(), vec![7.0, 8.0, 9.0], None)
            .unwrap();
        assert_eq!(slot2, 1); // New slot (old slot 0 is marked deleted)
        assert_eq!(store.len(), 1); // Still 1 live record

        // Check updated vector at new slot
        let record = store.get_by_slot(1).unwrap();
        assert_eq!(record.vector.as_slice(), &[7.0, 8.0, 9.0]);

        // Old slot is deleted (get_by_slot respects deleted bitmap)
        assert!(store.get_by_slot(0).is_none());
    }

    #[test]
    fn test_set_update_preserves_sparse_payload() {
        let store = RecordStore::new(3);

        let slot1 = store
            .set("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();
        let sparse = SparseVector::from_pairs(vec![(7, 0.5), (42, 1.5)]).unwrap();
        store.update_sparse(slot1, Some(sparse.clone())).unwrap();

        let slot2 = store
            .set("vec1".to_string(), vec![4.0, 5.0, 6.0], None)
            .unwrap();

        assert_eq!(slot2, 1);
        assert_eq!(store.get_sparse(slot2).as_ref(), Some(&sparse));
    }

    #[test]
    fn test_delete() {
        let store = RecordStore::new(3);

        store
            .set("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();
        store
            .set("vec2".to_string(), vec![4.0, 5.0, 6.0], None)
            .unwrap();
        assert_eq!(store.len(), 2);

        // Delete vec1
        let deleted_slot = store.delete("vec1");
        assert_eq!(deleted_slot, Some(0));
        assert_eq!(store.len(), 1);

        // vec1 is no longer accessible
        assert!(store.get("vec1").is_none());
        assert!(!store.is_live(0));

        // vec2 is still accessible
        assert!(store.get("vec2").is_some());
        assert!(store.is_live(1));
    }

    #[test]
    fn test_delete_nonexistent() {
        let store = RecordStore::new(3);
        store
            .set("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();

        assert!(store.delete("nonexistent").is_none());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_reinsert_after_delete() {
        let store = RecordStore::new(3);

        let slot1 = store
            .set("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();
        assert_eq!(slot1, 0);

        store.delete("vec1");
        assert_eq!(store.len(), 0);

        // Re-insert same ID gets new slot
        let slot2 = store
            .set("vec1".to_string(), vec![7.0, 8.0, 9.0], None)
            .unwrap();
        assert_eq!(slot2, 1); // New slot (old one is tombstoned)
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_dimension_mismatch() {
        let store = RecordStore::new(3);

        store
            .set("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();

        // Try to insert wrong dimension
        let result = store.set("vec2".to_string(), vec![1.0, 2.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_iter_live() {
        let store = RecordStore::new(3);

        store
            .set("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();
        store
            .set("vec2".to_string(), vec![4.0, 5.0, 6.0], None)
            .unwrap();
        store
            .set("vec3".to_string(), vec![7.0, 8.0, 9.0], None)
            .unwrap();

        store.delete("vec2");

        let live: Vec<_> = store.iter_live().collect();
        assert_eq!(live.len(), 2);
        assert_eq!(live[0].0, 0);
        assert_eq!(live[1].0, 2);
    }

    #[test]
    fn test_compact() {
        let store = RecordStore::new(3);

        store
            .set("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();
        store
            .set("vec2".to_string(), vec![4.0, 5.0, 6.0], None)
            .unwrap();
        store
            .set("vec3".to_string(), vec![7.0, 8.0, 9.0], None)
            .unwrap();

        store.delete("vec1");
        store.delete("vec2");

        assert_eq!(store.len(), 1);
        assert_eq!(store.slot_count(), 3);

        // Compact
        let mapping = store.compact();

        assert_eq!(store.len(), 1);
        assert_eq!(store.slot_count(), 1);
        assert_eq!(store.deleted_count(), 0);

        // vec3 moved from slot 2 to slot 0
        assert_eq!(mapping.get(&2), Some(&0));

        // vec3 still accessible with new slot
        assert!(store.get("vec3").is_some());
        assert_eq!(store.get_slot("vec3"), Some(0));
    }

    #[test]
    fn test_metadata() {
        let store = RecordStore::new(3);

        let meta = serde_json::json!({"key": "value"});
        store
            .set("vec1".to_string(), vec![1.0, 2.0, 3.0], Some(meta.clone()))
            .unwrap();

        let record = store.get("vec1").unwrap();
        assert_eq!(record.metadata, Some(meta));
    }

    #[test]
    fn test_get_result_fields_by_slot() {
        let store = RecordStore::new(3);

        let meta = serde_json::json!({"key": "value"});
        let slot = store
            .set("vec1".to_string(), vec![1.0, 2.0, 3.0], Some(meta.clone()))
            .unwrap();

        let (id, metadata) = store.get_result_fields_by_slot(slot).unwrap();
        assert_eq!(id, "vec1");
        assert_eq!(metadata, Some(meta));
    }

    #[test]
    fn test_from_snapshot() {
        let mut deleted = RoaringBitmap::new();
        deleted.insert(1);

        let slots = vec![
            Some(Record::new("vec1".to_string(), vec![1.0, 2.0, 3.0], None)),
            Some(Record::new("vec2".to_string(), vec![4.0, 5.0, 6.0], None)), // deleted
            Some(Record::new("vec3".to_string(), vec![7.0, 8.0, 9.0], None)),
        ];

        let store = RecordStore::from_snapshot(slots, deleted, 3);

        assert_eq!(store.len(), 2);
        assert!(store.is_live(0));
        assert!(!store.is_live(1));
        assert!(store.is_live(2));

        assert_eq!(store.get_slot("vec1"), Some(0));
        assert_eq!(store.get_slot("vec2"), None); // deleted
        assert_eq!(store.get_slot("vec3"), Some(2));
    }
}
