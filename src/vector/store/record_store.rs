//! RecordStore - Single source of truth for vector records
//!
//! RecordStore owns: vectors, ids, deleted bitmap, metadata.
//! HNSW owns: graph structure only.
//! OmenFile: pure I/O (no state duplication).

use roaring::RoaringBitmap;
use rustc_hash::{FxBuildHasher, FxHashMap};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// A single record in the store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<JsonValue>,
}

impl Record {
    /// Create a new record
    #[inline]
    pub fn new(id: String, vector: Vec<f32>, metadata: Option<JsonValue>) -> Self {
        Self {
            id,
            vector,
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
    slots: Vec<Option<Record>>,

    /// Single deleted bitmap (RoaringBitmap for O(1) operations)
    deleted: RoaringBitmap,

    /// Single ID to slot mapping
    id_to_slot: FxHashMap<String, u32>,

    /// Derived live count (cached for O(1) len())
    live_count: u32,

    /// Vector dimensions (fixed after first insert)
    dimensions: u32,
}

impl RecordStore {
    /// Create a new empty RecordStore
    pub fn new(dimensions: u32) -> Self {
        Self {
            slots: Vec::new(),
            deleted: RoaringBitmap::new(),
            id_to_slot: FxHashMap::default(),
            live_count: 0,
            dimensions,
        }
    }

    /// Create RecordStore with pre-allocated capacity
    pub fn with_capacity(dimensions: u32, capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            deleted: RoaringBitmap::new(),
            id_to_slot: FxHashMap::with_capacity_and_hasher(capacity, FxBuildHasher),
            live_count: 0,
            dimensions,
        }
    }

    /// Restore from snapshot (for persistence loading)
    pub fn from_snapshot(
        slots: Vec<Option<Record>>,
        deleted: RoaringBitmap,
        dimensions: u32,
    ) -> Self {
        // Rebuild id_to_slot mapping from slots
        let mut id_to_slot = FxHashMap::default();
        let mut live_count = 0u32;

        for (slot, record_opt) in slots.iter().enumerate() {
            let slot = slot as u32;
            if deleted.contains(slot) {
                continue;
            }
            if let Some(ref record) = record_opt {
                id_to_slot.insert(record.id.clone(), slot);
                live_count += 1;
            }
        }

        Self {
            slots,
            deleted,
            id_to_slot,
            live_count,
            dimensions,
        }
    }

    // =========================================================================
    // Core Operations
    // =========================================================================

    /// Upsert a record (insert or update)
    ///
    /// Returns the slot index where the record was stored.
    /// For updates, returns existing slot. For inserts, returns new slot.
    pub fn upsert(
        &mut self,
        id: String,
        vector: Vec<f32>,
        metadata: Option<JsonValue>,
    ) -> anyhow::Result<u32> {
        // Validate dimensions
        if self.dimensions == 0 {
            self.dimensions = vector.len() as u32;
        } else if vector.len() != self.dimensions as usize {
            anyhow::bail!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            );
        }

        // Check for existing record (update case)
        // IMPORTANT: We do NOT reuse slots on update to maintain HNSW node ID == RecordStore slot.
        // HNSW assigns sequential node IDs, so RecordStore must do the same.
        // On update: mark old slot deleted, insert at new slot.
        if let Some(&old_slot) = self.id_to_slot.get(&id) {
            // Mark old slot as deleted (don't clear data yet - compaction handles that)
            if !self.deleted.contains(old_slot) {
                self.deleted.insert(old_slot);
                self.live_count -= 1;
            }
            // Fall through to insert at new slot
        }

        // Insert at new slot (both new records and updates)
        let slot = self.slots.len() as u32;
        self.slots
            .push(Some(Record::new(id.clone(), vector, metadata)));
        self.id_to_slot.insert(id, slot);
        self.live_count += 1;

        Ok(slot)
    }

    /// Delete a record by ID - O(1)
    ///
    /// Returns the slot index if found and deleted, None if not found.
    pub fn delete(&mut self, id: &str) -> Option<u32> {
        let slot = *self.id_to_slot.get(id)?;

        // Already deleted?
        if self.deleted.contains(slot) {
            return None;
        }

        // Mark as deleted
        self.deleted.insert(slot);
        self.live_count = self.live_count.saturating_sub(1);

        // Remove from ID mapping so it can be re-inserted later
        self.id_to_slot.remove(id);

        Some(slot)
    }

    /// Get a record by ID
    pub fn get(&self, id: &str) -> Option<&Record> {
        let &slot = self.id_to_slot.get(id)?;
        if self.deleted.contains(slot) {
            return None;
        }
        self.slots.get(slot as usize).and_then(|r| r.as_ref())
    }

    /// Get a record by slot index
    pub fn get_by_slot(&self, slot: u32) -> Option<&Record> {
        if self.deleted.contains(slot) {
            return None;
        }
        self.slots.get(slot as usize).and_then(|r| r.as_ref())
    }

    /// Get a vector by slot index (for HNSW distance calculations)
    #[inline]
    pub fn get_vector(&self, slot: u32) -> Option<&[f32]> {
        self.slots
            .get(slot as usize)
            .and_then(|r| r.as_ref())
            .map(|r| r.vector.as_slice())
    }

    /// Check if a slot is live (not deleted)
    #[inline]
    pub fn is_live(&self, slot: u32) -> bool {
        !self.deleted.contains(slot) && (slot as usize) < self.slots.len()
    }

    /// Get the slot for a string ID
    #[inline]
    pub fn get_slot(&self, id: &str) -> Option<u32> {
        self.id_to_slot.get(id).copied()
    }

    /// Get the ID for a slot
    pub fn get_id(&self, slot: u32) -> Option<&str> {
        self.slots
            .get(slot as usize)
            .and_then(|r| r.as_ref())
            .map(|r| r.id.as_str())
    }

    // =========================================================================
    // Counts and Dimensions
    // =========================================================================

    /// Get live record count - O(1)
    #[inline]
    pub fn len(&self) -> u32 {
        self.live_count
    }

    /// Check if store is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Get total slot count (including deleted)
    #[inline]
    pub fn slot_count(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Get vector dimensions
    #[inline]
    pub fn dimensions(&self) -> u32 {
        self.dimensions
    }

    /// Set dimensions (only if currently 0)
    pub fn set_dimensions(&mut self, dimensions: u32) {
        if self.dimensions == 0 {
            self.dimensions = dimensions;
        }
    }

    /// Get deleted count
    #[inline]
    pub fn deleted_count(&self) -> u32 {
        self.deleted.len() as u32
    }

    /// Get delete ratio
    pub fn delete_ratio(&self) -> f64 {
        let total = self.slots.len();
        if total == 0 {
            return 0.0;
        }
        self.deleted.len() as f64 / total as f64
    }

    // =========================================================================
    // Iteration
    // =========================================================================

    /// Iterate over live records with their slot indices
    pub fn iter_live(&self) -> impl Iterator<Item = (u32, &Record)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, record_opt)| {
                let slot = slot as u32;
                if self.deleted.contains(slot) {
                    return None;
                }
                record_opt.as_ref().map(|r| (slot, r))
            })
    }

    /// Iterate over all slots (including deleted) with their records
    pub fn iter_all(&self) -> impl Iterator<Item = (u32, Option<&Record>)> {
        self.slots
            .iter()
            .enumerate()
            .map(|(slot, record_opt)| (slot as u32, record_opt.as_ref()))
    }

    /// Get all live vectors as Vec<Vec<f32>> (for batch HNSW operations)
    pub fn collect_vectors(&self) -> Vec<Vec<f32>> {
        self.iter_live()
            .map(|(_, record)| record.vector.clone())
            .collect()
    }

    // =========================================================================
    // Persistence Support
    // =========================================================================

    /// Get reference to the deleted bitmap
    #[inline]
    pub fn deleted_bitmap(&self) -> &RoaringBitmap {
        &self.deleted
    }

    /// Get reference to the slots
    #[inline]
    pub fn slots(&self) -> &[Option<Record>] {
        &self.slots
    }

    /// Get reference to id_to_slot mapping
    #[inline]
    pub fn id_to_slot(&self) -> &FxHashMap<String, u32> {
        &self.id_to_slot
    }

    /// Update metadata for a record by slot
    pub fn update_metadata(&mut self, slot: u32, metadata: JsonValue) -> anyhow::Result<()> {
        let record = self
            .slots
            .get_mut(slot as usize)
            .and_then(|r| r.as_mut())
            .ok_or_else(|| anyhow::anyhow!("Slot {slot} not found"))?;

        record.metadata = Some(metadata);
        Ok(())
    }

    /// Update vector for a record by slot
    pub fn update_vector(&mut self, slot: u32, vector: Vec<f32>) -> anyhow::Result<()> {
        if vector.len() != self.dimensions as usize {
            anyhow::bail!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            );
        }

        let record = self
            .slots
            .get_mut(slot as usize)
            .and_then(|r| r.as_mut())
            .ok_or_else(|| anyhow::anyhow!("Slot {slot} not found"))?;

        record.vector = vector;
        Ok(())
    }

    /// Check if needs compaction (delete ratio > 20%)
    pub fn needs_compaction(&self) -> bool {
        self.delete_ratio() > 0.20
    }

    // =========================================================================
    // Checkpoint Export
    // =========================================================================

    /// Export vectors for checkpoint (slot-indexed)
    pub fn export_vectors(&self) -> Vec<Option<Vec<f32>>> {
        self.slots
            .iter()
            .map(|opt| opt.as_ref().map(|r| r.vector.clone()))
            .collect()
    }

    /// Export ID mappings for checkpoint
    pub fn export_id_to_slot(&self) -> std::collections::HashMap<String, u32> {
        self.id_to_slot
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect()
    }

    /// Export deleted slots for checkpoint
    pub fn export_deleted(&self) -> Vec<u32> {
        self.deleted.iter().collect()
    }

    /// Export metadata for checkpoint
    pub fn export_metadata(&self) -> std::collections::HashMap<u32, JsonValue> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, opt)| {
                opt.as_ref()
                    .and_then(|r| r.metadata.clone())
                    .map(|m| (slot as u32, m))
            })
            .collect()
    }

    /// Compact the store - removes deleted records and reassigns slots
    ///
    /// Returns mapping from old slot to new slot for live records.
    pub fn compact(&mut self) -> FxHashMap<u32, u32> {
        let mut old_to_new: FxHashMap<u32, u32> = FxHashMap::default();
        let mut new_slots: Vec<Option<Record>> = Vec::with_capacity(self.live_count as usize);
        let mut new_id_to_slot: FxHashMap<String, u32> = FxHashMap::default();

        for (old_slot, record_opt) in self.slots.iter().enumerate() {
            let old_slot = old_slot as u32;

            // Skip deleted slots
            if self.deleted.contains(old_slot) {
                continue;
            }

            if let Some(record) = record_opt {
                let new_slot = new_slots.len() as u32;
                old_to_new.insert(old_slot, new_slot);
                new_id_to_slot.insert(record.id.clone(), new_slot);
                new_slots.push(Some(record.clone()));
            }
        }

        // Update state
        self.slots = new_slots;
        self.id_to_slot = new_id_to_slot;
        self.deleted.clear();
        // live_count stays the same

        old_to_new
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let store = RecordStore::new(128);
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        assert_eq!(store.dimensions(), 128);
    }

    #[test]
    fn test_upsert_insert() {
        let mut store = RecordStore::new(3);

        let slot = store
            .upsert("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();
        assert_eq!(slot, 0);
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());

        let slot2 = store
            .upsert("vec2".to_string(), vec![4.0, 5.0, 6.0], None)
            .unwrap();
        assert_eq!(slot2, 1);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_upsert_update() {
        let mut store = RecordStore::new(3);

        let slot1 = store
            .upsert("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();
        assert_eq!(slot1, 0);

        // Update same ID - creates new slot (to maintain slot == HNSW node ID)
        let slot2 = store
            .upsert("vec1".to_string(), vec![7.0, 8.0, 9.0], None)
            .unwrap();
        assert_eq!(slot2, 1); // New slot (old slot 0 is marked deleted)
        assert_eq!(store.len(), 1); // Still 1 live record

        // Check updated vector at new slot
        let vec = store.get_vector(1).unwrap();
        assert_eq!(vec, &[7.0, 8.0, 9.0]);

        // Old slot is deleted (get_by_slot respects deleted bitmap)
        assert!(store.get_by_slot(0).is_none());
    }

    #[test]
    fn test_delete() {
        let mut store = RecordStore::new(3);

        store
            .upsert("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();
        store
            .upsert("vec2".to_string(), vec![4.0, 5.0, 6.0], None)
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
        let mut store = RecordStore::new(3);
        store
            .upsert("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();

        assert!(store.delete("nonexistent").is_none());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_reinsert_after_delete() {
        let mut store = RecordStore::new(3);

        let slot1 = store
            .upsert("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();
        assert_eq!(slot1, 0);

        store.delete("vec1");
        assert_eq!(store.len(), 0);

        // Re-insert same ID gets new slot
        let slot2 = store
            .upsert("vec1".to_string(), vec![7.0, 8.0, 9.0], None)
            .unwrap();
        assert_eq!(slot2, 1); // New slot (old one is tombstoned)
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut store = RecordStore::new(3);

        store
            .upsert("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();

        // Try to insert wrong dimension
        let result = store.upsert("vec2".to_string(), vec![1.0, 2.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_iter_live() {
        let mut store = RecordStore::new(3);

        store
            .upsert("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();
        store
            .upsert("vec2".to_string(), vec![4.0, 5.0, 6.0], None)
            .unwrap();
        store
            .upsert("vec3".to_string(), vec![7.0, 8.0, 9.0], None)
            .unwrap();

        store.delete("vec2");

        let live: Vec<_> = store.iter_live().collect();
        assert_eq!(live.len(), 2);
        assert_eq!(live[0].0, 0);
        assert_eq!(live[1].0, 2);
    }

    #[test]
    fn test_compact() {
        let mut store = RecordStore::new(3);

        store
            .upsert("vec1".to_string(), vec![1.0, 2.0, 3.0], None)
            .unwrap();
        store
            .upsert("vec2".to_string(), vec![4.0, 5.0, 6.0], None)
            .unwrap();
        store
            .upsert("vec3".to_string(), vec![7.0, 8.0, 9.0], None)
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
        let mut store = RecordStore::new(3);

        let meta = serde_json::json!({"key": "value"});
        store
            .upsert("vec1".to_string(), vec![1.0, 2.0, 3.0], Some(meta.clone()))
            .unwrap();

        let record = store.get("vec1").unwrap();
        assert_eq!(record.metadata, Some(meta));
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

    #[test]
    fn test_delete_ratio() {
        let mut store = RecordStore::new(3);

        // Empty store
        assert_eq!(store.delete_ratio(), 0.0);

        // Add 10 records
        for i in 0..10 {
            store
                .upsert(format!("vec{i}"), vec![i as f32, 0.0, 0.0], None)
                .unwrap();
        }
        assert_eq!(store.delete_ratio(), 0.0);

        // Delete 3 records (30%)
        store.delete("vec0");
        store.delete("vec1");
        store.delete("vec2");
        assert!((store.delete_ratio() - 0.3).abs() < 0.01);
        assert!(store.needs_compaction());
    }
}
