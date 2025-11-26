//! Point-in-time consistent snapshots for seerdb

use crate::memtable::Memtable;
use crate::range::RangeIterator;
use crate::sstable::SSTable;
use crate::types::SnapshotHandle;
use crate::MergeOperator;
use bytes::Bytes;
use std::sync::{Arc, Mutex};

pub struct Snapshot {
    memtables: Vec<Arc<Memtable>>,
    immutable_memtables: Option<Arc<Vec<Arc<Memtable>>>>,
    sstables: Vec<Vec<Arc<Mutex<SSTable>>>>,
    sequence_number: u64,
    merge_operator: Option<Arc<dyn MergeOperator>>,
    /// Handle for automatic unregistration from SnapshotTracker on drop.
    /// When this Snapshot is dropped, the handle unregisters from the tracker,
    /// allowing compaction to GC versions older than this snapshot.
    #[allow(dead_code)] // Drop impl uses this implicitly
    gc_handle: Option<SnapshotHandle>,
}

impl Snapshot {
    /// Create a snapshot with GC tracking.
    /// The snapshot will automatically unregister from the tracker when dropped.
    pub(crate) fn with_gc_handle(
        memtables: Vec<Arc<Memtable>>,
        immutable_memtables: Option<Arc<Vec<Arc<Memtable>>>>,
        sstables: Vec<Vec<Arc<Mutex<SSTable>>>>,
        sequence_number: u64,
        merge_operator: Option<Arc<dyn MergeOperator>>,
        gc_handle: SnapshotHandle,
    ) -> Self {
        Self {
            memtables,
            immutable_memtables,
            sstables,
            sequence_number,
            merge_operator,
            gc_handle: Some(gc_handle),
        }
    }

    pub fn get(&self, key: &[u8]) -> crate::db::Result<Option<Bytes>> {
        for memtable in &self.memtables {
            if let Some((value, _seq)) = memtable.get(key, self.sequence_number) {
                if value.is_empty() {
                    return Ok(None);
                }
                return Ok(Some(Bytes::from(value.to_vec())));
            }
        }

        if let Some(ref immutables) = self.immutable_memtables {
            for memtable in immutables.iter() {
                if let Some((value, _seq)) = memtable.get(key, self.sequence_number) {
                    if value.is_empty() {
                        return Ok(None);
                    }
                    return Ok(Some(Bytes::from(value.to_vec())));
                }
            }
        }

        for level_sstables in self.sstables.iter() {
            let sstables_iter = level_sstables.iter().rev();
            for sstable_arc in sstables_iter {
                let mut sstable_guard = sstable_arc.lock().expect("SSTable lock poisoned");
                // Use MVCC-aware lookup to respect snapshot sequence number
                if let Ok(Some(value)) = sstable_guard.get_mvcc(key, self.sequence_number) {
                    drop(sstable_guard);
                    if value.is_empty() {
                        return Ok(None);
                    }
                    return Ok(Some(value));
                }
            }
        }

        Ok(None)
    }

    pub fn range(
        &self,
        start_key: &[u8],
        end_key: Option<&[u8]>,
    ) -> crate::db::Result<RangeIterator> {
        let mut partition_refs: Vec<&Memtable> =
            self.memtables.iter().map(|arc| arc.as_ref()).collect();

        if let Some(ref immutables) = self.immutable_memtables {
            partition_refs.extend(immutables.iter().map(|arc| arc.as_ref()));
        }

        let mut sstables = Vec::new();
        for level_sstables in &self.sstables {
            for sstable_arc in level_sstables {
                let sstable_guard = sstable_arc.lock().expect("SSTable lock poisoned");
                let overlaps = sstable_guard.overlaps_range(start_key, end_key);

                if overlaps {
                    let iter = sstable_guard.scan_range(start_key, end_key);
                    drop(sstable_guard);
                    sstables.push(iter);
                } else {
                    drop(sstable_guard);
                }
            }
        }

        RangeIterator::new(
            start_key,
            end_key,
            &partition_refs,
            sstables,
            self.merge_operator.clone(),
        )
    }

    pub fn sequence_number(&self) -> u64 {
        self.sequence_number
    }
}

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_sstables: usize = self.sstables.iter().map(|l| l.len()).sum();
        f.debug_struct("Snapshot")
            .field("sequence_number", &self.sequence_number)
            .field("memtable_partitions", &self.memtables.len())
            .field(
                "has_immutable_memtables",
                &self.immutable_memtables.is_some(),
            )
            .field("lsm_levels", &self.sstables.len())
            .field("total_sstables", &total_sstables)
            .finish()
    }
}

#[cfg(test)]
mod tests {}
