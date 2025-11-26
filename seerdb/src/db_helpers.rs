// Database helper functions
//
// Utility functions used by the main DB implementation that are relatively
// self-contained and don't require the full DB struct.

use crate::db::{partition_for_key, DBError, DBOptions, Result};
use crate::memtable::{Entry, Memtable};
use crate::merge_operator::MergeOperator;
use crate::wal::{Record, RecoveryMode, WALReader};
use bytes::Bytes;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{debug, error, warn};

/// Recover partitioned memtables from WAL
///
/// Reads records one by one and distributes them across partitions using hash function.
/// Behavior on corruption depends on recovery_mode:
/// - `RecoveryMode::Strict`: Returns error on any corruption
/// - `RecoveryMode::BestEffort`: Stops reading but recovers all valid records before corruption
pub(crate) fn recover_partitioned(
    wal_path: &Path,
    memtables: &[Memtable],
    merge_operator: Option<&Arc<dyn MergeOperator>>,
    recovery_mode: RecoveryMode,
) -> Result<()> {
    let mut reader =
        WALReader::open(wal_path).map_err(|e| DBError::Io(std::io::Error::other(e)))?;

    let mut records_recovered = 0u64;

    // Read records one by one
    loop {
        match reader.read_next() {
            Ok(Some(record)) => {
                match record {
                    Record::Put { key, value, seq } => {
                        // Hash key to determine partition
                        let partition = partition_for_key(&key);
                        memtables[partition].put(key, value, seq);
                    }
                    Record::Delete { key, seq } => {
                        // Hash key to determine partition
                        let partition = partition_for_key(&key);
                        memtables[partition].delete(key, seq);
                    }
                    Record::Batch {
                        operations,
                        base_seq,
                    } => {
                        // Apply all operations in the batch atomically
                        let mut current_seq = base_seq;
                        for op in operations {
                            match op {
                                crate::wal::BatchOp::Put { key, value } => {
                                    let partition = partition_for_key(&key);
                                    memtables[partition].put(key, value, current_seq);
                                }
                                crate::wal::BatchOp::Delete { key } => {
                                    let partition = partition_for_key(&key);
                                    memtables[partition].delete(key, current_seq);
                                }
                                crate::wal::BatchOp::Merge { key, operand } => {
                                    let partition = partition_for_key(&key);
                                    apply_merge(&memtables[partition], key, operand, merge_operator);
                                }
                            }
                            current_seq += 1;
                        }
                    }
                    Record::Merge {
                        key,
                        operand,
                        seq: _,
                    } => {
                        // Note: apply_merge currently uses u64::MAX internally via put_entry
                        // Ideally it should use the record seq.
                        let partition = partition_for_key(&key);
                        apply_merge(&memtables[partition], key, operand, merge_operator);
                    }
                }
                records_recovered += 1;
            }
            Ok(None) => {
                // End of WAL reached
                break;
            }
            Err(e) => {
                // Corruption or truncation encountered
                match recovery_mode {
                    RecoveryMode::Strict => {
                        error!(
                            error = %e,
                            records_recovered = records_recovered,
                            "WAL corruption detected in strict mode"
                        );
                        return Err(DBError::WalCorruption(format!(
                            "WAL corruption after {} records: {}",
                            records_recovered, e
                        )));
                    }
                    RecoveryMode::BestEffort => {
                        // Stop reading but don't fail - we've recovered all valid records
                        warn!(
                            error = %e,
                            records_recovered = records_recovered,
                            "WAL recovery stopped due to corrupt/truncated record (best-effort mode)"
                        );
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Helper to apply merge during recovery
fn apply_merge(
    mt: &Memtable,
    key: Bytes,
    operand: Bytes,
    merge_operator: Option<&Arc<dyn MergeOperator>>,
) {
    let new_entry = match mt.get_entry(&key) {
        Some(Entry::Value(v)) => {
            if let Some(op) = merge_operator {
                let operands = [operand.as_ref()];
                if let Some(merged) = op.full_merge(&key, Some(&v), &operands) {
                    Entry::Value(Bytes::from(merged))
                } else {
                    // Merge failed, fallback to stack (overwrites value with merge list)
                    Entry::Merge(vec![operand])
                }
            } else {
                Entry::Merge(vec![operand])
            }
        }
        Some(Entry::Merge(ops)) => {
            let mut new_ops = ops.clone();
            if let Some(op) = merge_operator {
                let pushed = if let Some(last) = new_ops.last() {
                    if let Some(merged) = op.partial_merge(&key, last, &operand) {
                        new_ops.pop();
                        new_ops.push(Bytes::from(merged));
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                if !pushed {
                    new_ops.push(operand);
                }
            } else {
                new_ops.push(operand);
            }
            Entry::Merge(new_ops)
        }
        Some(Entry::Tombstone) => Entry::Merge(vec![operand]),
        None => Entry::Merge(vec![operand]),
    };
    mt.put_entry(key, new_entry);
}

/// Clean up old SSTable deletion queue
///
/// Files are queued for deletion after compaction but kept for 5 seconds
/// to allow concurrent readers to finish. This function periodically removes
/// files that are old enough.
///
/// This is called after each compaction to safely delete old SSTable files
/// without interfering with concurrent readers.
pub(crate) fn cleanup_old_deletions(
    pending_deletions: &Arc<Mutex<Vec<(PathBuf, std::time::Instant)>>>,
) {
    const DELETION_DELAY: std::time::Duration = std::time::Duration::from_secs(5);

    let mut pending = pending_deletions
        .lock()
        .expect("pending_deletions lock poisoned");
    let now = std::time::Instant::now();

    // Separate files into (ready_to_delete, still_pending)
    let mut still_pending = Vec::new();
    let mut ready_to_delete = Vec::new();

    for (path, queued_at) in pending.drain(..) {
        if now.duration_since(queued_at) >= DELETION_DELAY {
            ready_to_delete.push(path);
        } else {
            still_pending.push((path, queued_at));
        }
    }

    // Keep files that aren't old enough yet
    *pending = still_pending;
    drop(pending); // Release lock before doing I/O

    // Delete old files (outside the lock to avoid blocking)
    for path in ready_to_delete {
        if let Err(e) = std::fs::remove_file(&path) {
            // Log but don't fail - file might already be deleted
            warn!(path = ?path, error = %e, "Failed to delete old SSTable");
        } else {
            debug!(path = ?path, "Deleted old SSTable after safe delay");
        }
    }
}

/// Check if there is sufficient disk space
///
/// If min_disk_space_bytes is configured in options, verifies that the disk
/// containing the data directory has at least that much space available.
///
/// Returns an error if disk space is insufficient.
#[allow(dead_code)] // Reserved for future disk space monitoring
pub(crate) fn check_disk_space(options: &DBOptions) -> Result<()> {
    if let Some(min_space) = options.min_disk_space_bytes {
        use sysinfo::{DiskExt, System, SystemExt};

        // Get disk information (only refresh disks, not all system info)
        let mut sys = System::new();
        sys.refresh_disks_list();

        // Find the disk containing our data directory
        let data_dir = &options.data_dir;
        if let Some(disk) = sys
            .disks()
            .iter()
            .find(|d| data_dir.starts_with(d.mount_point()))
        {
            let available = disk.available_space();
            if available < min_space {
                return Err(DBError::DiskSpaceFull {
                    available,
                    required: min_space,
                });
            }
        }
    }
    Ok(())
}
