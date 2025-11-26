#![no_main]

use libfuzzer_sys::fuzz_target;
use libfuzzer_sys::arbitrary::{self, Arbitrary, Unstructured};
use seerdb::{DB, DBOptions, Snapshot};
use tempfile::TempDir;
use bytes::Bytes;
use std::error::Error;

#[derive(Debug, Clone)]
enum DBOp {
    Put { key: Vec<u8>, value: Vec<u8> },
    Get { key: Vec<u8> },
    Delete { key: Vec<u8> },
    Range { start: Vec<u8>, end: Option<Vec<u8>> },
    Flush,
    Snapshot,
    SnapshotConsistent,
    Iter,
    Prefix { prefix: Vec<u8> },
}

impl<'a> Arbitrary<'a> for DBOp {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let op_type: u8 = u.int_in_range(0..=8)?;

        match op_type {
            0 => {
                let key = u.arbitrary::<Vec<u8>>()?;
                let value = u.arbitrary::<Vec<u8>>()?;
                Ok(DBOp::Put { key, value })
            }
            1 => {
                let key = u.arbitrary::<Vec<u8>>()?;
                Ok(DBOp::Get { key })
            }
            2 => {
                let key = u.arbitrary::<Vec<u8>>()?;
                Ok(DBOp::Delete { key })
            }
            3 => {
                let start = u.arbitrary::<Vec<u8>>()?;
                let has_end: bool = u.arbitrary()?;
                let end = if has_end {
                    Some(u.arbitrary::<Vec<u8>>()?)
                } else {
                    None
                };
                Ok(DBOp::Range { start, end })
            }
            4 => Ok(DBOp::Flush),
            5 => Ok(DBOp::Snapshot),
            6 => Ok(DBOp::SnapshotConsistent),
            7 => Ok(DBOp::Iter),
            8 => {
                let prefix = u.arbitrary::<Vec<u8>>()?;
                Ok(DBOp::Prefix { prefix })
            }
            _ => unreachable!(),
        }
    }
}

fuzz_target!(|data: &[u8]| {
    // Parse fuzzed data into a sequence of DB operations
    let mut u = Unstructured::new(data);

    let ops: Vec<DBOp> = match u.arbitrary() {
        Ok(ops) => ops,
        Err(_) => return,
    };

    // Skip if no operations
    if ops.is_empty() {
        return;
    }

    // Create temporary database
    let temp_dir = match TempDir::new() {
        Ok(dir) => dir,
        Err(_) => return,
    };

    let mut options = DBOptions::default();
    options.data_dir = temp_dir.path().to_path_buf();

    let db: DB = match DB::open(options) {
        Ok(db) => db,
        Err(_) => return,
    };

    // Execute all operations
    // None should panic - all errors should be graceful
    let mut snapshots: Vec<Snapshot> = Vec::new();

    for op in ops {
        match op {
            DBOp::Put { key, value } => {
                // Limit size to avoid OOM
                if key.len() > 64 * 1024 || value.len() > 1024 * 1024 {
                    continue;
                }
                let _ = db.put(&key, &value);
            }
            DBOp::Get { key } => {
                if key.len() > 64 * 1024 {
                    continue;
                }
                let _ = db.get(&key);
            }
            DBOp::Delete { key } => {
                if key.len() > 64 * 1024 {
                    continue;
                }
                let _ = db.delete(&key);
            }
            DBOp::Range { start, end } => {
                if start.len() > 64 * 1024 {
                    continue;
                }
                if let Some(ref e) = end {
                    if e.len() > 64 * 1024 {
                        continue;
                    }
                }
                if let Ok(iter) = db.range(&start, end.as_deref()) {
                    // Consume iterator (limit to avoid OOM)
                    for (i, _item) in iter.enumerate() {
                        let _: Result<(Bytes, Bytes), Box<dyn Error>> = _item;
                        if i > 1000 {
                            break;
                        }
                    }
                }
            }
            DBOp::Flush => {
                let _ = db.flush();
            }
            DBOp::Snapshot => {
                if let Ok(snap) = db.snapshot() {
                    // Limit number of snapshots to avoid memory explosion
                    if snapshots.len() < 10 {
                        snapshots.push(snap);
                    }
                }
            }
            DBOp::SnapshotConsistent => {
                if let Ok(snap) = db.snapshot_consistent() {
                    if snapshots.len() < 10 {
                        snapshots.push(snap);
                    }
                }
            }
            DBOp::Iter => {
                if let Ok(iter) = db.iter() {
                    // Consume iterator (limit to avoid OOM)
                    for (i, _item) in iter.enumerate() {
                        let _: Result<(Bytes, Bytes), Box<dyn Error>> = _item;
                        if i > 1000 {
                            break;
                        }
                    }
                }
            }
            DBOp::Prefix { prefix } => {
                if prefix.len() > 64 * 1024 {
                    continue;
                }
                if let Ok(iter) = db.prefix(&prefix) {
                    // Consume iterator (limit to avoid OOM)
                    for (i, _item) in iter.enumerate() {
                        let _: Result<(Bytes, Bytes), Box<dyn Error>> = _item;
                        if i > 1000 {
                            break;
                        }
                    }
                }
            }
        }
    }

    // Optionally read from snapshots
    for snap in &snapshots {
        // Try reading a few keys from each snapshot
        let _: Result<Option<Bytes>, _> = snap.get(&[0u8]);
        let _: Result<Option<Bytes>, _> = snap.get(&[255u8]);
    }

    // Database is automatically cleaned up when temp_dir is dropped
});
