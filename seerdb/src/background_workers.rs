// Background worker threads for async operations
//
// This module contains the background worker implementation for:
// - WAL writer: Batches writes to the write-ahead log
// - Flush worker: Converts memtables to SSTables
// - Compaction worker: Merges SSTables to reduce read amplification

use crate::compaction::CompactionFilter;
use crate::compaction::LSMTree;
use crate::memtable::{Entry, Memtable};
use crate::metrics::MetricsCollector;
use crate::sstable::SSTableBuilder;
#[cfg(feature = "object-store")]
use crate::storage::Storage;
use crate::vlog::VLog;
use crate::wal::WAL;
use arc_swap::ArcSwap;
use bytes::Bytes;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;
use tracing::{error, info};

use crate::db::Result;

/// Number of memtable partitions (must match db.rs)
const NUM_PARTITIONS: usize = 16;

/// Messages sent to the background compaction worker thread
#[derive(Debug)]
pub(crate) enum CompactionTask {
    /// Compact a specific level
    CompactLevel(usize),
    /// Shutdown signal
    Shutdown,
}

/// Messages sent to the background flush worker thread
#[derive(Debug)]
pub(crate) enum FlushTask {
    /// Flush the memtable to SSTable
    Flush,
    /// Shutdown signal
    Shutdown,
}

/// Static compaction method for background worker thread
/// This is called from the worker thread without &self
///
/// Supports tiered storage: when `cold_tier_level` is set, SSTables at that level
/// and above are written to `cold_storage_backend` instead of local disk.
pub(crate) fn run_compaction(
    lsm: &Arc<ArcSwap<LSMTree>>,
    lsm_mutex: &Arc<Mutex<()>>,
    sstable_counter: &Arc<Mutex<u64>>,
    data_dir: &Path,
    level_num: usize,
    metrics: &Arc<MetricsCollector>,
    max_flushed_seq: &Arc<AtomicU64>,
    pending_deletions: &Arc<Mutex<Vec<(PathBuf, std::time::Instant)>>>,
    filter: &Option<Arc<dyn CompactionFilter>>,
    #[cfg(feature = "object-store")] storage_backend: &Option<Arc<dyn Storage>>,
    snapshot_tracker: &Arc<crate::types::SnapshotTracker>,
    #[cfg(feature = "object-store")] cold_tier_level: Option<usize>,
    #[cfg(feature = "object-store")] cold_storage_backend: &Option<Arc<dyn Storage>>,
) -> Result<()> {
    use crate::db::DB;

    #[cfg(feature = "object-store")]
    {
        DB::do_compact_level(
            lsm,
            lsm_mutex,
            sstable_counter,
            data_dir,
            level_num,
            metrics,
            max_flushed_seq,
            pending_deletions,
            filter,
            storage_backend,
            snapshot_tracker,
            cold_tier_level,
            cold_storage_backend,
        )
    }

    #[cfg(not(feature = "object-store"))]
    {
        DB::do_compact_level(
            lsm,
            lsm_mutex,
            sstable_counter,
            data_dir,
            level_num,
            metrics,
            max_flushed_seq,
            pending_deletions,
            filter,
            snapshot_tracker,
        )
    }
}

/// Static flush method for background worker thread
/// This is called from the worker thread without &self
///
/// NOTE: Memtable swap already happened in try_swap_memtable() before signal was sent.
/// This method just builds the SSTable from immutable_memtable (slow part).
pub(crate) fn run_background_flush_partitioned(
    _memtables: &Arc<[ArcSwap<Memtable>; NUM_PARTITIONS]>,
    immutable_memtables: &Arc<ArcSwap<Option<Arc<Vec<Arc<Memtable>>>>>>,
    wal: &Arc<Mutex<WAL>>,
    lsm: &Arc<ArcSwap<LSMTree>>,
    lsm_mutex: &Arc<Mutex<()>>,
    vlog: &Arc<Mutex<Option<VLog>>>,
    sstable_counter: &Arc<Mutex<u64>>,
    data_dir: &Path,
    metrics: &Arc<MetricsCollector>,
    _memtable_capacity: usize,
    vlog_threshold: Option<usize>,
    flush_mutex: &Arc<Mutex<()>>,
    max_flushed_seq: &Arc<AtomicU64>,
    compaction_tx: &Option<Sender<CompactionTask>>,
    #[cfg(feature = "object-store")] storage_backend: &Option<Arc<dyn Storage>>,
) -> Result<()> {
    // Serialize all flushes to prevent concurrent SSTable builds
    let _flush_lock = flush_mutex.lock().expect("Flush mutex poisoned");

    let flush_start = Instant::now();

    // Check if there are immutable_memtables to flush (LOCK-FREE!)
    let immut_arc = immutable_memtables.load();
    let has_immutable = immut_arc.is_some();

    if !has_immutable {
        // No immutable memtables - another thread might have already flushed them
        return Ok(());
    }

    // Generate SSTable filename
    let mut counter = sstable_counter
        .lock()
        .expect("SSTable counter mutex poisoned");
    let flush_sequence = *counter; // Capture sequence for this background flush
    let sstable_path = data_dir.join(format!("L0_{:06}.sst", *counter));
    *counter += 1;
    drop(counter);

    // Build SSTable from immutable memtable partitions (slow part - this is why it's in background) (LOCK-FREE!)
    // Keep Arc alive and get reference to the Vec
    let immutable_partitions_arc = immut_arc
        .as_ref()
        .as_ref()
        .expect("Immutable partitions should be present");

    // Collect entries from ALL partitions and sort
    let mut all_entries: Vec<(Bytes, Entry)> = Vec::new();
    for partition_mt in immutable_partitions_arc.iter() {
        for (key, entry) in partition_mt.iter_entries() {
            all_entries.push((key, entry));
        }
    }
    all_entries.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));

    // Check if we're using cloud storage (feature-gated)
    #[cfg(feature = "object-store")]
    let use_cloud_storage = storage_backend.is_some();
    #[cfg(not(feature = "object-store"))]
    let use_cloud_storage = false;

    // Build SSTable with optional vLog support
    let mut vlog_guard = vlog.lock().expect("vLog mutex poisoned");

    if let (Some(threshold), Some(ref mut vlog_ref)) = (vlog_threshold, vlog_guard.as_mut()) {
        // KV separation enabled - use vLog for large values

        if use_cloud_storage {
            #[cfg(feature = "object-store")]
            {
                // Cloud storage + vLog: use buffered builder
                let mut builder = SSTableBuilder::new_buffered()
                    .with_vlog_threshold(threshold)
                    .with_max_sequence(flush_sequence);

                for (key, entry) in &all_entries {
                    match entry {
                        Entry::Value(value) => {
                            builder.add_with_vlog(key.clone(), value.clone(), vlog_ref)?;
                        }
                        Entry::Tombstone => {
                            builder.add_tombstone(key.clone())?;
                        }
                        Entry::Merge(ops) => {
                            // Store merge operands as individual merge entries
                            for op in ops {
                                builder.add_merge(key.clone(), op.clone())?;
                            }
                        }
                    }
                }

                // ALWAYS sync vLog after flush
                vlog_ref.sync()?;

                // Build SSTable in memory
                let bytes = builder.finish_to_bytes()?;

                // Write to local disk (single syscall)
                std::fs::write(&sstable_path, &bytes)?;

                // Upload to cloud storage
                if let Some(ref backend) = storage_backend {
                    backend.write_sstable(&sstable_path, &bytes)?;
                    info!(
                        path = ?sstable_path,
                        size_bytes = bytes.len(),
                        "SSTable with vLog uploaded to cloud storage (background flush)"
                    );
                }
            }
        } else {
            // No cloud storage + vLog: use traditional SSTableBuilder
            let mut builder = SSTableBuilder::create(&sstable_path)?
                .with_vlog_threshold(threshold)
                .with_max_sequence(flush_sequence);

            for (key, entry) in &all_entries {
                match entry {
                    Entry::Value(value) => {
                        builder.add_with_vlog(key.clone(), value.clone(), vlog_ref)?;
                    }
                    Entry::Tombstone => {
                        builder.add_tombstone(key.clone())?;
                    }
                    Entry::Merge(ops) => {
                        for op in ops {
                            builder.add_merge(key.clone(), op.clone())?;
                        }
                    }
                }
            }

            builder.finish()?;

            // Sync vLog after flush
            vlog_ref.sync()?;
        }
    } else {
        // No KV separation - traditional flush
        drop(vlog_guard);

        if use_cloud_storage {
            #[cfg(feature = "object-store")]
            {
                // Use buffered builder when cloud storage is enabled
                let mut builder = SSTableBuilder::new_buffered().with_max_sequence(flush_sequence);

                for (key, entry) in &all_entries {
                    match entry {
                        Entry::Value(value) => {
                            builder.add(key.clone(), value.clone())?;
                        }
                        Entry::Tombstone => {
                            builder.add_tombstone(key.clone())?;
                        }
                        Entry::Merge(ops) => {
                            // Store merge operands as individual merge entries
                            for op in ops {
                                builder.add_merge(key.clone(), op.clone())?;
                            }
                        }
                    }
                }

                // Build SSTable in memory
                let bytes = builder.finish_to_bytes()?;

                // Write to local disk (single syscall)
                std::fs::write(&sstable_path, &bytes)?;

                // Upload to cloud storage
                if let Some(ref backend) = storage_backend {
                    backend.write_sstable(&sstable_path, &bytes)?;
                    info!(
                        path = ?sstable_path,
                        size_bytes = bytes.len(),
                        "SSTable uploaded to cloud storage (background flush)"
                    );
                }
            }
        } else {
            // No cloud storage - use traditional SSTableBuilder
            let mut builder =
                SSTableBuilder::create(&sstable_path)?.with_max_sequence(flush_sequence);
            for (key, entry) in &all_entries {
                match entry {
                    Entry::Value(value) => {
                        builder.add(key.clone(), value.clone())?;
                    }
                    Entry::Tombstone => {
                        builder.add_tombstone(key.clone())?;
                    }
                    Entry::Merge(ops) => {
                        for op in ops {
                            builder.add_merge(key.clone(), op.clone())?;
                        }
                    }
                }
            }
            builder.finish()?;
        }
    }
    // Arc automatically dropped (lock-free, no explicit drop needed!)

    let size = std::fs::metadata(&sstable_path)?.len();

    // Track physical bytes written
    metrics.record_physical_bytes(size);

    // CRITICAL FIX (Bug #7c): Serialize LSM tree updates to prevent ABA race
    // Hold mutex during read-modify-write to ensure atomicity
    {
        let _lsm_lock = lsm_mutex.lock().expect("LSM mutex poisoned");

        // Add to LSM tree L0 (serialized)
        let mut lsm_clone = (**lsm.load()).clone();
        lsm_clone.add_l0_sstable(sstable_path.clone(), size);
        lsm.store(Arc::new(lsm_clone));

        // Lock released here (automatic drop)
    }

    // Clear immutable memtables + WAL after successful flush (LOCK-FREE!)
    immutable_memtables.store(Arc::new(None));

    {
        let mut wal_guard = wal.lock().expect("WAL mutex poisoned");
        wal_guard.clear()?;
    }

    // CRITICAL FIX (Bug #7d): Update max_flushed_seq to allow compaction of this SSTable
    // This MUST happen after immutable_memtables is cleared to prevent data loss
    // Without this, compaction will skip all background-flushed SSTables forever!
    // Use fetch_max to handle out-of-order flush completions (only update if new value is greater)
    max_flushed_seq.fetch_max(flush_sequence, Ordering::SeqCst);

    let flush_duration_ms = flush_start.elapsed().as_millis();
    info!(
        duration_ms = flush_duration_ms,
        sstable_path = ?sstable_path,
        sstable_size_bytes = size,
        partitions_merged = NUM_PARTITIONS,
        "Background partitioned memtable flush complete"
    );

    // Record flush metric
    metrics.record_flush();

    // Check if compaction is needed (LOCK-FREE!)
    if let Some(level_num) = lsm.load().needs_compaction() {
        info!(
            level = level_num,
            "Compaction triggered by background flush"
        );
        if let Some(tx) = compaction_tx {
            // Background compaction: send signal (non-blocking)
            let _ = tx.send(CompactionTask::CompactLevel(level_num));
        }
    }

    Ok(())
}

/// Spawn background compaction worker thread if enabled
///
/// Returns (Option<Sender>, Option<JoinHandle>) for sending tasks and joining the thread
pub(crate) fn spawn_compaction_worker(
    enabled: bool,
    lsm: Arc<ArcSwap<LSMTree>>,
    lsm_mutex: Arc<Mutex<()>>,
    sstable_counter: Arc<Mutex<u64>>,
    data_dir: PathBuf,
    metrics: Arc<MetricsCollector>,
    max_flushed_seq: Arc<AtomicU64>,
    compaction_healthy: Arc<AtomicBool>,
    pending_deletions: Arc<Mutex<Vec<(PathBuf, Instant)>>>,
    filter: Option<Arc<dyn CompactionFilter>>,
    #[cfg(feature = "object-store")] storage_backend: Option<Arc<dyn Storage>>,
    snapshot_tracker: Arc<crate::types::SnapshotTracker>,
    #[cfg(feature = "object-store")] cold_tier_level: Option<usize>,
    #[cfg(feature = "object-store")] cold_storage_backend: Option<Arc<dyn Storage>>,
) -> (Option<Sender<CompactionTask>>, Option<JoinHandle<()>>) {
    if !enabled {
        return (None, None);
    }

    let (tx, rx) = channel::<CompactionTask>();

    // Spawn compaction worker thread with panic detection
    let worker = thread::Builder::new()
        .name("compaction-worker".to_string())
        .spawn(move || {
            // Wrap in catch_unwind to detect panics and mark health status
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                while let Ok(task) = rx.recv() {
                    match task {
                        CompactionTask::CompactLevel(level_num) => {
                            // Perform compaction
                            #[cfg(feature = "object-store")]
                            let res = run_compaction(
                                &lsm,
                                &lsm_mutex,
                                &sstable_counter,
                                &data_dir,
                                level_num,
                                &metrics,
                                &max_flushed_seq,
                                &pending_deletions,
                                &filter,
                                &storage_backend,
                                &snapshot_tracker,
                                cold_tier_level,
                                &cold_storage_backend,
                            );

                            #[cfg(not(feature = "object-store"))]
                            let res = run_compaction(
                                &lsm,
                                &lsm_mutex,
                                &sstable_counter,
                                &data_dir,
                                level_num,
                                &metrics,
                                &max_flushed_seq,
                                &pending_deletions,
                                &filter,
                                &snapshot_tracker,
                            );

                            if let Err(e) = res {
                                error!(error = %e, level = level_num, "Background compaction failed");
                            }
                        }
                        CompactionTask::Shutdown => {
                            // Exit worker thread
                            break;
                        }
                    }
                }
            }));

            // If panicked, mark as unhealthy
            if result.is_err() {
                error!("Compaction worker thread panicked");
                compaction_healthy.store(false, Ordering::SeqCst);
            }
        })
        .expect("Failed to spawn compaction worker thread");

    (Some(tx), Some(worker))
}

/// Spawn background flush worker thread if enabled
///
/// Returns (Option<Sender>, Option<JoinHandle>) for sending tasks and joining the thread
pub(crate) fn spawn_flush_worker(
    enabled: bool,
    memtables: Arc<[ArcSwap<Memtable>; NUM_PARTITIONS]>,
    immutable_memtables: Arc<ArcSwap<Option<Arc<Vec<Arc<Memtable>>>>>>,
    wal: Arc<Mutex<WAL>>,
    lsm: Arc<ArcSwap<LSMTree>>,
    lsm_mutex: Arc<Mutex<()>>,
    vlog: Arc<Mutex<Option<VLog>>>,
    sstable_counter: Arc<Mutex<u64>>,
    data_dir: PathBuf,
    metrics: Arc<MetricsCollector>,
    memtable_capacity: usize,
    vlog_threshold: Option<usize>,
    flush_mutex: Arc<Mutex<()>>,
    max_flushed_seq: Arc<AtomicU64>,
    flush_healthy: Arc<AtomicBool>,
    compaction_tx: Option<Sender<CompactionTask>>,
    #[cfg(feature = "object-store")] storage_backend: Option<Arc<dyn Storage>>,
) -> (Option<Sender<FlushTask>>, Option<JoinHandle<()>>) {
    if !enabled {
        return (None, None);
    }

    let (tx, rx) = channel::<FlushTask>();

    // Spawn flush worker thread with panic detection
    let worker = thread::Builder::new()
        .name("flush-worker".to_string())
        .spawn(move || {
            // Wrap in catch_unwind to detect panics and mark health status
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                while let Ok(task) = rx.recv() {
                    match task {
                        FlushTask::Flush => {
                            // Perform background flush (now with partitioned memtables)
                            #[cfg(feature = "object-store")]
                            let res = run_background_flush_partitioned(
                                &memtables,
                                &immutable_memtables,
                                &wal,
                                &lsm,
                                &lsm_mutex,
                                &vlog,
                                &sstable_counter,
                                &data_dir,
                                &metrics,
                                memtable_capacity,
                                vlog_threshold,
                                &flush_mutex,
                                &max_flushed_seq,
                                &compaction_tx,
                                &storage_backend,
                            );

                            #[cfg(not(feature = "object-store"))]
                            let res = run_background_flush_partitioned(
                                &memtables,
                                &immutable_memtables,
                                &wal,
                                &lsm,
                                &lsm_mutex,
                                &vlog,
                                &sstable_counter,
                                &data_dir,
                                &metrics,
                                memtable_capacity,
                                vlog_threshold,
                                &flush_mutex,
                                &max_flushed_seq,
                                &compaction_tx,
                            );

                            if let Err(e) = res {
                                error!(error = %e, "Background flush failed");
                            }
                        }
                        FlushTask::Shutdown => {
                            // Exit worker thread
                            break;
                        }
                    }
                }
            }));

            // If panicked, mark as unhealthy
            if result.is_err() {
                error!("Flush worker thread panicked");
                flush_healthy.store(false, Ordering::SeqCst);
            }
        })
        .expect("Failed to spawn flush worker thread");

    (Some(tx), Some(worker))
}

// spawn_wal_writer and flush_and_ack removed (deprecated by PipelinedWAL)
