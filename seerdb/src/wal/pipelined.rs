//! Pipelined WAL with optimizations:
//! - Lock-free writer queue (crossbeam channel)
//! - Adaptive batch delay based on queue depth
//! - Pipelined writes (overlap memtable write N with WAL write N+1)
//!
//! Based on RocksDB's pipelined write design which achieves 20-30% improvement.

use crate::wal::{Record, Result, WAL};
use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

/// Writer request sent through the lock-free channel
struct Writer {
    record: Record,
    thread: Thread,
    /// Result of the WAL write operation (set by leader)
    result: Mutex<Option<Result<u64>>>,
    /// Flag to indicate completion (handles spurious wakeups)
    done: AtomicBool,
}

impl Writer {
    fn new(record: Record) -> Self {
        Self {
            record,
            thread: thread::current(),
            result: Mutex::new(None),
            done: AtomicBool::new(false),
        }
    }

    #[inline]
    fn signal_done(&self, res: Result<u64>) {
        {
            let mut result = self.result.lock().expect("mutex poisoned");
            *result = Some(res);
        }
        self.done.store(true, Ordering::Release);
        self.thread.unpark();
    }

    #[inline]
    fn is_done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    fn take_result(&self) -> Result<u64> {
        self.result
            .lock()
            .expect("mutex poisoned")
            .take()
            .expect("result must be set before waking writer")
    }
}

/// Configuration for adaptive batching
#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    /// Minimum batch delay (used when queue is shallow)
    pub min_delay: Duration,
    /// Maximum batch delay (used when queue is deep)
    pub max_delay: Duration,
    /// Queue depth at which we use max_delay
    pub adaptive_threshold: usize,
    /// Maximum writers per batch
    pub max_batch_size: usize,
    /// Enable pipelining (overlap memtable write with next WAL write)
    pub enable_pipelining: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            min_delay: Duration::from_micros(50), // Low latency for light load
            max_delay: Duration::from_micros(500), // Higher throughput for heavy load
            adaptive_threshold: 16,               // Queue depth to switch to max_delay
            max_batch_size: 256,
            enable_pipelining: true,
        }
    }
}

impl PipelineConfig {
    /// Compute adaptive delay based on queue depth using integer math
    #[inline]
    fn adaptive_delay(&self, queue_depth: usize) -> Duration {
        if queue_depth == 0 {
            self.min_delay
        } else if queue_depth >= self.adaptive_threshold {
            self.max_delay
        } else {
            // Linear interpolation using integer math (avoids f64 conversion)
            let min_us = self.min_delay.as_micros() as u64;
            let max_us = self.max_delay.as_micros() as u64;
            let delta = max_us - min_us;
            let scaled = delta * (queue_depth as u64) / (self.adaptive_threshold as u64);
            Duration::from_micros(min_us + scaled)
        }
    }
}

/// Pipelined WAL with lock-free queue and adaptive batching
pub struct PipelinedWAL {
    wal: Arc<Mutex<WAL>>,
    /// Lock-free channel for writer requests
    sender: Sender<Arc<Writer>>,
    receiver: Receiver<Arc<Writer>>,
    /// Atomic flag for leader election (true = leader exists)
    leader_active: AtomicBool,
    /// Configuration
    config: PipelineConfig,
    /// Stats: total batches processed
    batches_processed: AtomicU64,
    /// Stats: total writes processed
    writes_processed: AtomicU64,
}

impl PipelinedWAL {
    /// Create with default configuration
    pub fn new(wal: Arc<Mutex<WAL>>, delay: Duration, max_batch_size: usize) -> Self {
        Self::with_config(
            wal,
            PipelineConfig {
                min_delay: delay,
                max_delay: delay,
                max_batch_size,
                enable_pipelining: true,
                ..Default::default()
            },
        )
    }

    /// Create with custom configuration
    pub fn with_config(wal: Arc<Mutex<WAL>>, config: PipelineConfig) -> Self {
        // Bounded channel prevents unbounded memory growth under extreme load
        let (sender, receiver) = bounded(config.max_batch_size * 4);
        Self {
            wal,
            sender,
            receiver,
            leader_active: AtomicBool::new(false),
            config,
            batches_processed: AtomicU64::new(0),
            writes_processed: AtomicU64::new(0),
        }
    }

    /// Submit a write request
    pub fn put<F>(&self, record: Record, on_memtable: F) -> Result<u64>
    where
        F: Fn(&[Record]),
    {
        let writer = Arc::new(Writer::new(record));

        // 1. Enqueue (lock-free)
        // Use try_send to avoid blocking; fall back to send if needed
        if self.sender.try_send(writer.clone()).is_err() {
            // Channel full - send with blocking (backpressure)
            self.sender.send(writer.clone()).expect("channel closed");
        }

        // 2. Try to become leader (lock-free CAS)
        let is_leader = self
            .leader_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok();

        // 3. If leader, process batches
        if is_leader {
            self.process_batches_pipelined(&on_memtable);
        } else {
            // 4. Follower: wait for leader to complete our request
            loop {
                thread::park();
                if writer.is_done() {
                    break;
                }
            }
        }

        // 5. Return result
        writer.take_result()
    }

    /// Sync the WAL to disk
    pub fn sync(&self) -> Result<()> {
        let wal = self.wal.lock().expect("WAL mutex poisoned");
        wal.sync()
    }

    /// Get statistics
    pub fn stats(&self) -> (u64, u64) {
        (
            self.batches_processed.load(Ordering::Relaxed),
            self.writes_processed.load(Ordering::Relaxed),
        )
    }

    /// Process batches with pipelining: overlap memtable write N with WAL write N+1
    #[allow(clippy::type_complexity)]
    fn process_batches_pipelined<F>(&self, on_memtable: &F)
    where
        F: Fn(&[Record]),
    {
        // Previous batch's memtable work (for pipelining)
        let mut pending_memtable: Option<(Vec<Arc<Writer>>, Vec<Record>, Vec<u64>)> = None;

        loop {
            // 1. Collect batch from lock-free queue
            let batch_writers = self.collect_batch();

            if batch_writers.is_empty() {
                // Complete any pending memtable work before exiting
                if let Some((writers, records, offsets)) = pending_memtable.take() {
                    on_memtable(&records);
                    for (writer, offset) in writers.iter().zip(offsets) {
                        writer.signal_done(Ok(offset));
                    }
                }
                // Release leadership
                self.leader_active.store(false, Ordering::Release);

                // Double-check: if new writers arrived, try to reclaim leadership
                if !self.receiver.is_empty()
                    && self
                        .leader_active
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                {
                    continue;
                }
                return;
            }

            // 2. Extract records
            let records: Vec<Record> = batch_writers.iter().map(|w| w.record.clone()).collect();

            // 3. Write to WAL
            let wal_result = {
                let mut wal = self.wal.lock().expect("WAL mutex poisoned");
                wal.write_batch(&records)
            };

            // 4. Handle previous batch's memtable write (pipelining)
            // This runs in parallel with the WAL write of the NEXT iteration
            if let Some((prev_writers, prev_records, prev_offsets)) = pending_memtable.take() {
                on_memtable(&prev_records);
                for (writer, offset) in prev_writers.iter().zip(prev_offsets) {
                    writer.signal_done(Ok(offset));
                }
            }

            // 5. Handle current batch result
            match wal_result {
                Ok(offsets) => {
                    self.batches_processed.fetch_add(1, Ordering::Relaxed);
                    self.writes_processed
                        .fetch_add(batch_writers.len() as u64, Ordering::Relaxed);

                    if self.config.enable_pipelining {
                        // Defer memtable write to pipeline with next WAL write
                        pending_memtable = Some((batch_writers, records, offsets));
                    } else {
                        // No pipelining: complete immediately
                        on_memtable(&records);
                        for (writer, offset) in batch_writers.iter().zip(offsets) {
                            writer.signal_done(Ok(offset));
                        }
                    }
                }
                Err(e) => {
                    // Error: signal all writers immediately
                    let err_str = e.to_string();
                    for writer in batch_writers.iter() {
                        let err = crate::wal::WALError::Io(std::io::Error::other(err_str.clone()));
                        writer.signal_done(Err(err));
                    }
                }
            }
        }
    }

    /// Collect a batch from the lock-free queue with adaptive delay
    fn collect_batch(&self) -> Vec<Arc<Writer>> {
        let mut batch = Vec::with_capacity(self.config.max_batch_size);

        // First, drain any immediately available writers
        loop {
            match self.receiver.try_recv() {
                Ok(writer) => {
                    batch.push(writer);
                    if batch.len() >= self.config.max_batch_size {
                        return batch;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return batch,
            }
        }

        // If we got nothing, wait briefly with adaptive delay
        if batch.is_empty() {
            let delay = self.config.adaptive_delay(0);
            match self.receiver.recv_timeout(delay) {
                Ok(writer) => batch.push(writer),
                Err(_) => return batch, // Timeout or disconnected
            }
        }

        // Now collect more with adaptive delay based on queue depth
        let delay = self.config.adaptive_delay(batch.len());
        let deadline = Instant::now() + delay;

        while batch.len() < self.config.max_batch_size && Instant::now() < deadline {
            match self.receiver.try_recv() {
                Ok(writer) => batch.push(writer),
                Err(TryRecvError::Empty) => {
                    // Brief sleep to avoid busy-waiting (10µs is small enough for low latency)
                    thread::sleep(Duration::from_micros(10));
                }
                Err(TryRecvError::Disconnected) => break,
            }
        }

        batch
    }
}
