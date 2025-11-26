// Write-Ahead Log (WAL) implementation
// Provides durability guarantees for memtable operations

pub mod pipelined;
pub mod reader;
pub mod record;

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;

pub use pipelined::{PipelineConfig, PipelinedWAL};
pub use reader::WALReader;
pub use record::{BatchOp, Record};

// WAL file format magic number: "WLOG"
const MAGIC: u32 = 0x574C4F47;
// Version 2 adds per-record CRC32C checksums
const VERSION: u32 = 0x00000002;
const HEADER_SIZE: u64 = 8; // magic (4) + version (4)

#[derive(Debug, Error)]
pub enum WALError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Record error: {0}")]
    Record(#[from] record::RecordError),

    #[error("Invalid WAL format: bad magic or version")]
    InvalidFormat,
}

pub type Result<T> = std::result::Result<T, WALError>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SyncPolicy {
    /// Sync data and metadata on every write (safest, slowest)
    SyncAll,
    /// Sync data only on every write (safe, faster)
    SyncData,
    /// No sync, rely on OS (fastest, least safe)
    None,
}

/// Recovery mode for WAL replay during DB::open()
///
/// Controls how the database handles corruption or truncation in the WAL.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RecoveryMode {
    /// Stop on any corruption and return an error
    ///
    /// Use this mode when data integrity is critical and you want to
    /// investigate corruption before proceeding.
    Strict,

    /// Skip corrupted/truncated records and continue recovery (default)
    ///
    /// Recovers all valid records up to the first corruption point.
    /// This is the recommended mode for most use cases, as it maximizes
    /// data recovery after crashes or power failures.
    #[default]
    BestEffort,
}

/// Configuration for WAL batching
#[derive(Debug, Clone, Copy)]
pub struct BatchConfig {
    /// Maximum batch size in bytes before forcing flush (default: 4MB)
    pub max_batch_size: usize,
    /// Maximum time to wait before forcing flush (default: 50ms)
    pub max_batch_timeout: Duration,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            // Increased from 8MB to 32MB to reduce syscall frequency (profiling showed 47% time in syscalls)
            max_batch_size: 32 * 1024 * 1024, // 32MB
            // Reduced from 100ms to 10ms for better batching while maintaining low latency
            max_batch_timeout: Duration::from_millis(10), // 10ms
        }
    }
}

/// Write-Ahead Log writer with automatic batching
pub struct WAL {
    file: Arc<Mutex<File>>,
    path: PathBuf,
    offset: u64,
    sync_policy: SyncPolicy,
    // Batching fields
    batch: Vec<Record>,
    batch_size_bytes: usize,
    batch_config: BatchConfig,
    last_flush: Instant,
}

impl WAL {
    /// Create a new WAL file with default batch configuration
    pub fn create(path: impl AsRef<Path>, sync_policy: SyncPolicy) -> Result<Self> {
        Self::create_with_batch_config(path, sync_policy, BatchConfig::default())
    }

    /// Create a new WAL file with custom batch configuration
    pub fn create_with_batch_config(
        path: impl AsRef<Path>,
        sync_policy: SyncPolicy,
        batch_config: BatchConfig,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        // Write header: [magic: u32][version: u32]
        file.write_all(&MAGIC.to_le_bytes())?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.sync_all()?;

        Ok(Self {
            file: Arc::new(Mutex::new(file)),
            path,
            offset: HEADER_SIZE,
            sync_policy,
            batch: Vec::new(),
            batch_size_bytes: 0,
            batch_config,
            last_flush: Instant::now(),
        })
    }

    /// Open an existing WAL file with default batch configuration
    pub fn open(path: impl AsRef<Path>, sync_policy: SyncPolicy) -> Result<Self> {
        Self::open_with_batch_config(path, sync_policy, BatchConfig::default())
    }

    /// Open an existing WAL file with custom batch configuration
    pub fn open_with_batch_config(
        path: impl AsRef<Path>,
        sync_policy: SyncPolicy,
        batch_config: BatchConfig,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;

        // Read and validate header
        let mut header = [0u8; 8];
        file.read_exact(&mut header)?;

        let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);

        if magic != MAGIC || version != VERSION {
            return Err(WALError::InvalidFormat);
        }

        // Get file size for offset
        let offset = file.metadata()?.len();

        Ok(Self {
            file: Arc::new(Mutex::new(file)),
            path,
            offset,
            sync_policy,
            batch: Vec::new(),
            batch_size_bytes: 0,
            batch_config,
            last_flush: Instant::now(),
        })
    }

    /// Write a record to the WAL with automatic batching
    pub fn write(&mut self, record: &Record) -> Result<u64> {
        let encoded_size = record.encoded_len();
        let record_offset = self.offset + self.batch_size_bytes as u64;

        // Add to batch
        self.batch.push(record.clone());
        self.batch_size_bytes += encoded_size;

        // Check if we should flush
        let should_flush = self.batch_size_bytes >= self.batch_config.max_batch_size
            || self.last_flush.elapsed() >= self.batch_config.max_batch_timeout;

        if should_flush {
            self.flush_batch()?;
        }

        Ok(record_offset)
    }

    /// Force flush any pending batch
    pub fn flush_batch(&mut self) -> Result<()> {
        if self.batch.is_empty() {
            return Ok(());
        }

        // Write all batched records
        let records: Vec<Record> = self.batch.drain(..).collect();
        self.write_batch(&records)?;

        self.batch_size_bytes = 0;
        self.last_flush = Instant::now();

        Ok(())
    }

    /// Write a batch of records
    ///
    /// Record format (v2): `[crc32c:4 LE][len:4 BE][record_data:len]`
    /// CRC32C is computed over `[len:4 BE][record_data]`
    pub fn write_batch(&mut self, records: &[Record]) -> Result<Vec<u64>> {
        let mut offsets = Vec::with_capacity(records.len());

        {
            let mut file = self.file.lock().expect("WAL file mutex poisoned");

            // OPTIMIZATION: Accumulate all records into a single buffer to reduce syscalls
            // Previous: N records = N write_all() calls (47% of total time in syscalls!)
            // Now: N records = 1 write_all() call (massive reduction in syscall overhead)
            let mut batch_buffer = Vec::new();

            for record in records {
                let encoded = record.encode();
                offsets.push(self.offset);

                // Build the record frame: [len:4 BE][record_data]
                let len = encoded.len() as u32;
                let len_bytes = len.to_be_bytes();

                // Compute CRC32C over [len:4 BE][record_data]
                let crc = crc32c::crc32c_append(crc32c::crc32c(&len_bytes), &encoded);

                // Write: [crc32c:4 LE][len:4 BE][record_data]
                batch_buffer.extend_from_slice(&crc.to_le_bytes());
                batch_buffer.extend_from_slice(&len_bytes);
                batch_buffer.extend_from_slice(&encoded);
                self.offset += (4 + 4 + encoded.len()) as u64; // crc(4) + len(4) + data
            }

            // Single syscall for entire batch (key optimization!)
            if !batch_buffer.is_empty() {
                file.write_all(&batch_buffer)?;
            }

            // Sync once at the end for batch
            match self.sync_policy {
                SyncPolicy::SyncAll => file.sync_all()?,
                SyncPolicy::SyncData => file.sync_data()?,
                SyncPolicy::None => {}
            }
        }

        Ok(offsets)
    }

    /// Get the current offset (end of file)
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Get the file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Sync the WAL to disk
    pub fn sync(&self) -> Result<()> {
        let file = self.file.lock().expect("WAL file mutex poisoned");
        file.sync_all()?;
        Ok(())
    }

    /// Clear the WAL (truncate to zero)
    ///
    /// This should be called after a successful flush to remove committed data.
    pub fn clear(&mut self) -> Result<()> {
        // Flush any pending batch first
        self.flush_batch()?;

        let mut file = self.file.lock().expect("WAL file mutex poisoned");
        // CRITICAL FIX (Bug #8): Truncate to HEADER_SIZE (not 0!) to preserve magic + version
        // WALReader::open() expects a valid 8-byte header, truncating to 0 causes UnexpectedEof
        file.set_len(HEADER_SIZE)?;

        // CRITICAL: Seek to HEADER_SIZE after truncating
        // set_len() doesn't move the file cursor, so subsequent writes would be at the wrong position
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(HEADER_SIZE))?;

        file.sync_all()?;
        // Reset offset to after header (not 0!)
        self.offset = HEADER_SIZE;
        Ok(())
    }
}

impl Drop for WAL {
    fn drop(&mut self) {
        // Flush any pending batch when WAL is dropped
        let _ = self.flush_batch();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use tempfile::tempdir;

    #[test]
    fn test_wal_create_and_write() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut wal = WAL::create(&wal_path, SyncPolicy::SyncAll).unwrap();

        let record = Record::Put {
            key: Bytes::from("key1"),
            value: Bytes::from("value1"),
            seq: 1,
        };

        let offset = wal.write(&record).unwrap();
        assert_eq!(offset, HEADER_SIZE); // First record starts after 8-byte header

        assert!(wal_path.exists());
    }

    #[test]
    fn test_wal_write_batch() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut wal = WAL::create(&wal_path, SyncPolicy::SyncData).unwrap();

        let records = vec![
            Record::Put {
                key: Bytes::from("key1"),
                value: Bytes::from("value1"),
                seq: 1,
            },
            Record::Put {
                key: Bytes::from("key2"),
                value: Bytes::from("value2"),
                seq: 2,
            },
            Record::Delete {
                key: Bytes::from("key1"),
                seq: 3,
            },
        ];

        let offsets = wal.write_batch(&records).unwrap();
        assert_eq!(offsets.len(), 3);
        assert_eq!(offsets[0], HEADER_SIZE); // First record starts after header
    }

    #[test]
    fn test_wal_reopen() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let mut wal = WAL::create(&wal_path, SyncPolicy::SyncAll).unwrap();
            let record = Record::Put {
                key: Bytes::from("key1"),
                value: Bytes::from("value1"),
                seq: 1,
            };
            wal.write(&record).unwrap();
        }

        let wal = WAL::open(&wal_path, SyncPolicy::SyncAll).unwrap();
        assert!(wal.offset() > 0);
    }
}
