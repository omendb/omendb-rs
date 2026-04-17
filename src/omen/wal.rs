//! Write-Ahead Log for crash-consistent operations
//!
//! Based on P-HNSW research: `NLog` (node ops) + `NlistLog` (neighbor ops)

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Configure OpenOptions for cross-platform compatibility.
/// On Windows, enables full file sharing to avoid "Access is denied" errors.
#[cfg(windows)]
fn configure_open_options(opts: &mut OpenOptions) {
    use std::os::windows::fs::OpenOptionsExt;
    // FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
    opts.share_mode(0x1 | 0x2 | 0x4);
}

#[cfg(not(windows))]
fn configure_open_options(_opts: &mut OpenOptions) {
    // No-op on Unix
}

/// WAL entry types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WalEntryType {
    /// Upsert a record with dense, sparse, and/or multi payloads
    UpsertRecord = 1,
    /// Delete a node: {id}
    DeleteNode = 2,
    /// Insert an edge: {from_id, to_id, edge_type, weight, metadata}
    InsertEdge = 3,
    /// Delete an edge: {from_id, to_id, edge_type}
    DeleteEdge = 4,
    /// Checkpoint marker - safe truncation point
    Checkpoint = 100,
}

impl WalEntryType {
    /// Try to parse a WAL entry type from a byte.
    ///
    /// Returns `None` for unknown entry types (skipped during recovery for forward compat).
    #[must_use]
    pub fn from_byte(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::UpsertRecord),
            2 => Some(Self::DeleteNode),
            3 => Some(Self::InsertEdge),
            4 => Some(Self::DeleteEdge),
            100 => Some(Self::Checkpoint),
            _ => None,
        }
    }
}

/// WAL entry header (20 bytes)
/// Layout: `entry_type(1)` + reserved(3) + timestamp(8) + `data_len(4)` + checksum(4)
#[derive(Debug, Clone)]
pub struct WalEntryHeader {
    pub entry_type: WalEntryType,
    pub timestamp: u64, // Monotonic counter
    pub data_len: u32,
    pub checksum: u32,
}

impl WalEntryHeader {
    pub const SIZE: usize = 20;

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0] = self.entry_type as u8;
        // bytes 1-3: reserved/padding
        buf[4..12].copy_from_slice(&self.timestamp.to_le_bytes());
        buf[12..16].copy_from_slice(&self.data_len.to_le_bytes());
        buf[16..20].copy_from_slice(&self.checksum.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; Self::SIZE]) -> io::Result<Self> {
        let entry_type = WalEntryType::from_byte(buf[0]).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown WAL entry type: {}", buf[0]),
            )
        })?;
        Ok(Self {
            entry_type,
            timestamp: u64::from_le_bytes([
                buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11],
            ]),
            data_len: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
            checksum: u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]),
        })
    }
}

/// WAL entry (header + data)
#[derive(Debug, Clone)]
pub struct WalEntry {
    pub header: WalEntryHeader,
    pub data: Vec<u8>,
}

impl WalEntry {
    /// Create unified upsert record entry
    #[must_use]
    pub fn upsert_record(
        timestamp: u64,
        string_id: &str,
        level: u8,
        dense: Option<&[f32]>,
        sparse: Option<(&[u32], &[f32])>,
        multi: Option<&[&[f32]]>,
        metadata: Option<&[u8]>,
    ) -> Self {
        let meta_bytes = metadata.unwrap_or(&[]);
        let mut capacity = 4 + string_id.len() + 1 + 4 + 4 + 4 + 4 + meta_bytes.len();

        if let Some(v) = dense {
            capacity += v.len() * 4;
        }
        if let Some((idx, val)) = sparse {
            capacity += 4 + idx.len() * 4 + 4 + val.len() * 4;
        }
        if let Some(tokens) = multi {
            let token_values: usize = tokens.iter().map(|t| t.len()).sum();
            capacity += 4 + 4 + token_values * 4; // 4 for token_count, 4 for token_dim
        }

        let mut data = Vec::with_capacity(capacity);

        // String ID (length-prefixed)
        data.extend_from_slice(&(string_id.len() as u32).to_le_bytes());
        data.extend_from_slice(string_id.as_bytes());

        // Level
        data.push(level);

        // Dense (Option)
        if let Some(v) = dense {
            data.extend_from_slice(&(v.len() as u32).to_le_bytes());
            let byte_slice =
                unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
            data.extend_from_slice(byte_slice);
        } else {
            data.extend_from_slice(&0u32.to_le_bytes());
        }

        // Sparse (Option)
        if let Some((idx, val)) = sparse {
            data.extend_from_slice(&(idx.len() as u32).to_le_bytes());
            let idx_bytes =
                unsafe { std::slice::from_raw_parts(idx.as_ptr() as *const u8, idx.len() * 4) };
            data.extend_from_slice(idx_bytes);
            data.extend_from_slice(&(val.len() as u32).to_le_bytes());
            let val_bytes =
                unsafe { std::slice::from_raw_parts(val.as_ptr() as *const u8, val.len() * 4) };
            data.extend_from_slice(val_bytes);
        } else {
            data.extend_from_slice(&0u32.to_le_bytes());
        }

        // Multi (Option)
        if let Some(tokens) = multi {
            data.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
            if tokens.is_empty() {
                data.extend_from_slice(&0u32.to_le_bytes());
            } else {
                let token_dim = tokens[0].len();
                data.extend_from_slice(&(token_dim as u32).to_le_bytes());
                for t in tokens {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(t.as_ptr() as *const u8, token_dim * 4)
                    };
                    data.extend_from_slice(bytes);
                }
            }
        } else {
            data.extend_from_slice(&0u32.to_le_bytes());
        }

        // Metadata (length-prefixed)
        data.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(meta_bytes);

        let checksum = crc32fast::hash(&data);

        Self {
            header: WalEntryHeader {
                entry_type: WalEntryType::UpsertRecord,
                timestamp,
                data_len: data.len() as u32,
                checksum,
            },
            data,
        }
    }

    /// Create delete node entry
    #[must_use]
    pub fn delete_node(timestamp: u64, string_id: &str) -> Self {
        // Pre-calculate exact capacity: 4 (id len) + id bytes
        let mut data = Vec::with_capacity(4 + string_id.len());
        data.extend_from_slice(&(string_id.len() as u32).to_le_bytes());
        data.extend_from_slice(string_id.as_bytes());

        let checksum = crc32fast::hash(&data);

        Self {
            header: WalEntryHeader {
                entry_type: WalEntryType::DeleteNode,
                timestamp,
                data_len: data.len() as u32,
                checksum,
            },
            data,
        }
    }

    /// Create insert edge entry: {from_id, to_id, edge_type, weight, metadata}
    #[must_use]
    pub fn insert_edge(
        timestamp: u64,
        from_id: &str,
        to_id: &str,
        edge_type: &str,
        weight: f32,
        metadata: Option<&[u8]>,
    ) -> Self {
        let meta_bytes = metadata.unwrap_or(&[]);
        let capacity =
            4 + from_id.len() + 4 + to_id.len() + 4 + edge_type.len() + 4 + 4 + meta_bytes.len();
        let mut data = Vec::with_capacity(capacity);

        data.extend_from_slice(&(from_id.len() as u32).to_le_bytes());
        data.extend_from_slice(from_id.as_bytes());
        data.extend_from_slice(&(to_id.len() as u32).to_le_bytes());
        data.extend_from_slice(to_id.as_bytes());
        data.extend_from_slice(&(edge_type.len() as u32).to_le_bytes());
        data.extend_from_slice(edge_type.as_bytes());
        data.extend_from_slice(&weight.to_le_bytes());
        data.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(meta_bytes);

        let checksum = crc32fast::hash(&data);
        Self {
            header: WalEntryHeader {
                entry_type: WalEntryType::InsertEdge,
                timestamp,
                data_len: data.len() as u32,
                checksum,
            },
            data,
        }
    }

    /// Create delete edge entry: {from_id, to_id, edge_type}
    #[must_use]
    pub fn delete_edge(timestamp: u64, from_id: &str, to_id: &str, edge_type: &str) -> Self {
        let capacity = 4 + from_id.len() + 4 + to_id.len() + 4 + edge_type.len();
        let mut data = Vec::with_capacity(capacity);

        data.extend_from_slice(&(from_id.len() as u32).to_le_bytes());
        data.extend_from_slice(from_id.as_bytes());
        data.extend_from_slice(&(to_id.len() as u32).to_le_bytes());
        data.extend_from_slice(to_id.as_bytes());
        data.extend_from_slice(&(edge_type.len() as u32).to_le_bytes());
        data.extend_from_slice(edge_type.as_bytes());

        let checksum = crc32fast::hash(&data);
        Self {
            header: WalEntryHeader {
                entry_type: WalEntryType::DeleteEdge,
                timestamp,
                data_len: data.len() as u32,
                checksum,
            },
            data,
        }
    }

    /// Create checkpoint entry
    #[must_use]
    pub fn checkpoint(timestamp: u64) -> Self {
        Self {
            header: WalEntryHeader {
                entry_type: WalEntryType::Checkpoint,
                timestamp,
                data_len: 0,
                checksum: 0,
            },
            data: Vec::new(),
        }
    }

    /// Verify entry checksum
    #[must_use]
    pub fn verify(&self) -> bool {
        if self.data.is_empty() {
            return self.header.checksum == 0;
        }
        crc32fast::hash(&self.data) == self.header.checksum
    }
}

/// WAL sidecar metadata (stored in `.wal.meta`)
///
/// 32 bytes: [checkpoint_offset: u64] [max_timestamp: u64] [entry_count: u64] [truncation_epoch: u64]
///
/// `truncation_epoch` increments on every `truncate()` call. The slim records snapshot
/// stores the epoch at the time it was written. On recovery, if the current WAL epoch is
/// greater than the snapshot's epoch, the WAL was truncated after the snapshot was taken,
/// meaning all current WAL entries are new and must be replayed. If epochs match, the WAL
/// was not truncated after the snapshot and its entries are already incorporated — skip replay.
///
/// Only `truncation_epoch` is authoritative across restarts. `max_timestamp` and
/// `entry_count` are rebuilt from the WAL body on open so restart behavior does not depend
/// on syncing a second metadata file on every append.
const WAL_META_SIZE: usize = 32;
/// Legacy 3-field format: [checkpoint_offset: u64] [max_timestamp: u64] [entry_count: u64]
const WAL_META_SIZE_V1: usize = 24;

fn meta_path(wal_path: &std::path::Path) -> std::path::PathBuf {
    let mut p = wal_path.as_os_str().to_os_string();
    p.push(".meta");
    std::path::PathBuf::from(p)
}

fn read_wal_meta(path: &std::path::Path) -> Option<(u64, u64, u64, u64)> {
    let data = std::fs::read(path).ok()?;
    // Support both old 24-byte format (epoch defaults to 0) and new 32-byte format
    if data.len() != WAL_META_SIZE && data.len() != WAL_META_SIZE_V1 {
        return None;
    }
    let checkpoint_offset = u64::from_le_bytes(data[0..8].try_into().ok()?);
    let max_timestamp = u64::from_le_bytes(data[8..16].try_into().ok()?);
    let entry_count = u64::from_le_bytes(data[16..24].try_into().ok()?);
    let truncation_epoch = if data.len() == WAL_META_SIZE {
        u64::from_le_bytes(data[24..32].try_into().ok()?)
    } else {
        0
    };
    Some((
        checkpoint_offset,
        max_timestamp,
        entry_count,
        truncation_epoch,
    ))
}

fn open_meta_file(path: &std::path::Path) -> Option<File> {
    let mut opts = OpenOptions::new();
    opts.write(true).create(true);
    configure_open_options(&mut opts);
    opts.open(path).ok()
}

/// Write-Ahead Log
pub struct Wal {
    file: File,
    /// Kept open across truncate() calls to avoid one open() syscall per checkpoint.
    meta_file: Option<File>,
    path: std::path::PathBuf,
    next_timestamp: u64,
    entry_count: u64,
    /// Incremented on every truncate(). Stored in slim snapshots so recovery can tell
    /// whether WAL entries predate the snapshot (skip) or postdate it (replay).
    truncation_epoch: u64,
}

impl Wal {
    /// Open or create WAL file
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut opts = OpenOptions::new();
        // Use write mode instead of append for Windows compatibility
        // (append mode on Windows may prevent truncation)
        opts.read(true).write(true).create(true);
        configure_open_options(&mut opts);
        let mut file = opts.open(&path)?;

        let metadata = file.metadata()?;
        let file_len = metadata.len();

        // Seek to end for append-like behavior
        if file_len > 0 {
            file.seek(SeekFrom::End(0))?;
        }

        let meta_path_buf = meta_path(&path);
        let meta_file = open_meta_file(&meta_path_buf);

        let mut wal = Self {
            file,
            meta_file,
            path,
            next_timestamp: 0,
            entry_count: 0,
            truncation_epoch: 0,
        };

        if file_len > 0 {
            if let Some((_cp_offset, _max_ts, _count, epoch)) = read_wal_meta(&meta_path(&wal.path))
            {
                wal.truncation_epoch = epoch;
            }
            // Rebuild max timestamp and entry count from the WAL body so restart behavior
            // does not depend on how recently `.wal.meta` was refreshed.
            wal.scan_for_timestamp()?;
        }

        Ok(wal)
    }

    /// Scan WAL to find highest timestamp
    fn scan_for_timestamp(&mut self) -> io::Result<()> {
        let file = &mut self.file;
        file.seek(SeekFrom::Start(0))?;

        let mut header_buf = [0u8; WalEntryHeader::SIZE];
        let mut max_timestamp = 0u64;
        let mut count = 0u64;

        // Maximum reasonable entry size (100MB) - protects against corrupted data_len
        const MAX_ENTRY_SIZE: u32 = 100 * 1024 * 1024;

        loop {
            match file.read_exact(&mut header_buf) {
                Ok(()) => {
                    if skip_unknown_entry(file, &header_buf)? {
                        continue;
                    }

                    let header = WalEntryHeader::from_bytes(&header_buf)?;

                    // Sanity check: reject obviously corrupted entries
                    if header.data_len > MAX_ENTRY_SIZE {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "WAL entry has suspicious data_len: {} bytes (max: {})",
                                header.data_len, MAX_ENTRY_SIZE
                            ),
                        ));
                    }

                    max_timestamp = max_timestamp.max(header.timestamp);
                    count += 1;

                    // Skip data
                    if header.data_len > 0 {
                        file.seek(SeekFrom::Current(header.data_len as i64))?;
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
        }

        self.next_timestamp = max_timestamp + 1;
        self.entry_count = count;

        // Seek to end for appending
        file.seek(SeekFrom::End(0))?;

        Ok(())
    }

    /// Append entry to WAL
    pub fn append(&mut self, mut entry: WalEntry) -> io::Result<()> {
        entry.header.timestamp = self.next_timestamp;
        self.next_timestamp += 1;

        self.file.write_all(&entry.header.to_bytes())?;
        if !entry.data.is_empty() {
            self.file.write_all(&entry.data)?;
        }

        self.entry_count += 1;
        Ok(())
    }

    /// Flush WAL to disk
    ///
    /// Only fsyncs the WAL data file. `.wal.meta` is refreshed on truncation/checkpoint
    /// only — not on every sync — to avoid a second fsync per write. Restarted WAL state
    /// rebuilds max timestamp and entry count from the WAL body; the sidecar is used for
    /// truncation epoch only.
    pub fn sync(&mut self) -> io::Result<()> {
        self.file.sync_all()
    }

    /// Return the current truncation epoch.
    ///
    /// Stored in slim snapshots so recovery can tell whether WAL entries are new
    /// (epoch increased since snapshot) or stale (epoch unchanged).
    #[must_use]
    pub fn truncation_epoch(&self) -> u64 {
        self.truncation_epoch
    }

    /// Return the highest timestamp currently present in the WAL.
    ///
    /// Used by full-manifest checkpoints to record which WAL entries were already
    /// incorporated into the published manifest.
    #[must_use]
    pub fn max_timestamp(&self) -> Option<u64> {
        if self.entry_count == 0 {
            None
        } else {
            Some(self.next_timestamp.saturating_sub(1))
        }
    }

    /// Read all entries after last checkpoint
    ///
    /// Note: Entries are validated via checksum. Invalid entries are skipped.
    /// Unknown entry types are also skipped (not treated as checkpoints).
    pub fn entries_after_checkpoint(&mut self) -> io::Result<Vec<WalEntry>> {
        let file = &mut self.file;
        file.seek(SeekFrom::Start(0))?;

        let mut all_entries = Vec::new();
        let mut last_checkpoint_idx: Option<usize> = None;
        let mut header_buf = [0u8; WalEntryHeader::SIZE];

        // Maximum reasonable entry size (100MB)
        const MAX_ENTRY_SIZE: u32 = 100 * 1024 * 1024;

        loop {
            match file.read_exact(&mut header_buf) {
                Ok(()) => {
                    if skip_unknown_entry(file, &header_buf)? {
                        continue;
                    }

                    let header = WalEntryHeader::from_bytes(&header_buf)?;

                    // Sanity check on data_len
                    if header.data_len > MAX_ENTRY_SIZE {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "WAL entry has suspicious data_len: {} bytes",
                                header.data_len
                            ),
                        ));
                    }

                    let mut data = vec![0u8; header.data_len as usize];
                    if header.data_len > 0 {
                        file.read_exact(&mut data)?;
                    }

                    let entry = WalEntry { header, data };

                    // Skip entries that fail checksum verification
                    if !entry.verify() {
                        tracing::warn!(
                            timestamp = entry.header.timestamp,
                            entry_type = entry.header.entry_type as u8,
                            data_len = entry.header.data_len,
                            "Skipping WAL entry with invalid checksum during recovery"
                        );
                        continue;
                    }

                    if entry.header.entry_type == WalEntryType::Checkpoint {
                        last_checkpoint_idx = Some(all_entries.len());
                    }

                    all_entries.push(entry);
                }
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
        }

        // Return entries after last checkpoint
        match last_checkpoint_idx {
            Some(idx) => Ok(all_entries.split_off(idx + 1)),
            None => Ok(all_entries),
        }
    }

    /// Get entry count
    #[must_use]
    pub fn len(&self) -> u64 {
        self.entry_count
    }

    /// Check if WAL is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Truncate WAL (after checkpoint)
    pub fn truncate(&mut self) -> io::Result<()> {
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        self.entry_count = 0;
        // next_timestamp intentionally resets to 0 — timestamps are session-scoped,
        // not globally monotonic. Recovery uses Checkpoint entry type, not timestamps.
        let previous_epoch = self.truncation_epoch;
        self.truncation_epoch += 1;
        if let Err(err) = self.write_meta() {
            self.truncation_epoch = previous_epoch;
            return Err(err);
        }
        Ok(())
    }

    /// Write .wal.meta using the cached file handle (avoids open() syscall on each checkpoint).
    /// Falls back to opening a fresh file if the handle is unavailable.
    fn write_meta(&mut self) -> io::Result<()> {
        // Layout: checkpoint_offset(8) + max_timestamp(8) + entry_count(8) + truncation_epoch(8)
        // After truncate: checkpoint_offset=0, max_timestamp=0, entry_count=0; only epoch varies.
        let mut buf = [0u8; WAL_META_SIZE];
        buf[24..32].copy_from_slice(&self.truncation_epoch.to_le_bytes());

        if let Some(ref mut f) = self.meta_file {
            f.seek(SeekFrom::Start(0))?;
            f.write_all(&buf)?;
            f.sync_all()
        } else {
            // Fallback: open (or create) and write
            let path = meta_path(&self.path);
            let mut opts = OpenOptions::new();
            opts.write(true).create(true).truncate(true);
            configure_open_options(&mut opts);
            let mut f = opts.open(&path)?;
            f.write_all(&buf)?;
            f.sync_all()
        }
    }
}

/// Skip unknown WAL entry types for forward compatibility.
///
/// Returns `Ok(true)` when the entry was skipped and the caller should continue
/// scanning from the next record.
fn skip_unknown_entry(
    file: &mut File,
    header_buf: &[u8; WalEntryHeader::SIZE],
) -> io::Result<bool> {
    // Maximum reasonable entry size (100MB) - protects against corrupted data_len
    const MAX_ENTRY_SIZE: u32 = 100 * 1024 * 1024;

    if WalEntryType::from_byte(header_buf[0]).is_some() {
        return Ok(false);
    }

    let data_len = u32::from_le_bytes([
        header_buf[12],
        header_buf[13],
        header_buf[14],
        header_buf[15],
    ]);
    if data_len > MAX_ENTRY_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("WAL entry has suspicious data_len: {data_len} bytes (max: {MAX_ENTRY_SIZE})"),
        ));
    }

    if data_len > 0 {
        file.seek(SeekFrom::Current(data_len as i64))?;
    }

    Ok(true)
}

/// Maximum string ID length (64KB) - prevents DoS via malicious length field
const MAX_STRING_ID_LEN: usize = 65536;
/// Maximum vector dimensions (1M) - prevents DoS via malicious length field
const MAX_VECTOR_DIM: usize = 1 << 20;
/// Maximum metadata length (16MB) - prevents DoS via malicious length field
const MAX_METADATA_LEN: usize = 16 << 20;

fn read_string_id(cursor: &mut std::io::Cursor<&[u8]>) -> io::Result<String> {
    let mut len_buf = [0u8; 4];
    cursor.read_exact(&mut len_buf)?;
    let id_len = u32::from_le_bytes(len_buf) as usize;
    if id_len > MAX_STRING_ID_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("String ID length {id_len} exceeds maximum {MAX_STRING_ID_LEN}"),
        ));
    }
    let mut id_buf = vec![0u8; id_len];
    cursor.read_exact(&mut id_buf)?;
    String::from_utf8(id_buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_vector_from_bytes(bytes: &[u8], dimensions: usize) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .take(dimensions)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap_or([0; 4])))
        .collect()
}

/// Parsed insert data from a WAL entry
#[derive(Debug, Clone)]
pub struct WalDeleteData {
    pub id: String,
}

pub struct WalUpsertData {
    pub id: String,
    pub level: u8,
    pub dense: Option<Vec<f32>>,
    pub sparse: Option<(Vec<u32>, Vec<f32>)>,
    pub multi: Option<Vec<Vec<f32>>>,
    pub metadata: Option<Vec<u8>>,
}

/// Parse WAL upsert record entry data
/// Parse WAL delete entry data
///
/// Returns parsed ID.
pub fn parse_wal_delete(data: &[u8]) -> std::io::Result<WalDeleteData> {
    let mut cursor = std::io::Cursor::new(data);
    let id = crate::omen::wal::read_string_id(&mut cursor)?;
    Ok(WalDeleteData { id })
}

pub fn parse_wal_upsert_record(data: &[u8]) -> io::Result<WalUpsertData> {
    let mut cursor = std::io::Cursor::new(data);
    let string_id = read_string_id(&mut cursor)?;

    let mut buf1 = [0u8; 1];
    let mut buf4 = [0u8; 4];

    // Read level
    cursor.read_exact(&mut buf1)?;
    let level = buf1[0];

    // Read dense
    cursor.read_exact(&mut buf4)?;
    let vec_len = u32::from_le_bytes(buf4) as usize;
    let dense = if vec_len > 0 {
        if vec_len > MAX_VECTOR_DIM {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Vector dimension {vec_len} exceeds maximum {MAX_VECTOR_DIM}"),
            ));
        }
        let byte_len = vec_len.checked_mul(4).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Vector byte size overflow")
        })?;
        let mut vec_bytes = vec![0u8; byte_len];
        cursor.read_exact(&mut vec_bytes)?;
        Some(read_vector_from_bytes(&vec_bytes, vec_len))
    } else {
        None
    };

    // Read sparse
    cursor.read_exact(&mut buf4)?;
    let sparse_len = u32::from_le_bytes(buf4) as usize;
    let sparse = if sparse_len > 0 {
        if sparse_len > MAX_VECTOR_DIM {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Sparse length exceeds maximum",
            ));
        }
        let mut indices = Vec::with_capacity(sparse_len);
        for _ in 0..sparse_len {
            cursor.read_exact(&mut buf4)?;
            indices.push(u32::from_le_bytes(buf4));
        }
        cursor.read_exact(&mut buf4)?;
        let values_len = u32::from_le_bytes(buf4) as usize;
        if values_len != sparse_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Sparse indices/values mismatch",
            ));
        }
        let byte_len = values_len.checked_mul(4).unwrap();
        let mut value_bytes = vec![0u8; byte_len];
        cursor.read_exact(&mut value_bytes)?;
        Some((indices, read_vector_from_bytes(&value_bytes, values_len)))
    } else {
        None
    };

    // Read multi
    cursor.read_exact(&mut buf4)?;
    let token_count = u32::from_le_bytes(buf4) as usize;
    let multi = if token_count > 0 {
        cursor.read_exact(&mut buf4)?;
        let token_dim = u32::from_le_bytes(buf4) as usize;
        if token_dim > MAX_VECTOR_DIM {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Token dimension exceeds maximum",
            ));
        }
        let mut tokens = Vec::with_capacity(token_count);
        let byte_len = token_dim.checked_mul(4).unwrap();
        for _ in 0..token_count {
            let mut token_bytes = vec![0u8; byte_len];
            cursor.read_exact(&mut token_bytes)?;
            tokens.push(read_vector_from_bytes(&token_bytes, token_dim));
        }
        Some(tokens)
    } else {
        None
    };

    // Read metadata
    cursor.read_exact(&mut buf4)?;
    let meta_len = u32::from_le_bytes(buf4) as usize;
    let metadata = if meta_len > 0 {
        if meta_len > MAX_METADATA_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Metadata length exceeds maximum",
            ));
        }
        let mut bytes = vec![0u8; meta_len];
        cursor.read_exact(&mut bytes)?;
        Some(bytes)
    } else {
        None
    };

    Ok(WalUpsertData {
        id: string_id,
        level,
        dense,
        sparse,
        multi,
        metadata,
    })
}

/// Parsed insert-edge data from a WAL entry
#[derive(Debug, Clone)]
pub struct WalInsertEdgeData {
    pub from_id: String,
    pub to_id: String,
    pub edge_type: String,
    pub weight: f32,
    pub metadata: Option<Vec<u8>>,
}

/// Parsed delete-edge data from a WAL entry
#[derive(Debug, Clone)]
pub struct WalDeleteEdgeData {
    pub from_id: String,
    pub to_id: String,
    pub edge_type: String,
}

/// Parse WAL insert-edge entry data.
pub fn parse_wal_insert_edge(data: &[u8]) -> io::Result<WalInsertEdgeData> {
    let mut cursor = std::io::Cursor::new(data);
    let from_id = read_string_id(&mut cursor)?;
    let to_id = read_string_id(&mut cursor)?;
    let edge_type = read_string_id(&mut cursor)?;

    let mut buf4 = [0u8; 4];
    cursor.read_exact(&mut buf4)?;
    let weight = f32::from_le_bytes(buf4);

    cursor.read_exact(&mut buf4)?;
    let meta_len = u32::from_le_bytes(buf4) as usize;
    let metadata = if meta_len > 0 {
        if meta_len > MAX_METADATA_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Edge metadata length {meta_len} exceeds maximum {MAX_METADATA_LEN}"),
            ));
        }
        let mut meta_bytes = vec![0u8; meta_len];
        cursor.read_exact(&mut meta_bytes)?;
        Some(meta_bytes)
    } else {
        None
    };

    Ok(WalInsertEdgeData {
        from_id,
        to_id,
        edge_type,
        weight,
        metadata,
    })
}

/// Parse WAL delete-edge entry data.
pub fn parse_wal_delete_edge(data: &[u8]) -> io::Result<WalDeleteEdgeData> {
    let mut cursor = std::io::Cursor::new(data);
    let from_id = read_string_id(&mut cursor)?;
    let to_id = read_string_id(&mut cursor)?;
    let edge_type = read_string_id(&mut cursor)?;
    Ok(WalDeleteEdgeData {
        from_id,
        to_id,
        edge_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_wal_roundtrip() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append(WalEntry::upsert_record(
                0,
                "vec1",
                0,
                Some(&[1.0, 2.0, 3.0]),
                None,
                None,
                Some(b"{}"),
            ))
            .unwrap();
            wal.append(WalEntry::delete_node(0, "vec2")).unwrap();
            wal.append(WalEntry::checkpoint(0)).unwrap();
            wal.append(WalEntry::upsert_record(
                0,
                "vec3",
                1,
                Some(&[4.0, 5.0, 6.0]),
                None,
                None,
                Some(b"{}"),
            ))
            .unwrap();
            wal.sync().unwrap();
        }

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            let entries = wal.entries_after_checkpoint().unwrap();

            // Should only have entries after checkpoint
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].header.entry_type, WalEntryType::UpsertRecord);
        }
    }

    #[test]
    fn test_entry_checksum() {
        let entry = WalEntry::upsert_record(
            1,
            "test",
            0,
            Some(&[1.0, 2.0]),
            None,
            None,
            Some(b"metadata"),
        );
        assert!(entry.verify());
    }

    #[test]
    fn test_corrupted_entry_data_detected() {
        let mut entry = WalEntry::upsert_record(
            1,
            "test",
            0,
            Some(&[1.0, 2.0]),
            None,
            None,
            Some(b"metadata"),
        );
        assert!(entry.verify());

        // Corrupt the data
        if !entry.data.is_empty() {
            entry.data[0] ^= 0xFF;
        }

        // Verify should now fail
        assert!(!entry.verify(), "Corrupted data should fail verification");
    }

    #[test]
    fn test_corrupted_entry_checksum_detected() {
        let mut entry = WalEntry::upsert_record(
            1,
            "test",
            0,
            Some(&[1.0, 2.0]),
            None,
            None,
            Some(b"metadata"),
        );
        assert!(entry.verify());

        // Corrupt the checksum
        entry.header.checksum ^= 0xFFFF_FFFF;

        // Verify should now fail
        assert!(
            !entry.verify(),
            "Corrupted checksum should fail verification"
        );
    }

    #[test]
    fn test_wal_recovery_skips_corrupted_entries() {
        use std::io::Write;

        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test_corrupt.wal");

        // Write valid entries
        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append(WalEntry::upsert_record(
                0,
                "vec1",
                0,
                Some(&[1.0, 2.0, 3.0]),
                None,
                None,
                Some(b"{}"),
            ))
            .unwrap();
            wal.append(WalEntry::upsert_record(
                0,
                "vec2",
                0,
                Some(&[4.0, 5.0, 6.0]),
                None,
                None,
                Some(b"{}"),
            ))
            .unwrap();
            wal.sync().unwrap();
        }

        // Corrupt the middle of the file (corrupt first entry's data)
        {
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&wal_path)
                .unwrap();

            // Corrupt bytes in the first entry's data section (after header)
            // Header is 20 bytes, so corrupt data at offset 25
            file.seek(SeekFrom::Start(25)).unwrap();
            file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF]).unwrap();
            file.sync_all().unwrap();
        }

        // Read entries - corrupted entries should be SKIPPED (not returned)
        {
            let mut wal = Wal::open(&wal_path).unwrap();
            let entries = wal.entries_after_checkpoint().unwrap();

            // All returned entries should pass verification (corrupted ones are skipped)
            for entry in &entries {
                assert!(entry.verify(), "All returned entries should be valid");
            }

            // We started with 2 entries, at least one should have been corrupted and skipped
            // The exact count depends on how corruption affects parsing
            assert!(
                entries.len() <= 2,
                "Should have at most 2 entries after skipping corrupted ones"
            );
        }
    }

    #[test]
    fn test_wal_reopen_recovers_entry_count_and_timestamps() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test_reopen_counts.wal");

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append(WalEntry::upsert_record(
                0,
                "vec1",
                0,
                Some(&[1.0, 2.0, 3.0]),
                None,
                None,
                Some(b"{}"),
            ))
            .unwrap();
            wal.append(WalEntry::delete_node(0, "vec1")).unwrap();
            wal.sync().unwrap();
            assert_eq!(wal.len(), 2);
            assert_eq!(wal.max_timestamp(), Some(1));
        }

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            assert_eq!(wal.len(), 2);
            assert_eq!(wal.max_timestamp(), Some(1));

            wal.append(WalEntry::upsert_record(
                0,
                "vec2",
                1,
                Some(&[4.0, 5.0, 6.0]),
                None,
                None,
                Some(b"{}"),
            ))
            .unwrap();
            assert_eq!(wal.len(), 3);
            assert_eq!(wal.max_timestamp(), Some(2));
        }
    }

    #[test]
    fn test_wal_reopen_skips_unknown_entry_types_when_scanning() {
        use std::io::Write;

        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test_unknown_entry.wal");

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append(WalEntry::upsert_record(
                0,
                "vec1",
                0,
                Some(&[1.0, 2.0, 3.0]),
                None,
                None,
                Some(b"{}"),
            ))
            .unwrap();
            wal.sync().unwrap();
        }

        {
            let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
            let mut header = [0u8; WalEntryHeader::SIZE];
            header[0] = 0xFF;
            header[4..12].copy_from_slice(&99u64.to_le_bytes());
            header[12..16].copy_from_slice(&4u32.to_le_bytes());
            file.write_all(&header).unwrap();
            file.write_all(&[1, 2, 3, 4]).unwrap();
            file.sync_all().unwrap();
        }

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            assert_eq!(wal.len(), 1);
            assert_eq!(wal.max_timestamp(), Some(0));

            wal.append(WalEntry::delete_node(0, "vec1")).unwrap();
            assert_eq!(wal.len(), 2);
            assert_eq!(wal.max_timestamp(), Some(1));
        }
    }

    #[test]
    fn test_truncate_rolls_back_epoch_on_meta_write_failure() {
        use std::fs;

        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test_truncate_epoch_rollback.wal");

        let mut wal = Wal::open(&wal_path).unwrap();
        wal.append(WalEntry::upsert_record(
            0,
            "vec1",
            0,
            Some(&[1.0, 2.0, 3.0]),
            None,
            None,
            Some(b"{}"),
        ))
        .unwrap();
        wal.sync().unwrap();

        let original_epoch = wal.truncation_epoch();
        let meta_path = meta_path(&wal_path);
        wal.meta_file = None;
        fs::remove_file(&meta_path).unwrap();
        fs::create_dir(&meta_path).unwrap();

        let err = wal.truncate().unwrap_err();
        assert!(matches!(
            err.kind(),
            io::ErrorKind::IsADirectory | io::ErrorKind::PermissionDenied
        ));
        assert_eq!(wal.truncation_epoch(), original_epoch);
    }
}
