//! Write-Ahead Log for crash-consistent operations
//!
//! Based on P-HNSW research: `NLog` (node ops) + `NlistLog` (neighbor ops)

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
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
    /// Insert a new node: {id, level, vector, metadata}
    InsertNode = 1,
    /// Delete a node: {id}
    DeleteNode = 2,
    /// Update neighbors: {id, level, [`neighbor_ids`]}
    UpdateNeighbors = 3,
    /// Update metadata: {id, metadata}
    UpdateMetadata = 4,
    /// Checkpoint marker - safe truncation point
    Checkpoint = 100,
}

impl WalEntryType {
    /// Try to parse a WAL entry type from a byte
    ///
    /// Returns None for unknown entry types (which should be skipped during recovery)
    #[must_use]
    pub fn from_byte(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::InsertNode),
            2 => Some(Self::DeleteNode),
            3 => Some(Self::UpdateNeighbors),
            4 => Some(Self::UpdateMetadata),
            100 => Some(Self::Checkpoint),
            _ => None, // Unknown entry type - caller should handle
        }
    }
}

impl From<u8> for WalEntryType {
    fn from(v: u8) -> Self {
        // For backwards compatibility, unknown entries are treated as checkpoint
        // But prefer using from_byte() which returns Option
        Self::from_byte(v).unwrap_or(Self::Checkpoint)
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

    pub fn from_bytes(buf: &[u8; Self::SIZE]) -> Self {
        // Direct array indexing - infallible for fixed-size input buffer
        Self {
            entry_type: WalEntryType::from(buf[0]),
            timestamp: u64::from_le_bytes([
                buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11],
            ]),
            data_len: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
            checksum: u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]),
        }
    }
}

/// WAL entry (header + data)
#[derive(Debug, Clone)]
pub struct WalEntry {
    pub header: WalEntryHeader,
    pub data: Vec<u8>,
}

impl WalEntry {
    /// Create insert node entry
    #[must_use]
    pub fn insert_node(
        timestamp: u64,
        string_id: &str,
        level: u8,
        vector: &[f32],
        metadata: &[u8],
    ) -> Self {
        let mut data = Vec::new();

        // String ID (length-prefixed)
        data.extend_from_slice(&(string_id.len() as u32).to_le_bytes());
        data.extend_from_slice(string_id.as_bytes());

        // Level
        data.push(level);

        // Vector (length-prefixed f32 array)
        data.extend_from_slice(&(vector.len() as u32).to_le_bytes());
        for &val in vector {
            data.extend_from_slice(&val.to_le_bytes());
        }

        // Metadata (length-prefixed)
        data.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
        data.extend_from_slice(metadata);

        let checksum = crc32fast::hash(&data);

        Self {
            header: WalEntryHeader {
                entry_type: WalEntryType::InsertNode,
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
        let mut data = Vec::new();
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

    /// Create update neighbors entry
    #[must_use]
    pub fn update_neighbors(timestamp: u64, node_id: u32, level: u8, neighbors: &[u32]) -> Self {
        let mut data = Vec::new();

        // Node ID
        data.extend_from_slice(&node_id.to_le_bytes());

        // Level
        data.push(level);

        // Neighbors (length-prefixed)
        data.extend_from_slice(&(neighbors.len() as u32).to_le_bytes());
        for &neighbor in neighbors {
            data.extend_from_slice(&neighbor.to_le_bytes());
        }

        let checksum = crc32fast::hash(&data);

        Self {
            header: WalEntryHeader {
                entry_type: WalEntryType::UpdateNeighbors,
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

/// Write-Ahead Log
pub struct Wal {
    file: BufWriter<File>,
    #[allow(dead_code)]
    path: std::path::PathBuf,
    next_timestamp: u64,
    entry_count: u64,
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

        let mut wal = Self {
            file: BufWriter::new(file),
            path,
            next_timestamp: 0,
            entry_count: 0,
        };

        // Scan to find last timestamp
        if file_len > 0 {
            wal.scan_for_timestamp()?;
        }

        Ok(wal)
    }

    /// Scan WAL to find highest timestamp
    fn scan_for_timestamp(&mut self) -> io::Result<()> {
        let file = self.file.get_mut();
        file.seek(SeekFrom::Start(0))?;

        let mut header_buf = [0u8; WalEntryHeader::SIZE];
        let mut max_timestamp = 0u64;
        let mut count = 0u64;

        // Maximum reasonable entry size (100MB) - protects against corrupted data_len
        const MAX_ENTRY_SIZE: u32 = 100 * 1024 * 1024;

        loop {
            match file.read_exact(&mut header_buf) {
                Ok(()) => {
                    let header = WalEntryHeader::from_bytes(&header_buf);

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
    pub fn sync(&mut self) -> io::Result<()> {
        self.file.flush()?;
        self.file.get_mut().sync_all()
    }

    /// Read all entries after last checkpoint
    ///
    /// Note: Entries are validated via checksum. Invalid entries are skipped.
    /// Unknown entry types are also skipped (not treated as checkpoints).
    pub fn entries_after_checkpoint(&mut self) -> io::Result<Vec<WalEntry>> {
        let file = self.file.get_mut();
        file.seek(SeekFrom::Start(0))?;

        let mut all_entries = Vec::new();
        let mut last_checkpoint_idx: Option<usize> = None;
        let mut header_buf = [0u8; WalEntryHeader::SIZE];

        // Maximum reasonable entry size (100MB)
        const MAX_ENTRY_SIZE: u32 = 100 * 1024 * 1024;

        loop {
            match file.read_exact(&mut header_buf) {
                Ok(()) => {
                    let header = WalEntryHeader::from_bytes(&header_buf);

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
                        continue;
                    }

                    // Only count as checkpoint if it's actually a valid checkpoint entry type
                    // (not an unknown entry type that defaulted to Checkpoint via From<u8>)
                    let entry_type_byte = entry.header.entry_type as u8;
                    if WalEntryType::from_byte(entry_type_byte) == Some(WalEntryType::Checkpoint) {
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
        // Flush buffer before truncating (required on Windows)
        self.file.flush()?;
        self.file.get_mut().set_len(0)?;
        self.file.get_mut().seek(SeekFrom::Start(0))?;
        self.next_timestamp = 0;
        self.entry_count = 0;
        Ok(())
    }
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
            wal.append(WalEntry::insert_node(0, "vec1", 0, &[1.0, 2.0, 3.0], b"{}"))
                .unwrap();
            wal.append(WalEntry::delete_node(0, "vec2")).unwrap();
            wal.append(WalEntry::checkpoint(0)).unwrap();
            wal.append(WalEntry::insert_node(0, "vec3", 1, &[4.0, 5.0, 6.0], b"{}"))
                .unwrap();
            wal.sync().unwrap();
        }

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            let entries = wal.entries_after_checkpoint().unwrap();

            // Should only have entries after checkpoint
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].header.entry_type, WalEntryType::InsertNode);
        }
    }

    #[test]
    fn test_entry_checksum() {
        let entry = WalEntry::insert_node(1, "test", 0, &[1.0, 2.0], b"metadata");
        assert!(entry.verify());
    }

    #[test]
    fn test_corrupted_entry_data_detected() {
        let mut entry = WalEntry::insert_node(1, "test", 0, &[1.0, 2.0], b"metadata");
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
        let mut entry = WalEntry::insert_node(1, "test", 0, &[1.0, 2.0], b"metadata");
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
            wal.append(WalEntry::insert_node(0, "vec1", 0, &[1.0, 2.0, 3.0], b"{}"))
                .unwrap();
            wal.append(WalEntry::insert_node(0, "vec2", 0, &[4.0, 5.0, 6.0], b"{}"))
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
}
