// Value Log (vLog) implementation
// Based on WiscKey paper (USENIX FAST 2016)
//
// Architecture: Append-only log for storing values separately from keys
// - LSM tree stores: key + vLog pointer (offset + length)
// - vLog stores: actual values sequentially
// - Benefit: Compaction only rewrites keys, not values (10-100x less write amp)

use bytes::{Bytes, BytesMut};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

// VLog file format magic number: "VLOG"
const MAGIC: u32 = 0x564C4F47;
const VERSION: u32 = 0x00000001;
const HEADER_SIZE: u64 = 8; // magic (4) + version (4)

#[derive(Debug, Error)]
pub enum VLogError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("CRC mismatch: expected {expected:x}, got {actual:x}")]
    CrcMismatch { expected: u32, actual: u32 },

    #[error("Invalid record format")]
    InvalidRecordFormat,

    #[error("Invalid VLog format: bad magic or version")]
    InvalidFormat,
}

pub type Result<T> = std::result::Result<T, VLogError>;

/// Value pointer stored in LSM tree
/// Points to value location in vLog
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValuePointer {
    /// Offset in vLog file
    pub offset: u64,
    /// Length of value
    pub length: u32,
}

impl ValuePointer {
    /// Encode pointer to bytes for SSTable storage
    ///
    /// Format: [offset: u64][length: u32] = 12 bytes total
    pub fn to_bytes(&self) -> Bytes {
        let mut buf = bytes::BytesMut::with_capacity(12);
        buf.extend_from_slice(&self.offset.to_le_bytes());
        buf.extend_from_slice(&self.length.to_le_bytes());
        buf.freeze()
    }
}

/// Value Log record format:
/// [key_len: u32][key: bytes][value_len: u32][value: bytes][crc: u32]
///
/// Key is stored for GC validation (check if value is still valid)
#[derive(Debug, Clone)]
pub struct VLogRecord {
    pub key: Bytes,
    pub value: Bytes,
}

impl VLogRecord {
    /// Encode record to bytes
    pub fn encode(&self) -> Bytes {
        let key_len = self.key.len() as u32;
        let value_len = self.value.len() as u32;

        let total_len = 4 + key_len as usize + 4 + value_len as usize + 4;
        let mut buf = BytesMut::with_capacity(total_len);

        // Write key
        buf.extend_from_slice(&key_len.to_le_bytes());
        buf.extend_from_slice(&self.key);

        // Write value
        buf.extend_from_slice(&value_len.to_le_bytes());
        buf.extend_from_slice(&self.value);

        // Calculate CRC over key + value (hardware-accelerated CRC32C)
        let mut crc = crc32c::crc32c(&key_len.to_le_bytes());
        crc = crc32c::crc32c_append(crc, &self.key);
        crc = crc32c::crc32c_append(crc, &value_len.to_le_bytes());
        crc = crc32c::crc32c_append(crc, &self.value);

        buf.extend_from_slice(&crc.to_le_bytes());

        buf.freeze()
    }

    /// Decode record from bytes
    pub fn decode(data: Bytes) -> Result<Self> {
        if data.len() < 12 {
            // Minimum: key_len(4) + value_len(4) + crc(4)
            return Err(VLogError::InvalidRecordFormat);
        }

        let mut offset = 0;

        // Read key length
        let key_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        offset += 4;

        if offset + key_len > data.len() {
            return Err(VLogError::InvalidRecordFormat);
        }

        // Read key
        let key = data.slice(offset..offset + key_len);
        offset += key_len;

        if offset + 4 > data.len() {
            return Err(VLogError::InvalidRecordFormat);
        }

        // Read value length
        let value_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + value_len + 4 > data.len() {
            return Err(VLogError::InvalidRecordFormat);
        }

        // Read value
        let value = data.slice(offset..offset + value_len);
        offset += value_len;

        // Read CRC
        let stored_crc = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);

        // Verify CRC (hardware-accelerated CRC32C)
        let mut computed_crc = crc32c::crc32c(&(key_len as u32).to_le_bytes());
        computed_crc = crc32c::crc32c_append(computed_crc, &key);
        computed_crc = crc32c::crc32c_append(computed_crc, &(value_len as u32).to_le_bytes());
        computed_crc = crc32c::crc32c_append(computed_crc, &value);

        if stored_crc != computed_crc {
            return Err(VLogError::CrcMismatch {
                expected: stored_crc,
                actual: computed_crc,
            });
        }

        Ok(Self { key, value })
    }
}

/// Value Log - append-only log for storing values
#[derive(Debug)]
pub struct VLog {
    file: File,
    path: PathBuf,
    head: u64, // Append offset (where to write next)
    tail: u64, // GC offset (where GC starts reading)
}

impl VLog {
    /// Create a new vLog
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&path)?;

        // Write header: [magic: u32][version: u32]
        file.write_all(&MAGIC.to_le_bytes())?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.sync_all()?;

        Ok(Self {
            file,
            path,
            head: HEADER_SIZE,
            tail: HEADER_SIZE,
        })
    }

    /// Open an existing vLog
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new().write(true).read(true).open(&path)?;

        // Read and validate header
        file.seek(SeekFrom::Start(0))?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header)?;

        let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);

        if magic != MAGIC || version != VERSION {
            return Err(VLogError::InvalidFormat);
        }

        let head = file.metadata()?.len();

        Ok(Self {
            file,
            path,
            head,
            tail: HEADER_SIZE,
        })
    }

    /// Append a value to the vLog
    /// Returns pointer (offset, length) to be stored in LSM tree
    /// NOTE: Does NOT sync - caller must call sync() or sync_data() explicitly
    pub fn append(&mut self, key: &[u8], value: &[u8]) -> Result<ValuePointer> {
        let record = VLogRecord {
            key: Bytes::copy_from_slice(key),
            value: Bytes::copy_from_slice(value),
        };

        let encoded = record.encode();
        let record_offset = self.head;

        // Write to file (buffered, not synced yet)
        self.file.seek(SeekFrom::Start(self.head))?;
        self.file.write_all(&encoded)?;

        self.head += encoded.len() as u64;

        // Return pointer to value (skip key_len + key + value_len prefix)
        let key_len = key.len() as u64;
        let value_offset = record_offset + 4 + key_len + 4;
        let value_len = value.len() as u32;

        Ok(ValuePointer {
            offset: value_offset,
            length: value_len,
        })
    }

    /// Sync vLog to disk (durability)
    pub fn sync(&mut self) -> Result<()> {
        self.file.sync_data()?;
        Ok(())
    }

    /// Read a value from vLog using pointer
    pub fn read(&mut self, pointer: ValuePointer) -> Result<Bytes> {
        let mut buf = vec![0u8; pointer.length as usize];
        self.file.seek(SeekFrom::Start(pointer.offset))?;
        self.file.read_exact(&mut buf)?;
        Ok(Bytes::from(buf))
    }

    /// Read full record (key + value) at offset
    /// Used for GC validation
    pub fn read_record(&mut self, offset: u64) -> Result<(VLogRecord, u64)> {
        self.file.seek(SeekFrom::Start(offset))?;

        // Read key length
        let mut len_buf = [0u8; 4];
        self.file.read_exact(&mut len_buf)?;
        let key_len = u32::from_le_bytes(len_buf) as usize;

        // Read key
        let mut key_buf = vec![0u8; key_len];
        self.file.read_exact(&mut key_buf)?;

        // Read value length
        self.file.read_exact(&mut len_buf)?;
        let value_len = u32::from_le_bytes(len_buf) as usize;

        // Read value
        let mut value_buf = vec![0u8; value_len];
        self.file.read_exact(&mut value_buf)?;

        // Read CRC
        self.file.read_exact(&mut len_buf)?;

        // Build full record data for decoding
        let total_len = 4 + key_len + 4 + value_len + 4;
        let mut record_data = BytesMut::with_capacity(total_len);
        record_data.extend_from_slice(&(key_len as u32).to_le_bytes());
        record_data.extend_from_slice(&key_buf);
        record_data.extend_from_slice(&(value_len as u32).to_le_bytes());
        record_data.extend_from_slice(&value_buf);
        record_data.extend_from_slice(&len_buf);

        let record = VLogRecord::decode(record_data.freeze())?;
        let next_offset = offset + total_len as u64;

        Ok((record, next_offset))
    }

    /// Get current head position
    pub fn head(&self) -> u64 {
        self.head
    }

    /// Get current tail position
    pub fn tail(&self) -> u64 {
        self.tail
    }

    /// Set tail position (for GC)
    pub fn set_tail(&mut self, tail: u64) {
        self.tail = tail;
    }

    /// Get vLog file size
    pub fn size(&self) -> Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    /// Get vLog path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Verify vLog integrity by checking all record checksums
    ///
    /// Reads and validates the CRC32C checksum of every record in the vLog.
    /// Returns the number of records verified and total bytes read.
    ///
    /// # Errors
    ///
    /// Returns an error if any record has a checksum mismatch or I/O error.
    pub fn verify(&mut self) -> Result<VLogVerifyResult> {
        let mut records_verified = 0u64;
        let mut bytes_verified = 0u64;
        let mut offset = HEADER_SIZE;

        while offset < self.head {
            match self.read_record(offset) {
                Ok((_, next_offset)) => {
                    bytes_verified += next_offset - offset;
                    records_verified += 1;
                    offset = next_offset;
                }
                Err(VLogError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // Reached end of valid records
                    break;
                }
                Err(e) => return Err(e),
            }
        }

        Ok(VLogVerifyResult {
            records_verified,
            bytes_verified,
        })
    }
}

/// Result of vLog verification
#[derive(Debug, Clone, Default)]
pub struct VLogVerifyResult {
    /// Number of records verified
    pub records_verified: u64,
    /// Total bytes read and verified
    pub bytes_verified: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_vlog_record_encode_decode() {
        let record = VLogRecord {
            key: Bytes::from("key1"),
            value: Bytes::from("value1"),
        };

        let encoded = record.encode();
        let decoded = VLogRecord::decode(encoded).unwrap();

        assert_eq!(record.key, decoded.key);
        assert_eq!(record.value, decoded.value);
    }

    #[test]
    fn test_vlog_append_and_read() {
        let dir = tempdir().unwrap();
        let vlog_path = dir.path().join("test.vlog");

        let mut vlog = VLog::create(&vlog_path).unwrap();

        // Append value
        let pointer = vlog.append(b"key1", b"value1").unwrap();
        assert_eq!(pointer.length, 6); // "value1".len()

        // Read value back
        let value = vlog.read(pointer).unwrap();
        assert_eq!(value, Bytes::from("value1"));
    }

    #[test]
    fn test_vlog_multiple_values() {
        let dir = tempdir().unwrap();
        let vlog_path = dir.path().join("test.vlog");

        let mut vlog = VLog::create(&vlog_path).unwrap();

        // Append multiple values
        let p1 = vlog.append(b"key1", b"value1").unwrap();
        let p2 = vlog.append(b"key2", b"value2").unwrap();
        let p3 = vlog.append(b"key3", b"value3").unwrap();

        // Read values back
        assert_eq!(vlog.read(p1).unwrap(), Bytes::from("value1"));
        assert_eq!(vlog.read(p2).unwrap(), Bytes::from("value2"));
        assert_eq!(vlog.read(p3).unwrap(), Bytes::from("value3"));
    }

    #[test]
    fn test_vlog_reopen() {
        let dir = tempdir().unwrap();
        let vlog_path = dir.path().join("test.vlog");

        let pointer = {
            let mut vlog = VLog::create(&vlog_path).unwrap();
            vlog.append(b"key1", b"value1").unwrap()
        };

        // Reopen and read
        let mut vlog = VLog::open(&vlog_path).unwrap();
        let value = vlog.read(pointer).unwrap();
        assert_eq!(value, Bytes::from("value1"));
    }

    #[test]
    fn test_vlog_read_record() {
        let dir = tempdir().unwrap();
        let vlog_path = dir.path().join("test.vlog");

        let mut vlog = VLog::create(&vlog_path).unwrap();

        vlog.append(b"key1", b"value1").unwrap();
        vlog.append(b"key2", b"value2").unwrap();

        // Read first record (starts after 8-byte header)
        let (record, next_offset) = vlog.read_record(HEADER_SIZE).unwrap();
        assert_eq!(record.key, Bytes::from("key1"));
        assert_eq!(record.value, Bytes::from("value1"));

        // Read second record
        let (record, _) = vlog.read_record(next_offset).unwrap();
        assert_eq!(record.key, Bytes::from("key2"));
        assert_eq!(record.value, Bytes::from("value2"));
    }

    #[test]
    fn test_vlog_large_values() {
        let dir = tempdir().unwrap();
        let vlog_path = dir.path().join("test.vlog");

        let mut vlog = VLog::create(&vlog_path).unwrap();

        // Test with 4KB value (typical embedding size)
        let large_value = vec![b'x'; 4096];
        let pointer = vlog.append(b"key1", &large_value).unwrap();

        let value = vlog.read(pointer).unwrap();
        assert_eq!(value.len(), 4096);
        assert_eq!(value, Bytes::from(large_value));
    }
}
