// Block-based storage for SSTable
// Implements RocksDB-style block format with prefix compression + varint + LZ4
//
// Block Structure (4KB default, compressed):
// ┌─────────────────────────────────────────────────────┐
// │ LZ4 Compressed Block Data:                         │
// │   Entry 0 (restart): [0][key_len][key][value_len][value] │
// │   Entry 1: [prefix_len][suffix_len][suffix][value_len][value] │
// │   ...                                                │
// │   Restart Points: [offset_0: varint][offset_16: varint]... │
// │   Num Restart Points: varint                        │
// ├─────────────────────────────────────────────────────┤
// │ Uncompressed Size: u32 (original size before LZ4)  │
// │ Compressed Flag: u8 (1=compressed, 0=uncompressed)  │
// │ Restart Offset: u32 (offset in *uncompressed* data) │
// │ Checksum: u32 (over compressed data + metadata)     │
// └─────────────────────────────────────────────────────┘
//
// Optimizations:
// - Prefix compression: 30-50% space savings
// - Varint encoding: 3-5% space savings
// - LZ4 compression: 40-60% space savings (CRITICAL - +30-40% performance)
// - Decompressed block cache: 2x cache efficiency

use crate::buffer::manager::FrameRef;
use bytes::{Bytes, BytesMut};
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use std::io::{self};
use std::sync::{Arc, OnceLock};
use thiserror::Error;
#[cfg(not(feature = "simd"))]
use varint_rs::VarintReader;
use varint_rs::VarintWriter;

/// Compression algorithm for SSTable blocks
///
/// Controls the compression algorithm used for data blocks. Different algorithms
/// offer different trade-offs between compression ratio and speed.
///
/// # Examples
///
/// ```rust
/// use seerdb::sstable::CompressionType;
///
/// // LZ4 for speed-critical workloads
/// let fast = CompressionType::Lz4;
///
/// // ZSTD for space-critical workloads (vectors, embeddings)
/// let compact = CompressionType::Zstd;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionType {
    /// No compression (fastest writes, largest files)
    None,
    /// LZ4 compression (fast, moderate ratio)
    /// Good for general workloads where speed matters more than size
    #[default]
    Lz4,
    /// ZSTD compression (slower, best ratio)
    /// Excellent for vector embeddings and highly compressible data
    Zstd,
}

impl CompressionType {
    /// Convert to block footer byte
    pub(crate) fn to_byte(self) -> u8 {
        match self {
            CompressionType::None => 0,
            CompressionType::Lz4 => 1,
            CompressionType::Zstd => 2,
        }
    }

    /// Parse from block footer byte
    pub(crate) fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(CompressionType::None),
            1 => Some(CompressionType::Lz4),
            2 => Some(CompressionType::Zstd),
            _ => None,
        }
    }
}

/// Helper to write varint to BytesMut
fn write_varint(buf: &mut BytesMut, value: u64) {
    let mut temp = Vec::new();
    temp.write_u64_varint(value)
        .expect("write to memory failed");
    buf.extend_from_slice(&temp);
}

/// Helper to read varint from slice, advancing offset
/// Uses SIMD acceleration if enabled
#[inline(always)]
fn read_varint(data: &[u8], offset: &mut usize) -> Option<u64> {
    #[cfg(feature = "simd")]
    {
        // Use our internal SIMD implementation (portable std::simd)
        // This will handle both long and short buffers efficiently
        if let Some((val, len)) = crate::simd::decode_varint(&data[*offset..]) {
            *offset += len;
            return Some(val);
        }
        None
    }
    #[cfg(not(feature = "simd"))]
    {
        let mut slice = &data[*offset..];
        match slice.read_u64_varint() {
            Ok(val) => {
                let read = data.len() - *offset - slice.len();
                *offset += read;
                Some(val)
            }
            Err(_) => None,
        }
    }
}

// Use SIMD-accelerated comparison when available, fallback to standard otherwise
#[cfg(feature = "simd")]
use crate::simd;

#[cfg(not(feature = "simd"))]
mod simd {
    use std::cmp::Ordering;
    #[inline]
    pub fn shared_prefix_len(a: &[u8], b: &[u8]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }
    #[inline]
    pub fn compare_keys(a: &[u8], b: &[u8]) -> Ordering {
        a.cmp(b)
    }
}

#[derive(Debug, Error)]
pub enum BlockError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Block corrupted: checksum mismatch")]
    Corruption,

    #[error("Invalid block format")]
    InvalidFormat,

    #[error("Block full")]
    BlockFull,
}

pub type Result<T> = std::result::Result<T, BlockError>;

/// Default block size (4KB)
pub const DEFAULT_BLOCK_SIZE: usize = 4096;

/// Restart interval for prefix compression (every N entries)
const RESTART_INTERVAL: usize = 16;

/// Block builder for writing entries
pub struct BlockBuilder {
    /// Buffer for block data
    buffer: BytesMut,
    /// Restart points (offsets to full keys)
    restart_points: Vec<u32>,
    /// Number of entries since last restart
    counter: usize,
    /// Last key added (for prefix compression)
    last_key: Bytes,
    /// Maximum block size
    max_size: usize,
    /// Compression algorithm to use (default: LZ4)
    compression_type: CompressionType,
}

impl BlockBuilder {
    /// Create a new block builder with default size
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_BLOCK_SIZE)
    }

    /// Create a new block builder with custom capacity
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            buffer: BytesMut::with_capacity(max_size),
            restart_points: vec![0], // First entry is always a restart point
            counter: 0,
            last_key: Bytes::new(),
            max_size,
            compression_type: CompressionType::Lz4,
        }
    }

    /// Set compression algorithm
    pub fn set_compression_type(&mut self, compression_type: CompressionType) {
        self.compression_type = compression_type;
    }

    /// Enable or disable compression (legacy API)
    #[deprecated(since = "0.1.0", note = "use set_compression_type instead")]
    pub fn set_compression(&mut self, enabled: bool) {
        self.compression_type = if enabled {
            CompressionType::Lz4
        } else {
            CompressionType::None
        };
    }

    /// Add an entry to the block
    /// Returns false if block is full
    #[inline]
    pub fn add(&mut self, key: &[u8], value: &[u8]) -> bool {
        // Calculate shared prefix length (0 for restart points) using SIMD
        let prefix_len = if self.counter > 0 && !self.last_key.is_empty() {
            simd::shared_prefix_len(key, &self.last_key)
        } else {
            0
        };

        let suffix_len = key.len() - prefix_len;

        // Calculate entry size with prefix compression + varint encoding
        // Format: [prefix_len: varint][suffix_len: varint][suffix][value_len: varint][value]
        // Conservative estimate: varint can be up to 10 bytes for u64
        let entry_size = 10 + 10 + suffix_len + 10 + value.len();

        // Check if we have space (reserve space for footer)
        // Footer: restart_offsets (varint each) + num_restarts (varint) + checksum (4 bytes)
        // Conservative estimate: 10 bytes per restart point + 10 bytes for count + 4 bytes checksum
        let footer_size = (self.restart_points.len() + 1) * 10 + 14;
        if self.buffer.len() + entry_size + footer_size > self.max_size {
            return false;
        }

        // Check if this should be a restart point
        if self.counter >= RESTART_INTERVAL {
            self.restart_points.push(self.buffer.len() as u32);
            self.counter = 0;
            // Restart point: full key (no prefix compression)
            return self.add(key, value);
        }

        // Write entry with prefix compression + varint encoding
        // [prefix_len: varint][suffix_len: varint][suffix][value_len: varint][value]
        write_varint(&mut self.buffer, prefix_len as u64);
        write_varint(&mut self.buffer, suffix_len as u64);
        self.buffer.extend_from_slice(&key[prefix_len..]);
        write_varint(&mut self.buffer, value.len() as u64);
        self.buffer.extend_from_slice(value);

        self.last_key = Bytes::copy_from_slice(key);
        self.counter += 1;
        true
    }

    /// Get the current size of the block (excluding footer)
    pub fn current_size(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the block is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get the last key added (for index building)
    pub fn last_key(&self) -> &[u8] {
        &self.last_key
    }

    /// Finalize the block and return bytes
    #[inline]
    pub fn finish(mut self) -> Bytes {
        // Save restart offset (where restart points begin in uncompressed data)
        let restart_offset = self.buffer.len() as u32;

        // Write restart points (varint-encoded)
        for offset in &self.restart_points {
            write_varint(&mut self.buffer, *offset as u64);
        }

        // Write number of restart points (varint-encoded)
        write_varint(&mut self.buffer, self.restart_points.len() as u64);

        // Save uncompressed size before compression
        let uncompressed_size = self.buffer.len() as u32;

        match self.compression_type {
            CompressionType::None => {
                // Uncompressed: append metadata directly to buffer
                self.buffer
                    .extend_from_slice(&uncompressed_size.to_le_bytes()); // 4 bytes
                self.buffer
                    .extend_from_slice(&[CompressionType::None.to_byte()]); // 1 byte
                self.buffer.extend_from_slice(&restart_offset.to_le_bytes()); // 4 bytes

                // Calculate checksum (over data + metadata so far)
                let checksum = crc32c::crc32c(&self.buffer);
                self.buffer.extend_from_slice(&checksum.to_le_bytes()); // 4 bytes

                self.buffer.freeze()
            }
            CompressionType::Lz4 => {
                // Compress block data with LZ4 (includes size prefix)
                let uncompressed_data = self.buffer.to_vec();
                let compressed_data = compress_prepend_size(&uncompressed_data);

                // Create final block with metadata
                let mut final_buffer = BytesMut::with_capacity(compressed_data.len() + 13);
                final_buffer.extend_from_slice(&compressed_data);

                // Write metadata
                final_buffer.extend_from_slice(&uncompressed_size.to_le_bytes()); // 4 bytes
                final_buffer.extend_from_slice(&[CompressionType::Lz4.to_byte()]); // 1 byte
                final_buffer.extend_from_slice(&restart_offset.to_le_bytes()); // 4 bytes

                // Calculate checksum over compressed data + metadata (hardware-accelerated CRC32C)
                let checksum = crc32c::crc32c(&final_buffer);
                final_buffer.extend_from_slice(&checksum.to_le_bytes()); // 4 bytes

                final_buffer.freeze()
            }
            CompressionType::Zstd => {
                // Compress block data with ZSTD (level 3 = balanced speed/ratio)
                let uncompressed_data = self.buffer.to_vec();
                let compressed_data =
                    zstd::encode_all(uncompressed_data.as_slice(), 3).unwrap_or(uncompressed_data);

                // Create final block with metadata
                let mut final_buffer = BytesMut::with_capacity(compressed_data.len() + 13);
                final_buffer.extend_from_slice(&compressed_data);

                // Write metadata
                final_buffer.extend_from_slice(&uncompressed_size.to_le_bytes()); // 4 bytes
                final_buffer.extend_from_slice(&[CompressionType::Zstd.to_byte()]); // 1 byte
                final_buffer.extend_from_slice(&restart_offset.to_le_bytes()); // 4 bytes

                // Calculate checksum over compressed data + metadata (hardware-accelerated CRC32C)
                let checksum = crc32c::crc32c(&final_buffer);
                final_buffer.extend_from_slice(&checksum.to_le_bytes()); // 4 bytes

                final_buffer.freeze()
            }
        }
    }

    /// Reset the builder for reuse
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.restart_points.clear();
        self.restart_points.push(0);
        self.counter = 0;
        self.last_key = Bytes::new();
    }
}

impl Default for BlockBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Data storage for a Block: either Owned (Bytes) or Borrowed (FrameRef)
#[derive(Clone, Debug)]
pub enum BlockData {
    Owned(Bytes),
    Borrowed(FrameRef),
}

impl BlockData {
    /// Access the underlying byte slice
    pub fn as_slice(&self) -> &[u8] {
        match self {
            BlockData::Owned(bytes) => bytes.as_ref(),
            BlockData::Borrowed(frame) => unsafe { frame.data_unchecked() },
        }
    }

    /// Create a sub-slice as Bytes
    /// For Borrowed data, this performs a copy to ensure the Bytes object is valid independently
    pub fn slice(&self, range: std::ops::Range<usize>) -> Bytes {
        match self {
            BlockData::Owned(bytes) => bytes.slice(range),
            BlockData::Borrowed(frame) => unsafe {
                let data = frame.data_unchecked();
                Bytes::copy_from_slice(&data[range])
            },
        }
    }
}

/// Block reader for parsing block data
#[derive(Clone)]
pub struct Block {
    data: BlockData,
    restart_offset: usize,
    num_restarts: usize,
    /// Decompressed entries cache (lazy initialized on first iter())
    /// Arc allows sharing across clones, OnceLock ensures thread-safe lazy init
    decompressed_cache: Arc<OnceLock<Vec<(Bytes, Bytes)>>>,
}

impl Block {
    /// Parse a block from block data (Owned or Borrowed)
    pub fn new(data: BlockData) -> Result<Self> {
        // Use a small scope to access slice for validation
        let (restart_offset, num_restarts) = {
            let raw_data = data.as_slice();
            if raw_data.len() < 13 {
                return Err(BlockError::InvalidFormat);
            }

            // Read checksum from end (fixed-width)
            let stored_checksum = u32::from_le_bytes([
                raw_data[raw_data.len() - 4],
                raw_data[raw_data.len() - 3],
                raw_data[raw_data.len() - 2],
                raw_data[raw_data.len() - 1],
            ]);

            // Verify checksum
            let computed_checksum = crc32c::crc32c(&raw_data[..raw_data.len() - 4]);
            if stored_checksum != computed_checksum {
                return Err(BlockError::Corruption);
            }

            // Read restart_offset
            let restart_offset = u32::from_le_bytes([
                raw_data[raw_data.len() - 8],
                raw_data[raw_data.len() - 7],
                raw_data[raw_data.len() - 6],
                raw_data[raw_data.len() - 5],
            ]) as usize;

            // Read compression type
            let compression_byte = raw_data[raw_data.len() - 9];
            let compression_type =
                CompressionType::from_byte(compression_byte).ok_or(BlockError::InvalidFormat)?;

            if compression_type != CompressionType::None {
                // If compressed, we MUST decompress into a new buffer (Owned)
                let compressed_slice = &raw_data[..raw_data.len() - 13];
                let uncompressed_data = match compression_type {
                    CompressionType::Lz4 => decompress_size_prepended(compressed_slice)
                        .map_err(|_| BlockError::InvalidFormat)?,
                    CompressionType::Zstd => {
                        zstd::decode_all(compressed_slice).map_err(|_| BlockError::InvalidFormat)?
                    }
                    CompressionType::None => unreachable!(),
                };
                let data = Bytes::from(uncompressed_data);

                // Parse num_restarts from uncompressed data
                // The uncompressed data contains [Entries... | Restart Points... | Num Restarts]
                // restart_offset points to the start of Restart Points
                if restart_offset >= data.len() {
                    return Err(BlockError::InvalidFormat);
                }

                let mut offset = restart_offset;
                let mut num_restarts = 0;
                while offset < data.len() {
                    if let Some(_offset_val) = read_varint(&data, &mut offset) {
                        num_restarts += 1;
                        let pos_after = offset;
                        if let Some(count) = read_varint(&data, &mut offset) {
                            if count as usize == num_restarts {
                                num_restarts = count as usize;
                                break;
                            } else {
                                offset = pos_after;
                            }
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                return Ok(Self {
                    data: BlockData::Owned(data),
                    restart_offset,
                    num_restarts,
                    decompressed_cache: Arc::new(OnceLock::new()),
                });
            }

            if restart_offset >= raw_data.len() - 13 {
                return Err(BlockError::InvalidFormat);
            }

            // Count restarts
            // For uncompressed blocks, we must stop before the footer (last 13 bytes)
            let content_limit = raw_data.len() - 13;
            let mut offset = restart_offset;
            let mut num_restarts = 0;
            while offset < content_limit {
                if let Some(_offset_val) = read_varint(raw_data, &mut offset) {
                    num_restarts += 1;
                    let pos_after = offset;
                    if let Some(count) = read_varint(raw_data, &mut offset) {
                        if count as usize == num_restarts {
                            num_restarts = count as usize;
                            break;
                        } else {
                            offset = pos_after;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            (restart_offset, num_restarts)
        };

        Ok(Self {
            data,
            restart_offset,
            num_restarts,
            decompressed_cache: Arc::new(OnceLock::new()),
        })
    }

    /// Legacy constructor for Bytes
    pub fn from_bytes(data: Bytes) -> Result<Self> {
        Self::new(BlockData::Owned(data))
    }

    /// Iterate over all entries in the block
    pub fn iter(&self) -> BlockIterator<'_> {
        // Populate decompressed cache on first access (lazy, thread-safe)
        let entries = self
            .decompressed_cache
            .get_or_init(|| self.decompress_all_entries());

        BlockIterator::new_cached(entries)
    }

    /// Find exact key match using binary search (for data blocks)
    /// Returns Some((key, value)) if found, None otherwise
    #[inline]
    pub fn find_exact(&self, key: &[u8]) -> Option<(Bytes, Bytes)> {
        let entries = self
            .decompressed_cache
            .get_or_init(|| self.decompress_all_entries());

        // Binary search for exact match using SIMD comparison
        match entries.binary_search_by(|(k, _)| simd::compare_keys(k.as_ref(), key)) {
            Ok(idx) => Some(entries[idx].clone()),
            Err(_) => None,
        }
    }

    /// Find first key >= target using binary search (for index blocks)
    /// Returns Some((key, value)) if found, None otherwise
    #[inline]
    pub fn find_lower_bound(&self, key: &[u8]) -> Option<(Bytes, Bytes)> {
        let entries = self
            .decompressed_cache
            .get_or_init(|| self.decompress_all_entries());

        // Binary search for first entry where entry_key >= key using SIMD
        let idx = entries.partition_point(|(k, _)| simd::compare_keys(k.as_ref(), key).is_lt());

        if idx < entries.len() {
            Some(entries[idx].clone())
        } else {
            None
        }
    }

    /// Find entry for MVCC lookup by user_key.
    ///
    /// This handles the InternalKey encoding correctly: when searching for a user_key,
    /// the encoded search key (user_key + inverted MAX seq) may sort before entries
    /// with longer user_keys that share the same prefix. This method scans forward
    /// from the lower bound position to find the first entry with matching user_key.
    ///
    /// Key insight: InternalKey encoding is `[user_key][8-byte-inverted-trailer]`.
    /// For prefix keys like "key1" and "key10":
    /// - key10 encodes as [k,e,y,1,0,trailer...]
    /// - key1 encodes as  [k,e,y,1,trailer...]
    /// Since '0' (0x30) < 0xFF (first trailer byte), key10 < key1 in encoded order!
    /// But in user_key order: "key1" < "key10" (shorter string is smaller).
    ///
    /// So we must scan forward until we find a matching user_key, and we can only
    /// terminate early when the encoded key passes beyond what any version of our
    /// target user_key could be (i.e., when the user_key prefix no longer matches).
    ///
    /// Returns Some((encoded_key, value)) if found, None otherwise.
    #[inline]
    pub fn find_mvcc(&self, encoded_search_key: &[u8], user_key: &[u8]) -> Option<(Bytes, Bytes)> {
        let entries = self
            .decompressed_cache
            .get_or_init(|| self.decompress_all_entries());

        // Binary search for first entry where entry_key >= search_key
        let start_idx = entries
            .partition_point(|(k, _)| simd::compare_keys(k.as_ref(), encoded_search_key).is_lt());

        // Scan forward from start_idx looking for matching user_key
        for i in start_idx..entries.len() {
            let (entry_key, entry_value) = &entries[i];

            // Extract user_key from encoded InternalKey (all but last 8 bytes)
            if entry_key.len() < 8 {
                continue; // Invalid entry, skip
            }
            let entry_user_key = &entry_key[..entry_key.len() - 8];

            if entry_user_key == user_key {
                // Found matching user_key
                return Some((entry_key.clone(), entry_value.clone()));
            }

            // Early termination check: can we stop scanning?
            // We can stop when the entry's user_key no longer shares our prefix AND
            // the encoded key is > our search key. This happens when:
            // 1. entry_user_key doesn't start with our user_key as prefix, AND
            // 2. entry_user_key > user_key lexicographically (for non-prefix cases)
            //
            // But for prefix cases (entry is longer and starts with target), we must
            // continue because our target might appear later.
            if !entry_user_key.starts_with(user_key) && entry_user_key > user_key {
                // This entry's user_key is strictly greater and not a prefix extension
                // Any remaining entries will also be > our target, so stop
                return None;
            }
        }

        // Reached end of entries without finding
        None
    }

    /// Get number of entries (approximate - counts restart points)
    pub fn num_entries_approx(&self) -> usize {
        self.num_restarts * RESTART_INTERVAL
    }

    /// Decompress all entries in the block (called once per block)
    fn decompress_all_entries(&self) -> Vec<(Bytes, Bytes)> {
        let mut entries = Vec::with_capacity(self.num_entries_approx());
        // Access data slice directly for SIMD compatibility
        let raw_data = self.data.as_slice();
        let data = &raw_data[..self.restart_offset];
        let mut offset = 0;
        let mut last_key = Bytes::new();

        while offset < data.len() {
            // Read prefix length (varint)
            let prefix_len = match read_varint(data, &mut offset) {
                Some(len) => len as usize,
                None => break,
            };

            // Read suffix length (varint)
            let suffix_len = match read_varint(data, &mut offset) {
                Some(len) => len as usize,
                None => break,
            };

            // Read suffix
            if offset + suffix_len > data.len() {
                break;
            }
            // Use BlockData::slice helper to handle both Owned and Borrowed cases
            let suffix = self.data.slice(offset..offset + suffix_len);
            offset += suffix_len;

            // Reconstruct full key from prefix + suffix
            let key = if prefix_len == 0 {
                // Restart point: suffix is the full key
                suffix.clone()
            } else {
                // Combine prefix from last_key with suffix
                if prefix_len > last_key.len() {
                    break; // Invalid format
                }
                let mut key_data = BytesMut::with_capacity(prefix_len + suffix_len);
                key_data.extend_from_slice(&last_key[..prefix_len]);
                key_data.extend_from_slice(&suffix);
                key_data.freeze()
            };

            // Read value length (varint)
            let value_len = match read_varint(data, &mut offset) {
                Some(len) => len as usize,
                None => break,
            };

            // Read value
            if offset + value_len > data.len() {
                break;
            }
            let value = self.data.slice(offset..offset + value_len);
            offset += value_len;

            // Update last_key for next entry
            last_key = key.clone();

            // Add to decompressed entries
            entries.push((key, value));
        }

        entries
    }
}

/// Iterator over block entries (now iterates over decompressed cache)
pub struct BlockIterator<'a> {
    iter: std::slice::Iter<'a, (Bytes, Bytes)>,
}

impl<'a> BlockIterator<'a> {
    fn new_cached(entries: &'a [(Bytes, Bytes)]) -> Self {
        Self {
            iter: entries.iter(),
        }
    }
}

impl<'a> Iterator for BlockIterator<'a> {
    type Item = Result<(Bytes, Bytes)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, v)| Ok((k.clone(), v.clone())))
    }
}

impl<'a> DoubleEndedIterator for BlockIterator<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .next_back()
            .map(|(k, v)| Ok((k.clone(), v.clone())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_builder_single_entry() {
        let mut builder = BlockBuilder::new();
        assert!(builder.add(b"key1", b"value1"));

        let block_data = builder.finish();
        let block = Block::from_bytes(block_data).unwrap();

        let entries: Vec<_> = block.iter().collect();
        assert_eq!(entries.len(), 1);

        let (key, value) = entries[0].as_ref().unwrap();
        assert_eq!(key, &Bytes::from("key1"));
        assert_eq!(value, &Bytes::from("value1"));
    }

    #[test]
    fn test_block_builder_multiple_entries() {
        let mut builder = BlockBuilder::new();
        assert!(builder.add(b"key1", b"value1"));
        assert!(builder.add(b"key2", b"value2"));
        assert!(builder.add(b"key3", b"value3"));

        let block_data = builder.finish();
        let block = Block::from_bytes(block_data).unwrap();

        let entries: Vec<_> = block.iter().map(|r| r.unwrap()).collect();
        assert_eq!(entries.len(), 3);

        assert_eq!(entries[0].0, Bytes::from("key1"));
        assert_eq!(entries[1].0, Bytes::from("key2"));
        assert_eq!(entries[2].0, Bytes::from("key3"));
    }

    #[test]
    fn test_block_builder_full() {
        let mut builder = BlockBuilder::with_capacity(256); // Small block

        // Fill until full
        let mut count = 0;
        for i in 0..100 {
            let key = format!("key{:04}", i);
            let value = format!("value{:04}", i);
            if !builder.add(key.as_bytes(), value.as_bytes()) {
                break;
            }
            count += 1;
        }

        assert!(
            count > 0 && count < 100,
            "Block should fill before 100 entries"
        );

        let block_data = builder.finish();
        let block = Block::from_bytes(block_data).unwrap();

        let entries: Vec<_> = block.iter().collect();
        assert_eq!(entries.len(), count);
    }

    #[test]
    fn test_block_checksum_validation() {
        let mut builder = BlockBuilder::new();
        builder.add(b"key1", b"value1");
        let mut block_data = builder.finish().to_vec();

        // Corrupt a byte
        block_data[0] ^= 0xFF;

        let result = Block::from_bytes(Bytes::from(block_data));
        assert!(matches!(result, Err(BlockError::Corruption)));
    }

    #[test]
    fn test_block_restart_points() {
        let mut builder = BlockBuilder::new();

        // Add more than RESTART_INTERVAL entries
        for i in 0..40 {
            let key = format!("key{:04}", i);
            let value = format!("value{:04}", i);
            assert!(builder.add(key.as_bytes(), value.as_bytes()));
        }

        // Should have multiple restart points
        assert!(builder.restart_points.len() > 1);

        let block_data = builder.finish();
        let block = Block::from_bytes(block_data).unwrap();

        let entries: Vec<_> = block.iter().map(|r| r.unwrap()).collect();
        assert_eq!(entries.len(), 40);
    }

    #[test]
    fn test_block_large_values() {
        let mut builder = BlockBuilder::new();
        let large_value = vec![b'x'; 2000]; // 2KB value

        assert!(builder.add(b"key1", &large_value));

        let block_data = builder.finish();
        let block = Block::from_bytes(block_data).unwrap();

        let entries: Vec<_> = block.iter().collect();
        assert_eq!(entries.len(), 1);

        let (_, value) = entries[0].as_ref().unwrap();
        assert_eq!(value.len(), 2000);
    }

    #[test]
    fn test_block_zstd_compression() {
        let mut builder = BlockBuilder::new();
        builder.set_compression_type(CompressionType::Zstd);

        // Add multiple entries
        for i in 0..20 {
            let key = format!("key{:04}", i);
            let value = format!("value{:04}", i);
            assert!(builder.add(key.as_bytes(), value.as_bytes()));
        }

        let block_data = builder.finish();
        let block = Block::from_bytes(block_data).unwrap();

        let entries: Vec<_> = block.iter().map(|r| r.unwrap()).collect();
        assert_eq!(entries.len(), 20);
        assert_eq!(entries[0].0, Bytes::from("key0000"));
        assert_eq!(entries[19].0, Bytes::from("key0019"));
    }

    #[test]
    fn test_block_no_compression() {
        let mut builder = BlockBuilder::new();
        builder.set_compression_type(CompressionType::None);

        assert!(builder.add(b"key1", b"value1"));
        assert!(builder.add(b"key2", b"value2"));

        let block_data = builder.finish();
        let block = Block::from_bytes(block_data).unwrap();

        let entries: Vec<_> = block.iter().map(|r| r.unwrap()).collect();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, Bytes::from("key1"));
        assert_eq!(entries[1].0, Bytes::from("key2"));
    }

    #[test]
    fn test_compression_ratio_comparison() {
        // Create repetitive data that compresses well
        let test_data: Vec<(Vec<u8>, Vec<u8>)> = (0..50)
            .map(|i| {
                let key = format!("user_profile_{:08}", i).into_bytes();
                // Highly compressible repeated pattern
                let value = format!("{{\"name\":\"user{}\",\"email\":\"user{}@example.com\",\"bio\":\"This is a sample biography that contains repetitive text patterns for testing compression.\"}}", i, i).into_bytes();
                (key, value)
            })
            .collect();

        // Build with no compression
        let mut none_builder = BlockBuilder::with_capacity(16384);
        none_builder.set_compression_type(CompressionType::None);
        for (k, v) in &test_data {
            none_builder.add(k, v);
        }
        let none_size = none_builder.finish().len();

        // Build with LZ4
        let mut lz4_builder = BlockBuilder::with_capacity(16384);
        lz4_builder.set_compression_type(CompressionType::Lz4);
        for (k, v) in &test_data {
            lz4_builder.add(k, v);
        }
        let lz4_size = lz4_builder.finish().len();

        // Build with ZSTD
        let mut zstd_builder = BlockBuilder::with_capacity(16384);
        zstd_builder.set_compression_type(CompressionType::Zstd);
        for (k, v) in &test_data {
            zstd_builder.add(k, v);
        }
        let zstd_size = zstd_builder.finish().len();

        // ZSTD should compress better than LZ4 for this kind of data
        assert!(lz4_size < none_size, "LZ4 should compress data");
        assert!(zstd_size < none_size, "ZSTD should compress data");
        assert!(
            zstd_size <= lz4_size,
            "ZSTD ({}) should compress at least as well as LZ4 ({})",
            zstd_size,
            lz4_size
        );
    }

    #[test]
    fn test_zstd_large_values() {
        // Use larger block for embeddings
        let mut builder = BlockBuilder::with_capacity(16384);
        builder.set_compression_type(CompressionType::Zstd);

        // Simulate vector embedding (768 floats as bytes = 3KB)
        let embedding: Vec<u8> = (0..768 * 4).map(|i| (i % 256) as u8).collect();
        assert!(builder.add(b"embedding_key", &embedding));

        let block_data = builder.finish();
        let block = Block::from_bytes(block_data).unwrap();

        let entries: Vec<_> = block.iter().collect();
        assert_eq!(entries.len(), 1);

        let (key, value) = entries[0].as_ref().unwrap();
        assert_eq!(key, &Bytes::from("embedding_key"));
        assert_eq!(value.len(), 768 * 4);
    }
}
