//! Segment persistence - save/load frozen segments
//!
//! File format for FrozenSegment:
//! ```text
//! [Magic: b"OMSEG\0\0\0"] (8 bytes)
//! [Version: u32] (4 bytes)
//! [Segment ID: u64] (8 bytes)
//! [Entry point: Option<u32>] (1 + 0-4 bytes)
//! [Distance function] (length-prefixed postcard)
//! [Params] (length-prefixed postcard)
//! [Storage header: 28 bytes]
//!   - len: u32
//!   - node_size: u32
//!   - neighbors_offset: u32
//!   - vector_offset: u32
//!   - metadata_offset: u32
//!   - dimensions: u32
//!   - max_neighbors: u32
//! [Storage data: raw bytes] (len * node_size bytes)
//! ```
//!
//! The data section starts at a known offset for mmap support.

use crate::vector::hnsw::error::{HNSWError, Result};
use crate::vector::hnsw::node_storage::NodeStorage;
use crate::vector::hnsw::segment::FrozenSegment;
use crate::vector::hnsw::types::{DistanceFunction, HNSWParams};
use std::fs::OpenOptions;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use tracing::{error, info, instrument};

/// Current segment file format version
const SEGMENT_FORMAT_VERSION: u32 = 1;

/// Magic bytes for segment files
const SEGMENT_MAGIC: &[u8; 8] = b"OMSEG\0\0\0";

/// Storage header size (7 u32 fields)
const STORAGE_HEADER_SIZE: usize = 7 * 4;

/// Configure OpenOptions for cross-platform compatibility.
#[cfg(windows)]
fn configure_open_options(opts: &mut OpenOptions) {
    use std::os::windows::fs::OpenOptionsExt;
    opts.share_mode(0x1 | 0x2 | 0x4);
}

#[cfg(not(windows))]
fn configure_open_options(_opts: &mut OpenOptions) {}

impl FrozenSegment {
    /// Save frozen segment to disk
    ///
    /// Writes the segment data in a format suitable for both
    /// regular loading and memory-mapped access.
    #[instrument(skip(self, path), fields(segment_id = self.id(), num_vectors = self.len()))]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        info!("Starting segment save");
        let start = std::time::Instant::now();

        let mut opts = OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        configure_open_options(&mut opts);
        let file = opts.open(path).map_err(|e| {
            error!(error = ?e, "Failed to create segment file");
            HNSWError::from(e)
        })?;
        let mut writer = BufWriter::new(file);

        // Write magic bytes
        writer.write_all(SEGMENT_MAGIC)?;

        // Write version
        writer.write_all(&SEGMENT_FORMAT_VERSION.to_le_bytes())?;

        // Write segment ID
        writer.write_all(&self.id().to_le_bytes())?;

        // Write entry point
        match self.entry_point() {
            Some(ep) => {
                writer.write_all(&[1u8])?;
                writer.write_all(&ep.to_le_bytes())?;
            }
            None => {
                writer.write_all(&[0u8])?;
            }
        }

        // Write distance function (length-prefixed postcard)
        let df_bytes = postcard::to_allocvec(&self.distance_function())?;
        writer.write_all(&(df_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&df_bytes)?;

        // Write params (length-prefixed postcard)
        let params_bytes = postcard::to_allocvec(self.params())?;
        writer.write_all(&(params_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&params_bytes)?;

        // Write storage header
        let storage = self.storage();
        writer.write_all(&(storage.len() as u32).to_le_bytes())?;
        writer.write_all(&(storage.node_size() as u32).to_le_bytes())?;
        writer.write_all(&(storage.neighbors_offset() as u32).to_le_bytes())?;
        writer.write_all(&(storage.vector_offset() as u32).to_le_bytes())?;
        writer.write_all(&(storage.metadata_offset() as u32).to_le_bytes())?;
        writer.write_all(&(storage.dimensions() as u32).to_le_bytes())?;
        writer.write_all(&(storage.max_neighbors() as u32).to_le_bytes())?;

        // Write storage data (raw bytes)
        if !storage.is_empty() {
            let data_bytes = storage.as_bytes();
            writer.write_all(data_bytes)?;
        }

        writer.flush()?;

        let elapsed = start.elapsed();
        info!(
            duration_ms = elapsed.as_millis(),
            bytes_written = storage.len() * storage.node_size(),
            "Segment save completed"
        );

        Ok(())
    }

    /// Load frozen segment from disk (into memory)
    #[instrument(skip(path))]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        info!("Starting segment load");
        let start = std::time::Instant::now();

        let mut opts = OpenOptions::new();
        opts.read(true);
        configure_open_options(&mut opts);
        let file = opts.open(&path)?;
        let mut reader = BufReader::new(file);

        // Read and verify magic bytes
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != SEGMENT_MAGIC {
            error!(magic = ?magic, "Invalid magic bytes in segment file");
            return Err(HNSWError::Storage(format!(
                "Invalid segment magic: {magic:?}"
            )));
        }

        // Read version
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != SEGMENT_FORMAT_VERSION {
            error!(
                version,
                expected = SEGMENT_FORMAT_VERSION,
                "Unsupported segment version"
            );
            return Err(HNSWError::Storage(format!(
                "Unsupported segment version: {version} (expected {SEGMENT_FORMAT_VERSION})"
            )));
        }

        // Read segment ID
        let mut id_bytes = [0u8; 8];
        reader.read_exact(&mut id_bytes)?;
        let id = u64::from_le_bytes(id_bytes);

        // Read entry point
        let mut ep_flag = [0u8; 1];
        reader.read_exact(&mut ep_flag)?;
        let entry_point = if ep_flag[0] == 1 {
            let mut ep_bytes = [0u8; 4];
            reader.read_exact(&mut ep_bytes)?;
            Some(u32::from_le_bytes(ep_bytes))
        } else {
            None
        };

        // Read distance function
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        let df_len = u32::from_le_bytes(len_bytes) as usize;
        let mut df_bytes = vec![0u8; df_len];
        reader.read_exact(&mut df_bytes)?;
        let distance_fn: DistanceFunction = postcard::from_bytes(&df_bytes)?;

        // Read params
        reader.read_exact(&mut len_bytes)?;
        let params_len = u32::from_le_bytes(len_bytes) as usize;
        let mut params_bytes = vec![0u8; params_len];
        reader.read_exact(&mut params_bytes)?;
        let params: HNSWParams = postcard::from_bytes(&params_bytes)?;

        // Read storage header
        let mut header = [0u8; STORAGE_HEADER_SIZE];
        reader.read_exact(&mut header)?;

        let len = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let node_size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
        let neighbors_offset =
            u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
        let vector_offset =
            u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;
        let metadata_offset =
            u32::from_le_bytes([header[16], header[17], header[18], header[19]]) as usize;
        let dimensions =
            u32::from_le_bytes([header[20], header[21], header[22], header[23]]) as usize;
        let max_neighbors =
            u32::from_le_bytes([header[24], header[25], header[26], header[27]]) as usize;

        // Read storage data (with overflow check)
        let data_size = len.checked_mul(node_size).ok_or_else(|| {
            HNSWError::Storage(format!(
                "Segment data size overflow: len={len} * node_size={node_size}"
            ))
        })?;
        let mut data = vec![0u8; data_size];
        if data_size > 0 {
            reader.read_exact(&mut data)?;
        }

        // Reconstruct storage
        let storage = NodeStorage::from_bytes(
            data,
            len,
            node_size,
            neighbors_offset,
            vector_offset,
            metadata_offset,
            dimensions,
            max_neighbors,
        );

        let elapsed = start.elapsed();
        info!(
            duration_ms = elapsed.as_millis(),
            segment_id = id,
            num_vectors = len,
            "Segment load completed"
        );

        Ok(Self::from_parts(
            id,
            entry_point,
            params,
            distance_fn,
            storage,
        ))
    }

    /// Load frozen segment with memory-mapped storage
    ///
    /// The segment metadata is loaded into memory, but the storage data
    /// is memory-mapped for zero-copy access and reduced memory usage.
    #[cfg(feature = "mmap")]
    #[instrument(skip(path))]
    pub fn load_mmap<P: AsRef<Path>>(path: P) -> Result<Self> {
        use memmap2::MmapOptions;

        info!("Starting mmap segment load");
        let start = std::time::Instant::now();

        let mut opts = OpenOptions::new();
        opts.read(true);
        configure_open_options(&mut opts);
        let file = opts.open(&path)?;

        // First, read metadata (everything except storage data)
        let mut reader = BufReader::new(&file);

        // Read and verify magic bytes
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != SEGMENT_MAGIC {
            return Err(HNSWError::Storage(format!(
                "Invalid segment magic: {magic:?}"
            )));
        }

        // Read version
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != SEGMENT_FORMAT_VERSION {
            return Err(HNSWError::Storage(format!(
                "Unsupported segment version: {version}"
            )));
        }

        // Read segment ID
        let mut id_bytes = [0u8; 8];
        reader.read_exact(&mut id_bytes)?;
        let id = u64::from_le_bytes(id_bytes);

        // Read entry point
        let mut ep_flag = [0u8; 1];
        reader.read_exact(&mut ep_flag)?;
        let entry_point = if ep_flag[0] == 1 {
            let mut ep_bytes = [0u8; 4];
            reader.read_exact(&mut ep_bytes)?;
            Some(u32::from_le_bytes(ep_bytes))
        } else {
            None
        };

        // Read distance function
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        let df_len = u32::from_le_bytes(len_bytes) as usize;
        let mut df_bytes = vec![0u8; df_len];
        reader.read_exact(&mut df_bytes)?;
        let distance_fn: DistanceFunction = postcard::from_bytes(&df_bytes)?;

        // Read params
        reader.read_exact(&mut len_bytes)?;
        let params_len = u32::from_le_bytes(len_bytes) as usize;
        let mut params_bytes = vec![0u8; params_len];
        reader.read_exact(&mut params_bytes)?;
        let params: HNSWParams = postcard::from_bytes(&params_bytes)?;

        // Read storage header
        let mut header = [0u8; STORAGE_HEADER_SIZE];
        reader.read_exact(&mut header)?;

        let len = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let node_size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
        let neighbors_offset =
            u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
        let vector_offset =
            u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;
        let metadata_offset =
            u32::from_le_bytes([header[16], header[17], header[18], header[19]]) as usize;
        let dimensions =
            u32::from_le_bytes([header[20], header[21], header[22], header[23]]) as usize;
        let max_neighbors =
            u32::from_le_bytes([header[24], header[25], header[26], header[27]]) as usize;

        // Calculate data offset
        // magic(8) + version(4) + id(8) + entry_point(1+0|4) + df(4+len) + params(4+len) + header(28)
        let data_offset = 8
            + 4
            + 8
            + 1
            + (if entry_point.is_some() { 4 } else { 0 })
            + 4
            + df_len
            + 4
            + params_len
            + STORAGE_HEADER_SIZE;

        // Memory-map the data section (or use empty storage for zero-length)
        let data_size = len * node_size;
        let storage = if data_size > 0 {
            // Validate file is large enough for declared data
            let file_len = file.metadata()?.len();
            let required_len = data_offset as u64 + data_size as u64;
            if file_len < required_len {
                return Err(HNSWError::Storage(format!(
                    "File too small: {file_len} bytes, need {required_len} bytes (data_offset={data_offset}, data_size={data_size})"
                )));
            }

            let mmap = unsafe {
                MmapOptions::new()
                    .offset(data_offset as u64)
                    .len(data_size)
                    .map(&file)?
            };
            NodeStorage::from_mmap(
                mmap,
                len,
                node_size,
                neighbors_offset,
                vector_offset,
                metadata_offset,
                dimensions,
                max_neighbors,
            )
        } else {
            // Empty segment - use empty owned storage
            NodeStorage::new(dimensions, max_neighbors, 8)
        };

        let elapsed = start.elapsed();
        info!(
            duration_ms = elapsed.as_millis(),
            segment_id = id,
            num_vectors = len,
            "Mmap segment load completed"
        );

        Ok(Self::from_parts(
            id,
            entry_point,
            params,
            distance_fn,
            storage,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::hnsw::segment::MutableSegment;
    use tempfile::tempdir;

    fn default_params() -> HNSWParams {
        HNSWParams {
            m: 8,
            ef_construction: 50,
            ..Default::default()
        }
    }

    #[test]
    fn test_segment_save_load_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("segment.bin");

        // Create mutable segment and insert vectors
        let mut mutable = MutableSegment::new(4, default_params(), DistanceFunction::L2).unwrap();
        for i in 0..10 {
            mutable.insert(&[i as f32, 0.0, 0.0, 0.0]).unwrap();
        }

        // Freeze and save
        let frozen = mutable.freeze();
        let original_len = frozen.len();
        let original_entry = frozen.entry_point();
        frozen.save(&path).unwrap();

        // Load and verify
        let loaded = FrozenSegment::load(&path).unwrap();
        assert_eq!(loaded.len(), original_len);
        assert_eq!(loaded.entry_point(), original_entry);
        assert_eq!(loaded.params().m, 8);

        // Verify search still works
        let results = loaded.search(&[5.0, 0.0, 0.0, 0.0], 3, 50);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 5); // Should find exact match
    }

    #[test]
    fn test_segment_save_load_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty_segment.bin");

        // Create empty frozen segment
        let mutable = MutableSegment::new(4, default_params(), DistanceFunction::L2).unwrap();
        let frozen = mutable.freeze();
        assert_eq!(frozen.len(), 0);

        // Save and load
        frozen.save(&path).unwrap();
        let loaded = FrozenSegment::load(&path).unwrap();

        assert_eq!(loaded.len(), 0);
        assert!(loaded.entry_point().is_none());
    }

    #[test]
    fn test_segment_save_load_large() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("large_segment.bin");

        // Create segment with many vectors
        let mut mutable =
            MutableSegment::with_capacity(128, default_params(), DistanceFunction::L2, 1000)
                .unwrap();
        for i in 0..500 {
            let vector: Vec<f32> = (0..128)
                .map(|j| ((i * 128 + j) % 256) as f32 / 256.0)
                .collect();
            mutable.insert(&vector).unwrap();
        }

        // Freeze and save
        let frozen = mutable.freeze();
        frozen.save(&path).unwrap();

        // Load and verify
        let loaded = FrozenSegment::load(&path).unwrap();
        assert_eq!(loaded.len(), 500);

        // Search should return sorted results
        let query: Vec<f32> = (0..128)
            .map(|j| (250 * 128 + j) as f32 / 256.0 % 1.0)
            .collect();
        let results = loaded.search(&query, 10, 100);
        assert_eq!(results.len(), 10);
        for i in 1..results.len() {
            assert!(results[i - 1].distance <= results[i].distance);
        }
    }

    #[test]
    fn test_segment_cosine_distance() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cosine_segment.bin");

        let mut mutable =
            MutableSegment::new(4, default_params(), DistanceFunction::Cosine).unwrap();
        mutable.insert(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        mutable.insert(&[0.0, 1.0, 0.0, 0.0]).unwrap();
        mutable.insert(&[0.707, 0.707, 0.0, 0.0]).unwrap();

        let frozen = mutable.freeze();
        frozen.save(&path).unwrap();

        let loaded = FrozenSegment::load(&path).unwrap();
        assert_eq!(loaded.distance_function(), DistanceFunction::Cosine);

        let results = loaded.search(&[1.0, 0.0, 0.0, 0.0], 3, 50);
        assert!(!results.is_empty());
    }
}
