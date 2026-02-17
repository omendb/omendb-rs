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
use crate::vector::hnsw::types::{HNSWParams, Metric};
use std::fs::OpenOptions;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use tracing::{error, info, instrument};

/// Current segment file format version
///
/// v1: Raw node data only (no quantization state)
/// v2: Raw node data + auxiliary data (mode, SQ8, upper neighbors)
const SEGMENT_FORMAT_VERSION: u32 = 2;

/// Magic bytes for segment files
const SEGMENT_MAGIC: &[u8; 8] = b"OMSEG\0\0\0";

/// Storage header size (7 u32 fields)
const STORAGE_HEADER_SIZE: usize = 7 * 4;

/// Alignment for the raw node data section (cache line).
/// Ensures mmap'd pointers are safe for f32/SIMD access.
const DATA_ALIGNMENT: usize = 64;

/// Maximum size for serialized distance function or params.
/// Protects against corrupted files causing OOM.
const MAX_SERIALIZED_LEN: usize = 4096;

/// Maximum auxiliary data size (1GB).
/// Protects against corrupted files causing OOM.
const MAX_AUX_DATA_SIZE: usize = 1 << 30;

/// Configure OpenOptions for cross-platform compatibility.
#[cfg(windows)]
fn configure_open_options(opts: &mut OpenOptions) {
    use std::os::windows::fs::OpenOptionsExt;
    opts.share_mode(0x1 | 0x2 | 0x4);
}

#[cfg(not(windows))]
fn configure_open_options(_opts: &mut OpenOptions) {}

/// Parsed segment file header (shared between load and load_mmap)
struct SegmentHeader {
    version: u32,
    id: u64,
    entry_point: Option<u32>,
    distance_fn: Metric,
    params: HNSWParams,
    len: usize,
    node_size: usize,
    neighbors_offset: usize,
    vector_offset: usize,
    metadata_offset: usize,
    dimensions: usize,
    max_neighbors: usize,
    /// Absolute byte offset where the raw node data starts
    data_offset: usize,
}

/// Parse segment header from reader, advancing past padding to data start.
///
/// After this returns, the reader is positioned at the start of the raw data section.
fn parse_header(reader: &mut impl Read) -> Result<SegmentHeader> {
    // Read and verify magic bytes
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != SEGMENT_MAGIC {
        error!(magic = ?magic, "Invalid magic bytes in segment file");
        return Err(HNSWError::Storage(format!(
            "Invalid segment magic: {magic:?}"
        )));
    }

    // Read version (accept v1 and v2)
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != 1 && version != SEGMENT_FORMAT_VERSION {
        error!(
            version,
            expected = SEGMENT_FORMAT_VERSION,
            "Unsupported segment version"
        );
        return Err(HNSWError::Storage(format!(
            "Unsupported segment version: {version} (expected 1 or {SEGMENT_FORMAT_VERSION})"
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

    // Read distance function (length-prefixed postcard)
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let df_len = u32::from_le_bytes(len_bytes) as usize;
    if df_len > MAX_SERIALIZED_LEN {
        return Err(HNSWError::Storage(format!(
            "Distance function data too large: {df_len} bytes (max {MAX_SERIALIZED_LEN})"
        )));
    }
    let mut df_bytes = vec![0u8; df_len];
    reader.read_exact(&mut df_bytes)?;
    let distance_fn: Metric = postcard::from_bytes(&df_bytes)?;

    // Read params (length-prefixed postcard)
    reader.read_exact(&mut len_bytes)?;
    let params_len = u32::from_le_bytes(len_bytes) as usize;
    if params_len > MAX_SERIALIZED_LEN {
        return Err(HNSWError::Storage(format!(
            "Params data too large: {params_len} bytes (max {MAX_SERIALIZED_LEN})"
        )));
    }
    let mut params_bytes = vec![0u8; params_len];
    reader.read_exact(&mut params_bytes)?;
    let params: HNSWParams = postcard::from_bytes(&params_bytes)?;

    // Read storage header
    let mut header_buf = [0u8; STORAGE_HEADER_SIZE];
    reader.read_exact(&mut header_buf)?;

    let len =
        u32::from_le_bytes([header_buf[0], header_buf[1], header_buf[2], header_buf[3]]) as usize;
    let node_size =
        u32::from_le_bytes([header_buf[4], header_buf[5], header_buf[6], header_buf[7]]) as usize;
    let neighbors_offset =
        u32::from_le_bytes([header_buf[8], header_buf[9], header_buf[10], header_buf[11]]) as usize;
    let vector_offset = u32::from_le_bytes([
        header_buf[12],
        header_buf[13],
        header_buf[14],
        header_buf[15],
    ]) as usize;
    let metadata_offset = u32::from_le_bytes([
        header_buf[16],
        header_buf[17],
        header_buf[18],
        header_buf[19],
    ]) as usize;
    let dimensions = u32::from_le_bytes([
        header_buf[20],
        header_buf[21],
        header_buf[22],
        header_buf[23],
    ]) as usize;
    let max_neighbors = u32::from_le_bytes([
        header_buf[24],
        header_buf[25],
        header_buf[26],
        header_buf[27],
    ]) as usize;

    // Calculate data offset with alignment
    let header_end = 8
        + 4
        + 8
        + 1
        + (if entry_point.is_some() { 4 } else { 0 })
        + 4
        + df_len
        + 4
        + params_len
        + STORAGE_HEADER_SIZE;
    let data_offset = if version >= 2 {
        let padding = (DATA_ALIGNMENT - (header_end % DATA_ALIGNMENT)) % DATA_ALIGNMENT;
        if padding > 0 {
            let mut pad = [0u8; DATA_ALIGNMENT];
            reader.read_exact(&mut pad[..padding])?;
        }
        header_end + padding
    } else {
        header_end
    };

    Ok(SegmentHeader {
        version,
        id,
        entry_point,
        distance_fn,
        params,
        len,
        node_size,
        neighbors_offset,
        vector_offset,
        metadata_offset,
        dimensions,
        max_neighbors,
        data_offset,
    })
}

/// Read and deserialize auxiliary data (v2+) from current reader position.
fn read_auxiliary(reader: &mut impl Read, storage: &mut NodeStorage) -> Result<()> {
    let mut aux_len_bytes = [0u8; 8];
    reader.read_exact(&mut aux_len_bytes)?;
    let aux_len = u64::from_le_bytes(aux_len_bytes) as usize;
    if aux_len > MAX_AUX_DATA_SIZE {
        return Err(HNSWError::Storage(format!(
            "Auxiliary data too large: {aux_len} bytes (max {MAX_AUX_DATA_SIZE})"
        )));
    }
    if aux_len > 0 {
        let mut aux_data = vec![0u8; aux_len];
        reader.read_exact(&mut aux_data)?;
        storage.deserialize_auxiliary(&aux_data).map_err(|e| {
            HNSWError::Storage(format!("Failed to deserialize auxiliary data: {e}"))
        })?;
    }
    Ok(())
}

impl FrozenSegment {
    /// Save frozen segment to disk (atomic: writes to temp file, then renames)
    ///
    /// Writes the segment data in a format suitable for both
    /// regular loading and memory-mapped access.
    #[instrument(skip(self, path), fields(segment_id = self.id(), num_vectors = self.len()))]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        info!("Starting segment save");
        let start = std::time::Instant::now();

        let path = path.as_ref();
        let tmp_path = path.with_extension("bin.tmp");

        // Write to temp file for atomic rename
        let file = {
            let mut opts = OpenOptions::new();
            opts.write(true).create(true).truncate(true);
            configure_open_options(&mut opts);
            let file = opts.open(&tmp_path).map_err(|e| {
                error!(error = ?e, "Failed to create segment temp file");
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

            // v2: Write alignment padding so raw data starts at a cache-line boundary.
            // This ensures mmap'd pointers are safe for f32/SIMD access.
            {
                let current_pos = 8
                    + 4
                    + 8
                    + 1
                    + (if self.entry_point().is_some() { 4 } else { 0 })
                    + 4
                    + df_bytes.len()
                    + 4
                    + params_bytes.len()
                    + STORAGE_HEADER_SIZE;
                let padding = (DATA_ALIGNMENT - (current_pos % DATA_ALIGNMENT)) % DATA_ALIGNMENT;
                if padding > 0 {
                    let zeros = [0u8; 64];
                    writer.write_all(&zeros[..padding])?;
                }
            }

            // Write storage data (raw bytes)
            if !storage.is_empty() {
                let data_bytes = storage.as_bytes();
                writer.write_all(data_bytes)?;
            }

            // v2: Write auxiliary data (mode, SQ8, upper neighbors)
            let aux_data = storage.serialize_auxiliary();
            writer.write_all(&(aux_data.len() as u64).to_le_bytes())?;
            writer.write_all(&aux_data)?;

            writer.flush()?;
            writer
                .into_inner()
                .map_err(|e| HNSWError::from(e.into_error()))?
        };

        // Sync to disk before rename for crash consistency
        file.sync_all()?;

        // Atomic rename (same filesystem guaranteed — tmp is sibling of target)
        std::fs::rename(&tmp_path, path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp_path);
            HNSWError::Storage(format!("Failed to rename segment file: {e}"))
        })?;

        let elapsed = start.elapsed();
        info!(
            duration_ms = elapsed.as_millis(),
            bytes_written = self.storage().len() * self.storage().node_size(),
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

        let header = parse_header(&mut reader)?;

        // Read storage data (with overflow check)
        let data_size = header.len.checked_mul(header.node_size).ok_or_else(|| {
            HNSWError::Storage(format!(
                "Segment data size overflow: len={} * node_size={}",
                header.len, header.node_size
            ))
        })?;
        let mut data = vec![0u8; data_size];
        if data_size > 0 {
            reader.read_exact(&mut data)?;
        }

        let mut storage = NodeStorage::from_bytes(
            data,
            header.len,
            header.node_size,
            header.neighbors_offset,
            header.vector_offset,
            header.metadata_offset,
            header.dimensions,
            header.max_neighbors,
        );

        if header.version >= 2 {
            read_auxiliary(&mut reader, &mut storage)?;
        }

        let elapsed = start.elapsed();
        info!(
            duration_ms = elapsed.as_millis(),
            segment_id = header.id,
            num_vectors = header.len,
            "Segment load completed"
        );

        Ok(Self::from_parts(
            header.id,
            header.entry_point,
            header.params,
            header.distance_fn,
            storage,
        ))
    }

    /// Load frozen segment with memory-mapped storage
    ///
    /// Metadata is loaded into memory; storage data is memory-mapped for
    /// zero-copy access. Kernel is advised for random access patterns
    /// and async page prefaulting.
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
        let mut reader = BufReader::new(&file);

        let header = parse_header(&mut reader)?;

        let data_size = header.len.checked_mul(header.node_size).ok_or_else(|| {
            HNSWError::Storage(format!(
                "Segment data size overflow: len={} * node_size={}",
                header.len, header.node_size
            ))
        })?;

        let mut storage = if data_size > 0 {
            // Validate file is large enough for declared data
            let file_len = file.metadata()?.len();
            let required_len = header.data_offset as u64 + data_size as u64;
            if file_len < required_len {
                return Err(HNSWError::Storage(format!(
                    "File too small: {file_len} bytes, need {required_len} bytes \
                     (data_offset={}, data_size={data_size})",
                    header.data_offset
                )));
            }

            // SAFETY: File size validated above, offset within file bounds
            let mmap = unsafe {
                MmapOptions::new()
                    .offset(header.data_offset as u64)
                    .len(data_size)
                    .map(&file)?
            };

            // Advise kernel on access patterns
            #[cfg(unix)]
            {
                use memmap2::Advice;
                // HNSW search is random access — disable readahead
                mmap.advise(Advice::Random)?;
                // Begin faulting pages asynchronously for warm cache
                mmap.advise(Advice::WillNeed)?;
            }

            // Transparent huge pages reduce TLB pressure at scale
            #[cfg(target_os = "linux")]
            {
                if let Err(e) = mmap.advise(memmap2::Advice::HugePage) {
                    tracing::debug!(error = %e, "HugePage advice not available");
                }
            }

            NodeStorage::from_mmap(
                mmap,
                header.len,
                header.node_size,
                header.neighbors_offset,
                header.vector_offset,
                header.metadata_offset,
                header.dimensions,
                header.max_neighbors,
            )
        } else {
            // Empty segment - use empty owned storage
            NodeStorage::new(header.dimensions, header.max_neighbors, 8)
        };

        // v2: Read auxiliary data from after the mmap section
        if header.version >= 2 {
            use std::io::Seek;
            let aux_offset = header.data_offset as u64 + data_size as u64;
            let mut reader = BufReader::new(&file);
            reader.seek(std::io::SeekFrom::Start(aux_offset))?;
            read_auxiliary(&mut reader, &mut storage)?;
        }

        let elapsed = start.elapsed();
        info!(
            duration_ms = elapsed.as_millis(),
            segment_id = header.id,
            num_vectors = header.len,
            "Mmap segment load completed"
        );

        Ok(Self::from_parts(
            header.id,
            header.entry_point,
            header.params,
            header.distance_fn,
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
        let mut mutable = MutableSegment::new(4, default_params(), Metric::L2).unwrap();
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
        let mutable = MutableSegment::new(4, default_params(), Metric::L2).unwrap();
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
            MutableSegment::with_capacity(128, default_params(), Metric::L2, 1000).unwrap();
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

        let mut mutable = MutableSegment::new(4, default_params(), Metric::Cosine).unwrap();
        mutable.insert(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        mutable.insert(&[0.0, 1.0, 0.0, 0.0]).unwrap();
        mutable.insert(&[0.707, 0.707, 0.0, 0.0]).unwrap();

        let frozen = mutable.freeze();
        frozen.save(&path).unwrap();

        let loaded = FrozenSegment::load(&path).unwrap();
        assert_eq!(loaded.distance_function(), Metric::Cosine);

        let results = loaded.search(&[1.0, 0.0, 0.0, 0.0], 3, 50);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_segment_atomic_save() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("segment.bin");

        let mut mutable = MutableSegment::new(4, default_params(), Metric::L2).unwrap();
        mutable.insert(&[1.0, 0.0, 0.0, 0.0]).unwrap();

        let frozen = mutable.freeze();
        frozen.save(&path).unwrap();

        // No temp file should remain after successful save
        let tmp_path = path.with_extension("bin.tmp");
        assert!(!tmp_path.exists());
        assert!(path.exists());

        // Verify loadable
        let loaded = FrozenSegment::load(&path).unwrap();
        assert_eq!(loaded.len(), 1);
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_segment_mmap_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("segment.bin");

        let mut mutable = MutableSegment::new(4, default_params(), Metric::L2).unwrap();
        for i in 0..10 {
            mutable.insert(&[i as f32, 0.0, 0.0, 0.0]).unwrap();
        }

        let frozen = mutable.freeze();
        let original_len = frozen.len();
        let original_entry = frozen.entry_point();
        frozen.save(&path).unwrap();

        let loaded = FrozenSegment::load_mmap(&path).unwrap();
        assert_eq!(loaded.len(), original_len);
        assert_eq!(loaded.entry_point(), original_entry);
        assert_eq!(loaded.params().m, 8);

        let results = loaded.search(&[5.0, 0.0, 0.0, 0.0], 3, 50);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 5);
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_segment_mmap_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty_segment.bin");

        let mutable = MutableSegment::new(4, default_params(), Metric::L2).unwrap();
        let frozen = mutable.freeze();
        frozen.save(&path).unwrap();

        let loaded = FrozenSegment::load_mmap(&path).unwrap();
        assert_eq!(loaded.len(), 0);
        assert!(loaded.entry_point().is_none());
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_segment_mmap_search_matches_heap() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("segment.bin");

        let mut mutable =
            MutableSegment::with_capacity(128, default_params(), Metric::L2, 1000).unwrap();
        for i in 0..500 {
            let vector: Vec<f32> = (0..128)
                .map(|j| ((i * 128 + j) % 256) as f32 / 256.0)
                .collect();
            mutable.insert(&vector).unwrap();
        }

        let frozen = mutable.freeze();
        frozen.save(&path).unwrap();

        let mmap_loaded = FrozenSegment::load_mmap(&path).unwrap();
        let heap_loaded = FrozenSegment::load(&path).unwrap();
        assert_eq!(mmap_loaded.len(), heap_loaded.len());

        let query: Vec<f32> = (0..128)
            .map(|j| (250 * 128 + j) as f32 / 256.0 % 1.0)
            .collect();
        let mmap_results = mmap_loaded.search(&query, 10, 100);
        let heap_results = heap_loaded.search(&query, 10, 100);
        assert_eq!(mmap_results.len(), heap_results.len());
        for (m, h) in mmap_results.iter().zip(heap_results.iter()) {
            assert_eq!(m.id, h.id);
            assert!((m.distance - h.distance).abs() < 1e-6);
        }
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_segment_mmap_cosine() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cosine_segment.bin");

        let mut mutable = MutableSegment::new(4, default_params(), Metric::Cosine).unwrap();
        mutable.insert(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        mutable.insert(&[0.0, 1.0, 0.0, 0.0]).unwrap();
        mutable.insert(&[0.707, 0.707, 0.0, 0.0]).unwrap();

        let frozen = mutable.freeze();
        frozen.save(&path).unwrap();

        let loaded = FrozenSegment::load_mmap(&path).unwrap();
        assert_eq!(loaded.distance_function(), Metric::Cosine);
        let results = loaded.search(&[1.0, 0.0, 0.0, 0.0], 3, 50);
        assert!(!results.is_empty());
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_segment_mmap_sq8() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sq8_segment.bin");

        let mut mutable = MutableSegment::new_quantized(4, default_params(), Metric::L2).unwrap();
        for i in 0..20 {
            mutable.insert(&[i as f32, 0.0, 0.0, 0.0]).unwrap();
        }

        let frozen = mutable.freeze();
        frozen.save(&path).unwrap();

        let loaded = FrozenSegment::load_mmap(&path).unwrap();
        assert_eq!(loaded.len(), 20);

        let results = loaded.search(&[10.0, 0.0, 0.0, 0.0], 5, 50);
        assert_eq!(results.len(), 5);
    }
}
