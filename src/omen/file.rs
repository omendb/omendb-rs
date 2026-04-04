//! `OmenFile` - main API for .omen format
//!
//! Storage backend for `VectorStore`. Uses postcard for efficient binary serialization.

use crate::omen::{
    ManifestHeader, NodeLocation, OmenFooter, OmenManifest, SegmentType, align_to_page,
    header::{HEADER_SIZE, Metric, OmenHeader},
    wal::{Wal, WalEntry},
};

// Re-export WAL parsing functions for external use
pub use crate::omen::wal::{
    WalDeleteData, WalDeleteEdgeData, WalInsertData, WalInsertEdgeData, parse_wal_delete,
    parse_wal_delete_edge, parse_wal_insert, parse_wal_insert_edge,
};
use crate::vector::store::record_store::RecordStore;
use anyhow::Result;
use fs2::FileExt;
use memmap2::MmapMut;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};

#[cfg(not(target_endian = "little"))]
compile_error!("OmenFile raw vector I/O requires little-endian targets");
use std::path::{Path, PathBuf};

/// Configure OpenOptions for cross-platform compatibility.
/// On Windows, enables full file sharing to avoid "Access is denied" errors.
#[cfg(windows)]
fn configure_open_options(opts: &mut OpenOptions) {
    use std::os::windows::fs::OpenOptionsExt;
    // FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
    opts.share_mode(0x1 | 0x2 | 0x4);
}

#[cfg(not(windows))]
fn configure_open_options(_opts: &mut OpenOptions) {}

fn lock_exclusive(file: &File) -> io::Result<()> {
    file.try_lock_exclusive().map_err(|_| {
        io::Error::new(
            io::ErrorKind::WouldBlock,
            "Database is locked by another process",
        )
    })
}

const MANIFEST_WAL_EPOCH_KEY: &str = "manifest_wal_epoch";
const MANIFEST_WAL_MAX_TS_KEY: &str = "manifest_wal_max_ts";

fn set_manifest_wal_replay_cutoff(config: &mut HashMap<String, u64>, wal: &Wal) {
    config.insert(MANIFEST_WAL_EPOCH_KEY.to_string(), wal.truncation_epoch());
    if let Some(max_ts) = wal.max_timestamp() {
        config.insert(MANIFEST_WAL_MAX_TS_KEY.to_string(), max_ts);
    } else {
        config.remove(MANIFEST_WAL_MAX_TS_KEY);
    }
}

// Note: DatabaseState removed in Phase 5 - state now managed by RecordStore at VectorStore level.
// OmenFile is pure I/O: WAL append + checkpoint_from_snapshot.

/// Helper for writing aligned segments to an Omen file
struct SegmentWriter<'a> {
    file: &'a mut File,
    current_offset: u64,
}

impl<'a> SegmentWriter<'a> {
    fn new(file: &'a mut File, start_offset: u64) -> Self {
        Self {
            file,
            current_offset: start_offset,
        }
    }

    /// Write data at the current offset (page-aligned) and return its location
    fn write_aligned(
        &mut self,
        data: &[u8],
        segment_type: SegmentType,
    ) -> io::Result<NodeLocation> {
        self.current_offset = align_to_page(self.current_offset as usize) as u64;
        self.file.seek(SeekFrom::Start(self.current_offset))?;
        self.file.write_all(data)?;

        let length = u32::try_from(data.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Segment data exceeds u32::MAX bytes",
            )
        })?;
        let location = NodeLocation {
            offset: self.current_offset,
            length,
            segment_type,
        };

        self.current_offset += data.len() as u64;
        Ok(location)
    }
}

/// `OmenFile` - single-file vector database
///
/// Storage layer for vectors, metadata, and serialized HNSW index.
/// Graph traversal is handled by `HNSWIndex` in the vector layer.
pub struct OmenFile {
    path: PathBuf,
    file: Option<File>,
    mmap: Option<MmapMut>,
    header: OmenHeader,

    // Configuration (dimensions, quantization mode, etc.)
    config: HashMap<String, u64>,

    // WAL for durability
    wal: Wal,

    // Serialized HNSW index (persisted on checkpoint, loaded on open)
    hnsw_index_bytes: Option<Vec<u8>>,

    // Omen Manifest
    manifest: OmenManifest,

    // Split .vecs file for incremental checkpoint
    vec_mmap: Option<MmapMut>,
    vec_file: Option<File>,
    vec_path: PathBuf,
    records_path: PathBuf,

    // Accumulated dirty slots since last full flush. Unioned into each auto-checkpoint's
    // .records snapshot so crash recovery can rebuild partial segments correctly across
    // multiple auto-checkpoints. Reset on full flush (write_omen_manifest /
    // checkpoint_from_snapshot). Avoids reading the .records file on every auto-checkpoint.
    accumulated_dirty: RoaringBitmap,
}

impl OmenFile {
    /// Compute .omen path by appending extension (preserves full filename)
    ///
    /// Handles filenames with multiple dots (e.g., `test.db_64` → `test.db_64.omen`)
    /// by appending `.omen` rather than replacing the extension.
    #[must_use]
    pub fn compute_omen_path(path: &Path) -> PathBuf {
        if path.extension().is_some_and(|ext| ext == "omen") {
            path.to_path_buf()
        } else {
            let mut omen = path.as_os_str().to_os_string();
            omen.push(".omen");
            PathBuf::from(omen)
        }
    }

    /// Compute .wal path by appending extension
    fn compute_wal_path(path: &Path) -> PathBuf {
        let mut wal = path.as_os_str().to_os_string();
        wal.push(".wal");
        PathBuf::from(wal)
    }

    /// Compute .vecs path by appending extension
    fn compute_vecs_path(path: &Path) -> PathBuf {
        let mut p = path.as_os_str().to_os_string();
        p.push(".vecs");
        PathBuf::from(p)
    }

    /// Compute .records path (slim records snapshot) by appending extension
    fn compute_records_path(path: &Path) -> PathBuf {
        let mut p = path.as_os_str().to_os_string();
        p.push(".records");
        PathBuf::from(p)
    }

    pub fn create(path: impl AsRef<Path>, dimensions: u32) -> io::Result<Self> {
        let path = path.as_ref();
        let omen_path = Self::compute_omen_path(path);
        let wal_path = Self::compute_wal_path(path);
        let vec_path = Self::compute_vecs_path(path);
        let records_path = Self::compute_records_path(path);

        let mut opts = OpenOptions::new();
        opts.read(true).write(true).create(true).truncate(true);
        configure_open_options(&mut opts);
        let mut file = opts.open(&omen_path)?;
        lock_exclusive(&file)?;

        let header = OmenHeader::new(dimensions);
        file.write_all(&header.to_bytes())?;

        // Write initial empty Manifest with header and Footer
        let manifest = OmenManifest::new();
        let manifest_bytes = postcard::to_allocvec(&manifest)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let manifest_header = ManifestHeader::new(&manifest_bytes);
        let manifest_offset = file.stream_position()?;
        file.write_all(&manifest_header.to_bytes())?;
        file.write_all(&manifest_bytes)?;

        let total_len = file.stream_position()?;
        let footer = OmenFooter::new(manifest_offset, total_len);
        file.write_all(&footer.to_bytes())?;

        file.sync_all()?;

        Ok(Self {
            path: omen_path,
            file: Some(file),
            mmap: None,
            header,
            config: HashMap::from([("dimensions".to_string(), u64::from(dimensions))]),
            wal: Wal::open(&wal_path)?,
            hnsw_index_bytes: None,
            manifest,
            vec_mmap: None,
            vec_file: None,
            vec_path,
            records_path,
            accumulated_dirty: RoaringBitmap::new(),
        })
    }

    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let omen_path = Self::compute_omen_path(path);
        let wal_path = Self::compute_wal_path(path);
        let vec_path = Self::compute_vecs_path(path);
        let records_path = Self::compute_records_path(path);

        let mut opts = OpenOptions::new();
        opts.read(true).write(true);
        configure_open_options(&mut opts);
        let mut file = opts.open(&omen_path)?;
        lock_exclusive(&file)?;

        // Try to read footer from the end of the file
        let file_len = file.metadata()?.len();
        let mut footer = None;
        if file_len >= (HEADER_SIZE + OmenFooter::SIZE) as u64 {
            // Seek to absolute end - Footer size
            #[allow(clippy::cast_possible_wrap)]
            file.seek(SeekFrom::End(-(OmenFooter::SIZE as i64)))?;
            let mut footer_buf = [0u8; OmenFooter::SIZE];
            file.read_exact(&mut footer_buf)?;
            let f = OmenFooter::from_bytes(&footer_buf)?;
            if f.verify() {
                footer = Some(f);
            }
        }

        // Mandatory Footer check (0.0.x Policy: no shims)
        let footer = footer.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid or missing OmenFooter. Legacy V1 files are no longer supported.",
            )
        })?;

        let mut header_buf = [0u8; HEADER_SIZE];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut header_buf)?;
        let header = OmenHeader::from_bytes(&header_buf)?;

        let mmap = if file_len > HEADER_SIZE as u64 {
            // SAFETY: File exclusively locked via flock, valid file descriptor
            Some(unsafe { MmapMut::map_mut(&file)? })
        } else {
            None
        };

        let wal = Wal::open(&wal_path)?;

        let mut manifest = OmenManifest::new();

        // Load manifest with CRC validation
        if let Some(ref mmap) = mmap {
            let manifest_offset = footer.manifest_offset as usize;
            let header_end = manifest_offset
                .checked_add(ManifestHeader::SIZE)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "Manifest offset overflow")
                })?;

            if header_end > mmap.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Manifest header out of bounds",
                ));
            }

            let manifest_header = ManifestHeader::from_bytes(&mmap[manifest_offset..header_end])?;
            let data_end = header_end
                .checked_add(manifest_header.length as usize)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "Manifest length overflow")
                })?;

            if data_end > mmap.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Manifest data out of bounds",
                ));
            }

            let manifest_bytes = &mmap[header_end..data_end];

            if !manifest_header.verify(manifest_bytes) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Manifest CRC mismatch - data may be corrupted",
                ));
            }

            manifest = postcard::from_bytes::<OmenManifest>(manifest_bytes).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Failed to decode manifest: {e}"),
                )
            })?;
        }

        // Note: vectors, id_to_index, index_to_id, metadata are now managed by RecordStore.
        // Only config is loaded into OmenFile.
        let config = manifest.config.clone();

        // Config values from manifest (source of truth for append-only)
        let count = manifest
            .config
            .get("count")
            .copied()
            .unwrap_or(manifest.id_to_index.len() as u64);
        let dimensions = manifest
            .config
            .get("dimensions")
            .copied()
            .unwrap_or(u64::from(header.dimensions));
        let dimensions = u32::try_from(dimensions).unwrap_or(header.dimensions);
        let hnsw_m = manifest
            .config
            .get("hnsw_m")
            .copied()
            .unwrap_or(u64::from(header.hnsw_m));
        let hnsw_m = u32::try_from(hnsw_m).unwrap_or(header.hnsw_m);
        let hnsw_ef_construction = manifest
            .config
            .get("hnsw_ef_construction")
            .copied()
            .unwrap_or(u64::from(header.hnsw_ef_construction));
        let hnsw_ef_construction =
            u32::try_from(hnsw_ef_construction).unwrap_or(header.hnsw_ef_construction);
        let hnsw_ef_search = manifest
            .config
            .get("hnsw_ef_search")
            .copied()
            .unwrap_or(u64::from(header.hnsw_ef_search));
        let hnsw_ef_search = u32::try_from(hnsw_ef_search).unwrap_or(header.hnsw_ef_search);
        let metric = manifest.config.get("metric").map_or(header.metric, |&v| {
            crate::omen::header::Metric::from(v as u8)
        });

        // Load HNSW index bytes from manifest (if mmap exists)
        let hnsw_index_bytes = mmap.as_ref().and_then(|m| {
            manifest.nodes.iter().find_map(|location| {
                if location.segment_type == SegmentType::IndexMetadata {
                    let start = location.offset as usize;
                    let end = start + location.length as usize;
                    if end <= m.len() {
                        return Some(m[start..end].to_vec());
                    }
                }
                None
            })
        });

        // Update header from manifest (source of truth for append-only)
        let mut header = header;
        header.count = count;
        header.dimensions = dimensions;
        header.hnsw_m = hnsw_m;
        header.hnsw_ef_construction = hnsw_ef_construction;
        header.hnsw_ef_search = hnsw_ef_search;
        header.metric = metric;

        // Open .vecs file if it exists (incremental checkpoint format)
        let has_vec_file_flag = manifest.config.get("vec_file").copied().unwrap_or(0) == 1;
        let (vec_mmap, vec_file) = if has_vec_file_flag && vec_path.exists() {
            let mut opts = OpenOptions::new();
            opts.read(true).write(true);
            configure_open_options(&mut opts);
            let vf = opts.open(&vec_path)?;
            let vf_len = vf.metadata()?.len();
            let vm = if vf_len > 0 {
                // SAFETY: .vecs exclusively owned via .omen lock
                Some(unsafe { MmapMut::map_mut(&vf)? })
            } else {
                None
            };
            (vm, Some(vf))
        } else {
            (None, None)
        };

        // Note: WAL replay happens at VectorStore level, not here (Phase 5 architecture)
        // State (vectors, ids, deleted) is managed by RecordStore via load_persisted_snapshot()
        Ok(Self {
            path: omen_path,
            file: Some(file),
            mmap,
            header,
            config,
            wal,
            hnsw_index_bytes,
            manifest,
            vec_mmap,
            vec_file,
            vec_path,
            records_path,
            accumulated_dirty: RoaringBitmap::new(),
        })
    }

    // Note: insert(), find_nearest(), search() removed in Phase 5.
    // VectorStore uses wal_append_insert() for inserts and RecordStore for search.

    /// Get vector count
    #[must_use]
    pub fn len(&self) -> u64 {
        self.header.count
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.header.count == 0
    }

    /// Get dimensions
    #[must_use]
    pub fn dimensions(&self) -> u32 {
        self.header.dimensions
    }

    // Note: checkpoint() removed in Phase 5.
    // VectorStore uses checkpoint_from_snapshot() which takes data from RecordStore.
}

impl crate::omen::StorageBackend for OmenFile {
    fn dimensions(&self) -> usize {
        self.header.dimensions as usize
    }

    fn metric(&self) -> crate::omen::Metric {
        self.header.metric
    }

    fn set_dimensions(&mut self, dimensions: u32) -> anyhow::Result<()> {
        self.set_dimensions(dimensions);
        Ok(())
    }

    fn set_metric(&mut self, metric: crate::omen::Metric) -> anyhow::Result<()> {
        self.set_metric(metric);
        Ok(())
    }

    fn set_hnsw_params(
        &mut self,
        m: u16,
        ef_construction: u16,
        ef_search: u16,
    ) -> anyhow::Result<()> {
        self.set_hnsw_params(m, ef_construction, ef_search);
        Ok(())
    }

    fn log_insert(
        &mut self,
        id: &str,
        vector: &[f32],
        metadata: &serde_json::Value,
    ) -> anyhow::Result<()> {
        let metadata_bytes = serde_json::to_vec(metadata)?;
        self.wal_append_insert(id, vector, Some(&metadata_bytes))
            .map_err(Into::into)
    }

    fn log_delete(&mut self, id: &str) -> anyhow::Result<()> {
        self.wal_append_delete(id).map_err(Into::into)
    }

    fn log_insert_edge(
        &mut self,
        from_id: &str,
        to_id: &str,
        edge_type: &str,
        weight: f32,
        metadata: Option<&[u8]>,
    ) -> anyhow::Result<()> {
        self.wal_append_insert_edge(from_id, to_id, edge_type, weight, metadata)
            .map_err(Into::into)
    }

    fn log_delete_edge(
        &mut self,
        from_id: &str,
        to_id: &str,
        edge_type: &str,
    ) -> anyhow::Result<()> {
        self.wal_append_delete_edge(from_id, to_id, edge_type)
            .map_err(Into::into)
    }

    fn checkpoint(&mut self) -> anyhow::Result<()> {
        // checkpoint_from_snapshot is the new way (Phase 5).
        // For now, this is a no-op or error until we wire it up.
        anyhow::bail!("OmenFile::checkpoint requires a snapshot (use checkpoint_from_snapshot)")
    }

    fn sync(&mut self) -> anyhow::Result<()> {
        self.wal_sync().map_err(Into::into)
    }

    fn put_config(&mut self, key: &str, value: u64) -> anyhow::Result<()> {
        self.put_config(key, value)
    }

    fn wal_len(&self) -> usize {
        self.wal_len() as usize
    }

    fn has_vec_file(&self) -> bool {
        self.has_vec_file()
    }

    fn checkpoint_vectors_only(
        &mut self,
        records: &crate::vector::store::record_store::RecordStore,
        dirty_slots: &roaring::RoaringBitmap,
    ) -> anyhow::Result<()> {
        self.checkpoint_vectors_only(records, dirty_slots)
            .map_err(Into::into)
    }

    fn checkpoint_incremental(
        &mut self,
        records: &crate::vector::store::record_store::RecordStore,
        dirty_slots: &roaring::RoaringBitmap,
        options: crate::omen::CheckpointOptions,
    ) -> anyhow::Result<()> {
        self.checkpoint_incremental(records, dirty_slots, options)
            .map_err(Into::into)
    }

    fn checkpoint_full(
        &mut self,
        records: &crate::vector::store::record_store::RecordStore,
        options: crate::omen::CheckpointOptions,
    ) -> anyhow::Result<()> {
        self.checkpoint_full(records, options).map_err(Into::into)
    }
}

// Note: Many methods removed in Phase 5. VectorStore uses RecordStore for state.
// OmenFile is now pure I/O: WAL + checkpoint_from_snapshot.

impl OmenFile {
    /// Store configuration value
    pub fn put_config(&mut self, key: &str, value: u64) -> Result<()> {
        self.config.insert(key.to_string(), value);
        // Sync to header
        match key {
            "dimensions" => {
                self.header.dimensions =
                    u32::try_from(value).map_err(|_| anyhow::anyhow!("dimensions overflow u32"))?;
            }
            "quantization" => {
                self.header.quantization = crate::omen::header::QuantizationCode::from(value as u8);
            }
            _ => {}
        }
        Ok(())
    }

    /// Get configuration value
    pub fn get_config(&self, key: &str) -> Result<Option<u64>> {
        Ok(self.config.get(key).copied())
    }

    /// Store quantization mode
    ///
    /// Mode values: 0=none, 1=sq8
    pub fn put_quantization_mode(&mut self, mode: u64) -> Result<()> {
        self.put_config("quantization", mode)
    }

    /// Get quantization mode
    ///
    /// Returns: 0=none, 1=sq8
    pub fn get_quantization_mode(&self) -> Result<Option<u64>> {
        self.get_config("quantization")
    }

    /// Check if store was created with quantization
    pub fn is_quantized(&self) -> Result<bool> {
        Ok(self.get_quantization_mode()?.unwrap_or(0) > 0)
    }

    // Note: load_all_metadata, load_all_id_mappings, put_deleted, is_deleted,
    // remove_deleted, load_all_deleted removed in Phase 5.
    // VectorStore uses RecordStore for state, loads via load_persisted_snapshot().

    /// Store serialized HNSW index bytes
    ///
    /// The bytes are persisted on the next checkpoint/flush.
    /// `VectorStore` serializes `HNSWIndex` and stores it here.
    pub fn put_hnsw_index(&mut self, bytes: Vec<u8>) {
        self.hnsw_index_bytes = Some(bytes);
    }

    /// Get serialized HNSW index bytes (if present)
    ///
    /// Returns the bytes previously stored by `put_hnsw_index()`,
    /// or loaded from disk on open.
    #[must_use]
    pub fn get_hnsw_index(&self) -> Option<&[u8]> {
        self.hnsw_index_bytes.as_deref()
    }

    /// Check if HNSW index is stored
    #[must_use]
    pub fn has_hnsw_index(&self) -> bool {
        self.hnsw_index_bytes.is_some()
    }

    /// Update HNSW parameters in the header
    ///
    /// These values are persisted to disk on the next checkpoint/flush.
    pub fn set_hnsw_params(&mut self, m: u16, ef_construction: u16, ef_search: u16) {
        self.header.hnsw_m = m as u32;
        self.header.hnsw_ef_construction = ef_construction as u32;
        self.header.hnsw_ef_search = ef_search as u32;
    }

    /// Update distance metric in the header
    ///
    /// Persisted to disk on the next checkpoint/flush.
    pub fn set_metric(&mut self, metric: Metric) {
        self.header.metric = metric;
    }

    /// Set dimensions in header
    ///
    /// Used when dimensions are inferred from vectors after initial creation.
    pub fn set_dimensions(&mut self, dimensions: u32) {
        self.header.dimensions = dimensions;
    }

    /// Get storage path
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get reference to the header
    #[must_use]
    pub fn header(&self) -> &OmenHeader {
        &self.header
    }

    // Note: flush(), delete_ratio(), needs_compaction(), compact() removed in Phase 5.
    // VectorStore handles flushing via checkpoint_from_snapshot().
    // Compaction will be implemented at VectorStore level using RecordStore data.
}

/// Persisted MUVERA configuration for multi-vector storage.
#[derive(Debug, Clone)]
pub struct PersistedMuveraConfig {
    pub repetitions: u8,
    pub partition_bits: u8,
    pub seed: u64,
    pub token_dim: usize,
    pub d_proj: Option<u8>,
    pub pool_factor: Option<u8>,
    pub max_tokens: Option<usize>,
}

/// Snapshot data loaded from OmenFile
#[derive(Debug, Default)]
pub struct OmenSnapshot {
    /// Vectors loaded from storage
    pub vectors: Vec<Option<Vec<f32>>>,
    /// ID to slot mappings
    pub id_to_slot: HashMap<String, u32>,
    /// Deleted slot bitmap (as Vec for compatibility)
    pub deleted: Vec<u32>,
    /// Metadata by slot
    pub metadata: HashMap<u32, serde_json::Value>,
    /// Vector dimensions
    pub dimensions: u32,
    /// HNSW index bytes (if persisted)
    pub hnsw_bytes: Option<Vec<u8>>,
    /// Serialized MetadataIndex (if persisted)
    pub metadata_index_bytes: Option<Vec<u8>>,
    /// Multi-vector token data (if persisted)
    pub multivec_bytes: Option<Vec<u8>>,
    /// Multi-vector offset table (if persisted)
    pub multivec_offsets: Option<Vec<u8>>,
    /// MUVERA config
    pub multivec_config: Option<PersistedMuveraConfig>,
    /// Serialized SparseIndex (if persisted)
    pub sparse_index_bytes: Option<Vec<u8>>,
    /// Serialized EdgeStore (if persisted)
    pub edge_store_bytes: Option<Vec<u8>>,
}

/// Slim records snapshot: ID mappings, deleted slots, metadata, and dirty slots since last flush.
///
/// Written alongside `.vecs` during auto-checkpoint to allow WAL truncation without
/// a full manifest rewrite. Recovery loads this snapshot when it is newer than the manifest.
/// The `dirty_since_flush` field accumulates across checkpoints so partial segment rebuild
/// works correctly even when the WAL has been truncated multiple times.
///
/// `wal_truncation_epoch` is the WAL's epoch at the time this snapshot was written (after
/// the WAL was truncated). On recovery, if the WAL's current epoch > this value, the WAL
/// was truncated again after this snapshot, meaning current WAL entries are new and must be
/// replayed. If equal, the WAL was not truncated after the snapshot — entries are stale.
#[derive(Debug, Default)]
pub struct SlimRecordsSnapshot {
    pub id_to_slot: HashMap<String, u32>,
    pub deleted: Vec<u32>,
    pub metadata: HashMap<u32, serde_json::Value>,
    pub dirty_since_flush: RoaringBitmap,
    pub wal_truncation_epoch: u64,
}

/// Options for checkpoint_from_snapshot
#[derive(Debug, Default)]
pub struct CheckpointOptions<'a> {
    /// Serialized HNSW index bytes
    pub hnsw_bytes: Option<&'a [u8]>,
    /// Serialized MetadataIndex bytes
    pub metadata_index_bytes: Option<&'a [u8]>,
    /// Multi-vector token data (from MultiVecStorage::vectors_to_bytes)
    pub multivec_bytes: Option<&'a [u8]>,
    /// Multi-vector offset table (from MultiVecStorage::offsets_to_bytes)
    pub multivec_offsets: Option<&'a [u8]>,
    /// MUVERA config
    pub multivec_config: Option<PersistedMuveraConfig>,
    /// Serialized SparseIndex bytes
    pub sparse_index_bytes: Option<&'a [u8]>,
    /// Serialized EdgeStore bytes
    pub edge_store_bytes: Option<&'a [u8]>,
}

impl OmenFile {
    /// Append insert entry to WAL without updating internal state
    ///
    /// WAL-only, no state mutation. State is managed by RecordStore.
    /// Note: Does not sync to disk. Call `wal_sync()` for durability.
    pub fn wal_append_insert(
        &mut self,
        id: &str,
        vector: &[f32],
        metadata: Option<&[u8]>,
    ) -> io::Result<()> {
        let metadata_bytes = metadata.unwrap_or(b"{}");
        let entry = WalEntry::insert_node(0, id, 0, vector, metadata_bytes);
        self.wal.append(entry)
    }

    /// Append delete entry to WAL without updating internal state
    ///
    /// WAL-only, no state mutation. State is managed by RecordStore.
    /// Note: Does not sync to disk. Call `wal_sync()` for durability.
    pub fn wal_append_delete(&mut self, id: &str) -> io::Result<()> {
        self.wal.append(WalEntry::delete_node(0, id))
    }

    /// Append insert-edge entry to WAL.
    pub fn wal_append_insert_edge(
        &mut self,
        from_id: &str,
        to_id: &str,
        edge_type: &str,
        weight: f32,
        metadata: Option<&[u8]>,
    ) -> io::Result<()> {
        self.wal.append(WalEntry::insert_edge(
            0, from_id, to_id, edge_type, weight, metadata,
        ))
    }

    /// Append delete-edge entry to WAL.
    pub fn wal_append_delete_edge(
        &mut self,
        from_id: &str,
        to_id: &str,
        edge_type: &str,
    ) -> io::Result<()> {
        self.wal
            .append(WalEntry::delete_edge(0, from_id, to_id, edge_type))
    }

    /// Sync WAL to disk for durability
    pub fn wal_sync(&mut self) -> io::Result<()> {
        self.wal.sync()
    }

    /// Get pending WAL entries (entries after last checkpoint)
    ///
    /// These entries have not been persisted to the checkpoint yet.
    /// VectorStore uses this to replay WAL directly into RecordStore.
    pub fn pending_wal_entries(&mut self) -> io::Result<Vec<WalEntry>> {
        self.wal.entries_after_checkpoint()
    }

    /// Load snapshot from persisted data only (manifest + mmap)
    ///
    /// Does NOT include WAL entries - caller must replay WAL separately.
    /// This is the Phase 5 API where state is managed externally by RecordStore.
    pub fn load_persisted_snapshot(&self) -> OmenSnapshot {
        self.load_persisted_snapshot_with_live_slots(None)
    }

    /// Load the durable snapshot, treating any hinted live slots as authoritative for `.vecs`.
    ///
    /// Recovery uses this when a newer slim `.records` snapshot exists: slots introduced after
    /// the last full manifest flush may legitimately contain all-zero vectors, so the stale
    /// manifest alone is not enough to decide whether a zero-filled slot is live.
    pub(crate) fn load_persisted_snapshot_with_live_slots(
        &self,
        extra_live_slots: Option<&RoaringBitmap>,
    ) -> OmenSnapshot {
        let dim = self.header.dimensions as usize;
        let mut snapshot = OmenSnapshot {
            dimensions: self.header.dimensions,
            ..Default::default()
        };
        let mut known_live_slots: RoaringBitmap =
            self.manifest.id_to_index.values().copied().collect();
        if let Some(extra_live_slots) = extra_live_slots {
            known_live_slots |= extra_live_slots;
        }

        // Load vectors from .vecs mmap (incremental format) or .omen manifest (legacy)
        let has_vec_file = self.manifest.config.get("vec_file").copied().unwrap_or(0) == 1;
        if has_vec_file {
            if let Some(ref vm) = self.vec_mmap {
                // Fixed-slot format: slot i at offset i * dim * 4
                let slot_bytes = dim * 4;
                if slot_bytes > 0 {
                    let slot_count = vm.len() / slot_bytes;
                    snapshot.vectors.resize(slot_count, None);
                    for slot in 0..slot_count {
                        if self.manifest.deleted.contains(slot as u32) {
                            continue;
                        }
                        let start = slot * slot_bytes;
                        let end = start + slot_bytes;
                        if end <= vm.len() {
                            let vec = read_vector_from_bytes(&vm[start..end], dim);
                            // Zero-filled slots can still be live when a newer slim `.records`
                            // snapshot exists (for example, legitimate zero vectors or sparse
                            // placeholders written after the last full manifest flush).
                            if vec.iter().any(|&v| v != 0.0)
                                || known_live_slots.contains(slot as u32)
                            {
                                snapshot.vectors[slot] = Some(vec);
                            }
                        }
                    }
                }
            }

            // Load HNSW index bytes from .omen manifest nodes
            if let Some(ref mmap) = self.mmap {
                for location in &self.manifest.nodes {
                    if location.segment_type == SegmentType::IndexMetadata {
                        let start = location.offset as usize;
                        let end = start + location.length as usize;
                        if end <= mmap.len() {
                            snapshot.hnsw_bytes = Some(mmap[start..end].to_vec());
                            break;
                        }
                    }
                }
            }
        } else if let Some(ref mmap) = self.mmap {
            // Legacy format: vectors stored in .omen via NodeLocations
            for (idx, location) in self.manifest.nodes.iter().enumerate() {
                if location.segment_type == SegmentType::Vectors {
                    while snapshot.vectors.len() <= idx {
                        snapshot.vectors.push(None);
                    }
                    // Skip deleted sentinels (length=0 + in deleted bitmap)
                    if location.length == 0 && self.manifest.deleted.contains(idx as u32) {
                        continue;
                    }
                    let start = location.offset as usize;
                    let end = start + location.length as usize;
                    if end <= mmap.len() {
                        let vec = read_vector_from_bytes(&mmap[start..end], dim);
                        // Infer dimensions from first vector if header says 0
                        if snapshot.dimensions == 0 && !vec.is_empty() {
                            snapshot.dimensions = u32::try_from(vec.len()).unwrap_or(u32::MAX);
                        }
                        snapshot.vectors[idx] = Some(vec);
                    }
                }
            }

            // Load HNSW index bytes
            for location in &self.manifest.nodes {
                if location.segment_type == SegmentType::IndexMetadata {
                    let start = location.offset as usize;
                    let end = start + location.length as usize;
                    if end <= mmap.len() {
                        snapshot.hnsw_bytes = Some(mmap[start..end].to_vec());
                        break;
                    }
                }
            }
        }

        // Load ID mappings from manifest
        snapshot.id_to_slot.clone_from(&self.manifest.id_to_index);

        // Load deleted bitmap from manifest (RoaringBitmap -> Vec<u32>)
        snapshot.deleted = self.manifest.deleted.iter().collect();

        // Load metadata from manifest (bytes -> JsonValue)
        for (&idx, bytes) in &self.manifest.metadata {
            match serde_json::from_slice(bytes) {
                Ok(json) => {
                    snapshot.metadata.insert(idx, json);
                }
                Err(e) => {
                    tracing::warn!(
                        "Corrupt metadata for slot {} during manifest load: {}",
                        idx,
                        e
                    );
                }
            }
        }

        // Load serialized MetadataIndex if available
        snapshot
            .metadata_index_bytes
            .clone_from(&self.manifest.metadata_index);

        // Load multi-vector data if present
        if let Some(ref mmap) = self.mmap {
            // Load MultiVectors segment (token vectors)
            for location in &self.manifest.nodes {
                if location.segment_type == SegmentType::MultiVectors {
                    let start = location.offset as usize;
                    let end = start + location.length as usize;
                    if end <= mmap.len() {
                        snapshot.multivec_bytes = Some(mmap[start..end].to_vec());
                        break;
                    }
                }
            }
        }

        // Load multi-vector offsets from manifest
        snapshot
            .multivec_offsets
            .clone_from(&self.manifest.multivec_offsets);

        // Load sparse index from manifest
        snapshot
            .sparse_index_bytes
            .clone_from(&self.manifest.sparse_index_bytes);

        // Load edge store from manifest
        snapshot
            .edge_store_bytes
            .clone_from(&self.manifest.edge_store_bytes);

        // Extract MUVERA config from manifest.config if present
        let reps = self.manifest.config.get("muvera_repetitions").copied();
        let bits = self.manifest.config.get("muvera_partition_bits").copied();
        let seed = self.manifest.config.get("muvera_seed").copied();
        let token_dim = self.manifest.config.get("muvera_token_dim").copied();
        let d_proj = self.manifest.config.get("muvera_d_proj").map(|&v| v as u8);
        let pool_factor = self
            .manifest
            .config
            .get("muvera_pool_factor")
            .map(|&v| v as u8);
        let max_tokens = self
            .manifest
            .config
            .get("muvera_max_tokens")
            .map(|&v| v as usize);

        if let (Some(reps), Some(bits), Some(seed), Some(token_dim)) = (reps, bits, seed, token_dim)
        {
            snapshot.multivec_config = Some(PersistedMuveraConfig {
                repetitions: reps as u8,
                partition_bits: bits as u8,
                seed,
                token_dim: token_dim as usize,
                d_proj,
                pool_factor,
                max_tokens,
            });
        }

        snapshot
    }

    // Note: load_snapshot() removed in Phase 5. VectorStore uses load_persisted_snapshot().

    /// Returns true if the .records snapshot exists and is at least as recent as the .omen manifest.
    ///
    /// Used during recovery to decide whether to use the slim snapshot's ID/metadata mappings
    /// instead of the manifest's (happens when auto-checkpoint ran but flush() hasn't been called).
    ///
    /// Uses `>=` (not `>`) to handle 1-second precision filesystems (HFS+, ext3, NFS): if
    /// `.records` and `.omen` have equal mtime, the snapshot was written after or during the
    /// same flush tick and is still valid. Using `>` would cause data loss on these filesystems
    /// because `checkpoint_vectors_only` truncates the WAL after writing `.records`, so ignoring
    /// an equal-mtime snapshot leaves an empty WAL and no in-between vectors.
    pub fn records_newer_than_omen(&self) -> bool {
        let records_mtime = match std::fs::metadata(&self.records_path).and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => return false,
        };
        let omen_mtime = match std::fs::metadata(&self.path).and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => return false,
        };
        records_mtime >= omen_mtime
    }

    /// Write a slim records snapshot atomically (tmp → fsync → rename).
    ///
    /// Format: magic "OREC" + version 2 + id_to_slot + deleted bitmap +
    /// metadata JSON + dirty_since_flush bitmap.
    ///
    /// `dirty_since_flush` should be the union of all dirty slots since the last flush(),
    /// accumulated by the caller across multiple auto-checkpoints.
    pub fn write_records_snapshot(
        &self,
        records: &RecordStore,
        dirty_since_flush: &RoaringBitmap,
        wal_truncation_epoch: u64,
    ) -> io::Result<()> {
        const MAGIC: &[u8; 4] = b"OREC";
        const VERSION: u32 = 2;

        let tmp_path = {
            let mut p = self.records_path.as_os_str().to_os_string();
            p.push(".tmp");
            PathBuf::from(p)
        };

        {
            let mut file = std::fs::File::create(&tmp_path)?;
            {
                let mut w = BufWriter::with_capacity(1 << 20, &mut file);

                // Header
                w.write_all(MAGIC)?;
                w.write_all(&VERSION.to_le_bytes())?;

                // id_to_slot section (zero-copy borrow — no HashMap clone)
                let id_to_slot = records.id_to_slot_ref();
                let count = id_to_slot.len() as u32;
                w.write_all(&count.to_le_bytes())?;
                for (id, slot) in &id_to_slot {
                    let id_bytes = id.as_bytes();
                    w.write_all(&(id_bytes.len() as u32).to_le_bytes())?;
                    w.write_all(id_bytes)?;
                    w.write_all(&slot.to_le_bytes())?;
                }

                // deleted bitmap section
                let deleted_bitmap = records.deleted_bitmap();
                let mut bitmap_buf: Vec<u8> = Vec::new();
                deleted_bitmap.serialize_into(&mut bitmap_buf)?;
                w.write_all(&(bitmap_buf.len() as u32).to_le_bytes())?;
                w.write_all(&bitmap_buf)?;

                // metadata section (zero-copy iterator — no HashMap clone)
                let metadata = records.iter_metadata();
                let meta_count = metadata.len() as u32;
                w.write_all(&meta_count.to_le_bytes())?;
                for (slot, value) in &metadata {
                    let json_bytes = serde_json::to_vec(value)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                    w.write_all(&slot.to_le_bytes())?;
                    w.write_all(&(json_bytes.len() as u32).to_le_bytes())?;
                    w.write_all(&json_bytes)?;
                }

                // dirty_since_flush bitmap section
                let mut dirty_buf: Vec<u8> = Vec::new();
                dirty_since_flush.serialize_into(&mut dirty_buf)?;
                w.write_all(&(dirty_buf.len() as u32).to_le_bytes())?;
                w.write_all(&dirty_buf)?;

                // wal_truncation_epoch (added to v2 format; absent in older v2 snapshots — defaults to 0 on load)
                w.write_all(&wal_truncation_epoch.to_le_bytes())?;

                w.flush()?;
            }
            file.sync_all()?;
        }

        std::fs::rename(&tmp_path, &self.records_path).inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp_path);
        })?;

        // Fsync parent directory to durabilize the rename
        if let Some(parent) = self.records_path.parent()
            && let Ok(dir) = std::fs::File::open(parent)
        {
            let _ = dir.sync_all();
        }

        Ok(())
    }

    /// Load the slim records snapshot if it exists.
    ///
    /// Returns `None` if the file doesn't exist or is corrupted (caller falls back to manifest).
    pub fn load_records_snapshot(&self) -> io::Result<Option<SlimRecordsSnapshot>> {
        if !self.records_path.exists() {
            return Ok(None);
        }

        let data = match std::fs::read(&self.records_path) {
            Ok(d) => d,
            Err(_) => return Ok(None),
        };

        if data.len() < 8 {
            return Ok(None);
        }

        const MAGIC: &[u8; 4] = b"OREC";
        if &data[0..4] != MAGIC {
            return Ok(None);
        }
        let version = u32::from_le_bytes(
            data[4..8]
                .try_into()
                .expect("data.len() >= 8 was checked above"),
        );
        if version != 2 {
            tracing::warn!(version, "Unrecognized slim records snapshot version");
            return Ok(None);
        }

        // Size limits match WAL parser constants to prevent OOM on corrupt files
        const MAX_ID_LEN: usize = 65536;
        const MAX_BITMAP_BYTES: usize = 16 << 20; // 16 MB
        const MAX_METADATA_JSON_BYTES: usize = 16 << 20; // 16 MB
        const MAX_RECORD_COUNT: usize = 100_000_000; // 100M

        // Physical bound: each entry requires at minimum 8 bytes on disk.
        // Cap with_capacity to avoid multi-GB pre-allocation from a corrupt count field.
        let max_entries_by_size = data.len() / 8;

        let mut cursor = std::io::Cursor::new(&data[8..]);
        let mut buf4 = [0u8; 4];

        // id_to_slot section
        cursor.read_exact(&mut buf4)?;
        let count = u32::from_le_bytes(buf4) as usize;
        if count > MAX_RECORD_COUNT {
            return Ok(None);
        }
        let mut id_to_slot = HashMap::with_capacity(count.min(max_entries_by_size));
        for _ in 0..count {
            cursor.read_exact(&mut buf4)?;
            let id_len = u32::from_le_bytes(buf4) as usize;
            if id_len > MAX_ID_LEN {
                return Ok(None);
            }
            let mut id_buf = vec![0u8; id_len];
            cursor.read_exact(&mut id_buf)?;
            let id = String::from_utf8(id_buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            cursor.read_exact(&mut buf4)?;
            let slot = u32::from_le_bytes(buf4);
            id_to_slot.insert(id, slot);
        }

        // deleted bitmap section
        cursor.read_exact(&mut buf4)?;
        let bitmap_len = u32::from_le_bytes(buf4) as usize;
        if bitmap_len > MAX_BITMAP_BYTES {
            return Ok(None);
        }
        let mut bitmap_buf = vec![0u8; bitmap_len];
        cursor.read_exact(&mut bitmap_buf)?;
        let deleted_bitmap = RoaringBitmap::deserialize_from(&bitmap_buf[..])?;
        let deleted: Vec<u32> = deleted_bitmap.iter().collect();

        // metadata section
        cursor.read_exact(&mut buf4)?;
        let meta_count = u32::from_le_bytes(buf4) as usize;
        if meta_count > MAX_RECORD_COUNT {
            return Ok(None);
        }
        let mut metadata = HashMap::with_capacity(meta_count.min(max_entries_by_size));
        for _ in 0..meta_count {
            cursor.read_exact(&mut buf4)?;
            let slot = u32::from_le_bytes(buf4);
            cursor.read_exact(&mut buf4)?;
            let json_len = u32::from_le_bytes(buf4) as usize;
            if json_len > MAX_METADATA_JSON_BYTES {
                return Ok(None);
            }
            let mut json_buf = vec![0u8; json_len];
            cursor.read_exact(&mut json_buf)?;
            match serde_json::from_slice::<serde_json::Value>(&json_buf) {
                Ok(value) => {
                    metadata.insert(slot, value);
                }
                Err(e) => {
                    tracing::warn!("Corrupt metadata at slot {slot} in records snapshot: {e}");
                }
            }
        }

        // dirty_since_flush bitmap section
        cursor.read_exact(&mut buf4)?;
        let dirty_len = u32::from_le_bytes(buf4) as usize;
        if dirty_len > MAX_BITMAP_BYTES {
            return Ok(None);
        }
        let mut dirty_buf = vec![0u8; dirty_len];
        cursor.read_exact(&mut dirty_buf)?;
        let dirty_since_flush = RoaringBitmap::deserialize_from(&dirty_buf[..])?;

        // wal_truncation_epoch section (absent in older v2 snapshots — defaults to 0)
        let mut epoch_buf = [0u8; 8];
        let wal_truncation_epoch = if cursor.read_exact(&mut epoch_buf).is_ok() {
            u64::from_le_bytes(epoch_buf)
        } else {
            0
        };

        Ok(Some(SlimRecordsSnapshot {
            id_to_slot,
            deleted,
            metadata,
            dirty_since_flush,
            wal_truncation_epoch,
        }))
    }

    /// Check if this storage has a .vecs file for incremental checkpoints
    #[must_use]
    pub fn has_vec_file(&self) -> bool {
        self.vec_mmap.is_some()
    }

    /// Full checkpoint: create .vecs file from all vectors in RecordStore.
    ///
    /// Used on first flush (migration from legacy format) and when .vecs doesn't exist.
    /// After this, future checkpoints use the incremental path.
    pub fn checkpoint_full(
        &mut self,
        records: &RecordStore,
        options: CheckpointOptions<'_>,
    ) -> io::Result<()> {
        let dim = self.header.dimensions as usize;
        let slot_count = records.slot_count() as usize;
        let slot_bytes = dim * 4;

        // Create .vecs file with all vector data
        if dim > 0 && slot_count > 0 {
            let total_size = slot_count * slot_bytes;
            let mut opts = OpenOptions::new();
            opts.read(true).write(true).create(true).truncate(true);
            configure_open_options(&mut opts);
            let vf = opts.open(&self.vec_path)?;
            vf.set_len(total_size as u64)?;

            // SAFETY: File exclusively owned, just created
            let mut vm = unsafe { MmapMut::map_mut(&vf)? };

            // Write all slot vectors
            for slot in 0..slot_count {
                let offset = slot * slot_bytes;
                if records.deleted_bitmap().contains(slot as u32) {
                    // Zero-fill deleted slots
                    vm[offset..offset + slot_bytes].fill(0);
                } else if let Some(vec_data) = records.get_vector(slot as u32) {
                    if vec_data.len() == dim {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                vec_data.as_ptr() as *const u8,
                                vec_data.len() * 4,
                            )
                        };
                        vm[offset..offset + slot_bytes].copy_from_slice(bytes);
                    } else {
                        vm[offset..offset + slot_bytes].fill(0);
                    }
                } else {
                    vm[offset..offset + slot_bytes].fill(0);
                }
            }

            vm.flush()?;
            vf.sync_all()?;

            self.vec_mmap = Some(vm);
            self.vec_file = Some(vf);
        }

        // Write .omen with non-vector segments (HNSW, MultiVec) via atomic temp+rename
        self.write_omen_manifest(records, options)?;

        Ok(())
    }

    /// Vector-only checkpoint: write dirty slots to .vecs, write a slim records snapshot,
    /// and truncate the WAL.
    ///
    /// Used by auto-checkpoint to persist vector data cheaply. Skips the expensive manifest
    /// rewrite (~40MB at 1M vectors). The slim records snapshot captures id_to_slot, deleted,
    /// metadata, and accumulated dirty_since_flush so the WAL can be truncated safely.
    /// On crash, recovery loads the slim snapshot instead of replaying WAL entries.
    ///
    /// Requires .vecs to already exist (call `checkpoint_full()` first).
    pub fn checkpoint_vectors_only(
        &mut self,
        records: &RecordStore,
        dirty: &RoaringBitmap,
    ) -> io::Result<()> {
        let dim = self.header.dimensions as usize;
        let slot_count = records.slot_count() as usize;
        let slot_bytes = dim * 4;

        if dim == 0 || slot_count == 0 || dirty.is_empty() {
            return Ok(());
        }

        let required_size = (slot_count * slot_bytes) as u64;

        // Grow .vecs if needed
        let vf = self
            .vec_file
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, ".vecs file not open"))?;
        let current_size = vf.metadata()?.len();

        if required_size > current_size {
            self.vec_mmap = None;
            let vf = self.vec_file.as_ref().expect(".vecs file is open");
            vf.set_len(required_size)?;
            self.vec_mmap = Some(unsafe { MmapMut::map_mut(vf)? });
        }

        // Write dirty slots
        let vm = self
            .vec_mmap
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, ".vecs mmap not available"))?;

        let dirty_count = dirty.len() as usize;
        let total_slots = slot_count;

        for slot in dirty {
            let offset = slot as usize * slot_bytes;
            let end = offset + slot_bytes;
            if end > vm.len() {
                continue;
            }

            if records.deleted_bitmap().contains(slot) {
                vm[offset..end].fill(0);
            } else if let Some(vec_data) = records.get_vector(slot) {
                if vec_data.len() == dim {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            vec_data.as_ptr() as *const u8,
                            vec_data.len() * 4,
                        )
                    };
                    vm[offset..end].copy_from_slice(bytes);
                } else {
                    vm[offset..end].fill(0);
                }
            } else {
                vm[offset..end].fill(0);
            }
        }

        // Flush: batch flush if >25% dirty, otherwise range flush
        if dirty_count * 4 > total_slots {
            vm.flush()?;
        } else {
            for slot in dirty {
                let offset = slot as usize * slot_bytes;
                let end = offset + slot_bytes;
                if end <= vm.len() {
                    vm.flush_range(offset, slot_bytes)?;
                }
            }
        }

        // fsync .vecs
        self.vec_file
            .as_ref()
            .expect(".vecs file is open")
            .sync_all()?;

        // Accumulate dirty slots across auto-checkpoints in memory. The .records snapshot
        // must contain the union of all dirty slots since the last full flush so crash
        // recovery can rebuild partial segments correctly without reading the previous
        // snapshot from disk on every auto-checkpoint.
        self.accumulated_dirty |= dirty;

        // Record the pre-truncation epoch. After truncate() the epoch increments to N+1.
        // Recovery sees WAL epoch N+1 > snapshot epoch N → current WAL entries are new
        // (written after this checkpoint) and must be replayed.
        // If truncation fails (epoch stays N), WAL epoch == snapshot epoch → skip replay
        // (entries are already in the snapshot).
        let pre_truncation_epoch = self.wal.truncation_epoch();
        self.write_records_snapshot(records, &self.accumulated_dirty, pre_truncation_epoch)?;
        self.wal.truncate()?;

        Ok(())
    }

    /// Incremental checkpoint: write only dirty slots to .vecs mmap.
    ///
    /// Much faster than full checkpoint for large databases with small updates.
    pub fn checkpoint_incremental(
        &mut self,
        records: &RecordStore,
        dirty: &RoaringBitmap,
        options: CheckpointOptions<'_>,
    ) -> io::Result<()> {
        let dim = self.header.dimensions as usize;
        let slot_count = records.slot_count() as usize;
        let slot_bytes = dim * 4;

        if dim == 0 || slot_count == 0 {
            // Truncate .vecs when all records are gone so stale vectors
            // aren't reconstructed as phantom records on reopen.
            if slot_count == 0 {
                self.vec_mmap = None;
                if let Some(ref vf) = self.vec_file {
                    vf.set_len(0)?;
                }
            }
            self.write_omen_manifest(records, options)?;
            return Ok(());
        }

        let required_size = (slot_count * slot_bytes) as u64;

        // Grow .vecs if needed
        let vf = self
            .vec_file
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, ".vecs file not open"))?;
        let current_size = vf.metadata()?.len();

        if required_size > current_size {
            // Drop mmap before resize
            self.vec_mmap = None;
            let vf = self.vec_file.as_ref().expect(".vecs file is open");
            vf.set_len(required_size)?;
            // Re-mmap
            // SAFETY: File exclusively owned via .omen lock
            self.vec_mmap = Some(unsafe { MmapMut::map_mut(vf)? });
        }

        // Write dirty slots
        let vm = self
            .vec_mmap
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, ".vecs mmap not available"))?;

        let dirty_count = dirty.len() as usize;
        let total_slots = slot_count;

        for slot in dirty {
            let offset = slot as usize * slot_bytes;
            let end = offset + slot_bytes;
            if end > vm.len() {
                continue;
            }

            if records.deleted_bitmap().contains(slot) {
                vm[offset..end].fill(0);
            } else if let Some(vec_data) = records.get_vector(slot) {
                if vec_data.len() == dim {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            vec_data.as_ptr() as *const u8,
                            vec_data.len() * 4,
                        )
                    };
                    vm[offset..end].copy_from_slice(bytes);
                } else {
                    vm[offset..end].fill(0);
                }
            } else {
                vm[offset..end].fill(0);
            }
        }

        // Flush: batch flush if >25% dirty, otherwise range flush
        if dirty_count * 4 > total_slots {
            vm.flush()?;
        } else {
            for slot in dirty {
                let offset = slot as usize * slot_bytes;
                let end = offset + slot_bytes;
                if end <= vm.len() {
                    vm.flush_range(offset, slot_bytes)?;
                }
            }
        }

        // fsync .vecs
        self.vec_file
            .as_ref()
            .expect(".vecs file is open")
            .sync_all()?;

        // Write .omen manifest (atomic temp+rename)
        self.write_omen_manifest(records, options)?;

        Ok(())
    }

    /// Write .omen manifest file with non-vector segments.
    ///
    /// Shared by checkpoint_full and checkpoint_incremental. Uses atomic
    /// temp+rename for crash safety.
    fn write_omen_manifest(
        &mut self,
        records: &RecordStore,
        options: CheckpointOptions<'_>,
    ) -> io::Result<()> {
        // Drop .omen mmap before writing
        self.mmap = None;

        let temp_path = {
            let mut p = self.path.as_os_str().to_os_string();
            p.push(".tmp");
            PathBuf::from(p)
        };

        let mut temp_file = {
            let mut opts = OpenOptions::new();
            opts.read(true).write(true).create(true).truncate(true);
            configure_open_options(&mut opts);
            opts.open(&temp_path)?
        };

        // Write header
        temp_file.write_all(&self.header.to_bytes())?;

        // Write non-vector segments (HNSW, MultiVec)
        let current_offset = align_to_page(HEADER_SIZE) as u64;
        let mut writer = SegmentWriter::new(&mut temp_file, current_offset);

        let mut new_nodes: Vec<NodeLocation> = Vec::new();

        if let Some(hnsw_data) = options.hnsw_bytes {
            let location = writer.write_aligned(hnsw_data, SegmentType::IndexMetadata)?;
            new_nodes.push(location);
        }

        if let Some(multivec_data) = options.multivec_bytes {
            let location = writer.write_aligned(multivec_data, SegmentType::MultiVectors)?;
            new_nodes.push(location);
        }

        // Build manifest (no vector NodeLocations — vectors are in .vecs)
        let mut manifest = OmenManifest::new();
        manifest.nodes = new_nodes;

        // Zero-copy borrows — no HashMap/Vec clone; String clones are unavoidable (manifest owns)
        let id_to_slot = records.id_to_slot_ref();
        manifest.max_node_id = id_to_slot.values().copied().max().unwrap_or(0);
        manifest.id_to_index = id_to_slot.iter().map(|(k, v)| (k.clone(), *v)).collect();
        manifest.index_to_id = id_to_slot
            .iter()
            .map(|(id, slot)| (*slot, id.clone()))
            .collect();

        // Convert metadata to bytes (skip deleted slots)
        let deleted_bitmap = records.deleted_bitmap();
        let mut metadata_bytes: HashMap<u32, Vec<u8>> = HashMap::new();
        for (idx, json) in records.iter_metadata() {
            if !deleted_bitmap.contains(idx)
                && let Ok(bytes) = serde_json::to_vec(&json)
            {
                metadata_bytes.insert(idx, bytes);
            }
        }
        manifest.metadata = metadata_bytes;
        manifest.deleted.clone_from(&deleted_bitmap);
        manifest.metadata_index = options.metadata_index_bytes.map(<[u8]>::to_vec);
        manifest.multivec_offsets = options.multivec_offsets.map(<[u8]>::to_vec);
        manifest.sparse_index_bytes = options.sparse_index_bytes.map(<[u8]>::to_vec);
        manifest.edge_store_bytes = options.edge_store_bytes.map(<[u8]>::to_vec);

        let live_count = records.len() as usize;
        for (key, val) in [
            ("count", live_count as u64),
            ("dimensions", u64::from(self.header.dimensions)),
            ("hnsw_m", u64::from(self.header.hnsw_m)),
            (
                "hnsw_ef_construction",
                u64::from(self.header.hnsw_ef_construction),
            ),
            ("hnsw_ef_search", u64::from(self.header.hnsw_ef_search)),
            ("metric", self.header.metric as u64),
        ] {
            manifest.config.insert(key.to_string(), val);
        }
        set_manifest_wal_replay_cutoff(&mut manifest.config, &self.wal);
        // Only mark vec_file if .vecs was actually created
        if self.vec_mmap.is_some() {
            manifest.config.insert("vec_file".to_string(), 1);
        }

        // Store MUVERA config in manifest.config
        if let Some(cfg) = options.multivec_config {
            manifest
                .config
                .insert("muvera_repetitions".to_string(), cfg.repetitions as u64);
            manifest.config.insert(
                "muvera_partition_bits".to_string(),
                cfg.partition_bits as u64,
            );
            manifest.config.insert("muvera_seed".to_string(), cfg.seed);
            manifest
                .config
                .insert("muvera_token_dim".to_string(), cfg.token_dim as u64);
            if let Some(d) = cfg.d_proj {
                manifest
                    .config
                    .insert("muvera_d_proj".to_string(), d as u64);
            }
            if let Some(pf) = cfg.pool_factor {
                manifest
                    .config
                    .insert("muvera_pool_factor".to_string(), pf as u64);
            }
            if let Some(mt) = cfg.max_tokens {
                manifest
                    .config
                    .insert("muvera_max_tokens".to_string(), mt as u64);
            }
        }

        // Write Manifest with CRC header
        let manifest_bytes = postcard::to_allocvec(&manifest)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let manifest_header = ManifestHeader::new(&manifest_bytes);

        writer.current_offset = align_to_page(writer.current_offset as usize) as u64;
        let manifest_offset = writer.current_offset;
        writer.file.seek(SeekFrom::Start(writer.current_offset))?;
        writer.file.write_all(&manifest_header.to_bytes())?;
        writer.file.write_all(&manifest_bytes)?;

        // Write Footer
        let total_len = writer.file.stream_position()?;
        let footer = OmenFooter::new(manifest_offset, total_len);
        writer.file.write_all(&footer.to_bytes())?;

        // Truncate to exact size and sync
        let final_len = writer.file.stream_position()?;
        writer.file.set_len(final_len)?;
        writer.file.sync_all()?;
        drop(temp_file);

        // Close old file handle (releases advisory lock)
        self.file = None;

        // Atomic rename: temp -> original
        std::fs::rename(&temp_path, &self.path)?;

        // Fsync parent directory
        if let Some(parent) = self.path.parent()
            && let Ok(dir) = std::fs::File::open(parent)
        {
            let _ = dir.sync_all();
        }

        // Re-open with lock + mmap
        let mut opts = OpenOptions::new();
        opts.read(true).write(true);
        configure_open_options(&mut opts);
        let file = opts.open(&self.path)?;
        lock_exclusive(&file)?;
        // SAFETY: File exclusively locked via flock, valid file descriptor
        self.mmap = Some(unsafe { MmapMut::map_mut(&file)? });
        self.file = Some(file);

        // Update in-memory state
        self.manifest = manifest;
        self.header.count = live_count as u64;

        // Truncate WAL after the new manifest is durable. Recovery filters stale same-epoch
        // WAL entries using the cutoff recorded in the published manifest, so there is no
        // need for a pre-rename checkpoint marker.
        self.wal.truncate()?;

        // Full flush complete — reset accumulated dirty set for the next cycle.
        self.accumulated_dirty = RoaringBitmap::new();

        Ok(())
    }

    /// Checkpoint from external snapshot (legacy full-rewrite path)
    ///
    /// Writes a complete .omen file to a temp path, then atomically renames it
    /// over the original. This ensures a crash at any point leaves either the
    /// old file or the new file intact -- never a half-written file.
    pub fn checkpoint_from_snapshot(
        &mut self,
        vectors: &[Option<&[f32]>],
        id_to_slot: &HashMap<String, u32>,
        deleted: &[u32],
        metadata: &HashMap<u32, serde_json::Value>,
        options: CheckpointOptions<'_>,
    ) -> io::Result<()> {
        // Drop mmap before writing (releases the mapping on the old file)
        self.mmap = None;

        let dim = self.header.dimensions as usize;
        let vec_size = dim
            .checked_mul(4)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Vector size overflow: dim={dim}"),
                )
            })?;
        if vectors.len() > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Vector count {} exceeds u32 maximum", vectors.len()),
            ));
        }
        // Build deleted bitmap once — reused for vector loop, metadata filter, and manifest.deleted
        let deleted_bitmap: RoaringBitmap = deleted.iter().copied().collect();

        // Write complete checkpoint to temp file
        let temp_path = {
            let mut p = self.path.as_os_str().to_os_string();
            p.push(".tmp");
            PathBuf::from(p)
        };

        let mut temp_file = {
            let mut opts = OpenOptions::new();
            opts.read(true).write(true).create(true).truncate(true);
            configure_open_options(&mut opts);
            opts.open(&temp_path)?
        };

        // Write header
        temp_file.write_all(&self.header.to_bytes())?;

        // Write vectors to temp file (packed contiguously, skip deleted)
        let vec_block_start = align_to_page(HEADER_SIZE) as u64;
        let mut buf_writer = BufWriter::with_capacity(1 << 20, &mut temp_file);
        buf_writer.seek(SeekFrom::Start(vec_block_start))?;
        let mut current_offset = vec_block_start;
        let mut new_nodes: Vec<NodeLocation> = Vec::with_capacity(vectors.len());

        for (idx, vec_opt) in vectors.iter().enumerate() {
            if deleted_bitmap.contains(idx as u32) {
                new_nodes.push(NodeLocation {
                    offset: 0,
                    length: 0,
                    segment_type: SegmentType::Vectors,
                });
                continue;
            }
            if let Some(data) = vec_opt {
                if data.len() != dim {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Vector {} has {} dims, expected {dim}", idx, data.len()),
                    ));
                }
                // SAFETY: &[f32] → &[u8] is sound: f32 alignment ≥ u8,
                // all targets are little-endian (matching f32::from_le_bytes on read).
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                buf_writer.write_all(bytes)?;
                new_nodes.push(NodeLocation {
                    offset: current_offset,
                    length: vec_size,
                    segment_type: SegmentType::Vectors,
                });
                current_offset += vec_size as u64;
            } else {
                new_nodes.push(NodeLocation {
                    offset: 0,
                    length: 0,
                    segment_type: SegmentType::Vectors,
                });
            }
        }
        buf_writer.flush()?;
        drop(buf_writer);

        // Write HNSW index if provided
        let mut writer = SegmentWriter::new(&mut temp_file, current_offset);
        if let Some(hnsw_data) = options.hnsw_bytes {
            let location = writer.write_aligned(hnsw_data, SegmentType::IndexMetadata)?;
            new_nodes.push(location);
        }

        // Write MultiVectors segment if provided
        if let Some(multivec_data) = options.multivec_bytes {
            let location = writer.write_aligned(multivec_data, SegmentType::MultiVectors)?;
            new_nodes.push(location);
        }

        // Build new manifest
        let mut manifest = OmenManifest::new();
        manifest.nodes = new_nodes;
        manifest.max_node_id = u32::try_from(vectors.len())
            .unwrap_or(u32::MAX)
            .saturating_sub(1);

        // Build index_to_id from id_to_slot
        let index_to_id: HashMap<u32, String> = id_to_slot
            .iter()
            .map(|(id, &slot)| (slot, id.clone()))
            .collect();

        manifest.id_to_index.clone_from(id_to_slot);
        manifest.index_to_id = index_to_id;

        // Convert metadata to bytes (skip deleted slots)
        let mut metadata_bytes: HashMap<u32, Vec<u8>> = HashMap::new();
        for (&idx, json) in metadata {
            if !deleted_bitmap.contains(idx)
                && let Ok(bytes) = serde_json::to_vec(json)
            {
                metadata_bytes.insert(idx, bytes);
            }
        }
        manifest.metadata = metadata_bytes;
        manifest.deleted = deleted_bitmap; // move — reuses the bitmap built above
        manifest.metadata_index = options.metadata_index_bytes.map(<[u8]>::to_vec);

        // Store multi-vector offsets in manifest (small, atomic with manifest)
        manifest.multivec_offsets = options.multivec_offsets.map(<[u8]>::to_vec);

        // Store sparse index in manifest
        manifest.sparse_index_bytes = options.sparse_index_bytes.map(<[u8]>::to_vec);

        // Store edge store in manifest
        manifest.edge_store_bytes = options.edge_store_bytes.map(<[u8]>::to_vec);

        let live_count = vectors.len().saturating_sub(deleted.len());
        for (key, val) in [
            ("count", live_count as u64),
            ("dimensions", u64::from(self.header.dimensions)),
            ("hnsw_m", u64::from(self.header.hnsw_m)),
            (
                "hnsw_ef_construction",
                u64::from(self.header.hnsw_ef_construction),
            ),
            ("hnsw_ef_search", u64::from(self.header.hnsw_ef_search)),
            ("metric", self.header.metric as u64),
        ] {
            manifest.config.insert(key.to_string(), val);
        }
        set_manifest_wal_replay_cutoff(&mut manifest.config, &self.wal);

        // Store MUVERA config in manifest.config
        if let Some(cfg) = options.multivec_config {
            manifest
                .config
                .insert("muvera_repetitions".to_string(), cfg.repetitions as u64);
            manifest.config.insert(
                "muvera_partition_bits".to_string(),
                cfg.partition_bits as u64,
            );
            manifest.config.insert("muvera_seed".to_string(), cfg.seed);
            manifest
                .config
                .insert("muvera_token_dim".to_string(), cfg.token_dim as u64);
            if let Some(d) = cfg.d_proj {
                manifest
                    .config
                    .insert("muvera_d_proj".to_string(), d as u64);
            }
            if let Some(pf) = cfg.pool_factor {
                manifest
                    .config
                    .insert("muvera_pool_factor".to_string(), pf as u64);
            }
            if let Some(mt) = cfg.max_tokens {
                manifest
                    .config
                    .insert("muvera_max_tokens".to_string(), mt as u64);
            }
        }

        // Write Manifest with CRC header
        let manifest_bytes = postcard::to_allocvec(&manifest)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let manifest_header = ManifestHeader::new(&manifest_bytes);

        // Write header + data at page-aligned offset
        writer.current_offset = align_to_page(writer.current_offset as usize) as u64;
        let manifest_offset = writer.current_offset;
        writer.file.seek(SeekFrom::Start(writer.current_offset))?;
        writer.file.write_all(&manifest_header.to_bytes())?;
        writer.file.write_all(&manifest_bytes)?;

        // Write Footer
        let total_len = writer.file.stream_position()?;
        let footer = OmenFooter::new(manifest_offset, total_len);
        writer.file.write_all(&footer.to_bytes())?;

        // Truncate to exact size and sync
        let final_len = writer.file.stream_position()?;
        writer.file.set_len(final_len)?;
        writer.file.sync_all()?;
        drop(temp_file);

        // Close old file handle (releases advisory lock)
        self.file = None;

        // Atomic rename: temp -> original
        // On Unix: atomic on same filesystem. On Windows (NTFS): atomic for same-dir rename.
        std::fs::rename(&temp_path, &self.path)?;

        // Fsync parent directory to ensure rename is durable (required on Linux ext4)
        if let Some(parent) = self.path.parent()
            && let Ok(dir) = std::fs::File::open(parent)
        {
            let _ = dir.sync_all();
        }

        // Re-open with lock + mmap
        let mut opts = OpenOptions::new();
        opts.read(true).write(true);
        configure_open_options(&mut opts);
        let file = opts.open(&self.path)?;
        lock_exclusive(&file)?;
        // SAFETY: File exclusively locked via flock, valid file descriptor
        self.mmap = Some(unsafe { MmapMut::map_mut(&file)? });
        self.file = Some(file);

        // Update in-memory state
        self.manifest = manifest;
        self.header.count = live_count as u64;

        // Truncate WAL after the new manifest is durable. Recovery filters stale same-epoch
        // WAL entries using the cutoff recorded in the published manifest, so there is no
        // need for a pre-rename checkpoint marker.
        self.wal.truncate()?;

        // Full flush complete — reset accumulated dirty set for the next cycle.
        self.accumulated_dirty = RoaringBitmap::new();

        Ok(())
    }

    /// Get WAL length (for determining if checkpoint needed)
    #[must_use]
    pub fn wal_len(&self) -> u64 {
        self.wal.len()
    }

    /// Return the WAL truncation epoch for slim snapshot recovery.
    pub fn wal_truncation_epoch(&self) -> u64 {
        self.wal.truncation_epoch()
    }

    /// Return the replay cutoff recorded in the published manifest for full checkpoints.
    #[must_use]
    pub fn manifest_wal_replay_cutoff(&self) -> Option<(u64, Option<u64>)> {
        let epoch = self.config.get(MANIFEST_WAL_EPOCH_KEY).copied()?;
        let max_ts = self.config.get(MANIFEST_WAL_MAX_TS_KEY).copied();
        Some((epoch, max_ts))
    }
}

// WAL entry parsing moved to wal.rs - re-exported above

fn read_vector_from_bytes(bytes: &[u8], dimensions: usize) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .take(dimensions)
        .map(|chunk| {
            f32::from_le_bytes(
                chunk
                    .try_into()
                    .expect("chunks_exact(4) guarantees 4 bytes"),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_create_and_open() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.omen");

        // Create empty file
        {
            let db = OmenFile::create(&db_path, 3).unwrap();
            assert_eq!(db.len(), 0);
            assert_eq!(db.dimensions(), 3);
        }

        // Reopen
        {
            let db = OmenFile::open(&db_path).unwrap();
            assert_eq!(db.len(), 0);
            assert_eq!(db.dimensions(), 3);
        }
    }

    #[test]
    fn test_wal_append_insert() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_wal_append.omen");

        let mut db = OmenFile::create(&db_path, 3).unwrap();

        // WAL only, no state mutation
        db.wal_append_insert("vec1", &[1.0, 2.0, 3.0], None)
            .unwrap();
        db.wal_append_insert("vec2", &[4.0, 5.0, 6.0], Some(br#"{"key":"value"}"#))
            .unwrap();

        // WAL should have entries
        assert!(db.wal_len() > 0);
        // Header count not updated by WAL-only writes
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_wal_recovery() {
        // Phase 5: WAL replay happens at VectorStore level
        // This test verifies pending_wal_entries() returns correct data
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.omen");

        {
            let mut db = OmenFile::create(&db_path, 3).unwrap();
            db.wal_append_insert("vec1", &[1.0, 2.0, 3.0], None)
                .unwrap();
            // Don't checkpoint - data is only in WAL
        }

        {
            let mut db = OmenFile::open(&db_path).unwrap();
            // WAL replay now happens externally - verify entries are available
            let entries = db.pending_wal_entries().unwrap();
            assert_eq!(entries.len(), 1);

            // Parse and verify the entry
            let insert_data = parse_wal_insert(&entries[0].data).unwrap();
            assert_eq!(insert_data.id, "vec1");
            assert_eq!(insert_data.vector, vec![1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn test_wal_delete_recovery() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.omen");

        {
            let mut db = OmenFile::create(&db_path, 3).unwrap();
            db.wal_append_insert("vec1", &[1.0, 2.0, 3.0], None)
                .unwrap();
            db.wal_append_delete("vec1").unwrap();
        }

        {
            let mut db = OmenFile::open(&db_path).unwrap();
            let entries = db.pending_wal_entries().unwrap();
            assert_eq!(entries.len(), 2);

            // First entry is insert
            let insert_data = parse_wal_insert(&entries[0].data).unwrap();
            assert_eq!(insert_data.id, "vec1");

            // Second entry is delete
            let delete_data = parse_wal_delete(&entries[1].data).unwrap();
            assert_eq!(delete_data.id, "vec1");
        }
    }

    #[test]
    fn test_checkpoint_from_snapshot() {
        use std::collections::HashMap;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_ext_checkpoint.omen");

        // Create empty DB
        let mut db = OmenFile::create(&db_path, 3).unwrap();

        // Build snapshot data externally (simulating RecordStore)
        let v1 = vec![1.0f32, 2.0, 3.0];
        let v2 = vec![4.0f32, 5.0, 6.0];
        let v3 = vec![7.0f32, 8.0, 9.0];
        let vectors: Vec<Option<&[f32]>> = vec![Some(&v1), Some(&v2), Some(&v3)];
        let mut id_to_slot: HashMap<String, u32> = HashMap::new();
        id_to_slot.insert("vec1".to_string(), 0);
        id_to_slot.insert("vec2".to_string(), 1);
        // vec3 at slot 2 is deleted

        let deleted: Vec<u32> = vec![2]; // Slot 2 is deleted

        let mut metadata: HashMap<u32, serde_json::Value> = HashMap::new();
        metadata.insert(0, serde_json::json!({"key": "value1"}));
        metadata.insert(1, serde_json::json!({"key": "value2"}));

        // Checkpoint from external snapshot
        db.checkpoint_from_snapshot(
            &vectors,
            &id_to_slot,
            &deleted,
            &metadata,
            CheckpointOptions::default(),
        )
        .unwrap();

        // Verify header count updated
        assert_eq!(db.len(), 2);

        drop(db);

        // Reopen and verify count persisted
        let db2 = OmenFile::open(&db_path).unwrap();
        assert_eq!(db2.len(), 2);
    }

    #[test]
    fn test_load_persisted_snapshot() {
        use std::collections::HashMap;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_snapshot.omen");

        // Create and checkpoint with data
        {
            let mut db = OmenFile::create(&db_path, 3).unwrap();

            let v1 = vec![1.0f32, 2.0, 3.0];
            let v2 = vec![4.0f32, 5.0, 6.0];
            let vectors: Vec<Option<&[f32]>> = vec![Some(v1.as_slice()), Some(v2.as_slice()), None];
            let mut id_to_slot: HashMap<String, u32> = HashMap::new();
            id_to_slot.insert("vec1".to_string(), 0);
            id_to_slot.insert("vec2".to_string(), 1);

            let deleted: Vec<u32> = vec![2];

            let mut metadata: HashMap<u32, serde_json::Value> = HashMap::new();
            metadata.insert(0, serde_json::json!({"k":"v1"}));
            metadata.insert(1, serde_json::json!({"k":"v2"}));

            db.checkpoint_from_snapshot(
                &vectors,
                &id_to_slot,
                &deleted,
                &metadata,
                CheckpointOptions::default(),
            )
            .unwrap();
        }

        // Reopen and load snapshot
        {
            let db = OmenFile::open(&db_path).unwrap();
            let snapshot = db.load_persisted_snapshot();

            assert_eq!(snapshot.dimensions, 3);
            assert_eq!(snapshot.id_to_slot.len(), 2);
            assert!(snapshot.id_to_slot.contains_key("vec1"));
            assert!(snapshot.id_to_slot.contains_key("vec2"));

            // Check vectors loaded correctly
            let slot0 = snapshot.id_to_slot["vec1"] as usize;
            let slot1 = snapshot.id_to_slot["vec2"] as usize;

            assert!(snapshot.vectors[slot0].is_some());
            assert!(snapshot.vectors[slot1].is_some());
            assert_eq!(snapshot.vectors[slot0].as_ref().unwrap(), &[1.0, 2.0, 3.0]);
            assert_eq!(snapshot.vectors[slot1].as_ref().unwrap(), &[4.0, 5.0, 6.0]);

            // Check deleted bitmap
            assert!(snapshot.deleted.contains(&2));

            // Check metadata
            assert!(snapshot.metadata.contains_key(&0));
            assert!(snapshot.metadata.contains_key(&1));
        }
    }

    #[test]
    fn test_checkpoint_clears_wal() {
        use std::collections::HashMap;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_wal_clear.omen");

        {
            let mut db = OmenFile::create(&db_path, 3).unwrap();

            // Write to WAL
            db.wal_append_insert("vec1", &[1.0, 2.0, 3.0], None)
                .unwrap();
            assert!(db.wal_len() > 0);

            // Checkpoint should clear WAL completely
            let v1 = vec![1.0f32, 2.0, 3.0];
            let vectors: Vec<Option<&[f32]>> = vec![Some(&v1)];
            let mut id_to_slot: HashMap<String, u32> = HashMap::new();
            id_to_slot.insert("vec1".to_string(), 0);
            let metadata: HashMap<u32, serde_json::Value> = HashMap::new();

            db.checkpoint_from_snapshot(
                &vectors,
                &id_to_slot,
                &[],
                &metadata,
                CheckpointOptions::default(),
            )
            .unwrap();

            // WAL is empty after checkpoint (no post-truncate marker needed —
            // empty WAL replays nothing; the pre-rename checkpoint entry covers
            // the crash window between rename and truncation)
            assert_eq!(db.wal_len(), 0);
        }

        // After reopen, no pending WAL entries (checkpoint marker is not returned)
        {
            let mut db = OmenFile::open(&db_path).unwrap();
            let entries = db.pending_wal_entries().unwrap();
            assert!(entries.is_empty());
        }
    }

    #[test]
    fn test_footer_recovery() {
        use std::collections::HashMap;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_footer.omen");

        {
            let mut db = OmenFile::create(&db_path, 3).unwrap();

            let v1 = vec![1.0f32, 2.0, 3.0];
            let vectors: Vec<Option<&[f32]>> = vec![Some(&v1)];
            let mut id_to_slot: HashMap<String, u32> = HashMap::new();
            id_to_slot.insert("vec1".to_string(), 0);
            let metadata: HashMap<u32, serde_json::Value> = HashMap::new();

            db.checkpoint_from_snapshot(
                &vectors,
                &id_to_slot,
                &[],
                &metadata,
                CheckpointOptions::default(),
            )
            .unwrap();

            // Check that footer is there
            let file = File::open(&db_path).unwrap();
            let len = file.metadata().unwrap().len();
            assert!(len > (HEADER_SIZE + OmenFooter::SIZE) as u64);
        }

        {
            // Open and check if manifest was recovered
            let db = OmenFile::open(&db_path).unwrap();
            assert_eq!(db.len(), 1);
            assert!(!db.manifest.nodes.is_empty());
            assert_eq!(db.manifest.nodes[0].segment_type, SegmentType::Vectors);
        }
    }

    #[test]
    fn test_config_persistence() {
        use std::collections::HashMap;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_config.omen");

        {
            let mut db = OmenFile::create(&db_path, 3).unwrap();

            // Set dimensions (in-memory only)
            db.set_dimensions(128);
            assert_eq!(db.dimensions(), 128);

            // Checkpoint to persist header changes
            let vectors: Vec<Option<&[f32]>> = vec![];
            let id_to_slot: HashMap<String, u32> = HashMap::new();
            let metadata: HashMap<u32, serde_json::Value> = HashMap::new();

            db.checkpoint_from_snapshot(
                &vectors,
                &id_to_slot,
                &[],
                &metadata,
                CheckpointOptions::default(),
            )
            .unwrap();
        }

        {
            let db = OmenFile::open(&db_path).unwrap();
            assert_eq!(db.dimensions(), 128);
        }
    }

    #[test]
    fn test_manifest_crc_validation() {
        use std::collections::HashMap;
        use std::io::Write;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_crc.omen");

        // Create a valid file
        {
            let mut db = OmenFile::create(&db_path, 3).unwrap();

            let v1 = vec![1.0f32, 2.0, 3.0];
            let vectors: Vec<Option<&[f32]>> = vec![Some(&v1)];
            let mut id_to_slot: HashMap<String, u32> = HashMap::new();
            id_to_slot.insert("vec1".to_string(), 0);
            let metadata: HashMap<u32, serde_json::Value> = HashMap::new();

            db.checkpoint_from_snapshot(
                &vectors,
                &id_to_slot,
                &[],
                &metadata,
                CheckpointOptions::default(),
            )
            .unwrap();
        }

        // Verify it opens successfully
        {
            let db = OmenFile::open(&db_path).unwrap();
            assert_eq!(db.len(), 1);
        }

        // Corrupt the manifest data (after header, before footer)
        {
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&db_path)
                .unwrap();
            let len = file.metadata().unwrap().len();

            // Read footer to get manifest offset
            #[allow(clippy::cast_possible_wrap)]
            file.seek(SeekFrom::End(-(OmenFooter::SIZE as i64)))
                .unwrap();
            let mut footer_buf = [0u8; OmenFooter::SIZE];
            file.read_exact(&mut footer_buf).unwrap();
            let footer = OmenFooter::from_bytes(&footer_buf).unwrap();

            // Corrupt one byte of manifest data (after the 8-byte header)
            let corrupt_offset = footer.manifest_offset + 8 + 1; // Skip header, corrupt second byte
            if corrupt_offset < len - OmenFooter::SIZE as u64 {
                file.seek(SeekFrom::Start(corrupt_offset)).unwrap();
                file.write_all(&[0xFF]).unwrap();
                file.sync_all().unwrap();
            }
        }

        // Verify it fails to open with CRC error
        {
            let result = OmenFile::open(&db_path);
            match result {
                Ok(_) => panic!("Expected CRC error, but file opened successfully"),
                Err(e) => {
                    assert!(
                        e.to_string().contains("CRC mismatch"),
                        "Expected CRC error, got: {e}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_checkpoint_skips_deleted_compact_file() {
        use std::collections::HashMap;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_deleted_skip.omen");

        let vecs: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32; 3]).collect();

        // Checkpoint with all 10 vectors live
        {
            let mut db = OmenFile::create(&db_path, 3).unwrap();
            let vectors: Vec<Option<&[f32]>> = vecs.iter().map(|v| Some(v.as_slice())).collect();
            let mut id_to_slot: HashMap<String, u32> = HashMap::new();
            for i in 0..10u32 {
                id_to_slot.insert(format!("vec{i}"), i);
            }
            let metadata: HashMap<u32, serde_json::Value> = HashMap::new();
            db.checkpoint_from_snapshot(
                &vectors,
                &id_to_slot,
                &[],
                &metadata,
                CheckpointOptions::default(),
            )
            .unwrap();
        }
        let file_size_all = std::fs::metadata(&db_path).unwrap().len();

        // Checkpoint with 50% deleted
        {
            let mut db = OmenFile::open(&db_path).unwrap();
            let vectors: Vec<Option<&[f32]>> = vecs.iter().map(|v| Some(v.as_slice())).collect();
            let mut id_to_slot: HashMap<String, u32> = HashMap::new();
            for i in (0..10u32).step_by(2) {
                id_to_slot.insert(format!("vec{i}"), i);
            }
            let deleted: Vec<u32> = (0..10u32).filter(|i| i % 2 != 0).collect();
            let metadata: HashMap<u32, serde_json::Value> = HashMap::new();
            db.checkpoint_from_snapshot(
                &vectors,
                &id_to_slot,
                &deleted,
                &metadata,
                CheckpointOptions::default(),
            )
            .unwrap();
        }
        let file_size_half_deleted = std::fs::metadata(&db_path).unwrap().len();

        // 50% deleted should produce smaller file
        assert!(
            file_size_half_deleted < file_size_all,
            "File with 50% deleted ({file_size_half_deleted}) should be smaller \
             than all live ({file_size_all})"
        );

        // Verify roundtrip: deleted slots are None, live slots correct
        {
            let db = OmenFile::open(&db_path).unwrap();
            assert_eq!(db.len(), 5);
            let snapshot = db.load_persisted_snapshot();

            // Live (even) slots have correct vectors
            for i in (0..10u32).step_by(2) {
                let vec_data = snapshot.vectors.get(i as usize).and_then(|v| v.as_ref());
                assert!(vec_data.is_some(), "Live slot {i} should have a vector");
                assert_eq!(vec_data.unwrap(), &vec![i as f32; 3]);
            }

            // Deleted (odd) slots have no vector data
            for i in (1..10u32).step_by(2) {
                let vec_data = snapshot.vectors.get(i as usize).and_then(|v| v.as_ref());
                assert!(vec_data.is_none(), "Deleted slot {i} should be None");
            }
        }
    }
}
