//! `OmenFile` - main API for .omen format
//!
//! Storage backend for `VectorStore`. Uses postcard for efficient binary serialization.

use crate::omen::{
    align_to_page,
    header::{Metric, OmenHeader, HEADER_SIZE},
    wal::{Wal, WalEntry},
    ManifestHeader, NodeLocation, OmenFooter, OmenManifest, SegmentType,
};

// Re-export WAL parsing functions for external use
pub use crate::omen::wal::{parse_wal_delete, parse_wal_insert, WalDeleteData, WalInsertData};
use anyhow::Result;
use fs2::FileExt;
use memmap2::MmapMut;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
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

    pub fn create(path: impl AsRef<Path>, dimensions: u32) -> io::Result<Self> {
        let path = path.as_ref();
        let omen_path = Self::compute_omen_path(path);
        let wal_path = Self::compute_wal_path(path);

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
        })
    }

    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let omen_path = Self::compute_omen_path(path);
        let wal_path = Self::compute_wal_path(path);

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
            let f = OmenFooter::from_bytes(&footer_buf);
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
            .unwrap_or(u64::from(header.dimensions)) as u32;
        let hnsw_m = manifest
            .config
            .get("hnsw_m")
            .copied()
            .unwrap_or(u64::from(header.hnsw_m)) as u32;
        let hnsw_ef_construction = manifest
            .config
            .get("hnsw_ef_construction")
            .copied()
            .unwrap_or(u64::from(header.hnsw_ef_construction))
            as u32;
        let hnsw_ef_search = manifest
            .config
            .get("hnsw_ef_search")
            .copied()
            .unwrap_or(u64::from(header.hnsw_ef_search)) as u32;
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
        })
    }

    // Note: insert(), find_nearest(), search() removed in Phase 5.
    // VectorStore uses wal_append_insert() for inserts and RecordStore for search.

    pub fn delete(&mut self, id: &str) -> io::Result<bool> {
        // Write delete to WAL - existence check is done at VectorStore level
        self.wal.append(WalEntry::delete_node(0, id))?;
        self.wal.sync()?;
        Ok(true)
    }

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

// Note: Many methods removed in Phase 5. VectorStore uses RecordStore for state.
// OmenFile is now pure I/O: WAL + checkpoint_from_snapshot.

impl OmenFile {
    /// Store metadata for a vector (no-op, RecordStore is source of truth)
    #[allow(clippy::unused_self)]
    pub fn put_metadata(&mut self, _id: usize, _metadata: &JsonValue) -> Result<()> {
        Ok(())
    }

    /// Store configuration value
    pub fn put_config(&mut self, key: &str, value: u64) -> Result<()> {
        self.config.insert(key.to_string(), value);
        // Sync to header
        match key {
            "dimensions" => self.header.dimensions = value as u32,
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
    pub fn load_persisted_snapshot(&self) -> io::Result<OmenSnapshot> {
        let dim = self.header.dimensions as usize;
        let mut snapshot = OmenSnapshot {
            dimensions: self.header.dimensions,
            ..Default::default()
        };

        // Load vectors from mmap using manifest locations
        if let Some(ref mmap) = self.mmap {
            for (idx, location) in self.manifest.nodes.iter().enumerate() {
                if location.segment_type == SegmentType::Vectors {
                    while snapshot.vectors.len() <= idx {
                        snapshot.vectors.push(None);
                    }
                    let start = location.offset as usize;
                    let end = start + location.length as usize;
                    if end <= mmap.len() {
                        let vec = read_vector_from_bytes(&mmap[start..end], dim);
                        // Infer dimensions from first vector if header says 0
                        if snapshot.dimensions == 0 && !vec.is_empty() {
                            snapshot.dimensions = vec.len() as u32;
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

        if let (Some(reps), Some(bits), Some(seed), Some(token_dim)) = (reps, bits, seed, token_dim)
        {
            snapshot.multivec_config = Some(PersistedMuveraConfig {
                repetitions: reps as u8,
                partition_bits: bits as u8,
                seed,
                token_dim: token_dim as usize,
                d_proj,
                pool_factor,
            });
        }

        Ok(snapshot)
    }

    // Note: load_snapshot() removed in Phase 5. VectorStore uses load_persisted_snapshot().

    /// Checkpoint from external snapshot
    ///
    /// Writes a complete .omen file to a temp path, then atomically renames it
    /// over the original. This ensures a crash at any point leaves either the
    /// old file or the new file intact -- never a half-written file.
    pub fn checkpoint_from_snapshot(
        &mut self,
        vectors: &[Option<Vec<f32>>],
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
        let deleted_set: std::collections::HashSet<u32> = deleted.iter().copied().collect();

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

        // Write ALL vectors to temp file
        let mut writer = SegmentWriter::new(&mut temp_file, HEADER_SIZE as u64);
        let mut new_nodes: Vec<NodeLocation> = Vec::new();

        for (idx, vec_opt) in vectors.iter().enumerate() {
            let to_write = if deleted_set.contains(&(idx as u32)) {
                vec![0.0f32; dim]
            } else {
                vec_opt.clone().unwrap_or_else(|| vec![0.0f32; dim])
            };

            writer.current_offset = align_to_page(writer.current_offset as usize) as u64;
            writer.file.seek(SeekFrom::Start(writer.current_offset))?;
            for &val in &to_write {
                writer.file.write_all(&val.to_le_bytes())?;
            }

            new_nodes.push(NodeLocation {
                offset: writer.current_offset,
                length: vec_size,
                segment_type: SegmentType::Vectors,
            });

            writer.current_offset += vec_size as u64;
        }

        // Write HNSW index if provided
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
        manifest.max_node_id = (vectors.len() as u32).saturating_sub(1);

        // Build index_to_id from id_to_slot
        let index_to_id: HashMap<u32, String> = id_to_slot
            .iter()
            .map(|(id, &slot)| (slot, id.clone()))
            .collect();

        manifest.id_to_index.clone_from(id_to_slot);
        manifest.index_to_id = index_to_id;
        manifest.deleted = deleted.iter().copied().collect();

        // Convert metadata to bytes
        let mut metadata_bytes: HashMap<u32, Vec<u8>> = HashMap::new();
        for (&idx, json) in metadata {
            if !deleted_set.contains(&idx) {
                if let Ok(bytes) = serde_json::to_vec(json) {
                    metadata_bytes.insert(idx, bytes);
                }
            }
        }
        manifest.metadata = metadata_bytes;
        manifest.metadata_index = options.metadata_index_bytes.map(<[u8]>::to_vec);

        // Store multi-vector offsets in manifest (small, atomic with manifest)
        manifest.multivec_offsets = options.multivec_offsets.map(<[u8]>::to_vec);

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
        if let Some(parent) = self.path.parent() {
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
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

        // Truncate WAL (safe: if crash here, WAL replay is idempotent)
        self.wal.truncate()?;
        self.wal.append(WalEntry::checkpoint(0))?;
        self.wal.sync()?;

        Ok(())
    }

    /// Get WAL length (for determining if checkpoint needed)
    #[must_use]
    pub fn wal_len(&self) -> u64 {
        self.wal.len()
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
        let vectors: Vec<Option<Vec<f32>>> = vec![
            Some(vec![1.0, 2.0, 3.0]),
            Some(vec![4.0, 5.0, 6.0]),
            Some(vec![7.0, 8.0, 9.0]),
        ];
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

            let vectors: Vec<Option<Vec<f32>>> = vec![
                Some(vec![1.0, 2.0, 3.0]),
                Some(vec![4.0, 5.0, 6.0]),
                None, // Slot 2 deleted
            ];
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
            let snapshot = db.load_persisted_snapshot().unwrap();

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

            // Checkpoint should clear WAL (leaves 1 checkpoint marker)
            let vectors: Vec<Option<Vec<f32>>> = vec![Some(vec![1.0, 2.0, 3.0])];
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

            // WAL has 1 entry (checkpoint marker)
            assert_eq!(db.wal_len(), 1);
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

            let vectors: Vec<Option<Vec<f32>>> = vec![Some(vec![1.0, 2.0, 3.0])];
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
            let vectors: Vec<Option<Vec<f32>>> = vec![];
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

            let vectors: Vec<Option<Vec<f32>>> = vec![Some(vec![1.0, 2.0, 3.0])];
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
            let footer = OmenFooter::from_bytes(&footer_buf);

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
}
