//! `OmenFile` - main API for .omen format
//!
//! Storage backend for `VectorStore`. Uses postcard for efficient binary serialization.

use crate::omen::{
    align_to_page,
    header::{OmenHeader, HEADER_SIZE},
    wal::{Wal, WalEntry, WalEntryType},
    NodeLocation, OmenFooter, OmenManifest, SegmentType,
};
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

/// In-memory state of the database before persistence
#[derive(Default)]
pub struct DatabaseState {
    pub vectors: Vec<Vec<f32>>,
    pub id_to_index: HashMap<String, u32>,
    pub index_to_id: HashMap<u32, String>,
    pub metadata: HashMap<u32, Vec<u8>>,
    pub deleted: HashMap<u32, bool>,
    pub config: HashMap<String, u64>,
}

impl DatabaseState {
    pub fn new(dimensions: u32) -> Self {
        Self {
            config: HashMap::from([("dimensions".to_string(), u64::from(dimensions))]),
            ..Default::default()
        }
    }
}

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

        let location = NodeLocation {
            offset: self.current_offset,
            length: data.len() as u32,
            segment_type,
        };

        self.current_offset += data.len() as u64;
        Ok(location)
    }
}

/// Checkpoint threshold (number of WAL entries before compaction)
const CHECKPOINT_THRESHOLD: u64 = 1000;

/// `OmenFile` - single-file vector database
///
/// Storage layer for vectors, metadata, and serialized HNSW index.
/// Graph traversal is handled by `HNSWIndex` in the vector layer.
pub struct OmenFile {
    path: PathBuf,
    file: Option<File>,
    mmap: Option<MmapMut>,
    header: OmenHeader,

    // In-memory state (for writes before checkpoint)
    state: DatabaseState,

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

        // Write initial empty Manifest and Footer
        let manifest = OmenManifest::new();
        let manifest_bytes = postcard::to_allocvec(&manifest).unwrap();
        let manifest_offset = file.stream_position()?;
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
            state: DatabaseState::new(dimensions),
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
            Some(unsafe { MmapMut::map_mut(&file)? })
        } else {
            None
        };

        let wal = Wal::open(&wal_path)?;

        let mut manifest = OmenManifest::new();

        // Load manifest
        if let Some(ref mmap) = mmap {
            let manifest_offset = footer.manifest_offset as usize;
            if manifest_offset < mmap.len() {
                let manifest_bytes = &mmap[manifest_offset..footer.total_len as usize];
                manifest = postcard::from_bytes::<OmenManifest>(manifest_bytes).map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Failed to decode manifest: {e}"),
                    )
                })?;
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Manifest offset out of bounds",
                ));
            }
        }

        let mut vectors_mem = Vec::new();
        let id_to_index = manifest.id_to_index.clone();
        let index_to_id = manifest.index_to_id.clone();
        let metadata_mem = manifest.metadata.clone();
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
        let metric = manifest
            .config
            .get("metric")
            .map(|&v| crate::omen::header::Metric::from(v as u8))
            .unwrap_or(header.metric);

        if let Some(ref mmap) = mmap {
            // Load vectors from manifest using dimensions from manifest.config
            let dim = dimensions as usize;
            for location in &manifest.nodes {
                if location.segment_type == SegmentType::Vectors {
                    let start = location.offset as usize;
                    let end = start + location.length as usize;
                    if end <= mmap.len() {
                        vectors_mem.push(read_vector_from_bytes(&mmap[start..end], dim));
                    }
                }
            }

            // Load HNSW index bytes from manifest
            let mut hnsw_index_bytes = None;
            for location in &manifest.nodes {
                if location.segment_type == SegmentType::IndexMetadata {
                    let start = location.offset as usize;
                    let end = start + location.length as usize;
                    if end <= mmap.len() {
                        hnsw_index_bytes = Some(mmap[start..end].to_vec());
                        break;
                    }
                }
            }

            // Restore deleted bitmap from manifest (LanceDB pattern)
            let deleted: HashMap<u32, bool> =
                manifest.deleted.iter().map(|idx| (idx, true)).collect();

            let state = DatabaseState {
                vectors: vectors_mem,
                id_to_index,
                index_to_id,
                metadata: metadata_mem,
                config,
                deleted,
            };

            // Update header from manifest (source of truth for append-only)
            let mut header = header;
            header.count = count;
            header.dimensions = dimensions;
            header.hnsw_m = hnsw_m;
            header.hnsw_ef_construction = hnsw_ef_construction;
            header.hnsw_ef_search = hnsw_ef_search;
            header.metric = metric;

            let mmap = Some(unsafe { MmapMut::map_mut(&file)? });
            let mut db = Self {
                path: omen_path,
                file: Some(file),
                mmap,
                header,
                state,
                wal,
                hnsw_index_bytes,
                manifest,
            };

            // Replay WAL
            db.recover()?;

            return Ok(db);
        }

        // Restore deleted bitmap from manifest (LanceDB pattern)
        let deleted: HashMap<u32, bool> = manifest.deleted.iter().map(|idx| (idx, true)).collect();

        let state = DatabaseState {
            vectors: vectors_mem,
            id_to_index,
            index_to_id,
            metadata: metadata_mem,
            config,
            deleted,
        };

        // Update header from manifest (source of truth for append-only)
        let mut header = header;
        header.count = count;
        header.dimensions = dimensions;
        header.hnsw_m = hnsw_m;
        header.hnsw_ef_construction = hnsw_ef_construction;
        header.hnsw_ef_search = hnsw_ef_search;
        header.metric = metric;

        let mut db = Self {
            path: omen_path,
            file: Some(file),
            mmap: None,
            header,
            state,
            wal,
            hnsw_index_bytes: None,
            manifest,
        };

        // Replay WAL
        db.recover()?;

        Ok(db)
    }

    /// Recover from WAL
    fn recover(&mut self) -> io::Result<()> {
        let entries = self.wal.entries_after_checkpoint()?;

        for entry in entries {
            if !entry.verify() {
                // Log and skip corrupted entries
                tracing::warn!(
                    entry_type = ?entry.header.entry_type,
                    timestamp = entry.header.timestamp,
                    "Skipping corrupted WAL entry during recovery"
                );
                continue;
            }

            match entry.header.entry_type {
                WalEntryType::InsertNode => {
                    self.replay_insert(&entry.data)?;
                }
                WalEntryType::DeleteNode => {
                    self.replay_delete(&entry.data)?;
                }
                WalEntryType::UpdateNeighbors => {
                    self.replay_neighbors(&entry.data)?;
                }
                WalEntryType::UpdateMetadata | WalEntryType::Checkpoint => {
                    // No-op: metadata updates tracked in cloud-4uv, checkpoint is marker only
                }
            }
        }

        Ok(())
    }

    fn replay_insert(&mut self, data: &[u8]) -> io::Result<()> {
        let mut cursor = std::io::Cursor::new(data);
        let string_id = read_string_id(&mut cursor)?;

        let mut buf = [0u8; 4];

        // Skip level byte (HNSW graph managed by HNSWIndex)
        cursor.read_exact(&mut buf[..1])?;

        // Read vector
        cursor.read_exact(&mut buf)?;
        let vec_len = u32::from_le_bytes(buf) as usize;
        let mut vec_bytes = vec![0u8; vec_len * 4];
        cursor.read_exact(&mut vec_bytes)?;
        let vector = read_vector_from_bytes(&vec_bytes, vec_len);

        // Read metadata
        cursor.read_exact(&mut buf)?;
        let meta_len = u32::from_le_bytes(buf) as usize;
        let mut metadata = vec![0u8; meta_len];
        cursor.read_exact(&mut metadata)?;

        let index = self.state.vectors.len() as u32;
        self.state.vectors.push(vector);
        self.state.id_to_index.insert(string_id.clone(), index);
        self.state.index_to_id.insert(index, string_id);
        if !metadata.is_empty() {
            self.state.metadata.insert(index, metadata);
        }

        Ok(())
    }

    fn replay_delete(&mut self, data: &[u8]) -> io::Result<()> {
        let mut cursor = std::io::Cursor::new(data);
        let string_id = read_string_id(&mut cursor)?;

        if let Some(&index) = self.state.id_to_index.get(&string_id) {
            self.state.deleted.insert(index, true);
        }

        Ok(())
    }

    /// Replay neighbors update from WAL (no-op: graph managed by `HNSWIndex`)
    #[allow(clippy::unused_self, clippy::unnecessary_wraps)]
    fn replay_neighbors(&mut self, _data: &[u8]) -> io::Result<()> {
        // Neighbor updates are consumed from WAL but not stored.
        // HNSWIndex rebuilds graph from vectors on recovery.
        Ok(())
    }

    /// Insert a vector
    ///
    /// Note: Graph management (HNSW) is handled by `HNSWIndex` in the vector layer.
    /// This method only handles storage: WAL, vectors, metadata.
    pub fn insert(&mut self, id: &str, vector: &[f32], metadata: Option<&[u8]>) -> io::Result<()> {
        if vector.len() != self.header.dimensions as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Vector dimensions mismatch: expected {}, got {}",
                    self.header.dimensions,
                    vector.len()
                ),
            ));
        }

        let metadata_bytes = metadata.unwrap_or(b"{}");

        // 1. Append to WAL (durable)
        // Level 0 is placeholder - actual HNSW levels managed by HNSWIndex
        let entry = WalEntry::insert_node(0, id, 0, vector, metadata_bytes);
        self.wal.append(entry)?;
        self.wal.sync()?;

        // 2. Update in-memory state
        let index = self.state.vectors.len() as u32;
        self.state.vectors.push(vector.to_vec());
        self.state.id_to_index.insert(id.to_string(), index);
        self.state.index_to_id.insert(index, id.to_string());
        if metadata_bytes != b"{}" {
            self.state.metadata.insert(index, metadata_bytes.to_vec());
        }

        self.header.count += 1;

        // 3. Periodic checkpoint
        if self.wal.len() > CHECKPOINT_THRESHOLD {
            self.checkpoint()?;
        }

        Ok(())
    }

    fn find_nearest(&self, query: &[f32], k: usize) -> Vec<u32> {
        let mut distances: Vec<(u32, f32)> = self
            .state
            .vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.state.deleted.contains_key(&(*i as u32)))
            .map(|(i, v)| (i as u32, l2_distance(query, v)))
            .collect();

        distances.sort_by(|a, b| a.1.total_cmp(&b.1));
        distances.truncate(k);
        distances.into_iter().map(|(id, _)| id).collect()
    }

    /// Search for k nearest neighbors
    #[must_use]
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        if query.len() != self.header.dimensions as usize {
            return Vec::new();
        }

        let indices = self.find_nearest(query, k);

        indices
            .into_iter()
            .filter_map(|idx| {
                let id = self.state.index_to_id.get(&idx)?;
                let vector = self.state.vectors.get(idx as usize)?;
                let distance = l2_distance(query, vector);
                Some((id.clone(), distance))
            })
            .collect()
    }

    pub fn delete(&mut self, id: &str) -> io::Result<bool> {
        // Write delete to WAL unconditionally - existence check is done at VectorStore level
        // (RecordStore.delete returns None if not found, so we only get here for valid deletes)
        self.wal.append(WalEntry::delete_node(0, id))?;
        self.wal.sync()?;

        // Update legacy state for compatibility (will be removed in Phase 5)
        if let Some(&index) = self.state.id_to_index.get(id) {
            self.state.deleted.insert(index, true);
            self.state.id_to_index.remove(id);
            self.state.index_to_id.remove(&index);
        }

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

    /// Checkpoint - append-only persistence (O(1) for metadata-only changes)
    ///
    /// Append-only design:
    /// 1. Compute append point (end of last data segment)
    /// 2. Append only NEW vectors (not already in manifest)
    /// 3. Append new manifest with updated locations + deleted bitmap
    /// 4. Append footer
    /// 5. Fsync
    /// 6. Truncate WAL
    pub fn checkpoint(&mut self) -> io::Result<()> {
        if self.state.vectors.is_empty() && self.hnsw_index_bytes.is_none() {
            return Ok(());
        }

        // Drop mmap before writing (required for file modification)
        self.mmap = None;

        let file = self
            .file
            .as_mut()
            .ok_or_else(|| io::Error::other("File not open"))?;

        // Find append point: end of last data segment (before old manifest/footer)
        // This is the manifest_offset from the current file layout
        let file_len = file.metadata()?.len();
        let append_offset = if file_len > (HEADER_SIZE + OmenFooter::SIZE) as u64 {
            // Read current footer to find where to append
            file.seek(SeekFrom::End(-(OmenFooter::SIZE as i64)))?;
            let mut footer_buf = [0u8; OmenFooter::SIZE];
            file.read_exact(&mut footer_buf)?;
            let old_footer = OmenFooter::from_bytes(&footer_buf);
            if old_footer.verify() {
                // Append after last data, before old manifest
                // Find end of vector data from manifest.nodes
                let data_end = self
                    .manifest
                    .nodes
                    .iter()
                    .filter(|n| {
                        n.segment_type == SegmentType::Vectors
                            || n.segment_type == SegmentType::IndexMetadata
                    })
                    .map(|n| n.offset + n.length as u64)
                    .max()
                    .unwrap_or(HEADER_SIZE as u64);
                data_end
            } else {
                HEADER_SIZE as u64
            }
        } else {
            HEADER_SIZE as u64
        };

        let mut writer = SegmentWriter::new(file, append_offset);

        // Count of vectors already persisted
        let persisted_count = self
            .manifest
            .nodes
            .iter()
            .filter(|n| n.segment_type == SegmentType::Vectors)
            .count();

        // Clone existing vector locations
        let mut new_nodes: Vec<NodeLocation> = self
            .manifest
            .nodes
            .iter()
            .filter(|n| n.segment_type == SegmentType::Vectors)
            .copied()
            .collect();

        // Append only NEW vectors (those not yet persisted)
        let dim = self.header.dimensions as usize;
        let vec_size = (dim * 4) as u32;

        for (idx, vector) in self.state.vectors.iter().enumerate().skip(persisted_count) {
            // Skip deleted vectors - write zeros
            let to_write = if self.state.deleted.contains_key(&(idx as u32)) {
                vec![0.0f32; dim]
            } else {
                vector.clone()
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

        // Write HNSW index (if present and changed)
        if let Some(ref hnsw_bytes) = self.hnsw_index_bytes {
            let location = writer.write_aligned(hnsw_bytes, SegmentType::IndexMetadata)?;
            new_nodes.push(location);
        }

        // Update config before building manifest (source of truth for append-only)
        let vector_count = self.state.vectors.len() as u64;
        self.state.config.insert("count".to_string(), vector_count);
        self.state
            .config
            .insert("hnsw_m".to_string(), u64::from(self.header.hnsw_m));
        self.state.config.insert(
            "hnsw_ef_construction".to_string(),
            u64::from(self.header.hnsw_ef_construction),
        );
        self.state.config.insert(
            "hnsw_ef_search".to_string(),
            u64::from(self.header.hnsw_ef_search),
        );
        self.state
            .config
            .insert("metric".to_string(), self.header.metric as u64);
        self.header.count = vector_count;

        // Build new manifest
        let mut manifest = OmenManifest::new();
        manifest.nodes = new_nodes;
        manifest.max_node_id = (self.state.vectors.len() as u32).saturating_sub(1);
        manifest.id_to_index = self.state.id_to_index.clone();
        manifest.index_to_id = self.state.index_to_id.clone();
        manifest.deleted = self.state.deleted.keys().copied().collect();

        // Filter deleted from metadata
        let mut metadata = self.state.metadata.clone();
        metadata.retain(|k, _| !self.state.deleted.contains_key(k));
        manifest.metadata = metadata;
        manifest.config = self.state.config.clone();

        // Write Manifest
        let manifest_bytes = postcard::to_allocvec(&manifest)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let manifest_location = writer.write_aligned(&manifest_bytes, SegmentType::Manifest)?;

        // Write Footer at new file end
        let total_len = writer.file.stream_position()?;
        let footer = OmenFooter::new(manifest_location.offset, total_len);
        writer.file.write_all(&footer.to_bytes())?;

        // Truncate file to remove any trailing garbage from old layout
        let final_len = writer.file.stream_position()?;
        writer.file.set_len(final_len)?;

        // Fsync
        writer.file.sync_all()?;

        // Update in-memory manifest
        self.manifest = manifest;

        // Truncate WAL and write checkpoint marker
        self.wal.truncate()?;
        self.wal.append(WalEntry::checkpoint(0))?;
        self.wal.sync()?;

        // Re-establish mmap
        let file = self.file.as_ref().unwrap();
        self.mmap = Some(unsafe { MmapMut::map_mut(file)? });

        // Purge deleted vectors from memory
        for (idx, vector) in self.state.vectors.iter_mut().enumerate() {
            if self.state.deleted.contains_key(&(idx as u32)) {
                vector.clear();
                vector.shrink_to_fit();
            }
        }
        self.state.deleted.clear();

        Ok(())
    }
}

// ============================================================================
// Storage API for VectorStore
// ============================================================================

impl OmenFile {
    /// Store a vector by internal index
    ///
    /// Note: This is a no-op. VectorStore owns vector data via RecordStore.
    /// Persistence happens via checkpoint_from_snapshot which reads from RecordStore.
    #[allow(clippy::unused_self)]
    pub fn put_vector(&mut self, _id: usize, _vector: &[f32]) -> Result<()> {
        // No-op: RecordStore is source of truth, checkpoint reads from it
        Ok(())
    }

    pub fn get_vector(&self, id: usize) -> Result<Option<Vec<f32>>> {
        if id < self.state.vectors.len() && !self.state.vectors[id].is_empty() {
            return Ok(Some(self.state.vectors[id].clone()));
        }

        let Some(ref mmap) = self.mmap else {
            return Ok(None);
        };

        // Use Manifest to locate vector on disk
        let Some(location) = self.manifest.nodes.get(id) else {
            return Ok(None);
        };

        if location.segment_type != SegmentType::Vectors {
            return Ok(None);
        }

        let dim = self.header.dimensions as usize;
        let start = location.offset as usize;
        let end = start + location.length as usize;

        if end <= mmap.len() {
            Ok(Some(read_vector_from_bytes(&mmap[start..end], dim)))
        } else {
            Ok(None)
        }
    }

    /// Store metadata for a vector (as JSON)
    ///
    /// Note: This is a no-op. VectorStore owns metadata via RecordStore.
    /// Persistence happens via checkpoint_from_snapshot which reads from RecordStore.
    #[allow(clippy::unused_self)]
    pub fn put_metadata(&mut self, _id: usize, _metadata: &JsonValue) -> Result<()> {
        // No-op: RecordStore is source of truth, checkpoint reads from it
        Ok(())
    }

    pub fn get_metadata(&self, id: usize) -> Result<Option<JsonValue>> {
        self.state
            .metadata
            .get(&(id as u32))
            .map(|bytes| serde_json::from_slice(bytes))
            .transpose()
            .map_err(Into::into)
    }

    /// Store string ID to internal index mapping
    ///
    /// Note: This is a no-op. VectorStore owns ID mappings via RecordStore.
    /// Persistence happens via checkpoint_from_snapshot which reads from RecordStore.
    #[allow(clippy::unused_self)]
    pub fn put_id_mapping(&mut self, _string_id: &str, _index: usize) -> Result<()> {
        // No-op: RecordStore is source of truth, checkpoint reads from it
        Ok(())
    }

    /// Get internal index for a string ID
    pub fn get_id_mapping(&self, string_id: &str) -> Result<Option<usize>> {
        Ok(self
            .state
            .id_to_index
            .get(string_id)
            .map(|&idx| idx as usize))
    }

    /// Get string ID for an internal index (reverse lookup)
    pub fn get_string_id(&self, index: usize) -> Result<Option<String>> {
        Ok(self.state.index_to_id.get(&(index as u32)).cloned())
    }

    /// Delete string ID mapping
    pub fn delete_id_mapping(&mut self, string_id: &str) -> Result<()> {
        if let Some(&index) = self.state.id_to_index.get(string_id) {
            self.state.index_to_id.remove(&index);
        }
        self.state.id_to_index.remove(string_id);
        Ok(())
    }

    /// Store configuration value
    pub fn put_config(&mut self, key: &str, value: u64) -> Result<()> {
        self.state.config.insert(key.to_string(), value);
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
        Ok(self.state.config.get(key).copied())
    }

    /// Load all vectors from storage
    pub fn load_all_vectors(&self) -> Result<Vec<(usize, Vec<f32>)>> {
        Ok(self
            .state
            .vectors
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .map(|(id, v)| (id, v.clone()))
            .collect())
    }

    /// Increment vector count in storage
    pub fn increment_count(&mut self) -> Result<usize> {
        let count = self.state.config.get("count").copied().unwrap_or(0) as usize;
        let new_count = count + 1;
        self.state
            .config
            .insert("count".to_string(), new_count as u64);
        self.header.count = new_count as u64;
        Ok(new_count)
    }

    /// Get current vector count
    pub fn get_count(&self) -> Result<usize> {
        Ok(self.state.config.get("count").copied().unwrap_or(0) as usize)
    }

    /// Store quantization mode
    ///
    /// Mode values: 0=none, 1=sq8, 2=rabitq-4, 3=rabitq-2, 4=rabitq-8
    pub fn put_quantization_mode(&mut self, mode: u64) -> Result<()> {
        self.put_config("quantization", mode)
    }

    /// Get quantization mode
    ///
    /// Returns: 0=none, 1=sq8, 2=rabitq-4, 3=rabitq-2, 4=rabitq-8
    pub fn get_quantization_mode(&self) -> Result<Option<u64>> {
        self.get_config("quantization")
    }

    /// Check if store was created with quantization
    pub fn is_quantized(&self) -> Result<bool> {
        Ok(self.get_quantization_mode()?.unwrap_or(0) > 0)
    }

    pub fn load_all_metadata(&self) -> Result<HashMap<usize, JsonValue>> {
        Ok(self
            .state
            .metadata
            .iter()
            .filter_map(|(&id, bytes)| {
                serde_json::from_slice(bytes)
                    .ok()
                    .map(|meta| (id as usize, meta))
            })
            .collect())
    }

    /// Load all ID mappings from storage
    pub fn load_all_id_mappings(&self) -> Result<HashMap<String, usize>> {
        Ok(self
            .state
            .id_to_index
            .iter()
            .map(|(id, &idx)| (id.clone(), idx as usize))
            .collect())
    }

    /// Mark a vector as deleted (tombstone)
    pub fn put_deleted(&mut self, id: usize) -> Result<()> {
        self.state.deleted.insert(id as u32, true);
        Ok(())
    }

    pub fn is_deleted(&self, id: usize) -> Result<bool> {
        Ok(self.state.deleted.contains_key(&(id as u32)))
    }

    /// Remove deleted marker (for re-insertion)
    pub fn remove_deleted(&mut self, id: usize) -> Result<()> {
        self.state.deleted.remove(&(id as u32));
        Ok(())
    }

    /// Load all deleted IDs from storage
    pub fn load_all_deleted(&self) -> Result<HashMap<usize, bool>> {
        Ok(self
            .state
            .deleted
            .iter()
            .map(|(&id, &v)| (id as usize, v))
            .collect())
    }

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

    /// Flush all pending writes to disk
    pub fn flush(&mut self) -> Result<()> {
        self.checkpoint()?;
        Ok(())
    }

    /// Returns the ratio of deleted vectors to total vectors (0.0 to 1.0)
    #[must_use]
    pub fn delete_ratio(&self) -> f64 {
        let total = self.state.vectors.len();
        if total == 0 {
            return 0.0;
        }
        self.state.deleted.len() as f64 / total as f64
    }

    /// Check if compaction is recommended (delete ratio > 20%)
    #[must_use]
    pub fn needs_compaction(&self) -> bool {
        self.delete_ratio() > 0.20
    }

    /// Compact the database by rewriting only live vectors
    ///
    /// This reclaims space from deleted vectors by:
    /// 1. Creating a new file with only live data
    /// 2. Rebuilding all mappings with new sequential indices
    /// 3. Atomically swapping the new file for the old one
    ///
    /// Note: The caller (VectorStore) should rebuild the HNSW index after
    /// compaction since internal indices change.
    ///
    /// Returns the number of vectors removed (space reclaimed).
    pub fn compact(&mut self) -> io::Result<usize> {
        let deleted_count = self.state.deleted.len();
        if deleted_count == 0 {
            return Ok(0);
        }

        // 1. Checkpoint current state first (ensure WAL is flushed)
        self.checkpoint()?;

        // 2. Create temporary path for compacted data
        // Use .compact suffix on base path (without .omen) so create() adds .omen correctly
        let base_path = self.path.with_extension(""); // Remove .omen
        let temp_base = base_path.with_extension("compact"); // Add .compact
        let temp_omen_path = Self::compute_omen_path(&temp_base); // -> .compact.omen
        let wal_path = Self::compute_wal_path(&self.path);

        // Close current file
        self.mmap = None;
        self.file = None;

        // Create new compacted file (this will create temp_base.omen)
        let mut new_db = OmenFile::create(&temp_base, self.header.dimensions)?;

        // Copy HNSW params and metric
        new_db.header.hnsw_m = self.header.hnsw_m;
        new_db.header.hnsw_ef_construction = self.header.hnsw_ef_construction;
        new_db.header.hnsw_ef_search = self.header.hnsw_ef_search;
        new_db.header.metric = self.header.metric;

        // Copy config (except count which will be recalculated)
        for (k, v) in &self.state.config {
            if k != "count" {
                new_db.state.config.insert(k.clone(), *v);
            }
        }

        // 3. Copy only live vectors with new sequential indices
        let mut old_to_new: HashMap<u32, u32> = HashMap::new();
        let mut new_index = 0u32;

        for (old_idx, vector) in self.state.vectors.iter().enumerate() {
            let old_idx = old_idx as u32;

            // Skip deleted vectors
            if self.state.deleted.contains_key(&old_idx) {
                continue;
            }

            // Skip empty vectors (already cleared)
            if vector.is_empty() {
                continue;
            }

            // Get string ID for this index
            let Some(string_id) = self.state.index_to_id.get(&old_idx) else {
                continue;
            };

            // Copy vector to new file
            new_db.state.vectors.push(vector.clone());
            new_db
                .state
                .id_to_index
                .insert(string_id.clone(), new_index);
            new_db
                .state
                .index_to_id
                .insert(new_index, string_id.clone());

            // Copy metadata if exists
            if let Some(meta) = self.state.metadata.get(&old_idx) {
                new_db.state.metadata.insert(new_index, meta.clone());
            }

            old_to_new.insert(old_idx, new_index);
            new_index += 1;
        }

        // Update count
        new_db.header.count = new_index as u64;
        new_db
            .state
            .config
            .insert("count".to_string(), new_index as u64);

        // 4. Checkpoint the new file
        new_db.checkpoint()?;

        // Close new file before swap
        drop(new_db);

        // 5. Atomic swap: rename compacted file to original path
        // First remove old file
        std::fs::remove_file(&self.path)?;

        // Move compacted file to original path
        std::fs::rename(&temp_omen_path, &self.path)?;

        // Clean up temp WAL if it exists
        let temp_wal = Self::compute_wal_path(&temp_base);
        let _ = std::fs::remove_file(&temp_wal);

        // 6. Reopen the compacted file
        let mut opts = OpenOptions::new();
        opts.read(true).write(true);
        configure_open_options(&mut opts);
        let mut file = opts.open(&self.path)?;
        lock_exclusive(&file)?;

        // Read footer
        file.seek(SeekFrom::End(-(OmenFooter::SIZE as i64)))?;
        let mut footer_buf = [0u8; OmenFooter::SIZE];
        file.read_exact(&mut footer_buf)?;
        let footer = OmenFooter::from_bytes(&footer_buf);

        // Read manifest
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        let manifest_bytes = &mmap[footer.manifest_offset as usize..footer.total_len as usize];
        let manifest: OmenManifest = postcard::from_bytes(manifest_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Load vectors from manifest
        let dim = self.header.dimensions as usize;
        let mut vectors = Vec::new();
        for location in &manifest.nodes {
            if location.segment_type == SegmentType::Vectors {
                let start = location.offset as usize;
                let end = start + location.length as usize;
                if end <= mmap.len() {
                    vectors.push(read_vector_from_bytes(&mmap[start..end], dim));
                }
            }
        }

        // Update self with compacted state
        self.file = Some(file);
        self.mmap = Some(mmap);
        self.manifest = manifest.clone();
        self.state = DatabaseState {
            vectors,
            id_to_index: manifest.id_to_index,
            index_to_id: manifest.index_to_id,
            metadata: manifest.metadata,
            deleted: HashMap::new(), // No deleted after compaction
            config: manifest.config,
        };
        self.header.count = new_index as u64;
        self.wal = Wal::open(&wal_path)?;
        self.hnsw_index_bytes = None; // HNSW must be rebuilt by caller

        Ok(deleted_count)
    }

    /// Batch set vectors with metadata and ID mappings
    ///
    /// Note: This is a no-op. VectorStore owns all data via RecordStore.
    /// Persistence happens via checkpoint_from_snapshot which reads from RecordStore.
    #[allow(clippy::unused_self)]
    pub fn put_batch(&mut self, _items: Vec<(usize, String, Vec<f32>, JsonValue)>) -> Result<()> {
        // No-op: RecordStore is source of truth, checkpoint reads from it
        Ok(())
    }
}

// ============================================================================
// Pure I/O API
// ============================================================================

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
        snapshot.id_to_slot = self.manifest.id_to_index.clone();

        // Load deleted bitmap from manifest (RoaringBitmap -> Vec<u32>)
        snapshot.deleted = self.manifest.deleted.iter().collect();

        // Load metadata from manifest (bytes -> JsonValue)
        for (&idx, bytes) in &self.manifest.metadata {
            if let Ok(json) = serde_json::from_slice(bytes) {
                snapshot.metadata.insert(idx, json);
            }
        }

        Ok(snapshot)
    }

    /// Load snapshot from storage
    ///
    /// Returns all persisted data for initializing RecordStore. Does not modify internal state.
    pub fn load_snapshot(&self) -> io::Result<OmenSnapshot> {
        let mut snapshot = OmenSnapshot {
            dimensions: self.header.dimensions,
            ..Default::default()
        };

        // Load vectors from state
        for (idx, vec) in self.state.vectors.iter().enumerate() {
            while snapshot.vectors.len() <= idx {
                snapshot.vectors.push(None);
            }
            if !vec.is_empty() {
                snapshot.vectors[idx] = Some(vec.clone());
                // Infer dimensions from first vector if header says 0
                if snapshot.dimensions == 0 {
                    snapshot.dimensions = vec.len() as u32;
                }
            }
        }

        let dim = snapshot.dimensions as usize;

        // Load from mmap if we have persisted data
        if let Some(ref mmap) = self.mmap {
            for (idx, location) in self.manifest.nodes.iter().enumerate() {
                if location.segment_type == SegmentType::Vectors {
                    while snapshot.vectors.len() <= idx {
                        snapshot.vectors.push(None);
                    }
                    if snapshot.vectors[idx].is_none() {
                        let start = location.offset as usize;
                        let end = start + location.length as usize;
                        if end <= mmap.len() {
                            snapshot.vectors[idx] =
                                Some(read_vector_from_bytes(&mmap[start..end], dim));
                        }
                    }
                }
            }
        }

        // Copy mappings
        snapshot.id_to_slot = self
            .state
            .id_to_index
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();

        // Copy deleted
        snapshot.deleted = self.state.deleted.keys().copied().collect();

        // Copy metadata (converting from bytes to JsonValue)
        for (&idx, bytes) in &self.state.metadata {
            if let Ok(json) = serde_json::from_slice(bytes) {
                snapshot.metadata.insert(idx, json);
            }
        }

        // Copy HNSW bytes if present
        snapshot.hnsw_bytes = self.hnsw_index_bytes.clone();

        Ok(snapshot)
    }

    /// Checkpoint from external snapshot
    ///
    /// Writes vectors, metadata, and mappings from the provided data. Does not read from internal state.
    pub fn checkpoint_from_snapshot(
        &mut self,
        vectors: &[Option<Vec<f32>>],
        id_to_slot: &HashMap<String, u32>,
        deleted: &[u32],
        metadata: &HashMap<u32, serde_json::Value>,
        hnsw_bytes: Option<&[u8]>,
    ) -> io::Result<()> {
        // Drop mmap before writing
        self.mmap = None;

        let file = self
            .file
            .as_mut()
            .ok_or_else(|| io::Error::other("File not open"))?;

        // Find append point
        let file_len = file.metadata()?.len();
        let append_offset = if file_len > (HEADER_SIZE + OmenFooter::SIZE) as u64 {
            file.seek(SeekFrom::End(-(OmenFooter::SIZE as i64)))?;
            let mut footer_buf = [0u8; OmenFooter::SIZE];
            file.read_exact(&mut footer_buf)?;
            let old_footer = OmenFooter::from_bytes(&footer_buf);
            if old_footer.verify() {
                self.manifest
                    .nodes
                    .iter()
                    .filter(|n| {
                        n.segment_type == SegmentType::Vectors
                            || n.segment_type == SegmentType::IndexMetadata
                    })
                    .map(|n| n.offset + n.length as u64)
                    .max()
                    .unwrap_or(HEADER_SIZE as u64)
            } else {
                HEADER_SIZE as u64
            }
        } else {
            HEADER_SIZE as u64
        };

        let mut writer = SegmentWriter::new(file, append_offset);

        // Count of vectors already persisted
        let persisted_count = self
            .manifest
            .nodes
            .iter()
            .filter(|n| n.segment_type == SegmentType::Vectors)
            .count();

        // Clone existing vector locations
        let mut new_nodes: Vec<NodeLocation> = self
            .manifest
            .nodes
            .iter()
            .filter(|n| n.segment_type == SegmentType::Vectors)
            .copied()
            .collect();

        // Append only NEW vectors
        let dim = self.header.dimensions as usize;
        let vec_size = (dim * 4) as u32;
        let deleted_set: std::collections::HashSet<u32> = deleted.iter().copied().collect();

        for (idx, vec_opt) in vectors.iter().enumerate().skip(persisted_count) {
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
        if let Some(hnsw_data) = hnsw_bytes {
            let location = writer.write_aligned(hnsw_data, SegmentType::IndexMetadata)?;
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

        manifest.id_to_index = id_to_slot.clone();
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

        // Update config
        let live_count = vectors.len() - deleted.len();
        manifest
            .config
            .insert("count".to_string(), live_count as u64);
        manifest
            .config
            .insert("dimensions".to_string(), u64::from(self.header.dimensions));
        manifest
            .config
            .insert("hnsw_m".to_string(), u64::from(self.header.hnsw_m));
        manifest.config.insert(
            "hnsw_ef_construction".to_string(),
            u64::from(self.header.hnsw_ef_construction),
        );
        manifest.config.insert(
            "hnsw_ef_search".to_string(),
            u64::from(self.header.hnsw_ef_search),
        );
        manifest
            .config
            .insert("metric".to_string(), self.header.metric as u64);

        // Write Manifest
        let manifest_bytes = postcard::to_allocvec(&manifest)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let manifest_location = writer.write_aligned(&manifest_bytes, SegmentType::Manifest)?;

        // Write Footer
        let total_len = writer.file.stream_position()?;
        let footer = OmenFooter::new(manifest_location.offset, total_len);
        writer.file.write_all(&footer.to_bytes())?;

        // Truncate and sync
        let final_len = writer.file.stream_position()?;
        writer.file.set_len(final_len)?;
        writer.file.sync_all()?;

        // Update in-memory manifest
        self.manifest = manifest;

        // Truncate WAL
        self.wal.truncate()?;
        self.wal.append(WalEntry::checkpoint(0))?;
        self.wal.sync()?;

        // Re-establish mmap
        let file = self.file.as_ref().unwrap();
        self.mmap = Some(unsafe { MmapMut::map_mut(file)? });

        // Update header count
        self.header.count = live_count as u64;

        Ok(())
    }

    /// Get WAL length (for determining if checkpoint needed)
    #[must_use]
    pub fn wal_len(&self) -> u64 {
        self.wal.len()
    }
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn read_string_id(cursor: &mut std::io::Cursor<&[u8]>) -> io::Result<String> {
    let mut len_buf = [0u8; 4];
    cursor.read_exact(&mut len_buf)?;
    let id_len = u32::from_le_bytes(len_buf) as usize;
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

// ============================================================================
// WAL Entry Parsing (Phase 5 API)
// ============================================================================

/// Parsed insert data from a WAL entry
#[derive(Debug, Clone)]
pub struct WalInsertData {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<Vec<u8>>,
}

/// Parsed delete data from a WAL entry
#[derive(Debug, Clone)]
pub struct WalDeleteData {
    pub id: String,
}

/// Parse WAL insert entry data
///
/// Returns parsed ID, vector, and optional metadata bytes.
pub fn parse_wal_insert(data: &[u8]) -> io::Result<WalInsertData> {
    let mut cursor = std::io::Cursor::new(data);
    let string_id = read_string_id(&mut cursor)?;

    let mut buf = [0u8; 4];

    // Skip level byte (HNSW graph managed by HNSWIndex)
    cursor.read_exact(&mut buf[..1])?;

    // Read vector
    cursor.read_exact(&mut buf)?;
    let vec_len = u32::from_le_bytes(buf) as usize;
    let mut vec_bytes = vec![0u8; vec_len * 4];
    cursor.read_exact(&mut vec_bytes)?;
    let vector = read_vector_from_bytes(&vec_bytes, vec_len);

    // Read metadata
    cursor.read_exact(&mut buf)?;
    let meta_len = u32::from_le_bytes(buf) as usize;
    let metadata = if meta_len > 0 {
        let mut meta_bytes = vec![0u8; meta_len];
        cursor.read_exact(&mut meta_bytes)?;
        Some(meta_bytes)
    } else {
        None
    };

    Ok(WalInsertData {
        id: string_id,
        vector,
        metadata,
    })
}

/// Parse WAL delete entry data
///
/// Returns parsed ID.
pub fn parse_wal_delete(data: &[u8]) -> io::Result<WalDeleteData> {
    let mut cursor = std::io::Cursor::new(data);
    let id = read_string_id(&mut cursor)?;
    Ok(WalDeleteData { id })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_create_and_insert() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.omen");

        let mut db = OmenFile::create(&db_path, 3).unwrap();
        db.insert("vec1", &[1.0, 2.0, 3.0], None).unwrap();
        db.insert("vec2", &[4.0, 5.0, 6.0], None).unwrap();

        assert_eq!(db.len(), 2);
    }

    #[test]
    fn test_search() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.omen");

        let mut db = OmenFile::create(&db_path, 3).unwrap();
        db.insert("vec1", &[1.0, 0.0, 0.0], None).unwrap();
        db.insert("vec2", &[0.0, 1.0, 0.0], None).unwrap();
        db.insert("vec3", &[0.0, 0.0, 1.0], None).unwrap();

        let results = db.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "vec1");
    }

    #[test]
    fn test_checkpoint_and_reopen() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.omen");

        {
            let mut db = OmenFile::create(&db_path, 3).unwrap();
            db.insert("vec1", &[1.0, 2.0, 3.0], None).unwrap();
            db.insert("vec2", &[4.0, 5.0, 6.0], None).unwrap();
            db.checkpoint().unwrap();
        }

        {
            let db = OmenFile::open(&db_path).unwrap();
            assert_eq!(db.len(), 2);
        }
    }

    #[test]
    fn test_wal_recovery() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.omen");

        {
            let mut db = OmenFile::create(&db_path, 3).unwrap();
            db.insert("vec1", &[1.0, 2.0, 3.0], None).unwrap();
            // Don't checkpoint - data is only in WAL
        }

        {
            let db = OmenFile::open(&db_path).unwrap();
            // Should recover from WAL
            let results = db.search(&[1.0, 2.0, 3.0], 1);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, "vec1");
        }
    }

    #[test]
    fn test_footer_recovery() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_footer.omen");

        {
            let mut db = OmenFile::create(&db_path, 3).unwrap();
            db.insert("vec1", &[1.0, 2.0, 3.0], None).unwrap();
            db.checkpoint().unwrap();

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

            let results = db.search(&[1.0, 2.0, 3.0], 1);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, "vec1");
        }
    }

    #[test]
    fn test_delete_ratio() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_ratio.omen");

        let mut db = OmenFile::create(&db_path, 3).unwrap();

        // Empty db has 0 ratio
        assert_eq!(db.delete_ratio(), 0.0);
        assert!(!db.needs_compaction());

        // Insert 10 vectors
        for i in 0..10 {
            db.insert(&format!("vec{i}"), &[i as f32, 0.0, 0.0], None)
                .unwrap();
        }
        assert_eq!(db.delete_ratio(), 0.0);

        // Delete 2 vectors (20%)
        db.delete("vec0").unwrap();
        db.delete("vec1").unwrap();
        assert!((db.delete_ratio() - 0.2).abs() < 0.01);
        assert!(!db.needs_compaction()); // exactly 20% doesn't trigger

        // Delete one more (30%)
        db.delete("vec2").unwrap();
        assert!((db.delete_ratio() - 0.3).abs() < 0.01);
        assert!(db.needs_compaction()); // > 20% triggers
    }

    #[test]
    fn test_compact_basic() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_compact.omen");

        let mut db = OmenFile::create(&db_path, 3).unwrap();

        // Insert 10 vectors
        for i in 0..10 {
            db.insert(&format!("vec{i}"), &[i as f32, 0.0, 0.0], None)
                .unwrap();
        }
        db.checkpoint().unwrap();

        // Delete 5 vectors
        for i in 0..5 {
            db.delete(&format!("vec{i}")).unwrap();
        }
        assert_eq!(db.delete_ratio(), 0.5);

        // Compact
        let removed = db.compact().unwrap();
        assert_eq!(removed, 5);

        // Verify state after compaction
        assert_eq!(db.len(), 5);
        assert_eq!(db.delete_ratio(), 0.0);
        assert!(!db.needs_compaction());

        // Verify live vectors are searchable
        let results = db.search(&[5.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "vec5");

        // Verify deleted vectors are gone
        let results = db.search(&[0.0, 0.0, 0.0], 5);
        for (id, _) in &results {
            assert!(!id.starts_with("vec0"));
            assert!(!id.starts_with("vec1"));
            assert!(!id.starts_with("vec2"));
            assert!(!id.starts_with("vec3"));
            assert!(!id.starts_with("vec4"));
        }
    }

    #[test]
    fn test_compact_persistence() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_compact_persist.omen");

        {
            let mut db = OmenFile::create(&db_path, 3).unwrap();

            // Insert 10 vectors
            for i in 0..10 {
                db.insert(&format!("vec{i}"), &[i as f32, 0.0, 0.0], None)
                    .unwrap();
            }

            // Delete 5 vectors
            for i in 0..5 {
                db.delete(&format!("vec{i}")).unwrap();
            }

            // Compact
            let removed = db.compact().unwrap();
            assert_eq!(removed, 5);
        }

        // Reopen and verify
        {
            let db = OmenFile::open(&db_path).unwrap();
            assert_eq!(db.len(), 5);

            // Verify live vectors
            let results = db.search(&[7.0, 0.0, 0.0], 5);
            assert_eq!(results.len(), 5);

            // All results should be vec5-vec9
            for (id, _) in &results {
                let num: i32 = id.strip_prefix("vec").unwrap().parse().unwrap();
                assert!(num >= 5 && num <= 9, "Expected vec5-vec9, got {id}");
            }
        }
    }

    #[test]
    fn test_compact_no_deletes() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_compact_noop.omen");

        let mut db = OmenFile::create(&db_path, 3).unwrap();

        // Insert without deleting
        for i in 0..5 {
            db.insert(&format!("vec{i}"), &[i as f32, 0.0, 0.0], None)
                .unwrap();
        }
        db.checkpoint().unwrap();

        // Compact should be no-op
        let removed = db.compact().unwrap();
        assert_eq!(removed, 0);
        assert_eq!(db.len(), 5);
    }

    #[test]
    fn test_compact_preserves_metadata() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_compact_meta.omen");

        let mut db = OmenFile::create(&db_path, 3).unwrap();

        // Insert with metadata
        db.insert("vec0", &[0.0, 0.0, 0.0], Some(br#"{"key":"delete_me"}"#))
            .unwrap();
        db.insert("vec1", &[1.0, 0.0, 0.0], Some(br#"{"key":"keep"}"#))
            .unwrap();
        db.checkpoint().unwrap();

        // Delete vec0
        db.delete("vec0").unwrap();

        // Compact
        db.compact().unwrap();

        // Verify metadata preserved for vec1
        // vec1 was at index 1, now should be at index 0
        let meta = db.get_metadata(0).unwrap();
        assert!(meta.is_some());
        let meta_json = meta.unwrap();
        assert_eq!(meta_json["key"], "keep");
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

        // WAL should have entries, but state is NOT updated
        assert!(db.wal_len() > 0);
        // Internal state should be empty since we only wrote to WAL
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_load_snapshot() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_snapshot.omen");

        // Create and populate using old API
        {
            let mut db = OmenFile::create(&db_path, 3).unwrap();
            db.insert("vec1", &[1.0, 2.0, 3.0], Some(br#"{"k":"v1"}"#))
                .unwrap();
            db.insert("vec2", &[4.0, 5.0, 6.0], Some(br#"{"k":"v2"}"#))
                .unwrap();
            db.delete("vec1").unwrap();
            db.checkpoint().unwrap();
        }

        // Open and load snapshot
        {
            let db = OmenFile::open(&db_path).unwrap();
            let snapshot = db.load_snapshot().unwrap();

            assert_eq!(snapshot.dimensions, 3);
            assert_eq!(snapshot.id_to_slot.len(), 1); // Only vec2 in mapping
            assert!(snapshot.id_to_slot.contains_key("vec2"));
            assert!(!snapshot.id_to_slot.contains_key("vec1")); // Deleted

            // vec2's slot should have vector data
            let slot = snapshot.id_to_slot["vec2"] as usize;
            assert!(snapshot.vectors[slot].is_some());
            let vec_data = snapshot.vectors[slot].as_ref().unwrap();
            assert_eq!(vec_data, &[4.0, 5.0, 6.0]);

            // Metadata should be present
            assert!(snapshot.metadata.contains_key(&(slot as u32)));
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
        db.checkpoint_from_snapshot(&vectors, &id_to_slot, &deleted, &metadata, None)
            .unwrap();

        // Verify internal state was updated
        assert_eq!(db.len(), 2);

        drop(db);

        // Reopen and verify
        let db2 = OmenFile::open(&db_path).unwrap();
        assert_eq!(db2.len(), 2);

        // Search should find vec1 and vec2
        let results = db2.search(&[1.0, 2.0, 3.0], 2);
        assert_eq!(results.len(), 2);

        // vec1 should be closest to query
        assert_eq!(results[0].0, "vec1");
    }

    #[test]
    fn test_delete_round_trip() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_delete_rt.omen");

        // Phase 1: Create and populate
        {
            let mut db = OmenFile::create(&db_path, 3).unwrap();
            db.insert("vec1", &[1.0, 0.0, 0.0], None).unwrap();
            db.insert("vec2", &[0.0, 1.0, 0.0], None).unwrap();
            db.insert("vec3", &[0.0, 0.0, 1.0], None).unwrap();
            db.checkpoint().unwrap();
        }

        // Phase 2: Reopen and delete using WAL-only
        {
            let mut db = OmenFile::open(&db_path).unwrap();
            assert_eq!(db.len(), 3);

            // WAL-only delete (RecordStore-based deletion)
            db.wal_append_delete("vec2").unwrap();

            // Checkpoint with updated deleted set
            let snapshot = db.load_snapshot().unwrap();

            // Convert snapshot to checkpoint format, adding vec2's slot to deleted
            let mut deleted: Vec<u32> = snapshot.deleted;
            if let Some(&slot) = snapshot.id_to_slot.get("vec2") {
                deleted.push(slot);
            }

            // Remove vec2 from id_to_slot
            let mut id_to_slot = snapshot.id_to_slot.clone();
            id_to_slot.remove("vec2");

            // Build vectors slice
            let vectors: Vec<Option<Vec<f32>>> = snapshot.vectors;

            db.checkpoint_from_snapshot(&vectors, &id_to_slot, &deleted, &snapshot.metadata, None)
                .unwrap();
        }

        // Phase 3: Reopen and verify deletion persisted
        {
            let db = OmenFile::open(&db_path).unwrap();
            assert_eq!(db.len(), 2);

            // Search should only find vec1 and vec3
            let results = db.search(&[0.0, 1.0, 0.0], 3);
            assert_eq!(results.len(), 2);

            // vec2 should NOT be in results (it was deleted)
            for (id, _) in &results {
                assert_ne!(id, "vec2", "vec2 should have been deleted");
            }
        }
    }
}
