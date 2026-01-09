//! `OmenFile` - main API for .omen format
//!
//! Storage backend for `VectorStore`. Uses postcard for efficient binary serialization.

use crate::omen::{
    align_to_page,
    header::{OmenHeader, HEADER_SIZE},
    section::{SectionEntry, SectionType},
    vectors::VectorSection,
    wal::{Wal, WalEntry, WalEntryType},
};
use anyhow::Result;
use fs2::FileExt;
use memmap2::MmapMut;
use serde::{Deserialize, Serialize};
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

#[derive(Serialize, Deserialize, Default)]
struct CheckpointMetadata {
    id_to_index: HashMap<String, u32>,
    index_to_id: HashMap<u32, String>,
    deleted: HashMap<u32, bool>,
    config: HashMap<String, u64>,
    metadata: HashMap<u32, Vec<u8>>,
}

/// Checkpoint threshold (number of WAL entries before compaction)
const CHECKPOINT_THRESHOLD: u64 = 1000;

/// `OmenFile` - single-file vector database
///
/// Storage layer for vectors, metadata, and serialized HNSW index.
/// Graph traversal is handled by `HNSWIndex` in the vector layer.
pub struct OmenFile {
    path: PathBuf,
    file: File,
    mmap: Option<MmapMut>,
    header: OmenHeader,

    // In-memory state (for writes before checkpoint)
    vectors_mem: Vec<Vec<f32>>,
    id_to_index: HashMap<String, u32>,
    index_to_id: HashMap<u32, String>,
    metadata_mem: HashMap<u32, Vec<u8>>,
    deleted: HashMap<u32, bool>,
    config: HashMap<String, u64>,

    // WAL for durability
    wal: Wal,

    // Serialized HNSW index (persisted on checkpoint, loaded on open)
    hnsw_index_bytes: Option<Vec<u8>>,
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
        file.sync_all()?;

        Ok(Self {
            path: omen_path,
            file,
            mmap: None,
            header,
            vectors_mem: Vec::new(),
            id_to_index: HashMap::new(),
            index_to_id: HashMap::new(),
            metadata_mem: HashMap::new(),
            deleted: HashMap::new(),
            config: HashMap::from([("dimensions".to_string(), u64::from(dimensions))]),
            wal: Wal::open(&wal_path)?,
            hnsw_index_bytes: None,
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

        let mut header_buf = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_buf)?;
        let header = OmenHeader::from_bytes(&header_buf)?;

        let file_len = file.metadata()?.len() as usize;
        let mmap = if file_len > HEADER_SIZE {
            Some(unsafe { MmapMut::map_mut(&file)? })
        } else {
            None
        };

        let wal = Wal::open(&wal_path)?;
        let mut config = HashMap::from([
            ("dimensions".to_string(), u64::from(header.dimensions)),
            ("count".to_string(), header.count),
        ]);

        let mut vectors_mem = Vec::new();
        let mut id_to_index = HashMap::new();
        let mut index_to_id = HashMap::new();
        let mut metadata_mem = HashMap::new();
        let mut deleted = HashMap::new();

        if let Some(ref mmap) = mmap {
            // Load vectors
            if let Some(vec_section) = header.get_section(SectionType::Vectors) {
                let vec_offset = vec_section.offset as usize;
                let dim = header.dimensions as usize;
                let count = header.count as usize;
                let vec_size = dim * 4;

                for i in 0..count {
                    let start = vec_offset + i * vec_size;
                    let end = start + vec_size;
                    if end <= mmap.len() {
                        vectors_mem.push(read_vector_from_bytes(&mmap[start..end], dim));
                    }
                }
            }

            // Load metadata section (postcard-encoded CheckpointMetadata)
            if let Some(meta_section) = header.get_section(SectionType::MetadataRaw) {
                let meta_offset = meta_section.offset as usize;
                let meta_len = meta_section.length as usize;
                if meta_offset + meta_len <= mmap.len() {
                    let meta_bytes = &mmap[meta_offset..meta_offset + meta_len];
                    if let Ok(meta) = postcard::from_bytes::<CheckpointMetadata>(meta_bytes) {
                        id_to_index = meta.id_to_index;
                        index_to_id = meta.index_to_id;
                        deleted = meta.deleted;
                        config.extend(meta.config);
                        metadata_mem = meta.metadata;
                    }
                }
            }
        }

        let hnsw_index_bytes = mmap
            .as_ref()
            .and_then(|m| header.get_section(SectionType::HnswIndex).map(|s| (m, s)))
            .and_then(|(m, s)| {
                let start = s.offset as usize;
                let end = start + s.length as usize;
                (end <= m.len()).then(|| m[start..end].to_vec())
            });

        let mut db = Self {
            path: omen_path,
            file,
            mmap,
            header,
            vectors_mem,
            id_to_index,
            index_to_id,
            metadata_mem,
            deleted,
            config,
            wal,
            hnsw_index_bytes,
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

        let index = self.vectors_mem.len() as u32;
        self.vectors_mem.push(vector);
        self.id_to_index.insert(string_id.clone(), index);
        self.index_to_id.insert(index, string_id);
        if !metadata.is_empty() {
            self.metadata_mem.insert(index, metadata);
        }

        Ok(())
    }

    fn replay_delete(&mut self, data: &[u8]) -> io::Result<()> {
        let mut cursor = std::io::Cursor::new(data);
        let string_id = read_string_id(&mut cursor)?;

        if let Some(&index) = self.id_to_index.get(&string_id) {
            self.deleted.insert(index, true);
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
        let index = self.vectors_mem.len() as u32;
        self.vectors_mem.push(vector.to_vec());
        self.id_to_index.insert(id.to_string(), index);
        self.index_to_id.insert(index, id.to_string());
        if metadata_bytes != b"{}" {
            self.metadata_mem.insert(index, metadata_bytes.to_vec());
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
            .vectors_mem
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.deleted.contains_key(&(*i as u32)))
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
                let id = self.index_to_id.get(&idx)?;
                let vector = self.vectors_mem.get(idx as usize)?;
                let distance = l2_distance(query, vector);
                Some((id.clone(), distance))
            })
            .collect()
    }

    pub fn delete(&mut self, id: &str) -> io::Result<bool> {
        let Some(&index) = self.id_to_index.get(id) else {
            return Ok(false);
        };

        self.wal.append(WalEntry::delete_node(0, id))?;
        self.wal.sync()?;
        self.deleted.insert(index, true);
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

    /// Checkpoint - compact WAL into main file (atomic via temp-file-rename)
    ///
    /// Uses write-to-temp-then-rename pattern for crash safety:
    /// 1. Write complete file to `.omen.tmp`
    /// 2. Fsync temp file
    /// 3. Rename temp to actual (atomic on POSIX)
    /// 4. Fsync directory
    /// 5. Truncate WAL
    pub fn checkpoint(&mut self) -> io::Result<()> {
        if self.vectors_mem.is_empty() && self.hnsw_index_bytes.is_none() {
            return Ok(());
        }

        // Serialize metadata with postcard (compact, fast, actively maintained)
        let checkpoint_meta = CheckpointMetadata {
            id_to_index: self.id_to_index.clone(),
            index_to_id: self.index_to_id.clone(),
            deleted: self.deleted.clone(),
            config: self.config.clone(),
            metadata: self.metadata_mem.clone(),
        };
        let metadata_bytes = postcard::to_allocvec(&checkpoint_meta)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Calculate section sizes
        let vector_size =
            VectorSection::size_for_count(self.header.dimensions, self.vectors_mem.len() as u64)
                as usize;
        let graph_size = 0; // Graph managed by HNSWIndex, not stored in OmenFile
        let metadata_size = metadata_bytes.len();
        let hnsw_size = self.hnsw_index_bytes.as_ref().map_or(0, Vec::len);

        // Calculate offsets (page-aligned)
        let vector_offset = align_to_page(HEADER_SIZE);
        let graph_offset = align_to_page(vector_offset + vector_size);
        let metadata_offset = align_to_page(graph_offset + graph_size);
        let hnsw_offset = align_to_page(metadata_offset + metadata_size);
        let total_size = align_to_page(hnsw_offset + hnsw_size);

        // Create temp file path
        let temp_path = {
            let mut p = self.path.as_os_str().to_os_string();
            p.push(".tmp");
            PathBuf::from(p)
        };

        // Update header for new checkpoint
        self.header.count = self.vectors_mem.len() as u64;
        self.header.entry_point = 0; // Entry point managed by HNSWIndex
        self.header.set_section(SectionEntry::new(
            SectionType::Vectors,
            vector_offset as u64,
            vector_size as u64,
        ));
        self.header.set_section(SectionEntry::new(
            SectionType::Graph,
            graph_offset as u64,
            graph_size as u64,
        ));
        self.header.set_section(SectionEntry::new(
            SectionType::MetadataRaw,
            metadata_offset as u64,
            metadata_size as u64,
        ));
        if hnsw_size > 0 {
            self.header.set_section(SectionEntry::new(
                SectionType::HnswIndex,
                hnsw_offset as u64,
                hnsw_size as u64,
            ));
        }

        // Write to temp file
        {
            let mut opts = OpenOptions::new();
            opts.write(true).create(true).truncate(true);
            configure_open_options(&mut opts);
            let mut temp_file = opts.open(&temp_path)?;
            temp_file.set_len(total_size as u64)?;

            // Write header
            temp_file.write_all(&self.header.to_bytes())?;

            // Write vectors
            temp_file.seek(SeekFrom::Start(vector_offset as u64))?;
            for vector in &self.vectors_mem {
                for &val in vector {
                    temp_file.write_all(&val.to_le_bytes())?;
                }
            }

            // Write metadata
            temp_file.seek(SeekFrom::Start(metadata_offset as u64))?;
            temp_file.write_all(&metadata_bytes)?;

            // Write HNSW index (if present)
            if let Some(ref hnsw_bytes) = self.hnsw_index_bytes {
                temp_file.seek(SeekFrom::Start(hnsw_offset as u64))?;
                temp_file.write_all(hnsw_bytes)?;
            }

            // Fsync temp file before rename
            temp_file.sync_all()?;
        }

        // Drop mmap before rename (required - file handle must be released)
        self.mmap = None;

        // Atomic rename (POSIX guarantees atomicity for same-filesystem rename)
        std::fs::rename(&temp_path, &self.path)?;

        // Fsync directory to ensure rename is durable
        #[cfg(unix)]
        if let Some(parent) = self.path.parent() {
            if let Ok(dir) = File::open(parent) {
                let _ = dir.sync_all();
            }
        }

        let mut opts = OpenOptions::new();
        opts.read(true).write(true);
        configure_open_options(&mut opts);
        self.file = opts.open(&self.path)?;
        lock_exclusive(&self.file)?;

        self.wal.truncate()?;
        self.wal.append(WalEntry::checkpoint(0))?;
        self.wal.sync()?;
        self.mmap = Some(unsafe { MmapMut::map_mut(&self.file)? });

        Ok(())
    }
}

// ============================================================================
// Storage API for VectorStore
// ============================================================================

impl OmenFile {
    /// Store a vector by internal index
    pub fn put_vector(&mut self, id: usize, vector: &[f32]) -> Result<()> {
        let new_len = id + 1;
        if self.vectors_mem.len() < new_len {
            self.vectors_mem.resize_with(new_len, Vec::new);
        }
        self.vectors_mem[id] = vector.to_vec();
        Ok(())
    }

    pub fn get_vector(&self, id: usize) -> Result<Option<Vec<f32>>> {
        if id < self.vectors_mem.len() && !self.vectors_mem[id].is_empty() {
            return Ok(Some(self.vectors_mem[id].clone()));
        }

        let Some(ref mmap) = self.mmap else {
            return Ok(None);
        };
        let Some(vec_section) = self.header.get_section(SectionType::Vectors) else {
            return Ok(None);
        };

        let dim = self.header.dimensions as usize;
        if id >= self.header.count as usize {
            return Ok(None);
        }

        let vec_size = dim * 4;
        let start = vec_section.offset as usize + id * vec_size;
        let end = start + vec_size;

        if end <= mmap.len() {
            Ok(Some(read_vector_from_bytes(&mmap[start..end], dim)))
        } else {
            Ok(None)
        }
    }

    /// Store metadata for a vector (as JSON)
    pub fn put_metadata(&mut self, id: usize, metadata: &JsonValue) -> Result<()> {
        let bytes = serde_json::to_vec(metadata)?;
        self.metadata_mem.insert(id as u32, bytes);
        Ok(())
    }

    pub fn get_metadata(&self, id: usize) -> Result<Option<JsonValue>> {
        self.metadata_mem
            .get(&(id as u32))
            .map(|bytes| serde_json::from_slice(bytes))
            .transpose()
            .map_err(Into::into)
    }

    /// Store string ID to internal index mapping
    pub fn put_id_mapping(&mut self, string_id: &str, index: usize) -> Result<()> {
        self.id_to_index.insert(string_id.to_string(), index as u32);
        self.index_to_id.insert(index as u32, string_id.to_string());
        Ok(())
    }

    /// Get internal index for a string ID
    pub fn get_id_mapping(&self, string_id: &str) -> Result<Option<usize>> {
        Ok(self.id_to_index.get(string_id).map(|&idx| idx as usize))
    }

    /// Get string ID for an internal index (reverse lookup)
    pub fn get_string_id(&self, index: usize) -> Result<Option<String>> {
        Ok(self.index_to_id.get(&(index as u32)).cloned())
    }

    /// Delete string ID mapping
    pub fn delete_id_mapping(&mut self, string_id: &str) -> Result<()> {
        if let Some(&index) = self.id_to_index.get(string_id) {
            self.index_to_id.remove(&index);
        }
        self.id_to_index.remove(string_id);
        Ok(())
    }

    /// Store configuration value
    pub fn put_config(&mut self, key: &str, value: u64) -> Result<()> {
        self.config.insert(key.to_string(), value);
        // Sync dimensions to header
        if key == "dimensions" {
            self.header.dimensions = value as u32;
        }
        Ok(())
    }

    /// Get configuration value
    pub fn get_config(&self, key: &str) -> Result<Option<u64>> {
        Ok(self.config.get(key).copied())
    }

    /// Load all vectors from storage
    pub fn load_all_vectors(&self) -> Result<Vec<(usize, Vec<f32>)>> {
        Ok(self
            .vectors_mem
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .map(|(id, v)| (id, v.clone()))
            .collect())
    }

    /// Increment vector count in storage
    pub fn increment_count(&mut self) -> Result<usize> {
        let count = self.config.get("count").copied().unwrap_or(0) as usize;
        let new_count = count + 1;
        self.config.insert("count".to_string(), new_count as u64);
        self.header.count = new_count as u64;
        Ok(new_count)
    }

    /// Get current vector count
    pub fn get_count(&self) -> Result<usize> {
        Ok(self.config.get("count").copied().unwrap_or(0) as usize)
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
            .metadata_mem
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
            .id_to_index
            .iter()
            .map(|(id, &idx)| (id.clone(), idx as usize))
            .collect())
    }

    /// Mark a vector as deleted (tombstone)
    pub fn put_deleted(&mut self, id: usize) -> Result<()> {
        self.deleted.insert(id as u32, true);
        Ok(())
    }

    pub fn is_deleted(&self, id: usize) -> Result<bool> {
        Ok(self.deleted.contains_key(&(id as u32)))
    }

    /// Remove deleted marker (for re-insertion)
    pub fn remove_deleted(&mut self, id: usize) -> Result<()> {
        self.deleted.remove(&(id as u32));
        Ok(())
    }

    /// Load all deleted IDs from storage
    pub fn load_all_deleted(&self) -> Result<HashMap<usize, bool>> {
        Ok(self
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
        self.header.m = m;
        self.header.ef_construction = ef_construction;
        self.header.ef_search = ef_search;
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

    /// Batch set vectors with metadata and ID mappings
    pub fn put_batch(&mut self, items: Vec<(usize, String, Vec<f32>, JsonValue)>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }

        for (idx, string_id, vector, metadata) in items {
            self.put_vector(idx, &vector)?;
            self.put_metadata(idx, &metadata)?;
            self.put_id_mapping(&string_id, idx)?;
        }

        // Update count
        let current_count = self.get_count()?;
        let new_count = self
            .vectors_mem
            .iter()
            .filter(|v| !v.is_empty())
            .count()
            .max(current_count);
        self.config.insert("count".to_string(), new_count as u64);
        self.header.count = new_count as u64;

        Ok(())
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
}
