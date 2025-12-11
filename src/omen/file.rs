//! OmenFile - main API for .omen format

use crate::omen::{
    align_to_page,
    graph::GraphSection,
    header::{OmenHeader, HEADER_SIZE},
    section::{SectionEntry, SectionType},
    vectors::VectorSection,
    wal::{Wal, WalEntry, WalEntryType},
};
use memmap2::MmapMut;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Checkpoint threshold (number of WAL entries before compaction)
const CHECKPOINT_THRESHOLD: u64 = 1000;

/// OmenFile - single-file vector database
pub struct OmenFile {
    path: PathBuf,
    file: File,
    mmap: Option<MmapMut>,
    header: OmenHeader,

    // In-memory state (for writes before checkpoint)
    vectors_mem: Vec<Vec<f32>>,
    graph_mem: Vec<Vec<u32>>,
    levels_mem: Vec<u8>,
    id_to_index: HashMap<String, u32>,
    index_to_id: HashMap<u32, String>,
    metadata_mem: HashMap<u32, Vec<u8>>,
    deleted: HashMap<u32, bool>,

    // WAL for durability
    wal: Wal,

    // Entry point for HNSW search
    entry_point: Option<u32>,
}

impl OmenFile {
    /// Create a new .omen database
    pub fn create(path: impl AsRef<Path>, dimensions: u32) -> io::Result<Self> {
        let path = path.as_ref();
        let omen_path = if path.extension().is_some_and(|ext| ext == "omen") {
            path.to_path_buf()
        } else {
            path.with_extension("omen")
        };

        let wal_path = omen_path.with_extension("wal");

        // Create empty file with header
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&omen_path)?;

        let header = OmenHeader::new(dimensions);
        file.write_all(&header.to_bytes())?;
        file.sync_all()?;

        let wal = Wal::open(&wal_path)?;

        Ok(Self {
            path: omen_path,
            file,
            mmap: None,
            header,
            vectors_mem: Vec::new(),
            graph_mem: Vec::new(),
            levels_mem: Vec::new(),
            id_to_index: HashMap::new(),
            index_to_id: HashMap::new(),
            metadata_mem: HashMap::new(),
            deleted: HashMap::new(),
            wal,
            entry_point: None,
        })
    }

    /// Open an existing .omen database
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let omen_path = if path.extension().is_some_and(|ext| ext == "omen") {
            path.to_path_buf()
        } else {
            path.with_extension("omen")
        };

        let wal_path = omen_path.with_extension("wal");

        let mut file = OpenOptions::new().read(true).write(true).open(&omen_path)?;

        // Read header
        let mut header_buf = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_buf)?;
        let header = OmenHeader::from_bytes(&header_buf)?;

        // Create mmap if file has data
        let file_len = file.metadata()?.len() as usize;
        let mmap = if file_len > HEADER_SIZE {
            Some(unsafe { MmapMut::map_mut(&file)? })
        } else {
            None
        };

        // Open WAL
        let wal = Wal::open(&wal_path)?;

        // Save entry point before moving header
        let entry_point = if header.entry_point > 0 {
            Some(header.entry_point)
        } else {
            None
        };

        let mut db = Self {
            path: omen_path,
            file,
            mmap,
            header,
            vectors_mem: Vec::new(),
            graph_mem: Vec::new(),
            levels_mem: Vec::new(),
            id_to_index: HashMap::new(),
            index_to_id: HashMap::new(),
            metadata_mem: HashMap::new(),
            deleted: HashMap::new(),
            wal,
            entry_point,
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
                // Skip corrupted entries
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
                WalEntryType::UpdateMetadata => {
                    // TODO: implement metadata replay
                }
                WalEntryType::Checkpoint => {
                    // Checkpoint - nothing to replay
                }
            }
        }

        Ok(())
    }

    /// Replay insert from WAL
    fn replay_insert(&mut self, data: &[u8]) -> io::Result<()> {
        let mut cursor = std::io::Cursor::new(data);

        // Read string ID
        let mut len_buf = [0u8; 4];
        cursor.read_exact(&mut len_buf)?;
        let id_len = u32::from_le_bytes(len_buf) as usize;
        let mut id_buf = vec![0u8; id_len];
        cursor.read_exact(&mut id_buf)?;
        let string_id = String::from_utf8_lossy(&id_buf).to_string();

        // Read level
        let mut level_buf = [0u8; 1];
        cursor.read_exact(&mut level_buf)?;
        let level = level_buf[0];

        // Read vector
        cursor.read_exact(&mut len_buf)?;
        let vec_len = u32::from_le_bytes(len_buf) as usize;
        let mut vector = vec![0.0f32; vec_len];
        for val in &mut vector {
            let mut f32_buf = [0u8; 4];
            cursor.read_exact(&mut f32_buf)?;
            *val = f32::from_le_bytes(f32_buf);
        }

        // Read metadata
        cursor.read_exact(&mut len_buf)?;
        let meta_len = u32::from_le_bytes(len_buf) as usize;
        let mut metadata = vec![0u8; meta_len];
        cursor.read_exact(&mut metadata)?;

        // Apply insert
        let index = self.vectors_mem.len() as u32;
        self.vectors_mem.push(vector);
        self.levels_mem.push(level);
        self.graph_mem.push(Vec::new());
        self.id_to_index.insert(string_id.clone(), index);
        self.index_to_id.insert(index, string_id);
        if !metadata.is_empty() {
            self.metadata_mem.insert(index, metadata);
        }

        if self.entry_point.is_none() {
            self.entry_point = Some(index);
        }

        Ok(())
    }

    /// Replay delete from WAL
    fn replay_delete(&mut self, data: &[u8]) -> io::Result<()> {
        let mut cursor = std::io::Cursor::new(data);

        // Read string ID
        let mut len_buf = [0u8; 4];
        cursor.read_exact(&mut len_buf)?;
        let id_len = u32::from_le_bytes(len_buf) as usize;
        let mut id_buf = vec![0u8; id_len];
        cursor.read_exact(&mut id_buf)?;
        let string_id = String::from_utf8_lossy(&id_buf).to_string();

        if let Some(&index) = self.id_to_index.get(&string_id) {
            self.deleted.insert(index, true);
        }

        Ok(())
    }

    /// Replay neighbors update from WAL
    fn replay_neighbors(&mut self, data: &[u8]) -> io::Result<()> {
        let mut cursor = std::io::Cursor::new(data);

        // Read node ID
        let mut u32_buf = [0u8; 4];
        cursor.read_exact(&mut u32_buf)?;
        let node_id = u32::from_le_bytes(u32_buf);

        // Read level (unused for now - we only store level 0)
        let mut level_buf = [0u8; 1];
        cursor.read_exact(&mut level_buf)?;

        // Read neighbors
        cursor.read_exact(&mut u32_buf)?;
        let neighbor_count = u32::from_le_bytes(u32_buf) as usize;
        let mut neighbors = Vec::with_capacity(neighbor_count);
        for _ in 0..neighbor_count {
            cursor.read_exact(&mut u32_buf)?;
            neighbors.push(u32::from_le_bytes(u32_buf));
        }

        if (node_id as usize) < self.graph_mem.len() {
            self.graph_mem[node_id as usize] = neighbors;
        }

        Ok(())
    }

    /// Insert a vector
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
        let level = self.random_level();

        // 1. Append to WAL (durable)
        let entry = WalEntry::insert_node(0, id, level, vector, metadata_bytes);
        self.wal.append(entry)?;
        self.wal.sync()?;

        // 2. Update in-memory index
        let index = self.vectors_mem.len() as u32;
        self.vectors_mem.push(vector.to_vec());
        self.levels_mem.push(level);
        self.graph_mem.push(Vec::new());
        self.id_to_index.insert(id.to_string(), index);
        self.index_to_id.insert(index, id.to_string());
        if metadata_bytes != b"{}" {
            self.metadata_mem.insert(index, metadata_bytes.to_vec());
        }

        // 3. Connect to graph (simplified - full HNSW would be more complex)
        if self.entry_point.is_none() {
            self.entry_point = Some(index);
        } else {
            // Simple greedy insert - find nearest neighbors
            let neighbors = self.find_nearest(vector, self.header.m as usize);

            // Update this node's neighbors
            self.graph_mem[index as usize] = neighbors.clone();

            // Log neighbor update
            let neighbor_entry = WalEntry::update_neighbors(0, index, 0, &neighbors);
            self.wal.append(neighbor_entry)?;

            // Update bidirectional connections
            for &neighbor in &neighbors {
                if (neighbor as usize) < self.graph_mem.len() {
                    let neighbor_list = &mut self.graph_mem[neighbor as usize];
                    if !neighbor_list.contains(&index)
                        && neighbor_list.len() < self.header.m as usize * 2
                    {
                        neighbor_list.push(index);

                        // Log neighbor update
                        let update = WalEntry::update_neighbors(0, neighbor, 0, neighbor_list);
                        self.wal.append(update)?;
                    }
                }
            }
        }

        self.header.count += 1;

        // 4. Periodic checkpoint
        if self.wal.len() > CHECKPOINT_THRESHOLD {
            self.checkpoint()?;
        }

        Ok(())
    }

    /// Find k nearest neighbors (simple greedy search)
    fn find_nearest(&self, query: &[f32], k: usize) -> Vec<u32> {
        if self.vectors_mem.is_empty() {
            return Vec::new();
        }

        // Simple brute force for now - full HNSW search would be more efficient
        let mut distances: Vec<(u32, f32)> = self
            .vectors_mem
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.deleted.get(&(*i as u32)).copied().unwrap_or(false))
            .map(|(i, v)| (i as u32, l2_distance(query, v)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        distances.into_iter().take(k).map(|(id, _)| id).collect()
    }

    /// Search for k nearest neighbors
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

    /// Delete a vector by ID
    pub fn delete(&mut self, id: &str) -> io::Result<bool> {
        if let Some(&index) = self.id_to_index.get(id) {
            // Log to WAL
            let entry = WalEntry::delete_node(0, id);
            self.wal.append(entry)?;
            self.wal.sync()?;

            self.deleted.insert(index, true);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get vector count
    pub fn len(&self) -> u64 {
        self.header.count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.header.count == 0
    }

    /// Get dimensions
    pub fn dimensions(&self) -> u32 {
        self.header.dimensions
    }

    /// Generate random level for HNSW (simplified)
    fn random_level(&self) -> u8 {
        let ml = 1.0 / (self.header.m as f64).ln();
        let mut level = 0u8;
        let mut r = rand::random::<f64>();
        while r < ml && level < 16 {
            level += 1;
            r = rand::random::<f64>();
        }
        level
    }

    /// Checkpoint - compact WAL into main file
    pub fn checkpoint(&mut self) -> io::Result<()> {
        if self.vectors_mem.is_empty() {
            return Ok(());
        }

        // Calculate section sizes
        let vector_size =
            VectorSection::size_for_count(self.header.dimensions, self.vectors_mem.len() as u64)
                as usize;
        let graph_size = GraphSection::size_for_graph(&self.levels_mem, &self.graph_mem);

        // Calculate offsets (page-aligned)
        let vector_offset = align_to_page(HEADER_SIZE);
        let graph_offset = align_to_page(vector_offset + vector_size);
        let total_size = align_to_page(graph_offset + graph_size);

        // Extend file
        self.file.set_len(total_size as u64)?;

        // Write vectors
        self.file.seek(SeekFrom::Start(vector_offset as u64))?;
        for vector in &self.vectors_mem {
            for &val in vector {
                self.file.write_all(&val.to_le_bytes())?;
            }
        }

        // Write graph
        self.file.seek(SeekFrom::Start(graph_offset as u64))?;
        GraphSection::write_graph(&mut self.file, &self.levels_mem, &self.graph_mem)?;

        // Update header
        self.header.count = self.vectors_mem.len() as u64;
        self.header.entry_point = self.entry_point.unwrap_or(0);
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

        // Write header
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&self.header.to_bytes())?;
        self.file.sync_all()?;

        // Truncate WAL
        self.wal.truncate()?;

        // Write checkpoint marker
        self.wal.append(WalEntry::checkpoint(0))?;
        self.wal.sync()?;

        // Update mmap
        self.mmap = Some(unsafe { MmapMut::map_mut(&self.file)? });

        Ok(())
    }
}

/// L2 distance between two vectors
#[inline]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
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
