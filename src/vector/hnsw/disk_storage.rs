//! Disk-based storage for HNSW-IF using memory-mapped files
//!
//! This module provides mmap-based disk storage for HNSW graphs,
//! enabling billion-scale vector search on commodity hardware.
//!
//! Features:
//! - Memory-mapped file I/O (OS handles paging)
//! - madvise for access pattern hints (`MADV_RANDOM`)
//! - Optional pre-population (populate parameter)
//! - Versioned binary format for forward compatibility
//!
//! File format:
//! - metadata.bin: Graph metadata (entry point, params, version)
//! - `layer_0.graph`: Node neighbor lists (mmap-friendly format)

use super::error::{HNSWError, Result};
use super::node_storage::{Level, NodeId, NodeStorage};
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Magic number for file format validation
const MAGIC_NUMBER: &[u8; 8] = b"OMENDBIF"; // OmenDB Inverted File

/// Current file format version (2 = delta encoding with vbyte)
const FORMAT_VERSION: u32 = 2;

/// Metadata header size (64 bytes, cache-line aligned)
/// Note: Not page-aligned since metadata is read once at startup, not mmap'd for random access
const METADATA_HEADER_SIZE: usize = 64;

// ============================================================================
// Variable-byte (VByte) encoding for delta-compressed neighbor IDs
// ============================================================================

/// Encode a u32 value using variable-byte encoding (7 bits per byte, MSB=continuation)
/// Returns the number of bytes written
#[inline]
fn vbyte_encode(mut value: u32, buf: &mut [u8]) -> usize {
    let mut i = 0;
    while value >= 0x80 {
        buf[i] = (value as u8) | 0x80;
        value >>= 7;
        i += 1;
    }
    buf[i] = value as u8;
    i + 1
}

/// Decode a vbyte-encoded value from a byte slice.
/// Returns (value, `bytes_read`), or (0, 0) if buffer is empty/invalid.
#[inline]
fn vbyte_decode(buf: &[u8]) -> (u32, usize) {
    if buf.is_empty() {
        return (0, 0);
    }
    let mut value: u32 = 0;
    let mut shift = 0;
    let mut i = 0;
    loop {
        if i >= buf.len() {
            // Truncated vbyte - return what we have
            return (value, i);
        }
        let byte = buf[i];
        value |= ((byte & 0x7F) as u32) << shift;
        i += 1;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 35 {
            // Overflow protection (max 5 bytes for u32)
            break;
        }
    }
    (value, i)
}

/// Write neighbor IDs as vbyte-encoded absolute values (shared helper)
/// Preserves original order for HNSW distance-based neighbor lists
fn write_neighbors_vbyte<W: Write>(writer: &mut W, neighbors: &[NodeId]) -> std::io::Result<()> {
    let mut vbyte_buf = [0u8; 5]; // Max 5 bytes for u32 vbyte
    for &neighbor_id in neighbors {
        let len = vbyte_encode(neighbor_id, &mut vbyte_buf);
        writer.write_all(&vbyte_buf[..len])?;
    }
    Ok(())
}

/// Write neighbor IDs as vbyte and return total bytes written
fn write_neighbors_vbyte_counted<W: Write>(
    writer: &mut W,
    neighbors: &[NodeId],
) -> std::io::Result<usize> {
    let mut vbyte_buf = [0u8; 5];
    let mut total = 0;
    for &neighbor_id in neighbors {
        let len = vbyte_encode(neighbor_id, &mut vbyte_buf);
        writer.write_all(&vbyte_buf[..len])?;
        total += len;
    }
    Ok(total)
}

/// Disk storage using memory-mapped files
///
/// This storage backend uses mmap for efficient disk I/O without loading
/// entire files into memory. The OS handles paging automatically.
#[derive(Debug)]
pub struct DiskStorage {
    /// Path to storage directory
    #[allow(dead_code)]
    path: PathBuf,

    /// Memory-mapped graph file (neighbor lists)
    graph_mmap: Arc<Mmap>,

    /// Memory-mapped offset index (`node_id` → file offset)
    ///
    /// Optional: If present, enables O(1) node access.
    /// If absent, falls back to O(n) scanning.
    offset_mmap: Option<Arc<Mmap>>,

    /// Graph metadata
    metadata: GraphMetadata,

    /// Whether to pre-populate pages (pre-fault all pages into memory)
    ///
    /// Stored for documentation but not read after initialization.
    #[allow(dead_code)]
    populate: bool,
}

/// Graph metadata stored in metadata.bin
#[derive(Debug, Clone)]
struct GraphMetadata {
    /// File format version
    version: u32,

    /// Storage mode (0=Memory, 1=Hybrid, 2=DiskHeavy)
    storage_mode: u32,

    /// Total number of nodes
    num_nodes: u64,

    /// Maximum level in graph
    max_level: u32,

    /// Entry node ID
    entry_node_id: u32,

    /// Entry node level
    entry_level: u32,

    /// M parameter (max neighbors per level)
    m: u32,

    /// M0 parameter (max neighbors at base layer)
    m0: u32,

    /// `ef_construction` parameter
    ef_construction: u32,
}

/// Node entry in graph file
///
/// Variable-size format:
/// - `num_levels`: u32 (4 bytes)
/// - For each level (`0..num_levels)`:
///   - `neighbor_count`: u32 (4 bytes)
///   - neighbors: [u32; count] (count * 4 bytes)
struct NodeEntry {
    num_levels: u32,
    neighbors_per_level: Vec<Vec<NodeId>>,
}

impl DiskStorage {
    /// Open existing disk storage
    ///
    /// # Arguments
    /// * `path` - Path to storage directory
    /// * `populate` - If true, pre-fault all pages into memory (slower open, faster queries)
    ///
    /// # Errors
    /// Returns error if files don't exist, are corrupted, or mmap fails
    pub fn open(path: &Path, populate: bool) -> Result<Self> {
        // Delegate to open_with_offsets (which auto-loads offset index if available)
        Self::open_with_offsets(path, populate)
    }

    /// Open with pre-computed offset index (O(1) access)
    ///
    /// This method loads the offset index file if it exists, enabling O(1) node access.
    /// If the offset index doesn't exist, falls back to O(n) scanning.
    ///
    /// # Arguments
    /// * `path` - Path to storage directory
    /// * `populate` - If true, pre-fault graph pages into memory
    ///
    /// # Errors
    /// Returns error if files don't exist, are corrupted, or mmap fails
    pub fn open_with_offsets(path: &Path, populate: bool) -> Result<Self> {
        if !path.exists() {
            return Err(HNSWError::Storage(format!(
                "Storage directory does not exist: {}",
                path.display()
            )));
        }

        // Load metadata
        let metadata = Self::load_metadata(path)?;

        // Open graph mmap
        let graph_path = path.join("layer_0.graph");
        let graph_mmap = Self::open_mmap(&graph_path, populate)?;

        // Load offset index if it exists
        let offset_path = path.join("layer_0.offsets");
        let offset_mmap = if offset_path.exists() {
            Some(Arc::new(Self::open_mmap(&offset_path, true)?))
        } else {
            None // Fall back to scanning
        };

        Ok(Self {
            path: path.to_path_buf(),
            graph_mmap: Arc::new(graph_mmap),
            offset_mmap,
            metadata,
            populate,
        })
    }

    /// Create new disk storage from in-memory data
    ///
    /// This is called during `save()` to write in-memory graph to disk.
    ///
    /// # Arguments
    /// * `path` - Path to storage directory
    /// * `nodes` - Node data to serialize (Vec of neighbor lists per level)
    /// * `max_level` - Maximum level in graph
    /// * `m` - M parameter
    ///
    /// # Errors
    /// Returns error if directory creation or file writing fails
    pub fn create(path: &Path, nodes: &[Vec<Vec<NodeId>>], max_level: u32, m: u32) -> Result<()> {
        // Create directory if it doesn't exist
        std::fs::create_dir_all(path)?;

        // Build metadata
        let metadata = GraphMetadata {
            version: FORMAT_VERSION,
            storage_mode: 1, // Hybrid (will be set properly later)
            num_nodes: nodes.len() as u64,
            max_level,
            entry_node_id: 0, // Will be set by caller
            entry_level: 0,   // Will be set by caller
            m,
            m0: m * 2,           // Base layer has 2x neighbors
            ef_construction: 64, // Default
        };

        // Save metadata
        Self::save_metadata(path, &metadata)?;

        // Serialize graph to file
        Self::save_graph(path, nodes)?;

        Ok(())
    }

    /// Load metadata from metadata.bin
    fn load_metadata(path: &Path) -> Result<GraphMetadata> {
        let metadata_path = path.join("metadata.bin");
        let mut file = File::open(&metadata_path)
            .map_err(|e| HNSWError::Storage(format!("Failed to open metadata.bin: {e}")))?;

        let mut buffer = vec![0u8; METADATA_HEADER_SIZE];
        file.read_exact(&mut buffer)
            .map_err(|e| HNSWError::Storage(format!("Failed to read metadata: {e}")))?;

        // Validate magic number
        if &buffer[0..8] != MAGIC_NUMBER {
            return Err(HNSWError::Storage(
                "Invalid magic number in metadata.bin".to_string(),
            ));
        }

        // Parse metadata
        let version = u32::from_le_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]);
        let storage_mode = u32::from_le_bytes([buffer[12], buffer[13], buffer[14], buffer[15]]);
        let num_nodes = u64::from_le_bytes([
            buffer[16], buffer[17], buffer[18], buffer[19], buffer[20], buffer[21], buffer[22],
            buffer[23],
        ]);
        let max_level = u32::from_le_bytes([buffer[24], buffer[25], buffer[26], buffer[27]]);
        let entry_node_id = u32::from_le_bytes([buffer[28], buffer[29], buffer[30], buffer[31]]);
        let entry_level = u32::from_le_bytes([buffer[32], buffer[33], buffer[34], buffer[35]]);
        let m = u32::from_le_bytes([buffer[36], buffer[37], buffer[38], buffer[39]]);
        let m0 = u32::from_le_bytes([buffer[40], buffer[41], buffer[42], buffer[43]]);
        let ef_construction = u32::from_le_bytes([buffer[44], buffer[45], buffer[46], buffer[47]]);

        Ok(GraphMetadata {
            version,
            storage_mode,
            num_nodes,
            max_level,
            entry_node_id,
            entry_level,
            m,
            m0,
            ef_construction,
        })
    }

    /// Save metadata to metadata.bin
    fn save_metadata(path: &Path, metadata: &GraphMetadata) -> Result<()> {
        let metadata_path = path.join("metadata.bin");
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&metadata_path)
            .map_err(|e| HNSWError::Storage(format!("Failed to create metadata.bin: {e}")))?;

        let mut buffer = vec![0u8; METADATA_HEADER_SIZE];

        // Magic number
        buffer[0..8].copy_from_slice(MAGIC_NUMBER);

        // Version
        buffer[8..12].copy_from_slice(&metadata.version.to_le_bytes());

        // Storage mode
        buffer[12..16].copy_from_slice(&metadata.storage_mode.to_le_bytes());

        // Number of nodes
        buffer[16..24].copy_from_slice(&metadata.num_nodes.to_le_bytes());

        // Max level
        buffer[24..28].copy_from_slice(&metadata.max_level.to_le_bytes());

        // Entry point
        buffer[28..32].copy_from_slice(&metadata.entry_node_id.to_le_bytes());
        buffer[32..36].copy_from_slice(&metadata.entry_level.to_le_bytes());

        // Parameters
        buffer[36..40].copy_from_slice(&metadata.m.to_le_bytes());
        buffer[40..44].copy_from_slice(&metadata.m0.to_le_bytes());
        buffer[44..48].copy_from_slice(&metadata.ef_construction.to_le_bytes());

        // Reserved bytes (48..64) remain zero

        file.write_all(&buffer)
            .map_err(|e| HNSWError::Storage(format!("Failed to write metadata: {e}")))?;

        Ok(())
    }

    /// Save graph to `layer_0.graph` using vbyte encoding (`FORMAT_VERSION` 2)
    ///
    /// Format per node:
    /// - `num_levels`: u32 (4 bytes)
    /// - For each level:
    ///   - `neighbor_count`: u32 (4 bytes)
    ///   - neighbors: vbyte encoded absolute IDs (preserves original order)
    fn save_graph(path: &Path, nodes: &[Vec<Vec<NodeId>>]) -> Result<()> {
        let graph_path = path.join("layer_0.graph");
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&graph_path)
            .map_err(|e| HNSWError::Storage(format!("Failed to create layer_0.graph: {e}")))?;

        // Use BufWriter to batch writes (100-1000x speedup vs unbuffered I/O)
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file); // 8MB buffer

        // Serialize each node
        for node_neighbors in nodes {
            // Write num_levels
            let num_levels = node_neighbors.len() as u32;
            writer.write_all(&num_levels.to_le_bytes())?;

            // Write neighbors for each level
            for neighbors in node_neighbors {
                // Write neighbor count
                let count = neighbors.len() as u32;
                writer.write_all(&count.to_le_bytes())?;

                // Write neighbors as vbyte-encoded absolute IDs (preserves distance order)
                write_neighbors_vbyte(&mut writer, neighbors)?;
            }
        }

        // Flush buffer to ensure all data is written
        writer.flush()?;

        Ok(())
    }

    /// Open memory-mapped file
    ///
    /// Applies `madvise(MADV_RANDOM)` for random access patterns
    fn open_mmap(path: &Path, populate: bool) -> Result<Mmap> {
        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| HNSWError::Storage(format!("Failed to open {}: {}", path.display(), e)))?;

        let mut mmap_options = memmap2::MmapOptions::new();

        // Conditionally pre-fault pages if requested
        if populate {
            mmap_options.populate();
        }

        let mmap = unsafe {
            mmap_options.map(&file).map_err(|e| {
                HNSWError::Storage(format!("Failed to mmap {}: {}", path.display(), e))
            })?
        };

        // madvise(MADV_RANDOM) could improve random access patterns but requires nix crate.
        // The OS handles this reasonably well by default for mmap workloads.

        Ok(mmap)
    }

    /// Get byte offset for a node (O(1) with index, O(n) without)
    ///
    /// Uses the memory-mapped offset index if available, otherwise falls back to scanning.
    ///
    /// # Arguments
    /// * `node_id` - Node ID to look up
    ///
    /// # Errors
    /// Returns error if `node_id` is out of bounds
    fn get_offset(&self, node_id: NodeId) -> Result<usize> {
        if let Some(offset_mmap) = &self.offset_mmap {
            // O(1) lookup from mmap
            let offset_bytes = node_id as usize * 8;
            if offset_bytes + 8 > offset_mmap.len() {
                return Err(HNSWError::NodeNotFound(node_id));
            }

            let offset = u64::from_le_bytes([
                offset_mmap[offset_bytes],
                offset_mmap[offset_bytes + 1],
                offset_mmap[offset_bytes + 2],
                offset_mmap[offset_bytes + 3],
                offset_mmap[offset_bytes + 4],
                offset_mmap[offset_bytes + 5],
                offset_mmap[offset_bytes + 6],
                offset_mmap[offset_bytes + 7],
            ]);

            Ok(offset as usize)
        } else {
            // Fall back to O(n) scanning
            self.calculate_offset(node_id)
        }
    }

    /// Calculate byte offset for a node in the graph file
    ///
    /// This requires scanning from the beginning since node entries are variable-size.
    /// Used as fallback when offset index is not available.
    fn calculate_offset(&self, target_node_id: NodeId) -> Result<usize> {
        let mut offset = 0;
        let mut current_node = 0u32;
        let is_v2 = self.metadata.version >= 2;

        while current_node < target_node_id {
            if offset + 4 > self.graph_mmap.len() {
                return Err(HNSWError::NodeNotFound(target_node_id));
            }

            // Read num_levels
            let num_levels = u32::from_le_bytes([
                self.graph_mmap[offset],
                self.graph_mmap[offset + 1],
                self.graph_mmap[offset + 2],
                self.graph_mmap[offset + 3],
            ]);
            offset += 4;

            // Skip all levels
            for _ in 0..num_levels {
                if offset + 4 > self.graph_mmap.len() {
                    return Err(HNSWError::NodeNotFound(target_node_id));
                }

                // Read neighbor count
                let count = u32::from_le_bytes([
                    self.graph_mmap[offset],
                    self.graph_mmap[offset + 1],
                    self.graph_mmap[offset + 2],
                    self.graph_mmap[offset + 3],
                ]) as usize;
                offset += 4;

                if is_v2 {
                    // v2 format: all neighbors are vbyte encoded
                    for _ in 0..count {
                        // Skip vbyte: read until MSB=0
                        while offset < self.graph_mmap.len() && self.graph_mmap[offset] & 0x80 != 0
                        {
                            offset += 1;
                        }
                        offset += 1; // Skip the final byte (MSB=0)
                    }
                } else {
                    // v1 format: all neighbors are 4 bytes each
                    offset += count * 4;
                }
            }

            current_node += 1;
        }

        Ok(offset)
    }

    /// Parse node entry at given offset (handles both v1 and v2 formats)
    fn parse_node_at_offset(&self, offset: usize) -> Result<NodeEntry> {
        if self.metadata.version >= 2 {
            self.parse_node_v2(offset)
        } else {
            self.parse_node_v1(offset)
        }
    }

    /// Parse v1 format (raw u32 neighbor IDs)
    fn parse_node_v1(&self, offset: usize) -> Result<NodeEntry> {
        if offset + 4 > self.graph_mmap.len() {
            return Err(HNSWError::Storage("Offset out of bounds".to_string()));
        }

        // Read num_levels
        let num_levels = u32::from_le_bytes([
            self.graph_mmap[offset],
            self.graph_mmap[offset + 1],
            self.graph_mmap[offset + 2],
            self.graph_mmap[offset + 3],
        ]);
        let mut current_offset = offset + 4;

        let mut neighbors_per_level = Vec::with_capacity(num_levels as usize);

        // Read each level's neighbors
        for _ in 0..num_levels {
            if current_offset + 4 > self.graph_mmap.len() {
                return Err(HNSWError::Storage("Unexpected end of file".to_string()));
            }

            // Read neighbor count
            let count = u32::from_le_bytes([
                self.graph_mmap[current_offset],
                self.graph_mmap[current_offset + 1],
                self.graph_mmap[current_offset + 2],
                self.graph_mmap[current_offset + 3],
            ]) as usize;
            current_offset += 4;

            // Read neighbors
            let mut neighbors = Vec::with_capacity(count);
            for _ in 0..count {
                if current_offset + 4 > self.graph_mmap.len() {
                    return Err(HNSWError::Storage("Unexpected end of file".to_string()));
                }

                let neighbor_id = u32::from_le_bytes([
                    self.graph_mmap[current_offset],
                    self.graph_mmap[current_offset + 1],
                    self.graph_mmap[current_offset + 2],
                    self.graph_mmap[current_offset + 3],
                ]);
                neighbors.push(neighbor_id);
                current_offset += 4;
            }

            neighbors_per_level.push(neighbors);
        }

        Ok(NodeEntry {
            num_levels,
            neighbors_per_level,
        })
    }

    /// Parse v2 format (vbyte-encoded absolute IDs)
    fn parse_node_v2(&self, offset: usize) -> Result<NodeEntry> {
        if offset + 4 > self.graph_mmap.len() {
            return Err(HNSWError::Storage("Offset out of bounds".to_string()));
        }

        // Read num_levels
        let num_levels = u32::from_le_bytes([
            self.graph_mmap[offset],
            self.graph_mmap[offset + 1],
            self.graph_mmap[offset + 2],
            self.graph_mmap[offset + 3],
        ]);
        let mut current_offset = offset + 4;

        let mut neighbors_per_level = Vec::with_capacity(num_levels as usize);

        // Read each level's neighbors
        for _ in 0..num_levels {
            if current_offset + 4 > self.graph_mmap.len() {
                return Err(HNSWError::Storage("Unexpected end of file".to_string()));
            }

            // Read neighbor count
            let count = u32::from_le_bytes([
                self.graph_mmap[current_offset],
                self.graph_mmap[current_offset + 1],
                self.graph_mmap[current_offset + 2],
                self.graph_mmap[current_offset + 3],
            ]) as usize;
            current_offset += 4;

            // Read all neighbors as vbyte-encoded absolute IDs
            let mut neighbors = Vec::with_capacity(count);
            for _ in 0..count {
                if current_offset >= self.graph_mmap.len() {
                    return Err(HNSWError::Storage("Unexpected end of file".to_string()));
                }
                let (neighbor_id, bytes_read) = vbyte_decode(&self.graph_mmap[current_offset..]);
                if bytes_read == 0 {
                    return Err(HNSWError::Storage("Invalid vbyte encoding".to_string()));
                }
                neighbors.push(neighbor_id);
                current_offset += bytes_read;
            }

            neighbors_per_level.push(neighbors);
        }

        Ok(NodeEntry {
            num_levels,
            neighbors_per_level,
        })
    }
}

impl NodeStorage for DiskStorage {
    fn read_neighbors(&self, node_id: NodeId, level: Level) -> Result<Vec<NodeId>> {
        if (node_id as u64) >= self.metadata.num_nodes {
            return Ok(Vec::new()); // Node doesn't exist
        }

        // Get offset for this node (O(1) with index, O(n) without)
        let offset = self.get_offset(node_id)?;

        // Parse node entry
        let entry = self.parse_node_at_offset(offset)?;

        // Return neighbors at requested level
        let level_idx = level as usize;
        if level_idx < entry.neighbors_per_level.len() {
            Ok(entry.neighbors_per_level[level_idx].clone())
        } else {
            Ok(Vec::new()) // Level doesn't exist for this node
        }
    }

    fn write_neighbors(
        &mut self,
        _node_id: NodeId,
        _level: Level,
        _neighbors: &[NodeId],
    ) -> Result<()> {
        // DiskStorage is read-only after creation
        // Writes happen during save() (build in memory, serialize to disk)
        Err(HNSWError::Storage(
            "DiskStorage is read-only. Use MemoryStorage for writes, then call save()".to_string(),
        ))
    }

    fn exists(&self, node_id: NodeId) -> bool {
        (node_id as u64) < self.metadata.num_nodes
    }

    fn num_levels(&self, node_id: NodeId) -> Result<usize> {
        if (node_id as u64) >= self.metadata.num_nodes {
            return Err(HNSWError::NodeNotFound(node_id));
        }

        let offset = self.get_offset(node_id)?;
        let entry = self.parse_node_at_offset(offset)?;
        Ok(entry.num_levels as usize)
    }

    fn len(&self) -> usize {
        self.metadata.num_nodes as usize
    }

    fn memory_usage(&self) -> usize {
        // DiskStorage uses mmap, which doesn't count as RSS
        // Only count metadata and Arc overhead
        std::mem::size_of::<Self>() + std::mem::size_of::<GraphMetadata>()
    }

    fn flush(&mut self) -> Result<()> {
        // mmap is read-only, no flush needed
        Ok(())
    }
}

/// Writable disk storage for incremental graph building
///
/// Supports append-only writes during HNSW construction, then converts
/// to read-only `DiskStorage` via `finalize()`.
///
/// # Usage
/// ```ignore
/// // Create writable storage
/// let mut storage = WritableDiskStorage::create(path, max_level, m)?;
///
/// // Write nodes incrementally
/// for node_id in 0..num_nodes {
///     let neighbors = compute_neighbors(node_id);
///     storage.write_node(node_id, &neighbors)?;
/// }
///
/// // Finalize and convert to read-only
/// let disk_storage = storage.finalize()?;
/// ```
pub struct WritableDiskStorage {
    /// Path to storage directory
    path: PathBuf,

    /// Buffered writer for graph file (append-only)
    file: BufWriter<File>,

    /// Offset index: `node_id` → byte offset in file
    offset_index: Vec<u64>,

    /// Current write position in file
    current_offset: u64,

    /// Graph metadata
    metadata: GraphMetadata,
}

impl WritableDiskStorage {
    /// Create new writable storage
    ///
    /// # Arguments
    /// * `path` - Path to storage directory
    /// * `max_level` - Maximum level in graph
    /// * `m` - M parameter (neighbors per level)
    ///
    /// # Errors
    /// Returns error if directory creation or file creation fails
    pub fn create(path: &Path, max_level: u32, m: u32) -> Result<Self> {
        // Create directory if it doesn't exist
        std::fs::create_dir_all(path)?;

        // Open graph file for writing
        let graph_path = path.join("layer_0.graph");
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&graph_path)
            .map_err(|e| HNSWError::Storage(format!("Failed to create layer_0.graph: {e}")))?;

        let file = BufWriter::with_capacity(8 * 1024 * 1024, file); // 8MB buffer

        // Initialize metadata
        let metadata = GraphMetadata {
            version: FORMAT_VERSION,
            storage_mode: 1, // Hybrid
            num_nodes: 0,    // Will be set on finalize
            max_level,
            entry_node_id: 0,
            entry_level: 0,
            m,
            m0: m * 2,
            ef_construction: 64,
        };

        Ok(Self {
            path: path.to_path_buf(),
            file,
            offset_index: Vec::new(),
            current_offset: 0,
            metadata,
        })
    }

    /// Write node with all its neighbors (all levels at once)
    ///
    /// Nodes must be written sequentially (0, 1, 2, ...).
    ///
    /// # Arguments
    /// * `node_id` - Node ID (must equal current node count)
    /// * `neighbors_per_level` - Neighbors for each level (level 0, 1, 2, ...)
    ///
    /// # Errors
    /// Returns error if:
    /// - Node ID is not sequential
    /// - Write fails
    pub fn write_node(
        &mut self,
        node_id: NodeId,
        neighbors_per_level: &[Vec<NodeId>],
    ) -> Result<()> {
        // Validate node_id is sequential
        if node_id as usize != self.offset_index.len() {
            return Err(HNSWError::Storage(format!(
                "Nodes must be written sequentially. Expected {}, got {}",
                self.offset_index.len(),
                node_id
            )));
        }

        // Track offset before writing
        self.offset_index.push(self.current_offset);

        // Write num_levels
        let num_levels = neighbors_per_level.len() as u32;
        self.file
            .write_all(&num_levels.to_le_bytes())
            .map_err(|e| HNSWError::Storage(format!("Write failed: {e}")))?;
        self.current_offset += 4;

        // Write each level's neighbors as vbyte-encoded absolute IDs (preserves order)
        for neighbors in neighbors_per_level {
            // Write neighbor count
            let count = neighbors.len() as u32;
            self.file
                .write_all(&count.to_le_bytes())
                .map_err(|e| HNSWError::Storage(format!("Write failed: {e}")))?;
            self.current_offset += 4;

            // Write neighbors as vbyte and track bytes written
            let bytes_written = write_neighbors_vbyte_counted(&mut self.file, neighbors)
                .map_err(|e| HNSWError::Storage(format!("Write failed: {e}")))?;
            self.current_offset += bytes_written as u64;
        }

        Ok(())
    }

    /// Finalize writes and convert to read-only `DiskStorage`
    ///
    /// This:
    /// 1. Flushes remaining writes
    /// 2. Saves offset index to disk
    /// 3. Updates metadata
    /// 4. Opens as read-only `DiskStorage`
    ///
    /// # Errors
    /// Returns error if flush, offset save, or open fails
    pub fn finalize(mut self) -> Result<DiskStorage> {
        // Flush and close graph file
        self.file
            .flush()
            .map_err(|e| HNSWError::Storage(format!("Flush failed: {e}")))?;
        drop(self.file);

        // Extract fields we need
        let path = self.path.clone();
        let offset_index = std::mem::take(&mut self.offset_index);
        let mut metadata = self.metadata.clone();

        // Save offset index
        let offset_path = path.join("layer_0.offsets");
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&offset_path)
            .map_err(|e| HNSWError::Storage(format!("Failed to create offset index: {e}")))?;

        let mut writer = BufWriter::new(file);

        for &offset in &offset_index {
            writer
                .write_all(&offset.to_le_bytes())
                .map_err(|e| HNSWError::Storage(format!("Failed to write offset: {e}")))?;
        }

        writer
            .flush()
            .map_err(|e| HNSWError::Storage(format!("Failed to flush offset index: {e}")))?;

        // Update and save metadata
        metadata.num_nodes = offset_index.len() as u64;
        DiskStorage::save_metadata(&path, &metadata)?;

        // Open as read-only DiskStorage
        DiskStorage::open_with_offsets(&path, false)
    }

    /// Get number of nodes written so far
    #[must_use]
    pub fn len(&self) -> usize {
        self.offset_index.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.offset_index.is_empty()
    }
}

impl NodeStorage for WritableDiskStorage {
    fn read_neighbors(&self, _node_id: NodeId, _level: Level) -> Result<Vec<NodeId>> {
        // WritableDiskStorage is write-only during building
        // Reads should happen after finalize() -> DiskStorage
        Err(HNSWError::Storage(
            "WritableDiskStorage is write-only. Call finalize() to get read-only DiskStorage"
                .to_string(),
        ))
    }

    fn write_neighbors(
        &mut self,
        _node_id: NodeId,
        _level: Level,
        _neighbors: &[NodeId],
    ) -> Result<()> {
        // WritableDiskStorage uses write_node() for atomic multi-level writes
        // Individual level writes not supported
        Err(HNSWError::Storage(
            "WritableDiskStorage requires write_node() for atomic writes".to_string(),
        ))
    }

    fn write_node(&mut self, node_id: NodeId, neighbors_per_level: &[Vec<NodeId>]) -> Result<()> {
        // Use the optimized write_node implementation
        self.write_node(node_id, neighbors_per_level)
    }

    fn exists(&self, node_id: NodeId) -> bool {
        (node_id as usize) < self.offset_index.len()
    }

    fn num_levels(&self, _node_id: NodeId) -> Result<usize> {
        // Can't determine num_levels without reading from disk
        // This is a write-only structure during building
        Err(HNSWError::Storage(
            "WritableDiskStorage is write-only. Call finalize() for reads".to_string(),
        ))
    }

    fn len(&self) -> usize {
        self.offset_index.len()
    }

    fn memory_usage(&self) -> usize {
        // Offset index + metadata
        self.offset_index.len() * std::mem::size_of::<u64>() + std::mem::size_of::<Self>()
    }

    fn flush(&mut self) -> Result<()> {
        self.file
            .flush()
            .map_err(|e| HNSWError::Storage(format!("Flush failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn create_test_data() -> Vec<Vec<Vec<NodeId>>> {
        vec![
            // Node 0: 2 levels
            vec![
                vec![1, 2], // Level 0
                vec![3],    // Level 1
            ],
            // Node 1: 1 level
            vec![
                vec![0, 2, 3], // Level 0
            ],
            // Node 2: 2 levels
            vec![
                vec![0, 1], // Level 0
                vec![3],    // Level 1
            ],
            // Node 3: 3 levels
            vec![
                vec![1, 2], // Level 0
                vec![0, 2], // Level 1
                vec![],     // Level 2 (empty)
            ],
        ]
    }

    #[test]
    fn test_disk_storage_create_and_open() {
        let path = PathBuf::from("/tmp/omendb_disk_storage_test");
        let _ = fs::remove_dir_all(&path); // Clean up

        // Create storage
        let nodes = create_test_data();
        DiskStorage::create(&path, &nodes, 2, 16).unwrap();

        // Verify files exist
        assert!(path.join("metadata.bin").exists());
        assert!(path.join("layer_0.graph").exists());

        // Open storage
        let storage = DiskStorage::open(&path, false).unwrap();
        assert_eq!(storage.len(), 4);

        // Clean up
        fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn test_disk_storage_read_neighbors() {
        let path = PathBuf::from("/tmp/omendb_disk_storage_read_test");
        let _ = fs::remove_dir_all(&path);

        // Create and open storage
        let nodes = create_test_data();
        DiskStorage::create(&path, &nodes, 2, 16).unwrap();
        let storage = DiskStorage::open(&path, false).unwrap();

        // Read neighbors for node 0, level 0
        let neighbors = storage.read_neighbors(0, 0).unwrap();
        assert_eq!(neighbors, vec![1, 2]);

        // Read neighbors for node 0, level 1
        let neighbors = storage.read_neighbors(0, 1).unwrap();
        assert_eq!(neighbors, vec![3]);

        // Read neighbors for node 3, level 2 (empty)
        let neighbors = storage.read_neighbors(3, 2).unwrap();
        assert!(neighbors.is_empty());

        fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn test_disk_storage_exists() {
        let path = PathBuf::from("/tmp/omendb_disk_storage_exists_test");
        let _ = fs::remove_dir_all(&path);

        let nodes = create_test_data();
        DiskStorage::create(&path, &nodes, 2, 16).unwrap();
        let storage = DiskStorage::open(&path, false).unwrap();

        assert!(storage.exists(0));
        assert!(storage.exists(1));
        assert!(storage.exists(2));
        assert!(storage.exists(3));
        assert!(!storage.exists(4)); // Doesn't exist

        fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn test_disk_storage_num_levels() {
        let path = PathBuf::from("/tmp/omendb_disk_storage_levels_test");
        let _ = fs::remove_dir_all(&path);

        let nodes = create_test_data();
        DiskStorage::create(&path, &nodes, 2, 16).unwrap();
        let storage = DiskStorage::open(&path, false).unwrap();

        assert_eq!(storage.num_levels(0).unwrap(), 2); // 2 levels
        assert_eq!(storage.num_levels(1).unwrap(), 1); // 1 level
        assert_eq!(storage.num_levels(3).unwrap(), 3); // 3 levels

        fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn test_disk_storage_read_only() {
        let path = PathBuf::from("/tmp/omendb_disk_storage_readonly_test");
        let _ = fs::remove_dir_all(&path);

        let nodes = create_test_data();
        DiskStorage::create(&path, &nodes, 2, 16).unwrap();
        let mut storage = DiskStorage::open(&path, false).unwrap();

        // Write should fail (read-only)
        let result = storage.write_neighbors(0, 0, &[99]);
        assert!(result.is_err());

        fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn test_disk_storage_nonexistent_path() {
        let path = PathBuf::from("/tmp/omendb_nonexistent_12345");
        let result = DiskStorage::open(&path, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_disk_storage_metadata_validation() {
        let path = PathBuf::from("/tmp/omendb_disk_storage_metadata_test");
        let _ = fs::remove_dir_all(&path);

        let nodes = create_test_data();
        DiskStorage::create(&path, &nodes, 2, 16).unwrap();

        // Corrupt metadata
        let metadata_path = path.join("metadata.bin");
        let mut file = OpenOptions::new().write(true).open(&metadata_path).unwrap();
        file.write_all(b"BADMAGIC").unwrap();

        // Should fail to open
        let result = DiskStorage::open(&path, false);
        assert!(result.is_err());

        fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn test_disk_storage_populate_parameter() {
        let path = PathBuf::from("/tmp/omendb_disk_storage_populate_test");
        let _ = fs::remove_dir_all(&path);

        let nodes = create_test_data();
        DiskStorage::create(&path, &nodes, 2, 16).unwrap();

        // Open with populate=false
        let storage_no_pop = DiskStorage::open(&path, false).unwrap();
        assert!(!storage_no_pop.populate);

        // Open with populate=true
        let storage_pop = DiskStorage::open(&path, true).unwrap();
        assert!(storage_pop.populate);

        fs::remove_dir_all(&path).ok();
    }

    // WritableDiskStorage tests

    #[test]
    fn test_writable_disk_storage_basic() {
        let path = PathBuf::from("/tmp/omendb_writable_basic_test");
        let _ = fs::remove_dir_all(&path);

        // Create writable storage
        let mut storage = WritableDiskStorage::create(&path, 2, 16).unwrap();
        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());

        // Write 4 nodes
        let nodes = create_test_data();
        for (node_id, neighbors_per_level) in nodes.iter().enumerate() {
            storage
                .write_node(node_id as NodeId, neighbors_per_level)
                .unwrap();
        }
        assert_eq!(storage.len(), 4);
        assert!(!storage.is_empty());

        // Finalize and convert to read-only
        let disk_storage = storage.finalize().unwrap();

        // Verify we can read the data
        assert_eq!(disk_storage.len(), 4);
        let neighbors = disk_storage.read_neighbors(0, 0).unwrap();
        assert_eq!(neighbors, vec![1, 2]);

        // Verify offset index file exists
        assert!(path.join("layer_0.offsets").exists());

        fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn test_writable_disk_storage_sequential_requirement() {
        let path = PathBuf::from("/tmp/omendb_writable_sequential_test");
        let _ = fs::remove_dir_all(&path);

        let mut storage = WritableDiskStorage::create(&path, 2, 16).unwrap();

        // Write node 0 (OK)
        storage.write_node(0, &[vec![1, 2]]).unwrap();

        // Try to write node 2 (should fail - expected 1)
        let result = storage.write_node(2, &[vec![0, 3]]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Nodes must be written sequentially"));

        fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn test_writable_disk_storage_read_after_write() {
        let path = PathBuf::from("/tmp/omendb_writable_read_after_write_test");
        let _ = fs::remove_dir_all(&path);

        // Write data
        let mut storage = WritableDiskStorage::create(&path, 2, 16).unwrap();
        let nodes = create_test_data();
        for (node_id, neighbors_per_level) in nodes.iter().enumerate() {
            storage
                .write_node(node_id as NodeId, neighbors_per_level)
                .unwrap();
        }

        // Finalize
        let disk_storage = storage.finalize().unwrap();

        // Verify all data is readable and correct
        for (node_id, expected_neighbors) in nodes.iter().enumerate() {
            for (level, expected) in expected_neighbors.iter().enumerate() {
                let actual = disk_storage
                    .read_neighbors(node_id as NodeId, level as Level)
                    .unwrap();
                assert_eq!(
                    actual, *expected,
                    "Mismatch for node {node_id} level {level}"
                );
            }
        }

        fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn test_writable_disk_storage_offset_index_persistence() {
        let path = PathBuf::from("/tmp/omendb_writable_offset_index_test");
        let _ = fs::remove_dir_all(&path);

        // Write and finalize
        let mut storage = WritableDiskStorage::create(&path, 2, 16).unwrap();
        let nodes = create_test_data();
        for (node_id, neighbors_per_level) in nodes.iter().enumerate() {
            storage
                .write_node(node_id as NodeId, neighbors_per_level)
                .unwrap();
        }
        let _disk_storage = storage.finalize().unwrap();

        // Close and reopen
        let reopened = DiskStorage::open(&path, false).unwrap();

        // Verify offset_mmap is loaded
        assert!(reopened.offset_mmap.is_some());

        // Verify we can still read correctly (using O(1) access)
        let neighbors = reopened.read_neighbors(3, 1).unwrap();
        assert_eq!(neighbors, vec![0, 2]);

        fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn test_writable_disk_storage_empty() {
        let path = PathBuf::from("/tmp/omendb_writable_empty_test");
        let _ = fs::remove_dir_all(&path);

        // Create and immediately finalize (no nodes)
        let storage = WritableDiskStorage::create(&path, 2, 16).unwrap();
        let disk_storage = storage.finalize().unwrap();

        assert_eq!(disk_storage.len(), 0);

        fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn test_writable_disk_storage_large_neighbors() {
        let path = PathBuf::from("/tmp/omendb_writable_large_test");
        let _ = fs::remove_dir_all(&path);

        let mut storage = WritableDiskStorage::create(&path, 2, 16).unwrap();

        // Write node with many neighbors
        let large_neighbors = vec![
            (0..100).collect::<Vec<NodeId>>(), // Level 0: 100 neighbors
            (0..50).collect::<Vec<NodeId>>(),  // Level 1: 50 neighbors
        ];
        storage.write_node(0, &large_neighbors).unwrap();

        let disk_storage = storage.finalize().unwrap();

        // Verify large neighbor lists are correct
        let level0 = disk_storage.read_neighbors(0, 0).unwrap();
        assert_eq!(level0.len(), 100);
        assert_eq!(level0[0], 0);
        assert_eq!(level0[99], 99);

        let level1 = disk_storage.read_neighbors(0, 1).unwrap();
        assert_eq!(level1.len(), 50);

        fs::remove_dir_all(&path).ok();
    }
}
