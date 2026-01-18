//! HNSW index persistence (save/load)
//!
//! Format versions:
//! - v2: Original postcard format with NeighborLists (deprecated, read-only)
//! - v3: Format with GraphStorage (deprecated, read-only)
//! - v4: NodeStorage format - unified colocated storage

use super::HNSWIndex;
use crate::vector::hnsw::error::{HNSWError, Result};
use crate::vector::hnsw::node_storage::NodeStorage;
use crate::vector::hnsw::types::{DistanceFunction, HNSWParams};
use std::fs::OpenOptions;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use tracing::{error, info, instrument, warn};

/// Current persistence format version
const FORMAT_VERSION: u32 = 4;

/// Configure OpenOptions for cross-platform compatibility.
/// On Windows, enables full file sharing to avoid "Access is denied" errors.
#[cfg(windows)]
fn configure_open_options(opts: &mut OpenOptions) {
    use std::os::windows::fs::OpenOptionsExt;
    // FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
    opts.share_mode(0x1 | 0x2 | 0x4);
}

#[cfg(not(windows))]
fn configure_open_options(_opts: &mut OpenOptions) {
    // No-op on Unix
}

impl HNSWIndex {
    /// Save index to disk (v4 format)
    ///
    /// Format:
    /// - Magic: b"HNSWIDX\0" (8 bytes)
    /// - Version: u32 (4 bytes)
    /// - Entry point: Option<u32> (1 + 4 bytes)
    /// - Distance function: `DistanceFunction` (length-prefixed postcard)
    /// - Params: `HNSWParams` (length-prefixed postcard)
    /// - RNG state: u64 (8 bytes)
    /// - Storage: `NodeStorage` (length-prefixed blob)
    #[instrument(skip(self, path), fields(index_size = self.len(), dimensions = self.dimensions()))]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        info!("Starting index save (v4 format)");
        let start = std::time::Instant::now();

        let mut opts = OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        configure_open_options(&mut opts);
        let file = opts.open(path).map_err(|e| {
            error!(error = ?e, "Failed to create index file");
            HNSWError::from(e)
        })?;
        let mut writer = BufWriter::new(file);

        // Write magic bytes
        writer.write_all(b"HNSWIDX\0")?;

        // Write version (4 = NodeStorage format)
        writer.write_all(&FORMAT_VERSION.to_le_bytes())?;

        // Write entry point
        match self.entry_point {
            Some(ep) => {
                writer.write_all(&[1u8])?;
                writer.write_all(&ep.to_le_bytes())?;
            }
            None => {
                writer.write_all(&[0u8])?;
            }
        }

        // Write distance function (length-prefixed postcard)
        let df_bytes = postcard::to_allocvec(&self.distance_fn)?;
        writer.write_all(&(df_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&df_bytes)?;

        // Write params (length-prefixed postcard)
        let params_bytes = postcard::to_allocvec(&self.params)?;
        writer.write_all(&(params_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&params_bytes)?;

        // Write RNG state
        writer.write_all(&self.rng_state.to_le_bytes())?;

        // Write storage (complete NodeStorage serialization including SQ8 state)
        let storage_bytes = self.storage.serialize_full();
        writer.write_all(&(storage_bytes.len() as u64).to_le_bytes())?;
        writer.write_all(&storage_bytes)?;

        let elapsed = start.elapsed();
        info!(
            duration_ms = elapsed.as_millis(),
            memory_bytes = self.memory_usage(),
            "Index save completed successfully"
        );

        Ok(())
    }

    /// Load index from disk
    #[instrument(skip(path))]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        info!("Starting index load");
        let start = std::time::Instant::now();
        let mut opts = OpenOptions::new();
        opts.read(true);
        configure_open_options(&mut opts);
        let file = opts.open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify magic bytes
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != b"HNSWIDX\0" {
            error!(magic = ?magic, "Invalid magic bytes in index file");
            return Err(HNSWError::Storage(format!(
                "Invalid magic bytes: {magic:?}"
            )));
        }

        // Read version
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);

        match version {
            4 => Self::load_v4(&mut reader, start),
            2 | 3 => {
                warn!(
                    version,
                    "Loading legacy format (v2/v3) - will be upgraded to v4 on next save"
                );
                Self::load_legacy(&mut reader, version, start)
            }
            _ => {
                error!(version, "Unsupported index file version");
                Err(HNSWError::Storage(format!(
                    "Unsupported version: {version} (expected 2, 3, or 4)"
                )))
            }
        }
    }

    /// Load v4 format (NodeStorage)
    fn load_v4<R: Read>(reader: &mut BufReader<R>, start: std::time::Instant) -> Result<Self> {
        // Read entry point
        let mut entry_point_flag = [0u8; 1];
        reader.read_exact(&mut entry_point_flag)?;
        let entry_point = if entry_point_flag[0] == 1 {
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
        let mut df_bytes = vec![0u8; df_len];
        reader.read_exact(&mut df_bytes)?;
        let distance_fn: DistanceFunction = postcard::from_bytes(&df_bytes)?;

        // Read params (length-prefixed postcard)
        reader.read_exact(&mut len_bytes)?;
        let params_len = u32::from_le_bytes(len_bytes) as usize;
        let mut params_bytes = vec![0u8; params_len];
        reader.read_exact(&mut params_bytes)?;
        let params: HNSWParams = postcard::from_bytes(&params_bytes)?;

        // Read RNG state
        let mut rng_state_bytes = [0u8; 8];
        reader.read_exact(&mut rng_state_bytes)?;
        let rng_state = u64::from_le_bytes(rng_state_bytes);

        // Read storage (length-prefixed blob)
        let mut storage_len_bytes = [0u8; 8];
        reader.read_exact(&mut storage_len_bytes)?;
        let storage_len = u64::from_le_bytes(storage_len_bytes) as usize;
        let mut storage_bytes = vec![0u8; storage_len];
        reader.read_exact(&mut storage_bytes)?;

        let storage = NodeStorage::deserialize_full(&storage_bytes)
            .map_err(|e| HNSWError::Storage(format!("Failed to deserialize NodeStorage: {e}")))?;

        let elapsed = start.elapsed();
        let index = Self {
            storage,
            entry_point,
            params,
            distance_fn,
            rng_state,
        };

        info!(
            duration_ms = elapsed.as_millis(),
            index_size = index.len(),
            dimensions = index.dimensions(),
            memory_bytes = index.memory_usage(),
            "Index load completed successfully (v4 format)"
        );

        Ok(index)
    }

    /// Load legacy v2/v3 format and convert to NodeStorage
    fn load_legacy<R: Read>(
        reader: &mut BufReader<R>,
        version: u32,
        start: std::time::Instant,
    ) -> Result<Self> {
        use crate::vector::hnsw::graph_storage::GraphStorage;
        use crate::vector::hnsw::storage::{NeighborLists, VectorStorage};
        use crate::vector::hnsw::types::HNSWNode;

        // Read dimensions
        let mut dimensions_bytes = [0u8; 4];
        reader.read_exact(&mut dimensions_bytes)?;
        let dimensions = u32::from_le_bytes(dimensions_bytes) as usize;

        // Read num nodes
        let mut num_nodes_bytes = [0u8; 4];
        reader.read_exact(&mut num_nodes_bytes)?;
        let num_nodes = u32::from_le_bytes(num_nodes_bytes) as usize;

        // Read entry point
        let mut entry_point_flag = [0u8; 1];
        reader.read_exact(&mut entry_point_flag)?;
        let entry_point = if entry_point_flag[0] == 1 {
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
        let mut df_bytes = vec![0u8; df_len];
        reader.read_exact(&mut df_bytes)?;
        let distance_fn: DistanceFunction = postcard::from_bytes(&df_bytes)?;

        // Read params (length-prefixed postcard)
        reader.read_exact(&mut len_bytes)?;
        let params_len = u32::from_le_bytes(len_bytes) as usize;
        let mut params_bytes = vec![0u8; params_len];
        reader.read_exact(&mut params_bytes)?;
        let params: HNSWParams = postcard::from_bytes(&params_bytes)?;

        // Read RNG state
        let mut rng_state_bytes = [0u8; 8];
        reader.read_exact(&mut rng_state_bytes)?;
        let rng_state = u64::from_le_bytes(rng_state_bytes);

        // Read nodes (raw bytes for fast I/O)
        let mut nodes = vec![HNSWNode::default(); num_nodes];
        if num_nodes > 0 {
            let nodes_bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    nodes.as_mut_ptr().cast::<u8>(),
                    nodes.len() * std::mem::size_of::<HNSWNode>(),
                )
            };
            reader.read_exact(nodes_bytes)?;
        }

        // Read neighbors (length-prefixed postcard)
        reader.read_exact(&mut len_bytes)?;
        let neighbors_len = u32::from_le_bytes(len_bytes) as usize;
        let mut neighbors_bytes = vec![0u8; neighbors_len];
        reader.read_exact(&mut neighbors_bytes)?;

        // Handle v2 (NeighborLists) and v3 (GraphStorage) formats
        let neighbors: GraphStorage = if version == 2 {
            let neighbor_lists: NeighborLists = postcard::from_bytes(&neighbors_bytes)?;
            GraphStorage::from_neighbor_lists(neighbor_lists)
        } else {
            postcard::from_bytes(&neighbors_bytes)?
        };

        // Read vectors (length-prefixed postcard)
        reader.read_exact(&mut len_bytes)?;
        let vectors_len = u32::from_le_bytes(len_bytes) as usize;
        let mut vectors_bytes = vec![0u8; vectors_len];
        reader.read_exact(&mut vectors_bytes)?;
        let vectors: VectorStorage = postcard::from_bytes(&vectors_bytes)?;

        // Verify dimensions match
        if vectors.dimensions() != dimensions {
            error!(
                expected_dim = dimensions,
                actual_dim = vectors.dimensions(),
                "Dimension mismatch in loaded index"
            );
            return Err(HNSWError::DimensionMismatch {
                expected: dimensions,
                actual: vectors.dimensions(),
            });
        }

        // Convert to NodeStorage
        let is_sq8 = vectors.is_sq8();
        let max_level = params.max_level as usize;
        let mut storage = if is_sq8 {
            NodeStorage::new_sq8(dimensions, params.m, max_level)
        } else {
            NodeStorage::new(dimensions, params.m, max_level)
        };

        // Copy nodes and vectors
        for node in &nodes {
            let node_id = storage.allocate_node();
            assert_eq!(node_id, node.id);

            // Copy vector
            if let Some(vec) = vectors.get_dequantized(node.id) {
                storage.set_vector(node_id, &vec);
            }

            // Set metadata
            storage.set_slot(node_id, node.slot);
            storage.set_level(node_id, node.level);

            // Allocate upper levels if needed
            if node.level > 0 {
                storage.allocate_upper_levels(node_id, node.level);
            }

            // Copy neighbors at each level
            for level in 0..=node.level {
                let neighbors_list = neighbors.get_neighbors(node_id, level);
                storage.set_neighbors_at_level(node_id, level, neighbors_list);
            }
        }

        let elapsed = start.elapsed();
        let index = Self {
            storage,
            entry_point,
            params,
            distance_fn,
            rng_state,
        };

        info!(
            duration_ms = elapsed.as_millis(),
            index_size = index.len(),
            dimensions = index.dimensions(),
            memory_bytes = index.memory_usage(),
            "Legacy index converted to v4 format"
        );

        Ok(index)
    }
}
