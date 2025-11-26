/*
 * IP-DiskANN Index - In-Place Updates with Bi-Directional Graph
 *
 * Based on:
 * - IP-DiskANN paper (arXiv 2502.13826, February 2025)
 * - Microsoft's Rust implementation (~/github/microsoft/DiskANN/rust/diskann/src/index/)
 * - Our Week 1 building blocks (types, graph, prune, search)
 *
 * Key features:
 * - In-place insert (no batch consolidation)
 * - In-place delete (efficient via bi-directional edges)
 * - Persistent storage (save/load)
 */

use super::graph::BiDirectionalGraph;
use super::prune::prune_neighbors;
use super::search::greedy_search;
use super::types::{IPDiskANNConfig, Neighbor, NodeId};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// IP-DiskANN Index with in-place updates
///
/// This is the main index structure that combines:
/// - Bi-directional graph (efficient deletes)
/// - Greedy search (k-NN queries)
/// - RobustPrune (neighbor selection)
#[derive(Debug)]
pub struct IPDiskANNIndex {
    /// Bi-directional graph (out-neighbors + in-neighbors)
    pub(crate) graph: BiDirectionalGraph,

    /// Vector storage (NodeId -> vector)
    pub(crate) vectors: HashMap<NodeId, Vec<f32>>,

    /// Index configuration (R, C, alpha, L)
    config: IPDiskANNConfig,

    /// Next available node ID
    next_id: NodeId,

    /// Vector dimensionality
    dimension: usize,
}

impl IPDiskANNIndex {
    /// Create a new IP-DiskANN index
    ///
    /// # Parameters
    /// - `dimension`: Vector dimensionality
    /// - `config`: Index configuration (optional, uses defaults if None)
    pub fn new(dimension: usize, config: Option<IPDiskANNConfig>) -> Self {
        let config = config.unwrap_or_default();

        Self {
            graph: BiDirectionalGraph::new(),
            vectors: HashMap::new(),
            config,
            next_id: 0,
            dimension,
        }
    }

    /// Insert a vector into the index (in-place, no batch consolidation)
    ///
    /// # Algorithm (from DiskANN paper + IP-DiskANN modifications)
    /// 1. Assign new node ID
    /// 2. Find candidate neighbors via greedy search
    /// 3. Prune to diverse neighbors via RobustPrune
    /// 4. Add bi-directional edges (out + in)
    /// 5. Update reverse neighbors (inter_insert)
    ///
    /// # Returns
    /// NodeId of the inserted vector
    pub fn insert(&mut self, vector: Vec<f32>) -> Result<NodeId, String> {
        // Validate dimension
        if vector.len() != self.dimension {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            ));
        }

        let node_id = self.next_id;
        self.next_id += 1;

        // Add vector to storage
        self.vectors.insert(node_id, vector);

        // Add node to graph
        self.graph.add_node(node_id);

        // If this is the first node, set as entry point and we're done
        if node_id == 0 {
            self.graph.set_entry_node(node_id);
            return Ok(node_id);
        }

        // Otherwise, find neighbors and connect
        let vector_ref = self.vectors.get(&node_id).unwrap();

        // Step 1: Greedy search to find candidate neighbors
        // Search for more candidates than we need, then prune to diverse set
        let mut candidates = greedy_search(
            vector_ref,
            &self.graph,
            &self.vectors,
            self.config.search_list_size, // Search for L candidates (more than R)
            self.config.search_list_size,
        );

        // Step 2: Prune to diverse neighbors
        let selected_neighbors = prune_neighbors(
            node_id,
            &mut candidates,
            self.config.max_degree,
            self.config.max_candidates,
            self.config.alpha,
            &self.vectors,
        );

        // Step 3: Add forward edges (node -> neighbors)
        for &neighbor_id in &selected_neighbors {
            self.graph.add_edge(node_id, neighbor_id);
        }

        // Step 4: Update reverse neighbors (inter_insert)
        // For each neighbor, add reverse edge and re-prune if over-degree
        self.inter_insert(node_id, &selected_neighbors)?;

        Ok(node_id)
    }

    /// Update reverse neighbors after insertion (inter_insert from Microsoft)
    ///
    /// For each neighbor in the pruned list:
    /// 1. Add reverse edge (neighbor -> node_id)
    /// 2. If neighbor is now over-degree, re-prune its neighbors
    fn inter_insert(&mut self, node_id: NodeId, pruned_list: &[NodeId]) -> Result<(), String> {
        for &neighbor_id in pruned_list {
            // Add reverse edge: neighbor -> node_id
            // (The forward edge node_id -> neighbor was already added in insert())
            self.graph.add_edge(neighbor_id, node_id);

            let neighbor_degree = self
                .graph
                .get_node(neighbor_id)
                .map(|n| n.out_neighbors.len())
                .unwrap_or(0);

            // If neighbor exceeds max_degree, re-prune its neighbors
            if neighbor_degree > self.config.max_degree {
                self.reprune_node(neighbor_id)?;
            }
        }

        Ok(())
    }

    /// Re-prune a node's neighbors when it exceeds max_degree
    fn reprune_node(&mut self, node_id: NodeId) -> Result<(), String> {
        // Get current neighbors as candidates
        let current_neighbors = self
            .graph
            .get_node(node_id)
            .map(|n| n.out_neighbors.clone())
            .ok_or_else(|| format!("Node {} not found", node_id))?;

        let node_vector = self
            .vectors
            .get(&node_id)
            .ok_or_else(|| format!("Vector for node {} not found", node_id))?;

        // Convert to Neighbor structs with distances
        let mut candidates: Vec<Neighbor> = current_neighbors
            .into_iter()
            .filter_map(|neighbor_id| {
                self.vectors.get(&neighbor_id).map(|neighbor_vec| {
                    let distance = l2_distance(node_vector, neighbor_vec);
                    Neighbor::new(neighbor_id, distance)
                })
            })
            .collect();

        // Prune to diverse neighbors
        let new_neighbors = prune_neighbors(
            node_id,
            &mut candidates,
            self.config.max_degree,
            self.config.max_candidates,
            self.config.alpha,
            &self.vectors,
        );

        // Get old neighbors for comparison
        let old_neighbors = self
            .graph
            .get_node(node_id)
            .map(|n| n.out_neighbors.clone())
            .unwrap_or_default();

        // Remove edges that are no longer in the new neighbor list
        for old_neighbor in &old_neighbors {
            if !new_neighbors.contains(old_neighbor) {
                self.graph.remove_edge(node_id, *old_neighbor);
            }
        }

        // Add edges for new neighbors (some may already exist, which is fine)
        for &new_neighbor in &new_neighbors {
            if !old_neighbors.contains(&new_neighbor) {
                self.graph.add_edge(node_id, new_neighbor);
            }
        }

        Ok(())
    }

    /// Delete a vector from the index (in-place, efficient via in-neighbors)
    ///
    /// IP-DiskANN advantage: Can efficiently find all in-neighbors
    /// and remove edges pointing to this node.
    ///
    /// # Algorithm
    /// 1. Remove all edges involving this node (via bi-directional tracking)
    /// 2. Remove vector from storage
    /// 3. Optionally: re-prune affected neighbors to maintain connectivity
    pub fn delete(&mut self, node_id: NodeId) -> Result<(), String> {
        // Verify node exists
        if !self.vectors.contains_key(&node_id) {
            return Err(format!("Node {} not found", node_id));
        }

        // Remove from graph (this handles both in and out edges)
        self.graph
            .remove_node(node_id)
            .ok_or_else(|| format!("Failed to remove node {} from graph", node_id))?;

        // Remove vector from storage
        self.vectors.remove(&node_id);

        // Note: We could optionally re-prune affected neighbors here
        // to maintain graph connectivity, but for Week 1 we'll keep it simple

        Ok(())
    }

    /// Search for k nearest neighbors
    ///
    /// # Parameters
    /// - `query`: Query vector
    /// - `k`: Number of nearest neighbors to return
    ///
    /// # Returns
    /// Vector of k nearest neighbors (sorted by distance, ascending)
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>, String> {
        // Validate dimension
        if query.len() != self.dimension {
            return Err(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension,
                query.len()
            ));
        }

        // Empty index
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        Ok(greedy_search(
            query,
            &self.graph,
            &self.vectors,
            k,
            self.config.search_list_size,
        ))
    }

    /// Get number of vectors in the index
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get index configuration
    pub fn config(&self) -> &IPDiskANNConfig {
        &self.config
    }

    /// Get vector by ID
    pub fn get_vector(&self, node_id: NodeId) -> Option<&Vec<f32>> {
        self.vectors.get(&node_id)
    }

    /// Save index to file
    ///
    /// Format:
    /// - Magic number (4 bytes): "IPDA" (IP-DiskANN)
    /// - Version (u32): 1
    /// - Dimension (u32)
    /// - Config (R, C, alpha, L)
    /// - Next ID (u32)
    /// - Entry node (Option<u32>)
    /// - Number of nodes (u32)
    /// - For each node:
    ///   - Node ID (u32)
    ///   - Vector (dimension * f32)
    ///   - Out-degree (u32)
    ///   - Out-neighbors (out-degree * u32)
    ///   - In-degree (u32)
    ///   - In-neighbors (in-degree * u32)
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let mut writer = BufWriter::new(file);

        // Magic number
        writer
            .write_all(b"IPDA")
            .map_err(|e| format!("Failed to write magic: {}", e))?;

        // Version
        writer
            .write_all(&1u32.to_le_bytes())
            .map_err(|e| format!("Failed to write version: {}", e))?;

        // Dimension
        writer
            .write_all(&(self.dimension as u32).to_le_bytes())
            .map_err(|e| format!("Failed to write dimension: {}", e))?;

        // Config
        writer
            .write_all(&(self.config.max_degree as u32).to_le_bytes())
            .map_err(|e| format!("Failed to write max_degree: {}", e))?;
        writer
            .write_all(&(self.config.max_candidates as u32).to_le_bytes())
            .map_err(|e| format!("Failed to write max_candidates: {}", e))?;
        writer
            .write_all(&self.config.alpha.to_le_bytes())
            .map_err(|e| format!("Failed to write alpha: {}", e))?;
        writer
            .write_all(&(self.config.search_list_size as u32).to_le_bytes())
            .map_err(|e| format!("Failed to write search_list_size: {}", e))?;

        // Next ID
        writer
            .write_all(&self.next_id.to_le_bytes())
            .map_err(|e| format!("Failed to write next_id: {}", e))?;

        // Entry node
        let has_entry = self.graph.entry_node().is_some();
        writer
            .write_all(&[if has_entry { 1u8 } else { 0u8 }])
            .map_err(|e| format!("Failed to write entry flag: {}", e))?;
        if let Some(entry) = self.graph.entry_node() {
            writer
                .write_all(&entry.to_le_bytes())
                .map_err(|e| format!("Failed to write entry node: {}", e))?;
        }

        // Number of nodes
        writer
            .write_all(&(self.vectors.len() as u32).to_le_bytes())
            .map_err(|e| format!("Failed to write node count: {}", e))?;

        // Write each node
        for (&node_id, vector) in &self.vectors {
            // Node ID
            writer
                .write_all(&node_id.to_le_bytes())
                .map_err(|e| format!("Failed to write node ID: {}", e))?;

            // Vector
            for &val in vector {
                writer
                    .write_all(&val.to_le_bytes())
                    .map_err(|e| format!("Failed to write vector value: {}", e))?;
            }

            // Get node edges
            if let Some(node) = self.graph.get_node(node_id) {
                // Out-degree and out-neighbors
                writer
                    .write_all(&(node.out_neighbors.len() as u32).to_le_bytes())
                    .map_err(|e| format!("Failed to write out-degree: {}", e))?;
                for &neighbor in &node.out_neighbors {
                    writer
                        .write_all(&neighbor.to_le_bytes())
                        .map_err(|e| format!("Failed to write out-neighbor: {}", e))?;
                }

                // In-degree and in-neighbors
                writer
                    .write_all(&(node.in_neighbors.len() as u32).to_le_bytes())
                    .map_err(|e| format!("Failed to write in-degree: {}", e))?;
                for &neighbor in &node.in_neighbors {
                    writer
                        .write_all(&neighbor.to_le_bytes())
                        .map_err(|e| format!("Failed to write in-neighbor: {}", e))?;
                }
            } else {
                // Node not in graph (shouldn't happen)
                writer
                    .write_all(&0u32.to_le_bytes())
                    .map_err(|e| format!("Failed to write out-degree: {}", e))?;
                writer
                    .write_all(&0u32.to_le_bytes())
                    .map_err(|e| format!("Failed to write in-degree: {}", e))?;
            }
        }

        writer
            .flush()
            .map_err(|e| format!("Failed to flush: {}", e))?;
        Ok(())
    }

    /// Load index from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mut reader = BufReader::new(file);

        // Magic number
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| format!("Failed to read magic: {}", e))?;
        if &magic != b"IPDA" {
            return Err(format!(
                "Invalid magic number: expected IPDA, got {:?}",
                magic
            ));
        }

        // Version
        let mut version_bytes = [0u8; 4];
        reader
            .read_exact(&mut version_bytes)
            .map_err(|e| format!("Failed to read version: {}", e))?;
        let version = u32::from_le_bytes(version_bytes);
        if version != 1 {
            return Err(format!("Unsupported version: {}", version));
        }

        // Dimension
        let mut dim_bytes = [0u8; 4];
        reader
            .read_exact(&mut dim_bytes)
            .map_err(|e| format!("Failed to read dimension: {}", e))?;
        let dimension = u32::from_le_bytes(dim_bytes) as usize;

        // Config
        let mut buf = [0u8; 4];
        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("Failed to read max_degree: {}", e))?;
        let max_degree = u32::from_le_bytes(buf) as usize;

        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("Failed to read max_candidates: {}", e))?;
        let max_candidates = u32::from_le_bytes(buf) as usize;

        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("Failed to read alpha: {}", e))?;
        let alpha = f32::from_le_bytes(buf);

        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("Failed to read search_list_size: {}", e))?;
        let search_list_size = u32::from_le_bytes(buf) as usize;

        let config = IPDiskANNConfig {
            max_degree,
            max_candidates,
            alpha,
            search_list_size,
        };

        // Next ID
        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("Failed to read next_id: {}", e))?;
        let next_id = u32::from_le_bytes(buf);

        // Entry node
        let mut entry_flag = [0u8; 1];
        reader
            .read_exact(&mut entry_flag)
            .map_err(|e| format!("Failed to read entry flag: {}", e))?;
        let entry_node = if entry_flag[0] == 1 {
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Failed to read entry node: {}", e))?;
            Some(u32::from_le_bytes(buf))
        } else {
            None
        };

        // Number of nodes
        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("Failed to read node count: {}", e))?;
        let node_count = u32::from_le_bytes(buf) as usize;

        // Create index
        let mut graph = BiDirectionalGraph::with_capacity(node_count);
        let mut vectors = HashMap::with_capacity(node_count);

        // Read each node
        for _ in 0..node_count {
            // Node ID
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Failed to read node ID: {}", e))?;
            let node_id = u32::from_le_bytes(buf);

            // Vector
            let mut vector = vec![0.0f32; dimension];
            for val in vector.iter_mut() {
                reader
                    .read_exact(&mut buf)
                    .map_err(|e| format!("Failed to read vector value: {}", e))?;
                *val = f32::from_le_bytes(buf);
            }

            // Out-degree and out-neighbors
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Failed to read out-degree: {}", e))?;
            let out_degree = u32::from_le_bytes(buf) as usize;

            let mut out_neighbors = Vec::with_capacity(out_degree);
            for _ in 0..out_degree {
                reader
                    .read_exact(&mut buf)
                    .map_err(|e| format!("Failed to read out-neighbor: {}", e))?;
                out_neighbors.push(u32::from_le_bytes(buf));
            }

            // In-degree and in-neighbors
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Failed to read in-degree: {}", e))?;
            let in_degree = u32::from_le_bytes(buf) as usize;

            let mut in_neighbors = Vec::with_capacity(in_degree);
            for _ in 0..in_degree {
                reader
                    .read_exact(&mut buf)
                    .map_err(|e| format!("Failed to read in-neighbor: {}", e))?;
                in_neighbors.push(u32::from_le_bytes(buf));
            }

            // Add to index
            vectors.insert(node_id, vector);
            graph.add_node(node_id);

            // Manually set edges (avoid double-counting via add_edge)
            if let Some(node) = graph.get_node_mut(node_id) {
                node.out_neighbors = out_neighbors;
                node.in_neighbors = in_neighbors;
            }
        }

        // Set entry node
        if let Some(entry) = entry_node {
            graph.set_entry_node(entry);
        }

        Ok(Self {
            graph,
            vectors,
            config,
            next_id,
            dimension,
        })
    }
}

/// L2 (Euclidean) distance between two vectors
#[inline]
fn l2_distance(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_creation() {
        let index = IPDiskANNIndex::new(128, None);
        assert_eq!(index.dimension, 128);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_single_insert() {
        let mut index = IPDiskANNIndex::new(2, None);
        let vector = vec![1.0, 2.0];

        let id = index.insert(vector.clone()).unwrap();
        assert_eq!(id, 0);
        assert_eq!(index.len(), 1);
        assert_eq!(index.get_vector(id), Some(&vector));
    }

    #[test]
    fn test_multiple_inserts() {
        let mut index = IPDiskANNIndex::new(2, None);

        let id0 = index.insert(vec![0.0, 0.0]).unwrap();
        let id1 = index.insert(vec![1.0, 0.0]).unwrap();
        let id2 = index.insert(vec![2.0, 0.0]).unwrap();

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_search_single_vector() {
        let mut index = IPDiskANNIndex::new(2, None);
        index.insert(vec![1.0, 1.0]).unwrap();

        let results = index.search(&[1.0, 1.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance < 0.01); // Should be very close to 0
    }

    #[test]
    fn test_search_finds_nearest() {
        let mut index = IPDiskANNIndex::new(2, None);

        index.insert(vec![0.0, 0.0]).unwrap();
        index.insert(vec![1.0, 0.0]).unwrap();
        index.insert(vec![2.0, 0.0]).unwrap();

        // Query close to node 1
        let results = index.search(&[0.9, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn test_search_returns_k_results() {
        let mut index = IPDiskANNIndex::new(2, None);

        index.insert(vec![0.0, 0.0]).unwrap();
        index.insert(vec![1.0, 0.0]).unwrap();
        index.insert(vec![2.0, 0.0]).unwrap();
        index.insert(vec![3.0, 0.0]).unwrap();

        let results = index.search(&[1.5, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);

        // Should return nodes 1 and 2 (closest to 1.5)
        let ids: Vec<NodeId> = results.iter().map(|n| n.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn test_delete() {
        let mut index = IPDiskANNIndex::new(2, None);

        let id0 = index.insert(vec![0.0, 0.0]).unwrap();
        let id1 = index.insert(vec![1.0, 0.0]).unwrap();
        let id2 = index.insert(vec![2.0, 0.0]).unwrap();

        assert_eq!(index.len(), 3);

        // Delete node 1
        index.delete(id1).unwrap();
        assert_eq!(index.len(), 2);
        assert!(index.get_vector(id1).is_none());
        assert!(index.get_vector(id0).is_some());
        assert!(index.get_vector(id2).is_some());
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut index = IPDiskANNIndex::new(2, None);
        index.insert(vec![0.0, 0.0]).unwrap();

        let result = index.delete(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_validation() {
        let mut index = IPDiskANNIndex::new(2, None);

        // Insert wrong dimension
        let result = index.insert(vec![1.0, 2.0, 3.0]);
        assert!(result.is_err());

        // Search wrong dimension
        index.insert(vec![1.0, 2.0]).unwrap();
        let result = index.search(&[1.0, 2.0, 3.0], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_empty_index() {
        let index = IPDiskANNIndex::new(2, None);
        let results = index.search(&[1.0, 2.0], 5).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_custom_config() {
        let config = IPDiskANNConfig {
            max_degree: 32,
            max_candidates: 500,
            alpha: 1.5,
            search_list_size: 75,
        };

        let index = IPDiskANNIndex::new(128, Some(config.clone()));
        assert_eq!(index.config().max_degree, 32);
        assert_eq!(index.config().alpha, 1.5);
    }

    #[test]
    fn test_save_and_load() {
        use std::fs;

        let mut index = IPDiskANNIndex::new(2, None);

        // Build index
        index.insert(vec![0.0, 0.0]).unwrap();
        index.insert(vec![1.0, 0.0]).unwrap();
        index.insert(vec![2.0, 0.0]).unwrap();

        let original_len = index.len();
        let original_config = index.config().clone();

        // Save
        let path = "/tmp/test_ip_diskann.bin";
        index.save(path).unwrap();

        // Load
        let loaded = IPDiskANNIndex::load(path).unwrap();

        // Verify
        assert_eq!(loaded.len(), original_len);
        assert_eq!(loaded.dimension, 2);
        assert_eq!(loaded.config().max_degree, original_config.max_degree);
        assert_eq!(loaded.config().alpha, original_config.alpha);

        // Verify vectors
        for i in 0..3 {
            assert!(loaded.get_vector(i).is_some());
        }

        // Verify search works
        let results = loaded.search(&[0.9, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1); // Should find node 1 as nearest

        // Cleanup
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_save_and_load_empty_index() {
        use std::fs;

        let index = IPDiskANNIndex::new(128, None);

        let path = "/tmp/test_ip_diskann_empty.bin";
        index.save(path).unwrap();

        let loaded = IPDiskANNIndex::load(path).unwrap();
        assert_eq!(loaded.len(), 0);
        assert_eq!(loaded.dimension, 128);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_load_invalid_magic() {
        use std::fs::File;
        use std::io::Write;

        let path = "/tmp/test_invalid_magic.bin";
        let mut file = File::create(path).unwrap();
        file.write_all(b"XXXX").unwrap(); // Wrong magic

        let result = IPDiskANNIndex::load(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid magic"));

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_save_and_load_preserves_graph_structure() {
        use std::fs;

        let mut index = IPDiskANNIndex::new(2, None);

        // Build a simple graph
        index.insert(vec![0.0, 0.0]).unwrap();
        index.insert(vec![1.0, 0.0]).unwrap();
        index.insert(vec![2.0, 0.0]).unwrap();

        // Get original graph structure
        let original_edges: Vec<(NodeId, Vec<NodeId>)> = (0..3)
            .filter_map(|id| {
                index
                    .graph
                    .get_node(id)
                    .map(|n| (id, n.out_neighbors.clone()))
            })
            .collect();

        // Save and load
        let path = "/tmp/test_graph_structure.bin";
        index.save(path).unwrap();
        let loaded = IPDiskANNIndex::load(path).unwrap();

        // Verify graph structure
        for (node_id, expected_neighbors) in original_edges {
            if let Some(node) = loaded.graph.get_node(node_id) {
                assert_eq!(node.out_neighbors, expected_neighbors);
            }
        }

        fs::remove_file(path).unwrap();
    }
}
