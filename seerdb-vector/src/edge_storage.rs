//! Graph edge storage using seerdb
//!
//! Provides persistent storage for HNSW graph edges using seerdb as the backend.
//! This replaces fjall-based storage with better performance characteristics.

use crate::{config::SeerdbVectorConfig, Result, SeerdbVectorError};
use seerdb::db::{DBOptions, DB};
use seerdb::SyncPolicy;
use std::path::PathBuf;
use std::sync::Arc;

/// Size of edge key (node_id + level = 9 bytes)
const KEY_SIZE: usize = 9;

/// Edge storage using seerdb
///
/// Stores HNSW graph edges in seerdb as adjacency lists.
/// Key format: `node_id (8 bytes) || level (1 byte)`
/// Value format: `Vec<u64>` serialized via bincode
#[derive(Clone)]
pub struct EdgeStorage {
    db: Arc<DB>,
}

impl EdgeStorage {
    /// Create new edge storage at path
    pub fn new(path: PathBuf, config: &SeerdbVectorConfig) -> Result<Self> {
        // Optimal configuration for HNSW graph storage
        let options = DBOptions {
            data_dir: path,

            // Configurable durability (default: None for performance)
            wal_sync_policy: if config.sync_writes {
                SyncPolicy::SyncData
            } else {
                SyncPolicy::None
            },

            // Large memtable: Fewer flushes during graph construction
            memtable_capacity: config.memtable_capacity,

            // Large cache: High hit rate for adjacency list lookups
            block_cache_capacity: config.block_cache_capacity,

            // Background compaction: Non-blocking writes
            background_compaction: config.background_compaction,

            // Use vLog for large neighbor lists if needed
            // M=48 -> 384 bytes (inline)
            // M=1000 -> 8000 bytes (vLog)
            vlog_threshold: Some(4096),

            ..Default::default()
        };

        let db = DB::open(options).map_err(|e| SeerdbVectorError::Backend(e.to_string()))?;

        Ok(Self { db: Arc::new(db) })
    }

    /// Create from existing seerdb instance
    pub fn from_db(db: Arc<DB>) -> Self {
        Self { db }
    }

    /// Get reference to underlying seerdb
    pub fn db(&self) -> &DB {
        &self.db
    }

    /// Get cache statistics for performance monitoring
    pub fn cache_stats(&self) -> (u64, u64, f64) {
        let stats = self.db.stats();
        (stats.cache_hits, stats.cache_misses, stats.cache_hit_rate)
    }

    /// Encode edge key: node_id || level
    fn encode_key(node_id: u64, level: u8) -> [u8; KEY_SIZE] {
        let mut key = [0u8; KEY_SIZE];
        key[0..8].copy_from_slice(&node_id.to_be_bytes());
        key[8] = level;
        key
    }

    /// Add/Overwrite neighbor list for a node
    ///
    /// Overwrites any existing neighbor list.
    pub fn put_neighbors(&self, node_id: u64, level: u8, neighbors: &[u64]) -> Result<()> {
        let key = Self::encode_key(node_id, level);
        let value = bincode::serialize(neighbors)
            .map_err(|e| SeerdbVectorError::Serialization(e.to_string()))?;

        self.db
            .put(key, &value)
            .map_err(|e| SeerdbVectorError::Backend(e.to_string()))
    }

    /// Add multiple neighbor lists in a batch
    ///
    /// Takes `(node_id, level, neighbors)` tuples.
    pub fn put_neighbors_batch(&self, batch: &[(u64, u8, Vec<u64>)]) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        let mut db_batch = self.db.batch_with_capacity(batch.len());

        for (node_id, level, neighbors) in batch {
            let key = Self::encode_key(*node_id, *level);
            let value = bincode::serialize(neighbors)
                .map_err(|e| SeerdbVectorError::Serialization(e.to_string()))?;
            db_batch.put(key, &value);
        }

        db_batch
            .commit()
            .map_err(|e| SeerdbVectorError::Backend(e.to_string()))?;
        Ok(())
    }

    /// Get all neighbors of a node at a specific level
    ///
    /// Uses point lookup O(1) instead of prefix scan O(N_SST).
    pub fn get_neighbors(&self, node_id: u64, level: u8) -> Result<Vec<u64>> {
        let key = Self::encode_key(node_id, level);

        if let Some(value) = self
            .db
            .get(key)
            .map_err(|e| SeerdbVectorError::Backend(e.to_string()))?
        {
            // Deserialize Vec<u64>
            let neighbors: Vec<u64> = bincode::deserialize(&value)
                .map_err(|e| SeerdbVectorError::Serialization(e.to_string()))?;
            Ok(neighbors)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get neighbor count at level
    pub fn neighbor_count(&self, node_id: u64, level: u8) -> Result<usize> {
        Ok(self.get_neighbors(node_id, level)?.len())
    }

    /// Flush to disk
    pub fn flush(&self) -> Result<()> {
        self.db
            .flush()
            .map_err(|e| SeerdbVectorError::Backend(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_put_and_get_neighbors() {
        let temp_dir = TempDir::new().unwrap();
        let storage = EdgeStorage::new(
            temp_dir.path().to_path_buf(),
            &SeerdbVectorConfig::default(),
        )
        .unwrap();

        // Add neighbors
        let neighbors = vec![100, 200, 300];
        storage.put_neighbors(1, 0, &neighbors).unwrap();

        // Get neighbors
        let retrieved = storage.get_neighbors(1, 0).unwrap();
        assert_eq!(retrieved, neighbors);
    }

    #[test]
    fn test_put_neighbors_batch() {
        let temp_dir = TempDir::new().unwrap();
        let storage = EdgeStorage::new(
            temp_dir.path().to_path_buf(),
            &SeerdbVectorConfig::default(),
        )
        .unwrap();

        let batch = vec![(1, 0, vec![2, 3]), (2, 0, vec![1, 3]), (3, 0, vec![1, 2])];
        storage.put_neighbors_batch(&batch).unwrap();

        assert_eq!(storage.get_neighbors(1, 0).unwrap(), vec![2, 3]);
        assert_eq!(storage.get_neighbors(2, 0).unwrap(), vec![1, 3]);
        assert_eq!(storage.get_neighbors(3, 0).unwrap(), vec![1, 2]);
    }

    #[test]
    fn test_get_neighbors_empty() {
        let temp_dir = TempDir::new().unwrap();
        let storage = EdgeStorage::new(
            temp_dir.path().to_path_buf(),
            &SeerdbVectorConfig::default(),
        )
        .unwrap();

        let retrieved = storage.get_neighbors(999, 0).unwrap();
        assert!(retrieved.is_empty());
    }

    #[test]
    #[ignore] // Benchmark
    fn bench_get_neighbors_performance() {
        use std::time::Instant;
        let temp_dir = TempDir::new().unwrap();

        let config = SeerdbVectorConfig::builder()
            .memtable_capacity(1024 * 1024)
            .build();

        let storage = EdgeStorage::new(temp_dir.path().to_path_buf(), &config).unwrap();

        // 1. Insert noise to create multiple SSTables
        println!("Inserting noise to create SSTables...");
        for batch in 0..10 {
            let mut noise_batch = Vec::new();
            for i in 0..5_000 {
                let id = 2 + (batch * 5000) + i;
                noise_batch.push((id, 0, vec![2000 + id]));
            }
            storage.put_neighbors_batch(&noise_batch).unwrap();
            storage.flush().unwrap();
        }

        println!("Created ~10 SSTables");

        // 2. Insert 32 neighbors for node 1 (Last -> Newest)
        let neighbors: Vec<u64> = (0..32).map(|i| 1000 + i).collect();
        storage.put_neighbors(1, 0, &neighbors).unwrap();

        // 3. Benchmark Get
        let start = Instant::now();
        let iterations = 1000;
        for _ in 0..iterations {
            let _ = storage.get_neighbors(1, 0).unwrap();
        }
        let duration = start.elapsed();
        let avg = duration.as_micros() as f64 / iterations as f64;

        println!("Get Neighbors (32 items): {:.2} us/op", avg);
    }
}
