//! Mock storage backend for testing.
//!
//! Provides an in-memory implementation of StorageBackend for unit tests.

use crate::catalog::CollectionSchema;
use crate::omen::{CheckpointOptions, Metric, StorageBackend};
use anyhow::Result;
use std::collections::HashMap;

pub struct MockStorageBackend {
    dimensions: usize,
    metric: Metric,
    vectors: HashMap<u32, Vec<f32>>,
    config: HashMap<String, u64>,
    schema: Option<CollectionSchema>,
    wal: Vec<String>, // simplified WAL tracking
}

impl MockStorageBackend {
    pub fn new(dimensions: usize, metric: Metric) -> Self {
        Self {
            dimensions,
            metric,
            vectors: HashMap::new(),
            config: HashMap::new(),
            schema: None,
            wal: Vec::new(),
        }
    }
}

impl StorageBackend for MockStorageBackend {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn set_dimensions(&mut self, dimensions: u32) -> Result<()> {
        self.dimensions = dimensions as usize;
        Ok(())
    }

    fn set_metric(&mut self, metric: Metric) -> Result<()> {
        self.metric = metric;
        Ok(())
    }

    fn schema(&self) -> Option<CollectionSchema> {
        self.schema.clone()
    }

    fn set_schema(&mut self, schema: CollectionSchema) -> Result<()> {
        self.schema = Some(schema);
        Ok(())
    }

    fn set_hnsw_params(&mut self, _m: u16, _ef_construction: u16, _ef_search: u16) -> Result<()> {
        Ok(())
    }

    fn log_insert(
        &mut self,
        id: &str,
        _vector: &[f32],
        _metadata: &serde_json::Value,
    ) -> Result<()> {
        self.wal.push(format!("insert:{id}"));
        // In a real mock, we might want to store the vector here too if we want to
        // simulate recovery.
        Ok(())
    }

    fn log_delete(&mut self, id: &str) -> Result<()> {
        self.wal.push(format!("delete:{id}"));
        Ok(())
    }

    fn log_upsert_sparse(
        &mut self,
        id: &str,
        _sparse: &crate::vector::sparse::SparseVector,
        _metadata: &serde_json::Value,
    ) -> Result<()> {
        self.wal.push(format!("sparse:{id}"));
        Ok(())
    }

    fn log_upsert_multi(
        &mut self,
        id: &str,
        _tokens: &[Vec<f32>],
        _metadata: &serde_json::Value,
    ) -> Result<()> {
        self.wal.push(format!("multi:{id}"));
        Ok(())
    }

    fn log_insert_edge(
        &mut self,
        from_id: &str,
        to_id: &str,
        edge_type: &str,
        _weight: f32,
        _metadata: Option<&[u8]>,
    ) -> Result<()> {
        self.wal
            .push(format!("insert_edge:{from_id}:{to_id}:{edge_type}"));
        Ok(())
    }

    fn log_delete_edge(&mut self, from_id: &str, to_id: &str, edge_type: &str) -> Result<()> {
        self.wal
            .push(format!("delete_edge:{from_id}:{to_id}:{edge_type}"));
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<()> {
        self.wal.clear();
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        Ok(())
    }

    fn put_config(&mut self, key: &str, value: u64) -> Result<()> {
        self.config.insert(key.to_string(), value);
        Ok(())
    }

    fn wal_len(&self) -> usize {
        self.wal.len()
    }

    fn has_vec_file(&self) -> bool {
        !self.vectors.is_empty()
    }

    fn checkpoint_vectors_only(
        &mut self,
        records: &crate::vector::store::record_store::RecordStore,
        dirty_slots: &roaring::RoaringBitmap,
    ) -> Result<()> {
        for slot in dirty_slots {
            if let Some(vector) = records.get_vector(slot) {
                self.vectors.insert(slot, vector.to_vec());
            } else {
                self.vectors.remove(&slot);
            }
        }
        Ok(())
    }

    fn checkpoint_incremental(
        &mut self,
        records: &crate::vector::store::record_store::RecordStore,
        dirty_slots: &roaring::RoaringBitmap,
        _options: CheckpointOptions,
    ) -> Result<()> {
        self.checkpoint_vectors_only(records, dirty_slots)
    }

    fn checkpoint_full(
        &mut self,
        records: &crate::vector::store::record_store::RecordStore,
        _options: CheckpointOptions,
    ) -> Result<()> {
        self.vectors.clear();
        for (slot, record) in records.iter_live() {
            if let Some(vector) = record.vector {
                self.vectors.insert(slot, vector.to_vec());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::store::record_store::RecordStore;

    #[test]
    fn test_mock_backend_basics() {
        let mut backend = MockStorageBackend::new(4, Metric::L2);
        assert_eq!(backend.dimensions(), 4);
        assert_eq!(backend.metric(), Metric::L2);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        backend
            .log_insert("doc1", &v1, &serde_json::json!({}))
            .unwrap();
        assert_eq!(backend.wal_len(), 1);

        backend.checkpoint().unwrap();
        assert_eq!(backend.wal_len(), 0);
    }

    #[test]
    fn test_mock_backend_checkpoint() {
        let mut backend = MockStorageBackend::new(4, Metric::L2);
        let records = RecordStore::new(4);
        records
            .set("doc1".to_string(), vec![1.0, 2.0, 3.0, 4.0], None)
            .unwrap();

        let mut dirty = roaring::RoaringBitmap::new();
        dirty.insert(0);

        backend.checkpoint_vectors_only(&records, &dirty).unwrap();

        let v = backend.vectors.get(&0).expect("should have vector");
        assert_eq!(*v, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
