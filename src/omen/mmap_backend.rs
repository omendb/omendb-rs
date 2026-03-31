//! Mmap-based storage backend.
//!
//! A simpler backend than OmenFile, intended for cases where we want
//! direct mmap access to vectors without the single-file Omen container overhead.

use crate::omen::{Metric, StorageBackend};
use anyhow::Result;
use memmap2::Mmap;
use std::fs::File;
use std::path::{Path, PathBuf};

pub struct MmapBackend {
    dimensions: usize,
    metric: Metric,
    _path: PathBuf,
    mmap: Option<Mmap>,
}

impl MmapBackend {
    pub fn open<P: AsRef<Path>>(path: P, dimensions: usize, metric: Metric) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mmap = if path.exists() {
            let file = File::open(&path)?;
            if file.metadata()?.len() > 0 {
                Some(unsafe { Mmap::map(&file)? })
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            dimensions,
            metric,
            _path: path,
            mmap,
        })
    }
}

impl StorageBackend for MmapBackend {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn set_dimensions(&mut self, _dimensions: u32) -> Result<()> {
        Ok(())
    }

    fn set_metric(&mut self, _metric: Metric) -> Result<()> {
        Ok(())
    }

    fn set_hnsw_params(&mut self, _m: u16, _ef_construction: u16, _ef_search: u16) -> Result<()> {
        Ok(())
    }

    fn get_vector(&self, slot: u32) -> Result<Option<Vec<f32>>> {
        let mmap = match &self.mmap {
            Some(m) => m,
            None => return Ok(None),
        };

        let vec_size = self.dimensions * std::mem::size_of::<f32>();
        let offset = slot as usize * vec_size;

        if offset + vec_size > mmap.len() {
            return Ok(None);
        }

        let slice = &mmap[offset..offset + vec_size];
        let mut vector = vec![0.0f32; self.dimensions];
        
        // Unsafe block to copy f32s from bytes. 
        // In OmenDB, we usually use zerocopy or manual casting.
        unsafe {
            std::ptr::copy_nonoverlapping(
                slice.as_ptr(),
                vector.as_mut_ptr() as *mut u8,
                vec_size,
            );
        }

        Ok(Some(vector))
    }

    fn log_insert(&mut self, _id: &str, _vector: &[f32], _metadata: &serde_json::Value) -> Result<()> {
        // MmapBackend doesn't have a WAL. It's intended for read-only or 
        // direct-to-file scenarios.
        anyhow::bail!("MmapBackend does not support WAL logging")
    }

    fn log_delete(&mut self, _id: &str) -> Result<()> {
        anyhow::bail!("MmapBackend does not support WAL logging")
    }

    fn log_insert_edge(
        &mut self,
        _from_id: &str,
        _to_id: &str,
        _edge_type: &str,
        _weight: f32,
        _metadata: Option<&[u8]>,
    ) -> Result<()> {
        anyhow::bail!("MmapBackend does not support WAL logging")
    }

    fn log_delete_edge(&mut self, _from_id: &str, _to_id: &str, _edge_type: &str) -> Result<()> {
        anyhow::bail!("MmapBackend does not support WAL logging")
    }

    fn checkpoint(&mut self) -> Result<()> {
        // No-op for now
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        // No-op
        Ok(())
    }

    fn put_config(&mut self, _key: &str, _value: u64) -> Result<()> {
        // No-op for now
        Ok(())
    }

    fn wal_len(&self) -> usize {
        0
    }

    fn has_vec_file(&self) -> bool {
        false
    }

    fn checkpoint_vectors_only(
        &mut self,
        _records: &crate::vector::store::record_store::RecordStore,
        _dirty_slots: &roaring::RoaringBitmap,
    ) -> Result<()> {
        anyhow::bail!("MmapBackend does not support checkpointing")
    }

    fn checkpoint_incremental(
        &mut self,
        _records: &crate::vector::store::record_store::RecordStore,
        _dirty_slots: &roaring::RoaringBitmap,
        _options: crate::omen::CheckpointOptions,
    ) -> Result<()> {
        anyhow::bail!("MmapBackend does not support checkpointing")
    }

    fn checkpoint_full(
        &mut self,
        _records: &crate::vector::store::record_store::RecordStore,
        _options: crate::omen::CheckpointOptions,
    ) -> Result<()> {
        anyhow::bail!("MmapBackend does not support checkpointing")
    }
}
