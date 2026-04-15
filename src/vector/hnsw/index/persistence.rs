//! HNSW index persistence (save/load)
//!
//! Format versions:
//! - v5: HNSWStorage format - unified flat matrices (SOTA)

use super::HNSWIndex;
use crate::vector::hnsw::error::{HNSWError, Result};
use crate::vector::hnsw::storage::HNSWStorage;
use crate::vector::hnsw::types::{HNSWParams, Metric};
use std::fs::OpenOptions;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use tracing::instrument;

const FORMAT_VERSION: u32 = 5;

#[cfg(windows)]
fn configure_open_options(opts: &mut OpenOptions) {
    use std::os::windows::fs::OpenOptionsExt;
    opts.share_mode(0x1 | 0x2 | 0x4);
}

#[cfg(not(windows))]
fn configure_open_options(_opts: &mut OpenOptions) {}

impl HNSWIndex {
    #[instrument(skip(self, path))]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut opts = OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        configure_open_options(&mut opts);
        let file = opts.open(path)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(b"HNSWIDX\0")?;
        writer.write_all(&FORMAT_VERSION.to_le_bytes())?;

        // Entry point
        match self.entry_point {
            Some(ep) => {
                writer.write_all(&[1u8])?;
                writer.write_all(&ep.to_le_bytes())?;
            }
            None => writer.write_all(&[0u8])?,
        }

        // Distance function
        let df_bytes = postcard::to_allocvec(&self.distance_fn)
            .map_err(|e| HNSWError::Serialization(e.to_string()))?;
        writer.write_all(&(df_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&df_bytes)?;

        // Params
        let params_bytes = postcard::to_allocvec(&self.params)
            .map_err(|e| HNSWError::Serialization(e.to_string()))?;
        writer.write_all(&(params_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&params_bytes)?;

        // RNG state
        writer.write_all(&self.rng_state.to_le_bytes())?;

        // Storage (serialized via serde)
        let storage_bytes = postcard::to_allocvec(&self.storage)
            .map_err(|e| HNSWError::Serialization(e.to_string()))?;
        writer.write_all(&(storage_bytes.len() as u64).to_le_bytes())?;
        writer.write_all(&storage_bytes)?;

        writer.flush()?;
        Ok(())
    }

    #[instrument(skip(path))]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new().read(true).open(path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != b"HNSWIDX\0" {
            return Err(HNSWError::Storage(format!(
                "Invalid magic: expected HNSWIDX\\0, got {:?}",
                magic
            )));
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);

        if version != FORMAT_VERSION {
            return Err(HNSWError::Storage(format!(
                "Unsupported version: expected {}, got {}",
                FORMAT_VERSION, version
            )));
        }

        let mut ep_flag = [0u8; 1];
        reader.read_exact(&mut ep_flag)?;
        let entry_point = if ep_flag[0] == 1 {
            let mut ep_bytes = [0u8; 4];
            reader.read_exact(&mut ep_bytes)?;
            Some(u32::from_le_bytes(ep_bytes))
        } else {
            None
        };

        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        let df_len = u32::from_le_bytes(len_bytes) as usize;
        let mut df_bytes = vec![0u8; df_len];
        reader.read_exact(&mut df_bytes)?;
        let distance_fn: Metric =
            postcard::from_bytes(&df_bytes).map_err(|e| HNSWError::Serialization(e.to_string()))?;

        reader.read_exact(&mut len_bytes)?;
        let params_len = u32::from_le_bytes(len_bytes) as usize;
        let mut params_bytes = vec![0u8; params_len];
        reader.read_exact(&mut params_bytes)?;
        let params: HNSWParams = postcard::from_bytes(&params_bytes)
            .map_err(|e| HNSWError::Serialization(e.to_string()))?;

        let mut rng_bytes = [0u8; 8];
        reader.read_exact(&mut rng_bytes)?;
        let rng_state = u64::from_le_bytes(rng_bytes);

        let mut storage_len_bytes = [0u8; 8];
        reader.read_exact(&mut storage_len_bytes)?;
        let storage_len = u64::from_le_bytes(storage_len_bytes) as usize;
        let mut storage_bytes = vec![0u8; storage_len];
        reader.read_exact(&mut storage_bytes)?;
        let mut storage: HNSWStorage = postcard::from_bytes(&storage_bytes)
            .map_err(|e| HNSWError::Serialization(e.to_string()))?;

        storage.restore_locks();

        Ok(Self {
            storage,
            entry_point,
            params,
            distance_fn,
            rng_state,
        })
    }
}
