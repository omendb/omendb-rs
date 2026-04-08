//! Serialization and deserialization for NodeStorage
//!
//! Supports both owned heap allocations and memory-mapped files.

use super::{CACHE_LINE, NodeStorage, StorageBacking, upper_neighbors::UpperNeighborsStorage};
use crate::compression::scalar::ScalarParams;
use std::alloc::Layout;
use std::ptr::NonNull;

const MAX_AUX_ELEMENTS: usize = 1_000_000_000;

fn read_bytes<'a>(data: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], String> {
    if *pos + n > data.len() {
        return Err(format!(
            "Data too short: need {} bytes at position {}, have {}",
            n,
            *pos,
            data.len()
        ));
    }
    let result = &data[*pos..*pos + n];
    *pos += n;
    Ok(result)
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    let bytes = read_bytes(data, pos, 8)?;
    Ok(u64::from_le_bytes(
        bytes.try_into().map_err(|_| "Invalid u64")?,
    ))
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    let bytes = read_bytes(data, pos, 4)?;
    Ok(u32::from_le_bytes(
        bytes.try_into().map_err(|_| "Invalid u32")?,
    ))
}

fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16, String> {
    let bytes = read_bytes(data, pos, 2)?;
    Ok(u16::from_le_bytes(
        bytes.try_into().map_err(|_| "Invalid u16")?,
    ))
}

fn read_f32(data: &[u8], pos: &mut usize) -> Result<f32, String> {
    let bytes = read_bytes(data, pos, 4)?;
    Ok(f32::from_le_bytes(
        bytes.try_into().map_err(|_| "Invalid f32")?,
    ))
}

fn read_i32(data: &[u8], pos: &mut usize) -> Result<i32, String> {
    let bytes = read_bytes(data, pos, 4)?;
    Ok(i32::from_le_bytes(
        bytes.try_into().map_err(|_| "Invalid i32")?,
    ))
}

fn read_u8(data: &[u8], pos: &mut usize) -> Result<u8, String> {
    let bytes = read_bytes(data, pos, 1)?;
    Ok(bytes[0])
}

impl NodeStorage {
    /// Get raw bytes of storage data (for persistence)
    ///
    /// Returns a slice of all node data (len * node_size bytes).
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match &self.backing {
            StorageBacking::Owned { data, .. } => {
                if self.len == 0 {
                    &[]
                } else {
                    // SAFETY: self.len == 0 guard above, pointer from valid owned allocation, size = len * node_size
                    unsafe { std::slice::from_raw_parts(data.as_ptr(), self.len * self.node_size) }
                }
            }
            #[cfg(feature = "mmap")]
            StorageBacking::Mmap(mmap) => &mmap[..self.len * self.node_size],
        }
    }

    /// Construct storage from raw bytes (for loading)
    ///
    /// Takes ownership of the data vector.
    ///
    /// # Panics
    /// Panics if parameters are inconsistent with data size.
    #[allow(clippy::too_many_arguments)]
    pub fn from_bytes(
        data: Vec<u8>,
        len: usize,
        node_size: usize,
        neighbors_offset: usize,
        vector_offset: usize,
        metadata_offset: usize,
        dimensions: usize,
        max_neighbors: usize,
    ) -> Self {
        // Validate parameters to prevent memory safety issues
        let expected_size = len.checked_mul(node_size);
        assert!(
            expected_size.is_some() && expected_size.unwrap() <= data.len(),
            "Invalid segment: len={} * node_size={} exceeds data.len()={}",
            len,
            node_size,
            data.len()
        );
        assert!(
            node_size == 0 || neighbors_offset < node_size,
            "Invalid segment: neighbors_offset {neighbors_offset} >= node_size {node_size}",
        );
        assert!(
            node_size == 0 || vector_offset < node_size,
            "Invalid segment: vector_offset {vector_offset} >= node_size {node_size}",
        );

        let capacity = if node_size > 0 && !data.is_empty() {
            data.len() / node_size
        } else {
            0
        };

        // Convert Vec<u8> to owned allocation with proper alignment
        let backing = if data.is_empty() {
            StorageBacking::default()
        } else {
            // Allocate with CACHE_LINE alignment for optimal performance
            let layout = Layout::from_size_align(data.len(), CACHE_LINE).expect("Invalid layout");
            // SAFETY: We allocate with proper alignment and copy data
            let ptr = unsafe {
                use std::alloc::alloc;
                let raw = alloc(layout);
                if raw.is_null() {
                    std::alloc::handle_alloc_error(layout);
                }
                // Copy data to properly aligned allocation
                std::ptr::copy_nonoverlapping(data.as_ptr(), raw, data.len());
                // SAFETY: raw null-checked above
                NonNull::new_unchecked(raw)
            };
            // data is dropped here, freeing the original unaligned allocation
            StorageBacking::Owned {
                data: ptr,
                layout,
                capacity,
            }
        };

        // M = max_neighbors / 2 (level 0 has M*2)
        let max_neighbors_upper = max_neighbors / 2;

        Self {
            backing,
            len,
            node_size,
            neighbors_offset,
            vector_offset,
            metadata_offset,
            dimensions,
            max_neighbors,
            max_neighbors_upper,
            max_level: 8, // Default max level
            upper_neighbors: UpperNeighborsStorage::default(),
            sq8: false, // Default to full precision for loaded data
            sq8_params: None,
            norms: Vec::new(),
            sq8_sums: Vec::new(),
            training_buffer: Vec::new(),
            sq8_trained: false,
        }
    }

    /// Construct storage from memory-mapped file (for mmap loading)
    ///
    /// # Panics
    /// Panics if parameters are inconsistent with mmap size.
    #[cfg(feature = "mmap")]
    #[allow(clippy::too_many_arguments)]
    pub fn from_mmap(
        mmap: memmap2::Mmap,
        len: usize,
        node_size: usize,
        neighbors_offset: usize,
        vector_offset: usize,
        metadata_offset: usize,
        dimensions: usize,
        max_neighbors: usize,
    ) -> Self {
        // Same validation as from_bytes — prevents SIGBUS from corrupted files
        let expected_size = len.checked_mul(node_size);
        assert!(
            expected_size.is_some() && expected_size.unwrap() <= mmap.len(),
            "Invalid segment: len={} * node_size={} exceeds mmap.len()={}",
            len,
            node_size,
            mmap.len()
        );
        assert!(
            node_size == 0 || neighbors_offset < node_size,
            "Invalid segment: neighbors_offset {neighbors_offset} >= node_size {node_size}",
        );
        assert!(
            node_size == 0 || vector_offset < node_size,
            "Invalid segment: vector_offset {vector_offset} >= node_size {node_size}",
        );

        let max_neighbors_upper = max_neighbors / 2;

        Self {
            backing: StorageBacking::Mmap(mmap),
            len,
            node_size,
            neighbors_offset,
            vector_offset,
            metadata_offset,
            dimensions,
            max_neighbors,
            max_neighbors_upper,
            max_level: 8,
            upper_neighbors: UpperNeighborsStorage::default(),
            sq8: false,
            sq8_params: None,
            norms: Vec::new(),
            sq8_sums: Vec::new(),
            training_buffer: Vec::new(),
            sq8_trained: false,
        }
    }

    /// Serialize SQ8 params, norms, sums, and legacy compatibility fields.
    fn serialize_aux_body_without_upper(&self, out: &mut Vec<u8>) {
        // SQ8 params if present
        if let Some(ref params) = self.sq8_params {
            out.push(1);
            out.extend_from_slice(&params.scale.to_le_bytes());
            out.extend_from_slice(&params.offset.to_le_bytes());
        } else {
            out.push(0);
        }

        // Norms
        out.extend_from_slice(&(self.norms.len() as u64).to_le_bytes());
        for &norm in &self.norms {
            out.extend_from_slice(&norm.to_le_bytes());
        }

        // SQ8 sums
        out.extend_from_slice(&(self.sq8_sums.len() as u64).to_le_bytes());
        for &sum in &self.sq8_sums {
            out.extend_from_slice(&sum.to_le_bytes());
        }

        // Legacy PQ fields (write zeros for backward compatibility)
        out.push(0); // pq_trained = false
        out.push(0); // no PQ params
        out.extend_from_slice(&0u64.to_le_bytes()); // pq_codes length = 0

        // Legacy RaBitQ fields (write zeros for backward compatibility)
        out.push(0); // rabitq_trained = false
        out.push(0); // no rabitq params
        out.extend_from_slice(&0u64.to_le_bytes()); // rabitq_codes length = 0
        out.extend_from_slice(&0u64.to_le_bytes()); // rabitq_metadata length = 0
        out.extend_from_slice(&0u64.to_le_bytes()); // rabitq_originals length = 0
    }

    fn serialize_legacy_upper_neighbors(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.upper_neighbors.nodes_with_upper() as u64).to_le_bytes());
        for node_id in 0..self.len as u32 {
            let level_count = self.upper_neighbors.upper_level_count(node_id);
            if level_count == 0 {
                continue;
            }
            out.extend_from_slice(&node_id.to_le_bytes());
            out.push(level_count as u8);
            for level in 1..=level_count as u8 {
                let neighbors = self.upper_neighbors.neighbors_at_level_cow(
                    node_id,
                    level,
                    self.max_neighbors_upper,
                );
                out.extend_from_slice(&(neighbors.len() as u16).to_le_bytes());
                for &neighbor in neighbors.iter() {
                    out.extend_from_slice(&neighbor.to_le_bytes());
                }
            }
        }
    }

    /// Deserialize SQ8 params, norms, sums, and legacy compatibility fields.
    fn deserialize_aux_body_without_upper(
        &mut self,
        data: &[u8],
        pos: &mut usize,
    ) -> Result<(), String> {
        // SQ8 params
        let has_params = read_u8(data, pos)? != 0;
        let sq8_params = if has_params {
            let scale = read_f32(data, pos)?;
            let offset = read_f32(data, pos)?;
            Some(ScalarParams {
                scale,
                offset,
                dimensions: self.dimensions,
            })
        } else {
            None
        };

        // Norms
        let norms_len = read_u64(data, pos)? as usize;
        if norms_len > MAX_AUX_ELEMENTS {
            return Err(format!("Norms length {norms_len} exceeds safety cap"));
        }
        let mut norms = Vec::with_capacity(norms_len);
        for _ in 0..norms_len {
            norms.push(read_f32(data, pos)?);
        }

        // SQ8 sums
        let sums_len = read_u64(data, pos)? as usize;
        if sums_len > MAX_AUX_ELEMENTS {
            return Err(format!("SQ8 sums length {sums_len} exceeds safety cap"));
        }
        let mut sq8_sums = Vec::with_capacity(sums_len);
        for _ in 0..sums_len {
            sq8_sums.push(read_i32(data, pos)?);
        }

        // Legacy PQ state (skip)
        let _pq_trained = read_u8(data, pos)? != 0;
        let has_pq_params = read_u8(data, pos)? != 0;
        if has_pq_params {
            let codebook_len = read_u64(data, pos)? as usize;
            let _codebook_bytes = read_bytes(data, pos, codebook_len)?;
        }
        let pq_codes_len = read_u64(data, pos)? as usize;
        let _pq_codes = read_bytes(data, pos, pq_codes_len)?;

        // Legacy RaBitQ state (skip all fields)
        if *pos + 2 <= data.len() {
            let _trained = read_u8(data, pos)? != 0;
            let has_rabitq_params = read_u8(data, pos)? != 0;
            if has_rabitq_params {
                let param_len = read_u64(data, pos)? as usize;
                let _ = read_bytes(data, pos, param_len)?;
            }
            let codes_len = read_u64(data, pos)? as usize;
            if codes_len > MAX_AUX_ELEMENTS {
                return Err(format!(
                    "RaBitQ codes length {codes_len} exceeds safety cap"
                ));
            }
            for _ in 0..codes_len {
                let _ = read_u64(data, pos)?;
            }
            let meta_len = read_u64(data, pos)? as usize;
            if meta_len > MAX_AUX_ELEMENTS {
                return Err(format!(
                    "RaBitQ metadata length {meta_len} exceeds safety cap"
                ));
            }
            for _ in 0..meta_len {
                let _ = read_f32(data, pos)?;
            }
            if *pos + 8 <= data.len() {
                let orig_len = read_u64(data, pos)? as usize;
                if orig_len > MAX_AUX_ELEMENTS {
                    return Err(format!(
                        "RaBitQ originals length {orig_len} exceeds safety cap"
                    ));
                }
                for _ in 0..orig_len {
                    let _ = read_f32(data, pos)?;
                }
            }
        }

        // Apply to self
        self.sq8_params = sq8_params;
        self.norms = norms;
        self.sq8_sums = sq8_sums;

        Ok(())
    }

    fn deserialize_legacy_upper_neighbors(
        &mut self,
        data: &[u8],
        pos: &mut usize,
    ) -> Result<(), String> {
        let upper_count = read_u64(data, pos)? as usize;
        let mut upper_neighbors =
            rustc_hash::FxHashMap::with_capacity_and_hasher(upper_count, rustc_hash::FxBuildHasher);
        for _ in 0..upper_count {
            let node_id = read_u32(data, pos)?;
            let num_levels = read_u8(data, pos)? as usize;
            let mut levels = Vec::with_capacity(num_levels);
            for _ in 0..num_levels {
                let count = read_u16(data, pos)? as usize;
                let mut neighbors = Vec::with_capacity(count);
                for _ in 0..count {
                    neighbors.push(read_u32(data, pos)?);
                }
                levels.push(neighbors);
            }
            upper_neighbors.insert(node_id, levels);
        }

        self.upper_neighbors.replace_owned(upper_neighbors);
        Ok(())
    }

    /// Serialize auxiliary state (mode, SQ8, upper neighbors).
    ///
    /// Does NOT include the header or raw node data -- those are handled
    /// separately (as_bytes() for save, from_bytes()/from_mmap() for load).
    /// Used by segment persistence v2 format to append auxiliary data after
    /// the raw node data in the segment file.
    #[must_use]
    pub fn serialize_auxiliary(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(256 + self.norms.len() * 4);

        // Mode and trained flag
        let mode_byte = u8::from(self.sq8);
        out.push(mode_byte);
        out.push(u8::from(self.sq8_trained));

        self.serialize_aux_body_without_upper(&mut out);
        self.serialize_legacy_upper_neighbors(&mut out);

        out
    }

    /// Restore auxiliary state into an existing NodeStorage that already has
    /// raw node data loaded (via from_bytes or from_mmap).
    ///
    /// This is the inverse of `serialize_auxiliary()`.
    pub fn deserialize_auxiliary(&mut self, data: &[u8]) -> Result<(), String> {
        let mut pos = 0;

        // Mode and trained flag
        let mode_byte = read_u8(data, &mut pos)?;
        let sq8 = match mode_byte {
            0 | 3 => false,
            1 => true,
            2 => return Err("PQ storage mode is no longer supported".to_string()),
            _ => return Err(format!("Invalid storage mode: {mode_byte}")),
        };
        let sq8_trained = read_u8(data, &mut pos)? != 0;

        self.sq8 = sq8;
        self.sq8_trained = sq8_trained;

        self.deserialize_aux_body_without_upper(data, &mut pos)?;
        self.deserialize_legacy_upper_neighbors(data, &mut pos)
    }

    #[must_use]
    pub fn serialize_segment_auxiliary(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(128 + self.norms.len() * 4);
        out.push(u8::from(self.sq8));
        out.push(u8::from(self.sq8_trained));
        self.serialize_aux_body_without_upper(&mut out);
        out
    }

    #[must_use]
    pub fn serialize_upper_neighbors_region(&self) -> Vec<u8> {
        self.upper_neighbors.serialize_region(self.len)
    }

    pub fn deserialize_segment_auxiliary(&mut self, data: &[u8]) -> Result<(), String> {
        let mut pos = 0;
        let mode_byte = read_u8(data, &mut pos)?;
        let sq8 = match mode_byte {
            0 | 3 => false,
            1 => true,
            2 => return Err("PQ storage mode is no longer supported".to_string()),
            _ => return Err(format!("Invalid storage mode: {mode_byte}")),
        };
        let sq8_trained = read_u8(data, &mut pos)? != 0;

        self.sq8 = sq8;
        self.sq8_trained = sq8_trained;
        self.deserialize_aux_body_without_upper(data, &mut pos)
    }

    pub fn deserialize_upper_neighbors_region(&mut self, data: &[u8]) -> Result<(), String> {
        self.upper_neighbors = UpperNeighborsStorage::deserialize_region_owned(
            data,
            self.len,
            self.max_neighbors_upper,
        )?;
        Ok(())
    }

    #[cfg(feature = "mmap")]
    pub fn mmap_upper_neighbors_region(&mut self, mmap: memmap2::Mmap) -> Result<(), String> {
        self.upper_neighbors =
            UpperNeighborsStorage::mmap_region(mmap, self.len, self.max_neighbors_upper)?;
        Ok(())
    }

    /// Serialize complete storage state to bytes
    ///
    /// Format:
    /// - Header: len, node_size, offsets, dimensions, max_neighbors (7 * u64)
    /// - Mode: u8 (0 = FullPrecision, 1 = SQ8)
    /// - SQ8 trained: u8
    /// - Raw node data: len * node_size bytes
    /// - If SQ8: scale, offset (2 * f32), norms (len * f32), sq8_sums (len * i32)
    /// - Upper neighbors count: u64
    /// - For each node with upper neighbors: node_id (u32), num_levels (u8), then for each level: count (u16), neighbors ([u32])
    /// Get total serialized length in bytes.
    #[must_use]
    pub fn serialized_len_full(&self) -> usize {
        let mut len = 7 * 8; // Header (7 * u64)
        len += 2; // sq8 (u8) + sq8_trained (u8)
        len += 8; // raw_len (u64)
        len += self.len * self.node_size; // raw node data

        // Auxiliary body (SQ8 params, norms, upper neighbors)
        // Calculating this exactly without serializing is a bit complex due to
        // variable-length encoding of upper neighbors.
        // For now, we just use the serialized buffer size as it's the easiest.
        let mut aux = Vec::new();
        self.serialize_aux_body_without_upper(&mut aux);
        self.serialize_legacy_upper_neighbors(&mut aux);
        len += aux.len();

        len
    }

    /// Serialize complete storage state directly to a writer.
    ///
    /// Avoids allocating a large intermediate buffer for the entire storage.
    pub fn write_full<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Header
        writer.write_all(&(self.len as u64).to_le_bytes())?;
        writer.write_all(&(self.node_size as u64).to_le_bytes())?;
        writer.write_all(&(self.neighbors_offset as u64).to_le_bytes())?;
        writer.write_all(&(self.vector_offset as u64).to_le_bytes())?;
        writer.write_all(&(self.metadata_offset as u64).to_le_bytes())?;
        writer.write_all(&(self.dimensions as u64).to_le_bytes())?;
        writer.write_all(&(self.max_neighbors as u64).to_le_bytes())?;

        // Mode and trained flag
        writer.write_all(&[u8::from(self.sq8)])?;
        writer.write_all(&[u8::from(self.sq8_trained)])?;

        // Raw node data
        let raw_data = self.as_bytes();
        writer.write_all(&(raw_data.len() as u64).to_le_bytes())?;
        writer.write_all(raw_data)?;

        // Auxiliary body (SQ8 params, norms, upper neighbors)
        // For now we use intermediate buffers for these smaller sections
        // to avoid duplicating the complex logic in serialize_aux_body_without_upper
        // and serialize_legacy_upper_neighbors.
        let mut aux = Vec::new();
        self.serialize_aux_body_without_upper(&mut aux);
        self.serialize_legacy_upper_neighbors(&mut aux);
        writer.write_all(&aux)?;

        Ok(())
    }

    pub fn serialize_full(&self) -> Vec<u8> {
        let mut out = Vec::new();

        // Header
        out.extend_from_slice(&(self.len as u64).to_le_bytes());
        out.extend_from_slice(&(self.node_size as u64).to_le_bytes());
        out.extend_from_slice(&(self.neighbors_offset as u64).to_le_bytes());
        out.extend_from_slice(&(self.vector_offset as u64).to_le_bytes());
        out.extend_from_slice(&(self.metadata_offset as u64).to_le_bytes());
        out.extend_from_slice(&(self.dimensions as u64).to_le_bytes());
        out.extend_from_slice(&(self.max_neighbors as u64).to_le_bytes());

        // Mode and trained flag
        let mode_byte = u8::from(self.sq8);
        out.push(mode_byte);
        out.push(u8::from(self.sq8_trained));

        // Raw node data
        let raw_data = self.as_bytes();
        out.extend_from_slice(&(raw_data.len() as u64).to_le_bytes());
        out.extend_from_slice(raw_data);

        // Auxiliary body (SQ8 params, norms, upper neighbors)
        self.serialize_aux_body_without_upper(&mut out);
        self.serialize_legacy_upper_neighbors(&mut out);

        out
    }

    /// Deserialize complete storage state from bytes
    ///
    /// Returns the deserialized storage and the number of bytes consumed.
    pub fn deserialize_full(data: &[u8]) -> Result<Self, String> {
        if data.len() < 58 {
            return Err("Data too short for header".to_string());
        }

        let mut pos = 0;

        // Read header
        let len = read_u64(data, &mut pos)? as usize;
        let node_size = read_u64(data, &mut pos)? as usize;
        let neighbors_offset = read_u64(data, &mut pos)? as usize;
        let vector_offset = read_u64(data, &mut pos)? as usize;
        let metadata_offset = read_u64(data, &mut pos)? as usize;
        let dimensions = read_u64(data, &mut pos)? as usize;
        let max_neighbors = read_u64(data, &mut pos)? as usize;

        // Safety caps to prevent corrupt files from exhausting memory
        const MAX_NODES: usize = 100_000_000; // 100M vectors
        if len > MAX_NODES {
            return Err(format!("Node count {len} exceeds safety cap ({MAX_NODES})"));
        }
        if node_size > 0 {
            let total = len
                .checked_mul(node_size)
                .ok_or_else(|| format!("Node data size overflow: {len} * {node_size}"))?;
            if total > data.len() {
                return Err(format!(
                    "Node data size {total} exceeds file size {}",
                    data.len()
                ));
            }
        }

        // Mode and trained flag
        let mode_byte = read_u8(data, &mut pos)?;
        let sq8 = match mode_byte {
            0 | 3 => false,
            1 => true,
            2 => return Err("PQ storage mode is no longer supported".to_string()),
            _ => return Err(format!("Invalid storage mode: {mode_byte}")),
        };
        let sq8_trained = read_u8(data, &mut pos)? != 0;

        // Raw node data
        let raw_len = read_u64(data, &mut pos)? as usize;
        if raw_len > data.len() {
            return Err(format!(
                "Raw data length {raw_len} exceeds file size {}",
                data.len()
            ));
        }
        let raw_data = read_bytes(data, &mut pos, raw_len)?.to_vec();

        // Construct storage from raw bytes
        let mut storage = Self::from_bytes(
            raw_data,
            len,
            node_size,
            neighbors_offset,
            vector_offset,
            metadata_offset,
            dimensions,
            max_neighbors,
        );

        // Set sq8 and trained before deserializing aux body
        storage.sq8 = sq8;
        storage.sq8_trained = sq8_trained;

        // Deserialize auxiliary body (SQ8 params, norms, upper neighbors)
        storage.deserialize_aux_body_without_upper(data, &mut pos)?;
        storage.deserialize_legacy_upper_neighbors(data, &mut pos)?;

        Ok(storage)
    }
}
