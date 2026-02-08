//! Serialization and deserialization for NodeStorage
//!
//! Supports both owned heap allocations and memory-mapped files.

use super::{NodeStorage, StorageBacking, StorageMode, CACHE_LINE};
use crate::compression::scalar::ScalarParams;
use rustc_hash::FxHashMap;
use std::alloc::Layout;
use std::ptr::NonNull;

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
                NonNull::new(raw).expect("Allocation should not return null")
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
            upper_neighbors: FxHashMap::default(),
            mode: StorageMode::FullPrecision, // Default to full precision for loaded data
            sq8_params: None,
            norms: Vec::new(),
            sq8_sums: Vec::new(),
            training_buffer: Vec::new(),
            sq8_trained: false,
            rabitq_params: None,
            rabitq_codes: Vec::new(),
            rabitq_metadata: Vec::new(),
            rabitq_originals: Vec::new(),
            rabitq_trained: false,
        }
    }

    /// Construct storage from memory-mapped file (for mmap loading)
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
            upper_neighbors: FxHashMap::default(),
            mode: StorageMode::FullPrecision,
            sq8_params: None,
            norms: Vec::new(),
            sq8_sums: Vec::new(),
            training_buffer: Vec::new(),
            sq8_trained: false,
            rabitq_params: None,
            rabitq_codes: Vec::new(),
            rabitq_metadata: Vec::new(),
            rabitq_originals: Vec::new(),
            rabitq_trained: false,
        }
    }

    /// Serialize auxiliary state (mode, SQ8, RaBitQ, upper neighbors).
    ///
    /// Does NOT include the header or raw node data — those are handled
    /// separately (as_bytes() for save, from_bytes()/from_mmap() for load).
    /// Used by segment persistence v2 format to append auxiliary data after
    /// the raw node data in the segment file.
    #[must_use]
    pub fn serialize_auxiliary(&self) -> Vec<u8> {
        let mut out = Vec::new();

        // Mode and trained flag
        let mode_byte: u8 = match self.mode {
            StorageMode::FullPrecision => 0,
            StorageMode::SQ8 => 1,
            StorageMode::RaBitQ => 3,
        };
        out.push(mode_byte);
        out.push(u8::from(self.sq8_trained));

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

        // RaBitQ state
        out.push(u8::from(self.rabitq_trained));
        if let Some(ref params) = self.rabitq_params {
            out.push(1);
            let param_bytes = params.serialize_params();
            out.extend_from_slice(&(param_bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(&param_bytes);
        } else {
            out.push(0);
        }
        out.extend_from_slice(&(self.rabitq_codes.len() as u64).to_le_bytes());
        for &word in &self.rabitq_codes {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out.extend_from_slice(&(self.rabitq_metadata.len() as u64).to_le_bytes());
        for &m in &self.rabitq_metadata {
            out.extend_from_slice(&m.to_le_bytes());
        }
        let originals_to_write = if self.mode == StorageMode::RaBitQ
            && !self.rabitq_trained
            && !self.training_buffer.is_empty()
        {
            &self.training_buffer
        } else {
            &self.rabitq_originals
        };
        out.extend_from_slice(&(originals_to_write.len() as u64).to_le_bytes());
        for &v in originals_to_write {
            out.extend_from_slice(&v.to_le_bytes());
        }

        // Upper neighbors
        out.extend_from_slice(&(self.upper_neighbors.len() as u64).to_le_bytes());
        for (&node_id, levels) in &self.upper_neighbors {
            out.extend_from_slice(&node_id.to_le_bytes());
            out.push(levels.len() as u8);
            for level_neighbors in levels {
                out.extend_from_slice(&(level_neighbors.len() as u16).to_le_bytes());
                for &neighbor in level_neighbors {
                    out.extend_from_slice(&neighbor.to_le_bytes());
                }
            }
        }

        out
    }

    /// Restore auxiliary state into an existing NodeStorage that already has
    /// raw node data loaded (via from_bytes or from_mmap).
    ///
    /// This is the inverse of `serialize_auxiliary()`.
    pub fn deserialize_auxiliary(&mut self, data: &[u8]) -> Result<(), String> {
        fn read_bytes<'a>(data: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], String> {
            if *pos + n > data.len() {
                return Err(format!(
                    "Auxiliary data too short: need {} bytes at position {}, have {}",
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

        const MAX_AUX_ELEMENTS: usize = 1_000_000_000;
        let mut pos = 0;

        // Mode and trained flag
        let mode_byte = read_u8(data, &mut pos)?;
        let mode = match mode_byte {
            0 => StorageMode::FullPrecision,
            1 => StorageMode::SQ8,
            2 => return Err("PQ storage mode is no longer supported".to_string()),
            3 => StorageMode::RaBitQ,
            _ => return Err(format!("Invalid storage mode: {mode_byte}")),
        };
        let sq8_trained = read_u8(data, &mut pos)? != 0;

        // SQ8 params
        let has_params = read_u8(data, &mut pos)? != 0;
        let sq8_params = if has_params {
            let scale = read_f32(data, &mut pos)?;
            let offset = read_f32(data, &mut pos)?;
            Some(ScalarParams {
                scale,
                offset,
                dimensions: self.dimensions,
            })
        } else {
            None
        };

        // Norms
        let norms_len = read_u64(data, &mut pos)? as usize;
        if norms_len > MAX_AUX_ELEMENTS {
            return Err(format!("Norms length {norms_len} exceeds safety cap"));
        }
        let mut norms = Vec::with_capacity(norms_len);
        for _ in 0..norms_len {
            norms.push(read_f32(data, &mut pos)?);
        }

        // SQ8 sums
        let sums_len = read_u64(data, &mut pos)? as usize;
        if sums_len > MAX_AUX_ELEMENTS {
            return Err(format!("SQ8 sums length {sums_len} exceeds safety cap"));
        }
        let mut sq8_sums = Vec::with_capacity(sums_len);
        for _ in 0..sums_len {
            sq8_sums.push(read_i32(data, &mut pos)?);
        }

        // Legacy PQ state (skip)
        let _pq_trained = read_u8(data, &mut pos)? != 0;
        let has_pq_params = read_u8(data, &mut pos)? != 0;
        if has_pq_params {
            let codebook_len = read_u64(data, &mut pos)? as usize;
            let _codebook_bytes = read_bytes(data, &mut pos, codebook_len)?;
        }
        let pq_codes_len = read_u64(data, &mut pos)? as usize;
        let _pq_codes = read_bytes(data, &mut pos, pq_codes_len)?;

        // RaBitQ state
        let (rabitq_trained, rabitq_params, rabitq_codes, rabitq_metadata, rabitq_originals) =
            if pos + 2 <= data.len() && mode == StorageMode::RaBitQ {
                let trained = read_u8(data, &mut pos)? != 0;
                let has_params = read_u8(data, &mut pos)? != 0;
                let params = if has_params {
                    let param_len = read_u64(data, &mut pos)? as usize;
                    let param_bytes = read_bytes(data, &mut pos, param_len)?;
                    Some(
                        crate::compression::rabitq::RaBitQParams::deserialize_params(param_bytes)
                            .map_err(|e| format!("Failed to deserialize RaBitQ params: {e}"))?,
                    )
                } else {
                    None
                };
                let codes_len = read_u64(data, &mut pos)? as usize;
                if codes_len > MAX_AUX_ELEMENTS {
                    return Err(format!(
                        "RaBitQ codes length {codes_len} exceeds safety cap"
                    ));
                }
                let mut codes = Vec::with_capacity(codes_len);
                for _ in 0..codes_len {
                    codes.push(read_u64(data, &mut pos)?);
                }
                let meta_len = read_u64(data, &mut pos)? as usize;
                if meta_len > MAX_AUX_ELEMENTS {
                    return Err(format!(
                        "RaBitQ metadata length {meta_len} exceeds safety cap"
                    ));
                }
                let mut metadata = Vec::with_capacity(meta_len);
                for _ in 0..meta_len {
                    metadata.push(read_f32(data, &mut pos)?);
                }
                let originals = if pos + 8 <= data.len() {
                    let orig_len = read_u64(data, &mut pos)? as usize;
                    if orig_len > MAX_AUX_ELEMENTS {
                        return Err(format!(
                            "RaBitQ originals length {orig_len} exceeds safety cap"
                        ));
                    }
                    let mut orig = Vec::with_capacity(orig_len);
                    for _ in 0..orig_len {
                        orig.push(read_f32(data, &mut pos)?);
                    }
                    orig
                } else {
                    Vec::new()
                };
                (trained, params, codes, metadata, originals)
            } else if pos + 2 <= data.len() && mode != StorageMode::RaBitQ {
                let _trained = read_u8(data, &mut pos)? != 0;
                let has_params = read_u8(data, &mut pos)? != 0;
                if has_params {
                    let param_len = read_u64(data, &mut pos)? as usize;
                    let _ = read_bytes(data, &mut pos, param_len)?;
                }
                let codes_len = read_u64(data, &mut pos)? as usize;
                if codes_len > MAX_AUX_ELEMENTS {
                    return Err(format!(
                        "RaBitQ codes length {codes_len} exceeds safety cap"
                    ));
                }
                for _ in 0..codes_len {
                    let _ = read_u64(data, &mut pos)?;
                }
                let meta_len = read_u64(data, &mut pos)? as usize;
                if meta_len > MAX_AUX_ELEMENTS {
                    return Err(format!(
                        "RaBitQ metadata length {meta_len} exceeds safety cap"
                    ));
                }
                for _ in 0..meta_len {
                    let _ = read_f32(data, &mut pos)?;
                }
                if pos + 8 <= data.len() {
                    let orig_len = read_u64(data, &mut pos)? as usize;
                    if orig_len > MAX_AUX_ELEMENTS {
                        return Err(format!(
                            "RaBitQ originals length {orig_len} exceeds safety cap"
                        ));
                    }
                    for _ in 0..orig_len {
                        let _ = read_f32(data, &mut pos)?;
                    }
                }
                (false, None, Vec::new(), Vec::new(), Vec::new())
            } else {
                (false, None, Vec::new(), Vec::new(), Vec::new())
            };

        // Upper neighbors
        let upper_count = read_u64(data, &mut pos)? as usize;
        let mut upper_neighbors: FxHashMap<u32, Vec<Vec<u32>>> =
            FxHashMap::with_capacity_and_hasher(upper_count, rustc_hash::FxBuildHasher);
        for _ in 0..upper_count {
            let node_id = read_u32(data, &mut pos)?;
            let num_levels = read_u8(data, &mut pos)? as usize;
            let mut levels = Vec::with_capacity(num_levels);
            for _ in 0..num_levels {
                let count = read_u16(data, &mut pos)? as usize;
                let mut neighbors = Vec::with_capacity(count);
                for _ in 0..count {
                    neighbors.push(read_u32(data, &mut pos)?);
                }
                levels.push(neighbors);
            }
            upper_neighbors.insert(node_id, levels);
        }

        // Apply to self
        self.mode = mode;
        self.sq8_params = sq8_params;
        self.norms = norms;
        self.sq8_sums = sq8_sums;
        self.sq8_trained = sq8_trained;
        self.upper_neighbors = upper_neighbors;
        self.rabitq_params = rabitq_params;
        self.rabitq_codes = rabitq_codes;
        self.rabitq_metadata = rabitq_metadata;
        self.rabitq_trained = rabitq_trained;
        if mode == StorageMode::RaBitQ && !rabitq_trained && !rabitq_originals.is_empty() {
            self.training_buffer = rabitq_originals;
        } else {
            self.rabitq_originals = rabitq_originals;
        }

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
        let mode_byte: u8 = match self.mode {
            StorageMode::FullPrecision => 0,
            StorageMode::SQ8 => 1,
            StorageMode::RaBitQ => 3,
        };
        out.push(mode_byte);
        out.push(u8::from(self.sq8_trained));

        // Raw node data
        let raw_data = self.as_bytes();
        out.extend_from_slice(&(raw_data.len() as u64).to_le_bytes());
        out.extend_from_slice(raw_data);

        // SQ8 params if present
        if let Some(ref params) = self.sq8_params {
            out.push(1); // has params
            out.extend_from_slice(&params.scale.to_le_bytes());
            out.extend_from_slice(&params.offset.to_le_bytes());
        } else {
            out.push(0); // no params
        }

        // Norms (only if SQ8 trained)
        out.extend_from_slice(&(self.norms.len() as u64).to_le_bytes());
        for &norm in &self.norms {
            out.extend_from_slice(&norm.to_le_bytes());
        }

        // SQ8 sums (only if SQ8 trained)
        out.extend_from_slice(&(self.sq8_sums.len() as u64).to_le_bytes());
        for &sum in &self.sq8_sums {
            out.extend_from_slice(&sum.to_le_bytes());
        }

        // Legacy PQ fields (write zeros for backward compatibility)
        out.push(0); // pq_trained = false
        out.push(0); // no PQ params
        out.extend_from_slice(&0u64.to_le_bytes()); // pq_codes length = 0

        // RaBitQ state
        out.push(u8::from(self.rabitq_trained));
        if let Some(ref params) = self.rabitq_params {
            out.push(1); // has RaBitQ params
            let param_bytes = params.serialize_params();
            out.extend_from_slice(&(param_bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(&param_bytes);
        } else {
            out.push(0);
        }
        // RaBitQ codes (stored as u64 words, serialized as little-endian bytes)
        out.extend_from_slice(&(self.rabitq_codes.len() as u64).to_le_bytes());
        for &word in &self.rabitq_codes {
            out.extend_from_slice(&word.to_le_bytes());
        }
        // RaBitQ metadata
        out.extend_from_slice(&(self.rabitq_metadata.len() as u64).to_le_bytes());
        for &m in &self.rabitq_metadata {
            out.extend_from_slice(&m.to_le_bytes());
        }
        // RaBitQ original vectors (needed for graph construction + rescore)
        // When untrained, training_buffer holds the vectors (originals is empty)
        let originals_to_write = if self.mode == StorageMode::RaBitQ
            && !self.rabitq_trained
            && !self.training_buffer.is_empty()
        {
            &self.training_buffer
        } else {
            &self.rabitq_originals
        };
        out.extend_from_slice(&(originals_to_write.len() as u64).to_le_bytes());
        for &v in originals_to_write {
            out.extend_from_slice(&v.to_le_bytes());
        }

        // Upper neighbors (HashMap - only stores nodes with upper levels)
        out.extend_from_slice(&(self.upper_neighbors.len() as u64).to_le_bytes());

        for (&node_id, levels) in &self.upper_neighbors {
            out.extend_from_slice(&node_id.to_le_bytes());
            out.push(levels.len() as u8); // num levels (excluding level 0)
            for level_neighbors in levels {
                out.extend_from_slice(&(level_neighbors.len() as u16).to_le_bytes());
                for &neighbor in level_neighbors {
                    out.extend_from_slice(&neighbor.to_le_bytes());
                }
            }
        }

        out
    }

    /// Deserialize complete storage state from bytes
    ///
    /// Returns the deserialized storage and the number of bytes consumed.
    pub fn deserialize_full(data: &[u8]) -> Result<Self, String> {
        // Helper to read bytes safely
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
        let mode = match mode_byte {
            0 => StorageMode::FullPrecision,
            1 => StorageMode::SQ8,
            2 => return Err("PQ storage mode is no longer supported".to_string()),
            3 => StorageMode::RaBitQ,
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

        // SQ8 params
        let has_params = read_u8(data, &mut pos)? != 0;
        let sq8_params = if has_params {
            let scale = read_f32(data, &mut pos)?;
            let offset = read_f32(data, &mut pos)?;
            Some(ScalarParams {
                scale,
                offset,
                dimensions,
            })
        } else {
            None
        };

        // Safety caps for auxiliary arrays
        const MAX_AUX_ELEMENTS: usize = 1_000_000_000; // 1B elements

        // Norms
        let norms_len = read_u64(data, &mut pos)? as usize;
        if norms_len > MAX_AUX_ELEMENTS {
            return Err(format!("Norms length {norms_len} exceeds safety cap"));
        }
        let mut norms = Vec::with_capacity(norms_len);
        for _ in 0..norms_len {
            norms.push(read_f32(data, &mut pos)?);
        }

        // SQ8 sums
        let sums_len = read_u64(data, &mut pos)? as usize;
        if sums_len > MAX_AUX_ELEMENTS {
            return Err(format!("SQ8 sums length {sums_len} exceeds safety cap"));
        }
        let mut sq8_sums = Vec::with_capacity(sums_len);
        for _ in 0..sums_len {
            sq8_sums.push(read_i32(data, &mut pos)?);
        }

        // Legacy PQ state (skip but consume bytes for backward compatibility)
        let _pq_trained = read_u8(data, &mut pos)? != 0;
        let has_pq_params = read_u8(data, &mut pos)? != 0;
        if has_pq_params {
            let codebook_len = read_u64(data, &mut pos)? as usize;
            let _codebook_bytes = read_bytes(data, &mut pos, codebook_len)?;
        }
        // Skip PQ codes
        let pq_codes_len = read_u64(data, &mut pos)? as usize;
        let _pq_codes = read_bytes(data, &mut pos, pq_codes_len)?;

        // RaBitQ state (only present if data remains before upper neighbors)
        let (rabitq_trained, rabitq_params, rabitq_codes, rabitq_metadata, rabitq_originals) =
            if pos + 2 <= data.len() && mode == StorageMode::RaBitQ {
                let trained = read_u8(data, &mut pos)? != 0;
                let has_params = read_u8(data, &mut pos)? != 0;
                let params = if has_params {
                    let param_len = read_u64(data, &mut pos)? as usize;
                    let param_bytes = read_bytes(data, &mut pos, param_len)?;
                    Some(
                        crate::compression::rabitq::RaBitQParams::deserialize_params(param_bytes)
                            .map_err(|e| format!("Failed to deserialize RaBitQ params: {e}"))?,
                    )
                } else {
                    None
                };
                // Codes are stored as u64 words (len = number of u64 words)
                let codes_len = read_u64(data, &mut pos)? as usize;
                if codes_len > MAX_AUX_ELEMENTS {
                    return Err(format!(
                        "RaBitQ codes length {codes_len} exceeds safety cap"
                    ));
                }
                let mut codes = Vec::with_capacity(codes_len);
                for _ in 0..codes_len {
                    codes.push(read_u64(data, &mut pos)?);
                }
                let meta_len = read_u64(data, &mut pos)? as usize;
                if meta_len > MAX_AUX_ELEMENTS {
                    return Err(format!(
                        "RaBitQ metadata length {meta_len} exceeds safety cap"
                    ));
                }
                let mut metadata = Vec::with_capacity(meta_len);
                for _ in 0..meta_len {
                    metadata.push(read_f32(data, &mut pos)?);
                }
                // Original vectors (may not be present in older files)
                let originals = if pos + 8 <= data.len() {
                    let orig_len = read_u64(data, &mut pos)? as usize;
                    if orig_len > MAX_AUX_ELEMENTS {
                        return Err(format!(
                            "RaBitQ originals length {orig_len} exceeds safety cap"
                        ));
                    }
                    let mut orig = Vec::with_capacity(orig_len);
                    for _ in 0..orig_len {
                        orig.push(read_f32(data, &mut pos)?);
                    }
                    orig
                } else {
                    Vec::new()
                };
                (trained, params, codes, metadata, originals)
            } else if pos + 2 <= data.len() && mode != StorageMode::RaBitQ {
                // Non-RaBitQ mode: still read past the rabitq fields for forward compat
                let _trained = read_u8(data, &mut pos)? != 0;
                let has_params = read_u8(data, &mut pos)? != 0;
                let _params = if has_params {
                    let param_len = read_u64(data, &mut pos)? as usize;
                    let _param_bytes = read_bytes(data, &mut pos, param_len)?;
                    true
                } else {
                    false
                };
                // Skip codes (u64 words)
                let codes_len = read_u64(data, &mut pos)? as usize;
                if codes_len > MAX_AUX_ELEMENTS {
                    return Err(format!(
                        "RaBitQ codes length {codes_len} exceeds safety cap"
                    ));
                }
                for _ in 0..codes_len {
                    let _ = read_u64(data, &mut pos)?;
                }
                // Skip metadata
                let meta_len = read_u64(data, &mut pos)? as usize;
                if meta_len > MAX_AUX_ELEMENTS {
                    return Err(format!(
                        "RaBitQ metadata length {meta_len} exceeds safety cap"
                    ));
                }
                for _ in 0..meta_len {
                    let _ = read_f32(data, &mut pos)?;
                }
                // Skip originals if present
                if pos + 8 <= data.len() {
                    let orig_len = read_u64(data, &mut pos)? as usize;
                    if orig_len > MAX_AUX_ELEMENTS {
                        return Err(format!(
                            "RaBitQ originals length {orig_len} exceeds safety cap"
                        ));
                    }
                    for _ in 0..orig_len {
                        let _ = read_f32(data, &mut pos)?;
                    }
                }
                (false, None, Vec::new(), Vec::new(), Vec::new())
            } else {
                // Old format without RaBitQ fields
                (false, None, Vec::new(), Vec::new(), Vec::new())
            };

        // Upper neighbors (HashMap - only nodes with upper levels)
        let upper_count = read_u64(data, &mut pos)? as usize;
        let mut upper_neighbors: FxHashMap<u32, Vec<Vec<u32>>> =
            FxHashMap::with_capacity_and_hasher(upper_count, rustc_hash::FxBuildHasher);

        for _ in 0..upper_count {
            let node_id = read_u32(data, &mut pos)?;
            let num_levels = read_u8(data, &mut pos)? as usize;

            let mut levels = Vec::with_capacity(num_levels);
            for _ in 0..num_levels {
                let count = read_u16(data, &mut pos)? as usize;
                let mut neighbors = Vec::with_capacity(count);
                for _ in 0..count {
                    neighbors.push(read_u32(data, &mut pos)?);
                }
                levels.push(neighbors);
            }
            upper_neighbors.insert(node_id, levels);
        }

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

        // Restore SQ8 state
        storage.mode = mode;
        storage.sq8_params = sq8_params;
        storage.norms = norms;
        storage.sq8_sums = sq8_sums;
        storage.sq8_trained = sq8_trained;
        storage.upper_neighbors = upper_neighbors;

        // Restore RaBitQ state
        storage.rabitq_params = rabitq_params;
        storage.rabitq_codes = rabitq_codes;
        storage.rabitq_metadata = rabitq_metadata;
        storage.rabitq_trained = rabitq_trained;
        // When untrained, originals hold pre-training vectors → restore to training_buffer
        if mode == StorageMode::RaBitQ && !rabitq_trained && !rabitq_originals.is_empty() {
            storage.training_buffer = rabitq_originals;
        } else {
            storage.rabitq_originals = rabitq_originals;
        }

        Ok(storage)
    }
}
