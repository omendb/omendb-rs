//! Storage for original multi-vector tokens (used for MaxSim reranking).

use std::hash::BuildHasher;

use serde::{Deserialize, Serialize};

/// Storage for original multi-vector documents.
///
/// Stores all token vectors in a flat buffer with per-document offsets.
/// This is used for MaxSim reranking after FDE-based HNSW search.
///
/// # Memory Layout
///
/// ```text
/// vectors: [doc0_tok0, doc0_tok1, ..., doc1_tok0, doc1_tok1, ...]
/// offsets: [(start0, count0), (start1, count1), ...]
/// ```
///
/// Each slot corresponds to a document, with tokens stored contiguously.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultiVecStorage {
    /// Flat storage for all token vectors (concatenated).
    vectors: Vec<f32>,
    /// Per-document offsets: (start_index, token_count).
    /// start_index is the position in `vectors` array (not byte offset).
    offsets: Vec<(u32, u16)>,
    /// Token embedding dimension.
    dim: usize,
}

impl MultiVecStorage {
    /// Create a new empty storage for the given token dimension.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            vectors: Vec::new(),
            offsets: Vec::new(),
            dim,
        }
    }

    /// Create storage with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(dim: usize, num_docs: usize, avg_tokens_per_doc: usize) -> Self {
        Self {
            vectors: Vec::with_capacity(num_docs * avg_tokens_per_doc * dim),
            offsets: Vec::with_capacity(num_docs),
            dim,
        }
    }

    /// Add a multi-vector document and return its slot ID.
    ///
    /// # Errors
    ///
    /// Returns error if slot count exceeds u32, offset exceeds u32, or token count exceeds u16.
    pub fn add(&mut self, tokens: &[&[f32]]) -> Result<u32, &'static str> {
        let slot = u32::try_from(self.offsets.len())
            .map_err(|_| "Multi-vector slot overflow (>4B docs)")?;
        let start = u32::try_from(self.vectors.len() / self.dim)
            .map_err(|_| "Multi-vector offset overflow")?;
        let count = u16::try_from(tokens.len())
            .map_err(|_| "Token count overflow (>65K tokens per doc)")?;

        // Copy all tokens to flat storage
        for token in tokens {
            debug_assert_eq!(
                token.len(),
                self.dim,
                "Token dimension mismatch: expected {}, got {}",
                self.dim,
                token.len()
            );
            self.vectors.extend_from_slice(token);
        }

        self.offsets.push((start, count));
        Ok(slot)
    }

    /// Update tokens for an existing slot.
    ///
    /// # Errors
    /// Returns error if slot is out of bounds or if updating would cause offset/count overflow.
    ///
    /// NOTE: This is O(N) where N is total number of tokens after the updated slot,
    /// as it must shift subsequent token vectors in the flat buffer.
    pub fn update(&mut self, slot: u32, tokens: &[&[f32]]) -> Result<(), &'static str> {
        let slot_idx = slot as usize;
        if slot_idx >= self.offsets.len() {
            return Err("Slot index out of bounds");
        }

        let (old_start, old_count) = self.offsets[slot_idx];
        let new_count = u16::try_from(tokens.len())
            .map_err(|_| "Token count overflow (>65K tokens per doc)")?;

        let old_tokens_len = old_count as usize * self.dim;
        let new_tokens_len = new_count as usize * self.dim;
        let diff = (new_tokens_len as isize) - (old_tokens_len as isize);

        let start_idx = old_start as usize * self.dim;
        let end_idx = start_idx + old_tokens_len;

        if diff != 0 {
            // Shift subsequent vectors
            if diff > 0 {
                let additional = diff as usize;
                self.vectors.reserve(additional);
                unsafe {
                    self.vectors.set_len(self.vectors.len() + additional);
                }
                let src = end_idx;
                let dst = end_idx + additional;
                let len = self.vectors.len() - dst;
                if len > 0 {
                    self.vectors.copy_within(src..src + len, dst);
                }
            } else {
                let reduction = (-diff) as usize;
                let src = end_idx;
                let dst = end_idx - reduction;
                let len = self.vectors.len() - src;
                if len > 0 {
                    self.vectors.copy_within(src..src + len, dst);
                }
                self.vectors.truncate(self.vectors.len() - reduction);
            }

            // Update subsequent offsets
            let offset_diff = (diff / self.dim as isize) as i32;
            for i in (slot_idx + 1)..self.offsets.len() {
                let (start, _count) = &mut self.offsets[i];
                *start = (*start as i32 + offset_diff) as u32;
            }
        }

        // Copy new tokens
        for (i, token) in tokens.iter().enumerate() {
            let offset = start_idx + i * self.dim;
            self.vectors[offset..offset + self.dim].copy_from_slice(token);
        }

        self.offsets[slot_idx] = (old_start, new_count);
        Ok(())
    }

    /// Ensure capacity for up to `num_docs` slots.
    /// New slots are initialized with zero tokens.
    pub fn reserve_slots(&mut self, num_docs: usize) {
        if num_docs > self.offsets.len() {
            let current_len = self.vectors.len() / self.dim;
            self.offsets.resize(num_docs, (current_len as u32, 0));
        }
    }

    /// Get tokens for a document by slot ID.
    ///
    /// Returns an iterator over token slices.
    #[must_use]
    pub fn get(&self, slot: u32) -> Option<impl Iterator<Item = &[f32]>> {
        let (start, count) = *self.offsets.get(slot as usize)?;
        let start_idx = start as usize * self.dim;

        Some((0..count as usize).map(move |i| {
            let offset = start_idx + i * self.dim;
            &self.vectors[offset..offset + self.dim]
        }))
    }

    /// Get tokens as a Vec of slices (convenience method for reranking).
    #[must_use]
    pub fn get_tokens(&self, slot: u32) -> Option<Vec<&[f32]>> {
        self.get(slot).map(std::iter::Iterator::collect)
    }

    /// Number of documents stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Get the token dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get total number of tokens stored across all documents.
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.vectors.len() / self.dim
    }

    /// Get memory usage in bytes (approximate).
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.vectors.len() * std::mem::size_of::<f32>()
            + self.offsets.len() * std::mem::size_of::<(u32, u16)>()
    }

    /// Serialize vectors to bytes for persistence.
    ///
    /// Layout: flat f32 array in little-endian format.
    #[must_use]
    pub fn vectors_to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.vectors.len() * 4);
        for &val in &self.vectors {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Serialize offsets to bytes for persistence.
    ///
    /// Layout: [(start: u32, count: u16), ...] packed as 6 bytes each.
    #[must_use]
    pub fn offsets_to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.offsets.len() * 6);
        for &(start, count) in &self.offsets {
            bytes.extend_from_slice(&start.to_le_bytes());
            bytes.extend_from_slice(&count.to_le_bytes());
        }
        bytes
    }

    /// Reconstruct storage from persisted bytes.
    ///
    /// # Arguments
    ///
    /// * `vec_bytes` - Serialized vectors from `vectors_to_bytes()`
    /// * `off_bytes` - Serialized offsets from `offsets_to_bytes()`
    /// * `dim` - Token embedding dimension
    ///
    /// # Errors
    ///
    /// Returns error if bytes are malformed or don't match expected layout.
    pub fn from_bytes(vec_bytes: &[u8], off_bytes: &[u8], dim: usize) -> std::io::Result<Self> {
        // Validate vector bytes length
        if !vec_bytes.len().is_multiple_of(4) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Vector bytes not aligned to f32",
            ));
        }

        // Validate offset bytes length
        if !off_bytes.len().is_multiple_of(6) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Offset bytes not aligned to (u32, u16)",
            ));
        }

        // Parse vectors
        let vectors: Vec<f32> = vec_bytes
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact(4) guarantees 4-byte slice"),
                )
            })
            .collect();

        // Parse offsets
        let offsets: Vec<(u32, u16)> = off_bytes
            .chunks_exact(6)
            .map(|chunk| {
                let start = u32::from_le_bytes(
                    chunk[0..4]
                        .try_into()
                        .expect("chunks_exact(6) guarantees at least 4 bytes"),
                );
                let count = u16::from_le_bytes(
                    chunk[4..6]
                        .try_into()
                        .expect("chunks_exact(6) guarantees 6 bytes"),
                );
                (start, count)
            })
            .collect();

        // Validate vector count matches dimension
        if !vectors.is_empty() && !vectors.len().is_multiple_of(dim) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Vector count {} not divisible by dimension {dim}",
                    vectors.len()
                ),
            ));
        }

        Ok(Self {
            vectors,
            offsets,
            dim,
        })
    }

    /// Build helper storage from slot-indexed token payloads.
    #[must_use]
    pub fn from_slot_tokens(dim: usize, payloads: &[(u32, Vec<Vec<f32>>)]) -> Self {
        if payloads.is_empty() {
            return Self::new(dim);
        }

        let max_slot = payloads
            .iter()
            .map(|(slot, _)| *slot as usize)
            .max()
            .unwrap_or(0);
        let mut storage = Self {
            vectors: Vec::new(),
            offsets: vec![(0, 0); max_slot + 1],
            dim,
        };

        let mut sorted = payloads.to_vec();
        sorted.sort_unstable_by_key(|(slot, _)| *slot);

        for (slot, tokens) in sorted {
            let start = (storage.vectors.len() / dim) as u32;
            let count = u16::try_from(tokens.len()).expect("token count fits u16");
            for token in &tokens {
                debug_assert_eq!(token.len(), dim);
                storage.vectors.extend_from_slice(token);
            }
            storage.offsets[slot as usize] = (start, count);
        }

        storage
    }

    /// Remap slots after RecordStore compaction, reclaiming orphaned token data.
    ///
    /// `old_to_new` maps old_slot -> new_slot for live records only.
    /// Deleted slots are removed and their token vectors are freed.
    pub fn compact<S: BuildHasher>(&mut self, old_to_new: &std::collections::HashMap<u32, u32, S>) {
        if old_to_new.is_empty() {
            self.offsets.clear();
            self.vectors.clear();
            return;
        }

        let max_new_slot = old_to_new.values().copied().max().unwrap_or(0) as usize;
        let old_offsets = std::mem::take(&mut self.offsets);
        let old_vectors = std::mem::take(&mut self.vectors);
        let mut new_offsets = vec![(0u32, 0u16); max_new_slot + 1];
        let mut new_vectors = Vec::new();

        // Collect live slots sorted by new slot ID for sequential writes
        let mut live: Vec<(u32, u32, u16)> = Vec::with_capacity(old_to_new.len());
        for (old_slot, &(start, count)) in old_offsets.iter().enumerate() {
            if let Some(&new_slot) = old_to_new.get(&(old_slot as u32)) {
                live.push((new_slot, start, count));
            }
        }
        live.sort_unstable_by_key(|&(new_slot, _, _)| new_slot);

        // Copy live token data into compacted buffer
        for (new_slot, old_start, count) in live {
            let new_start = (new_vectors.len() / self.dim) as u32;
            let src_begin = old_start as usize * self.dim;
            let src_end = src_begin + count as usize * self.dim;
            new_vectors.extend_from_slice(&old_vectors[src_begin..src_end]);
            new_offsets[new_slot as usize] = (new_start, count);
        }

        self.offsets = new_offsets;
        self.vectors = new_vectors;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_storage() {
        let storage = MultiVecStorage::new(128);
        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());
        assert_eq!(storage.dim(), 128);
    }

    #[test]
    fn test_add_single_doc() {
        let mut storage = MultiVecStorage::new(4);
        let tokens: Vec<&[f32]> = vec![&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]];

        let slot = storage.add(&tokens).unwrap();

        assert_eq!(slot, 0);
        assert_eq!(storage.len(), 1);
        assert_eq!(storage.total_tokens(), 2);
    }

    #[test]
    fn test_roundtrip() {
        let mut storage = MultiVecStorage::new(4);
        let doc1: Vec<&[f32]> = vec![&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]];
        let doc2: Vec<&[f32]> = vec![
            &[9.0, 10.0, 11.0, 12.0],
            &[13.0, 14.0, 15.0, 16.0],
            &[17.0, 18.0, 19.0, 20.0],
        ];

        let slot1 = storage.add(&doc1).unwrap();
        let slot2 = storage.add(&doc2).unwrap();

        // Verify slot1
        let retrieved1: Vec<&[f32]> = storage.get(slot1).unwrap().collect();
        assert_eq!(retrieved1.len(), 2);
        assert_eq!(retrieved1[0], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(retrieved1[1], &[5.0, 6.0, 7.0, 8.0]);

        // Verify slot2
        let retrieved2: Vec<&[f32]> = storage.get(slot2).unwrap().collect();
        assert_eq!(retrieved2.len(), 3);
        assert_eq!(retrieved2[0], &[9.0, 10.0, 11.0, 12.0]);
        assert_eq!(retrieved2[1], &[13.0, 14.0, 15.0, 16.0]);
        assert_eq!(retrieved2[2], &[17.0, 18.0, 19.0, 20.0]);
    }

    #[test]
    fn test_get_invalid_slot() {
        let storage = MultiVecStorage::new(4);
        assert!(storage.get(0).is_none());
        assert!(storage.get(100).is_none());
    }

    #[test]
    fn test_empty_doc() {
        let mut storage = MultiVecStorage::new(4);
        let empty: Vec<&[f32]> = vec![];

        let slot = storage.add(&empty).unwrap();

        assert_eq!(slot, 0);
        assert_eq!(storage.len(), 1);
        let retrieved: Vec<&[f32]> = storage.get(slot).unwrap().collect();
        assert!(retrieved.is_empty());
    }

    #[test]
    fn test_get_tokens_convenience() {
        let mut storage = MultiVecStorage::new(4);
        let tokens: Vec<&[f32]> = vec![&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]];
        let slot = storage.add(&tokens).unwrap();

        let retrieved = storage.get_tokens(slot).unwrap();
        assert_eq!(retrieved.len(), 2);
        assert_eq!(retrieved[0], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_memory_bytes() {
        let mut storage = MultiVecStorage::new(128);
        let token = vec![0.0f32; 128];
        let tokens: Vec<&[f32]> = vec![&token; 100]; // 100 tokens

        storage.add(&tokens).unwrap();

        // 100 tokens * 128 dims * 4 bytes = 51,200 bytes for vectors
        // 1 doc * size_of::<(u32, u16)>() bytes for offset (8 bytes due to alignment)
        let expected = 100 * 128 * 4 + std::mem::size_of::<(u32, u16)>();
        assert_eq!(storage.memory_bytes(), expected);
    }

    #[test]
    fn test_compact_remaps_slots_and_reclaims_memory() {
        let mut storage = MultiVecStorage::new(4);

        // Add 5 documents
        let doc0: Vec<&[f32]> = vec![&[1.0, 2.0, 3.0, 4.0]];
        let doc1: Vec<&[f32]> = vec![&[5.0, 6.0, 7.0, 8.0]];
        let doc2: Vec<&[f32]> = vec![&[9.0, 10.0, 11.0, 12.0]];
        let doc3: Vec<&[f32]> = vec![&[13.0, 14.0, 15.0, 16.0]];
        let doc4: Vec<&[f32]> = vec![&[17.0, 18.0, 19.0, 20.0]];

        storage.add(&doc0).unwrap(); // slot 0
        storage.add(&doc1).unwrap(); // slot 1
        storage.add(&doc2).unwrap(); // slot 2
        storage.add(&doc3).unwrap(); // slot 3
        storage.add(&doc4).unwrap(); // slot 4

        assert_eq!(storage.total_tokens(), 5);

        // Simulate deleting slots 1 and 3 (keep 0, 2, 4)
        // After compaction: 0->0, 2->1, 4->2
        let mut old_to_new = std::collections::HashMap::new();
        old_to_new.insert(0u32, 0u32);
        old_to_new.insert(2u32, 1u32);
        old_to_new.insert(4u32, 2u32);

        storage.compact(&old_to_new);

        // Verify remapped slots
        assert_eq!(storage.len(), 3);

        // Verify orphaned token data was reclaimed
        assert_eq!(storage.total_tokens(), 3);

        // New slot 0 should have doc0's tokens
        let tokens0: Vec<&[f32]> = storage.get(0).unwrap().collect();
        assert_eq!(tokens0[0], &[1.0, 2.0, 3.0, 4.0]);

        // New slot 1 should have doc2's tokens (was slot 2)
        let tokens1: Vec<&[f32]> = storage.get(1).unwrap().collect();
        assert_eq!(tokens1[0], &[9.0, 10.0, 11.0, 12.0]);

        // New slot 2 should have doc4's tokens (was slot 4)
        let tokens2: Vec<&[f32]> = storage.get(2).unwrap().collect();
        assert_eq!(tokens2[0], &[17.0, 18.0, 19.0, 20.0]);
    }

    #[test]
    fn test_compact_empty() {
        let mut storage = MultiVecStorage::new(4);
        let doc: Vec<&[f32]> = vec![&[1.0, 2.0, 3.0, 4.0]];
        storage.add(&doc).unwrap();

        // All deleted - empty mapping
        let old_to_new: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        storage.compact(&old_to_new);

        assert!(storage.is_empty());
        assert_eq!(storage.total_tokens(), 0);
    }

    #[test]
    fn test_serialization_empty() {
        let storage = MultiVecStorage::new(128);
        let vec_bytes = storage.vectors_to_bytes();
        let off_bytes = storage.offsets_to_bytes();

        assert!(vec_bytes.is_empty());
        assert!(off_bytes.is_empty());

        let restored = MultiVecStorage::from_bytes(&vec_bytes, &off_bytes, 128).unwrap();
        assert!(restored.is_empty());
        assert_eq!(restored.dim(), 128);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut storage = MultiVecStorage::new(4);
        let doc1: Vec<&[f32]> = vec![&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]];
        let doc2: Vec<&[f32]> = vec![
            &[9.0, 10.0, 11.0, 12.0],
            &[13.0, 14.0, 15.0, 16.0],
            &[17.0, 18.0, 19.0, 20.0],
        ];

        storage.add(&doc1).unwrap();
        storage.add(&doc2).unwrap();

        // Serialize
        let vec_bytes = storage.vectors_to_bytes();
        let off_bytes = storage.offsets_to_bytes();

        // Expected sizes
        assert_eq!(vec_bytes.len(), 5 * 4 * 4); // 5 tokens * 4 dims * 4 bytes
        assert_eq!(off_bytes.len(), 2 * 6); // 2 docs * 6 bytes

        // Deserialize
        let restored = MultiVecStorage::from_bytes(&vec_bytes, &off_bytes, 4).unwrap();

        // Verify
        assert_eq!(restored.len(), 2);
        assert_eq!(restored.total_tokens(), 5);
        assert_eq!(restored.dim(), 4);

        // Verify doc1 tokens
        let tokens1: Vec<&[f32]> = restored.get(0).unwrap().collect();
        assert_eq!(tokens1.len(), 2);
        assert_eq!(tokens1[0], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tokens1[1], &[5.0, 6.0, 7.0, 8.0]);

        // Verify doc2 tokens
        let tokens2: Vec<&[f32]> = restored.get(1).unwrap().collect();
        assert_eq!(tokens2.len(), 3);
        assert_eq!(tokens2[0], &[9.0, 10.0, 11.0, 12.0]);
        assert_eq!(tokens2[1], &[13.0, 14.0, 15.0, 16.0]);
        assert_eq!(tokens2[2], &[17.0, 18.0, 19.0, 20.0]);
    }

    #[test]
    fn test_serialization_invalid_vec_bytes() {
        // Not divisible by 4
        let result = MultiVecStorage::from_bytes(&[0, 1, 2], &[], 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialization_invalid_off_bytes() {
        // Not divisible by 6
        let result = MultiVecStorage::from_bytes(&[], &[0, 1, 2, 3, 4], 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialization_dimension_mismatch() {
        // 5 floats not divisible by dim=4
        let vec_bytes: Vec<u8> = (0..5).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let result = MultiVecStorage::from_bytes(&vec_bytes, &[], 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialization_large_store() {
        let mut storage = MultiVecStorage::new(128);

        // Add 100 documents with varying token counts
        for i in 0..100 {
            let num_tokens = (i % 10) + 1; // 1-10 tokens
            let tokens: Vec<Vec<f32>> = (0..num_tokens)
                .map(|t| vec![(i * num_tokens + t) as f32; 128])
                .collect();
            let token_refs: Vec<&[f32]> = tokens.iter().map(std::vec::Vec::as_slice).collect();
            storage.add(&token_refs).unwrap();
        }

        // Serialize
        let vec_bytes = storage.vectors_to_bytes();
        let off_bytes = storage.offsets_to_bytes();

        // Deserialize
        let restored = MultiVecStorage::from_bytes(&vec_bytes, &off_bytes, 128).unwrap();

        // Verify
        assert_eq!(restored.len(), 100);
        assert_eq!(restored.total_tokens(), storage.total_tokens());

        // Spot check a few documents
        for slot in [0, 50, 99] {
            let orig_tokens: Vec<&[f32]> = storage.get(slot).unwrap().collect();
            let restored_tokens: Vec<&[f32]> = restored.get(slot).unwrap().collect();
            assert_eq!(orig_tokens.len(), restored_tokens.len());
            for (orig, rest) in orig_tokens.iter().zip(restored_tokens.iter()) {
                assert_eq!(*orig, *rest);
            }
        }
    }
}
