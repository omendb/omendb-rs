// Vector storage for HNSW (quantized or full precision)
//
// Supports:
// - Full precision f32 vectors with pre-computed norms
// - SQ8 scalar quantization (4x compression, ~97% recall)

use serde::{Deserialize, Serialize};

use crate::compression::scalar::QueryPrep;
use crate::compression::ScalarParams;
use crate::distance::dot_product;

/// Vector storage (quantized or full precision)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VectorStorage {
    /// Full precision f32 vectors - FLAT CONTIGUOUS STORAGE
    ///
    /// Memory: dimensions * 4 bytes per vector + 4 bytes for norm
    /// Example: 1536D = 6148 bytes per vector
    ///
    /// Vectors stored in single contiguous array for cache efficiency.
    /// Access: vectors[id * dimensions..(id + 1) * dimensions]
    ///
    /// Norms (||v||^2) are stored separately for L2 decomposition optimization:
    /// ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    /// This reduces L2 distance from 3N FLOPs to 2N+3 FLOPs (~7% faster).
    FullPrecision {
        /// Flat contiguous vector data (all vectors concatenated)
        vectors: Vec<f32>,
        /// Pre-computed squared norms (||v||^2) for L2 decomposition
        norms: Vec<f32>,
        /// Number of vectors stored
        count: usize,
        /// Dimensions per vector
        dimensions: usize,
    },

    /// Scalar quantized vectors (SQ8) - 4x compression, ~97% recall, 2-3x faster
    ///
    /// Memory: 1x (quantized only, no originals stored)
    /// Trade-off: 4x RAM savings for ~3% recall loss
    ///
    /// Uses uniform min/max scaling with integer SIMD distance computation.
    /// Lazy training: Buffers first 256 vectors, then trains and quantizes.
    ///
    /// Note: No rescore support - originals not stored to save memory.
    ScalarQuantized {
        /// Trained quantization parameters (global scale/offset)
        params: ScalarParams,

        /// Quantized vectors as flat contiguous u8 array
        /// Empty until training completes (after 256 vectors)
        /// Access: quantized[id * dimensions..(id + 1) * dimensions]
        quantized: Vec<u8>,

        /// Pre-computed squared norms of dequantized vectors for L2 decomposition
        /// ||dequant(q)||^2 = sum((code[d] * scale + offset)^2)
        /// Enables fast distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        norms: Vec<f32>,

        /// Pre-computed sums of quantized values for fast integer dot product
        /// sum = sum(quantized[d])
        sums: Vec<i32>,

        /// Buffer for training vectors (cleared after training)
        /// During training phase, stores f32 vectors until we have enough to train
        training_buffer: Vec<f32>,

        /// Number of vectors stored
        count: usize,

        /// Vector dimensions
        dimensions: usize,

        /// Whether quantization parameters have been trained
        /// Training happens automatically after 256 vectors are inserted
        trained: bool,
    },
}

impl VectorStorage {
    /// Create empty full precision storage
    #[must_use]
    pub fn new_full_precision(dimensions: usize) -> Self {
        Self::FullPrecision {
            vectors: Vec::new(),
            norms: Vec::new(),
            count: 0,
            dimensions,
        }
    }

    /// Create empty SQ8 (Scalar Quantized) storage
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality
    ///
    /// # Performance
    /// - Search: 2-3x faster than f32 (integer SIMD)
    /// - Memory: 4x smaller (quantized only, no originals)
    /// - Recall: ~97% (no rescore support)
    ///
    /// # Lazy Training
    /// Quantization parameters are trained automatically after 256 vectors.
    /// Before training completes, search falls back to f32 distance on
    /// the training buffer.
    #[must_use]
    pub fn new_sq8_quantized(dimensions: usize) -> Self {
        Self::ScalarQuantized {
            params: ScalarParams::uninitialized(dimensions),
            quantized: Vec::new(),
            norms: Vec::new(),
            sums: Vec::new(),
            training_buffer: Vec::new(),
            count: 0,
            dimensions,
            trained: false,
        }
    }

    /// Check if this storage uses asymmetric search (SQ8)
    ///
    /// SQ8 uses direct asymmetric L2 distance for search.
    /// This gives ~99.9% recall on SIFT-50K.
    #[must_use]
    pub fn is_asymmetric(&self) -> bool {
        matches!(self, Self::ScalarQuantized { .. })
    }

    /// Check if this storage uses SQ8 quantization
    #[must_use]
    pub fn is_sq8(&self) -> bool {
        matches!(self, Self::ScalarQuantized { .. })
    }

    /// Get number of vectors stored
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::FullPrecision { count, .. } | Self::ScalarQuantized { count, .. } => *count,
        }
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get dimensions
    #[must_use]
    pub fn dimensions(&self) -> usize {
        match self {
            Self::FullPrecision { dimensions, .. } | Self::ScalarQuantized { dimensions, .. } => {
                *dimensions
            }
        }
    }

    /// Insert a full precision vector
    pub fn insert(&mut self, vector: Vec<f32>) -> Result<u32, String> {
        match self {
            Self::FullPrecision {
                vectors,
                norms,
                count,
                dimensions,
            } => {
                if vector.len() != *dimensions {
                    return Err(format!(
                        "Vector dimension mismatch: expected {}, got {}",
                        dimensions,
                        vector.len()
                    ));
                }
                let id = *count as u32;
                // Compute and store squared norm for L2 decomposition
                let norm_sq: f32 = vector.iter().map(|&x| x * x).sum();
                norms.push(norm_sq);
                vectors.extend(vector);
                *count += 1;
                Ok(id)
            }
            Self::ScalarQuantized {
                params,
                quantized,
                norms,
                sums,
                training_buffer,
                count,
                dimensions,
                trained,
            } => {
                if vector.len() != *dimensions {
                    return Err(format!(
                        "Vector dimension mismatch: expected {}, got {}",
                        dimensions,
                        vector.len()
                    ));
                }

                let id = *count as u32;
                let dim = *dimensions;

                if *trained {
                    // Already trained - quantize directly, don't store original
                    let quant = params.quantize(&vector);
                    norms.push(quant.norm_sq);
                    sums.push(quant.sum);
                    quantized.extend(quant.data);
                    *count += 1;
                } else {
                    // Still in training phase - buffer the vector
                    training_buffer.extend(vector);
                    *count += 1;

                    if *count >= 256 {
                        // Time to train! Use buffered vectors as training sample
                        let training_refs: Vec<&[f32]> = (0..256)
                            .map(|i| &training_buffer[i * dim..(i + 1) * dim])
                            .collect();
                        *params =
                            ScalarParams::train(&training_refs).map_err(ToString::to_string)?;
                        *trained = true;

                        // Quantize all buffered vectors and store norms/sums
                        quantized.reserve(*count * dim);
                        norms.reserve(*count);
                        sums.reserve(*count);
                        for i in 0..*count {
                            let vec_slice = &training_buffer[i * dim..(i + 1) * dim];
                            let quant = params.quantize(vec_slice);
                            norms.push(quant.norm_sq);
                            sums.push(quant.sum);
                            quantized.extend(quant.data);
                        }

                        // Clear training buffer to free memory
                        training_buffer.clear();
                        training_buffer.shrink_to_fit();
                    }
                }
                // If not trained and count < 256, vectors stay in training_buffer
                // Search will fall back to f32 distance on training_buffer

                Ok(id)
            }
        }
    }

    /// Get a vector by ID (full precision)
    ///
    /// Returns slice directly into contiguous storage - zero-copy, cache-friendly.
    #[inline]
    #[must_use]
    pub fn get(&self, id: u32) -> Option<&[f32]> {
        match self {
            Self::FullPrecision {
                vectors,
                count,
                dimensions,
                ..
            } => {
                let idx = id as usize;
                if idx >= *count {
                    return None;
                }
                let start = idx * *dimensions;
                let end = start + *dimensions;
                Some(&vectors[start..end])
            }
            Self::ScalarQuantized {
                training_buffer,
                count,
                dimensions,
                trained,
                ..
            } => {
                // SQ8 doesn't store originals after training - no rescore support
                // During training phase, return from training buffer
                if *trained {
                    return None; // No originals stored
                }
                let idx = id as usize;
                if idx >= *count {
                    return None;
                }
                let start = idx * *dimensions;
                let end = start + *dimensions;
                Some(&training_buffer[start..end])
            }
        }
    }

    /// Get a vector by ID, dequantizing if necessary (returns owned Vec)
    ///
    /// For full precision storage, clones the slice.
    /// For quantized storage (SQ8), dequantizes the quantized bytes to f32.
    /// Used for neighbor-to-neighbor distance calculations during graph construction.
    #[must_use]
    pub fn get_dequantized(&self, id: u32) -> Option<Vec<f32>> {
        match self {
            Self::FullPrecision {
                vectors,
                count,
                dimensions,
                ..
            } => {
                let idx = id as usize;
                if idx >= *count {
                    return None;
                }
                let start = idx * *dimensions;
                let end = start + *dimensions;
                Some(vectors[start..end].to_vec())
            }
            Self::ScalarQuantized {
                params,
                quantized,
                training_buffer,
                count,
                dimensions,
                trained,
                ..
            } => {
                let idx = id as usize;
                if idx >= *count {
                    return None;
                }
                let dim = *dimensions;
                if *trained {
                    // Dequantize from quantized storage
                    let start = idx * dim;
                    let end = start + dim;
                    Some(params.dequantize(&quantized[start..end]))
                } else {
                    // Still in training phase, return from buffer
                    let start = idx * dim;
                    let end = start + dim;
                    Some(training_buffer[start..end].to_vec())
                }
            }
        }
    }

    /// Compute asymmetric L2 distance (query full precision, candidate quantized)
    ///
    /// This is the HOT PATH for asymmetric search. Works with `ScalarQuantized`
    /// storage. Returns None if storage is not quantized, not trained,
    /// or if id is out of bounds.
    ///
    /// # Performance (Apple Silicon M3 Max, 768D)
    /// - SQ8: Similar speed to full precision (1.07x)
    #[inline(always)]
    #[must_use]
    pub fn distance_asymmetric_l2(&self, query: &[f32], id: u32) -> Option<f32> {
        match self {
            Self::ScalarQuantized {
                params,
                quantized,
                norms,
                sums,
                count,
                dimensions,
                trained,
                ..
            } => {
                // Only use asymmetric distance if trained
                if !*trained {
                    return None;
                }

                let idx = id as usize;
                if idx >= *count {
                    return None;
                }

                let start = idx * *dimensions;
                let end = start + *dimensions;
                // NOTE: This path is inefficient - prepare_query is called per-vector!
                // Use distance_sq8_with_prep() instead for batch operations.
                let query_prep = params.prepare_query(query);
                Some(params.distance_l2_squared_raw(
                    &query_prep,
                    &quantized[start..end],
                    sums[idx],
                    norms[idx],
                ))
            }
            // FullPrecision uses regular L2 distance, not asymmetric
            Self::FullPrecision { .. } => None,
        }
    }

    /// Get ScalarParams reference for SQ8 storage (to prepare query once per search)
    #[inline]
    #[must_use]
    pub fn get_sq8_params(&self) -> Option<&ScalarParams> {
        if let Self::ScalarQuantized {
            params, trained, ..
        } = self
        {
            if *trained {
                return Some(params);
            }
        }
        None
    }

    /// Prepare query for efficient SQ8 distance computation
    ///
    /// Call this ONCE at the start of search, then use `distance_sq8_with_prep()`
    /// for each distance calculation. This avoids the O(D) query preparation
    /// overhead on each distance call.
    ///
    /// Returns None if storage is not SQ8 or not trained.
    #[inline]
    #[must_use]
    pub fn prepare_sq8_query(&self, query: &[f32]) -> Option<QueryPrep> {
        self.get_sq8_params()
            .map(|params| params.prepare_query(query))
    }

    /// Compute SQ8 distance with a pre-prepared query (efficient batch path)
    ///
    /// Use this instead of `distance_asymmetric_l2` when doing multiple distance
    /// computations with the same query. Call `params.prepare_query()` once,
    /// then use this method for each vector.
    #[inline(always)]
    #[must_use]
    pub fn distance_sq8_with_prep(&self, prep: &QueryPrep, id: u32) -> Option<f32> {
        if let Self::ScalarQuantized {
            params,
            quantized,
            norms,
            sums,
            count,
            dimensions,
            trained,
            ..
        } = self
        {
            if !*trained {
                return None;
            }
            let idx = id as usize;
            if idx >= *count {
                return None;
            }
            let start = idx * *dimensions;
            let end = start + *dimensions;
            Some(params.distance_l2_squared_raw(
                prep,
                &quantized[start..end],
                sums[idx],
                norms[idx],
            ))
        } else {
            None
        }
    }

    /// Batch compute SQ8 distances for multiple vectors
    ///
    /// More efficient than calling `distance_sq8_with_prep` in a loop:
    /// - Common terms computed once
    /// - Better cache utilization
    /// - Better instruction-level parallelism
    ///
    /// Returns the number of distances computed (may be less than ids.len() if some IDs are invalid).
    #[inline]
    pub fn distance_sq8_batch(
        &self,
        prep: &QueryPrep,
        ids: &[u32],
        distances: &mut [f32],
    ) -> usize {
        if let Self::ScalarQuantized {
            params,
            quantized,
            norms,
            sums,
            count,
            dimensions,
            trained,
            ..
        } = self
        {
            if !*trained {
                return 0;
            }

            let dim = *dimensions;
            let n = *count;
            let mut computed = 0;

            // Pre-compute common terms
            let scale_sq = params.scale * params.scale;
            let offset_term = params.offset * params.offset * dim as f32;
            let query_norm = prep.norm_sq;

            for (i, &id) in ids.iter().enumerate() {
                let idx = id as usize;
                if idx >= n {
                    continue;
                }

                let start = idx * dim;
                let vec_data = &quantized[start..start + dim];
                let vec_sum = sums[idx];
                let vec_norm_sq = norms[idx];

                let int_dot = params.int_dot_product_pub(&prep.quantized, vec_data);

                let dot = scale_sq * int_dot as f32
                    + params.scale * params.offset * (prep.sum + vec_sum) as f32
                    + offset_term;

                distances[i] = query_norm + vec_norm_sq - 2.0 * dot;
                computed += 1;
            }

            computed
        } else {
            0
        }
    }

    /// Get the pre-computed squared norm (||v||^2) for a vector
    ///
    /// Only available for FullPrecision storage. Used for L2 decomposition optimization.
    #[inline]
    #[must_use]
    pub fn get_norm(&self, id: u32) -> Option<f32> {
        match self {
            Self::FullPrecision { norms, count, .. } => {
                let idx = id as usize;
                if idx >= *count {
                    return None;
                }
                Some(norms[idx])
            }
            Self::ScalarQuantized { .. } => None,
        }
    }

    /// Check if L2 decomposition is available for this storage
    ///
    /// Returns true for:
    /// - FullPrecision storage (always has pre-computed norms)
    /// - ScalarQuantized storage when trained (uses multiversion dot_product)
    ///
    /// The decomposition path uses `dot_product` with `#[multiversion]` which
    /// provides better cross-compilation compatibility than raw NEON intrinsics.
    #[inline]
    #[must_use]
    pub fn supports_l2_decomposition(&self) -> bool {
        // SQ8 is excluded - L2 decomposition causes ~10% recall regression
        // due to numerical precision issues (catastrophic cancellation).
        // SQ8 uses the asymmetric path via distance_asymmetric_l2 for 99%+ recall.
        matches!(self, Self::FullPrecision { .. })
    }

    /// Compute L2 squared distance using decomposition: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    ///
    /// This is ~7-15% faster than direct L2/asymmetric computation because:
    /// - Vector norms are pre-computed during insert
    /// - Query norm is computed once per search (passed in)
    /// - Only dot product is computed per-vector (2N FLOPs vs 3N)
    ///
    /// Works for both FullPrecision and trained ScalarQuantized storage.
    /// Returns None if decomposition is not available.
    #[inline(always)]
    #[must_use]
    pub fn distance_l2_decomposed(&self, query: &[f32], query_norm: f32, id: u32) -> Option<f32> {
        match self {
            Self::FullPrecision {
                vectors,
                norms,
                count,
                dimensions,
            } => {
                let idx = id as usize;
                if idx >= *count {
                    return None;
                }
                let start = idx * *dimensions;
                let end = start + *dimensions;
                let vec = &vectors[start..end];
                let vec_norm = norms[idx];

                // ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
                // Uses SIMD-accelerated dot product for performance
                let dot = dot_product(query, vec);
                Some(query_norm + vec_norm - 2.0 * dot)
            }
            Self::ScalarQuantized {
                params,
                quantized,
                norms,
                sums,
                count,
                dimensions,
                trained,
                ..
            } => {
                if !*trained {
                    return None;
                }
                let idx = id as usize;
                if idx >= *count {
                    return None;
                }
                let start = idx * *dimensions;
                let end = start + *dimensions;
                let vec_norm = norms[idx];
                let vec_sum = sums[idx];

                // Use integer SIMD distance with precomputed sums
                let query_prep = params.prepare_query(query);
                let quantized_slice = &quantized[start..end];
                Some(params.distance_l2_squared_raw(
                    &query_prep,
                    quantized_slice,
                    vec_sum,
                    vec_norm,
                ))
            }
        }
    }

    /// Prefetch a vector's data into CPU cache (for HNSW search optimization)
    ///
    /// This hints to the CPU to load the vector data into cache before it's needed.
    /// Call this on neighbor[j+1] while computing distance to neighbor[j].
    /// ~10% search speedup per hnswlib benchmarks.
    ///
    /// NOTE: This gets the pointer directly without loading the data, so the
    /// prefetch hint can be issued before the data is needed.
    /// Prefetch vector data into L1 cache
    ///
    /// Simple single-cache-line prefetch (64 bytes).
    /// Hardware prefetcher handles subsequent cache lines.
    #[inline]
    pub fn prefetch(&self, id: u32) {
        let ptr: Option<*const u8> = match self {
            Self::FullPrecision {
                vectors,
                count,
                dimensions,
                ..
            } => {
                let idx = id as usize;
                if idx >= *count {
                    None
                } else {
                    let start = idx * *dimensions;
                    Some(vectors[start..].as_ptr().cast())
                }
            }
            Self::ScalarQuantized {
                quantized,
                training_buffer,
                count,
                dimensions,
                trained,
                ..
            } => {
                let idx = id as usize;
                if idx >= *count {
                    None
                } else if *trained {
                    // Prefetch quantized data for asymmetric search
                    let start = idx * *dimensions;
                    Some(quantized[start..].as_ptr())
                } else {
                    // Not trained yet - prefetch training buffer f32 data
                    let start = idx * *dimensions;
                    Some(training_buffer[start..].as_ptr().cast())
                }
            }
        };

        if let Some(ptr) = ptr {
            // SAFETY: ptr is valid and aligned since it comes from a valid Vec
            #[cfg(target_arch = "x86_64")]
            unsafe {
                std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
            }
            #[cfg(target_arch = "aarch64")]
            unsafe {
                std::arch::asm!(
                    "prfm pldl1keep, [{ptr}]",
                    ptr = in(reg) ptr,
                    options(nostack, preserves_flags)
                );
            }
        }
    }

    /// Compute quantization thresholds from sample vectors
    ///
    /// Only relevant for SQ8 (scalar quantization).
    pub fn train_quantization(&mut self, sample_vectors: &[Vec<f32>]) -> Result<(), String> {
        match self {
            Self::FullPrecision { .. } => {
                Err("Cannot train quantization on full precision storage".to_string())
            }
            Self::ScalarQuantized {
                params,
                quantized,
                norms,
                sums,
                training_buffer,
                count,
                dimensions,
                trained,
            } => {
                if sample_vectors.is_empty() {
                    return Err("Cannot train on empty sample".to_string());
                }

                // Train params from sample vectors
                let refs: Vec<&[f32]> =
                    sample_vectors.iter().map(std::vec::Vec::as_slice).collect();
                *params = ScalarParams::train(&refs).map_err(ToString::to_string)?;
                *trained = true;

                // If there are vectors in training buffer, quantize them now
                if *count > 0 && quantized.is_empty() && !training_buffer.is_empty() {
                    let dim = *dimensions;
                    quantized.reserve(*count * dim);
                    norms.reserve(*count);
                    sums.reserve(*count);
                    for i in 0..*count {
                        let vec_slice = &training_buffer[i * dim..(i + 1) * dim];
                        let quant = params.quantize(vec_slice);
                        norms.push(quant.norm_sq);
                        sums.push(quant.sum);
                        quantized.extend(quant.data);
                    }
                    // Clear training buffer to free memory
                    training_buffer.clear();
                    training_buffer.shrink_to_fit();
                }

                Ok(())
            }
        }
    }

    /// Get memory usage in bytes (approximate)
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        match self {
            Self::FullPrecision { vectors, norms, .. } => {
                vectors.len() * std::mem::size_of::<f32>()
                    + norms.len() * std::mem::size_of::<f32>()
            }
            Self::ScalarQuantized {
                quantized,
                norms,
                sums,
                training_buffer,
                ..
            } => {
                // Quantized u8 vectors + norms + sums + training buffer (usually empty after training) + params
                let quantized_size = quantized.len();
                let norms_size = norms.len() * std::mem::size_of::<f32>();
                let sums_size = sums.len() * std::mem::size_of::<i32>();
                let buffer_size = training_buffer.len() * std::mem::size_of::<f32>();
                // Uniform params: scale + offset + dimensions = 2 * f32 + usize
                let params_size = 2 * std::mem::size_of::<f32>() + std::mem::size_of::<usize>();
                quantized_size + norms_size + sums_size + buffer_size + params_size
            }
        }
    }

    /// Reorder vectors based on node ID mapping
    ///
    /// `old_to_new`[`old_id`] = `new_id`
    /// This reorders vectors to match the BFS-reordered neighbor lists.
    pub fn reorder(&mut self, old_to_new: &[u32]) {
        match self {
            Self::FullPrecision {
                vectors,
                norms,
                count,
                dimensions,
            } => {
                let dim = *dimensions;
                let n = *count;
                let mut new_vectors = vec![0.0f32; vectors.len()];
                let mut new_norms = vec![0.0f32; norms.len()];
                for (old_id, &new_id) in old_to_new.iter().enumerate() {
                    if old_id < n {
                        let old_start = old_id * dim;
                        let new_start = new_id as usize * dim;
                        new_vectors[new_start..new_start + dim]
                            .copy_from_slice(&vectors[old_start..old_start + dim]);
                        new_norms[new_id as usize] = norms[old_id];
                    }
                }
                *vectors = new_vectors;
                *norms = new_norms;
            }
            Self::ScalarQuantized {
                quantized,
                norms,
                sums,
                count,
                dimensions,
                ..
            } => {
                let dim = *dimensions;
                let n = *count;

                // Reorder quantized vectors, norms, and sums
                let mut new_quantized = vec![0u8; quantized.len()];
                let mut new_norms = vec![0.0f32; norms.len()];
                let mut new_sums = vec![0i32; sums.len()];
                for (old_id, &new_id) in old_to_new.iter().enumerate() {
                    if old_id < n {
                        let old_start = old_id * dim;
                        let new_start = new_id as usize * dim;
                        new_quantized[new_start..new_start + dim]
                            .copy_from_slice(&quantized[old_start..old_start + dim]);
                        if old_id < norms.len() {
                            new_norms[new_id as usize] = norms[old_id];
                        }
                        if old_id < sums.len() {
                            new_sums[new_id as usize] = sums[old_id];
                        }
                    }
                }
                *quantized = new_quantized;
                *norms = new_norms;
                *sums = new_sums;
            }
        }
    }

    /// Check if storage is SQ8 quantized and trained
    #[must_use]
    pub fn is_quantized_and_trained(&self) -> bool {
        matches!(self, Self::ScalarQuantized { trained: true, .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_storage_full_precision() {
        let mut storage = VectorStorage::new_full_precision(3);

        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];

        let id1 = storage.insert(vec1.clone()).unwrap();
        let id2 = storage.insert(vec2.clone()).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(storage.len(), 2);

        assert_eq!(storage.get(0), Some(vec1.as_slice()));
        assert_eq!(storage.get(1), Some(vec2.as_slice()));
    }

    #[test]
    fn test_vector_storage_dimension_check() {
        let mut storage = VectorStorage::new_full_precision(3);

        let wrong_dim = vec![1.0, 2.0]; // Only 2 dimensions
        assert!(storage.insert(wrong_dim).is_err());
    }

    #[test]
    fn test_sq8_train_empty_sample_rejected() {
        let mut storage = VectorStorage::new_sq8_quantized(4);
        let empty_samples: Vec<Vec<f32>> = vec![];
        let result = storage.train_quantization(&empty_samples);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty sample"));
    }
}
