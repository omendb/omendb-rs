//! Quantization support for NodeStorage (SQ8 and PQ)
//!
//! Implements scalar and product quantization with lazy training.
//! - SQ8: L2 decomposition for fast integer SIMD distance
//! - PQ: ADC lookup tables for M-byte compressed distance

use super::{NodeStorage, StorageMode};
use crate::compression::product::{PQParams, PQQueryPrep};
use crate::compression::scalar::{QueryPrep, ScalarParams};

impl NodeStorage {
    /// Set vector in SQ8 mode with lazy training
    pub(super) fn set_vector_sq8(&mut self, id: u32, vector: &[f32]) {
        let id_usize = id as usize;

        if self.sq8_trained {
            // Already trained - quantize directly
            let params = self.sq8_params.as_ref().expect("SQ8 params should exist");
            let quant = params.quantize(vector);

            // Store quantized vector
            let ptr = self.node_ptr_mut(id);
            unsafe {
                let vec_ptr = ptr.add(self.vector_offset);
                std::ptr::copy_nonoverlapping(quant.data.as_ptr(), vec_ptr, self.dimensions);
            }

            // Store norm and sum
            if id_usize >= self.norms.len() {
                self.norms.resize(id_usize + 1, 0.0);
            }
            if id_usize >= self.sq8_sums.len() {
                self.sq8_sums.resize(id_usize + 1, 0);
            }
            self.norms[id_usize] = quant.norm_sq;
            self.sq8_sums[id_usize] = quant.sum;
        } else {
            // Still in training phase - buffer the vector
            self.training_buffer.extend_from_slice(vector);

            // Store zeros in the colocated storage for now (will be filled after training)
            let ptr = self.node_ptr_mut(id);
            unsafe {
                let vec_ptr = ptr.add(self.vector_offset);
                std::ptr::write_bytes(vec_ptr, 0, self.dimensions);
            }

            // Check if we have enough vectors to train (256 threshold)
            let num_vectors = self.training_buffer.len() / self.dimensions;
            if num_vectors >= 256 {
                self.train_quantization();
            }
        }
    }

    /// Train SQ8 quantization from buffered vectors
    pub(super) fn train_quantization(&mut self) {
        let dim = self.dimensions;
        let num_vectors = self.training_buffer.len() / dim;

        // Build training sample (refs to slices)
        let training_refs: Vec<&[f32]> = (0..num_vectors)
            .map(|i| &self.training_buffer[i * dim..(i + 1) * dim])
            .collect();

        // Train quantization parameters
        let params = ScalarParams::train(&training_refs).expect("Failed to train SQ8 params");
        self.sq8_params = Some(params);
        self.sq8_trained = true;

        // Quantize all buffered vectors and store them
        self.norms.reserve(num_vectors);
        self.sq8_sums.reserve(num_vectors);

        for i in 0..num_vectors {
            let vec_slice = &self.training_buffer[i * dim..(i + 1) * dim];
            let quant = params.quantize(vec_slice);

            // Store quantized vector in colocated storage
            let ptr = self.node_ptr_mut(i as u32);
            unsafe {
                let vec_ptr = ptr.add(self.vector_offset);
                std::ptr::copy_nonoverlapping(quant.data.as_ptr(), vec_ptr, dim);
            }

            // Store norm and sum
            if i >= self.norms.len() {
                self.norms.push(quant.norm_sq);
            } else {
                self.norms[i] = quant.norm_sq;
            }
            if i >= self.sq8_sums.len() {
                self.sq8_sums.push(quant.sum);
            } else {
                self.sq8_sums[i] = quant.sum;
            }
        }

        // Clear training buffer
        self.training_buffer.clear();
        self.training_buffer.shrink_to_fit();
    }

    /// Prepare query for SQ8 distance calculation
    #[must_use]
    pub fn prepare_query(&self, query: &[f32]) -> Option<QueryPrep> {
        self.sq8_params
            .as_ref()
            .map(|params| params.prepare_query(query))
    }

    /// Compute SQ8 L2 distance (requires trained quantization)
    ///
    /// Uses integer SIMD for fast distance calculation.
    #[inline]
    #[must_use]
    pub fn distance_sq8(&self, prep: &QueryPrep, id: u32) -> Option<f32> {
        let params = self.sq8_params.as_ref()?;
        if !self.sq8_trained {
            return None;
        }

        let idx = id as usize;
        if idx >= self.len {
            return None;
        }

        let quantized = self.quantized_vector(id);
        let vec_norm_sq = self.norms.get(idx)?;
        let vec_sum = *self.sq8_sums.get(idx)?;

        // L2 decomposition: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        let scale_sq = params.scale * params.scale;
        let offset_term = params.offset * params.offset * self.dimensions as f32;

        let int_dot = params.int_dot_product_pub(&prep.quantized, quantized);

        let dot = scale_sq * int_dot as f32
            + params.scale * params.offset * (prep.sum + vec_sum) as f32
            + offset_term;

        Some(prep.norm_sq + vec_norm_sq - 2.0 * dot)
    }

    /// Batch compute SQ8 L2 distances
    ///
    /// Fills distances buffer with SQ8 distances for the given IDs.
    /// Returns the number of distances computed (some IDs may be out of range).
    #[inline]
    pub fn distance_sq8_batch(
        &self,
        prep: &QueryPrep,
        ids: &[u32],
        distances: &mut [f32],
    ) -> usize {
        let mut count = 0;
        for (&id, dist) in ids.iter().zip(distances.iter_mut()) {
            if let Some(d) = self.distance_sq8(prep, id) {
                *dist = d;
                count += 1;
            }
        }
        count
    }

    // ========================================================================
    // PQ (Product Quantization) support
    // ========================================================================

    /// Set vector in PQ mode with lazy training
    pub(super) fn set_vector_pq(&mut self, id: u32, vector: &[f32]) {
        let id_usize = id as usize;

        if self.pq_trained {
            let params = self.pq_params.as_ref().expect("PQ params should exist");
            let codes = params.quantize(vector);
            let num_subspaces = params.num_subspaces;

            // Store PQ codes in the colocated node storage
            let ptr = self.node_ptr_mut(id);
            unsafe {
                let vec_ptr = ptr.add(self.vector_offset);
                std::ptr::copy_nonoverlapping(codes.as_ptr(), vec_ptr, num_subspaces);
            }

            // Also store in contiguous pq_codes for batch access
            let start = id_usize * num_subspaces;
            let end = start + num_subspaces;
            if end > self.pq_codes.len() {
                self.pq_codes.resize(end, 0);
            }
            self.pq_codes[start..end].copy_from_slice(&codes);

            // Store norm for rescore
            let norm_sq: f32 = vector.iter().map(|&x| x * x).sum();
            if id_usize >= self.norms.len() {
                self.norms.resize(id_usize + 1, 0.0);
            }
            self.norms[id_usize] = norm_sq;
        } else {
            // Buffer for training
            self.training_buffer.extend_from_slice(vector);

            // Store zeros in colocated storage
            let num_subspaces = match self.mode {
                StorageMode::PQ(n) => n,
                _ => unreachable!(),
            };
            let ptr = self.node_ptr_mut(id);
            unsafe {
                let vec_ptr = ptr.add(self.vector_offset);
                std::ptr::write_bytes(vec_ptr, 0, num_subspaces);
            }

            // Train after 256 vectors
            let num_vectors = self.training_buffer.len() / self.dimensions;
            if num_vectors >= 256 {
                self.train_pq_quantization();
            }
        }
    }

    /// Train PQ quantization from buffered vectors
    pub(super) fn train_pq_quantization(&mut self) {
        let dim = self.dimensions;
        let num_vectors = self.training_buffer.len() / dim;
        let num_subspaces = match self.mode {
            StorageMode::PQ(n) => n,
            _ => unreachable!(),
        };

        // Build training sample
        let training_refs: Vec<&[f32]> = (0..num_vectors)
            .map(|i| &self.training_buffer[i * dim..(i + 1) * dim])
            .collect();

        // Train PQ codebooks
        let params =
            PQParams::train(&training_refs, num_subspaces).expect("Failed to train PQ params");
        self.pq_params = Some(params.clone());
        self.pq_trained = true;

        // Quantize all buffered vectors
        self.pq_codes.reserve(num_vectors * num_subspaces);
        self.norms.reserve(num_vectors);

        for i in 0..num_vectors {
            let vec_slice = &self.training_buffer[i * dim..(i + 1) * dim];
            let codes = params.quantize(vec_slice);
            let norm_sq: f32 = vec_slice.iter().map(|&x| x * x).sum();

            // Store in colocated storage
            let ptr = self.node_ptr_mut(i as u32);
            unsafe {
                let vec_ptr = ptr.add(self.vector_offset);
                std::ptr::copy_nonoverlapping(codes.as_ptr(), vec_ptr, num_subspaces);
            }

            // Store in contiguous codes array
            let start = i * num_subspaces;
            if start + num_subspaces > self.pq_codes.len() {
                self.pq_codes.resize(start + num_subspaces, 0);
            }
            self.pq_codes[start..start + num_subspaces].copy_from_slice(&codes);

            // Store norm
            if i >= self.norms.len() {
                self.norms.push(norm_sq);
            } else {
                self.norms[i] = norm_sq;
            }
        }

        self.training_buffer.clear();
        self.training_buffer.shrink_to_fit();
    }

    /// Prepare query for PQ distance calculation
    #[must_use]
    pub fn prepare_query_pq(&self, query: &[f32]) -> Option<PQQueryPrep> {
        self.pq_params
            .as_ref()
            .map(|params| params.build_adc_table(query))
    }

    /// Compute PQ approximate L2 distance
    #[inline]
    #[must_use]
    pub fn distance_pq(&self, prep: &PQQueryPrep, id: u32) -> Option<f32> {
        if !self.pq_trained {
            return None;
        }
        let num_subspaces = match self.mode {
            StorageMode::PQ(n) => n,
            _ => return None,
        };
        let idx = id as usize;
        let start = idx * num_subspaces;
        let end = start + num_subspaces;
        if end > self.pq_codes.len() {
            return None;
        }
        Some(PQParams::distance_adc(prep, &self.pq_codes[start..end]))
    }

    /// Batch compute PQ distances
    #[inline]
    pub fn distance_pq_batch(
        &self,
        prep: &PQQueryPrep,
        ids: &[u32],
        distances: &mut [f32],
    ) -> usize {
        let mut count = 0;
        for (&id, dist) in ids.iter().zip(distances.iter_mut()) {
            if let Some(d) = self.distance_pq(prep, id) {
                *dist = d;
                count += 1;
            }
        }
        count
    }

    /// Get PQ codes for a vector (from contiguous storage)
    #[inline]
    #[must_use]
    pub fn pq_codes(&self, id: u32) -> Option<&[u8]> {
        let num_subspaces = match self.mode {
            StorageMode::PQ(n) => n,
            _ => return None,
        };
        let idx = id as usize;
        let start = idx * num_subspaces;
        let end = start + num_subspaces;
        if end <= self.pq_codes.len() {
            Some(&self.pq_codes[start..end])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq8_lazy_training() {
        let mut storage = NodeStorage::new_sq8(4, 2, 8);
        assert!(!storage.is_trained());

        // Insert 255 vectors (not enough to train)
        for i in 0..255 {
            storage.allocate_node();
            let vector: Vec<f32> = (0..4).map(|j| (i * 4 + j) as f32).collect();
            storage.set_vector(i as u32, &vector);
        }
        assert!(!storage.is_trained());

        // Insert 256th vector - should trigger training
        storage.allocate_node();
        let vector: Vec<f32> = (0..4).map(|j| (255 * 4 + j) as f32).collect();
        storage.set_vector(255, &vector);
        assert!(storage.is_trained());

        // New vectors should be quantized directly
        storage.allocate_node();
        let vector: Vec<f32> = (0..4).map(|j| (256 * 4 + j) as f32).collect();
        storage.set_vector(256, &vector);
        assert_eq!(storage.len(), 257);
    }

    #[test]
    fn test_sq8_dequantization() {
        let mut storage = NodeStorage::new_sq8(4, 2, 8);

        // Insert enough vectors to trigger training
        for i in 0..256 {
            storage.allocate_node();
            let vector: Vec<f32> = (0..4).map(|j| (i + j) as f32 / 255.0).collect();
            storage.set_vector(i as u32, &vector);
        }
        assert!(storage.is_trained());

        // Dequantized should be approximately equal to original
        let original: Vec<f32> = (0..4).map(|j| (100 + j) as f32 / 255.0).collect();
        let dequantized = storage.get_dequantized(100).unwrap();

        // Check approximate equality (quantization introduces small errors)
        for (o, d) in original.iter().zip(dequantized.iter()) {
            assert!((o - d).abs() < 0.02, "Dequantization error too large");
        }
    }

    #[test]
    fn test_sq8_distance_calculation() {
        let mut storage = NodeStorage::new_sq8(128, 2, 8);

        // Insert vectors with known values (realistic high-dimensional data)
        for i in 0..256 {
            storage.allocate_node();
            // Random-ish distribution with meaningful variance
            let vector: Vec<f32> = (0..128)
                .map(|j| ((i * 128 + j) % 255) as f32 / 255.0)
                .collect();
            storage.set_vector(i as u32, &vector);
        }
        assert!(storage.is_trained());

        // Query vector (middle of range)
        let query: Vec<f32> = (0..128).map(|j| (j % 255) as f32 / 255.0).collect();
        let prep = storage.prepare_query(&query).expect("Should have params");

        // Calculate distance to vectors
        for id in [0, 50, 100, 150, 200, 250] {
            let dist = storage.distance_sq8(&prep, id);
            assert!(
                dist.is_some(),
                "Distance should be computable for vector {id}"
            );
            // Allow small negative values due to floating point precision
            let dist_val = dist.unwrap();
            assert!(
                dist_val >= -0.01,
                "Distance {} for vector {} is too negative",
                dist_val,
                id
            );
        }

        // Distance to self should be near zero
        storage.allocate_node();
        storage.set_vector(256, &query);
        let self_dist = storage.distance_sq8(&prep, 256).unwrap();
        assert!(
            self_dist.abs() < 0.1,
            "Self-distance should be near zero, got {}",
            self_dist
        );
    }

    #[test]
    fn test_sq8_norms_stored() {
        let mut storage = NodeStorage::new_sq8(4, 2, 8);

        // Insert enough vectors to trigger training
        for i in 0..256 {
            storage.allocate_node();
            let vector: Vec<f32> = (0..4).map(|j| (i + j) as f32).collect();
            storage.set_vector(i as u32, &vector);
        }

        // After training, norms should be stored
        for i in 0..256 {
            let norm = storage.get_norm(i as u32);
            assert!(norm.is_some(), "Norm should be stored for vector {i}");
            assert!(norm.unwrap() >= 0.0, "Norm should be non-negative");
        }
    }
}
