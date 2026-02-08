//! Quantization support for NodeStorage (SQ8 and RaBitQ)
//!
//! Implements scalar and binary quantization with lazy training.
//! - SQ8: L2 decomposition for fast integer SIMD distance
//! - RaBitQ: 1-bit random rotation with binary distance

use super::NodeStorage;
use crate::compression::rabitq::{RaBitQParams, RaBitQQueryPrep};
use crate::compression::scalar::{QueryPrep, ScalarParams};

impl NodeStorage {
    /// Set vector in SQ8 mode with lazy training
    pub(super) fn set_vector_sq8(&mut self, id: u32, vector: &[f32]) {
        let id_usize = id as usize;

        if self.sq8_trained {
            let params = self.sq8_params.as_ref().expect("SQ8 params should exist");
            let quant = params.quantize(vector);

            let ptr = self.node_ptr_mut(id);
            unsafe {
                let vec_ptr = ptr.add(self.vector_offset);
                std::ptr::copy_nonoverlapping(quant.data.as_ptr(), vec_ptr, self.dimensions);
            }

            if id_usize >= self.norms.len() {
                self.norms.resize(id_usize + 1, 0.0);
            }
            if id_usize >= self.sq8_sums.len() {
                self.sq8_sums.resize(id_usize + 1, 0);
            }
            self.norms[id_usize] = quant.norm_sq;
            self.sq8_sums[id_usize] = quant.sum;
        } else {
            self.training_buffer.extend_from_slice(vector);

            let ptr = self.node_ptr_mut(id);
            unsafe {
                let vec_ptr = ptr.add(self.vector_offset);
                std::ptr::write_bytes(vec_ptr, 0, self.dimensions);
            }

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

        let training_refs: Vec<&[f32]> = (0..num_vectors)
            .map(|i| &self.training_buffer[i * dim..(i + 1) * dim])
            .collect();

        let params = ScalarParams::train(&training_refs).expect("Failed to train SQ8 params");
        self.sq8_params = Some(params);
        self.sq8_trained = true;

        self.norms.reserve(num_vectors);
        self.sq8_sums.reserve(num_vectors);

        for i in 0..num_vectors {
            let vec_slice = &self.training_buffer[i * dim..(i + 1) * dim];
            let quant = params.quantize(vec_slice);

            let ptr = self.node_ptr_mut(i as u32);
            unsafe {
                let vec_ptr = ptr.add(self.vector_offset);
                std::ptr::copy_nonoverlapping(quant.data.as_ptr(), vec_ptr, dim);
            }

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

    /// Set vector in RaBitQ mode with lazy training
    pub(super) fn set_vector_rabitq(&mut self, id: u32, vector: &[f32]) {
        let id_usize = id as usize;
        let dim = self.dimensions;

        if self.rabitq_trained {
            let params = self
                .rabitq_params
                .as_ref()
                .expect("RaBitQ params should exist");
            let (codes, dis_u_2, factor_ip, factor_ppc, _factor_err) = params.quantize(vector);
            let code_words = params.code_words();

            let start = id_usize * code_words;
            let end = start + code_words;
            if end > self.rabitq_codes.len() {
                self.rabitq_codes.resize(end, 0);
            }
            self.rabitq_codes[start..end].copy_from_slice(&codes);

            let meta_start = id_usize * 3;
            let meta_end = meta_start + 3;
            if meta_end > self.rabitq_metadata.len() {
                self.rabitq_metadata.resize(meta_end, 0.0);
            }
            self.rabitq_metadata[meta_start] = dis_u_2;
            self.rabitq_metadata[meta_start + 1] = factor_ip;
            self.rabitq_metadata[meta_start + 2] = factor_ppc;

            let orig_start = id_usize * dim;
            let orig_end = orig_start + dim;
            if orig_end > self.rabitq_originals.len() {
                self.rabitq_originals.resize(orig_end, 0.0);
            }
            self.rabitq_originals[orig_start..orig_end].copy_from_slice(vector);

            let norm_sq: f32 = vector.iter().map(|&x| x * x).sum();
            if id_usize >= self.norms.len() {
                self.norms.resize(id_usize + 1, 0.0);
            }
            self.norms[id_usize] = norm_sq;
        } else {
            self.training_buffer.extend_from_slice(vector);

            let num_vectors = self.training_buffer.len() / dim;
            if num_vectors >= 256 {
                self.train_rabitq_quantization();
            }
        }
    }

    /// Train RaBitQ quantization from buffered vectors
    pub(super) fn train_rabitq_quantization(&mut self) {
        let dim = self.dimensions;
        let num_vectors = self.training_buffer.len() / dim;

        self.rabitq_originals = std::mem::take(&mut self.training_buffer);

        let training_refs: Vec<&[f32]> = (0..num_vectors)
            .map(|i| &self.rabitq_originals[i * dim..(i + 1) * dim])
            .collect();

        let params = RaBitQParams::train(&training_refs).expect("Failed to train RaBitQ params");
        let code_words = params.code_words();
        self.rabitq_params = Some(params.clone());
        self.rabitq_trained = true;

        self.rabitq_codes.reserve(num_vectors * code_words);
        self.rabitq_metadata.reserve(num_vectors * 3);
        self.norms.reserve(num_vectors);

        for i in 0..num_vectors {
            let vec_slice = &self.rabitq_originals[i * dim..(i + 1) * dim];
            let (codes, dis_u_2, factor_ip, factor_ppc, _factor_err) = params.quantize(vec_slice);
            let norm_sq: f32 = vec_slice.iter().map(|&x| x * x).sum();

            let start = i * code_words;
            if start + code_words > self.rabitq_codes.len() {
                self.rabitq_codes.resize(start + code_words, 0);
            }
            self.rabitq_codes[start..start + code_words].copy_from_slice(&codes);

            let meta_start = i * 3;
            if meta_start + 3 > self.rabitq_metadata.len() {
                self.rabitq_metadata.resize(meta_start + 3, 0.0);
            }
            self.rabitq_metadata[meta_start] = dis_u_2;
            self.rabitq_metadata[meta_start + 1] = factor_ip;
            self.rabitq_metadata[meta_start + 2] = factor_ppc;

            if i >= self.norms.len() {
                self.norms.push(norm_sq);
            } else {
                self.norms[i] = norm_sq;
            }
        }
    }

    /// Prepare query for RaBitQ distance calculation
    #[must_use]
    pub fn prepare_query_rabitq(&self, query: &[f32]) -> Option<RaBitQQueryPrep> {
        self.rabitq_params
            .as_ref()
            .map(|params| params.prepare_query(query))
    }

    /// Compute RaBitQ approximate L2 distance
    #[inline]
    #[must_use]
    pub fn distance_rabitq(&self, prep: &RaBitQQueryPrep, id: u32) -> Option<f32> {
        let params = self.rabitq_params.as_ref()?;
        if !self.rabitq_trained {
            return None;
        }

        let idx = id as usize;
        let code_words = params.code_words();
        let start = idx * code_words;
        let end = start + code_words;
        if end > self.rabitq_codes.len() {
            return None;
        }

        let meta_start = idx * 3;
        if meta_start + 3 > self.rabitq_metadata.len() {
            return None;
        }

        let signs = &self.rabitq_codes[start..end];
        let dis_u_2 = self.rabitq_metadata[meta_start];
        let factor_ip = self.rabitq_metadata[meta_start + 1];
        let factor_ppc = self.rabitq_metadata[meta_start + 2];

        Some(RaBitQParams::distance(
            prep, signs, dis_u_2, factor_ip, factor_ppc,
        ))
    }

    /// Batch compute RaBitQ distances
    #[inline]
    pub fn distance_rabitq_batch(
        &self,
        prep: &RaBitQQueryPrep,
        ids: &[u32],
        distances: &mut [f32],
    ) -> usize {
        let mut count = 0;
        for (&id, dist) in ids.iter().zip(distances.iter_mut()) {
            if let Some(d) = self.distance_rabitq(prep, id) {
                *dist = d;
                count += 1;
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq8_lazy_training() {
        let mut storage = NodeStorage::new_sq8(4, 2, 8);
        assert!(!storage.is_trained());

        for i in 0..255 {
            storage.allocate_node();
            let vector: Vec<f32> = (0..4).map(|j| (i * 4 + j) as f32).collect();
            storage.set_vector(i as u32, &vector);
        }
        assert!(!storage.is_trained());

        storage.allocate_node();
        let vector: Vec<f32> = (0..4).map(|j| (255 * 4 + j) as f32).collect();
        storage.set_vector(255, &vector);
        assert!(storage.is_trained());

        storage.allocate_node();
        let vector: Vec<f32> = (0..4).map(|j| (256 * 4 + j) as f32).collect();
        storage.set_vector(256, &vector);
        assert_eq!(storage.len(), 257);
    }

    #[test]
    fn test_sq8_dequantization() {
        let mut storage = NodeStorage::new_sq8(4, 2, 8);

        for i in 0..256 {
            storage.allocate_node();
            let vector: Vec<f32> = (0..4).map(|j| (i + j) as f32 / 255.0).collect();
            storage.set_vector(i as u32, &vector);
        }
        assert!(storage.is_trained());

        let original: Vec<f32> = (0..4).map(|j| (100 + j) as f32 / 255.0).collect();
        let dequantized = storage.get_dequantized(100).unwrap();

        for (o, d) in original.iter().zip(dequantized.iter()) {
            assert!((o - d).abs() < 0.02, "Dequantization error too large");
        }
    }

    #[test]
    fn test_sq8_distance_calculation() {
        let mut storage = NodeStorage::new_sq8(128, 2, 8);

        for i in 0..256 {
            storage.allocate_node();
            let vector: Vec<f32> = (0..128)
                .map(|j| ((i * 128 + j) % 255) as f32 / 255.0)
                .collect();
            storage.set_vector(i as u32, &vector);
        }
        assert!(storage.is_trained());

        let query: Vec<f32> = (0..128).map(|j| (j % 255) as f32 / 255.0).collect();
        let prep = storage.prepare_query(&query).expect("Should have params");

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
    fn test_rabitq_pretrain_persistence_roundtrip() {
        // Regression test: RaBitQ with < 256 vectors must survive save/load
        let dim = 32;
        let num_vectors = 50; // Well below 256 training threshold
        let mut storage = NodeStorage::new_rabitq(dim, 4, 8);

        let mut vectors = Vec::new();
        for i in 0..num_vectors {
            storage.allocate_node();
            let vector: Vec<f32> = (0..dim)
                .map(|j| ((i * dim + j) % 100) as f32 / 100.0)
                .collect();
            storage.set_vector(i as u32, &vector);
            vectors.push(vector);
        }
        assert!(
            !storage.is_trained(),
            "Should not be trained with only {num_vectors} vectors"
        );

        // Verify vectors are accessible before save
        for i in 0..num_vectors {
            let deq = storage.get_dequantized(i as u32);
            assert!(
                deq.is_some(),
                "Vector {i} should be retrievable before save"
            );
        }

        // Serialize and deserialize
        let bytes = storage.serialize_full();
        let restored =
            NodeStorage::deserialize_full(&bytes).expect("Deserialization should succeed");

        assert_eq!(restored.len(), num_vectors);
        assert!(
            !restored.is_trained(),
            "Should still be untrained after load"
        );

        // All vectors must survive the roundtrip
        for i in 0..num_vectors {
            let deq = restored.get_dequantized(i as u32);
            assert!(deq.is_some(), "Vector {i} lost after save/load roundtrip");
            let deq = deq.unwrap();
            for (j, (&orig, &loaded)) in vectors[i].iter().zip(deq.iter()).enumerate() {
                assert_eq!(
                    orig, loaded,
                    "Vector {i} dimension {j} differs after roundtrip"
                );
            }
        }
    }

    #[test]
    fn test_sq8_norms_stored() {
        let mut storage = NodeStorage::new_sq8(4, 2, 8);

        for i in 0..256 {
            storage.allocate_node();
            let vector: Vec<f32> = (0..4).map(|j| (i + j) as f32).collect();
            storage.set_vector(i as u32, &vector);
        }

        for i in 0..256 {
            let norm = storage.get_norm(i as u32);
            assert!(norm.is_some(), "Norm should be stored for vector {i}");
            assert!(norm.unwrap() >= 0.0, "Norm should be non-negative");
        }
    }
}
