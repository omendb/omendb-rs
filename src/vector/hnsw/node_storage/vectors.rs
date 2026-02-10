//! Vector access methods for NodeStorage
//!
//! Provides read/write access to vector data in both full precision and SQ8 modes.

use super::NodeStorage;

impl NodeStorage {
    /// Zero-copy access to vector (full precision mode only)
    ///
    /// Panics in SQ8 mode - use `get_dequantized()` instead.
    #[inline]
    #[must_use]
    pub fn vector(&self, id: u32) -> &[f32] {
        assert!(
            !self.sq8,
            "vector() only available in full precision mode, use get_dequantized()"
        );
        let ptr = self.node_ptr(id);
        // SAFETY: Pointer from node_ptr() (bounds checked), dimensions matches allocation layout
        unsafe {
            let vec_ptr = ptr.add(self.vector_offset) as *const f32;
            std::slice::from_raw_parts(vec_ptr, self.dimensions)
        }
    }

    /// Zero-copy access to quantized vector (SQ8 mode only)
    #[inline]
    #[must_use]
    pub fn quantized_vector(&self, id: u32) -> &[u8] {
        assert!(self.sq8, "quantized_vector() only available in SQ8 mode");
        let ptr = self.node_ptr(id);
        // SAFETY: Pointer from node_ptr() (bounds checked), quantized_dim matches allocation layout
        unsafe {
            let vec_ptr = ptr.add(self.vector_offset);
            std::slice::from_raw_parts(vec_ptr, self.dimensions)
        }
    }

    /// Get dequantized vector (handles all quantization modes)
    ///
    /// In full precision mode, returns the vector directly.
    /// In quantized modes before training, returns the vector from the training buffer.
    /// In quantized modes after training, returns the dequantized/reconstructed vector.
    #[must_use]
    pub fn get_dequantized(&self, id: u32) -> Option<Vec<f32>> {
        if !self.sq8 {
            return Some(self.vector(id).to_vec());
        }

        let id_usize = id as usize;
        if self.sq8_trained {
            let params = self.sq8_params.as_ref()?;
            let quantized = self.quantized_vector(id);
            Some(params.dequantize(quantized))
        } else {
            let dim = self.dimensions;
            let start = id_usize * dim;
            let end = start + dim;
            if end <= self.training_buffer.len() {
                Some(self.training_buffer[start..end].to_vec())
            } else {
                None
            }
        }
    }

    /// Zero-copy vector access for full precision mode.
    /// Panics if SQ8 (use get_dequantized instead).
    #[inline]
    #[must_use]
    pub fn get_vector_ref(&self, id: u32) -> &[f32] {
        assert!(
            !self.sq8,
            "get_vector_ref not supported for SQ8, use get_dequantized()"
        );
        self.vector(id)
    }

    /// Get squared norm for a vector (used in L2 decomposition)
    #[inline]
    #[must_use]
    pub fn get_norm(&self, id: u32) -> Option<f32> {
        self.norms.get(id as usize).copied()
    }

    /// Set vector data (handles both full precision and SQ8 modes)
    ///
    /// # Errors
    ///
    /// Returns error if SQ8 quantization training fails.
    pub fn set_vector(&mut self, id: u32, vector: &[f32]) -> Result<(), &'static str> {
        assert_eq!(
            vector.len(),
            self.dimensions,
            "Vector length {} doesn't match dimensions {}",
            vector.len(),
            self.dimensions
        );

        if self.sq8 {
            self.set_vector_sq8(id, vector)?;
        } else {
            // Store vector directly and compute norm
            let ptr = self.node_ptr_mut(id);
            // SAFETY: Pointer from node_ptr_mut() (bounds checked), size = dimensions * 4 matches allocation layout
            unsafe {
                let vec_ptr = ptr.add(self.vector_offset) as *mut f32;
                std::ptr::copy_nonoverlapping(vector.as_ptr(), vec_ptr, self.dimensions);
            }
            // Compute and store squared norm
            let norm_sq: f32 = vector.iter().map(|&x| x * x).sum();
            let id_usize = id as usize;
            if id_usize >= self.norms.len() {
                self.norms.resize(id_usize + 1, 0.0);
            }
            self.norms[id_usize] = norm_sq;
        }
        Ok(())
    }
}
