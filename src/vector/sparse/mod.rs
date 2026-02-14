//! Sparse vector support for learned sparse retrieval (SPLADE, etc.)
//!
//! Provides `SparseVector` for representing high-dimensional sparse embeddings
//! and `SparseIndex` for exact top-k search via inverted index.

mod index;

#[cfg(test)]
mod tests;

pub use index::SparseIndex;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sparse vector with sorted dimension indices and corresponding weights.
///
/// Used for learned sparse retrieval models like SPLADE, where documents
/// and queries are represented as sparse vectors over a vocabulary
/// (typically BERT WordPiece, 30,522 dimensions).
///
/// Indices must be sorted ascending and unique. This invariant is enforced
/// at construction and enables efficient merge-intersection for dot product.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseVector {
    indices: Vec<u32>,
    values: Vec<f32>,
}

impl SparseVector {
    /// Create from parallel arrays. Validates sorted + unique indices and finite weights.
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Result<Self> {
        let sv = Self { indices, values };
        sv.validate()?;
        Ok(sv)
    }

    /// Create from (index, value) pairs. Sorts internally.
    ///
    /// If duplicate indices exist, the first occurrence is kept.
    /// Returns an error if any weight is NaN or infinity.
    pub fn from_pairs(mut pairs: Vec<(u32, f32)>) -> Result<Self> {
        pairs.sort_by_key(|(idx, _)| *idx);
        pairs.dedup_by_key(|(idx, _)| *idx);

        let (indices, values): (Vec<u32>, Vec<f32>) = pairs.into_iter().unzip();
        let sv = Self { indices, values };
        sv.validate_finite()?;
        Ok(sv)
    }

    /// Create from a map of dimension -> weight. Sorts internally.
    ///
    /// Returns an error if any weight is NaN or infinity.
    pub fn from_map(map: HashMap<u32, f32>) -> Result<Self> {
        let mut pairs: Vec<(u32, f32)> = map.into_iter().collect();
        pairs.sort_by_key(|(idx, _)| *idx);

        let (indices, values): (Vec<u32>, Vec<f32>) = pairs.into_iter().unzip();
        let sv = Self { indices, values };
        sv.validate_finite()?;
        Ok(sv)
    }

    /// Dimension indices (sorted ascending, unique).
    #[must_use]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Corresponding weights.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Number of non-zero entries.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Check if the vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Dot product with another sparse vector.
    ///
    /// Uses merge-intersection on sorted indices for O(n + m) time.
    #[must_use]
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let mut result = 0.0f32;
        let mut i = 0;
        let mut j = 0;

        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Equal => {
                    result += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }

        result
    }

    /// Serialize to bytes (postcard format).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        postcard::to_allocvec(self).map_err(|e| anyhow::anyhow!("SparseVector serialize: {e}"))
    }

    /// Deserialize from bytes (postcard format).
    ///
    /// Validates all invariants after deserialization to guard against
    /// corrupted data.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let sv: Self = postcard::from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("SparseVector deserialize: {e}"))?;
        sv.validate()?;
        Ok(sv)
    }

    /// Validate all invariants: length match, sorted-unique indices, finite weights.
    fn validate(&self) -> Result<()> {
        if self.indices.len() != self.values.len() {
            anyhow::bail!(
                "indices and values must have same length: {} vs {}",
                self.indices.len(),
                self.values.len()
            );
        }
        for window in self.indices.windows(2) {
            if window[0] >= window[1] {
                anyhow::bail!(
                    "indices must be sorted ascending and unique: found {} before {}",
                    window[0],
                    window[1]
                );
            }
        }
        self.validate_finite()
    }

    /// Validate all weights are finite (not NaN or infinity).
    fn validate_finite(&self) -> Result<()> {
        for &v in &self.values {
            if !v.is_finite() {
                anyhow::bail!("sparse vector weights must be finite, found {v}");
            }
        }
        Ok(())
    }
}
