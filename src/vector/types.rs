//! Vector types and distance functions
//!
//! Provides the core Vector type with support for:
//! - L2 (Euclidean) distance
//! - Dot product (inner product)
//! - Cosine distance

use anyhow::{anyhow, Result};

/// High-dimensional vector (1536 dimensions for `OpenAI` embeddings)
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    /// Vector dimensions (typically 1536 for `OpenAI`, 768 for other models)
    pub data: Vec<f32>,
}

impl Vector {
    /// Create new vector from f32 array
    #[must_use]
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Get dimensionality
    #[must_use]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Compute L2 (Euclidean) distance between vectors
    pub fn l2_distance(&self, other: &Vector) -> Result<f32> {
        if self.dim() != other.dim() {
            return Err(anyhow!(
                "Dimension mismatch: {} vs {}",
                self.dim(),
                other.dim()
            ));
        }

        let sum: f32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum();

        Ok(sum.sqrt())
    }

    /// Compute dot product (for inner product similarity)
    pub fn dot_product(&self, other: &Vector) -> Result<f32> {
        if self.dim() != other.dim() {
            return Err(anyhow!(
                "Dimension mismatch: {} vs {}",
                self.dim(),
                other.dim()
            ));
        }

        let sum: f32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum();

        Ok(sum)
    }

    /// Compute cosine distance (1 - cosine similarity)
    pub fn cosine_distance(&self, other: &Vector) -> Result<f32> {
        let dot = self.dot_product(other)?;
        let norm_self = self.l2_norm();
        let norm_other = other.l2_norm();

        if norm_self == 0.0 || norm_other == 0.0 {
            return Err(anyhow!("Cannot compute cosine distance for zero vector"));
        }

        let cosine_sim = dot / (norm_self * norm_other);
        Ok(1.0 - cosine_sim)
    }

    /// Compute L2 norm (magnitude) of vector
    #[must_use]
    pub fn l2_norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize vector to unit length (L2 norm = 1.0)
    pub fn normalize(&self) -> Result<Vector> {
        let norm = self.l2_norm();

        if norm == 0.0 {
            return Err(anyhow!("Cannot normalize zero vector"));
        }

        let normalized_data: Vec<f32> = self.data.iter().map(|x| x / norm).collect();
        Ok(Vector::new(normalized_data))
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_l2_distance() {
        let v1 = Vector::new(vec![1.0, 0.0, 0.0]);
        let v2 = Vector::new(vec![0.0, 1.0, 0.0]);

        let dist = v1.l2_distance(&v2).unwrap();
        assert!((dist - 1.414).abs() < 0.01); // sqrt(2) ≈ 1.414
    }

    #[test]
    fn test_vector_dot_product() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);

        let dot = v1.dot_product(&v2).unwrap();
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_dimension_mismatch() {
        let v1 = Vector::new(vec![1.0, 2.0]);
        let v2 = Vector::new(vec![1.0, 2.0, 3.0]);

        assert!(v1.l2_distance(&v2).is_err());
        assert!(v1.dot_product(&v2).is_err());
    }
}
