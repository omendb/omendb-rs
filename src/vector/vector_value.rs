use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// PostgreSQL-compatible vector value
///
/// Represents a VECTOR(N) type compatible with pgvector extension.
/// Internally stores float32 values for memory efficiency and SIMD operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorValue {
    /// Vector data (float32 for memory efficiency)
    data: Vec<f32>,

    /// Number of dimensions
    dimensions: usize,
}

impl VectorValue {
    /// Create a new vector from float32 data
    ///
    /// # Arguments
    /// * `data` - Vector data (must not contain NaN or Inf)
    ///
    /// # Returns
    /// VectorValue if validation succeeds
    pub fn new(data: Vec<f32>) -> Result<Self> {
        // Validate: no NaN or Inf
        for (i, &value) in data.iter().enumerate() {
            if value.is_nan() {
                anyhow::bail!("Vector contains NaN at position {}", i);
            }
            if value.is_infinite() {
                anyhow::bail!("Vector contains Inf at position {}", i);
            }
        }

        let dimensions = data.len();
        Ok(VectorValue { data, dimensions })
    }

    /// Create vector from literal string: '[1.0, 2.0, 3.0]'
    ///
    /// Compatible with pgvector syntax.
    pub fn from_literal(s: &str) -> Result<Self> {
        let s = s.trim();

        // Remove brackets
        if !s.starts_with('[') || !s.ends_with(']') {
            anyhow::bail!("Vector literal must be enclosed in brackets: {}", s);
        }

        let inner = &s[1..s.len() - 1];

        // Parse comma-separated values
        let values: Result<Vec<f32>> = inner
            .split(',')
            .map(|v| {
                v.trim()
                    .parse::<f32>()
                    .map_err(|e| anyhow!("Failed to parse value '{}': {}", v.trim(), e))
            })
            .collect();

        Self::new(values?)
    }

    /// Create vector with specific dimensions (for type checking)
    pub fn with_dimensions(data: Vec<f32>, expected_dimensions: usize) -> Result<Self> {
        if data.len() != expected_dimensions {
            anyhow::bail!(
                "Dimension mismatch: expected {}, got {}",
                expected_dimensions,
                data.len()
            );
        }

        Self::new(data)
    }

    /// Get vector dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get vector data as slice
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Convert to PostgreSQL binary format
    ///
    /// Format: [dimensions: u32][unused: u32][values: f32...]
    /// All values in big-endian byte order (PostgreSQL standard)
    pub fn to_postgres_binary(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + self.dimensions * 4);

        // Dimensions (4 bytes, big-endian)
        bytes.extend_from_slice(&(self.dimensions as u32).to_be_bytes());

        // Unused (4 bytes)
        bytes.extend_from_slice(&0u32.to_be_bytes());

        // Values (4 bytes each, big-endian)
        for &value in &self.data {
            bytes.extend_from_slice(&value.to_be_bytes());
        }

        bytes
    }

    /// Parse from PostgreSQL binary format
    pub fn from_postgres_binary(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            anyhow::bail!(
                "Invalid vector binary: too short (need at least 8 bytes, got {})",
                bytes.len()
            );
        }

        // Read dimensions
        let dimensions =
            u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;

        // Validate size
        let expected_size = 8 + dimensions * 4;
        if bytes.len() != expected_size {
            anyhow::bail!(
                "Invalid vector binary: expected {} bytes for {} dimensions, got {}",
                expected_size,
                dimensions,
                bytes.len()
            );
        }

        // Read values
        let mut data = Vec::with_capacity(dimensions);
        for i in 0..dimensions {
            let offset = 8 + i * 4;
            let value = f32::from_be_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            data.push(value);
        }

        Self::new(data)
    }

    /// L2 (Euclidean) distance
    ///
    /// Formula: sqrt(sum((a[i] - b[i])^2))
    pub fn l2_distance(&self, other: &VectorValue) -> Result<f32> {
        if self.dimensions != other.dimensions {
            anyhow::bail!(
                "Dimension mismatch: {} vs {}",
                self.dimensions,
                other.dimensions
            );
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

    /// Inner product (dot product)
    ///
    /// Formula: sum(a[i] * b[i])
    pub fn inner_product(&self, other: &VectorValue) -> Result<f32> {
        if self.dimensions != other.dimensions {
            anyhow::bail!(
                "Dimension mismatch: {} vs {}",
                self.dimensions,
                other.dimensions
            );
        }

        let product: f32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum();

        Ok(product)
    }

    /// Cosine distance
    ///
    /// Formula: 1 - (dot(a, b) / (||a|| * ||b||))
    pub fn cosine_distance(&self, other: &VectorValue) -> Result<f32> {
        if self.dimensions != other.dimensions {
            anyhow::bail!(
                "Dimension mismatch: {} vs {}",
                self.dimensions,
                other.dimensions
            );
        }

        let dot_product = self.inner_product(other)?;
        let norm_a = self.l2_norm();
        let norm_b = other.l2_norm();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(1.0); // Maximum distance for zero vectors
        }

        Ok(1.0 - (dot_product / (norm_a * norm_b)))
    }

    /// L2 norm (magnitude)
    pub fn l2_norm(&self) -> f32 {
        self.data.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// L2 normalize vector (unit vector)
    pub fn l2_normalize(&self) -> VectorValue {
        let norm = self.l2_norm();
        if norm == 0.0 {
            return self.clone(); // Zero vector stays zero
        }

        let data: Vec<f32> = self.data.iter().map(|v| v / norm).collect();
        VectorValue {
            data,
            dimensions: self.dimensions,
        }
    }
}

impl fmt::Display for VectorValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, value) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", value)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_new() {
        let vec = VectorValue::new(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(vec.dimensions(), 3);
        assert_eq!(vec.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vector_reject_nan() {
        let result = VectorValue::new(vec![1.0, f32::NAN, 3.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("NaN"));
    }

    #[test]
    fn test_vector_reject_inf() {
        let result = VectorValue::new(vec![1.0, f32::INFINITY, 3.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Inf"));
    }

    #[test]
    fn test_from_literal() {
        let vec = VectorValue::from_literal("[1.0, 2.0, 3.0]").unwrap();
        assert_eq!(vec.dimensions(), 3);
        assert_eq!(vec.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_from_literal_no_spaces() {
        let vec = VectorValue::from_literal("[1.0,2.0,3.0]").unwrap();
        assert_eq!(vec.dimensions(), 3);
        assert_eq!(vec.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_from_literal_missing_brackets() {
        let result = VectorValue::from_literal("1.0, 2.0, 3.0");
        assert!(result.is_err());
    }

    #[test]
    fn test_postgres_binary_roundtrip() {
        let original = VectorValue::new(vec![1.0, 2.5, -3.75, 4.125]).unwrap();
        let bytes = original.to_postgres_binary();
        let restored = VectorValue::from_postgres_binary(&bytes).unwrap();

        assert_eq!(original, restored);
    }

    #[test]
    fn test_l2_distance() {
        let v1 = VectorValue::new(vec![1.0, 0.0, 0.0]).unwrap();
        let v2 = VectorValue::new(vec![0.0, 1.0, 0.0]).unwrap();

        let dist = v1.l2_distance(&v2).unwrap();
        assert!((dist - 1.414).abs() < 0.001); // sqrt(2)
    }

    #[test]
    fn test_l2_distance_identical() {
        let v1 = VectorValue::new(vec![1.0, 2.0, 3.0]).unwrap();
        let v2 = VectorValue::new(vec![1.0, 2.0, 3.0]).unwrap();

        let dist = v1.l2_distance(&v2).unwrap();
        assert!(dist < 0.0001); // Should be ~0
    }

    #[test]
    fn test_inner_product() {
        let v1 = VectorValue::new(vec![1.0, 2.0, 3.0]).unwrap();
        let v2 = VectorValue::new(vec![4.0, 5.0, 6.0]).unwrap();

        let product = v1.inner_product(&v2).unwrap();
        assert_eq!(product, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn test_cosine_distance() {
        let v1 = VectorValue::new(vec![1.0, 0.0, 0.0]).unwrap();
        let v2 = VectorValue::new(vec![1.0, 0.0, 0.0]).unwrap();

        let dist = v1.cosine_distance(&v2).unwrap();
        assert!(dist < 0.0001); // Same direction = 0 distance
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let v1 = VectorValue::new(vec![1.0, 0.0]).unwrap();
        let v2 = VectorValue::new(vec![0.0, 1.0]).unwrap();

        let dist = v1.cosine_distance(&v2).unwrap();
        assert!((dist - 1.0).abs() < 0.0001); // Orthogonal = distance 1
    }

    #[test]
    fn test_l2_normalize() {
        let v = VectorValue::new(vec![3.0, 4.0]).unwrap();
        let normalized = v.l2_normalize();

        assert!((normalized.l2_norm() - 1.0).abs() < 0.0001);
        assert!((normalized.data()[0] - 0.6).abs() < 0.0001);
        assert!((normalized.data()[1] - 0.8).abs() < 0.0001);
    }

    #[test]
    fn test_display() {
        let v = VectorValue::new(vec![1.0, 2.5, 3.75]).unwrap();
        assert_eq!(format!("{}", v), "[1,2.5,3.75]");
    }

    #[test]
    fn test_dimension_mismatch() {
        let v1 = VectorValue::new(vec![1.0, 2.0]).unwrap();
        let v2 = VectorValue::new(vec![1.0, 2.0, 3.0]).unwrap();

        assert!(v1.l2_distance(&v2).is_err());
        assert!(v1.inner_product(&v2).is_err());
        assert!(v1.cosine_distance(&v2).is_err());
    }
}
