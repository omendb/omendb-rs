//! Multi-vector configuration.

/// Configuration for multi-vector (ColBERT-style) document storage.
///
/// Multi-vector stores encode documents as sets of token embeddings, enabling
/// late-interaction retrieval patterns like ColBERT's MaxSim scoring.
///
/// # Example
///
/// ```rust
/// use omendb::MultiVectorConfig;
///
/// // Use defaults (good for most cases)
/// let config = MultiVectorConfig::default();
///
/// // Customize for higher quality
/// let config = MultiVectorConfig {
///     repetitions: 10,
///     ..Default::default()
/// };
/// ```
///
/// # Parameters
///
/// - `repetitions`: Number of independent hash functions. Higher = better quality,
///   larger index. Default: 5, range: 3-40.
///
/// - `partition_bits`: Log2 of bucket count per repetition. Default: 3 (8 buckets).
///   Higher values give finer granularity but larger encodings.
///
/// # Encoded Dimension
///
/// The encoded vector size is: `repetitions × 2^partition_bits × token_dim`
///
/// | Config | Encoded Size (128D tokens) |
/// |--------|---------------------------|
/// | default (5, 3) | 5,120 |
/// | (10, 3) | 10,240 |
/// | (5, 4) | 10,240 |
/// | (10, 4) | 20,480 |
#[derive(Debug, Clone)]
pub struct MultiVectorConfig {
    /// Number of independent hash repetitions. Higher = better quality, larger index.
    /// Default: 5, range: 3-40.
    pub repetitions: u8,

    /// Log2 of partition count (2^partition_bits buckets per repetition).
    /// Default: 3 (8 partitions), range: 3-6.
    pub partition_bits: u8,

    /// Random seed for reproducible encoding. Default: 42.
    pub seed: u64,
}

impl Default for MultiVectorConfig {
    fn default() -> Self {
        Self {
            repetitions: 5,
            partition_bits: 3,
            seed: 42,
        }
    }
}

impl MultiVectorConfig {
    /// Fast configuration - smaller encoding, faster search.
    ///
    /// Good for prototyping. Use reranking to maintain quality.
    /// Encoded size: 3 × 8 × token_dim = 24 × token_dim
    #[must_use]
    pub fn fast() -> Self {
        Self {
            repetitions: 3,
            partition_bits: 3,
            seed: 42,
        }
    }

    /// Quality configuration - larger encoding, better approximation.
    ///
    /// Use for production when recall matters.
    /// Encoded size: 10 × 16 × token_dim = 160 × token_dim
    #[must_use]
    pub fn quality() -> Self {
        Self {
            repetitions: 10,
            partition_bits: 4,
            seed: 42,
        }
    }

    /// Calculate the encoded vector dimension for a given token dimension.
    #[must_use]
    pub fn encoded_dimension(&self, token_dim: usize) -> usize {
        self.repetitions as usize * self.partitions() * token_dim
    }

    /// Number of partitions per repetition (2^partition_bits).
    #[must_use]
    pub fn partitions(&self) -> usize {
        1 << self.partition_bits
    }
}

// Keep old name as alias for internal migration
pub type MuveraConfig = MultiVectorConfig;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MultiVectorConfig::default();
        assert_eq!(config.repetitions, 5);
        assert_eq!(config.partition_bits, 3);
        assert_eq!(config.partitions(), 8);
    }

    #[test]
    fn test_encoded_dimension() {
        let config = MultiVectorConfig::default();
        // 5 reps * 8 partitions * 128 = 5,120
        assert_eq!(config.encoded_dimension(128), 5120);

        // Higher quality config
        let config = MultiVectorConfig {
            repetitions: 10,
            partition_bits: 4,
            ..Default::default()
        };
        // 10 reps * 16 partitions * 128 = 20,480
        assert_eq!(config.encoded_dimension(128), 20480);
    }

    #[test]
    fn test_struct_init() {
        let config = MultiVectorConfig {
            repetitions: 20,
            partition_bits: 5,
            seed: 123,
        };
        assert_eq!(config.repetitions, 20);
        assert_eq!(config.partitions(), 32);
        assert_eq!(config.seed, 123);
    }

    #[test]
    fn test_fast_preset() {
        let config = MultiVectorConfig::fast();
        assert_eq!(config.repetitions, 3);
        assert_eq!(config.partition_bits, 3);
        assert_eq!(config.partitions(), 8);
        // 3 * 8 * 128 = 3,072
        assert_eq!(config.encoded_dimension(128), 3072);
    }

    #[test]
    fn test_quality_preset() {
        let config = MultiVectorConfig::quality();
        assert_eq!(config.repetitions, 10);
        assert_eq!(config.partition_bits, 4);
        assert_eq!(config.partitions(), 16);
        // 10 * 16 * 128 = 20,480
        assert_eq!(config.encoded_dimension(128), 20480);
    }
}
