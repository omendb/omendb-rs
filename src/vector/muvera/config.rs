//! Multi-vector configuration.

/// Configuration for multi-vector (ColBERT-style) document storage.
///
/// Multi-vector stores encode documents as sets of token embeddings, enabling
/// late-interaction retrieval patterns like ColBERT's MaxSim scoring.
///
/// # Quick Start
///
/// ```rust
/// use omendb::MultiVectorConfig;
///
/// // Sensible defaults (good for most use cases)
/// let config = MultiVectorConfig::default();
///
/// // Higher quality (larger index, better recall)
/// let config = MultiVectorConfig::quality();
///
/// // Faster search (smaller index, lower recall)
/// let config = MultiVectorConfig::fast();
/// ```
///
/// # How It Works
///
/// Documents with variable-length token sets are encoded into fixed-dimensional
/// vectors (FDEs) that approximate MaxSim similarity via dot product. This enables
/// efficient ANN search followed by exact MaxSim reranking.
///
/// The encoding dimension is: `repetitions × 2^partition_bits × token_dim`
///
/// | Preset    | Repetitions | Partitions | FDE Size (128D tokens) |
/// |-----------|-------------|------------|------------------------|
/// | fast()    | 3           | 8          | 3,072                  |
/// | default() | 5           | 8          | 5,120                  |
/// | quality() | 10          | 16         | 20,480                 |
#[derive(Debug, Clone)]
pub struct MultiVectorConfig {
    /// Number of independent hash repetitions. Higher = better approximation quality.
    /// Range: 3-40, default: 5.
    pub repetitions: u8,

    /// Log2 of partition count (creates 2^partition_bits buckets per repetition).
    /// Range: 3-6, default: 3 (8 partitions).
    pub partition_bits: u8,

    /// Random seed for reproducible encoding. Default: 42.
    pub seed: u64,
}

impl Default for MultiVectorConfig {
    fn default() -> Self {
        Self {
            repetitions: 5,
            partition_bits: 3, // 8 partitions
            seed: 42,
        }
    }
}

impl MultiVectorConfig {
    /// Fast preset: smaller index, faster search, lower recall.
    ///
    /// Good for prototyping or when search speed is critical.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            repetitions: 3,
            partition_bits: 3, // 8 partitions
            seed: 42,
        }
    }

    /// Quality preset: larger index, slower search, higher recall.
    ///
    /// Good for production when recall matters more than latency.
    #[must_use]
    pub fn quality() -> Self {
        Self {
            repetitions: 10,
            partition_bits: 4, // 16 partitions
            seed: 42,
        }
    }

    /// Create custom configuration.
    ///
    /// # Arguments
    ///
    /// * `repetitions` - Number of hash repetitions (3-40). More = better quality.
    /// * `partition_bits` - Log2 of partitions (3-6). More = finer granularity.
    /// * `seed` - Random seed for reproducibility.
    #[must_use]
    pub fn custom(repetitions: u8, partition_bits: u8, seed: u64) -> Self {
        Self {
            repetitions,
            partition_bits,
            seed,
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

    // Internal: for backwards compatibility with encoder
    pub(crate) fn k_sim(&self) -> u8 {
        self.partition_bits
    }

    pub(crate) fn r_reps(&self) -> u8 {
        self.repetitions
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
    fn test_presets() {
        let fast = MultiVectorConfig::fast();
        assert_eq!(fast.repetitions, 3);
        assert_eq!(fast.partitions(), 8);

        let quality = MultiVectorConfig::quality();
        assert_eq!(quality.repetitions, 10);
        assert_eq!(quality.partitions(), 16);
    }

    #[test]
    fn test_encoded_dimension() {
        let config = MultiVectorConfig::default();
        // 5 reps * 8 partitions * 128 = 5,120
        assert_eq!(config.encoded_dimension(128), 5120);

        let quality = MultiVectorConfig::quality();
        // 10 reps * 16 partitions * 128 = 20,480
        assert_eq!(quality.encoded_dimension(128), 20480);
    }

    #[test]
    fn test_custom() {
        let config = MultiVectorConfig::custom(20, 5, 123);
        assert_eq!(config.repetitions, 20);
        assert_eq!(config.partitions(), 32);
        assert_eq!(config.seed, 123);
    }
}
