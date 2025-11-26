// Bloom filter with bit-packed storage
// Uses Vec<u64> instead of Vec<bool> (standard approach, 8x more memory efficient)
// SIMD-friendly for future optimizations

use std::hash::{Hash, Hasher};
use twox_hash::XxHash64;

/// Bloom filter with optimal bit-packed storage (`Vec<u64>`)
pub struct BloomFilter {
    bits: Vec<u64>,    // Bit-packed storage (64 bits per u64)
    num_bits: usize,   // Total number of bits
    num_hashes: usize, // Number of hash functions
    count: usize,      // Number of elements inserted
}

impl BloomFilter {
    /// Create a new bloom filter with expected capacity and desired false positive rate
    pub fn new(expected_elements: usize, false_positive_rate: f64) -> Self {
        // Calculate optimal number of bits: m = -n*ln(p) / (ln(2)^2)
        let num_bits = (-(expected_elements as f64) * false_positive_rate.ln()
            / (2.0_f64.ln().powi(2)))
        .ceil() as usize;

        // Calculate optimal number of hash functions: k = (m/n) * ln(2)
        let num_hashes =
            ((num_bits as f64 / expected_elements as f64) * 2.0_f64.ln()).ceil() as usize;

        // Round up to nearest multiple of 64 for alignment
        let num_words = num_bits.div_ceil(64);

        Self {
            bits: vec![0u64; num_words],
            num_bits,
            num_hashes: num_hashes.max(1), // At least 1 hash
            count: 0,
        }
    }

    /// Insert an element into the bloom filter
    pub fn insert<T: Hash + ?Sized>(&mut self, item: &T) {
        for i in 0..self.num_hashes {
            let hash = self.hash(item, i);
            let index = (hash % self.num_bits as u64) as usize;
            let word_index = index / 64;
            let bit_index = index % 64;
            self.bits[word_index] |= 1u64 << bit_index;
        }
        self.count += 1;
    }

    /// Check if an element might be in the set
    /// Returns true if possibly in set, false if definitely not in set
    pub fn contains<T: Hash + ?Sized>(&self, item: &T) -> bool {
        for i in 0..self.num_hashes {
            let hash = self.hash(item, i);
            let index = (hash % self.num_bits as u64) as usize;
            let word_index = index / 64;
            let bit_index = index % 64;

            if (self.bits[word_index] & (1u64 << bit_index)) == 0 {
                return false; // Definitely not in set
            }
        }
        true // Possibly in set
    }

    /// Get the number of elements inserted
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the bloom filter is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.bits.len() * 8 + std::mem::size_of::<Self>()
    }

    /// Calculate actual false positive rate based on current state
    pub fn false_positive_rate(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        // Actual FPR: (1 - e^(-kn/m))^k
        let k = self.num_hashes as f64;
        let n = self.count as f64;
        let m = self.num_bits as f64;

        (1.0 - (-k * n / m).exp()).powf(k)
    }

    /// Serialize bloom filter to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Write metadata
        bytes.extend_from_slice(&(self.num_bits as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.num_hashes as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.count as u64).to_le_bytes());

        // Write bit-packed data (already in u64 words)
        for &word in &self.bits {
            bytes.extend_from_slice(&word.to_le_bytes());
        }

        bytes
    }

    /// Deserialize bloom filter from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 20 {
            return None; // Not enough data for header
        }

        let num_bits = u64::from_le_bytes(bytes[0..8].try_into().ok()?) as usize;
        let num_hashes = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
        let count = u64::from_le_bytes(bytes[12..20].try_into().ok()?) as usize;

        let num_words = num_bits.div_ceil(64);
        let expected_len = 20 + num_words * 8;

        if bytes.len() != expected_len {
            return None;
        }

        // Read bit-packed data
        let mut bits = Vec::with_capacity(num_words);
        for i in 0..num_words {
            let start = 20 + i * 8;
            let word = u64::from_le_bytes(bytes[start..start + 8].try_into().ok()?);
            bits.push(word);
        }

        Some(Self {
            bits,
            num_bits,
            num_hashes,
            count,
        })
    }

    /// Compute hash for an item with a given seed
    fn hash<T: Hash + ?Sized>(&self, item: &T, seed: usize) -> u64 {
        // Always use XxHash64 for stability (persisted to disk)
        // DefaultHasher is randomized per-process and cannot be used for persistence
        let mut hasher = XxHash64::with_seed(seed as u64);
        item.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitpacked_basic() {
        let mut bloom = BloomFilter::new(100, 0.01);

        bloom.insert(&"hello");
        bloom.insert(&"world");

        assert!(bloom.contains(&"hello"));
        assert!(bloom.contains(&"world"));
        assert!(!bloom.contains(&"foo"));
    }

    #[test]
    fn test_bitpacked_serialization() {
        let mut bloom = BloomFilter::new(100, 0.01);
        bloom.insert(&"test1");
        bloom.insert(&"test2");

        let bytes = bloom.to_bytes();
        let bloom2 = BloomFilter::from_bytes(&bytes).unwrap();

        assert!(bloom2.contains(&"test1"));
        assert!(bloom2.contains(&"test2"));
        assert!(!bloom2.contains(&"test3"));
    }

    #[test]
    fn test_bitpacked_space_efficiency() {
        let bloom = BloomFilter::new(10000, 0.01);

        // Bit-packed should use ~1/8 the space of Vec<bool>
        // Expected: ~1.2 bytes per element for 1% FPR
        let bytes_per_element = bloom.size_bytes() as f64 / 10000.0;
        assert!(
            bytes_per_element < 2.0,
            "Space efficiency check: {} bytes/elem",
            bytes_per_element
        );
    }
}
