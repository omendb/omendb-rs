// Traditional Bloom Filter implementation
// Based on standard bloom filter with multiple hash functions

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use twox_hash::XxHash64;

/// Traditional Bloom Filter
///
/// Uses k hash functions to set bits in a bit array.
/// False positive rate: (1 - e^(-kn/m))^k
/// where k = number of hash functions, n = number of elements, m = bit array size
#[derive(Debug)]
pub struct BloomFilter {
    /// Bit array
    bits: Vec<bool>,
    /// Number of hash functions
    num_hashes: usize,
    /// Number of elements inserted
    count: usize,
}

impl BloomFilter {
    /// Create a new Bloom filter with optimal parameters for target false positive rate
    ///
    /// # Arguments
    /// * `expected_elements` - Expected number of elements
    /// * `false_positive_rate` - Target false positive rate (e.g., 0.01 for 1%)
    pub fn new(expected_elements: usize, false_positive_rate: f64) -> Self {
        // Optimal bit array size: m = -n * ln(p) / (ln(2)^2)
        let num_bits = (-(expected_elements as f64) * false_positive_rate.ln()
            / (2.0_f64.ln().powi(2)))
        .ceil() as usize;

        // Optimal number of hash functions: k = (m/n) * ln(2)
        let num_hashes =
            ((num_bits as f64 / expected_elements as f64) * 2.0_f64.ln()).ceil() as usize;
        let num_hashes = num_hashes.max(1); // At least one hash function

        Self {
            bits: vec![false; num_bits],
            num_hashes,
            count: 0,
        }
    }

    /// Insert an element into the bloom filter
    pub fn insert<T: Hash + ?Sized>(&mut self, item: &T) {
        for i in 0..self.num_hashes {
            let hash = self.hash(item, i);
            let index = (hash % self.bits.len() as u64) as usize;
            self.bits[index] = true;
        }
        self.count += 1;
    }

    /// Check if an element might be in the set
    /// Returns true if possibly in set, false if definitely not in set
    #[inline]
    pub fn contains<T: Hash + ?Sized>(&self, item: &T) -> bool {
        for i in 0..self.num_hashes {
            let hash = self.hash(item, i);
            let index = (hash % self.bits.len() as u64) as usize;
            if !self.bits[index] {
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
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the size in bytes (for benchmarking)
    pub fn size_bytes(&self) -> usize {
        // Vec<bool> uses ~1 byte per bool in memory (not packed)
        // This is the actual in-memory size, not serialized size
        self.bits.len() + std::mem::size_of::<Self>()
    }

    /// Calculate actual false positive rate based on current state
    #[allow(dead_code)]
    pub fn false_positive_rate(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        // Actual FPR: (1 - e^(-kn/m))^k
        let k = self.num_hashes as f64;
        let n = self.count as f64;
        let m = self.bits.len() as f64;

        (1.0 - (-k * n / m).exp()).powf(k)
    }

    /// Serialize bloom filter to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Write metadata
        bytes.extend_from_slice(&(self.bits.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.num_hashes as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.count as u64).to_le_bytes());

        // Pack bits into bytes (8 bits per byte)
        let mut packed_bits = Vec::new();
        for chunk in self.bits.chunks(8) {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                if bit {
                    byte |= 1 << i;
                }
            }
            packed_bits.push(byte);
        }
        bytes.extend_from_slice(&packed_bits);

        bytes
    }

    /// Deserialize bloom filter from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 20 {
            return None; // Not enough data for metadata
        }

        // Read metadata
        let num_bits = u64::from_le_bytes(bytes[0..8].try_into().ok()?) as usize;
        let num_hashes = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
        let count = u64::from_le_bytes(bytes[12..20].try_into().ok()?) as usize;

        // Unpack bits from bytes
        let packed_bits = &bytes[20..];
        let mut bits = Vec::with_capacity(num_bits);
        for byte in packed_bits {
            for i in 0..8 {
                if bits.len() < num_bits {
                    bits.push((byte & (1 << i)) != 0);
                }
            }
        }

        if bits.len() != num_bits {
            return None;
        }

        Some(Self {
            bits,
            num_hashes,
            count,
        })
    }

    /// Hash function with seed
    #[inline]
    fn hash<T: Hash + ?Sized>(&self, item: &T, seed: usize) -> u64 {
        if seed.is_multiple_of(2) {
            // Use DefaultHasher for even seeds
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            item.hash(&mut hasher);
            hasher.finish()
        } else {
            // Use XxHash64 for odd seeds
            let mut hasher = XxHash64::with_seed(seed as u64);
            item.hash(&mut hasher);
            hasher.finish()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut bf = BloomFilter::new(1000, 0.01);

        // Insert some elements
        bf.insert(&"hello");
        bf.insert(&"world");
        bf.insert(&42);

        // Check membership
        assert!(bf.contains(&"hello"));
        assert!(bf.contains(&"world"));
        assert!(bf.contains(&42));

        // Should not contain (might have false positives, but unlikely with low FPR)
        assert!(!bf.contains(&"not_there"));
    }

    #[test]
    fn test_false_positive_rate() {
        let mut bf = BloomFilter::new(1000, 0.01);

        // Insert elements
        for i in 0..1000 {
            bf.insert(&i);
        }

        // Count false positives
        let mut false_positives = 0;
        for i in 1000..2000 {
            if bf.contains(&i) {
                false_positives += 1;
            }
        }

        let actual_fpr = false_positives as f64 / 1000.0;
        println!("Target FPR: 0.01, Actual FPR: {}", actual_fpr);

        // Should be close to target (within reason)
        assert!(actual_fpr < 0.02, "FPR too high: {}", actual_fpr);
    }

    #[test]
    fn test_size_calculation() {
        let bf = BloomFilter::new(10000, 0.01);
        let size = bf.size_bytes();

        println!("Bloom filter size for 10k elements: {} bytes", size);
        assert!(size > 0);
    }

    #[test]
    fn test_serialization() {
        let mut bf = BloomFilter::new(1000, 0.01);

        // Insert some elements
        bf.insert(&"key1");
        bf.insert(&"key2");
        bf.insert(&"key3");

        // Serialize
        let bytes = bf.to_bytes();

        // Deserialize
        let bf2 = BloomFilter::from_bytes(&bytes).unwrap();

        // Verify same behavior
        assert!(bf2.contains(&"key1"));
        assert!(bf2.contains(&"key2"));
        assert!(bf2.contains(&"key3"));
        assert!(!bf2.contains(&"key4"));
        assert_eq!(bf2.len(), 3);
    }
}
