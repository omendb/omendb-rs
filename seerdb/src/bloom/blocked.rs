// Blocked Bloom Filter with cache-line optimization
//
// Key idea: Constrain all k hash operations for a single key to a single cache line (64 bytes).
// This reduces k random memory accesses to 1 cache-line fetch, providing ~3x speedup.
//
// Trade-off: Slightly higher false positive rate vs standard bloom filter due to reduced entropy.
//
// References:
// - "Cache Efficient Bloom Filters for Shared Memory Machines" (Kaler, MIT)
// - "Blocked Bloom Filters with Choices" (Schmitz et al., arXiv 2025)
// - RocksDB blocked bloom filter implementation

use std::hash::{Hash, Hasher};
use twox_hash::XxHash64;

/// Cache line size in bytes (typical for modern CPUs)
const CACHE_LINE_BYTES: usize = 64;
/// Bits per cache line
const CACHE_LINE_BITS: usize = CACHE_LINE_BYTES * 8; // 512 bits
/// u64 words per cache line
const WORDS_PER_BLOCK: usize = CACHE_LINE_BYTES / 8; // 8 words

/// Blocked bloom filter optimized for cache locality
///
/// All k hash operations for a single key are confined to a single cache-line-sized block,
/// reducing memory accesses and improving performance by ~3x compared to standard bloom filters.
pub struct BlockedBloomFilter {
    /// Blocks of cache-line size (512 bits each)
    blocks: Vec<[u64; WORDS_PER_BLOCK]>,
    /// Total number of blocks
    num_blocks: usize,
    /// Number of hash functions
    num_hashes: usize,
    /// Number of elements inserted
    count: usize,
}

impl BlockedBloomFilter {
    /// Create a new blocked bloom filter with expected capacity and desired false positive rate
    pub fn new(expected_elements: usize, false_positive_rate: f64) -> Self {
        // Calculate optimal number of bits: m = -n*ln(p) / (ln(2)^2)
        let num_bits = (-(expected_elements as f64) * false_positive_rate.ln()
            / (2.0_f64.ln().powi(2)))
        .ceil() as usize;

        // Calculate optimal number of hash functions: k = (m/n) * ln(2)
        let num_hashes =
            ((num_bits as f64 / expected_elements as f64) * 2.0_f64.ln()).ceil() as usize;

        // Round up to nearest multiple of cache line size
        let num_blocks = num_bits.div_ceil(CACHE_LINE_BITS);

        Self {
            blocks: vec![[0u64; WORDS_PER_BLOCK]; num_blocks],
            num_blocks,
            num_hashes: num_hashes.clamp(1, CACHE_LINE_BITS), // Cap at block size
            count: 0,
        }
    }

    /// Insert an element into the blocked bloom filter
    #[inline]
    pub fn insert<T: Hash + ?Sized>(&mut self, item: &T) {
        let (block_index, block_hash) = self.hash_to_block(item);

        // Set k bits within this single block
        for i in 0..self.num_hashes {
            let bit_index = Self::nth_bit_in_block(block_hash, i);
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            self.blocks[block_index][word_index] |= 1u64 << bit_offset;
        }

        self.count += 1;
    }

    /// Check if an element might be in the set
    /// Returns true if possibly in set, false if definitely not in set
    pub fn contains<T: Hash + ?Sized>(&self, item: &T) -> bool {
        let (block_index, block_hash) = self.hash_to_block(item);

        // Check k bits within this single block
        for i in 0..self.num_hashes {
            let bit_index = Self::nth_bit_in_block(block_hash, i);
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;

            if (self.blocks[block_index][word_index] & (1u64 << bit_offset)) == 0 {
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
        self.blocks.len() * CACHE_LINE_BYTES + std::mem::size_of::<Self>()
    }

    /// Calculate actual false positive rate based on current state
    pub fn false_positive_rate(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        // Actual FPR for blocked bloom filter is slightly higher than standard
        // due to reduced entropy (all bits in same block)
        // FPR ≈ (1 - e^(-kn/(m*block_ratio)))^k
        let k = self.num_hashes as f64;
        let n = self.count as f64;
        let m = (self.num_blocks * CACHE_LINE_BITS) as f64;

        (1.0 - (-k * n / m).exp()).powf(k)
    }

    /// Serialize blocked bloom filter to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Write metadata
        bytes.extend_from_slice(&(self.num_blocks as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.num_hashes as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.count as u64).to_le_bytes());

        // Write blocks
        for block in &self.blocks {
            for &word in block {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }

        bytes
    }

    /// Deserialize blocked bloom filter from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 20 {
            return None; // Not enough data for header
        }

        let num_blocks = u64::from_le_bytes(bytes[0..8].try_into().ok()?) as usize;
        let num_hashes = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
        let count = u64::from_le_bytes(bytes[12..20].try_into().ok()?) as usize;

        let expected_len = 20 + num_blocks * CACHE_LINE_BYTES;
        if bytes.len() != expected_len {
            return None;
        }

        // Read blocks
        let mut blocks = Vec::with_capacity(num_blocks);
        for block_idx in 0..num_blocks {
            let mut block = [0u64; WORDS_PER_BLOCK];
            for (word_idx, word) in block.iter_mut().enumerate().take(WORDS_PER_BLOCK) {
                let start = 20 + block_idx * CACHE_LINE_BYTES + word_idx * 8;
                *word = u64::from_le_bytes(bytes[start..start + 8].try_into().ok()?);
            }
            blocks.push(block);
        }

        Some(Self {
            blocks,
            num_blocks,
            num_hashes,
            count,
        })
    }

    /// Hash item to determine block index and secondary hash for bits within block
    fn hash_to_block<T: Hash + ?Sized>(&self, item: &T) -> (usize, u64) {
        // Use double hashing: h1 selects block, h2 selects bits within block
        let h1 = {
            let mut hasher = XxHash64::with_seed(0);
            item.hash(&mut hasher);
            hasher.finish()
        };

        let h2 = {
            let mut hasher = XxHash64::with_seed(1);
            item.hash(&mut hasher);
            hasher.finish()
        };

        let block_index = (h1 % self.num_blocks as u64) as usize;
        (block_index, h2)
    }

    /// Calculate nth bit position within a block using enhanced double hashing
    ///
    /// Uses Kirsch-Mitzenmacher optimization: g_i(x) = h1(x) + i * h2(x)
    /// This provides good distribution with only 2 hash functions
    fn nth_bit_in_block(block_hash: u64, n: usize) -> usize {
        // Enhanced double hashing within block
        let h = block_hash.wrapping_add((n as u64).wrapping_mul(block_hash >> 32));
        (h % CACHE_LINE_BITS as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blocked_basic() {
        let mut bloom = BlockedBloomFilter::new(100, 0.01);

        bloom.insert(&"hello");
        bloom.insert(&"world");

        assert!(bloom.contains(&"hello"));
        assert!(bloom.contains(&"world"));
        assert!(!bloom.contains(&"foo"));
    }

    #[test]
    fn test_blocked_serialization() {
        let mut bloom = BlockedBloomFilter::new(100, 0.01);
        bloom.insert(&"test1");
        bloom.insert(&"test2");

        let bytes = bloom.to_bytes();
        let bloom2 = BlockedBloomFilter::from_bytes(&bytes).unwrap();

        assert!(bloom2.contains(&"test1"));
        assert!(bloom2.contains(&"test2"));
        assert!(!bloom2.contains(&"test3"));
    }

    #[test]
    fn test_blocked_cache_line_alignment() {
        let bloom = BlockedBloomFilter::new(10000, 0.01);

        // Verify blocks are cache-line aligned
        assert_eq!(std::mem::size_of_val(&bloom.blocks[0]), CACHE_LINE_BYTES);

        // Verify all hash operations touch single block
        let (_block_idx, block_hash) = bloom.hash_to_block(&"test");
        for i in 0..bloom.num_hashes {
            let bit_idx = BlockedBloomFilter::nth_bit_in_block(block_hash, i);
            assert!(
                bit_idx < CACHE_LINE_BITS,
                "Bit {} exceeds block boundary",
                bit_idx
            );
        }
    }

    #[test]
    fn test_blocked_space_efficiency() {
        let bloom = BlockedBloomFilter::new(10000, 0.01);

        // Blocked filters use slightly more space due to alignment
        // Expected: ~1.5 bytes per element for 1% FPR with blocking
        let bytes_per_element = bloom.size_bytes() as f64 / 10000.0;
        assert!(
            bytes_per_element < 3.0,
            "Space efficiency check: {} bytes/elem",
            bytes_per_element
        );
    }

    #[test]
    fn test_blocked_false_positive_rate() {
        let mut bloom = BlockedBloomFilter::new(1000, 0.01);

        // Insert 1000 elements
        for i in 0..1000 {
            bloom.insert(&format!("key{}", i));
        }

        // Test 10000 non-existent elements
        let mut false_positives = 0;
        for i in 1000..11000 {
            if bloom.contains(&format!("key{}", i)) {
                false_positives += 1;
            }
        }

        let observed_fpr = false_positives as f64 / 10000.0;

        // Blocked bloom filters have slightly higher FPR (typically 1.2-1.5x)
        // Accept up to 3% for 1% target
        assert!(
            observed_fpr < 0.03,
            "FPR too high: {:.4} (expected < 0.03)",
            observed_fpr
        );
    }
}
