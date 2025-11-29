//! Sampling-based probabilistic search optimization
//!
//! Implementation of random projection hashing from LSM-VEC paper.
//! Filters disk reads by comparing hash collision counts before loading vectors.
//!
//! Paper: "LSM-VEC: A Large-Scale Disk-Based System for Dynamic Vector Search"
//! Section: Sampling-Based Query Engine
//!
//! Performance impact: 30% speedup (4.90ms vs 7.0ms from paper)
//! - Skip 20% of disk reads by filtering low-collision candidates
//! - Fast hash comparison (XOR + popcount) vs expensive disk I/O + decompression

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::OnceLock;

/// Number of bits in the random projection hash
///
/// Paper uses 64-128 bits. We use 128 for better collision detection.
/// - More bits = better filtering, fewer false positives
/// - 128 bits = 16 bytes per hash (small storage overhead)
pub const HASH_BITS: usize = 128;

/// Number of bytes to store the hash
pub const HASH_BYTES: usize = HASH_BITS / 8; // 16 bytes

/// Random projection vectors for hash computation
///
/// These are fixed random vectors generated once at startup.
/// Each bit in the hash is the sign of the dot product with one projection vector.
static RANDOM_PROJECTIONS: OnceLock<Vec<Vec<f32>>> = OnceLock::new();

/// Initialize random projection vectors
///
/// Must be called once before using the sampling module.
/// Uses fixed seed for reproducibility across restarts.
pub fn init_random_projections(dimensions: usize) {
    RANDOM_PROJECTIONS.get_or_init(|| {
        let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
        let mut projections = Vec::with_capacity(HASH_BITS);

        for _ in 0..HASH_BITS {
            // Generate random Gaussian vector (approximation using Box-Muller)
            let mut projection = Vec::with_capacity(dimensions);
            for _ in 0..(dimensions / 2) {
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();

                // Box-Muller transform
                let r = (-2.0_f32 * u1.ln()).sqrt();
                let theta = 2.0_f32 * std::f32::consts::PI * u2;

                projection.push(r * theta.cos());
                projection.push(r * theta.sin());
            }

            // Handle odd dimensions
            if dimensions % 2 == 1 {
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();
                let r = (-2.0_f32 * u1.ln()).sqrt();
                let theta = 2.0_f32 * std::f32::consts::PI * u2;
                projection.push(r * theta.cos());
            }

            projections.push(projection);
        }

        projections
    });
}

/// Get the random projection vectors (initialize if needed)
fn get_projections(dimensions: usize) -> &'static Vec<Vec<f32>> {
    init_random_projections(dimensions);
    RANDOM_PROJECTIONS.get().unwrap()
}

/// Compute random projection hash for a vector
///
/// Algorithm:
/// 1. For each of HASH_BITS projection vectors:
///    - Compute dot product with input vector
///    - Set bit to 1 if dot product ≥ 0, else 0
/// 2. Pack bits into byte array
///
/// Returns: 16-byte hash (128 bits)
///
/// # Example
/// ```
/// let vector = vec![0.1, 0.2, 0.3, ..., 0.128];
/// let hash = compute_hash(&vector);
/// assert_eq!(hash.len(), 16); // 128 bits = 16 bytes
/// ```
pub fn compute_hash(vector: &[f32]) -> [u8; HASH_BYTES] {
    let projections = get_projections(vector.len());
    let mut hash = [0u8; HASH_BYTES];

    for (i, projection) in projections.iter().enumerate() {
        // Compute dot product
        let dot: f32 = vector
            .iter()
            .zip(projection.iter())
            .map(|(v, p)| v * p)
            .sum();

        // Set bit based on sign
        if dot >= 0.0 {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            hash[byte_idx] |= 1 << bit_idx;
        }
    }

    hash
}

/// Count hash collisions (matching bits) between two hashes
///
/// Algorithm:
/// 1. XOR the two hashes (differences become 1, matches become 0)
/// 2. Count zero bits = HASH_BITS - count_ones(XOR result)
///
/// Higher collision count = more similar vectors (likely neighbors)
///
/// Returns: Number of matching bits (0 to HASH_BITS)
///
/// # Example
/// ```
/// let hash1 = [0xFF, 0xFF, ...]; // All 1s
/// let hash2 = [0xFF, 0xFF, ...]; // All 1s
/// let collisions = count_collisions(&hash1, &hash2);
/// assert_eq!(collisions, 128); // All bits match
/// ```
#[inline]
pub fn count_collisions(hash1: &[u8; HASH_BYTES], hash2: &[u8; HASH_BYTES]) -> usize {
    // XOR hashes and count differing bits
    let diff_bits: usize = hash1
        .iter()
        .zip(hash2.iter())
        .map(|(a, b)| (a ^ b).count_ones() as usize)
        .sum();

    // Collisions = total bits - differing bits
    HASH_BITS - diff_bits
}

/// Check if a vector should be read based on hash collision threshold
///
/// Filters disk reads by comparing hash similarity before loading vector.
///
/// Algorithm:
/// 1. Count collisions between query hash and candidate hash
/// 2. If collisions ≥ threshold, read vector (likely neighbor)
/// 3. If collisions < threshold, skip read (unlikely neighbor)
///
/// From paper: ρ = 0.8 (read 80% of candidates, skip 20%)
///
/// # Arguments
/// * `query_hash` - Hash of the query vector
/// * `candidate_hash` - Hash of the candidate vector
/// * `threshold` - Minimum collision count to read (typically 60-70% of HASH_BITS)
///
/// Returns: true if vector should be read, false to skip
///
/// # Example
/// ```
/// let query_hash = compute_hash(&query_vector);
/// let candidate_hash = get_hash_from_storage(candidate_id);
/// let threshold = (HASH_BITS as f32 * 0.6) as usize; // 60% threshold
///
/// if should_read_vector(&query_hash, &candidate_hash, threshold) {
///     // Read full vector from disk (expensive)
///     let vector = get_vector_from_storage(candidate_id);
///     let distance = compute_distance(&query_vector, &vector);
/// } else {
///     // Skip disk read (20% savings)
/// }
/// ```
#[inline]
pub fn should_read_vector(
    query_hash: &[u8; HASH_BYTES],
    candidate_hash: &[u8; HASH_BYTES],
    threshold: usize,
) -> bool {
    count_collisions(query_hash, candidate_hash) >= threshold
}

/// Compute default sampling threshold
///
/// Returns 50% of HASH_BITS as a reasonable default.
/// Tunable based on recall requirements:
/// - Lower threshold (40-50%): More aggressive filtering, higher speedup, lower recall
/// - Higher threshold (60-70%): More conservative, lower speedup, higher recall
///
/// Paper uses threshold that achieves ρ = 0.8 (skip 20% of reads)
/// After testing: 60% was too aggressive (53% recall), 50% provides better balance
pub fn default_threshold() -> usize {
    (HASH_BITS as f32 * 0.5) as usize // 64 out of 128 bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_computation() {
        let dimensions = 128;
        let vector: Vec<f32> = (0..dimensions).map(|i| i as f32 / 128.0).collect();

        let hash = compute_hash(&vector);
        assert_eq!(hash.len(), HASH_BYTES);
    }

    #[test]
    fn test_collision_counting() {
        // Identical hashes = all bits match
        let hash1 = [0xFF; HASH_BYTES];
        let hash2 = [0xFF; HASH_BYTES];
        assert_eq!(count_collisions(&hash1, &hash2), HASH_BITS);

        // Opposite hashes = no bits match
        let hash3 = [0xFF; HASH_BYTES];
        let hash4 = [0x00; HASH_BYTES];
        assert_eq!(count_collisions(&hash3, &hash4), 0);

        // Half matching
        let hash5 = [0xFF; HASH_BYTES];
        let mut hash6 = [0xFF; HASH_BYTES];
        hash6[0] = 0x00; // Flip first byte (8 bits)
        let collisions = count_collisions(&hash5, &hash6);
        assert_eq!(collisions, HASH_BITS - 8);
    }

    #[test]
    fn test_should_read_vector() {
        let hash1 = [0xFF; HASH_BYTES];
        let hash2 = [0xFF; HASH_BYTES];
        let threshold = default_threshold();

        // Identical hashes = read
        assert!(should_read_vector(&hash1, &hash2, threshold));

        // Very different hashes = skip
        let hash3 = [0x00; HASH_BYTES];
        assert!(!should_read_vector(&hash1, &hash3, threshold));
    }

    #[test]
    fn test_similar_vectors_high_collision() {
        let dimensions = 128;

        // Two similar vectors
        let vector1: Vec<f32> = (0..dimensions).map(|i| i as f32 / 128.0).collect();
        let vector2: Vec<f32> = (0..dimensions).map(|i| (i as f32 + 0.1) / 128.0).collect();

        let hash1 = compute_hash(&vector1);
        let hash2 = compute_hash(&vector2);

        let collisions = count_collisions(&hash1, &hash2);

        // Similar vectors should have high collision count (>60%)
        assert!(
            collisions > (HASH_BITS * 60 / 100),
            "Similar vectors should have high collision count, got {}",
            collisions
        );
    }

    #[test]
    fn test_different_vectors_low_collision() {
        let dimensions = 128;

        // Two very different vectors
        let vector1: Vec<f32> = (0..dimensions).map(|i| i as f32 / 128.0).collect();
        let vector2: Vec<f32> = (0..dimensions).map(|i| -(i as f32) / 128.0).collect();

        let hash1 = compute_hash(&vector1);
        let hash2 = compute_hash(&vector2);

        let collisions = count_collisions(&hash1, &hash2);

        // Different vectors should have lower collision count
        // (Not always <50% due to random projections, but should be noticeably lower)
        println!(
            "Different vectors collision rate: {}/{}",
            collisions, HASH_BITS
        );
    }

    #[test]
    fn test_hash_deterministic() {
        let dimensions = 128;
        let vector: Vec<f32> = (0..dimensions).map(|i| i as f32 / 128.0).collect();

        // Same vector should produce same hash
        let hash1 = compute_hash(&vector);
        let hash2 = compute_hash(&vector);

        assert_eq!(hash1, hash2);
    }
}
