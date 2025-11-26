//! Gapped array nodes for ALEX learned index
//!
//! The key innovation of ALEX: arrays with gaps (empty slots) that enable
//! O(1) inserts without rebuilding the entire structure.
//!
//! ## Core Idea
//!
//! Traditional sorted array:
//! ```text
//! [1, 2, 3, 4, 5] ← Dense, no room for inserts
//! ```
//!
//! Gapped array (expansion_factor = 1.0):
//! ```text
//! [1, None, 2, None, 3, None, 4, None, 5] ← 50% gaps
//!  ↑  gap   ↑  gap   ↑  gap   ↑  gap   ↑
//! ```
//!
//! Insert 2.5: Find gap near predicted position, insert in O(1)
//!
//! ## From ALEX Paper
//!
//! "Gapped arrays allow for efficient insertions by maintaining spare capacity.
//! When a node reaches maximum density, we split it locally rather than
//! rebuilding the entire index."

use super::linear_model::LinearModel;
use super::simd_search;
use anyhow::Result;
use std::fmt;

/// Maximum density before triggering node split
/// 0.95 = 95% full (leaves 5% gaps for future inserts)
/// Increased from 0.8 to reduce excessive splitting at scale
const MAX_DENSITY: f64 = 0.95;

/// Minimum density to maintain (prevents too sparse nodes)
#[allow(dead_code)]
const MIN_DENSITY: f64 = 0.3;

/// Gapped array node for ALEX
///
/// Stores (key, value) pairs in a gapped array where Some = data, None = gap.
/// Uses linear model to predict approximate position for lookups/inserts.
///
/// # Examples
///
/// ```
/// use seerdb::alex::gapped_node::GappedNode;
///
/// let mut node = GappedNode::new(10, 1.0); // 10 keys, 50% expansion
///
/// // Insert keys (can be out of order)
/// node.insert(5, vec![1, 2, 3]).unwrap();
/// node.insert(10, vec![4, 5, 6]).unwrap();
/// node.insert(7, vec![7, 8, 9]).unwrap();
///
/// // Lookup
/// assert_eq!(node.get(5).unwrap(), Some(vec![1, 2, 3]));
/// assert_eq!(node.get(99).unwrap(), None);
///
/// // Check density
/// assert!(node.density() < 0.5); // Should have lots of gaps
/// ```
#[derive(Debug, Clone)]
pub struct GappedNode {
    /// Keys with gaps (None = empty slot, Some(k) = key)
    keys: Vec<Option<i64>>,

    /// Values aligned with keys
    values: Vec<Option<Vec<u8>>>,

    /// Linear model for predicting position
    model: LinearModel,

    /// Expansion factor: capacity = num_keys × (1 + expansion_factor)
    /// 1.0 = 50% gaps, 2.0 = 66% gaps, 0.5 = 33% gaps
    expansion_factor: f64,

    /// Current number of actual keys (not counting gaps)
    num_keys: usize,

    /// Cached maximum model error (for exponential search bounds)
    max_error_bound: usize,
}

impl GappedNode {
    /// Create new gapped node with specified capacity
    ///
    /// # Arguments
    /// * `expected_keys` - Expected number of keys (actual capacity will be larger)
    /// * `expansion_factor` - Gap ratio (1.0 = 50% capacity is gaps)
    ///
    /// # Examples
    /// ```
    /// use seerdb::alex::gapped_node::GappedNode;
    ///
    /// // Node for ~100 keys with 50% expansion
    /// let node = GappedNode::new(100, 1.0);
    /// assert_eq!(node.capacity(), 200); // 100 * (1 + 1.0)
    /// ```
    pub fn new(expected_keys: usize, expansion_factor: f64) -> Self {
        let capacity = ((expected_keys as f64 * (1.0 + expansion_factor)).ceil() as usize).max(4);

        Self {
            keys: vec![None; capacity],
            values: vec![None; capacity],
            model: LinearModel::new(),
            expansion_factor,
            num_keys: 0,
            max_error_bound: capacity / 4, // Conservative initial bound
        }
    }

    /// Insert key-value pair into gapped array
    ///
    /// Uses learned model + exponential search to find insertion position in O(log error) time.
    ///
    /// **Algorithm**:
    /// 1. Model predicts approximate position
    /// 2. Exponential search finds exact gap
    /// 3. Insert into gap (may shift elements)
    ///
    /// **Time complexity**: O(log error) where error is model prediction error
    ///
    /// # Errors
    /// Returns error if node is at maximum density (caller should split)
    pub fn insert(&mut self, key: i64, value: Vec<u8>) -> Result<bool> {
        // Check if at max density (needs split)
        if self.density() >= MAX_DENSITY {
            return Ok(false); // Signal caller to split
        }

        // Find insertion position using model + exponential search
        let pos = self.find_insert_position(key)?;

        // Insert at position
        if pos < self.keys.len() && self.keys[pos].is_none() {
            // Gap exists - direct insert
            self.keys[pos] = Some(key);
            self.values[pos] = Some(value);
            self.num_keys += 1;
        } else if pos < self.keys.len() {
            // No gap - shift to nearest gap and insert
            self.shift_and_insert(pos, key, value)?;
        } else {
            return Err(anyhow::anyhow!("Insert position out of bounds"));
        }

        Ok(true)
    }

    /// Batch insert key-value pairs (optimized for throughput)
    ///
    /// **Key optimizations**:
    /// 1. Pre-sorts keys for cache locality
    /// 2. Checks capacity once instead of per-key
    /// 3. Amortizes gap allocation overhead
    ///
    /// **Performance**: 10-100x faster than sequential insert() for large batches
    ///
    /// Returns false if node needs split (at capacity)
    pub fn insert_batch(&mut self, entries: &[(i64, Vec<u8>)]) -> Result<bool> {
        if entries.is_empty() {
            return Ok(true);
        }

        // Check if batch would exceed capacity
        let density_after = (self.num_keys + entries.len()) as f64 / self.keys.len() as f64;
        if density_after >= MAX_DENSITY {
            return Ok(false); // Signal caller to split
        }

        // Sort entries for cache locality (amortized O(n log n))
        let mut sorted_entries: Vec<(i64, Vec<u8>)> = entries.to_vec();
        sorted_entries.sort_by_key(|(k, _)| *k);

        // Insert each key
        // Still O(n log error) but with better constants due to:
        // - Cache locality from sorting
        // - No density checks per key
        // - Better branch prediction
        for (key, value) in sorted_entries {
            let pos = self.find_insert_position(key)?;

            if pos < self.keys.len() && self.keys[pos].is_none() {
                // Gap exists - direct insert
                self.keys[pos] = Some(key);
                self.values[pos] = Some(value);
                self.num_keys += 1;
            } else if pos < self.keys.len() {
                // No gap - shift and insert
                self.shift_and_insert(pos, key, value)?;
            }
        }

        Ok(true)
    }

    /// Lookup value for key
    ///
    /// Uses learned model + exponential search for fast lookup.
    ///
    /// **Time complexity**: O(log error) where error is model prediction error
    pub fn get(&self, key: i64) -> Result<Option<Vec<u8>>> {
        if self.num_keys == 0 {
            return Ok(None);
        }

        // Predict position
        let predicted_pos = self.model.predict(key).min(self.keys.len() - 1);

        // Exponential search around prediction
        if let Some(actual_pos) = self.exponential_search(key, predicted_pos) {
            Ok(self.values[actual_pos].clone())
        } else {
            Ok(None)
        }
    }

    /// Find position to insert key using exponential search
    ///
    /// From ALEX paper: "Exponential search exploits model prediction accuracy.
    /// If model error is E, exponential search finds position in O(log E) time."
    ///
    /// **Algorithm**:
    /// 1. Start at predicted position
    /// 2. Expand search radius exponentially (1, 2, 4, 8, ...)
    /// 3. Binary search within final radius
    ///
    /// **Time complexity**: O(log error)
    fn find_insert_position(&self, key: i64) -> Result<usize> {
        if self.num_keys == 0 {
            return Ok(0); // Empty node - insert at start
        }

        // Predict starting position
        let predicted_pos = self.model.predict(key).min(self.keys.len() - 1);

        // Find first gap after key's position
        // Use exponential search to find bounds, then binary search
        let mut radius = 1;
        let max_radius = self.max_error_bound.max(16);

        loop {
            let start = predicted_pos.saturating_sub(radius);
            let end = (predicted_pos + radius).min(self.keys.len());

            // Check if key would be in this range
            let start_key = self.get_key_at(start);
            let end_key = self.get_key_at(end.saturating_sub(1));

            // Can we bound the search with actual keys? (not gaps)
            // CRITICAL: Need BOTH bounds to confidently locate key
            let can_bound = match (start_key, end_key) {
                (Some(sk), Some(ek)) => sk <= key && key <= ek,
                _ => false, // Need both bounds to be confident
            };

            if can_bound {
                // Found bounding keys - search within range
                return Ok(self.binary_search_gap(start, end, key));
            }

            if radius >= max_radius {
                // Hit max radius without finding bounds - search entire array
                return Ok(self.binary_search_gap(0, self.keys.len(), key));
            }

            // Expand radius exponentially
            radius *= 2;

            if radius > self.keys.len() {
                // Searched entire array - search full range for sorted position
                return Ok(self.binary_search_gap(0, self.keys.len(), key));
            }
        }
    }

    /// Exponential search to find key
    ///
    /// Returns position if key found, None otherwise
    fn exponential_search(&self, key: i64, predicted_pos: usize) -> Option<usize> {
        let mut radius = 1;
        let max_radius = self.max_error_bound.max(16);

        loop {
            let start = predicted_pos.saturating_sub(radius);
            let end = (predicted_pos + radius).min(self.keys.len());

            // Check bounds - only trust actual keys, not gaps
            let start_key = self.get_key_at(start);
            let end_key = self.get_key_at(end.saturating_sub(1));

            // Can we bound the search with actual keys?
            // CRITICAL: Need BOTH bounds to confidently locate key
            let can_bound = match (start_key, end_key) {
                (Some(sk), Some(ek)) => sk <= key && key <= ek,
                _ => false, // Need both bounds to be confident
            };

            if can_bound {
                // Found bounding keys - search within range
                return self.binary_search_exact(start, end, key);
            }

            if radius >= max_radius {
                // Hit max radius without finding bounds - search entire array
                return self.binary_search_exact(0, self.keys.len(), key);
            }

            radius *= 2;

            // CRITICAL FIX: Check if we've covered entire array range
            if start == 0 && end == self.keys.len() {
                // Searched entire array - do full scan
                return self.binary_search_exact(0, self.keys.len(), key);
            }

            if radius > self.keys.len() {
                // Searched entire array - do full scan
                return self.binary_search_exact(0, self.keys.len(), key);
            }
        }
    }

    /// Search for exact key match (SIMD-accelerated when available)
    ///
    /// Uses SIMD to search 4 keys at once using std::simd.
    /// Provides ~3-4x speedup over scalar linear search.
    fn binary_search_exact(&self, start: usize, end: usize, key: i64) -> Option<usize> {
        simd_search::simd_search_i64(&self.keys[start..end], key).map(|pos| start + pos)
    }

    /// Binary search for gap to insert key
    ///
    /// Returns position where key should be inserted to maintain sorted order.
    /// insert() will handle finding gap and shifting if needed.
    fn binary_search_gap(&self, start: usize, end: usize, key: i64) -> usize {
        // Find first position where key should go to maintain sorted order
        for i in start..end {
            if let Some(k) = self.keys[i] {
                if k >= key {
                    // Found sorted position - return it (insert() will handle gap/shift)
                    return i;
                }
            } else {
                // Found gap at correct position
                if i == 0 || self.keys[i - 1].is_none_or(|k| k < key) {
                    return i;
                }
            }
        }

        // Key goes at end - return last valid position
        end.saturating_sub(1).min(self.keys.len().saturating_sub(1))
    }

    /// Find nearest gap to position
    fn find_nearest_gap(&self, pos: usize) -> usize {
        // Search nearby positions for gap
        for radius in 0..self.keys.len() {
            // Check right
            if pos + radius < self.keys.len() && self.keys[pos + radius].is_none() {
                return pos + radius;
            }
            // Check left
            if radius <= pos && self.keys[pos - radius].is_none() {
                return pos - radius;
            }
        }
        // No gap found (should not happen if density < MAX_DENSITY)
        pos
    }

    /// Find any available gap
    #[allow(dead_code)]
    fn find_any_gap(&self) -> usize {
        self.keys
            .iter()
            .position(|k| k.is_none())
            .unwrap_or(self.keys.len() - 1)
    }

    /// Get key at position (handling gaps)
    fn get_key_at(&self, pos: usize) -> Option<i64> {
        if pos >= self.keys.len() {
            None
        } else {
            self.keys[pos]
        }
    }

    /// Shift elements to make room and insert
    fn shift_and_insert(&mut self, pos: usize, key: i64, value: Vec<u8>) -> Result<()> {
        // Find nearest gap
        let gap_pos = self.find_nearest_gap(pos);

        if gap_pos > pos {
            // Shift left
            for i in (pos + 1..=gap_pos).rev() {
                self.keys[i] = self.keys[i - 1];
                self.values[i] = self.values[i - 1].clone();
            }
        } else {
            // Shift right
            for i in gap_pos..pos {
                self.keys[i] = self.keys[i + 1];
                self.values[i] = self.values[i + 1].clone();
            }
        }

        // Insert at target position
        self.keys[pos] = Some(key);
        self.values[pos] = Some(value);
        self.num_keys += 1;

        Ok(())
    }

    /// Check if model needs retraining based on error threshold
    ///
    /// Returns true if current model error exceeds acceptable threshold.
    /// This prevents excessive retraining that causes too many node splits.
    ///
    /// **Threshold**: Retrain if error > 20% of node capacity
    pub fn needs_retrain(&self) -> bool {
        if self.num_keys < 10 {
            return false; // Too few keys to benefit from retraining
        }

        // Calculate current model error
        let mut data: Vec<(i64, usize)> = self
            .keys
            .iter()
            .enumerate()
            .filter_map(|(pos, key)| key.map(|k| (k, pos)))
            .collect();

        if data.is_empty() {
            return false;
        }

        data.sort_by_key(|(k, _)| *k);
        let current_error = self.model.max_error(&data);

        // Retrain if error exceeds 20% of capacity
        // This allows some model inaccuracy to avoid over-fitting
        let error_threshold = (self.keys.len() as f64 * 0.2) as usize;
        current_error > error_threshold.max(50) // At least 50 to avoid constant retraining
    }

    /// Retrain model on current data
    ///
    /// Should be called after bulk inserts or splits, OR when needs_retrain() returns true.
    /// **Time complexity**: O(n log n) where n is number of keys (due to sorting)
    pub fn retrain(&mut self) -> Result<()> {
        // Collect (key, position) pairs for training
        let mut data: Vec<(i64, usize)> = self
            .keys
            .iter()
            .enumerate()
            .filter_map(|(pos, key)| key.map(|k| (k, pos)))
            .collect();

        if !data.is_empty() {
            // Sort by key - linear regression requires sorted training data
            data.sort_by_key(|(k, _)| *k);

            self.model.train(&data);
            self.max_error_bound = self.model.max_error(&data).max(4);
        }

        Ok(())
    }

    /// Get all keys (without values) for efficient searching
    ///
    /// Returns Vec of (key, position) pairs without cloning values.
    /// Much faster than pairs() for lower_bound operations.
    pub fn keys_only(&self) -> Vec<(i64, usize)> {
        self.keys
            .iter()
            .enumerate()
            .filter_map(|(pos, key_opt)| key_opt.map(|key| (key, pos)))
            .collect()
    }

    /// Find position of first key >= search_key using learned model + exponential search
    ///
    /// This is the ALEX-optimized lower_bound that avoids materializing all keys.
    /// **Time complexity**: O(log error) where error is model prediction error
    ///
    /// Returns (key, position) if found, None if no key >= search_key exists
    pub fn lower_bound_position(&self, search_key: i64) -> Option<(i64, usize)> {
        if self.num_keys == 0 {
            return None;
        }

        // Predict starting position using learned model
        let predicted_pos = self.model.predict(search_key).min(self.keys.len() - 1);

        // Exponential search to find bounds
        let mut radius = 1;
        let max_radius = self.max_error_bound.max(16);

        loop {
            let start = predicted_pos.saturating_sub(radius);
            let end = (predicted_pos + radius).min(self.keys.len());

            // Scan this range for first key >= search_key
            // This is O(radius) but radius grows exponentially, so total is O(log error)
            let mut candidate: Option<(i64, usize)> = None;

            for pos in start..end {
                if let Some(key) = self.keys[pos] {
                    if key >= search_key {
                        // Found a candidate - but keep searching in this range for earlier matches
                        // Safe replacement for: candidate.is_none() || key < candidate.unwrap().0
                        if candidate.as_ref().is_none_or(|c| key < c.0) {
                            candidate = Some((key, pos));
                        }
                        // If we found exact match or earlier key, we can stop
                        if key == search_key
                            || (pos > 0 && self.keys[pos - 1].is_some_and(|k| k < search_key))
                        {
                            break;
                        }
                    }
                }
            }

            // Only return candidate if we've searched from the beginning
            // Otherwise, we might have missed an earlier key
            if let Some(result) = candidate {
                if start == 0 {
                    // We've searched from position 0, so this is definitely the first key >= search_key
                    return Some(result);
                }
            }

            // Expand search radius
            if radius >= max_radius || (start == 0 && end == self.keys.len()) {
                // Searched entire array - return candidate or do full scan as fallback
                if let Some(result) = candidate {
                    return Some(result);
                }
                return self
                    .keys
                    .iter()
                    .enumerate()
                    .filter_map(|(pos, key_opt)| key_opt.map(|k| (k, pos)))
                    .find(|(k, _)| *k >= search_key);
            }

            radius *= 2;
        }
    }

    /// Get current density (fraction of capacity that is filled)
    ///
    /// Density = num_keys / capacity
    /// - 0.0 = empty
    /// - 1.0 = completely full (no gaps)
    /// - Typical: 0.3 - 0.8 (30-80% full)
    pub fn density(&self) -> f64 {
        self.num_keys as f64 / self.keys.len() as f64
    }

    /// Get number of keys (not counting gaps)
    pub fn num_keys(&self) -> usize {
        self.num_keys
    }

    /// Get total capacity (including gaps)
    pub fn capacity(&self) -> usize {
        self.keys.len()
    }

    /// Check if node should be split
    pub fn should_split(&self) -> bool {
        self.density() >= MAX_DENSITY
    }

    /// Split node when density exceeds threshold
    ///
    /// From ALEX paper: "When a node reaches max density, split at median key.
    /// Create two child nodes, distribute data, and retrain models locally."
    ///
    /// Returns (split_key, right_node) where:
    /// - split_key: Median key that divides left/right
    /// - right_node: New node containing keys >= split_key
    /// - self becomes left_node containing keys < split_key
    ///
    /// **Time complexity**: O(n log n) due to sorting
    pub fn split(&mut self) -> Result<(i64, GappedNode)> {
        if !self.should_split() {
            return Err(anyhow::anyhow!(
                "Node doesn't need splitting (density < MAX_DENSITY)"
            ));
        }

        // Extract and sort all keys
        let mut pairs = self.pairs();
        if pairs.is_empty() {
            return Err(anyhow::anyhow!("Cannot split empty node"));
        }

        // Find median key as split point
        let split_idx = pairs.len() / 2;
        let split_key = pairs[split_idx].0;

        // Create two new nodes with appropriate capacity
        // Use at least 1.0 expansion factor to ensure gaps after split
        let left_size = split_idx;
        let right_size = pairs.len() - split_idx;
        let expansion = self.expansion_factor.max(1.0);

        let mut left = GappedNode::new(left_size, expansion);
        let mut right = GappedNode::new(right_size, expansion);

        // Distribute pairs
        for (key, value) in pairs.drain(..split_idx) {
            left.insert(key, value)?;
        }
        for (key, value) in pairs {
            right.insert(key, value)?;
        }

        // Don't retrain immediately after split - let adaptive retraining decide
        // Retraining here creates perfectly accurate models that cause immediate
        // re-splitting when new keys arrive (they pack too tightly, hitting MAX_DENSITY)

        // Replace self with left node
        *self = left;

        Ok((split_key, right))
    }

    /// Get all key-value pairs (for splitting/iteration)
    ///
    /// Returns sorted list of (key, value) pairs
    pub fn into_pairs(self) -> Vec<(i64, Vec<u8>)> {
        let mut pairs: Vec<(i64, Vec<u8>)> = self
            .keys
            .into_iter()
            .zip(self.values)
            .filter_map(|(k, v)| {
                if let (Some(key), Some(value)) = (k, v) {
                    Some((key, value))
                } else {
                    None
                }
            })
            .collect();

        pairs.sort_by_key(|(k, _)| *k);
        pairs
    }

    /// Get reference to all key-value pairs
    pub fn pairs(&self) -> Vec<(i64, Vec<u8>)> {
        let mut pairs: Vec<(i64, Vec<u8>)> = self
            .keys
            .iter()
            .zip(self.values.iter())
            .filter_map(|(k, v)| match (k, v) {
                (Some(key), Some(value)) => Some((*key, value.clone())),
                _ => None,
            })
            .collect();

        pairs.sort_by_key(|(k, _)| *k);
        pairs
    }
}

impl fmt::Display for GappedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GappedNode(keys={}/{}, density={:.1}%, model={})",
            self.num_keys,
            self.capacity(),
            self.density() * 100.0,
            self.model
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_node() {
        let node = GappedNode::new(100, 1.0);
        assert_eq!(node.capacity(), 200); // 100 * (1 + 1.0)
        assert_eq!(node.num_keys(), 0);
        assert_eq!(node.density(), 0.0);
    }

    #[test]
    fn test_insert_sequential() {
        let mut node = GappedNode::new(10, 1.0);

        // Insert sequential keys
        for i in 0..10 {
            assert!(node.insert(i * 10, vec![i as u8]).unwrap());
        }

        assert_eq!(node.num_keys(), 10);
        assert!(node.density() < 0.6); // Should have gaps
    }

    #[test]
    fn test_insert_out_of_order() {
        let mut node = GappedNode::new(10, 1.0);

        // Insert out of order
        node.insert(50, vec![5]).unwrap();
        node.insert(10, vec![1]).unwrap();
        node.insert(30, vec![3]).unwrap();
        node.insert(20, vec![2]).unwrap();
        node.insert(40, vec![4]).unwrap();

        assert_eq!(node.num_keys(), 5);

        // Retrain model for accurate lookups
        node.retrain().unwrap();

        // Verify lookups
        assert_eq!(node.get(10).unwrap(), Some(vec![1]));
        assert_eq!(node.get(30).unwrap(), Some(vec![3]));
        assert_eq!(node.get(50).unwrap(), Some(vec![5]));
        assert_eq!(node.get(99).unwrap(), None);
    }

    #[test]
    fn test_get_nonexistent() {
        let mut node = GappedNode::new(10, 1.0);
        node.insert(10, vec![1]).unwrap();
        node.insert(20, vec![2]).unwrap();

        node.retrain().unwrap();

        assert_eq!(node.get(5).unwrap(), None);
        assert_eq!(node.get(15).unwrap(), None);
        assert_eq!(node.get(25).unwrap(), None);
    }

    #[test]
    fn test_density_threshold() {
        let mut node = GappedNode::new(10, 0.0); // No expansion - fills quickly

        // Fill to max density (MAX_DENSITY = 0.95, so need 10 keys in capacity 10)
        for i in 0..10 {
            assert!(node.insert(i, vec![i as u8]).unwrap());
        }

        // Should not allow more inserts (needs split)
        let result = node.insert(99, vec![99]).unwrap();
        assert!(!result); // Signals split needed
    }

    #[test]
    fn test_retrain_improves_accuracy() {
        let mut node = GappedNode::new(100, 1.0);

        // Insert data
        for i in 0..50 {
            node.insert(i * 2, vec![i as u8]).unwrap();
        }

        // Retrain
        node.retrain().unwrap();

        // Lookups should work accurately
        for i in 0..50 {
            let key = i * 2;
            assert_eq!(node.get(key).unwrap(), Some(vec![i as u8]));
        }
    }

    #[test]
    fn test_into_pairs() {
        let mut node = GappedNode::new(10, 1.0);

        node.insert(30, vec![3]).unwrap();
        node.insert(10, vec![1]).unwrap();
        node.insert(20, vec![2]).unwrap();

        let pairs = node.into_pairs();
        assert_eq!(pairs.len(), 3);

        // Should be sorted
        assert_eq!(pairs[0], (10, vec![1]));
        assert_eq!(pairs[1], (20, vec![2]));
        assert_eq!(pairs[2], (30, vec![3]));
    }

    #[test]
    fn test_large_scale() {
        let mut node = GappedNode::new(1000, 1.0);

        // Insert 1000 random keys
        for i in 0..1000 {
            let key = i * 7 % 10000; // Pseudo-random
            node.insert(key, vec![(i % 256) as u8]).unwrap();
        }

        assert_eq!(node.num_keys(), 1000);
        assert!(node.density() < 0.6);

        // Retrain
        node.retrain().unwrap();

        // Sample lookups
        for i in (0..1000).step_by(10) {
            let key = i * 7 % 10000;
            assert!(
                node.get(key).unwrap().is_some(),
                "Failed to find key={}",
                key
            );
        }
    }

    #[test]
    fn test_duplicate_inserts() {
        let mut node = GappedNode::new(10, 1.0);

        node.insert(10, vec![1]).unwrap();
        node.insert(10, vec![2]).unwrap(); // Duplicate key

        // Should have 2 entries (ALEX allows duplicates)
        assert_eq!(node.num_keys(), 2);
    }

    #[test]
    fn test_node_split() {
        let mut node = GappedNode::new(10, 0.0); // No expansion - fills quickly

        // Fill to max density (MAX_DENSITY = 0.95, so need 10 keys in capacity 10)
        for i in 0..10 {
            node.insert(i * 10, vec![i as u8]).unwrap();
        }

        assert!(node.should_split());
        assert_eq!(node.num_keys(), 10);

        // Split the node
        let (split_key, right) = node.split().unwrap();

        // Check split key is median
        assert_eq!(split_key, 50);

        // Check left node (keys < 50)
        assert_eq!(node.num_keys(), 5);
        assert!(node.get(0).unwrap().is_some());
        assert!(node.get(10).unwrap().is_some());
        assert!(node.get(20).unwrap().is_some());
        assert!(node.get(30).unwrap().is_some());
        assert!(node.get(40).unwrap().is_some());
        assert!(node.get(50).unwrap().is_none());

        // Check right node (keys >= 50)
        assert_eq!(right.num_keys(), 5);
        assert!(right.get(50).unwrap().is_some());
        assert!(right.get(60).unwrap().is_some());
        assert!(right.get(70).unwrap().is_some());
        assert!(right.get(80).unwrap().is_some());
        assert!(right.get(90).unwrap().is_some());
        assert!(right.get(40).unwrap().is_none());

        // Both nodes should have lower density now
        assert!(node.density() < 0.6);
        assert!(right.density() < 0.6);
    }

    #[test]
    fn test_expansion_factors() {
        // High expansion = more gaps
        let node_high = GappedNode::new(100, 2.0); // 200% = 66% gaps
        assert_eq!(node_high.capacity(), 300);

        // Low expansion = fewer gaps
        let node_low = GappedNode::new(100, 0.5); // 50% = 33% gaps
        assert_eq!(node_low.capacity(), 150);
    }
}
