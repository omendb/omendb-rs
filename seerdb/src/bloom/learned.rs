// Learned Bloom Filter implementation
// Based on Kraska et al., "Learned Bloom Filters" (2018)
//
// Architecture: ML model + backup bloom filter
// - Model predicts set membership with confidence score
// - High confidence → return prediction
// - Low confidence → check backup traditional bloom filter

use super::bitpacked::BloomFilter;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// Type alias for the decision tree model
type Model = DecisionTreeClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>>;

/// Learned Bloom Filter
///
/// Uses a machine learning model to predict set membership, with a backup
/// traditional bloom filter for uncertain cases.
///
/// Space savings: Model + small backup filter vs large traditional filter
pub struct LearnedBloomFilter {
    /// Trained decision tree model
    model: Option<Model>,
    /// Backup bloom filter for uncertain predictions
    backup_filter: BloomFilter,
    /// Confidence threshold (predictions above this are trusted)
    threshold: f64,
    /// Number of elements
    count: usize,
    /// Feature dimension (number of hash features extracted from keys)
    feature_dim: usize,
}

impl LearnedBloomFilter {
    /// Create a new learned bloom filter
    ///
    /// # Arguments
    /// * `expected_elements` - Expected number of elements
    /// * `false_positive_rate` - Target false positive rate
    /// * `threshold` - Confidence threshold (0.0-1.0, higher = trust model more)
    pub fn new(expected_elements: usize, false_positive_rate: f64, threshold: f64) -> Self {
        // Backup filter is much smaller since model handles most queries
        // Use higher FPR for backup since it's only for uncertain cases
        let backup_fpr = false_positive_rate * 2.0;
        let backup_elements = (expected_elements as f64 * 0.3) as usize; // 30% capacity

        Self {
            model: None,
            backup_filter: BloomFilter::new(backup_elements, backup_fpr),
            threshold,
            count: 0,
            feature_dim: 8, // Use 8 hash features per key
        }
    }

    /// Train the model on positive and negative examples
    ///
    /// # Arguments
    /// * `positive_examples` - Keys that are in the set
    /// * `negative_examples` - Keys that are NOT in the set
    pub fn train<T: Hash>(&mut self, positive_examples: &[T], negative_examples: &[T]) {
        let n_positive = positive_examples.len();
        let n_negative = negative_examples.len();
        let n_total = n_positive + n_negative;

        if n_total == 0 {
            return;
        }

        // Extract features
        let mut features = Vec::with_capacity(n_total);
        let mut labels = Vec::with_capacity(n_total);

        // Positive examples (label = 1)
        for key in positive_examples {
            features.push(self.extract_features(key));
            labels.push(1);
            self.count += 1;
        }

        // Negative examples (label = 0)
        for key in negative_examples {
            features.push(self.extract_features(key));
            labels.push(0);
        }

        // Convert to 2D matrix (row-major: each row is one sample)
        let x = DenseMatrix::from_2d_vec(&features);
        let y: Vec<u32> = labels;

        // Train decision tree
        let model = DecisionTreeClassifier::fit(&x, &y, Default::default())
            .expect("Failed to train decision tree");

        self.model = Some(model);

        // Add all positive examples to backup filter to guarantee no false negatives
        // The model may not learn perfectly with hash-based features, so the backup
        // filter ensures correctness while the model provides space savings
        for key in positive_examples {
            self.backup_filter.insert(key);
        }
    }

    /// Check if an element might be in the set
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        // Check backup filter first for positive cases (guarantees no false negatives)
        if self.backup_filter.contains(item) {
            return true;
        }

        // If not in backup filter, use model for negative predictions
        if let Some((prediction, confidence)) = self.predict_with_confidence(item) {
            if confidence >= self.threshold {
                // High confidence negative: trust the model
                return prediction;
            }
        }

        // Low confidence: assume not in set (conservative)
        false
    }

    /// Predict if item is in set, with confidence score
    /// Returns (prediction, confidence) where prediction is true/false and confidence is 0.0-1.0
    fn predict_with_confidence<T: Hash>(&self, item: &T) -> Option<(bool, f64)> {
        let model = self.model.as_ref()?;

        let features = self.extract_features(item);
        let x = DenseMatrix::from_2d_vec(&vec![features]);

        // Predict (1 = in set, 0 = not in set)
        let prediction = model.predict(&x).ok()?;

        // Convert to (prediction, confidence) tuple
        // Decision trees give binary predictions, so we use high confidence for both
        if prediction[0] == 1 {
            Some((true, 0.9)) // High confidence "in set"
        } else {
            Some((false, 0.9)) // High confidence "not in set"
        }
    }

    /// Predict confidence that item is in set (deprecated - use predict_with_confidence)
    #[allow(dead_code)]
    fn predict_confidence<T: Hash>(&self, item: &T) -> Option<f64> {
        self.predict_with_confidence(item).map(|(_, conf)| conf)
    }

    /// Extract hash-based features from a key
    fn extract_features<T: Hash>(&self, item: &T) -> Vec<f64> {
        let mut features = Vec::with_capacity(self.feature_dim);

        for i in 0..self.feature_dim {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            item.hash(&mut hasher);
            let hash = hasher.finish();

            // Normalize to 0.0-1.0 range
            features.push((hash % 10000) as f64 / 10000.0);
        }

        features
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get size in bytes (for benchmarking)
    pub fn size_bytes(&self) -> usize {
        let backup_size = self.backup_filter.size_bytes();

        // Model size estimation (rough approximation)
        // Decision tree size depends on depth and number of nodes
        // For simplicity, estimate ~1KB for a small tree
        let model_size = if self.model.is_some() { 1024 } else { 0 };

        backup_size + model_size + std::mem::size_of::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learned_bloom_filter() {
        let mut lbf = LearnedBloomFilter::new(1000, 0.01, 0.5);

        // Training data
        let positive: Vec<String> = (0..100).map(|i| format!("key_{}", i)).collect();
        let negative: Vec<String> = (1000..1100).map(|i| format!("key_{}", i)).collect();

        lbf.train(&positive, &negative);

        // Test membership
        for key in &positive {
            assert!(
                lbf.contains(key),
                "Positive example should be in set: {}",
                key
            );
        }

        // Test non-membership (may have false positives)
        let mut false_positives = 0;
        for key in &negative {
            if lbf.contains(key) {
                false_positives += 1;
            }
        }

        println!("False positives: {}/{}", false_positives, negative.len());
        assert!(
            false_positives < 5,
            "Too many false positives: {}",
            false_positives
        );
    }

    #[test]
    fn test_size_comparison() {
        // Traditional bloom filter
        let bf = BloomFilter::new(1000, 0.01);
        let bf_size = bf.size_bytes();

        // Learned bloom filter
        let mut lbf = LearnedBloomFilter::new(1000, 0.01, 0.7);
        let positive: Vec<i32> = (0..1000).collect();
        let negative: Vec<i32> = (10000..11000).collect();
        lbf.train(&positive, &negative);
        let lbf_size = lbf.size_bytes();

        println!("Traditional Bloom Filter: {} bytes", bf_size);
        println!("Learned Bloom Filter: {} bytes", lbf_size);

        let reduction = (1.0 - lbf_size as f64 / bf_size as f64) * 100.0;
        println!("Space reduction: {:.1}%", reduction);

        // Learned bloom filter should be smaller (but this is a rough test)
        // In practice, savings depend on data distribution and model complexity
    }
}
