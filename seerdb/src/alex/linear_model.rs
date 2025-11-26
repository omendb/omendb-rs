//! Linear regression models for ALEX learned index
//!
//! ALEX uses simple linear models (y = slope * x + intercept) to predict
//! positions in sorted arrays. Linear models are:
//! - Fast to train: O(n) single pass
//! - Fast to predict: O(1) multiplication + addition
//! - Good enough: For sorted data, linear model approximates CDF well
//!
//! From ALEX paper: "Linear models provide a good balance between prediction
//! accuracy and computational cost for learned indexes."

use std::fmt;

/// Linear regression model: y = slope * x + intercept
///
/// Used to predict position of a key in a sorted array.
/// Trained using least squares regression on (key, position) pairs.
#[derive(Debug, Clone)]
pub struct LinearModel {
    slope: f64,
    intercept: f64,
}

impl LinearModel {
    /// Create a new untrained linear model
    ///
    /// Default model: y = x (identity function)
    pub fn new() -> Self {
        Self {
            slope: 1.0,
            intercept: 0.0,
        }
    }

    /// Train model using least squares linear regression
    ///
    /// Given (key, position) pairs, finds best-fit line that minimizes
    /// squared error: min Σ(predicted_pos - actual_pos)²
    ///
    /// **Algorithm**: Standard least squares regression
    /// ```text
    /// slope = (n*Σ(xy) - Σ(x)*Σ(y)) / (n*Σ(x²) - (Σ(x))²)
    /// intercept = (Σ(y) - slope*Σ(x)) / n
    /// ```
    ///
    /// **Time complexity**: O(n) single pass over data
    ///
    /// For large datasets (>10K), consider `train_sampled()` for 10-100x speedup.
    ///
    /// # Arguments
    /// * `data` - Slice of (key, position) pairs. Keys should be sorted.
    ///
    /// # Examples
    /// ```
    /// use seerdb::alex::linear_model::LinearModel;
    ///
    /// let data = vec![(0, 0), (10, 1), (20, 2), (30, 3)];
    /// let mut model = LinearModel::new();
    /// model.train(&data);
    ///
    /// // Model should predict positions close to actual
    /// assert!((model.predict(15) as i64 - 1).abs() <= 1);
    /// ```
    pub fn train(&mut self, data: &[(i64, usize)]) {
        // For large datasets, use adaptive sampling (CDFShop SIGMOD 2024)
        const SAMPLING_THRESHOLD: usize = 10_000;

        if data.len() >= SAMPLING_THRESHOLD {
            // Use √n sampling for 10-100x speedup
            let sample_size = (data.len() as f64).sqrt() as usize;
            self.train_sampled(data, sample_size);
        } else {
            // Small dataset - train on all data
            self.train_full(data);
        }
    }

    /// Train on full dataset (original implementation)
    ///
    /// Used internally for small datasets or when full accuracy is needed.
    /// Can also be called directly for benchmarking/comparison.
    pub fn train_full(&mut self, data: &[(i64, usize)]) {
        if data.is_empty() {
            // No data - keep identity function
            return;
        }

        if data.len() == 1 {
            // Single point - map to that position
            let (_key, pos) = data[0];
            self.slope = 0.0;
            self.intercept = pos as f64;
            return;
        }

        // Compute statistics in single pass
        let n = data.len() as f64;
        let sum_x = data.iter().map(|(k, _)| *k as f64).sum::<f64>();
        let sum_y = data.iter().map(|(_, p)| *p as f64).sum::<f64>();
        let sum_xy = data.iter().map(|(k, p)| *k as f64 * *p as f64).sum::<f64>();
        let sum_x2 = data.iter().map(|(k, _)| (*k as f64).powi(2)).sum::<f64>();

        // Compute slope and intercept
        let denominator = n * sum_x2 - sum_x * sum_x;

        if denominator.abs() < 1e-10 {
            // All keys are identical - vertical line
            // Map all to average position
            self.slope = 0.0;
            self.intercept = sum_y / n;
        } else {
            self.slope = (n * sum_xy - sum_x * sum_y) / denominator;
            self.intercept = (sum_y - self.slope * sum_x) / n;
        }
    }

    /// Train on sampled dataset (CDFShop SIGMOD 2024)
    ///
    /// Adaptive sampling: Train on √n samples instead of n keys for 10-100x speedup.
    ///
    /// **Algorithm**: Stratified sampling to preserve data distribution
    /// 1. Divide data into sample_size buckets
    /// 2. Sample one point from each bucket
    /// 3. Train linear regression on samples
    ///
    /// **Performance**:
    /// - 1M keys: Train on 1000 samples instead of 1M (1000x faster!)
    /// - Accuracy: Similar to full training for monotonic data
    /// - Best for: Sorted/nearly-sorted data (ALEX's primary use case)
    ///
    /// **Time complexity**: O(sample_size) vs O(n) for full training
    ///
    /// # Arguments
    /// * `data` - Full dataset (sorted)
    /// * `sample_size` - Number of samples to use (typically √n)
    pub fn train_sampled(&mut self, data: &[(i64, usize)], sample_size: usize) {
        if data.is_empty() {
            return;
        }

        if sample_size >= data.len() {
            // Sample size >= data size, just train on full data
            self.train_full(data);
            return;
        }

        // Stratified sampling: divide data into equal buckets and sample from each
        let stride = data.len() / sample_size;
        let samples: Vec<(i64, usize)> = (0..sample_size)
            .map(|i| {
                let idx = (i * stride).min(data.len() - 1);
                data[idx]
            })
            .collect();

        // Train on sampled data (O(sample_size) instead of O(n))
        self.train_full(&samples);
    }

    /// Predict position for given key
    ///
    /// Returns predicted array index for key using: y = slope * key + intercept
    ///
    /// **Time complexity**: O(1)
    ///
    /// # Arguments
    /// * `key` - Key to predict position for
    ///
    /// # Returns
    /// Predicted position (clamped to valid array indices)
    ///
    /// # Examples
    /// ```
    /// use seerdb::alex::linear_model::LinearModel;
    ///
    /// let data = vec![(0, 0), (100, 100), (200, 200)];
    /// let mut model = LinearModel::new();
    /// model.train(&data);
    ///
    /// // Predict position for key=50 (should be near 50)
    /// let pos = model.predict(50);
    /// assert!((pos as i64 - 50).abs() < 10);
    /// ```
    pub fn predict(&self, key: i64) -> usize {
        let pos = self.slope * key as f64 + self.intercept;
        // Clamp to valid range (non-negative)
        pos.max(0.0) as usize
    }

    /// Get model slope
    pub fn slope(&self) -> f64 {
        self.slope
    }

    /// Get model intercept
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Compute maximum prediction error on training data
    ///
    /// Returns worst-case error: max|predicted_pos - actual_pos|
    ///
    /// Used to determine search window size for exponential search.
    ///
    /// **Time complexity**: O(n)
    pub fn max_error(&self, data: &[(i64, usize)]) -> usize {
        data.iter()
            .map(|(key, pos)| {
                let predicted = self.predict(*key);
                (predicted as i64 - *pos as i64).unsigned_abs() as usize
            })
            .max()
            .unwrap_or(0)
    }

    /// Compute average prediction error on training data
    ///
    /// Returns mean absolute error: avg(|predicted_pos - actual_pos|)
    ///
    /// **Time complexity**: O(n)
    pub fn avg_error(&self, data: &[(i64, usize)]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let total_error: usize = data
            .iter()
            .map(|(key, pos)| {
                let predicted = self.predict(*key);
                (predicted as i64 - *pos as i64).unsigned_abs() as usize
            })
            .sum();

        total_error as f64 / data.len() as f64
    }
}

impl Default for LinearModel {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for LinearModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LinearModel(y = {:.6}x + {:.6})",
            self.slope, self.intercept
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_function() {
        // Default model should be y = x
        let model = LinearModel::new();
        assert_eq!(model.predict(0), 0);
        assert_eq!(model.predict(100), 100);
        assert_eq!(model.predict(1000), 1000);
    }

    #[test]
    fn test_perfect_linear_data() {
        // Perfect linear: keys = positions
        let data: Vec<(i64, usize)> = (0..100).map(|i| (i, i as usize)).collect();
        let mut model = LinearModel::new();
        model.train(&data);

        // Should have slope ≈ 1, intercept ≈ 0
        assert!((model.slope() - 1.0).abs() < 0.01);
        assert!(model.intercept().abs() < 0.01);

        // Predictions should be exact
        for i in 0..100 {
            let predicted = model.predict(i);
            assert_eq!(predicted, i as usize);
        }

        // Max error should be 0
        assert_eq!(model.max_error(&data), 0);
    }

    #[test]
    fn test_scaled_data() {
        // Keys = 10 * positions
        let data: Vec<(i64, usize)> = (0..100).map(|i| (i * 10, i as usize)).collect();
        let mut model = LinearModel::new();
        model.train(&data);

        // Should have slope ≈ 0.1, intercept ≈ 0
        assert!((model.slope() - 0.1).abs() < 0.01);
        assert!(model.intercept().abs() < 0.01);

        // Predictions should be close
        for i in 0..100 {
            let predicted = model.predict(i * 10);
            assert!((predicted as i64 - i).abs() <= 1);
        }
    }

    #[test]
    fn test_offset_data() {
        // Keys = positions + 1000 (shifted)
        let data: Vec<(i64, usize)> = (0..100).map(|i| (i + 1000, i as usize)).collect();
        let mut model = LinearModel::new();
        model.train(&data);

        // Should have slope ≈ 1, intercept ≈ -1000
        assert!((model.slope() - 1.0).abs() < 0.01);
        assert!((model.intercept() + 1000.0).abs() < 0.01);

        // Predictions should be exact
        for i in 0..100 {
            let predicted = model.predict(i + 1000);
            assert_eq!(predicted, i as usize);
        }
    }

    #[test]
    fn test_single_point() {
        // Single data point
        let data = vec![(42, 10)];
        let mut model = LinearModel::new();
        model.train(&data);

        // Should map all keys to that position
        assert_eq!(model.predict(0), 10);
        assert_eq!(model.predict(42), 10);
        assert_eq!(model.predict(100), 10);
    }

    #[test]
    fn test_duplicate_keys() {
        // All keys identical
        let data = vec![(5, 0), (5, 1), (5, 2), (5, 3)];
        let mut model = LinearModel::new();
        model.train(&data);

        // Should map to average position (≈ 1.5)
        let predicted = model.predict(5);
        assert!((predicted as i64 - 1).abs() <= 1);
    }

    #[test]
    fn test_sparse_data() {
        // Sparse keys with gaps
        let data = vec![(0, 0), (1000, 1), (2000, 2), (3000, 3)];
        let mut model = LinearModel::new();
        model.train(&data);

        // Should predict intermediate positions
        let mid = model.predict(1500);
        assert!((mid as i64 - 1).abs() <= 1);
    }

    #[test]
    fn test_error_metrics() {
        // Slightly noisy data
        let data = vec![(0, 0), (10, 2), (20, 3), (30, 5)];
        let mut model = LinearModel::new();
        model.train(&data);

        let max_err = model.max_error(&data);
        let avg_err = model.avg_error(&data);

        // Should have some error due to noise
        assert!(max_err > 0);
        assert!(avg_err > 0.0);
        assert!(max_err < 5); // But not too large
    }

    #[test]
    fn test_empty_data() {
        // Empty training data
        let data: Vec<(i64, usize)> = vec![];
        let mut model = LinearModel::new();
        model.train(&data);

        // Should keep identity function
        assert_eq!(model.slope(), 1.0);
        assert_eq!(model.intercept(), 0.0);
    }

    #[test]
    fn test_negative_keys() {
        // Negative keys
        let data: Vec<(i64, usize)> = (-50..50).map(|i| (i, (i + 50) as usize)).collect();
        let mut model = LinearModel::new();
        model.train(&data);

        // Should handle negative keys correctly
        assert_eq!(model.predict(-50), 0);
        assert_eq!(model.predict(0), 50);
        assert_eq!(model.predict(49), 99);
    }

    #[test]
    fn test_large_scale() {
        // 1M data points
        let data: Vec<(i64, usize)> = (0..1_000_000).map(|i| (i as i64, i as usize)).collect();
        let mut model = LinearModel::new();
        model.train(&data);

        // Should have perfect fit (uses sampling automatically for >10K)
        assert!((model.slope() - 1.0).abs() < 0.001);
        assert!(model.intercept().abs() < 0.001);

        // Sample predictions
        assert_eq!(model.predict(0), 0);
        assert_eq!(model.predict(500_000), 500_000);
        assert_eq!(model.predict(999_999), 999_999);
    }

    #[test]
    fn test_cdfshop_sampling() {
        // Test CDFShop adaptive sampling
        let data: Vec<(i64, usize)> = (0..100_000).map(|i| (i as i64, i as usize)).collect();

        // Train with sampling (√n = 316 samples from 100K)
        let mut sampled_model = LinearModel::new();
        let sample_size = (data.len() as f64).sqrt() as usize;
        sampled_model.train_sampled(&data, sample_size);

        // Train on full data for comparison
        let mut full_model = LinearModel::new();
        full_model.train_full(&data);

        // Sampled model should be very close to full model
        assert!((sampled_model.slope() - full_model.slope()).abs() < 0.01);
        assert!((sampled_model.intercept() - full_model.intercept()).abs() < 1.0);

        // Predictions should be accurate
        for i in (0..100_000).step_by(10_000) {
            let sampled_pred = sampled_model.predict(i);
            let actual = i as usize;
            let error = (sampled_pred as i64 - actual as i64).abs();
            assert!(error < 100, "Error too high: {} for key {}", error, i);
        }
    }

    #[test]
    fn test_sampling_threshold() {
        // Data below threshold (10K) - should use full training
        let small_data: Vec<(i64, usize)> = (0..5_000).map(|i| (i as i64, i as usize)).collect();

        let mut model = LinearModel::new();
        model.train(&small_data);

        // Should have perfect fit (used full training)
        assert!((model.slope() - 1.0).abs() < 0.01);
        assert!(model.intercept().abs() < 0.01);

        // Data above threshold (10K) - should use sampling
        let large_data: Vec<(i64, usize)> = (0..50_000).map(|i| (i as i64, i as usize)).collect();

        let mut model2 = LinearModel::new();
        model2.train(&large_data);

        // Should still have good fit (used sampling)
        assert!((model2.slope() - 1.0).abs() < 0.01);
        assert!(model2.intercept().abs() < 1.0);
    }
}
