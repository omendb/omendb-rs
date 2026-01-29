//! Token pooling using hierarchical clustering.
//!
//! Reduces multi-vector storage by grouping similar tokens and averaging each cluster.
//! Based on Answer.AI research showing pool_factor=2 achieves 50% storage reduction
//! with 100.6% quality (slight improvement over no pooling).
//!
//! # Algorithm
//!
//! 1. Compute pairwise cosine distances between all tokens
//! 2. Apply Ward's hierarchical clustering (minimizes within-cluster variance)
//! 3. Cut dendrogram at target_clusters = ceil(n / pool_factor)
//! 4. Average tokens within each cluster to produce pooled vectors
//!
//! # Performance
//!
//! - O(n²) for pairwise distances - fine for typical 100-500 tokens (~0.2ms)
//! - O(n² log n) for hierarchical clustering
//!
//! # References
//!
//! - [Answer.AI Token Pooling](https://www.answer.ai/posts/colbert-pooling.html)

/// Pool tokens using hierarchical clustering.
///
/// Groups semantically similar tokens and averages each cluster to reduce
/// token count while preserving retrieval quality.
///
/// # Arguments
///
/// * `tokens` - Input token embeddings (each is a slice of f32)
/// * `pool_factor` - Reduction factor (2 = halve tokens, 3 = reduce to 1/3, etc.)
///
/// # Returns
///
/// Pooled token embeddings. If input has n tokens, output has ceil(n / pool_factor) tokens.
///
/// # Algorithm
///
/// Uses Ward's hierarchical clustering:
/// 1. Compute pairwise cosine distances
/// 2. Build dendrogram via Ward linkage (minimizes variance increase)
/// 3. Cut to target number of clusters
/// 4. Average tokens within each cluster
///
/// # Example
///
/// ```ignore
/// let tokens: Vec<Vec<f32>> = vec![vec![0.1; 128]; 100];
/// let refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();
///
/// // Pool with factor 2: 100 tokens -> 50 tokens
/// let pooled = pool_tokens(&refs, 2);
/// assert_eq!(pooled.len(), 50);
/// ```
pub fn pool_tokens(tokens: &[&[f32]], pool_factor: u8) -> Vec<Vec<f32>> {
    let n = tokens.len();
    let target = n.div_ceil(pool_factor as usize);

    // Skip if too few tokens to pool meaningfully
    if n <= target || n < 2 {
        return tokens.iter().map(|t| t.to_vec()).collect();
    }

    // Step 1: Compute pairwise cosine distances
    let distances = pairwise_cosine_distances(tokens);

    // Step 2: Ward's hierarchical clustering
    let clusters = ward_clustering(&distances, n, target);

    // Step 3: Average tokens within each cluster
    mean_pool_clusters(tokens, &clusters)
}

/// Compute pairwise cosine distances between all tokens.
///
/// Returns a condensed distance matrix (upper triangle, row-major).
/// For n tokens, returns n*(n-1)/2 distances.
fn pairwise_cosine_distances(tokens: &[&[f32]]) -> Vec<f32> {
    let n = tokens.len();
    let mut distances = Vec::with_capacity(n * (n - 1) / 2);

    // Precompute norms for efficiency
    let norms: Vec<f32> = tokens
        .iter()
        .map(|t| {
            let sum_sq: f32 = t.iter().map(|x| x * x).sum();
            sum_sq.sqrt().max(1e-10)
        })
        .collect();

    // Compute condensed distance matrix (upper triangle)
    for i in 0..n {
        for j in (i + 1)..n {
            let dot: f32 = tokens[i]
                .iter()
                .zip(tokens[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            let cosine_sim = dot / (norms[i] * norms[j]);
            // Cosine distance = 1 - similarity, clamped to [0, 2]
            let distance = (1.0 - cosine_sim).clamp(0.0, 2.0);
            distances.push(distance);
        }
    }

    distances
}

/// Get index into condensed distance matrix for pair (i, j) where i < j.
#[inline]
fn condensed_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < j);
    // Formula: index = n*i - i*(i+1)/2 + j - i - 1
    n * i - (i * (i + 1)) / 2 + j - i - 1
}

/// Ward's hierarchical clustering.
///
/// Builds a dendrogram by iteratively merging clusters to minimize
/// the increase in total within-cluster variance (Ward's criterion).
///
/// # Arguments
///
/// * `distances` - Condensed pairwise distance matrix
/// * `n` - Number of original tokens
/// * `target_clusters` - Number of clusters to produce
///
/// # Returns
///
/// Cluster assignments: `clusters[i]` is the cluster ID for token i.
fn ward_clustering(distances: &[f32], n: usize, target_clusters: usize) -> Vec<usize> {
    // Each token starts in its own cluster
    let mut cluster_id: Vec<usize> = (0..n).collect();
    let mut cluster_sizes: Vec<usize> = vec![1; n];
    let mut active: Vec<bool> = vec![true; n];
    let mut num_clusters = n;

    // Distance matrix that gets updated during merges
    // We use a square matrix for simplicity (Ward updates are complex)
    let mut dist_matrix = vec![f32::INFINITY; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = distances[condensed_index(n, i, j)];
            dist_matrix[i * n + j] = d;
            dist_matrix[j * n + i] = d;
        }
        dist_matrix[i * n + i] = 0.0;
    }

    // Merge until we reach target number of clusters
    while num_clusters > target_clusters {
        // Find minimum distance pair among active clusters
        let mut min_dist = f32::INFINITY;
        let mut min_i = 0;
        let mut min_j = 0;

        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                let d = dist_matrix[i * n + j];
                if d < min_dist {
                    min_dist = d;
                    min_i = i;
                    min_j = j;
                }
            }
        }

        // Merge j into i
        let ni = cluster_sizes[min_i];
        let nj = cluster_sizes[min_j];
        let nij = ni + nj;

        // Update distances using Lance-Williams formula for Ward's method
        for k in 0..n {
            if !active[k] || k == min_i || k == min_j {
                continue;
            }

            let nk = cluster_sizes[k];
            let d_ik = dist_matrix[min_i * n + k];
            let d_jk = dist_matrix[min_j * n + k];
            let d_ij = dist_matrix[min_i * n + min_j];

            // Ward's update formula
            let new_dist = ((ni + nk) as f32 * d_ik + (nj + nk) as f32 * d_jk - nk as f32 * d_ij)
                / (nij + nk) as f32;

            dist_matrix[min_i * n + k] = new_dist;
            dist_matrix[k * n + min_i] = new_dist;
        }

        // Update cluster assignments: all tokens in cluster j now belong to cluster i
        for cid in &mut cluster_id {
            if *cid == min_j {
                *cid = min_i;
            }
        }

        cluster_sizes[min_i] = nij;
        active[min_j] = false;
        num_clusters -= 1;
    }

    // Renumber clusters to be contiguous 0..target_clusters
    let mut mapping = vec![usize::MAX; n];
    let mut next_id = 0;
    for cid in &mut cluster_id {
        if mapping[*cid] == usize::MAX {
            mapping[*cid] = next_id;
            next_id += 1;
        }
        *cid = mapping[*cid];
    }

    cluster_id
}

/// Average tokens within each cluster to produce pooled vectors.
fn mean_pool_clusters(tokens: &[&[f32]], clusters: &[usize]) -> Vec<Vec<f32>> {
    let dim = tokens[0].len();
    let num_clusters = clusters.iter().max().map_or(0, |&m| m + 1);

    // Accumulate sums and counts per cluster
    let mut sums: Vec<Vec<f32>> = vec![vec![0.0; dim]; num_clusters];
    let mut counts: Vec<usize> = vec![0; num_clusters];

    for (token, &cluster) in tokens.iter().zip(clusters.iter()) {
        counts[cluster] += 1;
        for (sum, &val) in sums[cluster].iter_mut().zip(token.iter()) {
            *sum += val;
        }
    }

    // Compute means
    sums.into_iter()
        .zip(counts.iter())
        .map(|(mut sum, &count)| {
            let scale = 1.0 / count as f32;
            for val in &mut sum {
                *val *= scale;
            }
            sum
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_factor_2() {
        // 4 tokens -> 2 pooled (factor 2)
        let tokens = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0], // Similar to token 0
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.9, 0.1], // Similar to token 2
        ];
        let refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();

        let pooled = pool_tokens(&refs, 2);
        assert_eq!(pooled.len(), 2);

        // Each pooled vector should have dim 4
        assert_eq!(pooled[0].len(), 4);
        assert_eq!(pooled[1].len(), 4);
    }

    #[test]
    fn test_pool_factor_3() {
        // 9 tokens -> 3 pooled (factor 3)
        let tokens: Vec<Vec<f32>> = (0..9).map(|i| vec![i as f32; 8]).collect();
        let refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();

        let pooled = pool_tokens(&refs, 3);
        assert_eq!(pooled.len(), 3);
    }

    #[test]
    fn test_skip_small() {
        // 3 tokens with pool_factor=2 -> 2 pooled (ceiling division)
        let tokens = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();

        let pooled = pool_tokens(&refs, 2);
        // ceil(3/2) = 2, so should pool
        assert_eq!(pooled.len(), 2);
    }

    #[test]
    fn test_single_token() {
        // 1 token -> 1 (no pooling possible)
        let tokens = vec![vec![1.0, 2.0, 3.0]];
        let refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();

        let pooled = pool_tokens(&refs, 2);
        assert_eq!(pooled.len(), 1);
        assert_eq!(pooled[0], tokens[0]);
    }

    #[test]
    fn test_deterministic() {
        // Same input -> same output
        let tokens = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();

        let pooled1 = pool_tokens(&refs, 2);
        let pooled2 = pool_tokens(&refs, 2);

        assert_eq!(pooled1.len(), pooled2.len());
        for (p1, p2) in pooled1.iter().zip(pooled2.iter()) {
            assert_eq!(p1, p2);
        }
    }

    #[test]
    fn test_condensed_index() {
        // Test condensed matrix indexing
        // For n=4, condensed order: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        assert_eq!(condensed_index(4, 0, 1), 0);
        assert_eq!(condensed_index(4, 0, 2), 1);
        assert_eq!(condensed_index(4, 0, 3), 2);
        assert_eq!(condensed_index(4, 1, 2), 3);
        assert_eq!(condensed_index(4, 1, 3), 4);
        assert_eq!(condensed_index(4, 2, 3), 5);
    }

    #[test]
    fn test_pairwise_distances() {
        // Identical vectors -> distance 0
        let tokens = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();
        let distances = pairwise_cosine_distances(&refs);
        assert_eq!(distances.len(), 1);
        assert!(distances[0].abs() < 1e-6);

        // Orthogonal vectors -> distance 1
        let tokens = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();
        let distances = pairwise_cosine_distances(&refs);
        assert!((distances[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_realistic_scale() {
        // 100 tokens of 128D -> 50 pooled
        let tokens: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect())
            .collect();
        let refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();

        let pooled = pool_tokens(&refs, 2);
        assert_eq!(pooled.len(), 50);

        // Verify dimensions preserved
        for p in &pooled {
            assert_eq!(p.len(), 128);
        }
    }
}
