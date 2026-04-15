//! HNSW index statistics and monitoring

use super::{HNSWIndex, IndexStats};
use crate::vector::VectorEngineView;

impl HNSWIndex {
    pub fn stats(&self) -> IndexStats {
        let num_vectors = VectorEngineView::len(self);
        let mut level_distribution = vec![0; self.params.max_level as usize + 1];
        let mut total_neighbors = 0;
        let mut max_neighbors_l0 = 0;

        for i in 0..num_vectors {
            let level = self.storage.get_node_level(i as u32);
            if (level as usize) < level_distribution.len() {
                level_distribution[level as usize] += 1;
            }

            self.storage.with_neighbors(i as u32, 0, |neighbors| {
                total_neighbors += neighbors.len();
                max_neighbors_l0 = max_neighbors_l0.max(neighbors.len());
            });
        }

        let avg_neighbors_l0 = if num_vectors > 0 {
            total_neighbors as f32 / num_vectors as f32
        } else {
            0.0
        };

        IndexStats {
            num_vectors,
            dimensions: self.storage.vectors.dim,
            entry_point: self.entry_point,
            max_level: self.params.max_level,
            level_distribution,
            avg_neighbors_l0,
            max_neighbors_l0,
            memory_bytes: VectorEngineView::total_memory(self),
            params: self.params.clone(),
            distance_function: self.distance_fn,
            quantization_enabled: false, // Unified flat storage is currently f32
        }
    }
}
