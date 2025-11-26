/*
 * Debug test to understand IP-DiskANN graph construction
 */

#[cfg(test)]
mod tests {
    use super::super::index::IPDiskANNIndex;
    use rand::Rng;

    fn generate_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        (0..count)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn debug_small_index() {
        println!("\n=== Debug: 10 vectors ===");

        let vectors = generate_vectors(10, 8); // Small dimension for easy debugging
        let mut index = IPDiskANNIndex::new(8, None);

        // Insert vectors and track graph structure
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(vec.clone()).unwrap();

            // Print graph structure after each insert
            println!("\nAfter inserting vector {}:", i);
            for node_id in 0..=i {
                if let Some(node) = index.graph.get_node(node_id as u32) {
                    println!("  Node {}: out={:?}, in={:?}, degree={}",
                        node_id,
                        node.out_neighbors,
                        node.in_neighbors,
                        node.out_neighbors.len()
                    );
                }
            }
        }

        // Test search
        println!("\n=== Search Test ===");
        let query = &vectors[5];
        let results = index.search(query, 3).unwrap();

        println!("Query vector index: 5");
        println!("Search results:");
        for (i, result) in results.iter().enumerate() {
            println!("  {}: id={}, distance={:.4}", i, result.id, result.distance);
        }

        // Brute force for comparison
        let mut distances: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(id, vec)| {
                let dist = query
                    .iter()
                    .zip(vec.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                (id, dist)
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        println!("\nGround truth (brute force):");
        for (i, (id, dist)) in distances.iter().take(3).enumerate() {
            println!("  {}: id={}, distance={:.4}", i, id, dist);
        }
    }
}
