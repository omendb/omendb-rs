#[cfg(test)]
mod tests {
    use crate::vector::hnsw::storage::NeighborMatrix;

    #[test]
    fn test_stride_limits() {
        let max_levels = 4;
        let m = 64; // This makes M0 = 128
        let mut matrix = NeighborMatrix::new(max_levels, m);
        matrix.ensure_capacity(1);

        // Attempt to store 128 neighbors
        let neighbors: Vec<u32> = (0..128).collect();
        matrix.set_neighbors(0, 0, &neighbors);

        // Read them back
        let read_back = matrix.with_neighbors(0, 0, |n| n.to_vec());
        assert_eq!(
            read_back.len(),
            128,
            "Failed to store exactly 128 neighbors"
        );
    }
}
