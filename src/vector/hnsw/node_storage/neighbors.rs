//! Neighbor access methods for NodeStorage
//!
//! Provides read/write access to neighbor lists at all HNSW levels.
//! Level 0 uses colocated storage (hot path). Upper levels use sparse storage.

use std::borrow::Cow;

use super::NodeStorage;

impl NodeStorage {
    /// Get neighbor count at level 0 (hot path, colocated storage)
    #[inline]
    #[must_use]
    pub fn neighbor_count(&self, id: u32) -> usize {
        let ptr = self.node_ptr(id);
        // SAFETY: Pointer from node_ptr() (bounds checked), first 2 bytes of neighbor region are the count
        unsafe { u16::from_le_bytes([*ptr, *ptr.add(1)]) as usize }
    }

    /// Get neighbor count at any level
    #[inline]
    #[must_use]
    pub fn neighbor_count_at_level(&self, id: u32, level: u8) -> usize {
        if level == 0 {
            return self.neighbor_count(id);
        }
        match self.upper_neighbors.get(&id) {
            Some(levels) => {
                let level_idx = level as usize - 1;
                if level_idx < levels.len() {
                    levels[level_idx].len()
                } else {
                    0
                }
            }
            None => 0,
        }
    }

    /// Zero-copy access to level 0 neighbors (hot path, colocated storage)
    #[inline]
    #[must_use]
    pub fn neighbors(&self, id: u32) -> &[u32] {
        let count = self.neighbor_count(id);
        if count == 0 {
            return &[];
        }
        // Clamp to max_neighbors to prevent buffer overread on corrupt data
        if count > self.max_neighbors {
            tracing::warn!(
                node_id = id,
                stored_count = count,
                max_neighbors = self.max_neighbors,
                "Neighbor count exceeds max_neighbors, clamping"
            );
        }
        let count = count.min(self.max_neighbors);
        let ptr = self.node_ptr(id);
        // SAFETY: Pointer from node_ptr() (bounds checked), count clamped to max_neighbors
        unsafe {
            let neighbors_ptr = ptr.add(self.neighbors_offset) as *const u32;
            std::slice::from_raw_parts(neighbors_ptr, count)
        }
    }

    /// Get neighbors at any level
    ///
    /// Level 0 uses the colocated storage (zero-copy).
    /// Upper levels use the sparse storage.
    #[inline]
    #[must_use]
    pub fn neighbors_at_level(&self, id: u32, level: u8) -> Vec<u32> {
        if level == 0 {
            return self.neighbors(id).to_vec();
        }
        match self.upper_neighbors.get(&id) {
            Some(levels) => {
                let level_idx = level as usize - 1;
                if level_idx < levels.len() {
                    levels[level_idx].clone()
                } else {
                    Vec::new()
                }
            }
            None => Vec::new(),
        }
    }

    /// Get neighbors at any level as Cow (zero-copy for all levels)
    ///
    /// Use this in performance-critical paths to avoid allocation.
    /// Returns borrowed slice for both level 0 (colocated) and upper levels (sparse).
    #[inline]
    #[must_use]
    pub fn neighbors_at_level_cow(&self, id: u32, level: u8) -> Cow<'_, [u32]> {
        if level == 0 {
            Cow::Borrowed(self.neighbors(id))
        } else {
            match self.upper_neighbors.get(&id) {
                Some(levels) => {
                    let level_idx = level as usize - 1;
                    if level_idx < levels.len() {
                        Cow::Borrowed(&levels[level_idx])
                    } else {
                        Cow::Borrowed(&[])
                    }
                }
                None => Cow::Borrowed(&[]),
            }
        }
    }

    /// Set level 0 neighbors (overwrites all, colocated storage)
    pub fn set_neighbors(&mut self, id: u32, neighbors: &[u32]) {
        assert!(
            neighbors.len() <= self.max_neighbors,
            "Too many neighbors: {} > {}",
            neighbors.len(),
            self.max_neighbors
        );
        let ptr = self.node_ptr_mut(id);
        // SAFETY: Pointer from node_ptr_mut() (bounds checked), count bounded by max_neighbors
        unsafe {
            // Write count
            let count = neighbors.len() as u16;
            let count_bytes = count.to_le_bytes();
            *ptr = count_bytes[0];
            *ptr.add(1) = count_bytes[1];

            // Write neighbors
            if !neighbors.is_empty() {
                let neighbors_ptr = ptr.add(self.neighbors_offset) as *mut u32;
                std::ptr::copy_nonoverlapping(neighbors.as_ptr(), neighbors_ptr, neighbors.len());
            }
        }
    }

    /// Set neighbors at any level
    ///
    /// Level 0 uses colocated storage. Upper levels use sparse storage.
    pub fn set_neighbors_at_level(&mut self, id: u32, level: u8, neighbors: Vec<u32>) {
        if level == 0 {
            self.set_neighbors(id, &neighbors);
            return;
        }

        // Ensure upper levels are allocated
        self.allocate_upper_levels(id, level);

        if let Some(levels) = self.upper_neighbors.get_mut(&id) {
            let level_idx = level as usize - 1;
            if level_idx < levels.len() {
                levels[level_idx] = neighbors;
            }
        }
    }

    /// Add a neighbor at a specific level (for incremental construction)
    pub fn add_neighbor(&mut self, id: u32, level: u8, neighbor: u32) {
        if level == 0 {
            // Level 0: append to colocated storage
            let count = self.neighbor_count(id);
            if count >= self.max_neighbors {
                return; // At capacity
            }
            let ptr = self.node_ptr_mut(id);
            // SAFETY: Count checked against max_neighbors, pointer from node_ptr_mut() (bounds checked)
            unsafe {
                // Update count
                let new_count = (count + 1) as u16;
                let count_bytes = new_count.to_le_bytes();
                *ptr = count_bytes[0];
                *ptr.add(1) = count_bytes[1];

                // Write new neighbor
                let neighbors_ptr = ptr.add(self.neighbors_offset) as *mut u32;
                *neighbors_ptr.add(count) = neighbor;
            }
        } else {
            // Upper level: append to sparse storage
            self.allocate_upper_levels(id, level);
            if let Some(levels) = self.upper_neighbors.get_mut(&id) {
                let level_idx = level as usize - 1;
                if level_idx < levels.len() && levels[level_idx].len() < self.max_neighbors_upper {
                    levels[level_idx].push(neighbor);
                }
            }
        }
    }

    /// Check if a neighbor exists at a specific level
    ///
    /// Used during parallel construction to avoid duplicate links.
    #[inline]
    #[must_use]
    pub fn contains_neighbor(&self, id: u32, level: u8, neighbor: u32) -> bool {
        if level == 0 {
            self.neighbors(id).contains(&neighbor)
        } else {
            self.neighbors_at_level(id, level).contains(&neighbor)
        }
    }

    /// Try to add a neighbor at a specific level, returns true if added
    ///
    /// Returns false if:
    /// - The neighbor already exists (duplicate)
    /// - The neighbor list is at capacity
    ///
    /// Used during parallel construction for atomic-style neighbor updates.
    pub fn try_add_neighbor(&mut self, id: u32, level: u8, neighbor: u32) -> bool {
        if self.contains_neighbor(id, level, neighbor) {
            return false; // Already exists
        }

        let max = if level == 0 {
            self.max_neighbors
        } else {
            self.max_neighbors_upper
        };

        if self.neighbor_count_at_level(id, level) >= max {
            return false; // At capacity
        }

        self.add_neighbor(id, level, neighbor);
        true
    }
}
