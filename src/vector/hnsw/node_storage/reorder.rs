//! BFS reordering for cache-friendly node layout
//!
//! Reorders nodes so that graph traversal follows memory layout,
//! improving cache locality during search.

use super::{NodeStorage, StorageBacking, CACHE_LINE};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::{HashSet, VecDeque};
use std::ptr::NonNull;

impl NodeStorage {
    /// Reorder nodes using BFS traversal for cache-friendly layout
    ///
    /// Returns the old-to-new ID mapping.
    pub fn reorder_bfs(&mut self, entry_point: u32, _max_level: u8) -> Vec<u32> {
        let n = self.len;
        if n == 0 {
            return Vec::new();
        }

        // BFS from entry point to determine optimal ordering
        let mut visited = HashSet::new();
        let mut bfs_order = Vec::with_capacity(n);
        let mut queue = VecDeque::new();

        // Start from entry point at highest level, then traverse down
        queue.push_back(entry_point);
        visited.insert(entry_point);

        while let Some(node_id) = queue.pop_front() {
            bfs_order.push(node_id);

            // Visit neighbors at level 0 (most important for cache locality)
            for &neighbor in self.neighbors(node_id) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        // Add any nodes not reachable from entry point
        for node_id in 0..n as u32 {
            if visited.insert(node_id) {
                bfs_order.push(node_id);
            }
        }

        // Create mapping: old_to_new[old_id] = new_id
        let mut old_to_new = vec![0u32; n];
        for (new_id, &old_id) in bfs_order.iter().enumerate() {
            old_to_new[old_id as usize] = new_id as u32;
        }

        // Reorder the storage (creates new backing, copies data in BFS order)
        if let Err(e) = self.apply_reorder(&bfs_order, &old_to_new) {
            tracing::error!("Failed to apply reorder: {}", e);
            return old_to_new;
        }

        old_to_new
    }

    /// Apply a reordering to the storage
    pub(super) fn apply_reorder(
        &mut self,
        bfs_order: &[u32],
        old_to_new: &[u32],
    ) -> Result<(), String> {
        let n = self.len;
        if n == 0 {
            return Ok(());
        }

        // Allocate new storage
        let new_size = n * self.node_size;
        let layout = Layout::from_size_align(new_size, CACHE_LINE)
            .map_err(|e| format!("Invalid layout for reorder: {e}"))?;
        let new_ptr = unsafe { alloc_zeroed(layout) };
        if new_ptr.is_null() {
            return Err(format!("Allocation failed for reorder: {new_size} bytes"));
        }

        // Copy nodes in BFS order
        for (new_idx, &old_id) in bfs_order.iter().enumerate() {
            let old_ptr = self.node_ptr(old_id);
            let new_node_ptr = unsafe { new_ptr.add(new_idx * self.node_size) };

            // Copy the node data
            unsafe {
                std::ptr::copy_nonoverlapping(old_ptr, new_node_ptr, self.node_size);
            }

            // Update neighbor IDs to use new indices
            let count_ptr = new_node_ptr as *mut u16;
            let count = unsafe { *count_ptr } as usize;
            let neighbors_ptr = unsafe { new_node_ptr.add(self.neighbors_offset) as *mut u32 };

            for i in 0..count.min(self.max_neighbors) {
                let old_neighbor = unsafe { *neighbors_ptr.add(i) };
                if (old_neighbor as usize) < old_to_new.len() {
                    unsafe {
                        *neighbors_ptr.add(i) = old_to_new[old_neighbor as usize];
                    }
                }
            }
        }

        // Reorder norms and sq8_sums
        if !self.norms.is_empty() {
            let old_norms = std::mem::take(&mut self.norms);
            self.norms = vec![0.0; n];
            for (new_idx, &old_id) in bfs_order.iter().enumerate() {
                if (old_id as usize) < old_norms.len() {
                    self.norms[new_idx] = old_norms[old_id as usize];
                }
            }
        }

        if !self.sq8_sums.is_empty() {
            let old_sums = std::mem::take(&mut self.sq8_sums);
            self.sq8_sums = vec![0; n];
            for (new_idx, &old_id) in bfs_order.iter().enumerate() {
                if (old_id as usize) < old_sums.len() {
                    self.sq8_sums[new_idx] = old_sums[old_id as usize];
                }
            }
        }

        // Reorder upper neighbors (HashMap: old_id -> new_id mapping)
        if !self.upper_neighbors.is_empty() {
            let old_upper = std::mem::take(&mut self.upper_neighbors);
            for (new_idx, &old_id) in bfs_order.iter().enumerate() {
                if let Some(levels) = old_upper.get(&old_id) {
                    // Remap neighbor IDs in upper levels
                    let new_levels: Vec<Vec<u32>> = levels
                        .iter()
                        .map(|neighbors| {
                            neighbors
                                .iter()
                                .filter_map(|&old_n| {
                                    if (old_n as usize) < old_to_new.len() {
                                        Some(old_to_new[old_n as usize])
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        })
                        .collect();
                    self.upper_neighbors.insert(new_idx as u32, new_levels);
                }
            }
        }

        // Swap in new backing
        let old_backing = std::mem::replace(
            &mut self.backing,
            StorageBacking::Owned {
                data: NonNull::new(new_ptr).expect("Allocation should not return null"),
                layout,
                capacity: n,
            },
        );

        // Deallocate old backing
        match old_backing {
            StorageBacking::Owned {
                data,
                layout: old_layout,
                ..
            } => unsafe { dealloc(data.as_ptr(), old_layout) },
            #[cfg(feature = "mmap")]
            StorageBacking::Mmap(_) => {} // Mmap dropped automatically
        }

        Ok(())
    }
}
