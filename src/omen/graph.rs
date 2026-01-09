//! Graph section - HNSW neighbor lists
#![allow(clippy::cast_ptr_alignment)] // mmap data is aligned by our serialization

use memmap2::MmapMut;
use std::io::{self, Write};

/// Graph section - stores HNSW neighbor lists
///
/// Layout:
/// - Node levels: [u8; count] - max level for each node
/// - Level 0 offsets: [u32; count] - offset into `level0_neighbors`
/// - Level 0 neighbors: [u32; ...] - neighbor IDs (varint would be better but simpler for now)
/// - Upper offsets: [u32; ...] - for nodes with level > 0
/// - Upper neighbors: [u32; ...] - upper level neighbor IDs
pub struct GraphSection {
    count: u64,
    m: u16, // Max neighbors per layer
    /// Memory-mapped data
    data: *const u8,
    data_len: usize,
    /// Parsed offsets for fast access
    level0_neighbors_start: usize,
    upper_section_start: usize,
}

// Safety: GraphSection is read-only after creation
unsafe impl Send for GraphSection {}
unsafe impl Sync for GraphSection {}

impl GraphSection {
    /// Create from mmap region
    ///
    /// # Safety
    /// The mmap must remain valid for the lifetime of this section.
    pub unsafe fn from_mmap(
        mmap: &MmapMut,
        offset: usize,
        length: usize,
        count: u64,
        m: u16,
    ) -> io::Result<Self> {
        if length == 0 || count == 0 {
            return Ok(Self {
                count: 0,
                m,
                data: std::ptr::null(),
                data_len: 0,
                level0_neighbors_start: 0,
                upper_section_start: 0,
            });
        }

        let ptr = mmap.as_ptr().add(offset);

        // Calculate section boundaries
        let levels_size = count as usize;
        let offsets_size = count as usize * 4; // u32 offsets
        let level0_neighbors_start = levels_size + offsets_size;

        Ok(Self {
            count,
            m,
            data: ptr,
            data_len: length,
            level0_neighbors_start,
            upper_section_start: length, // Will be set properly on load
        })
    }

    /// Create empty section for building
    #[must_use]
    pub fn new(m: u16) -> Self {
        Self {
            count: 0,
            m,
            data: std::ptr::null(),
            data_len: 0,
            level0_neighbors_start: 0,
            upper_section_start: 0,
        }
    }

    /// Get node level
    #[inline]
    #[must_use]
    pub fn get_level(&self, node_id: u32) -> Option<u8> {
        if node_id as u64 >= self.count || self.data.is_null() {
            return None;
        }
        // Safety: bounds checked
        unsafe { Some(*self.data.add(node_id as usize)) }
    }

    /// Get level 0 neighbors for a node
    #[inline]
    #[must_use]
    pub fn get_neighbors_level0(&self, node_id: u32) -> Option<&[u32]> {
        if node_id as u64 >= self.count || self.data.is_null() {
            return None;
        }

        // Read offset from offsets table (use checked arithmetic to prevent overflow)
        let offset_pos = (self.count as usize).checked_add((node_id as usize).checked_mul(4)?)?;
        if offset_pos.checked_add(4)? > self.data_len {
            return None;
        }

        let offset = unsafe {
            let ptr = self.data.add(offset_pos).cast::<u32>();
            ptr.read_unaligned().to_le() as usize
        };

        // Read neighbor count (first u32 at offset)
        let neighbor_start = self.level0_neighbors_start.checked_add(offset)?;
        if neighbor_start.checked_add(4)? > self.data_len {
            return None;
        }

        let neighbor_count = unsafe {
            let ptr = self.data.add(neighbor_start).cast::<u32>();
            ptr.read_unaligned().to_le() as usize
        };

        if neighbor_count == 0 {
            return Some(&[]);
        }

        // Safety: bounds should be valid if file is not corrupted
        let neighbors_ptr = neighbor_start.checked_add(4)?;
        let neighbors_end = neighbors_ptr.checked_add(neighbor_count.checked_mul(4)?)?;
        if neighbors_end > self.upper_section_start {
            return None;
        }

        unsafe {
            let ptr = self.data.add(neighbors_ptr).cast::<u32>();
            Some(std::slice::from_raw_parts(ptr, neighbor_count))
        }
    }

    /// Get max neighbors per layer
    #[must_use]
    pub fn m(&self) -> u16 {
        self.m
    }

    /// Get node count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Serialize graph to bytes
    ///
    /// Format:
    /// - levels: [u8; count]
    /// - offsets: [u32; count]
    /// - neighbors: for each node: [count: u32, `neighbor_ids`: [u32; count]]
    pub fn write_graph<W: Write>(
        writer: &mut W,
        levels: &[u8],
        neighbors: &[Vec<u32>],
    ) -> io::Result<()> {
        let count = levels.len();
        assert_eq!(count, neighbors.len());

        // Write levels
        writer.write_all(levels)?;

        // Calculate offsets and write
        let mut current_offset: u32 = 0;
        for node_neighbors in neighbors {
            writer.write_all(&current_offset.to_le_bytes())?;
            // Each neighbor list: count (u32) + neighbors (u32 * count)
            current_offset += 4 + (node_neighbors.len() as u32 * 4);
        }

        // Write neighbor lists
        for node_neighbors in neighbors {
            writer.write_all(&(node_neighbors.len() as u32).to_le_bytes())?;
            for &neighbor in node_neighbors {
                writer.write_all(&neighbor.to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Calculate size in bytes for graph
    #[must_use]
    pub fn size_for_graph(levels: &[u8], neighbors: &[Vec<u32>]) -> usize {
        let count = levels.len();
        let levels_size = count;
        let offsets_size = count * 4;
        let neighbors_size: usize = neighbors
            .iter()
            .map(|n| 4 + n.len() * 4) // count + neighbors
            .sum();
        levels_size + offsets_size + neighbors_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_size_calculation() {
        let levels = vec![0u8; 100];
        let neighbors: Vec<Vec<u32>> = (0..100).map(|_| vec![1, 2, 3, 4]).collect();

        let size = GraphSection::size_for_graph(&levels, &neighbors);
        // 100 levels + 100*4 offsets + 100*(4 + 4*4) neighbor lists
        assert_eq!(size, 100 + 400 + 100 * 20);
    }
}
