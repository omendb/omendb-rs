//! Vector section - contiguous f32 array with O(1) access
#![allow(clippy::cast_ptr_alignment)] // alignment verified before cast

use memmap2::MmapMut;
use std::io;

/// Vector section - contiguous array of vectors
pub struct VectorSection {
    dimensions: u32,
    count: u64,
    /// Memory-mapped vector data (read-only view)
    data: *const f32,
    data_len: usize,
}

// Safety: VectorSection is read-only after creation
unsafe impl Send for VectorSection {}
unsafe impl Sync for VectorSection {}

impl VectorSection {
    /// Create from mmap region
    ///
    /// # Safety
    /// The mmap must remain valid for the lifetime of this section.
    /// The data at offset must be properly aligned f32 values.
    pub unsafe fn from_mmap(
        mmap: &MmapMut,
        offset: usize,
        length: usize,
        dimensions: u32,
    ) -> io::Result<Self> {
        if length == 0 {
            return Ok(Self {
                dimensions,
                count: 0,
                data: std::ptr::null(),
                data_len: 0,
            });
        }

        let bytes_per_vector = dimensions as usize * std::mem::size_of::<f32>();
        if bytes_per_vector == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "dimensions must be > 0",
            ));
        }

        let count = length / bytes_per_vector;

        // Check alignment
        let ptr = mmap.as_ptr().add(offset);
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<f32>()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Vector data not properly aligned",
            ));
        }

        Ok(Self {
            dimensions,
            count: count as u64,
            data: ptr.cast::<f32>(),
            data_len: length / std::mem::size_of::<f32>(),
        })
    }

    /// Create empty section for building
    #[must_use]
    pub fn new(dimensions: u32) -> Self {
        Self {
            dimensions,
            count: 0,
            data: std::ptr::null(),
            data_len: 0,
        }
    }

    /// Get vector by index - O(1)
    #[inline]
    #[must_use]
    pub fn get(&self, index: u32) -> Option<&[f32]> {
        if index as u64 >= self.count || self.data.is_null() {
            return None;
        }

        let offset = index as usize * self.dimensions as usize;
        if offset + self.dimensions as usize > self.data_len {
            return None;
        }

        // Safety: We checked bounds above
        unsafe {
            let ptr = self.data.add(offset);
            Some(std::slice::from_raw_parts(ptr, self.dimensions as usize))
        }
    }

    /// Prefetch vector for cache warming (VSAG-style)
    #[inline]
    pub fn prefetch(&self, index: u32) {
        if index as u64 >= self.count || self.data.is_null() {
            return;
        }

        let offset = index as usize * self.dimensions as usize;

        // Safety: We checked bounds
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let ptr = self.data.add(offset).cast::<u8>();
            // Prefetch multiple cache lines (64 bytes each)
            let bytes_to_prefetch = self.dimensions as usize * std::mem::size_of::<f32>();
            let cache_lines = bytes_to_prefetch.div_ceil(64);

            for i in 0..cache_lines.min(8) {
                use std::arch::x86_64::_mm_prefetch;
                _mm_prefetch(ptr.add(i * 64).cast::<i8>(), 3); // _MM_HINT_T0
            }
        }

        // aarch64 prefetch requires nightly feature, skip for now
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = offset; // Suppress unused warning
        }
    }

    /// Get dimensions
    #[must_use]
    pub fn dimensions(&self) -> u32 {
        self.dimensions
    }

    /// Get vector count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Calculate size in bytes for given count
    #[must_use]
    pub fn size_for_count(dimensions: u32, count: u64) -> u64 {
        count * dimensions as u64 * std::mem::size_of::<f32>() as u64
    }

    /// Serialize vectors to bytes (for writing)
    pub fn write_vectors<W: io::Write>(writer: &mut W, vectors: &[&[f32]]) -> io::Result<()> {
        for vector in vectors {
            for &val in *vector {
                writer.write_all(&val.to_le_bytes())?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_calculation() {
        // 768 dimensions, 1000 vectors, f32
        let size = VectorSection::size_for_count(768, 1000);
        assert_eq!(size, 768 * 1000 * 4); // 3,072,000 bytes
    }
}
