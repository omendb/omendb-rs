//! .omen single-file storage format for `OmenDB`
//!
//! Layout:
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ HEADER (4KB, page 0)                                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │ VECTOR SECTION (page-aligned, mmap)                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │ GRAPH SECTION (page-aligned, mmap)                          │
//! ├─────────────────────────────────────────────────────────────┤
//! │ METADATA SECTION                                            │
//! ├─────────────────────────────────────────────────────────────┤
//! │ WAL SECTION (append-only, at end)                           │
//! └─────────────────────────────────────────────────────────────┘
//! ```

mod file;
mod graph;
mod header;
mod manifest;
mod metadata;
mod vectors;
mod wal;

pub use file::{
    parse_wal_delete, parse_wal_insert, OmenFile, OmenSnapshot, WalDeleteData, WalInsertData,
};
pub use graph::GraphSection;
pub use header::{Metric, OmenHeader, HEADER_SIZE, MAGIC, VERSION_MAJOR, VERSION_MINOR};
pub use manifest::{ManifestHeader, NodeLocation, OmenFooter, OmenManifest, SegmentType};
pub use metadata::{FieldIndex, Filter, FilterValue, MetadataIndex};
pub use vectors::VectorSection;
pub use wal::{Wal, WalEntry, WalEntryType};

/// Page size for alignment (8KB optimal for `NVMe`)
pub const PAGE_SIZE: usize = 8192;

/// Align a value to page boundary
#[inline]
#[must_use]
pub const fn align_to_page(value: usize) -> usize {
    (value + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_to_page() {
        assert_eq!(align_to_page(0), 0);
        assert_eq!(align_to_page(1), PAGE_SIZE);
        assert_eq!(align_to_page(PAGE_SIZE), PAGE_SIZE);
        assert_eq!(align_to_page(PAGE_SIZE + 1), PAGE_SIZE * 2);
    }
}
