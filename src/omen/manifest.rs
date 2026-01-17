//! Omen Storage Structures
//!
//! Implements Trailer-based persistence and Manifest-based node mapping.

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::io;

/// Manifest segment header (8 bytes)
///
/// Written immediately before manifest data to provide integrity checking.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ManifestHeader {
    /// Length of manifest data (excluding this header)
    pub length: u32,
    /// CRC32 of manifest data
    pub crc: u32,
}

impl ManifestHeader {
    pub const SIZE: usize = 8;

    /// Create a new header for the given manifest bytes
    #[must_use]
    pub fn new(manifest_bytes: &[u8]) -> Self {
        Self {
            length: manifest_bytes.len() as u32,
            crc: crc32fast::hash(manifest_bytes),
        }
    }

    /// Serialize to 8-byte array
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 8] {
        let mut bytes = [0u8; 8];
        bytes[0..4].copy_from_slice(&self.length.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.crc.to_le_bytes());
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Manifest header too short",
            ));
        }
        Ok(Self {
            length: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            crc: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        })
    }

    /// Verify CRC against manifest bytes
    #[must_use]
    pub fn verify(&self, manifest_bytes: &[u8]) -> bool {
        manifest_bytes.len() == self.length as usize && crc32fast::hash(manifest_bytes) == self.crc
    }
}

/// Omen Footer (64 bytes)
///
/// Written at the absolute end of the file.
/// Points to the active Manifest segment.
#[repr(C)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct OmenFooter {
    /// Magic number (0x0DEDB002)
    pub magic: u32,
    /// Footer version (1)
    pub version: u32,
    /// Offset to the active Manifest segment
    pub manifest_offset: u64,
    /// Total file length at commit time (excluding this footer)
    pub total_len: u64,
    /// Reserved for future use (padding to 60 bytes)
    pub _reserved: [u8; 32],
    /// Reserved 2 (padding to 60 bytes)
    pub _reserved2: u32,
    /// CRC32 checksum of the footer (excluding crc field)
    pub crc: u32,
}

impl OmenFooter {
    pub const SIZE: usize = 64;
    pub const MAGIC: u32 = 0x0DED_B002;
    pub const VERSION: u32 = 1;

    /// Create a new footer
    #[must_use]
    pub fn new(manifest_offset: u64, total_len: u64) -> Self {
        let mut footer = Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            manifest_offset,
            total_len,
            _reserved: [0; 32],
            _reserved2: 0,
            crc: 0,
        };
        footer.crc = footer.compute_crc();
        footer
    }

    /// Compute CRC32 of footer data
    #[must_use]
    pub fn compute_crc(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&self.magic.to_le_bytes());
        hasher.update(&self.version.to_le_bytes());
        hasher.update(&self.manifest_offset.to_le_bytes());
        hasher.update(&self.total_len.to_le_bytes());
        hasher.update(&self._reserved);
        hasher.update(&self._reserved2.to_le_bytes());
        hasher.finalize()
    }

    /// Verify footer magic and checksum
    #[must_use]
    pub fn verify(&self) -> bool {
        self.magic == Self::MAGIC && self.crc == self.compute_crc()
    }

    /// Serialize to 64-byte array
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut bytes = [0u8; 64];
        bytes[0..4].copy_from_slice(&self.magic.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.version.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.manifest_offset.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.total_len.to_le_bytes());
        bytes[24..56].copy_from_slice(&self._reserved);
        bytes[56..60].copy_from_slice(&self._reserved2.to_le_bytes());
        bytes[60..64].copy_from_slice(&self.crc.to_le_bytes());
        bytes
    }

    /// Deserialize from 64-byte array
    #[must_use]
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        Self {
            magic: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            version: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            manifest_offset: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            total_len: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            _reserved: bytes[24..56].try_into().unwrap(),
            _reserved2: u32::from_le_bytes(bytes[56..60].try_into().unwrap()),
            crc: u32::from_le_bytes(bytes[60..64].try_into().unwrap()),
        }
    }
}

/// Type of data stored in a segment
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum SegmentType {
    /// Flat vector data (f32 or u8)
    Vectors = 1,
    /// HNSW neighbor lists
    Neighbors = 2,
    /// Interleaved [Vector | Neighbors]
    InterleavedNode = 3,
    /// Columnar metadata (tantivy-columnar)
    Metadata = 4,
    /// Metadata (HNSW, etc.)
    IndexMetadata,
    /// Manifest segment itself
    Manifest,
}

/// Location of a node's data in the file
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct NodeLocation {
    /// Offset from start of file
    pub offset: u64,
    /// Length in bytes
    pub length: u32,
    /// Segment type
    pub segment_type: SegmentType,
}

/// Omen Manifest
///
/// Maps NodeID to its location. Implicitly indexed by NodeID.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OmenManifest {
    /// Node locations (index = NodeID)
    pub nodes: Vec<NodeLocation>,
    /// Previous manifest offset (for incremental updates)
    pub prev_manifest: Option<u64>,
    /// Highest NodeID in this manifest
    pub max_node_id: u32,
    /// Deleted node indices (persisted as RoaringBitmap)
    #[serde(default)]
    pub deleted: RoaringBitmap,

    pub id_to_index: std::collections::HashMap<String, u32>,
    pub index_to_id: std::collections::HashMap<u32, String>,
    pub metadata: std::collections::HashMap<u32, Vec<u8>>,
    /// Serialized MetadataIndex for fast recovery (optional for backward compat)
    #[serde(default)]
    pub metadata_index: Option<Vec<u8>>,
    /// Global configuration
    pub config: std::collections::HashMap<String, u64>,
}

impl OmenManifest {
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            prev_manifest: None,
            max_node_id: 0,
            deleted: RoaringBitmap::new(),
            id_to_index: std::collections::HashMap::new(),
            index_to_id: std::collections::HashMap::new(),
            metadata: std::collections::HashMap::new(),
            metadata_index: None,
            config: std::collections::HashMap::new(),
        }
    }

    /// Returns the count of live (non-deleted) vectors
    #[must_use]
    pub fn live_count(&self) -> u64 {
        let total = self.id_to_index.len() as u64;
        let deleted = self.deleted.len();
        total.saturating_sub(deleted)
    }
}

impl Default for OmenManifest {
    fn default() -> Self {
        Self::new()
    }
}
