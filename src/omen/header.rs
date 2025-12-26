//! .omen file header (4KB)

use crate::omen::section::{SectionEntry, SectionType};
use std::io::{self, Read};

/// Magic bytes: "OMEN"
pub const MAGIC: [u8; 4] = *b"OMEN";

/// Current format version
pub const VERSION_MAJOR: u16 = 1;
pub const VERSION_MINOR: u16 = 0;

/// Header size (4KB, one page)
pub const HEADER_SIZE: usize = 4096;

/// Maximum number of sections
pub const MAX_SECTIONS: usize = 8;

/// Quantization code for file format serialization.
///
/// This is a compact `repr(u8)` representation for storing in the .omen header.
/// For runtime API, use `crate::vector::QuantizationMode` instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QuantizationCode {
    F32 = 0,
    Sq8 = 1,
    RabitQ4 = 2,
    RabitQ2 = 3,
    RabitQ8 = 4,
    Binary = 5,
}

impl From<u8> for QuantizationCode {
    fn from(v: u8) -> Self {
        match v {
            1 => Self::Sq8,
            2 => Self::RabitQ4,
            3 => Self::RabitQ2,
            4 => Self::RabitQ8,
            5 => Self::Binary,
            _ => Self::F32,
        }
    }
}

impl From<&crate::vector::QuantizationMode> for QuantizationCode {
    fn from(mode: &crate::vector::QuantizationMode) -> Self {
        use crate::compression::QuantizationBits;
        match mode {
            crate::vector::QuantizationMode::Binary => Self::Binary,
            crate::vector::QuantizationMode::SQ8 => Self::Sq8,
            crate::vector::QuantizationMode::RaBitQ(params) => match params.bits_per_dim {
                QuantizationBits::Bits1 => Self::Binary,
                QuantizationBits::Bits2 => Self::RabitQ2,
                QuantizationBits::Bits3 | QuantizationBits::Bits4 => Self::RabitQ4,
                QuantizationBits::Bits5 | QuantizationBits::Bits7 | QuantizationBits::Bits8 => {
                    Self::RabitQ8
                }
            },
        }
    }
}

impl From<crate::vector::QuantizationMode> for QuantizationCode {
    fn from(mode: crate::vector::QuantizationMode) -> Self {
        Self::from(&mode)
    }
}

impl QuantizationCode {
    /// Convert to runtime `QuantizationMode`.
    ///
    /// Returns `None` for `F32` (no quantization).
    #[must_use]
    pub fn to_runtime(self) -> Option<crate::vector::QuantizationMode> {
        use crate::compression::RaBitQParams;
        match self {
            Self::F32 => None,
            Self::Sq8 => Some(crate::vector::QuantizationMode::SQ8),
            Self::Binary => Some(crate::vector::QuantizationMode::Binary),
            Self::RabitQ2 => Some(crate::vector::QuantizationMode::RaBitQ(
                RaBitQParams::bits2(),
            )),
            Self::RabitQ4 => Some(crate::vector::QuantizationMode::RaBitQ(
                RaBitQParams::bits4(),
            )),
            Self::RabitQ8 => Some(crate::vector::QuantizationMode::RaBitQ(
                RaBitQParams::bits8(),
            )),
        }
    }
}

/// Distance metric for similarity search (user-facing API type).
///
/// This is the serialization/API type stored in .omen file headers.
/// For runtime distance computation, see `crate::vector::hnsw::DistanceFunction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Metric {
    /// L2 / Euclidean distance
    L2 = 0,
    /// Cosine distance (1 - cosine similarity)
    Cosine = 1,
    /// Dot product / inner product (for MIPS)
    Dot = 2,
}

impl From<u8> for Metric {
    fn from(v: u8) -> Self {
        match v {
            1 => Self::Cosine,
            2 => Self::Dot,
            _ => Self::L2,
        }
    }
}

impl From<Metric> for crate::vector::hnsw::DistanceFunction {
    fn from(m: Metric) -> Self {
        match m {
            Metric::L2 => Self::L2,
            Metric::Cosine => Self::Cosine,
            Metric::Dot => Self::NegativeDotProduct,
        }
    }
}

impl From<crate::vector::hnsw::DistanceFunction> for Metric {
    fn from(d: crate::vector::hnsw::DistanceFunction) -> Self {
        match d {
            crate::vector::hnsw::DistanceFunction::L2 => Self::L2,
            crate::vector::hnsw::DistanceFunction::Cosine => Self::Cosine,
            crate::vector::hnsw::DistanceFunction::NegativeDotProduct => Self::Dot,
        }
    }
}

impl Metric {
    /// Parse from string (case-insensitive, with aliases).
    ///
    /// # Supported values
    /// - `"l2"` or `"euclidean"`: Euclidean distance (default)
    /// - `"cosine"`: Cosine distance (1 - cosine similarity)
    /// - `"dot"` or `"ip"`: Inner product (for MIPS)
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "l2" | "euclidean" => Ok(Self::L2),
            "cosine" => Ok(Self::Cosine),
            "dot" | "ip" => Ok(Self::Dot),
            _ => Err(format!(
                "Unknown metric: '{s}'. Valid: l2, euclidean, cosine, dot, ip"
            )),
        }
    }

    /// Get the string representation.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::Cosine => "cosine",
            Self::Dot => "dot",
        }
    }
}

/// .omen file header
#[derive(Debug, Clone)]
pub struct OmenHeader {
    // Magic and version (16 bytes)
    pub version_major: u16,
    pub version_minor: u16,
    pub flags: u64,

    // Database info (32 bytes)
    pub dimensions: u32,
    pub count: u64,
    pub quantization: QuantizationCode,
    pub distance_fn: Metric,

    // HNSW params (16 bytes)
    pub m: u16,
    pub ef_construction: u16,
    pub ef_search: u16,
    pub max_level: u8,
    pub entry_point: u32,

    // Section directory
    pub sections: [SectionEntry; MAX_SECTIONS],

    // Checksums
    pub header_checksum: u32,
    pub data_checksum: u32,
}

impl Default for OmenHeader {
    fn default() -> Self {
        Self {
            version_major: VERSION_MAJOR,
            version_minor: VERSION_MINOR,
            flags: 0,
            dimensions: 0,
            count: 0,
            quantization: QuantizationCode::F32,
            distance_fn: Metric::L2,
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            max_level: 0,
            entry_point: 0,
            sections: [SectionEntry::default(); MAX_SECTIONS],
            header_checksum: 0,
            data_checksum: 0,
        }
    }
}

impl OmenHeader {
    /// Create a new header with the given dimensions
    #[must_use]
    pub fn new(dimensions: u32) -> Self {
        Self {
            dimensions,
            ..Default::default()
        }
    }

    /// Serialize header to bytes (4KB)
    #[must_use]
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        let mut offset = 0;

        // Magic (4 bytes)
        buf[offset..offset + 4].copy_from_slice(&MAGIC);
        offset += 4;

        // Version (4 bytes)
        buf[offset..offset + 2].copy_from_slice(&self.version_major.to_le_bytes());
        offset += 2;
        buf[offset..offset + 2].copy_from_slice(&self.version_minor.to_le_bytes());
        offset += 2;

        // Flags (8 bytes)
        buf[offset..offset + 8].copy_from_slice(&self.flags.to_le_bytes());
        offset += 8;

        // Database info (32 bytes)
        buf[offset..offset + 4].copy_from_slice(&self.dimensions.to_le_bytes());
        offset += 4;
        buf[offset..offset + 8].copy_from_slice(&self.count.to_le_bytes());
        offset += 8;
        buf[offset] = self.quantization as u8;
        offset += 1;
        buf[offset] = self.distance_fn as u8;
        offset += 1;
        // 14 bytes reserved (already zeroed)
        offset += 14;

        // HNSW params (16 bytes)
        buf[offset..offset + 2].copy_from_slice(&self.m.to_le_bytes());
        offset += 2;
        buf[offset..offset + 2].copy_from_slice(&self.ef_construction.to_le_bytes());
        offset += 2;
        buf[offset..offset + 2].copy_from_slice(&self.ef_search.to_le_bytes());
        offset += 2;
        buf[offset] = self.max_level;
        offset += 1;
        buf[offset..offset + 4].copy_from_slice(&self.entry_point.to_le_bytes());
        offset += 4;
        // 3 bytes reserved (already zeroed)
        offset += 3;

        // Sections (8 * 24 bytes = 192 bytes)
        for section in &self.sections {
            buf[offset..offset + 24].copy_from_slice(&section.to_bytes());
            offset += 24;
        }

        // Checksums (8 bytes)
        buf[offset..offset + 4].copy_from_slice(&self.header_checksum.to_le_bytes());
        offset += 4;
        buf[offset..offset + 4].copy_from_slice(&self.data_checksum.to_le_bytes());

        // Calculate and write header checksum
        let checksum = crc32fast::hash(&buf[..HEADER_SIZE - 8]);
        buf[HEADER_SIZE - 8..HEADER_SIZE - 4].copy_from_slice(&checksum.to_le_bytes());

        buf
    }

    /// Parse header from bytes
    pub fn from_bytes(buf: &[u8; HEADER_SIZE]) -> io::Result<Self> {
        // Verify magic
        if buf[0..4] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid magic bytes",
            ));
        }

        // Verify checksum - direct array indexing for fixed-size buffer
        let stored_checksum = u32::from_le_bytes([
            buf[HEADER_SIZE - 8],
            buf[HEADER_SIZE - 7],
            buf[HEADER_SIZE - 6],
            buf[HEADER_SIZE - 5],
        ]);
        let computed_checksum = crc32fast::hash(&buf[..HEADER_SIZE - 8]);
        if stored_checksum != computed_checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Header checksum mismatch",
            ));
        }

        let mut cursor = io::Cursor::new(&buf[4..]); // Skip magic

        let mut u16_buf = [0u8; 2];
        let mut u32_buf = [0u8; 4];
        let mut u64_buf = [0u8; 8];
        let mut u8_buf = [0u8; 1];

        // Version
        cursor.read_exact(&mut u16_buf)?;
        let version_major = u16::from_le_bytes(u16_buf);
        cursor.read_exact(&mut u16_buf)?;
        let version_minor = u16::from_le_bytes(u16_buf);

        // Check version compatibility
        if version_major > VERSION_MAJOR {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported version: {version_major}.{version_minor}"),
            ));
        }

        // Flags
        cursor.read_exact(&mut u64_buf)?;
        let flags = u64::from_le_bytes(u64_buf);

        // Database info
        cursor.read_exact(&mut u32_buf)?;
        let dimensions = u32::from_le_bytes(u32_buf);
        cursor.read_exact(&mut u64_buf)?;
        let count = u64::from_le_bytes(u64_buf);
        cursor.read_exact(&mut u8_buf)?;
        let quantization = QuantizationCode::from(u8_buf[0]);
        cursor.read_exact(&mut u8_buf)?;
        let distance_fn = Metric::from(u8_buf[0]);

        // Skip reserved
        let mut reserved = [0u8; 14];
        cursor.read_exact(&mut reserved)?;

        // HNSW params
        cursor.read_exact(&mut u16_buf)?;
        let m = u16::from_le_bytes(u16_buf);
        cursor.read_exact(&mut u16_buf)?;
        let ef_construction = u16::from_le_bytes(u16_buf);
        cursor.read_exact(&mut u16_buf)?;
        let ef_search = u16::from_le_bytes(u16_buf);
        cursor.read_exact(&mut u8_buf)?;
        let max_level = u8_buf[0];
        cursor.read_exact(&mut u32_buf)?;
        let entry_point = u32::from_le_bytes(u32_buf);

        // Skip reserved
        let mut reserved2 = [0u8; 3];
        cursor.read_exact(&mut reserved2)?;

        // Sections
        let mut sections = [SectionEntry::default(); MAX_SECTIONS];
        for section in &mut sections {
            let mut section_buf = [0u8; 24];
            cursor.read_exact(&mut section_buf)?;
            *section = SectionEntry::from_bytes(&section_buf);
        }

        // Checksums
        cursor.read_exact(&mut u32_buf)?;
        let header_checksum = u32::from_le_bytes(u32_buf);
        cursor.read_exact(&mut u32_buf)?;
        let data_checksum = u32::from_le_bytes(u32_buf);

        Ok(Self {
            version_major,
            version_minor,
            flags,
            dimensions,
            count,
            quantization,
            distance_fn,
            m,
            ef_construction,
            ef_search,
            max_level,
            entry_point,
            sections,
            header_checksum,
            data_checksum,
        })
    }

    /// Get section by type
    #[must_use]
    pub fn get_section(&self, section_type: SectionType) -> Option<&SectionEntry> {
        self.sections
            .iter()
            .find(|s| s.section_type == section_type && s.length > 0)
    }

    /// Set section entry
    pub fn set_section(&mut self, entry: SectionEntry) {
        for section in &mut self.sections {
            if section.section_type == entry.section_type || section.length == 0 {
                *section = entry;
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let mut header = OmenHeader::new(768);
        header.count = 1000;
        header.m = 32;
        header.ef_construction = 200;
        header.entry_point = 42;

        let bytes = header.to_bytes();
        let parsed = OmenHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.dimensions, 768);
        assert_eq!(parsed.count, 1000);
        assert_eq!(parsed.m, 32);
        assert_eq!(parsed.ef_construction, 200);
        assert_eq!(parsed.entry_point, 42);
    }

    #[test]
    fn test_invalid_magic() {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(b"NOPE");

        let result = OmenHeader::from_bytes(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_corrupted_header_detected() {
        let header = OmenHeader::new(768);
        let mut bytes = header.to_bytes();

        // Corrupt a byte in the middle of the header (dimensions field)
        bytes[20] ^= 0xFF;

        let result = OmenHeader::from_bytes(&bytes);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("checksum mismatch"));
    }

    #[test]
    fn test_checksum_calculated_correctly() {
        let mut header = OmenHeader::new(768);
        header.count = 12345;
        header.m = 32;
        header.ef_construction = 200;

        let bytes = header.to_bytes();

        // Extract the stored checksum
        let stored_checksum =
            u32::from_le_bytes(bytes[HEADER_SIZE - 8..HEADER_SIZE - 4].try_into().unwrap());

        // Verify it's not zero (would indicate checksum wasn't calculated)
        assert_ne!(stored_checksum, 0);

        // Verify we can read it back (proves checksum is correct)
        let parsed = OmenHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.dimensions, 768);
        assert_eq!(parsed.count, 12345);
    }
}
