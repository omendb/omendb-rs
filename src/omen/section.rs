//! Section entries for .omen file format

/// Section types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u16)]
pub enum SectionType {
    #[default]
    None = 0,
    Vectors = 1,
    Graph = 2,
    MetadataIndex = 3,
    MetadataRaw = 4,
    TextIndex = 5,
    Wal = 6,
    HnswIndex = 7,
}

impl From<u16> for SectionType {
    fn from(v: u16) -> Self {
        match v {
            1 => Self::Vectors,
            2 => Self::Graph,
            3 => Self::MetadataIndex,
            4 => Self::MetadataRaw,
            5 => Self::TextIndex,
            6 => Self::Wal,
            7 => Self::HnswIndex,
            _ => Self::None,
        }
    }
}

/// Section entry in header (24 bytes)
#[derive(Debug, Clone, Copy, Default)]
pub struct SectionEntry {
    pub section_type: SectionType,
    pub flags: u16,
    pub offset: u64,
    pub length: u64,
}

impl SectionEntry {
    /// Create a new section entry
    #[must_use]
    pub fn new(section_type: SectionType, offset: u64, length: u64) -> Self {
        Self {
            section_type,
            flags: 0,
            offset,
            length,
        }
    }

    /// Serialize to bytes (24 bytes)
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 24] {
        let mut buf = [0u8; 24];
        buf[0..2].copy_from_slice(&(self.section_type as u16).to_le_bytes());
        buf[2..4].copy_from_slice(&self.flags.to_le_bytes());
        // 4 bytes padding for alignment
        buf[8..16].copy_from_slice(&self.offset.to_le_bytes());
        buf[16..24].copy_from_slice(&self.length.to_le_bytes());
        buf
    }

    /// Parse from bytes
    #[must_use]
    pub fn from_bytes(buf: &[u8; 24]) -> Self {
        // Direct array indexing - infallible for fixed-size input buffer
        Self {
            section_type: SectionType::from(u16::from_le_bytes([buf[0], buf[1]])),
            flags: u16::from_le_bytes([buf[2], buf[3]]),
            offset: u64::from_le_bytes([
                buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
            ]),
            length: u64::from_le_bytes([
                buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23],
            ]),
        }
    }

    /// Check if section is valid (has data)
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.section_type != SectionType::None && self.length > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_section_entry_roundtrip() {
        let entry = SectionEntry::new(SectionType::Vectors, 4096, 1024 * 1024);
        let bytes = entry.to_bytes();
        let parsed = SectionEntry::from_bytes(&bytes);

        assert_eq!(parsed.section_type, SectionType::Vectors);
        assert_eq!(parsed.offset, 4096);
        assert_eq!(parsed.length, 1024 * 1024);
    }
}
