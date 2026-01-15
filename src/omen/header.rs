//! .omen file header (4KB)

use std::io::{self, Read};

/// Magic bytes: "OMEN"
pub const MAGIC: [u8; 4] = *b"OMEN";

/// Current format version
pub const VERSION_MAJOR: u16 = 1;
pub const VERSION_MINOR: u16 = 0;

/// Header size (4KB, one page)
pub const HEADER_SIZE: usize = 4096;

/// Quantization code for file format serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QuantizationCode {
    /// No quantization (full f32 precision)
    F32 = 0,
    /// SQ8 scalar quantization (4x compression)
    Sq8 = 1,
    /// RaBitQ 1-bit with FFHT (32x compression)
    RaBitQ = 2,
    /// Extended-RaBitQ 4-bit with FFHT (8x compression)
    RaBitQ4 = 3,
}

impl From<u8> for QuantizationCode {
    fn from(v: u8) -> Self {
        match v {
            1 => Self::Sq8,
            2 => Self::RaBitQ,
            3 => Self::RaBitQ4,
            _ => Self::F32,
        }
    }
}

impl From<&crate::vector::QuantizationMode> for QuantizationCode {
    fn from(mode: &crate::vector::QuantizationMode) -> Self {
        match mode {
            crate::vector::QuantizationMode::SQ8 => Self::Sq8,
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
    /// Returns `None` for `F32` (no quantization) or unsupported legacy codes (RaBitQ).
    #[must_use]
    pub fn to_runtime(self) -> Option<crate::vector::QuantizationMode> {
        match self {
            Self::Sq8 => Some(crate::vector::QuantizationMode::SQ8),
            // F32 = no quantization, RaBitQ codes = legacy (no longer supported at runtime)
            Self::F32 | Self::RaBitQ | Self::RaBitQ4 => None,
        }
    }
}

/// Distance metric for similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Metric {
    L2 = 0,
    Cosine = 1,
    InnerProduct = 2,
}

impl Metric {
    /// Parse from string
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "l2" | "euclidean" => Ok(Self::L2),
            "cosine" | "cos" => Ok(Self::Cosine),
            "ip" | "inner_product" | "dot" => Ok(Self::InnerProduct),
            _ => Err(format!("Unknown metric: {s}")),
        }
    }
}

impl From<u8> for Metric {
    fn from(v: u8) -> Self {
        match v {
            1 => Self::Cosine,
            2 => Self::InnerProduct,
            _ => Self::L2,
        }
    }
}

impl From<crate::vector::hnsw::DistanceFunction> for Metric {
    fn from(df: crate::vector::hnsw::DistanceFunction) -> Self {
        match df {
            crate::vector::hnsw::DistanceFunction::L2 => Self::L2,
            crate::vector::hnsw::DistanceFunction::Cosine => Self::Cosine,
            crate::vector::hnsw::DistanceFunction::NegativeDotProduct => Self::InnerProduct,
        }
    }
}

impl From<Metric> for crate::vector::hnsw::DistanceFunction {
    fn from(m: Metric) -> Self {
        match m {
            Metric::L2 => Self::L2,
            Metric::Cosine => Self::Cosine,
            Metric::InnerProduct => Self::NegativeDotProduct,
        }
    }
}

/// .omen file header (4KB)
#[derive(Debug, Clone)]
pub struct OmenHeader {
    pub version_major: u16,
    pub version_minor: u16,
    pub dimensions: u32,
    pub count: u64,
    pub entry_point: u64,
    pub quantization: QuantizationCode,
    pub metric: Metric,
    pub hnsw_m: u32,
    pub hnsw_ef_construction: u32,
    pub hnsw_ef_search: u32,
}

impl Default for OmenHeader {
    fn default() -> Self {
        Self {
            version_major: VERSION_MAJOR,
            version_minor: VERSION_MINOR,
            dimensions: 0,
            count: 0,
            entry_point: 0,
            quantization: QuantizationCode::F32,
            metric: Metric::L2,
            hnsw_m: 16,
            hnsw_ef_construction: 100,
            hnsw_ef_search: 100,
        }
    }
}

impl OmenHeader {
    /// Create new header with dimensions
    #[must_use]
    pub fn new(dimensions: u32) -> Self {
        Self {
            dimensions,
            ..Default::default()
        }
    }

    /// Set quantization mode
    #[must_use]
    pub fn with_quantization(mut self, q: QuantizationCode) -> Self {
        self.quantization = q;
        self
    }

    /// Set distance metric
    #[must_use]
    pub fn with_metric(mut self, m: Metric) -> Self {
        self.metric = m;
        self
    }

    /// Set HNSW parameters
    #[must_use]
    pub fn with_hnsw(mut self, m: u32, ef_construction: u32, ef_search: u32) -> Self {
        self.hnsw_m = m;
        self.hnsw_ef_construction = ef_construction;
        self.hnsw_ef_search = ef_search;
        self
    }

    /// Serialize header to bytes
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = vec![0u8; HEADER_SIZE];

        // Magic
        buf[0..4].copy_from_slice(&MAGIC);

        // Version
        buf[4..6].copy_from_slice(&self.version_major.to_le_bytes());
        buf[6..8].copy_from_slice(&self.version_minor.to_le_bytes());

        // Dimensions and count
        buf[8..12].copy_from_slice(&self.dimensions.to_le_bytes());
        buf[12..20].copy_from_slice(&self.count.to_le_bytes());

        // Quantization and metric
        buf[20] = self.quantization as u8;
        buf[21] = self.metric as u8;

        // HNSW params
        buf[24..28].copy_from_slice(&self.hnsw_m.to_le_bytes());
        buf[28..32].copy_from_slice(&self.hnsw_ef_construction.to_le_bytes());
        buf[32..36].copy_from_slice(&self.hnsw_ef_search.to_le_bytes());

        // Section count and entry_point
        buf[36..40].copy_from_slice(&0u32.to_le_bytes()); // Section count always 0 in V2
        buf[40..48].copy_from_slice(&self.entry_point.to_le_bytes());

        buf
    }

    /// Parse header from bytes
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        if data.len() < HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Header too small",
            ));
        }

        // Check magic
        if data[0..4] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid magic bytes",
            ));
        }

        let version_major = u16::from_le_bytes([data[4], data[5]]);
        let version_minor = u16::from_le_bytes([data[6], data[7]]);
        let dimensions = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let count = u64::from_le_bytes([
            data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19],
        ]);
        let quantization = QuantizationCode::from(data[20]);
        let metric = Metric::from(data[21]);
        let hnsw_m = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
        let hnsw_ef_construction = u32::from_le_bytes([data[28], data[29], data[30], data[31]]);
        let hnsw_ef_search = u32::from_le_bytes([data[32], data[33], data[34], data[35]]);
        let entry_point = u64::from_le_bytes([
            data[40], data[41], data[42], data[43], data[44], data[45], data[46], data[47],
        ]);

        Ok(Self {
            version_major,
            version_minor,
            dimensions,
            count,
            entry_point,
            quantization,
            metric,
            hnsw_m,
            hnsw_ef_construction,
            hnsw_ef_search,
        })
    }

    /// Read header from a reader
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut buf = vec![0u8; HEADER_SIZE];
        reader.read_exact(&mut buf)?;
        Self::from_bytes(&buf)
    }
}
