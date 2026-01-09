//! Metadata indexing with Roaring bitmaps for O(1) filter evaluation
//!
//! Qdrant-style payload indexes: each indexed field has an inverted index
//! mapping values to document IDs via Roaring bitmaps.

use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::io::{self, Read, Write};

/// Field types for metadata indexing
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FieldType {
    /// String equality (inverted index)
    Keyword = 1,
    /// Integer equality/range
    Integer = 2,
    /// Float range
    Float = 3,
    /// Boolean (two bitmaps: true/false)
    Boolean = 4,
}

impl From<u8> for FieldType {
    fn from(v: u8) -> Self {
        match v {
            2 => Self::Integer,
            3 => Self::Float,
            4 => Self::Boolean,
            _ => Self::Keyword, // 1 or unknown -> Keyword
        }
    }
}

/// Keyword index - inverted index for string equality
#[derive(Debug, Clone, Default)]
pub struct KeywordIndex {
    /// term -> bitmap of doc IDs
    terms: HashMap<String, RoaringBitmap>,
}

impl KeywordIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a document with the given term
    pub fn insert(&mut self, doc_id: u32, term: &str) {
        self.terms
            .entry(term.to_string())
            .or_default()
            .insert(doc_id);
    }

    /// Remove a document from all terms
    pub fn remove(&mut self, doc_id: u32) {
        for bitmap in self.terms.values_mut() {
            bitmap.remove(doc_id);
        }
        // Clean up empty bitmaps to prevent memory leak
        self.terms.retain(|_, bitmap| !bitmap.is_empty());
    }

    /// Get documents matching a term
    pub fn get(&self, term: &str) -> Option<&RoaringBitmap> {
        self.terms.get(term)
    }

    /// Check if a document has a specific term
    #[inline]
    pub fn contains(&self, doc_id: u32, term: &str) -> bool {
        self.terms
            .get(term)
            .is_some_and(|bitmap| bitmap.contains(doc_id))
    }

    /// Get all terms
    pub fn terms(&self) -> impl Iterator<Item = &str> {
        self.terms.keys().map(std::string::String::as_str)
    }

    /// Serialize to bytes
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Write term count
        writer.write_all(&(self.terms.len() as u32).to_le_bytes())?;

        for (term, bitmap) in &self.terms {
            // Write term (length-prefixed)
            writer.write_all(&(term.len() as u32).to_le_bytes())?;
            writer.write_all(term.as_bytes())?;

            // Write bitmap (use native serialization)
            let mut bitmap_bytes = Vec::new();
            bitmap.serialize_into(&mut bitmap_bytes)?;
            writer.write_all(&(bitmap_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(&bitmap_bytes)?;
        }

        Ok(())
    }

    /// Deserialize from bytes
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf)?;
        let term_count = u32::from_le_bytes(len_buf) as usize;

        let mut terms = HashMap::with_capacity(term_count);

        for _ in 0..term_count {
            // Read term
            reader.read_exact(&mut len_buf)?;
            let term_len = u32::from_le_bytes(len_buf) as usize;
            let mut term_buf = vec![0u8; term_len];
            reader.read_exact(&mut term_buf)?;
            let term = String::from_utf8(term_buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            // Read bitmap
            reader.read_exact(&mut len_buf)?;
            let bitmap_len = u32::from_le_bytes(len_buf) as usize;
            let mut bitmap_buf = vec![0u8; bitmap_len];
            reader.read_exact(&mut bitmap_buf)?;
            let bitmap = RoaringBitmap::deserialize_from(&bitmap_buf[..])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            terms.insert(term, bitmap);
        }

        Ok(Self { terms })
    }
}

/// Boolean index - two bitmaps for true/false
#[derive(Debug, Clone, Default)]
pub struct BooleanIndex {
    true_docs: RoaringBitmap,
    false_docs: RoaringBitmap,
}

impl BooleanIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, doc_id: u32, value: bool) {
        if value {
            self.true_docs.insert(doc_id);
            self.false_docs.remove(doc_id);
        } else {
            self.false_docs.insert(doc_id);
            self.true_docs.remove(doc_id);
        }
    }

    pub fn remove(&mut self, doc_id: u32) {
        self.true_docs.remove(doc_id);
        self.false_docs.remove(doc_id);
    }

    #[inline]
    pub fn matches(&self, doc_id: u32, value: bool) -> bool {
        if value {
            self.true_docs.contains(doc_id)
        } else {
            self.false_docs.contains(doc_id)
        }
    }

    pub fn get_true(&self) -> &RoaringBitmap {
        &self.true_docs
    }

    pub fn get_false(&self) -> &RoaringBitmap {
        &self.false_docs
    }

    /// Serialize to bytes
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Write true_docs bitmap
        let mut true_bytes = Vec::new();
        self.true_docs.serialize_into(&mut true_bytes)?;
        writer.write_all(&(true_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&true_bytes)?;

        // Write false_docs bitmap
        let mut false_bytes = Vec::new();
        self.false_docs.serialize_into(&mut false_bytes)?;
        writer.write_all(&(false_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&false_bytes)?;

        Ok(())
    }

    /// Deserialize from bytes
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut len_buf = [0u8; 4];

        // Read true_docs bitmap
        reader.read_exact(&mut len_buf)?;
        let true_len = u32::from_le_bytes(len_buf) as usize;
        let mut true_buf = vec![0u8; true_len];
        reader.read_exact(&mut true_buf)?;
        let true_docs = RoaringBitmap::deserialize_from(&true_buf[..])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Read false_docs bitmap
        reader.read_exact(&mut len_buf)?;
        let false_len = u32::from_le_bytes(len_buf) as usize;
        let mut false_buf = vec![0u8; false_len];
        reader.read_exact(&mut false_buf)?;
        let false_docs = RoaringBitmap::deserialize_from(&false_buf[..])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        Ok(Self {
            true_docs,
            false_docs,
        })
    }
}

/// Numeric index for integer/float range queries
#[derive(Debug, Clone, Default)]
pub struct NumericIndex {
    /// Sorted (value, `doc_id`) pairs for range queries
    entries: Vec<(f64, u32)>,
    /// Optional: bitmap for common values (equality fast path)
    common_values: HashMap<i64, RoaringBitmap>,
}

impl NumericIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, doc_id: u32, value: f64) {
        // Skip NaN values (can't be compared or indexed meaningfully)
        if value.is_nan() {
            return;
        }

        // Add to sorted entries (total_cmp handles -0.0 vs 0.0, NaN already filtered)
        let pos = self
            .entries
            .binary_search_by(|(v, _)| v.total_cmp(&value))
            .unwrap_or_else(|p| p);
        self.entries.insert(pos, (value, doc_id));

        // Track common integer values for fast equality
        if value.fract() == 0.0 && value >= i64::MIN as f64 && value <= i64::MAX as f64 {
            let int_val = value as i64;
            self.common_values
                .entry(int_val)
                .or_default()
                .insert(doc_id);
        }
    }

    pub fn remove(&mut self, doc_id: u32) {
        self.entries.retain(|(_, id)| *id != doc_id);
        for bitmap in self.common_values.values_mut() {
            bitmap.remove(doc_id);
        }
        // Clean up empty bitmaps to prevent memory leak
        self.common_values.retain(|_, bitmap| !bitmap.is_empty());
    }

    /// Get documents where value == target (fast path for integers)
    pub fn get_eq(&self, value: f64) -> Option<&RoaringBitmap> {
        if value.fract() == 0.0 {
            self.common_values.get(&(value as i64))
        } else {
            None
        }
    }

    /// Get documents where value is in range [min, max]
    pub fn get_range(&self, min: f64, max: f64) -> RoaringBitmap {
        // Use partition_point to find FIRST position where value >= min
        // This handles duplicates correctly (binary_search may find any duplicate)
        let start = self.entries.partition_point(|(v, _)| *v < min);

        let mut result = RoaringBitmap::new();
        for (val, doc_id) in &self.entries[start..] {
            if *val > max {
                break;
            }
            result.insert(*doc_id);
        }
        result
    }

    /// Check if document matches value
    #[inline]
    pub fn matches_eq(&self, doc_id: u32, value: f64) -> bool {
        if let Some(bitmap) = self.get_eq(value) {
            bitmap.contains(doc_id)
        } else {
            // Slow path: linear scan
            self.entries
                .iter()
                .any(|(v, id)| *id == doc_id && (*v - value).abs() < f64::EPSILON)
        }
    }

    /// Check if document is in range [min, max] (inclusive)
    #[inline]
    pub fn matches_range(&self, doc_id: u32, min: f64, max: f64) -> bool {
        self.entries
            .iter()
            .any(|(v, id)| *id == doc_id && *v >= min && *v <= max)
    }

    /// Check if document value > threshold (strict greater than)
    #[inline]
    pub fn matches_gt(&self, doc_id: u32, threshold: f64) -> bool {
        self.entries
            .iter()
            .any(|(v, id)| *id == doc_id && *v > threshold)
    }

    /// Check if document value < threshold (strict less than)
    #[inline]
    pub fn matches_lt(&self, doc_id: u32, threshold: f64) -> bool {
        self.entries
            .iter()
            .any(|(v, id)| *id == doc_id && *v < threshold)
    }

    /// Serialize to bytes
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Write entries count and data
        writer.write_all(&(self.entries.len() as u32).to_le_bytes())?;
        for (value, doc_id) in &self.entries {
            writer.write_all(&value.to_le_bytes())?;
            writer.write_all(&doc_id.to_le_bytes())?;
        }

        // Write common_values count and data
        writer.write_all(&(self.common_values.len() as u32).to_le_bytes())?;
        for (int_val, bitmap) in &self.common_values {
            writer.write_all(&int_val.to_le_bytes())?;
            let mut bitmap_bytes = Vec::new();
            bitmap.serialize_into(&mut bitmap_bytes)?;
            writer.write_all(&(bitmap_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(&bitmap_bytes)?;
        }

        Ok(())
    }

    /// Deserialize from bytes
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        // Read entries
        reader.read_exact(&mut buf4)?;
        let entries_len = u32::from_le_bytes(buf4) as usize;
        let mut entries = Vec::with_capacity(entries_len);
        for _ in 0..entries_len {
            reader.read_exact(&mut buf8)?;
            let value = f64::from_le_bytes(buf8);
            reader.read_exact(&mut buf4)?;
            let doc_id = u32::from_le_bytes(buf4);
            entries.push((value, doc_id));
        }

        // Read common_values
        reader.read_exact(&mut buf4)?;
        let common_len = u32::from_le_bytes(buf4) as usize;
        let mut common_values = HashMap::with_capacity(common_len);
        for _ in 0..common_len {
            reader.read_exact(&mut buf8)?;
            let int_val = i64::from_le_bytes(buf8);
            reader.read_exact(&mut buf4)?;
            let bitmap_len = u32::from_le_bytes(buf4) as usize;
            let mut bitmap_buf = vec![0u8; bitmap_len];
            reader.read_exact(&mut bitmap_buf)?;
            let bitmap = RoaringBitmap::deserialize_from(&bitmap_buf[..])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            common_values.insert(int_val, bitmap);
        }

        Ok(Self {
            entries,
            common_values,
        })
    }
}

/// Field index - wraps different index types
#[derive(Debug, Clone)]
pub enum FieldIndex {
    Keyword(KeywordIndex),
    Boolean(BooleanIndex),
    Numeric(NumericIndex),
}

impl FieldIndex {
    #[must_use]
    pub fn keyword() -> Self {
        Self::Keyword(KeywordIndex::new())
    }

    #[must_use]
    pub fn boolean() -> Self {
        Self::Boolean(BooleanIndex::new())
    }

    #[must_use]
    pub fn numeric() -> Self {
        Self::Numeric(NumericIndex::new())
    }
}

/// Metadata index - collection of field indexes
#[derive(Debug, Clone, Default)]
pub struct MetadataIndex {
    fields: HashMap<String, FieldIndex>,
}

impl MetadataIndex {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create or get a keyword index for a field.
    /// Returns None if the field exists with a different type.
    pub fn keyword_index(&mut self, field: &str) -> Option<&mut KeywordIndex> {
        let index = self
            .fields
            .entry(field.to_string())
            .or_insert_with(FieldIndex::keyword);
        match index {
            FieldIndex::Keyword(idx) => Some(idx),
            _ => None, // Field exists with different type
        }
    }

    /// Create or get a boolean index for a field.
    /// Returns None if the field exists with a different type.
    pub fn boolean_index(&mut self, field: &str) -> Option<&mut BooleanIndex> {
        let index = self
            .fields
            .entry(field.to_string())
            .or_insert_with(FieldIndex::boolean);
        match index {
            FieldIndex::Boolean(idx) => Some(idx),
            _ => None, // Field exists with different type
        }
    }

    /// Create or get a numeric index for a field.
    /// Returns None if the field exists with a different type.
    pub fn numeric_index(&mut self, field: &str) -> Option<&mut NumericIndex> {
        let index = self
            .fields
            .entry(field.to_string())
            .or_insert_with(FieldIndex::numeric);
        match index {
            FieldIndex::Numeric(idx) => Some(idx),
            _ => None, // Field exists with different type
        }
    }

    /// Get a field index
    #[must_use]
    pub fn get(&self, field: &str) -> Option<&FieldIndex> {
        self.fields.get(field)
    }

    /// Index a JSON metadata object.
    /// Silently skips fields with inconsistent types across documents.
    pub fn index_json(&mut self, doc_id: u32, metadata: &serde_json::Value) {
        if let serde_json::Value::Object(map) = metadata {
            for (key, value) in map {
                match value {
                    serde_json::Value::String(s) => {
                        if let Some(idx) = self.keyword_index(key) {
                            idx.insert(doc_id, s);
                        }
                    }
                    serde_json::Value::Bool(b) => {
                        if let Some(idx) = self.boolean_index(key) {
                            idx.insert(doc_id, *b);
                        }
                    }
                    serde_json::Value::Number(n) => {
                        if let Some(f) = n.as_f64() {
                            if let Some(idx) = self.numeric_index(key) {
                                idx.insert(doc_id, f);
                            }
                        }
                    }
                    _ => {} // Skip arrays and nested objects
                }
            }
        }
    }

    /// Remove a document from all indexes
    pub fn remove(&mut self, doc_id: u32) {
        for index in self.fields.values_mut() {
            match index {
                FieldIndex::Keyword(idx) => idx.remove(doc_id),
                FieldIndex::Boolean(idx) => idx.remove(doc_id),
                FieldIndex::Numeric(idx) => idx.remove(doc_id),
            }
        }
    }

    /// Evaluate a filter expression (returns true if matches)
    #[inline]
    #[must_use]
    pub fn matches(&self, doc_id: u32, filter: &Filter) -> bool {
        match filter {
            Filter::Eq(field, value) => self.matches_eq(doc_id, field, value),
            Filter::Ne(field, value) => !self.matches_eq(doc_id, field, value),
            Filter::Gt(field, value) => self.matches_gt(doc_id, field, *value),
            Filter::Gte(field, value) => self.matches_gte(doc_id, field, *value),
            Filter::Lt(field, value) => self.matches_lt(doc_id, field, *value),
            Filter::Lte(field, value) => self.matches_lte(doc_id, field, *value),
            Filter::In(field, values) => values.iter().any(|v| self.matches_eq(doc_id, field, v)),
            Filter::And(filters) => filters.iter().all(|f| self.matches(doc_id, f)),
            Filter::Or(filters) => filters.iter().any(|f| self.matches(doc_id, f)),
            Filter::Not(inner) => !self.matches(doc_id, inner),
        }
    }

    fn matches_eq(&self, doc_id: u32, field: &str, value: &FilterValue) -> bool {
        match (self.get(field), value) {
            (Some(FieldIndex::Keyword(idx)), FilterValue::String(s)) => idx.contains(doc_id, s),
            (Some(FieldIndex::Boolean(idx)), FilterValue::Bool(b)) => idx.matches(doc_id, *b),
            (Some(FieldIndex::Numeric(idx)), FilterValue::Number(n)) => idx.matches_eq(doc_id, *n),
            _ => false,
        }
    }

    fn matches_gt(&self, doc_id: u32, field: &str, value: f64) -> bool {
        match self.get(field) {
            Some(FieldIndex::Numeric(idx)) => idx.matches_gt(doc_id, value),
            _ => false,
        }
    }

    fn matches_gte(&self, doc_id: u32, field: &str, value: f64) -> bool {
        match self.get(field) {
            Some(FieldIndex::Numeric(idx)) => idx.matches_range(doc_id, value, f64::INFINITY),
            _ => false,
        }
    }

    fn matches_lt(&self, doc_id: u32, field: &str, value: f64) -> bool {
        match self.get(field) {
            Some(FieldIndex::Numeric(idx)) => idx.matches_lt(doc_id, value),
            _ => false,
        }
    }

    fn matches_lte(&self, doc_id: u32, field: &str, value: f64) -> bool {
        match self.get(field) {
            Some(FieldIndex::Numeric(idx)) => idx.matches_range(doc_id, f64::NEG_INFINITY, value),
            _ => false,
        }
    }
}

/// Filter value types
#[derive(Debug, Clone)]
pub enum FilterValue {
    String(String),
    Number(f64),
    Bool(bool),
}

/// Filter expressions
#[derive(Debug, Clone)]
pub enum Filter {
    Eq(String, FilterValue),
    Ne(String, FilterValue),
    Gt(String, f64),
    Gte(String, f64),
    Lt(String, f64),
    Lte(String, f64),
    In(String, Vec<FilterValue>),
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_keyword_index() {
        let mut idx = KeywordIndex::new();
        idx.insert(1, "apple");
        idx.insert(2, "apple");
        idx.insert(3, "banana");

        assert!(idx.contains(1, "apple"));
        assert!(idx.contains(2, "apple"));
        assert!(!idx.contains(3, "apple"));
        assert!(idx.contains(3, "banana"));

        let bitmap = idx.get("apple").unwrap();
        assert_eq!(bitmap.len(), 2);
    }

    #[test]
    fn test_boolean_index() {
        let mut idx = BooleanIndex::new();
        idx.insert(1, true);
        idx.insert(2, false);
        idx.insert(3, true);

        assert!(idx.matches(1, true));
        assert!(!idx.matches(1, false));
        assert!(idx.matches(2, false));
        assert!(idx.matches(3, true));
    }

    #[test]
    fn test_numeric_index() {
        let mut idx = NumericIndex::new();
        idx.insert(1, 10.0);
        idx.insert(2, 20.0);
        idx.insert(3, 30.0);

        // Equality
        assert!(idx.matches_eq(1, 10.0));
        assert!(!idx.matches_eq(1, 20.0));

        // Range
        let range = idx.get_range(15.0, 25.0);
        assert!(range.contains(2));
        assert!(!range.contains(1));
        assert!(!range.contains(3));
    }

    #[test]
    fn test_metadata_index_json() {
        let mut idx = MetadataIndex::new();

        idx.index_json(1, &json!({"category": "tech", "score": 85, "active": true}));
        idx.index_json(
            2,
            &json!({"category": "tech", "score": 92, "active": false}),
        );
        idx.index_json(
            3,
            &json!({"category": "science", "score": 78, "active": true}),
        );

        // Keyword filter
        let filter = Filter::Eq("category".into(), FilterValue::String("tech".into()));
        assert!(idx.matches(1, &filter));
        assert!(idx.matches(2, &filter));
        assert!(!idx.matches(3, &filter));

        // Numeric filter
        let filter = Filter::Gte("score".into(), 80.0);
        assert!(idx.matches(1, &filter));
        assert!(idx.matches(2, &filter));
        assert!(!idx.matches(3, &filter));

        // Boolean filter
        let filter = Filter::Eq("active".into(), FilterValue::Bool(true));
        assert!(idx.matches(1, &filter));
        assert!(!idx.matches(2, &filter));
        assert!(idx.matches(3, &filter));

        // Combined filter
        let filter = Filter::And(vec![
            Filter::Eq("category".into(), FilterValue::String("tech".into())),
            Filter::Eq("active".into(), FilterValue::Bool(true)),
        ]);
        assert!(idx.matches(1, &filter));
        assert!(!idx.matches(2, &filter));
        assert!(!idx.matches(3, &filter));
    }

    #[test]
    fn test_keyword_serialize_roundtrip() {
        let mut idx = KeywordIndex::new();
        idx.insert(1, "apple");
        idx.insert(2, "apple");
        idx.insert(3, "banana");

        let mut buf = Vec::new();
        idx.serialize(&mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let idx2 = KeywordIndex::deserialize(&mut cursor).unwrap();

        assert!(idx2.contains(1, "apple"));
        assert!(idx2.contains(2, "apple"));
        assert!(idx2.contains(3, "banana"));
    }
}
