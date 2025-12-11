//! Metadata filtering for vector search
//!
//! Provides MongoDB-style filter operators for post-hoc filtering of search results.

use serde_json::Value as JsonValue;

/// Metadata filter for vector search (MongoDB-style operators)
#[derive(Debug, Clone)]
pub enum MetadataFilter {
    /// Equality: field == value
    Eq(String, JsonValue),
    /// Not equal: field != value
    Ne(String, JsonValue),
    /// Greater than or equal: field >= value
    Gte(String, f64),
    /// Less than: field < value
    Lt(String, f64),
    /// Greater than: field > value
    Gt(String, f64),
    /// Less than or equal: field <= value
    Lte(String, f64),
    /// In list: field in [values]
    In(String, Vec<JsonValue>),
    /// Contains substring: field.contains(value)
    Contains(String, String),
    /// Logical AND: all filters must match
    And(Vec<MetadataFilter>),
    /// Logical OR: at least one filter must match
    Or(Vec<MetadataFilter>),
}

impl MetadataFilter {
    /// Evaluate filter against metadata
    #[must_use]
    pub fn matches(&self, metadata: &JsonValue) -> bool {
        match self {
            MetadataFilter::Eq(field, value) => metadata.get(field) == Some(value),
            MetadataFilter::Ne(field, value) => metadata.get(field) != Some(value),
            MetadataFilter::Gte(field, threshold) => metadata
                .get(field)
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|v| v >= *threshold),
            MetadataFilter::Lt(field, threshold) => metadata
                .get(field)
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|v| v < *threshold),
            MetadataFilter::Gt(field, threshold) => metadata
                .get(field)
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|v| v > *threshold),
            MetadataFilter::Lte(field, threshold) => metadata
                .get(field)
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|v| v <= *threshold),
            MetadataFilter::In(field, values) => {
                metadata.get(field).is_some_and(|v| values.contains(v))
            }
            MetadataFilter::Contains(field, substring) => metadata
                .get(field)
                .and_then(|v| v.as_str())
                .is_some_and(|s| s.contains(substring)),
            MetadataFilter::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            MetadataFilter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
        }
    }
}
