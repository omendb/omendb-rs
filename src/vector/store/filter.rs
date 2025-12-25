//! Metadata filtering for vector search
//!
//! Provides MongoDB-style filter operators for post-hoc filtering of search results.
//! Supports both JSON-based evaluation and fast Roaring bitmap evaluation.

use crate::omen::{FieldIndex, MetadataIndex};
use roaring::RoaringBitmap;
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
    /// Combine this filter with another using AND
    #[must_use]
    pub fn and(self, other: MetadataFilter) -> Self {
        match self {
            MetadataFilter::And(mut filters) => {
                filters.push(other);
                MetadataFilter::And(filters)
            }
            _ => MetadataFilter::And(vec![self, other]),
        }
    }

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

    /// Evaluate filter using Roaring bitmap index for O(1) per-candidate filtering
    ///
    /// Returns a bitmap of all matching document IDs. For filters that can't be
    /// evaluated via bitmap (e.g., Contains), returns None to fall back to JSON-based filtering.
    #[must_use]
    pub fn evaluate_bitmap(&self, index: &MetadataIndex) -> Option<RoaringBitmap> {
        match self {
            MetadataFilter::Eq(field, value) => {
                match value {
                    JsonValue::String(s) => {
                        // Keyword equality - use inverted index
                        index.get(field).and_then(|field_idx| match field_idx {
                            FieldIndex::Keyword(kw_idx) => kw_idx.get(s).cloned(),
                            _ => None,
                        })
                    }
                    JsonValue::Bool(b) => {
                        // Boolean equality
                        index.get(field).and_then(|field_idx| match field_idx {
                            FieldIndex::Boolean(bool_idx) => Some(if *b {
                                bool_idx.get_true().clone()
                            } else {
                                bool_idx.get_false().clone()
                            }),
                            _ => None,
                        })
                    }
                    JsonValue::Number(n) => {
                        // Numeric equality
                        n.as_f64().and_then(|f| {
                            index.get(field).and_then(|field_idx| match field_idx {
                                FieldIndex::Numeric(num_idx) => num_idx.get_eq(f).cloned(),
                                _ => None,
                            })
                        })
                    }
                    _ => None,
                }
            }
            MetadataFilter::Gte(field, threshold) => {
                index.get(field).and_then(|field_idx| match field_idx {
                    FieldIndex::Numeric(num_idx) => Some(num_idx.get_range(*threshold, f64::MAX)),
                    _ => None,
                })
            }
            MetadataFilter::Gt(..) | MetadataFilter::Lt(..) => {
                // Strict inequalities have floating-point boundary issues with epsilon
                // Fall back to JSON-based filtering for correctness
                None
            }
            MetadataFilter::Lte(field, threshold) => {
                index.get(field).and_then(|field_idx| match field_idx {
                    FieldIndex::Numeric(num_idx) => Some(num_idx.get_range(f64::MIN, *threshold)),
                    _ => None,
                })
            }
            MetadataFilter::In(field, values) => {
                // Union of all matching values
                let mut result = RoaringBitmap::new();
                for value in values {
                    if let Some(bitmap) =
                        MetadataFilter::Eq(field.clone(), value.clone()).evaluate_bitmap(index)
                    {
                        result |= bitmap;
                    }
                }
                Some(result)
            }
            MetadataFilter::And(filters) => {
                // Intersection of all sub-filters
                let mut result: Option<RoaringBitmap> = None;
                for filter in filters {
                    match filter.evaluate_bitmap(index) {
                        Some(bitmap) => {
                            result = Some(match result {
                                Some(r) => r & bitmap,
                                None => bitmap,
                            });
                        }
                        None => return None, // Can't evaluate this filter via bitmap
                    }
                }
                result
            }
            MetadataFilter::Or(filters) => {
                // Union of all sub-filters
                let mut result = RoaringBitmap::new();
                for filter in filters {
                    match filter.evaluate_bitmap(index) {
                        Some(bitmap) => {
                            result |= bitmap;
                        }
                        None => return None, // Can't evaluate this filter via bitmap
                    }
                }
                Some(result)
            }
            // These can't be efficiently evaluated via bitmap
            MetadataFilter::Ne(..) | MetadataFilter::Contains(..) => None,
        }
    }
}
