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
    /// Parse a metadata filter from JSON (MongoDB-style query syntax)
    ///
    /// Supported operators:
    /// - `{"field": value}` - equality
    /// - `{"field": {"$eq": value}}` - equality
    /// - `{"field": {"$ne": value}}` - not equal
    /// - `{"field": {"$gt": number}}` - greater than
    /// - `{"field": {"$gte": number}}` - greater than or equal
    /// - `{"field": {"$lt": number}}` - less than
    /// - `{"field": {"$lte": number}}` - less than or equal
    /// - `{"field": {"$in": [values]}}` - in list
    /// - `{"field": {"$contains": "substring"}}` - contains substring
    /// - `{"$and": [filters]}` - logical AND
    /// - `{"$or": [filters]}` - logical OR
    pub fn from_json(value: &JsonValue) -> Result<Self, String> {
        let obj = value
            .as_object()
            .ok_or_else(|| "Filter must be a JSON object".to_string())?;

        if obj.is_empty() {
            return Err("Filter cannot be empty".to_string());
        }

        let mut filters = Vec::new();

        for (key, val) in obj {
            let filter = if key == "$and" {
                let arr = val
                    .as_array()
                    .ok_or_else(|| "$and must be an array".to_string())?;
                let sub_filters: Result<Vec<_>, _> =
                    arr.iter().map(MetadataFilter::from_json).collect();
                MetadataFilter::And(sub_filters?)
            } else if key == "$or" {
                let arr = val
                    .as_array()
                    .ok_or_else(|| "$or must be an array".to_string())?;
                let sub_filters: Result<Vec<_>, _> =
                    arr.iter().map(MetadataFilter::from_json).collect();
                MetadataFilter::Or(sub_filters?)
            } else if let Some(op_obj) = val.as_object() {
                // Field with operator(s)
                Self::parse_field_operators(key, op_obj)?
            } else {
                // Simple equality: {"field": value}
                MetadataFilter::Eq(key.clone(), val.clone())
            };
            filters.push(filter);
        }

        if filters.len() == 1 {
            Ok(filters.remove(0))
        } else {
            Ok(MetadataFilter::And(filters))
        }
    }

    /// Parse operators for a single field
    fn parse_field_operators(
        field: &str,
        ops: &serde_json::Map<String, JsonValue>,
    ) -> Result<Self, String> {
        let mut filters = Vec::new();

        for (op, val) in ops {
            let filter = match op.as_str() {
                "$eq" => MetadataFilter::Eq(field.to_string(), val.clone()),
                "$ne" => MetadataFilter::Ne(field.to_string(), val.clone()),
                "$gt" => {
                    let n = val
                        .as_f64()
                        .ok_or_else(|| format!("$gt requires a number, got {val}"))?;
                    MetadataFilter::Gt(field.to_string(), n)
                }
                "$gte" => {
                    let n = val
                        .as_f64()
                        .ok_or_else(|| format!("$gte requires a number, got {val}"))?;
                    MetadataFilter::Gte(field.to_string(), n)
                }
                "$lt" => {
                    let n = val
                        .as_f64()
                        .ok_or_else(|| format!("$lt requires a number, got {val}"))?;
                    MetadataFilter::Lt(field.to_string(), n)
                }
                "$lte" => {
                    let n = val
                        .as_f64()
                        .ok_or_else(|| format!("$lte requires a number, got {val}"))?;
                    MetadataFilter::Lte(field.to_string(), n)
                }
                "$in" => {
                    let arr = val
                        .as_array()
                        .ok_or_else(|| "$in requires an array".to_string())?;
                    MetadataFilter::In(field.to_string(), arr.clone())
                }
                "$contains" => {
                    let s = val
                        .as_str()
                        .ok_or_else(|| "$contains requires a string".to_string())?;
                    MetadataFilter::Contains(field.to_string(), s.to_string())
                }
                other => return Err(format!("Unknown operator: {other}")),
            };
            filters.push(filter);
        }

        if filters.len() == 1 {
            Ok(filters.remove(0))
        } else {
            Ok(MetadataFilter::And(filters))
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_from_json_simple_equality() {
        let filter = MetadataFilter::from_json(&json!({"category": "books"})).unwrap();
        assert!(filter.matches(&json!({"category": "books"})));
        assert!(!filter.matches(&json!({"category": "movies"})));
    }

    #[test]
    fn test_from_json_eq_operator() {
        let filter = MetadataFilter::from_json(&json!({"category": {"$eq": "books"}})).unwrap();
        assert!(filter.matches(&json!({"category": "books"})));
        assert!(!filter.matches(&json!({"category": "movies"})));
    }

    #[test]
    fn test_from_json_ne_operator() {
        let filter = MetadataFilter::from_json(&json!({"category": {"$ne": "books"}})).unwrap();
        assert!(!filter.matches(&json!({"category": "books"})));
        assert!(filter.matches(&json!({"category": "movies"})));
    }

    #[test]
    fn test_from_json_numeric_operators() {
        let gte = MetadataFilter::from_json(&json!({"price": {"$gte": 10.0}})).unwrap();
        assert!(gte.matches(&json!({"price": 10.0})));
        assert!(gte.matches(&json!({"price": 15.0})));
        assert!(!gte.matches(&json!({"price": 5.0})));

        let lt = MetadataFilter::from_json(&json!({"price": {"$lt": 10.0}})).unwrap();
        assert!(!lt.matches(&json!({"price": 10.0})));
        assert!(lt.matches(&json!({"price": 5.0})));

        let gt = MetadataFilter::from_json(&json!({"price": {"$gt": 10.0}})).unwrap();
        assert!(!gt.matches(&json!({"price": 10.0})));
        assert!(gt.matches(&json!({"price": 15.0})));

        let lte = MetadataFilter::from_json(&json!({"price": {"$lte": 10.0}})).unwrap();
        assert!(lte.matches(&json!({"price": 10.0})));
        assert!(lte.matches(&json!({"price": 5.0})));
        assert!(!lte.matches(&json!({"price": 15.0})));
    }

    #[test]
    fn test_from_json_in_operator() {
        let filter =
            MetadataFilter::from_json(&json!({"category": {"$in": ["books", "movies"]}})).unwrap();
        assert!(filter.matches(&json!({"category": "books"})));
        assert!(filter.matches(&json!({"category": "movies"})));
        assert!(!filter.matches(&json!({"category": "music"})));
    }

    #[test]
    fn test_from_json_contains_operator() {
        let filter = MetadataFilter::from_json(&json!({"title": {"$contains": "rust"}})).unwrap();
        assert!(filter.matches(&json!({"title": "learning rust programming"})));
        assert!(!filter.matches(&json!({"title": "learning python"})));
    }

    #[test]
    fn test_from_json_and_operator() {
        let filter = MetadataFilter::from_json(
            &json!({"$and": [{"category": "books"}, {"price": {"$lt": 20.0}}]}),
        )
        .unwrap();
        assert!(filter.matches(&json!({"category": "books", "price": 15.0})));
        assert!(!filter.matches(&json!({"category": "books", "price": 25.0})));
        assert!(!filter.matches(&json!({"category": "movies", "price": 15.0})));
    }

    #[test]
    fn test_from_json_or_operator() {
        let filter = MetadataFilter::from_json(
            &json!({"$or": [{"category": "books"}, {"category": "movies"}]}),
        )
        .unwrap();
        assert!(filter.matches(&json!({"category": "books"})));
        assert!(filter.matches(&json!({"category": "movies"})));
        assert!(!filter.matches(&json!({"category": "music"})));
    }

    #[test]
    fn test_from_json_multiple_fields() {
        let filter =
            MetadataFilter::from_json(&json!({"category": "books", "price": {"$gte": 10.0}}))
                .unwrap();
        assert!(filter.matches(&json!({"category": "books", "price": 15.0})));
        assert!(!filter.matches(&json!({"category": "books", "price": 5.0})));
        assert!(!filter.matches(&json!({"category": "movies", "price": 15.0})));
    }

    #[test]
    fn test_from_json_errors() {
        assert!(MetadataFilter::from_json(&json!([])).is_err());
        assert!(MetadataFilter::from_json(&json!({})).is_err());
        assert!(MetadataFilter::from_json(&json!({"price": {"$unknown": 10}})).is_err());
        assert!(MetadataFilter::from_json(&json!({"price": {"$gt": "not_a_number"}})).is_err());
    }
}
