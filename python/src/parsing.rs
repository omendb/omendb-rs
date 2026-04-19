use crate::conversions::{extract_multi_vector_query, pyobject_to_json};
use omendb_lib::vector::Vector;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value as JsonValue;

/// Parsed batch item with optional text for hybrid search and optional document for embedding
pub(crate) struct ParsedItem {
    pub(crate) id: String,
    pub(crate) vector: Vector,
    pub(crate) metadata: JsonValue,
    pub(crate) text: Option<String>,
    pub(crate) document: Option<String>,
}

/// Parsed multi-vector batch item
pub(crate) struct ParsedMultiVecItem {
    pub(crate) id: String,
    pub(crate) vectors: Vec<Vec<f32>>,
    pub(crate) metadata: JsonValue,
}

/// Parse batch items from a list of dicts, including optional text and document fields
pub(crate) fn parse_batch_items_with_text(items: &Bound<'_, PyList>) -> PyResult<Vec<ParsedItem>> {
    let mut batch = Vec::new();

    for (idx, item) in items.iter().enumerate() {
        let dict = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!("Item at index {} must be a dict", idx)))?;

        let id: String = dict
            .get_item("id")?
            .ok_or_else(|| {
                PyValueError::new_err(format!("Item at index {} missing 'id' field", idx))
            })?
            .extract()?;

        // Check for document field (auto-embedded via embedding_fn)
        let document: Option<String> = dict
            .get_item("document")?
            .map(|d| d.extract())
            .transpose()
            .map_err(|_| {
                PyValueError::new_err(format!("Item '{}': 'document' must be a string", id))
            })?;

        // Vector is optional when document is present
        let vector_obj = dict.get_item("vector")?;

        if vector_obj.is_some() && document.is_some() {
            return Err(PyValueError::new_err(format!(
                "Item '{}': cannot have both 'vector' and 'document' - use one or the other",
                id
            )));
        }

        let vector = if let Some(v) = vector_obj {
            let vector_data: Vec<f32> = v.extract()?;
            Vector::new(vector_data)
        } else if document.is_some() {
            // Placeholder - will be replaced after embedding
            Vector::new(vec![])
        } else {
            return Err(PyValueError::new_err(format!(
                "Item '{}' must have either 'vector' or 'document' field",
                id
            )));
        };

        let mut metadata_json = if let Some(metadata_dict) = dict.get_item("metadata")? {
            pyobject_to_json(&metadata_dict)?
        } else {
            serde_json::json!({})
        };

        // Handle optional text field for hybrid search
        let text: Option<String> = dict
            .get_item("text")?
            .map(|t| t.extract())
            .transpose()
            .map_err(|_| {
                PyValueError::new_err(format!("Item '{}': 'text' must be a string", id))
            })?;

        // Auto-store text in metadata for retrieval
        if let Some(ref text_str) = text
            && let Some(obj) = metadata_json.as_object_mut()
        {
            if obj.contains_key("text") {
                return Err(PyValueError::new_err(format!(
                    "Item '{}': cannot have both 'text' field and 'metadata.text' - use one or the other",
                    id
                )));
            }
            obj.insert("text".to_string(), serde_json::json!(text_str));
        }

        batch.push(ParsedItem {
            id,
            vector,
            metadata: metadata_json,
            text,
            document,
        });
    }

    Ok(batch)
}

/// Parse batch items for multi-vector store (uses "vectors" key)
pub(crate) fn parse_multi_vec_items(
    items: &Bound<'_, PyList>,
) -> PyResult<Vec<ParsedMultiVecItem>> {
    let mut batch = Vec::new();

    for (idx, item) in items.iter().enumerate() {
        let dict = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!("Item at index {} must be a dict", idx)))?;

        let id: String = dict
            .get_item("id")?
            .ok_or_else(|| {
                PyValueError::new_err(format!("Item at index {} missing 'id' field", idx))
            })?
            .extract()?;

        // Multi-vector uses "vectors" key (list of lists)
        let vectors_obj = dict.get_item("vectors")?.ok_or_else(|| {
            PyValueError::new_err(format!(
                "Item '{}' missing 'vectors' field (multi-vector store)",
                id
            ))
        })?;

        let vectors = extract_multi_vector_query(&vectors_obj)?;

        if vectors.is_empty() {
            return Err(PyValueError::new_err(format!(
                "Item '{}': 'vectors' must not be empty",
                id
            )));
        }

        let metadata_json = if let Some(metadata_dict) = dict.get_item("metadata")? {
            pyobject_to_json(&metadata_dict)?
        } else {
            serde_json::json!({})
        };

        batch.push(ParsedMultiVecItem {
            id,
            vectors,
            metadata: metadata_json,
        });
    }

    Ok(batch)
}
