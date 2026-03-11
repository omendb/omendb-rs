#![allow(clippy::useless_conversion)]

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use omendb_lib::omen::Metric;
use omendb_lib::vector::{muvera::MultiVectorConfig, SearchResult};
use pyo3::conversion::IntoPyObject;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Py;
use serde_json::Value as JsonValue;

/// Parse quantization parameter and return bool (true = SQ8 enabled)
///
/// Accepts:
/// - True -> SQ8 (4x compression, ~99% recall) - RECOMMENDED
/// - "sq8" or "scalar" -> SQ8 (explicit)
/// - None/False -> no quantization (full precision)
pub(crate) fn parse_quantization(ob: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
    let Some(value) = ob else {
        return Ok(false);
    };

    // Handle boolean: True enables SQ8 (default), False disables
    if let Ok(b) = value.extract::<bool>() {
        return Ok(b);
    }

    // Handle string quantization modes
    if let Ok(mode) = value.extract::<String>() {
        return match mode.to_lowercase().as_str() {
            "sq8" | "scalar" => Ok(true),
            _ => Err(PyValueError::new_err(format!(
                "Unknown quantization mode: '{}'. Valid: True, 'sq8'",
                mode
            ))),
        };
    }

    Err(PyValueError::new_err(
        "quantization must be True, False, or 'sq8'",
    ))
}

/// Parse multi_vector parameter and return MultiVectorConfig if enabled
///
/// Accepts:
/// - True -> Default config (repetitions=8, partition_bits=4, d_proj=16)
/// - dict -> Custom config {"repetitions": N, "partition_bits": M, "d_proj": D}
/// - None/False -> disabled
///
/// Returns Ok(Some(config)) if enabled, Ok(None) if disabled
pub(crate) fn parse_multi_vector(
    ob: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<MultiVectorConfig>> {
    let Some(value) = ob else {
        return Ok(None);
    };

    // Handle boolean: True enables with defaults, False disables
    if let Ok(b) = value.extract::<bool>() {
        return if b {
            Ok(Some(MultiVectorConfig::default()))
        } else {
            Ok(None)
        };
    }

    // Handle dict with custom config
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut config = MultiVectorConfig::default();

        if let Some(reps) = dict.get_item("repetitions")? {
            config.repetitions = reps.extract()?;
        }
        if let Some(bits) = dict.get_item("partition_bits")? {
            config.partition_bits = bits.extract()?;
        }
        if let Some(seed) = dict.get_item("seed")? {
            config.seed = seed.extract()?;
        }
        if let Some(d_proj) = dict.get_item("d_proj")? {
            let val: Option<u8> = d_proj.extract()?;
            config.d_proj = val;
        }
        if let Some(pool_factor) = dict.get_item("pool_factor")? {
            let val: Option<u8> = pool_factor.extract()?;
            config.pool_factor = val;
        }

        return Ok(Some(config));
    }

    Err(PyValueError::new_err(
        "multi_vector must be True, False, or dict with {repetitions, partition_bits, d_proj, pool_factor}",
    ))
}

/// Extract multi-vector query (list of lists or 2D numpy array)
pub(crate) fn extract_multi_vector_query(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    // Try 2D numpy array first (most efficient)
    if let Ok(arr) = ob.extract::<PyReadonlyArray2<'_, f32>>() {
        let shape = arr.shape();
        let n_tokens = shape[0];
        let dim = shape[1];
        let mut tokens = Vec::with_capacity(n_tokens);

        if let Ok(slice) = arr.as_slice() {
            for i in 0..n_tokens {
                let start = i * dim;
                let end = start + dim;
                tokens.push(slice[start..end].to_vec());
            }
            return Ok(tokens);
        } else {
            return Err(PyValueError::new_err("2D array must be contiguous"));
        }
    }

    // Try list of lists
    if let Ok(outer) = ob.cast::<PyList>() {
        let mut tokens = Vec::with_capacity(outer.len());
        for item in outer.iter() {
            let token: Vec<f32> = item.extract()?;
            tokens.push(token);
        }
        return Ok(tokens);
    }

    Err(PyValueError::new_err(
        "multi-vector query must be a 2D numpy array or list of lists",
    ))
}

/// Extract single query vector from Python object (list or 1D numpy array)
pub(crate) fn extract_query_vector(ob: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    // Try 1D numpy array first (more efficient)
    if let Ok(arr) = ob.extract::<PyReadonlyArray1<'_, f32>>() {
        return arr
            .as_slice()
            .map(|s| s.to_vec())
            .map_err(|e| PyValueError::new_err(format!("Invalid numpy array: {}", e)));
    }
    // Fall back to list of floats
    if let Ok(list) = ob.extract::<Vec<f32>>() {
        return Ok(list);
    }
    Err(PyValueError::new_err(
        "query must be a list of floats or 1D numpy array (dtype=float32)",
    ))
}

/// Extract batch of query vectors from Python object (list of lists or 2D numpy array)
pub(crate) fn extract_batch_queries(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    // Try 2D numpy array first (most efficient)
    if let Ok(arr) = ob.extract::<PyReadonlyArray2<'_, f32>>() {
        let shape = arr.shape();
        let n_queries = shape[0];
        let dim = shape[1];
        let mut queries = Vec::with_capacity(n_queries);

        if let Ok(slice) = arr.as_slice() {
            for i in 0..n_queries {
                let start = i * dim;
                let end = start + dim;
                queries.push(slice[start..end].to_vec());
            }
            return Ok(queries);
        } else {
            return Err(PyValueError::new_err("2D array must be contiguous"));
        }
    }

    // Try list of lists/arrays
    if let Ok(list) = ob.extract::<Vec<Vec<f32>>>() {
        return Ok(list);
    }

    Err(PyValueError::new_err(
        "queries must be a 2D numpy array or list of lists",
    ))
}

/// Call an embedding function with a batch of documents, returning Vec<Vec<f32>>
///
/// The function signature is: (list[str]) -> ndarray[n, dim] or list[list[float]]
pub(crate) fn call_embedding_fn(
    py: Python<'_>,
    embedding_fn: &Py<PyAny>,
    documents: &[String],
) -> PyResult<Vec<Vec<f32>>> {
    let py_list = PyList::new(py, documents)?;
    let result = embedding_fn.call1(py, (py_list,))?;
    let result = result.bind(py);

    // Try 2D numpy array first (most common return type)
    if let Ok(arr) = result.extract::<PyReadonlyArray2<'_, f32>>() {
        let shape = arr.shape();
        let n = shape[0];
        let dim = shape[1];
        if n != documents.len() {
            return Err(PyValueError::new_err(format!(
                "embedding_fn returned {} vectors for {} documents",
                n,
                documents.len()
            )));
        }
        let mut vectors = Vec::with_capacity(n);
        if let Ok(slice) = arr.as_slice() {
            for i in 0..n {
                vectors.push(slice[i * dim..(i + 1) * dim].to_vec());
            }
        } else {
            return Err(PyValueError::new_err(
                "embedding_fn returned non-contiguous array",
            ));
        }
        return Ok(vectors);
    }

    // Try list of lists
    if let Ok(list) = result.extract::<Vec<Vec<f32>>>() {
        if list.len() != documents.len() {
            return Err(PyValueError::new_err(format!(
                "embedding_fn returned {} vectors for {} documents",
                list.len(),
                documents.len()
            )));
        }
        return Ok(list);
    }

    Err(PyValueError::new_err(
        "embedding_fn must return a 2D numpy array or list of lists of floats",
    ))
}

/// Convert PyO3 errors to Python exceptions with proper type mapping
pub(crate) fn convert_error(err: anyhow::Error) -> PyErr {
    let msg = err.to_string();
    // Map to appropriate Python exception types
    if msg.contains("dimension")
        || msg.contains("not found")
        || msg.contains("does not exist")
        || msg.contains("k=0")
        || msg.contains("Requirement: k > 0")
        || msg.contains("zero vector")
        || msg.contains("NaN or Infinity")
    {
        PyValueError::new_err(msg)
    } else {
        pyo3::exceptions::PyRuntimeError::new_err(msg)
    }
}

/// Convert raw distance to a normalized similarity score (0-1, higher = more similar).
pub(crate) fn distance_to_score(distance: f32, metric: Metric) -> f64 {
    let d = distance as f64;
    match metric {
        Metric::L2 => 1.0 / (1.0 + d),
        Metric::Cosine => 1.0 - d,
        Metric::InnerProduct => -d,
    }
}

/// Convert search results to Python list of dicts
pub(crate) fn results_to_py(
    py: Python<'_>,
    results: &[SearchResult],
    metric: Metric,
) -> PyResult<Vec<Py<PyDict>>> {
    let mut py_results = Vec::with_capacity(results.len());

    for result in results {
        let dict = PyDict::new(py);

        // Use interned strings for dict keys (hot path optimization)
        dict.set_item(pyo3::intern!(py, "id"), &result.id)?;
        dict.set_item(pyo3::intern!(py, "distance"), result.distance)?;
        dict.set_item(
            pyo3::intern!(py, "score"),
            distance_to_score(result.distance, metric),
        )?;

        // Convert metadata to Python dict
        let metadata_dict = json_to_pyobject(py, &result.metadata)?;
        dict.set_item(pyo3::intern!(py, "metadata"), metadata_dict)?;

        py_results.push(dict.unbind());
    }

    Ok(py_results)
}

/// Helper: Convert Python object to serde_json::Value
pub(crate) fn pyobject_to_json(obj: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    // Check None first (fast path)
    if obj.is_none() {
        Ok(JsonValue::Null)
    // Check bool BEFORE int/float - Python bool is subclass of int (True == 1, False == 0)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(JsonValue::Bool(b))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(JsonValue::String(s))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(JsonValue::Number(i.into()))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(serde_json::Number::from_f64(f)
            .map(JsonValue::Number)
            .unwrap_or(JsonValue::Null))
    } else if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (key, value) in dict.iter() {
            let key_str: String = key.extract()?;
            map.insert(key_str, pyobject_to_json(&value)?);
        }
        Ok(JsonValue::Object(map))
    } else if let Ok(list) = obj.cast::<PyList>() {
        let values: Result<Vec<_>, _> = list.iter().map(|item| pyobject_to_json(&item)).collect();
        Ok(JsonValue::Array(values?))
    } else {
        let type_name = obj
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        Err(PyValueError::new_err(format!(
            "Unsupported type '{}' for metadata. Supported: str, int, float, bool, None, list, dict",
            type_name
        )))
    }
}

/// Helper: Convert serde_json::Value to Python object
pub(crate) fn json_to_pyobject(py: Python<'_>, value: &JsonValue) -> PyResult<Py<PyAny>> {
    match value {
        JsonValue::Null => Ok(py.None()),
        JsonValue::Bool(b) => Ok((*b).into_pyobject(py).unwrap().to_owned().unbind().into()),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py).unwrap().unbind().into())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py).unwrap().unbind().into())
            } else {
                Ok(py.None())
            }
        }
        JsonValue::String(s) => Ok(s.clone().into_pyobject(py).unwrap().unbind().into()),
        JsonValue::Array(arr) => {
            let py_list = PyList::new(
                py,
                arr.iter()
                    .map(|v| json_to_pyobject(py, v))
                    .collect::<PyResult<Vec<_>>>()?,
            )?;
            Ok(py_list.into())
        }
        JsonValue::Object(obj) => {
            let py_dict = PyDict::new(py);
            for (k, v) in obj {
                py_dict.set_item(k, json_to_pyobject(py, v)?)?;
            }
            Ok(py_dict.into())
        }
    }
}
