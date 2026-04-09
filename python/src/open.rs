use crate::conversions::{convert_error, parse_multi_vector, parse_quantization};
use crate::database::{VectorDatabase, VectorDatabaseInner};
use omendb_lib::catalog::{
    CollectionSchema, DenseSchema, FrozenDenseIndexKind, MultiEncoderKind, MultiSchema,
    MutableDenseIndexKind, QuantizationMode, SparseIndexKind, SparseSchema, TextSchema,
};
use omendb_lib::text::{TextSearchConfig, TokenizerPreset};
use omendb_lib::vector::{VectorStore, VectorStoreOptions};
use omendb_lib::Metric;
use parking_lot::RwLock;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Py;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn extract_dict<'py>(value: &'py Bound<'py, PyAny>, label: &str) -> PyResult<Bound<'py, PyDict>> {
    value.cast::<PyDict>()
        .map_err(|_| PyValueError::new_err(format!("{label} must be a dict")))
        .map(|dict| dict.clone())
}

fn parse_collection_schema(value: &Bound<'_, PyAny>) -> PyResult<CollectionSchema> {
    let dict = extract_dict(value, "schema")?;

    let dense = if let Some(dense) = dict.get_item("dense")? {
        if dense.is_none() {
            None
        } else {
            let dense = extract_dict(&dense, "schema['dense']")?;
            let dim: u32 = dense
                .get_item("dim")?
                .ok_or_else(|| PyValueError::new_err("schema['dense']['dim'] is required"))?
                .extract()?;
            let quantization = match dense
                .get_item("quantization")?
                .map(|value| value.extract::<String>())
                .transpose()?
                .as_deref()
                .unwrap_or("none")
            {
                "none" => QuantizationMode::None,
                "sq8" | "scalar" => QuantizationMode::Sq8,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown dense quantization: '{other}'"
                    )));
                }
            };
            Some(DenseSchema {
                dim,
                quantization,
                mutable_index: MutableDenseIndexKind::Hnsw,
                frozen_index: FrozenDenseIndexKind::Hnsw,
            })
        }
    } else {
        None
    };

    let sparse = if let Some(sparse) = dict.get_item("sparse")? {
        if sparse.is_none() {
            None
        } else {
            let sparse = extract_dict(&sparse, "schema['sparse']")?;
            Some(SparseSchema {
                index_kind: match sparse
                    .get_item("index_kind")?
                    .map(|value| value.extract::<String>())
                    .transpose()?
                    .as_deref()
                    .unwrap_or("inverted_exact")
                {
                    "inverted_exact" => SparseIndexKind::InvertedExact,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "Unknown sparse index kind: '{other}'"
                        )));
                    }
                },
                max_nonzero: sparse.get_item("max_nonzero")?.map(|value| value.extract()).transpose()?,
            })
        }
    } else {
        None
    };

    let multi = if let Some(multi) = dict.get_item("multi")? {
        if multi.is_none() {
            None
        } else {
            let multi = extract_dict(&multi, "schema['multi']")?;
            let token_dim: u32 = multi
                .get_item("token_dim")?
                .ok_or_else(|| PyValueError::new_err("schema['multi']['token_dim'] is required"))?
                .extract()?;
            Some(MultiSchema {
                token_dim,
                encoder: match multi
                    .get_item("encoder")?
                    .map(|value| value.extract::<String>())
                    .transpose()?
                    .as_deref()
                    .unwrap_or("muvera")
                {
                    "muvera" => MultiEncoderKind::Muvera,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "Unknown multi encoder: '{other}'"
                        )));
                    }
                },
                repetitions: multi
                    .get_item("repetitions")?
                    .map(|value| value.extract())
                    .transpose()?
                    .unwrap_or(8),
                partition_bits: multi
                    .get_item("partition_bits")?
                    .map(|value| value.extract())
                    .transpose()?
                    .unwrap_or(4),
                d_proj: multi.get_item("d_proj")?.map(|value| value.extract()).transpose()?,
                seed: multi
                    .get_item("seed")?
                    .map(|value| value.extract())
                    .transpose()?
                    .unwrap_or(42),
                max_tokens: multi.get_item("max_tokens")?.map(|value| value.extract()).transpose()?,
                pool_factor: multi.get_item("pool_factor")?.map(|value| value.extract()).transpose()?,
            })
        }
    } else {
        None
    };

    let text = if let Some(text) = dict.get_item("text")? {
        if text.is_none() {
            None
        } else {
            let text = extract_dict(&text, "schema['text']")?;
            let tokenizer = text
                .get_item("tokenizer")?
                .map(|value| value.extract::<String>())
                .transpose()?
                .map(|value| TokenizerPreset::parse(&value).map_err(convert_error))
                .transpose()?
                .unwrap_or_default();
            let writer_buffer_mb = text
                .get_item("writer_buffer_mb")?
                .or_else(|| text.get_item("buffer_mb").ok().flatten())
                .map(|value| value.extract())
                .transpose()?
                .unwrap_or(50usize);
            Some(TextSchema {
                tokenizer,
                writer_buffer_mb: writer_buffer_mb as u32,
            })
        }
    } else {
        None
    };

    let metric = if let Some(metric) = dict.get_item("metric")? {
        Metric::parse(&metric.extract::<String>()?).map_err(PyValueError::new_err)?
    } else if multi.is_some() {
        Metric::InnerProduct
    } else {
        Metric::L2
    };

    Ok(CollectionSchema {
        name: dict
            .get_item("name")?
            .map(|value| value.extract())
            .transpose()?
            .unwrap_or_default(),
        metric,
        dense,
        sparse,
        multi,
        text,
    })
}

/// Build VectorStoreOptions from open() parameters
pub(crate) fn build_store_options(
    dimensions: usize,
    m: Option<usize>,
    ef_construction: Option<usize>,
    ef_search: Option<usize>,
    quantization: bool,
    metric: Option<&str>,
    rescore: Option<bool>,
    oversample: Option<f32>,
) -> PyResult<VectorStoreOptions> {
    let mut options = VectorStoreOptions::default().dimensions(dimensions);

    if let Some(m_val) = m {
        options = options.m(m_val);
    }
    if let Some(ef_con) = ef_construction {
        options = options.ef_construction(ef_con);
    }
    if let Some(ef_s) = ef_search {
        options = options.ef_search(ef_s);
    }
    if quantization {
        options = options.quantization(true);
    }
    if let Some(metric_str) = metric {
        options = options.metric(metric_str).map_err(PyValueError::new_err)?;
    }
    if let Some(r) = rescore {
        options = options.rescore(r);
    }
    if let Some(o) = oversample {
        options = options.oversample(o);
    }

    Ok(options)
}

pub(crate) fn parse_text_search_config(
    text_search: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<TextSearchConfig>> {
    let Some(value) = text_search else {
        return Ok(None);
    };

    if let Ok(enabled) = value.extract::<bool>() {
        return Ok(enabled.then(TextSearchConfig::default));
    }

    let dict = value
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("text_search must be bool or dict"))?;

    let mut config = TextSearchConfig::default();

    if let Some(buffer_mb) = dict
        .get_item("buffer_mb")?
        .or_else(|| dict.get_item("writer_buffer_mb").ok().flatten())
    {
        config.writer_buffer_mb = buffer_mb.extract()?;
    }

    if let Some(tokenizer) = dict.get_item("tokenizer")? {
        let tokenizer_name: String = tokenizer.extract()?;
        config.tokenizer = TokenizerPreset::parse(&tokenizer_name).map_err(convert_error)?;
    }

    Ok(Some(config))
}

/// Open or create a vector database.
///
/// All parameters except `path` are optional with sensible defaults.
///
/// Args:
///     path (str): Database directory path, or ":memory:" for in-memory
///     dimensions (int): Vector dimensionality (default: infer on first insert for single-vector stores)
///     m (int): HNSW neighbors per node (default: 16, range: 4-64)
///     ef_construction (int): Build quality (default: 100, higher = better graph)
///     ef_search (int): Search quality (default: 100, higher = better recall)
///     quantization (bool|str): Enable quantization (default: None = full precision)
///         - True or "sq8" or "scalar": SQ8 ~4x smaller, ~99% recall (RECOMMENDED)
///         - False/None: Full precision (no quantization)
///     metric (str): Distance metric for similarity search (default: "l2")
///         - "l2" or "euclidean": Euclidean distance (default)
///         - "cosine": Cosine distance (1 - cosine similarity)
///         - "dot" or "ip": Inner product (for MIPS)
///     multi_vector (bool|dict): Enable multi-vector mode for ColBERT-style retrieval
///         - True: Enable with default config (repetitions=8, partition_bits=4, d_proj=16)
///         - dict: Custom config {"repetitions": 10, "partition_bits": 4, "d_proj": 16}
///         - d_proj: Dimension projection (16 = 8x smaller FDE, None = full token dim)
///         - False/None: Single-vector mode (default)
///
/// Returns:
///     VectorDatabase: Database instance
///
/// Raises:
///     ValueError: If parameters are invalid
///     RuntimeError: If database creation fails
///
/// Examples:
///     >>> import omendb
///
///     # Simple usage with defaults
///     >>> db = omendb.open("./my_vectors", dimensions=768)
///
///     # With SQ8 quantization (4x smaller, similar speed, ~99% recall)
///     >>> db = omendb.open("./vectors", dimensions=768, quantization=True)
///     >>> db = omendb.open("./vectors", dimensions=768, quantization="sq8")
///
///     # Multi-vector mode for ColBERT-style retrieval
///     >>> db = omendb.open("./vectors", dimensions=128, multi_vector=True)
///     >>> db.set([{"id": "doc1", "vectors": [[0.1]*128, [0.2]*128], "metadata": {}}])
///     >>> results = db.search([[0.1]*128], k=10)
///
///     # With cosine distance metric
///     >>> db = omendb.open("./vectors", dimensions=768, metric="cosine")
#[pyfunction]
#[pyo3(signature = (path, dimensions=0, m=None, ef_construction=None, ef_search=None, quantization=None, metric=None, multi_vector=None, text_search=None, embedding_fn=None, rescore=None, oversample=None))]
pub(crate) fn open(
    py: Python<'_>,
    path: String,
    dimensions: usize,
    m: Option<usize>,
    ef_construction: Option<usize>,
    ef_search: Option<usize>,
    quantization: Option<&Bound<'_, PyAny>>,
    metric: Option<String>,
    multi_vector: Option<&Bound<'_, PyAny>>,
    text_search: Option<&Bound<'_, PyAny>>,
    embedding_fn: Option<Py<PyAny>>,
    rescore: Option<bool>,
    oversample: Option<f32>,
) -> PyResult<VectorDatabase> {
    // Validate optional params
    if let Some(m_val) = m {
        if !(4..=64).contains(&m_val) {
            return Err(PyValueError::new_err(format!(
                "m must be between 4 and 64, got {}",
                m_val
            )));
        }
    }

    // Parse quantization mode
    let quant_mode = parse_quantization(quantization)?;

    if let (Some(ef_val), Some(m_val)) = (ef_construction, m) {
        if ef_val < m_val {
            return Err(PyValueError::new_err(format!(
                "ef_construction ({}) must be >= m ({})",
                ef_val, m_val
            )));
        }
    }

    // Validate metric
    if let Some(ref m) = metric {
        match m.to_lowercase().as_str() {
            "l2" | "euclidean" | "cosine" | "dot" | "ip" => {}
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown metric: '{}'. Valid: l2, euclidean, cosine, dot, ip",
                    m
                )));
            }
        }
    }

    let effective_dims = dimensions;

    // Parse multi-vector config
    let mv_config = parse_multi_vector(multi_vector)?;
    let is_multi_vec = mv_config.is_some();
    let text_search_config = parse_text_search_config(text_search)?;

    if is_multi_vec && effective_dims == 0 {
        return Err(PyValueError::new_err(
            "dimensions must be greater than 0 for multi-vector stores",
        ));
    }

    // Multi-vector stores don't support quantization yet
    if is_multi_vec && quant_mode {
        return Err(PyValueError::new_err(
            "Multi-vector stores don't support quantization yet",
        ));
    }

    // Quantization only supports L2 distance
    if quant_mode {
        let m = metric.as_deref().unwrap_or("l2");
        if m != "l2" && m != "euclidean" {
            return Err(PyValueError::new_err(format!(
                "Quantization only supports L2 distance, got metric='{m}'"
            )));
        }
    }

    // Handle :memory: for in-memory database (must check BEFORE path existence checks)
    if path == ":memory:" {
        // Multi-vector mode: use VectorStore::multi_vector() constructor
        if let Some(config) = mv_config {
            let mut store =
                VectorStore::multi_vector_with(effective_dims, config).map_err(convert_error)?;
            if let Some(config) = text_search_config.clone() {
                store
                    .enable_text_search_with_config(Some(config))
                    .map_err(convert_error)?;
            }

            return Ok(VectorDatabase {
                inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
                path,
                is_persistent: false,
                embedding_fn: embedding_fn.as_ref().map(|f| f.clone_ref(py)),
                collections_cache: RwLock::new(HashMap::new()),
            });
        }

        // Single-vector mode (original logic)
        let options = build_store_options(
            effective_dims,
            m,
            ef_construction,
            ef_search,
            quant_mode,
            metric.as_deref(),
            rescore,
            oversample,
        )?;
        let options = if let Some(ref config) = text_search_config {
            options.text_search_config(config.clone())
        } else {
            options
        };

        let store = options
            .build()
            .map_err(|e| PyValueError::new_err(format!("Failed to create store: {}", e)))?;

        return Ok(VectorDatabase {
            inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
            path,
            is_persistent: false,
            embedding_fn: embedding_fn.as_ref().map(|f| f.clone_ref(py)),
            collections_cache: RwLock::new(HashMap::new()),
        });
    }

    let db_path = Path::new(&path);
    // Compute .omen path by appending extension (preserves full filename)
    let omen_path = if db_path.extension().is_some_and(|ext| ext == "omen") {
        db_path.to_path_buf()
    } else {
        let mut omen = db_path.as_os_str().to_os_string();
        omen.push(".omen");
        PathBuf::from(omen)
    };

    // Check if this is a directory (persistent storage) or .omen file exists
    if db_path.is_dir() || omen_path.exists() || !db_path.exists() {
        // Check for existing database that may have multi-vector config
        if omen_path.exists() {
            let mut store = VectorStore::open(&path).map_err(convert_error)?;
            if let Some(config) = text_search_config.clone() {
                store
                    .enable_text_search_with_config(Some(config))
                    .map_err(convert_error)?;
            }
            let is_mv = store.is_multi_vector();

            // If multi_vector param conflicts with existing store, error
            if is_multi_vec && !is_mv {
                return Err(PyValueError::new_err(
                    "Cannot open existing single-vector database with multi_vector=True",
                ));
            }

            return Ok(VectorDatabase {
                inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
                path,
                is_persistent: true,
                embedding_fn: embedding_fn.as_ref().map(|f| f.clone_ref(py)),
                collections_cache: RwLock::new(HashMap::new()),
            });
        }

        // Create new persistent store
        if let Some(mv_cfg) = mv_config {
            // Create new multi-vector persistent store
            let mut store = VectorStore::multi_vector_with(effective_dims, mv_cfg)
                .map_err(convert_error)?
                .persist(&path)
                .map_err(convert_error)?;
            if let Some(config) = text_search_config.clone() {
                store
                    .enable_text_search_with_config(Some(config))
                    .map_err(convert_error)?;
            }

            return Ok(VectorDatabase {
                inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
                path,
                is_persistent: true,
                embedding_fn: embedding_fn.as_ref().map(|f| f.clone_ref(py)),
                collections_cache: RwLock::new(HashMap::new()),
            });
        }

        // Single-vector persistent store
        let mut options = build_store_options(
            effective_dims,
            m,
            ef_construction,
            ef_search,
            quant_mode,
            metric.as_deref(),
            rescore,
            oversample,
        )?;
        if let Some(ref config) = text_search_config {
            options = options.text_search_config(config.clone());
        }

        // Check if enabling quantization on existing non-empty database
        if db_path.exists() && quant_mode {
            let existing = VectorStore::open(&path).map_err(convert_error)?;
            if !existing.is_empty() {
                return Err(PyValueError::new_err(
                    "Cannot enable quantization on existing database. Create a new database with quantization.",
                ));
            }
        }

        // Open with options
        let mut store = options.open(&path).map_err(convert_error)?;
        if let Some(config) = text_search_config {
            store
                .enable_text_search_with_config(Some(config))
                .map_err(convert_error)?;
        }

        return Ok(VectorDatabase {
            inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
            path,
            is_persistent: true,
            embedding_fn: embedding_fn.as_ref().map(|f| f.clone_ref(py)),
            collections_cache: RwLock::new(HashMap::new()),
        });
    }

    // Fallback: create new in-memory database with configuration
    let options = build_store_options(
        effective_dims,
        m,
        ef_construction,
        ef_search,
        quant_mode,
        metric.as_deref(),
        rescore,
        oversample,
    )?;
    let options = if let Some(ref config) = text_search_config {
        options.text_search_config(config.clone())
    } else {
        options
    };

    let store = options
        .build()
        .map_err(|e| PyValueError::new_err(format!("Failed to create store: {}", e)))?;

    Ok(VectorDatabase {
        inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
        path,
        is_persistent: false,
        embedding_fn,
        collections_cache: RwLock::new(HashMap::new()),
    })
}

/// Create a database from an explicit collection schema.
#[pyfunction]
#[pyo3(signature = (path, schema, embedding_fn=None))]
pub(crate) fn create(
    py: Python<'_>,
    path: String,
    schema: &Bound<'_, PyAny>,
    embedding_fn: Option<Py<PyAny>>,
) -> PyResult<VectorDatabase> {
    let schema = parse_collection_schema(schema)?;
    let is_persistent = path != ":memory:";
    let store = if is_persistent {
        VectorStore::create(&path, schema).map_err(convert_error)?
    } else {
        VectorStore::create_in_memory(schema).map_err(convert_error)?
    };

    Ok(VectorDatabase {
        inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
        path,
        is_persistent,
        embedding_fn: embedding_fn.as_ref().map(|f| f.clone_ref(py)),
        collections_cache: RwLock::new(HashMap::new()),
    })
}
