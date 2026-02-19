use crate::conversions::{convert_error, parse_multi_vector, parse_quantization};
use crate::database::{VectorDatabase, VectorDatabaseInner};
use omendb_lib::vector::{VectorStore, VectorStoreOptions};
use parking_lot::RwLock;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Py;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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

/// Open or create a vector database.
///
/// All parameters except `path` are optional with sensible defaults.
///
/// Args:
///     path (str): Database directory path, or ":memory:" for in-memory
///     dimensions (int): Vector dimensionality (default: 128, auto-detected on first insert)
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
///     config (dict): Advanced config (deprecated, use top-level params instead)
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
#[pyo3(signature = (path, dimensions=0, m=None, ef_construction=None, ef_search=None, quantization=None, metric=None, multi_vector=None, config=None, embedding_fn=None, rescore=None, oversample=None))]
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
    config: Option<&Bound<'_, PyDict>>,
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

    // Resolve effective dimensions (use 128 as default if not specified)
    let effective_dims = if dimensions == 0 { 128 } else { dimensions };

    // Parse multi-vector config
    let mv_config = parse_multi_vector(multi_vector)?;
    let is_multi_vec = mv_config.is_some();

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
            let store =
                VectorStore::multi_vector_with(effective_dims, config).map_err(convert_error)?;

            return Ok(VectorDatabase {
                inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
                path,
                dimensions: effective_dims,
                is_persistent: false,
                is_multi_vector: true,
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

        let store = options
            .build()
            .map_err(|e| PyValueError::new_err(format!("Failed to create store: {}", e)))?;

        return Ok(VectorDatabase {
            inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
            path,
            dimensions: effective_dims,
            is_persistent: false,
            is_multi_vector: false,
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
            let store = VectorStore::open(&path).map_err(convert_error)?;
            let is_mv = store.is_multi_vector();
            let actual_dims = store.dimensions();

            // If multi_vector param conflicts with existing store, error
            if is_multi_vec && !is_mv {
                return Err(PyValueError::new_err(
                    "Cannot open existing single-vector database with multi_vector=True",
                ));
            }

            return Ok(VectorDatabase {
                inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
                path,
                dimensions: if actual_dims > 0 {
                    actual_dims
                } else {
                    effective_dims
                },
                is_persistent: true,
                is_multi_vector: is_mv,
                embedding_fn: embedding_fn.as_ref().map(|f| f.clone_ref(py)),
                collections_cache: RwLock::new(HashMap::new()),
            });
        }

        // Create new persistent store
        if let Some(mv_cfg) = mv_config {
            // Create new multi-vector persistent store
            let store = VectorStore::multi_vector_with(effective_dims, mv_cfg)
                .map_err(convert_error)?
                .persist(&path)
                .map_err(convert_error)?;

            return Ok(VectorDatabase {
                inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
                path,
                dimensions: effective_dims,
                is_persistent: true,
                is_multi_vector: true,
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

        // Handle config dict for backward compatibility
        if let Some(cfg) = config {
            if let Some(hnsw_dict) = cfg.get_item("hnsw")? {
                let hnsw = hnsw_dict
                    .cast::<PyDict>()
                    .map_err(|_| PyValueError::new_err("'hnsw' must be a dict"))?;

                if m.is_none() {
                    if let Some(m_item) = hnsw.get_item("m")? {
                        options = options.m(m_item.extract()?);
                    }
                }
                if ef_construction.is_none() {
                    if let Some(ef_item) = hnsw.get_item("ef_construction")? {
                        options = options.ef_construction(ef_item.extract()?);
                    }
                }
                if ef_search.is_none() {
                    if let Some(ef_item) = hnsw.get_item("ef_search")? {
                        options = options.ef_search(ef_item.extract()?);
                    }
                }
            }
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
        let store = options.open(&path).map_err(convert_error)?;

        return Ok(VectorDatabase {
            inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
            path,
            dimensions: effective_dims,
            is_persistent: true,
            is_multi_vector: false,
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

    let store = options
        .build()
        .map_err(|e| PyValueError::new_err(format!("Failed to create store: {}", e)))?;

    Ok(VectorDatabase {
        inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
        path,
        dimensions: effective_dims,
        is_persistent: false,
        is_multi_vector: false,
        embedding_fn,
        collections_cache: RwLock::new(HashMap::new()),
    })
}
