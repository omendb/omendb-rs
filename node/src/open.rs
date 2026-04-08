use napi::bindgen_prelude::*;
use napi_derive::napi;
use omendb_lib::vector::{VectorStore, VectorStoreOptions};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::conversions::{
    convert_error, parse_multi_vector, parse_quantization, parse_text_search_config,
};
use crate::database::{EmbeddingFn, VectorDatabase, VectorDatabaseInner};

/// Configuration options for opening a vector database.
///
/// All fields are optional with sensible defaults:
/// - dimensions: inferred from first insert when omitted for single-vector stores
/// - m: 16 (HNSW neighbors per node, higher = better recall, more memory)
/// - efConstruction: 100 (build quality, higher = better graph, slower build)
/// - efSearch: 100 (search quality, higher = better recall, slower search)
/// - quantization: null (true/"sq8" for 4x compression)
/// - metric: "l2" (distance metric: "l2", "euclidean", "cosine", "dot", "ip")
#[napi(object)]
pub struct OpenOptions {
    /// Vector dimensions (default: inferred on first insert for single-vector stores)
    pub dimensions: Option<u32>,
    /// HNSW M parameter: neighbors per node (default: 16, range: 4-64)
    pub m: Option<u32>,
    /// HNSW ef_construction: build quality (default: 100, must be >= m)
    pub ef_construction: Option<u32>,
    /// HNSW ef_search: search quality/speed tradeoff (default: 100)
    pub ef_search: Option<u32>,
    /// Quantization mode (default: null = no quantization)
    /// - true or "sq8": SQ8 4x compression, ~99% recall (RECOMMENDED)
    /// - false/null: Full precision (no quantization)
    #[napi(ts_type = "boolean | 'sq8' | 'scalar' | null | undefined")]
    pub quantization: Option<serde_json::Value>,
    /// Distance metric: "l2"/"euclidean" (default), "cosine", "dot"/"ip"
    pub metric: Option<String>,
    /// Enable multi-vector mode for ColBERT-style retrieval
    /// - true: Enable with default config (repetitions=8, partition_bits=4, dProj=16)
    /// - { repetitions?, partitionBits?, seed?, dProj? }: Custom config
    /// - dProj: Dimension projection (16 = 8x smaller FDE, null = full token dim)
    /// - false/null: Disabled (default, single-vector mode)
    #[napi(
        ts_type = "boolean | { repetitions?: number; partitionBits?: number; seed?: number; dProj?: number | null; poolFactor?: number | null } | null | undefined"
    )]
    pub multi_vector: Option<serde_json::Value>,
    /// SQ8 refiner: rescore with full precision (default: true when quantized)
    pub rescore: Option<bool>,
    /// Candidate multiplier for rescoring (default: 3.0)
    pub oversample: Option<f64>,
    /// Enable text search at open time.
    /// - true: default buffer/tokenizer
    /// - { bufferMb?, writerBufferMb?, tokenizer? }: custom text config
    /// - false/null: disabled
    #[napi(
        ts_type = "boolean | { bufferMb?: number; writerBufferMb?: number; tokenizer?: 'default' | 'code' | 'raw' } | null | undefined"
    )]
    pub text_search: Option<serde_json::Value>,
}

/// Open or create a vector database.
///
/// @param path - Database directory path (use ":memory:" for in-memory)
/// @param options - Optional configuration (see OpenOptions for defaults)
/// @returns VectorDatabase instance
///
/// @example
/// ```javascript
/// // Simple usage with defaults
/// const db = omendb.open("./mydb");
///
/// // With custom HNSW parameters
/// const db = omendb.open("./mydb", {
///   dimensions: 384,
///   m: 32,
///   efConstruction: 200,
///   efSearch: 150
/// });
///
/// // With SQ8 quantization (4x memory reduction, ~99% recall)
/// const db = omendb.open("./mydb", {
///   dimensions: 128,
///   quantization: true  // or "sq8"
/// });
///
/// ```
#[napi]
pub fn open(
    path: String,
    options: Option<OpenOptions>,
    #[napi(ts_arg_type = "((texts: string[]) => Float32Array[]) | undefined")] embedding_fn: Option<
        EmbeddingFn,
    >,
) -> Result<VectorDatabase> {
    let opts = options.unwrap_or(OpenOptions {
        dimensions: None,
        m: None,
        ef_construction: None,
        ef_search: None,
        quantization: None,
        metric: None,
        multi_vector: None,
        rescore: None,
        oversample: None,
        text_search: None,
    });

    let embedding_tsfn = embedding_fn.map(Arc::new);

    let dimensions = opts.dimensions.map_or(0, |v| v as usize);
    let m = opts.m.map(|v| v as usize);
    let ef_construction = opts.ef_construction.map(|v| v as usize);
    let ef_search = opts.ef_search.map(|v| v as usize);

    let quant_mode = opts
        .quantization
        .as_ref()
        .map(parse_quantization)
        .transpose()?
        .unwrap_or(false);

    let multi_vector_config = opts
        .multi_vector
        .as_ref()
        .map(parse_multi_vector)
        .transpose()?
        .flatten();
    let text_search_config = opts
        .text_search
        .as_ref()
        .map(parse_text_search_config)
        .transpose()?
        .flatten();

    let is_multi_vector = multi_vector_config.is_some();

    if is_multi_vector && quant_mode {
        return Err(Error::new(
            Status::InvalidArg,
            "multi-vector stores do not support quantization yet",
        ));
    }

    if quant_mode {
        let m = opts.metric.as_deref().unwrap_or("l2");
        if m != "l2" && m != "euclidean" {
            return Err(Error::new(
                Status::InvalidArg,
                format!("Quantization only supports L2 distance, got metric='{m}'"),
            ));
        }
    }

    if is_multi_vector && dimensions == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "dimensions must be greater than 0 for multi-vector stores",
        ));
    }

    if let Some(m_val) = m {
        if !(4..=64).contains(&m_val) {
            return Err(Error::new(
                Status::InvalidArg,
                format!("m must be between 4 and 64, got {}", m_val),
            ));
        }
    }

    if let (Some(ef_val), Some(m_val)) = (ef_construction, m) {
        if ef_val < m_val {
            return Err(Error::new(
                Status::InvalidArg,
                format!("ef_construction ({}) must be >= m ({})", ef_val, m_val),
            ));
        }
    }

    if let Some(ref m) = opts.metric {
        match m.to_lowercase().as_str() {
            "l2" | "euclidean" | "cosine" | "dot" | "ip" => {}
            _ => {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!(
                        "Unknown metric: '{}'. Valid: l2, euclidean, cosine, dot, ip",
                        m
                    ),
                ));
            }
        }
    }

    let mut store_options = VectorStoreOptions::default().dimensions(dimensions);

    if let Some(m_val) = m {
        store_options = store_options.m(m_val);
    }
    if let Some(ef_con) = ef_construction {
        store_options = store_options.ef_construction(ef_con);
    }
    if let Some(ef_s) = ef_search {
        store_options = store_options.ef_search(ef_s);
    }
    if quant_mode {
        store_options = store_options.quantization(true);
    }
    if let Some(ref metric_str) = opts.metric {
        store_options = store_options
            .metric(metric_str)
            .map_err(|e| Error::new(Status::InvalidArg, e))?;
    }
    if let Some(rescore) = opts.rescore {
        store_options = store_options.rescore(rescore);
    }
    if let Some(oversample) = opts.oversample {
        store_options = store_options.oversample(oversample as f32);
    }
    if let Some(ref text_config) = text_search_config {
        store_options = store_options.text_search_config(text_config.clone());
    }

    if path == ":memory:" {
        let mut store = if let Some(mv_config) = multi_vector_config {
            VectorStore::multi_vector_with(dimensions, mv_config).map_err(|e| {
                Error::new(
                    Status::InvalidArg,
                    format!("Invalid multi-vector config: {}", e),
                )
            })?
        } else {
            store_options.build().map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Failed to create store: {}", e),
                )
            })?
        };

        if let Some(config) = text_search_config.clone() {
            store
                .enable_text_search_with_config(Some(config))
                .map_err(convert_error)?;
        }
        return Ok(VectorDatabase {
            inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
            path,
            is_persistent: false,
            embedding_fn: embedding_tsfn.clone(),
            collections_cache: RwLock::new(HashMap::new()),
        });
    }

    let db_path = std::path::Path::new(&path);
    let omen_path = if db_path.extension().is_some_and(|ext| ext == "omen") {
        db_path.to_path_buf()
    } else {
        let mut omen = db_path.as_os_str().to_os_string();
        omen.push(".omen");
        std::path::PathBuf::from(omen)
    };

    if omen_path.exists() {
        let mut store = VectorStore::open(&path).map_err(convert_error)?;
        if let Some(config) = text_search_config.clone() {
            store
                .enable_text_search_with_config(Some(config))
                .map_err(convert_error)?;
        }
        let is_mv = store.is_multi_vector();

        if is_multi_vector && !is_mv {
            return Err(Error::new(
                Status::InvalidArg,
                "Cannot open existing single-vector database with multiVector: true",
            ));
        }

        return Ok(VectorDatabase {
            inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
            path,
            is_persistent: true,
            embedding_fn: embedding_tsfn.clone(),
            collections_cache: RwLock::new(HashMap::new()),
        });
    }

    if let Some(mv_config) = multi_vector_config {
        let mut store = VectorStore::multi_vector_with(dimensions, mv_config)
            .map_err(|e| {
                Error::new(
                    Status::InvalidArg,
                    format!("Invalid multi-vector config: {}", e),
                )
            })?
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
            embedding_fn: embedding_tsfn.clone(),
            collections_cache: RwLock::new(HashMap::new()),
        });
    }

    if db_path.exists() && quant_mode {
        let existing = VectorStore::open(&path).map_err(convert_error)?;
        if !existing.is_empty() {
            return Err(Error::new(
                Status::InvalidArg,
                "Cannot enable quantization on existing database. Create a new database with quantization.",
            ));
        }
    }

    let mut store = store_options.open(&path).map_err(convert_error)?;
    if let Some(config) = text_search_config {
        store
            .enable_text_search_with_config(Some(config))
            .map_err(convert_error)?;
    }
    Ok(VectorDatabase {
        inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
        path,
        is_persistent: true,
        embedding_fn: embedding_tsfn,
        collections_cache: RwLock::new(HashMap::new()),
    })
}
