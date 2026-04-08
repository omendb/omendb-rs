use napi::bindgen_prelude::*;
use omendb_lib::omen::Metric;
use omendb_lib::text::{TextSearchConfig, TokenizerPreset};
use omendb_lib::vector::muvera::MultiVectorConfig;

/// Convert raw distance to a normalized similarity score (0-1, higher = more similar).
pub(crate) fn distance_to_score(distance: f64, metric: Metric) -> f64 {
    match metric {
        Metric::L2 => 1.0 / (1.0 + distance),
        Metric::Cosine => 1.0 - distance,
        Metric::InnerProduct => -distance, // IP distance is -dot, so score is dot product
    }
}

/// Extract query vector from JS - accepts number[] or Float32Array
pub(crate) fn extract_query_vector(query: Either<Vec<f64>, Float32Array>) -> Vec<f32> {
    match query {
        Either::A(arr) => arr.into_iter().map(|x| x as f32).collect(),
        Either::B(typed) => typed.to_vec(),
    }
}

/// Extract multi-vector query from JS - accepts number[][] or Float32Array[]
pub(crate) fn extract_multi_vector_query(
    query: Either<Vec<Vec<f64>>, Vec<Float32Array>>,
) -> Result<Vec<Vec<f32>>> {
    match query {
        Either::A(nested) => {
            if nested.is_empty() {
                return Err(Error::new(
                    Status::InvalidArg,
                    "multi-vector query must not be empty",
                ));
            }
            Ok(nested
                .into_iter()
                .map(|arr| arr.into_iter().map(|x| x as f32).collect())
                .collect())
        }
        Either::B(typed_arrays) => {
            if typed_arrays.is_empty() {
                return Err(Error::new(
                    Status::InvalidArg,
                    "multi-vector query must not be empty",
                ));
            }
            Ok(typed_arrays.into_iter().map(|t| t.to_vec()).collect())
        }
    }
}

/// Parse multi_vector option from JS value
pub(crate) fn parse_multi_vector(value: &serde_json::Value) -> Result<Option<MultiVectorConfig>> {
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Bool(true) => Ok(Some(MultiVectorConfig::default())),
        serde_json::Value::Bool(false) => Ok(None),
        serde_json::Value::Object(obj) => {
            let mut config = MultiVectorConfig::default();
            if let Some(reps) = obj.get("repetitions") {
                config.repetitions = reps
                    .as_u64()
                    .ok_or_else(|| Error::new(Status::InvalidArg, "repetitions must be a number"))?
                    as u8;
            }
            if let Some(bits) = obj.get("partitionBits") {
                config.partition_bits = bits.as_u64().ok_or_else(|| {
                    Error::new(Status::InvalidArg, "partitionBits must be a number")
                })? as u8;
            }
            if let Some(seed) = obj.get("seed") {
                config.seed = seed
                    .as_u64()
                    .ok_or_else(|| Error::new(Status::InvalidArg, "seed must be a number"))?;
            }
            if let Some(d_proj) = obj.get("dProj") {
                config.d_proj = if d_proj.is_null() {
                    None
                } else {
                    Some(d_proj.as_u64().ok_or_else(|| {
                        Error::new(Status::InvalidArg, "dProj must be a number or null")
                    })? as u8)
                };
            }
            if let Some(pool_factor) = obj.get("poolFactor") {
                config.pool_factor = if pool_factor.is_null() {
                    None
                } else {
                    Some(pool_factor.as_u64().ok_or_else(|| {
                        Error::new(Status::InvalidArg, "poolFactor must be a number or null")
                    })? as u8)
                };
            }
            Ok(Some(config))
        }
        _ => Err(Error::new(
            Status::InvalidArg,
            "multiVector must be true, false, or { repetitions?, partitionBits?, seed?, dProj?, poolFactor? }",
        )),
    }
}

/// Parse quantization option from JS value (bool or string) -> bool
pub(crate) fn parse_quantization(value: &serde_json::Value) -> Result<bool> {
    match value {
        serde_json::Value::Null => Ok(false),
        serde_json::Value::Bool(b) => Ok(*b),
        serde_json::Value::String(s) => {
            let lower = s.to_lowercase();
            match lower.as_str() {
                "sq8" | "scalar" => Ok(true),
                _ => Err(Error::new(
                    Status::InvalidArg,
                    format!("Unknown quantization mode: '{}'. Valid: true, 'sq8'", s),
                )),
            }
        }
        _ => Err(Error::new(
            Status::InvalidArg,
            "quantization must be true, false, or 'sq8'",
        )),
    }
}

/// Parse text search option from JS value.
pub(crate) fn parse_text_search_config(
    value: &serde_json::Value,
) -> Result<Option<TextSearchConfig>> {
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Bool(true) => Ok(Some(TextSearchConfig::default())),
        serde_json::Value::Bool(false) => Ok(None),
        serde_json::Value::Object(obj) => {
            let mut config = TextSearchConfig::default();
            if let Some(buffer_mb) = obj.get("bufferMb").or_else(|| obj.get("writerBufferMb")) {
                config.writer_buffer_mb = buffer_mb.as_u64().ok_or_else(|| {
                    Error::new(Status::InvalidArg, "textSearch.bufferMb must be a number")
                })? as usize;
            }
            if let Some(tokenizer) = obj.get("tokenizer") {
                let tokenizer_name = tokenizer.as_str().ok_or_else(|| {
                    Error::new(Status::InvalidArg, "textSearch.tokenizer must be a string")
                })?;
                config.tokenizer = TokenizerPreset::parse(tokenizer_name).map_err(convert_error)?;
            }
            Ok(Some(config))
        }
        _ => Err(Error::new(
            Status::InvalidArg,
            "textSearch must be true, false, or { bufferMb?, tokenizer? }",
        )),
    }
}

/// Convert Rust error to napi Error with appropriate status
pub(crate) fn convert_error(err: anyhow::Error) -> Error {
    let msg = err.to_string();
    if msg.contains("dimension")
        || msg.contains("k=0")
        || msg.contains("Requirement: k > 0")
        || msg.contains("zero vector")
        || msg.contains("NaN or Infinity")
    {
        Error::new(Status::InvalidArg, msg)
    } else {
        Error::new(Status::GenericFailure, msg)
    }
}
