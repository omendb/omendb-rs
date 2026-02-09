use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use omendb::omen::OmenFile;
use omendb::vector::QuantizationMode;
use omendb::{MetadataFilter, Vector, VectorStore};
use serde_json::json;

#[derive(Parser)]
#[command(name = "omendb", version, about = "OmenDB vector database CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show database metadata
    Info {
        path: PathBuf,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Detailed statistics and memory usage
    Stats { path: PathBuf },
    /// List collections
    Collections { path: PathBuf },
    /// Search by vector
    Search {
        path: PathBuf,
        /// Comma-separated query vector
        #[arg(short = 'v', long)]
        vector: Option<String>,
        /// Read query vector from JSON file
        #[arg(long)]
        file: Option<PathBuf>,
        /// Number of results
        #[arg(short, default_value = "10")]
        k: usize,
        /// ef_search override
        #[arg(long)]
        ef: Option<usize>,
        /// Metadata filter (JSON)
        #[arg(long)]
        filter: Option<String>,
        /// Maximum distance threshold
        #[arg(long)]
        max_distance: Option<f32>,
        /// Search within a collection
        #[arg(long)]
        collection: Option<String>,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Get vector by ID
    Get {
        path: PathBuf,
        /// Vector ID to retrieve
        id: String,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// List vector IDs
    Ids {
        path: PathBuf,
        /// Only print count
        #[arg(long)]
        count: bool,
        /// List IDs from a collection
        #[arg(long)]
        collection: Option<String>,
    },
    /// Count vectors
    Count {
        path: PathBuf,
        /// Count only vectors matching filter (JSON)
        #[arg(long)]
        filter: Option<String>,
        /// Count within a collection
        #[arg(long)]
        collection: Option<String>,
    },
    /// Benchmark search performance
    Bench {
        path: PathBuf,
        /// Number of queries per run
        #[arg(long, default_value = "100")]
        queries: usize,
        /// Number of runs
        #[arg(long, default_value = "3")]
        runs: usize,
        /// ef_search parameter
        #[arg(long, default_value = "100")]
        ef: usize,
        /// Number of results per query
        #[arg(short, default_value = "10")]
        k: usize,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Export vectors to JSON/JSONL
    Export {
        path: PathBuf,
        /// Output format: json or jsonl
        #[arg(long, default_value = "jsonl")]
        format: String,
        /// Export from a collection
        #[arg(long)]
        collection: Option<String>,
        /// Only export IDs (no vectors)
        #[arg(long)]
        ids_only: bool,
    },
    /// Compact database (remove tombstones)
    Compact { path: PathBuf },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Info { path, json } => cmd_info(&path, json),
        Commands::Stats { path } => cmd_stats(&path),
        Commands::Collections { path } => cmd_collections(&path),
        Commands::Search {
            path,
            vector,
            file,
            k,
            ef,
            filter,
            max_distance,
            collection,
            json,
        } => cmd_search(
            &path,
            vector,
            file,
            k,
            ef,
            filter,
            max_distance,
            collection,
            json,
        ),
        Commands::Get { path, id, json } => cmd_get(&path, &id, json),
        Commands::Ids {
            path,
            count,
            collection,
        } => cmd_ids(&path, count, collection),
        Commands::Count {
            path,
            filter,
            collection,
        } => cmd_count(&path, filter, collection),
        Commands::Bench {
            path,
            queries,
            runs,
            ef,
            k,
            json,
        } => cmd_bench(&path, queries, runs, ef, k, json),
        Commands::Export {
            path,
            format,
            collection,
            ids_only,
        } => cmd_export(&path, &format, collection, ids_only),
        Commands::Compact { path } => cmd_compact(&path),
    }
}

fn open_store(path: &Path, collection: Option<&str>) -> Result<VectorStore> {
    let store_path = if let Some(col) = collection {
        path.join("collections").join(col)
    } else {
        path.to_path_buf()
    };
    VectorStore::open(&store_path)
        .with_context(|| format!("Failed to open database at {}", store_path.display()))
}

fn metric_name(store: &VectorStore) -> &'static str {
    match store.metric() {
        omendb::omen::Metric::L2 => "L2",
        omendb::omen::Metric::Cosine => "Cosine",
        omendb::omen::Metric::InnerProduct => "InnerProduct",
    }
}

fn quant_name(store: &VectorStore) -> &'static str {
    match store.quantization_mode() {
        Some(QuantizationMode::SQ8) => "SQ8",
        Some(QuantizationMode::RaBitQ) => "RaBitQ",
        None => "None",
    }
}

fn file_size(path: &Path) -> Result<u64> {
    let omen_path = OmenFile::compute_omen_path(path);
    let mut total = 0u64;
    if omen_path.exists() {
        total += fs::metadata(&omen_path)?.len();
    }
    // WAL file
    let wal_path = path.with_extension("wal");
    if wal_path.exists() {
        total += fs::metadata(&wal_path)?.len();
    }
    // Segments directory
    let seg_dir = PathBuf::from(format!("{}.segments", omen_path.display()));
    if seg_dir.exists() {
        for entry in fs::read_dir(&seg_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                total += entry.metadata()?.len();
            }
        }
    }
    Ok(total)
}

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

fn cmd_info(path: &Path, json_output: bool) -> Result<()> {
    let store = open_store(path, None)?;
    let size = file_size(path)?;
    let collections = list_collections(path);

    if json_output {
        let info = json!({
            "version": env!("CARGO_PKG_VERSION"),
            "path": path.display().to_string(),
            "size_bytes": size,
            "vectors": store.len(),
            "dimensions": store.dimensions(),
            "metric": metric_name(&store),
            "quantization": quant_name(&store),
            "hnsw_m": store.hnsw_m(),
            "hnsw_ef_construction": store.hnsw_ef_construction(),
            "hnsw_ef_search": store.ef_search(),
            "collections": collections.len(),
            "text_search": store.has_text_search(),
        });
        println!("{}", serde_json::to_string_pretty(&info)?);
    } else {
        println!("OmenDB v{}", env!("CARGO_PKG_VERSION"));
        println!(
            "Path:           {} ({})",
            path.display(),
            format_bytes(size)
        );
        println!("Vectors:        {}", store.len());
        println!("Dimensions:     {}", store.dimensions());
        println!("Metric:         {}", metric_name(&store));
        println!("Quantization:   {}", quant_name(&store));
        println!(
            "HNSW:           M={}, ef_construction={}, ef_search={}",
            store.hnsw_m(),
            store.hnsw_ef_construction(),
            store.ef_search()
        );
        println!("Collections:    {}", collections.len());
        println!(
            "Text search:    {}",
            if store.has_text_search() {
                "enabled"
            } else {
                "disabled"
            }
        );
    }
    Ok(())
}

fn cmd_stats(path: &Path) -> Result<()> {
    let store = open_store(path, None)?;
    let size = file_size(path)?;
    let vec_data = store.memory_usage();
    let deleted = store.deleted_count();

    println!(
        "Vectors:        {} ({}D, f32)",
        store.len(),
        store.dimensions()
    );
    println!("Deleted:        {} tombstones", deleted);
    println!("File size:      {}", format_bytes(size));
    println!(
        "Vector data:    {} ({:.0} bytes/vec)",
        format_bytes(vec_data as u64),
        store.bytes_per_vector()
    );

    if let Some(ref segments) = store.segments {
        let frozen = segments.frozen_count();
        let mutable_len = segments.mutable_len();
        println!("Segments:       {} frozen + 1 mutable", frozen);
        for (i, seg) in segments.frozen_segments().iter().enumerate() {
            println!("  segment_{}:    {} vectors (frozen)", i, seg.len());
        }
        println!("  mutable:      {} vectors", mutable_len);
    } else {
        println!("Segments:       (no index built)");
    }

    Ok(())
}

fn list_collections(path: &Path) -> Vec<String> {
    let collections_dir = path.join("collections");
    if !collections_dir.exists() {
        return Vec::new();
    }
    let Ok(entries) = fs::read_dir(&collections_dir) else {
        return Vec::new();
    };
    let mut names = Vec::new();
    for entry in entries.flatten() {
        if entry.file_type().is_ok_and(|ft| ft.is_file()) {
            if let Some(name) = entry.file_name().to_str() {
                if let Some(col_name) = name.strip_suffix(".omen") {
                    names.push(col_name.to_string());
                }
            }
        }
    }
    names.sort();
    names
}

fn cmd_collections(path: &Path) -> Result<()> {
    let names = list_collections(path);
    if names.is_empty() {
        println!("No collections");
        return Ok(());
    }
    for name in &names {
        let col_store = open_store(path, Some(name))?;
        println!("{:<20} {} vectors", name, col_store.len());
    }
    Ok(())
}

fn parse_vector(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|v| {
            v.trim()
                .parse::<f32>()
                .with_context(|| format!("Invalid float: '{}'", v.trim()))
        })
        .collect()
}

fn parse_vector_file(path: &Path) -> Result<Vec<f32>> {
    let content =
        fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&content)?;
    match value {
        serde_json::Value::Array(arr) => arr
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|f| f as f32)
                    .with_context(|| format!("Expected number, got: {v}"))
            })
            .collect(),
        _ => bail!("Expected JSON array of floats"),
    }
}

fn parse_filter(s: &str) -> Result<MetadataFilter> {
    let value: serde_json::Value = serde_json::from_str(s).context("Invalid JSON filter")?;
    MetadataFilter::from_json(&value).map_err(|e| anyhow::anyhow!("{e}"))
}

#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
fn cmd_search(
    path: &Path,
    vector_str: Option<String>,
    file: Option<PathBuf>,
    k: usize,
    ef: Option<usize>,
    filter_str: Option<String>,
    max_distance: Option<f32>,
    collection: Option<String>,
    json_output: bool,
) -> Result<()> {
    let query_data = if let Some(ref v) = vector_str {
        parse_vector(v)?
    } else if let Some(ref f) = file {
        parse_vector_file(f)?
    } else {
        bail!("Provide a query vector with -v or --file");
    };

    let store = open_store(path, collection.as_deref())?;
    let query = Vector::new(query_data);
    let filter = filter_str.as_deref().map(parse_filter).transpose()?;

    let results = store.search_with_options(&query, k, filter.as_ref(), ef, max_distance)?;

    if json_output {
        let arr: Vec<_> = results
            .iter()
            .map(|r| {
                json!({
                    "id": r.id,
                    "distance": r.distance,
                    "metadata": r.metadata,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
    } else {
        println!("{:<20} {:>10}  Metadata", "ID", "Distance");
        for r in &results {
            let meta = if r.metadata.is_null() {
                String::new()
            } else {
                serde_json::to_string(&r.metadata)?
            };
            println!("{:<20} {:>10.4}  {}", r.id, r.distance, meta);
        }
    }
    Ok(())
}

fn cmd_get(path: &Path, id: &str, json_output: bool) -> Result<()> {
    let store = open_store(path, None)?;
    let Some((vector, metadata)) = store.get(id) else {
        bail!("Vector '{id}' not found");
    };

    if json_output {
        let dims = vector.data.len();
        let obj = json!({
            "id": id,
            "vector": vector.data,
            "dimensions": dims,
            "metadata": metadata,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        let dims = vector.data.len();
        let preview: Vec<String> = vector
            .data
            .iter()
            .take(5)
            .map(|f| format!("{f:.3}"))
            .collect();
        let suffix = if dims > 5 { ", ..." } else { "" };
        println!("ID:       {id}");
        println!("Vector:   [{}{}] ({}D)", preview.join(", "), suffix, dims);
        if !metadata.is_null() {
            println!("Metadata: {}", serde_json::to_string(&metadata)?);
        }
    }
    Ok(())
}

fn cmd_ids(path: &Path, count_only: bool, collection: Option<String>) -> Result<()> {
    let store = open_store(path, collection.as_deref())?;
    if count_only {
        println!("{}", store.len());
    } else {
        for id in store.ids() {
            println!("{id}");
        }
    }
    Ok(())
}

fn cmd_count(path: &Path, filter_str: Option<String>, collection: Option<String>) -> Result<()> {
    let store = open_store(path, collection.as_deref())?;
    if let Some(ref f) = filter_str {
        let filter = parse_filter(f)?;
        println!("{}", store.count_by_filter(&filter));
    } else {
        println!("{}", store.len());
    }
    Ok(())
}

fn cmd_bench(
    path: &Path,
    queries: usize,
    runs: usize,
    ef: usize,
    k: usize,
    json_output: bool,
) -> Result<()> {
    let store = open_store(path, None)?;
    let count = store.len();
    let dims = store.dimensions();

    if count == 0 {
        bail!("Database is empty, nothing to benchmark");
    }

    // Sample query vectors from the database
    let items = store.items();
    let num_queries = queries.min(count);
    let query_vecs: Vec<Vector> = items
        .iter()
        .take(num_queries)
        .map(|(_, v, _)| Vector::new(v.clone()))
        .collect();

    println!("OmenDB Benchmark ({} vectors, {}D)", count, dims);
    println!();

    // Warm up
    for q in query_vecs.iter().take(10.min(num_queries)) {
        let _ = store.search_with_ef(q, k, None, Some(ef));
    }

    let mut run_results = Vec::new();

    for run in 0..runs {
        let start = Instant::now();
        for q in &query_vecs {
            let _ = store.search_with_ef(q, k, None, Some(ef));
        }
        let elapsed = start.elapsed();
        let qps = query_vecs.len() as f64 / elapsed.as_secs_f64();
        run_results.push(qps);

        if !json_output {
            println!(
                "Run {}: {:.0} QPS ({} queries in {:.1}ms)",
                run + 1,
                qps,
                query_vecs.len(),
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }

    let avg_qps = run_results.iter().sum::<f64>() / run_results.len() as f64;

    if json_output {
        let obj = json!({
            "vectors": count,
            "dimensions": dims,
            "k": k,
            "ef": ef,
            "queries": num_queries,
            "runs": runs,
            "qps_per_run": run_results,
            "avg_qps": avg_qps,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!();
        println!("Average: {:.0} QPS (k={}, ef={})", avg_qps, k, ef);
    }

    Ok(())
}

fn cmd_export(path: &Path, format: &str, collection: Option<String>, ids_only: bool) -> Result<()> {
    let store = open_store(path, collection.as_deref())?;

    if ids_only {
        for id in store.ids() {
            println!("{id}");
        }
        return Ok(());
    }

    match format {
        "jsonl" => {
            for (id, vector, metadata) in store.items() {
                let obj = json!({
                    "id": id,
                    "vector": vector,
                    "metadata": metadata,
                });
                println!("{}", serde_json::to_string(&obj)?);
            }
        }
        "json" => {
            let items: Vec<_> = store
                .items()
                .into_iter()
                .map(|(id, vector, metadata)| {
                    json!({
                        "id": id,
                        "vector": vector,
                        "metadata": metadata,
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&items)?);
        }
        _ => bail!("Unknown format: '{}'. Use 'json' or 'jsonl'.", format),
    }
    Ok(())
}

fn cmd_compact(path: &Path) -> Result<()> {
    let mut store = open_store(path, None)?;
    let removed = store.compact()?;
    if removed > 0 {
        store.flush()?;
        println!("Removed {} tombstones", removed);
    } else {
        println!("No tombstones to remove");
    }
    Ok(())
}
