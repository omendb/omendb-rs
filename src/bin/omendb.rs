use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

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
    /// Show database info and statistics
    Info {
        path: PathBuf,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Search by vector
    Search {
        path: PathBuf,
        /// Comma-separated query vector
        #[arg(short = 'q', long)]
        query: Option<String>,
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
        /// Get from a collection
        #[arg(long)]
        collection: Option<String>,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Export vectors to JSON/JSONL
    Export {
        path: PathBuf,
        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output format: json or jsonl
        #[arg(long, default_value = "jsonl")]
        format: String,
        /// Export from a collection
        #[arg(long)]
        collection: Option<String>,
    },
    /// Compact database (remove tombstones)
    Compact { path: PathBuf },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Info { path, json } => cmd_info(&path, json),
        Commands::Search {
            path,
            query,
            file,
            k,
            ef,
            filter,
            max_distance,
            collection,
            json,
        } => cmd_search(
            &path,
            query,
            file,
            k,
            ef,
            filter,
            max_distance,
            collection,
            json,
        ),
        Commands::Get {
            path,
            id,
            collection,
            json,
        } => cmd_get(&path, &id, collection.as_deref(), json),
        Commands::Export {
            path,
            output,
            format,
            collection,
        } => cmd_export(&path, output, &format, collection),
        Commands::Compact { path } => cmd_compact(&path),
    }
}

fn resolve_store_path(path: &Path, collection: Option<&str>) -> Result<PathBuf> {
    let store_path = if let Some(col) = collection {
        if col.is_empty() || col.contains('/') || col.contains('\\') || col.contains("..") {
            bail!("Invalid collection name: '{col}'");
        }
        path.join("collections").join(col)
    } else {
        path.to_path_buf()
    };
    Ok(store_path)
}

fn require_exists(path: &Path) -> Result<()> {
    let omen_path = OmenFile::compute_omen_path(path);
    if !omen_path.exists() {
        bail!("Database not found at {}", path.display());
    }
    Ok(())
}

fn open_store(path: &Path, collection: Option<&str>) -> Result<VectorStore> {
    let store_path = resolve_store_path(path, collection)?;
    require_exists(&store_path)?;
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

fn append_ext(path: &Path, ext: &str) -> PathBuf {
    let mut s: OsString = path.as_os_str().into();
    s.push(ext);
    PathBuf::from(s)
}

fn file_size(path: &Path) -> Result<u64> {
    let omen_path = OmenFile::compute_omen_path(path);
    let mut total = 0u64;
    if omen_path.exists() {
        total += fs::metadata(&omen_path)?.len();
    }
    let wal_path = append_ext(path, ".wal");
    if wal_path.exists() {
        total += fs::metadata(&wal_path)?.len();
    }
    let seg_dir = append_ext(path, ".segments");
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

fn cmd_info(path: &Path, json_output: bool) -> Result<()> {
    let store = open_store(path, None)?;
    let size = file_size(path)?;
    let collections = list_collections(path);
    let deleted = store.deleted_count();

    if json_output {
        let info = json!({
            "version": env!("CARGO_PKG_VERSION"),
            "path": path.display().to_string(),
            "size_bytes": size,
            "vectors": store.len(),
            "deleted": deleted,
            "dimensions": store.dimensions(),
            "metric": metric_name(&store),
            "quantization": quant_name(&store),
            "hnsw_m": store.hnsw_m(),
            "hnsw_ef_construction": store.hnsw_ef_construction(),
            "hnsw_ef_search": store.ef_search(),
            "collections": collections,
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
        if deleted > 0 {
            println!("Deleted:        {} tombstones", deleted);
        }
        println!("Dimensions:     {}", store.dimensions());
        println!("Metric:         {}", metric_name(&store));
        println!("Quantization:   {}", quant_name(&store));
        println!(
            "HNSW:           M={}, ef_construction={}, ef_search={}",
            store.hnsw_m(),
            store.hnsw_ef_construction(),
            store.ef_search()
        );
        if !collections.is_empty() {
            println!(
                "Collections:    {} ({})",
                collections.len(),
                collections.join(", ")
            );
        }
        if store.has_text_search() {
            println!("Text search:    enabled");
        }
    }
    Ok(())
}

fn parse_vector(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|v| {
            let f = v
                .trim()
                .parse::<f32>()
                .with_context(|| format!("Invalid float: '{}'", v.trim()))?;
            if !f.is_finite() {
                bail!("Non-finite value: '{}'", v.trim());
            }
            Ok(f)
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
                let f = v
                    .as_f64()
                    .with_context(|| format!("Expected number, got: {v}"))?
                    as f32;
                if !f.is_finite() {
                    bail!("Non-finite value in vector file");
                }
                Ok(f)
            })
            .collect(),
        _ => bail!("Expected JSON array of floats"),
    }
}

fn parse_filter(s: &str) -> Result<MetadataFilter> {
    let value: serde_json::Value = serde_json::from_str(s).context("Invalid JSON filter")?;
    MetadataFilter::from_json(&value).map_err(|e| anyhow::anyhow!("{e}"))
}

fn cmd_search(
    path: &Path,
    query_str: Option<String>,
    file: Option<PathBuf>,
    k: usize,
    ef: Option<usize>,
    filter_str: Option<String>,
    max_distance: Option<f32>,
    collection: Option<String>,
    json_output: bool,
) -> Result<()> {
    let query_data = if let Some(ref v) = query_str {
        parse_vector(v)?
    } else if let Some(ref f) = file {
        parse_vector_file(f)?
    } else {
        bail!("Provide a query vector with -q or --file");
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

fn cmd_get(path: &Path, id: &str, collection: Option<&str>, json_output: bool) -> Result<()> {
    let store = open_store(path, collection)?;
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

fn cmd_export(
    path: &Path,
    output: Option<PathBuf>,
    format: &str,
    collection: Option<String>,
) -> Result<()> {
    use std::io::{BufWriter, Write};

    let store = open_store(path, collection.as_deref())?;

    let mut writer: Box<dyn Write> = if let Some(ref out_path) = output {
        Box::new(BufWriter::new(fs::File::create(out_path).with_context(
            || format!("Failed to create {}", out_path.display()),
        )?))
    } else {
        Box::new(BufWriter::new(std::io::stdout().lock()))
    };

    match format {
        "jsonl" => {
            for (id, vector, metadata) in store.items() {
                let obj = json!({
                    "id": id,
                    "vector": vector,
                    "metadata": metadata,
                });
                serde_json::to_writer(&mut writer, &obj)?;
                writeln!(writer)?;
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
            serde_json::to_writer_pretty(&mut writer, &items)?;
            writeln!(writer)?;
        }
        _ => bail!("Unknown format: '{}'. Use 'json' or 'jsonl'.", format),
    }

    if let Some(ref out_path) = output {
        eprintln!("Exported {} vectors to {}", store.len(), out_path.display());
    }
    Ok(())
}

fn cmd_compact(path: &Path) -> Result<()> {
    let store_path = resolve_store_path(path, None)?;
    require_exists(&store_path)?;
    let mut store = VectorStore::open(&store_path)
        .with_context(|| format!("Failed to open database at {}", store_path.display()))?;
    let removed = store.compact()?;
    if removed > 0 {
        println!("Removed {} tombstones", removed);
    } else {
        println!("No tombstones to remove");
    }
    Ok(())
}
