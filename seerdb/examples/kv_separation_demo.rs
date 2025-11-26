// Demonstration of KV separation (WiscKey-style)
// Shows how large values are stored separately from keys

use bytes::Bytes;
use seerdb::{SSTableBuilder, VLog};
use tempfile::tempdir;

fn main() {
    println!("=== KV Separation Demo (WiscKey) ===\n");

    let dir = tempdir().unwrap();
    let sstable_path = dir.path().join("demo.sst");
    let vlog_path = dir.path().join("demo.vlog");

    // Create vLog
    let mut vlog = VLog::create(&vlog_path).unwrap();

    // Build SSTable with 4KB threshold (like embeddings)
    let mut builder = SSTableBuilder::create(&sstable_path)
        .unwrap()
        .with_vlog_threshold(4096);

    println!("Adding entries:");

    // Small values (stored inline in SSTable)
    println!("  key1: small_value (11 bytes) → inline");
    builder
        .add_with_vlog(Bytes::from("key1"), Bytes::from("small_value"), &mut vlog)
        .unwrap();

    // Large values (stored in vLog)
    let large_value = vec![b'X'; 8192]; // 8KB
    println!("  key2: [8KB value] → vLog");
    builder
        .add_with_vlog(Bytes::from("key2"), Bytes::from(large_value), &mut vlog)
        .unwrap();

    let embedding = vec![0.1f32; 1024]; // 4KB (1024 floats * 4 bytes)
    let embedding_bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
    println!("  key3: [4KB embedding] → vLog");
    builder
        .add_with_vlog(Bytes::from("key3"), Bytes::from(embedding_bytes), &mut vlog)
        .unwrap();

    println!();

    // Finish writing SSTable
    builder.finish().unwrap();

    // Open SSTable for reading
    let mut sstable = seerdb::SSTable::open(&sstable_path).unwrap();

    // Check file sizes
    let sstable_size = std::fs::metadata(&sstable_path).unwrap().len();
    let vlog_size = std::fs::metadata(&vlog_path).unwrap().len();

    println!("File sizes:");
    println!(
        "  SSTable: {} bytes (keys + small values + pointers)",
        sstable_size
    );
    println!("  VLog:    {} bytes (large values)", vlog_size);
    println!("  Total:   {} bytes", sstable_size + vlog_size);
    println!();

    println!("Write amplification benefit:");
    println!(
        "  Traditional LSM: Compaction rewrites {} bytes",
        sstable_size + vlog_size
    );
    println!(
        "  WiscKey:         Compaction rewrites {} bytes (just SSTable)",
        sstable_size
    );
    let reduction = ((vlog_size as f64 / (sstable_size + vlog_size) as f64) * 100.0) as u32;
    println!("  Reduction:       {}%", reduction);
    println!();

    // Read small value (no vLog needed)
    println!("Reading small value (inline):");
    let value = sstable.get(b"key1").unwrap().unwrap();
    println!("  key1 = {}", String::from_utf8_lossy(&value));
    println!();

    // Read large value (requires vLog)
    println!("Reading large value (from vLog):");
    println!("  key2 without vLog: {:?}", sstable.get(b"key2"));
    println!("  ^ Error: Value pointer found but no vLog attached");
    println!();

    // Attach vLog
    let vlog = VLog::open(&vlog_path).unwrap();
    let mut sstable = seerdb::SSTable::open(&sstable_path)
        .unwrap()
        .with_vlog(vlog);

    println!(
        "  key2 with vLog: OK ({} bytes)",
        sstable.get(b"key2").unwrap().unwrap().len()
    );
    println!(
        "  key3 with vLog: OK ({} bytes)",
        sstable.get(b"key3").unwrap().unwrap().len()
    );
    println!();

    println!("✅ KV Separation working!");
    println!();
    println!("Benefits:");
    println!(
        "  - Compaction only rewrites keys ({}% less data)",
        reduction
    );
    println!("  - Large values never moved (10-100x write amp reduction)");
    println!("  - Small values stay inline (no random read penalty)");
}
