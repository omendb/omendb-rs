// Benchmark: ALEX learned index vs binary search for SSTable index lookups
// Tests whether learned index provides speedup for finding index blocks

use seerdb::AlexTree;
use std::time::Instant;

/// Convert Bytes to i64 preserving lexicographic ordering
/// Uses first 8 bytes (big-endian) with zero padding for short keys
fn bytes_to_i64(bytes: &[u8]) -> i64 {
    let mut buf = [0u8; 8];
    let len = bytes.len().min(8);
    buf[..len].copy_from_slice(&bytes[..len]);
    i64::from_be_bytes(buf) // Big-endian preserves lexicographic ordering
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ALEX vs Binary Search Benchmark ===\n");

    // Test with different dataset sizes
    let test_sizes = vec![100, 1_000, 10_000];
    let lookups = 10_000;

    for &num_index_blocks in &test_sizes {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "Dataset: {} index blocks, {} lookups",
            num_index_blocks, lookups
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        run_benchmark(num_index_blocks, lookups)?;
        println!();
    }

    Ok(())
}

fn run_benchmark(
    num_index_blocks: usize,
    lookups: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Generate sorted keys (simulating SSTable index block last_keys)
    let keys: Vec<Vec<u8>> = (0..num_index_blocks)
        .map(|i| format!("key_{:010}", i * 100).into_bytes())
        .collect();

    // ========================================
    // Baseline: Binary search on Vec
    // ========================================
    println!("1. Binary Search (Vec)");

    let start = Instant::now();
    for i in 0..lookups {
        let target = format!("key_{:010}", i % (num_index_blocks * 100)).into_bytes();
        let _idx = keys.binary_search(&target).unwrap_or_else(|idx| idx);
    }
    let binary_search_time = start.elapsed();

    let ns_per_lookup_binary = binary_search_time.as_nanos() as f64 / lookups as f64;
    println!("   Total time:       {:?}", binary_search_time);
    println!("   Time per lookup:  {:.1} ns", ns_per_lookup_binary);
    println!(
        "   Throughput:       {:.0} ops/sec\n",
        1_000_000_000.0 / ns_per_lookup_binary
    );

    // ========================================
    // ALEX learned index
    // ========================================
    println!("2. ALEX Learned Index");

    // Build ALEX index
    let build_start = Instant::now();
    let mut alex = AlexTree::new();
    for (i, key) in keys.iter().enumerate() {
        let key_i64 = bytes_to_i64(key);
        let value = i.to_le_bytes().to_vec(); // Store index as value
        alex.insert(key_i64, value)?;
    }
    let build_time = build_start.elapsed();
    println!("   Build time:       {:?}", build_time);

    // Lookup benchmark
    let start = Instant::now();
    for i in 0..lookups {
        let target = format!("key_{:010}", i % (num_index_blocks * 100)).into_bytes();
        let target_i64 = bytes_to_i64(&target);
        let _ = alex.get(target_i64)?;
    }
    let alex_time = start.elapsed();

    let ns_per_lookup_alex = alex_time.as_nanos() as f64 / lookups as f64;
    println!("   Total time:       {:?}", alex_time);
    println!("   Time per lookup:  {:.1} ns", ns_per_lookup_alex);
    println!(
        "   Throughput:       {:.0} ops/sec\n",
        1_000_000_000.0 / ns_per_lookup_alex
    );

    // ========================================
    // Comparison
    // ========================================
    println!("=== Results ===");
    println!("Binary search:    {:.1} ns/lookup", ns_per_lookup_binary);
    println!("ALEX:             {:.1} ns/lookup", ns_per_lookup_alex);

    if alex_time < binary_search_time {
        let speedup = binary_search_time.as_nanos() as f64 / alex_time.as_nanos() as f64;
        println!("ALEX speedup:     {:.2}x faster", speedup);
        println!("\n✅ ALEX wins! Consider integrating into SSTable.");
    } else {
        let slowdown = alex_time.as_nanos() as f64 / binary_search_time.as_nanos() as f64;
        println!("ALEX slowdown:    {:.2}x slower", slowdown);
        println!("\n❌ Binary search faster. Skip ALEX integration.");
    }

    // Memory footprint estimate
    println!("\nMemory footprint:");
    let binary_mem = num_index_blocks * 32;
    let alex_mem = alex.num_leaves() * 200;
    println!("Binary search:    ~{} bytes (Vec<Entry>)", binary_mem);
    println!("ALEX:             ~{} bytes (estimated)", alex_mem);
    let mem_reduction = 1.0 - (alex_mem as f64 / binary_mem as f64);
    println!("Memory reduction: {:.0}%", mem_reduction * 100.0);

    Ok(())
}
