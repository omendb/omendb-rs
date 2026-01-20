//! Production stress tests for VectorStore
//!
//! These tests verify correctness under heavy load:
//! - Concurrent access (many readers + writers)
//! - Memory pressure (large vector counts, high dimensions)
//! - Crash recovery (simulated mid-operation failures)
//! - Edge cases (rapid insert/delete cycles, concurrent deletes)
//!
//! Run with: cargo test --lib stress_ --release -- --nocapture

use super::{ThreadSafeVectorStore, VectorStore, VectorStoreOptions};
use crate::vector::types::Vector;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Generate random vector with reproducible seed
fn random_vector(seed: usize, dim: usize) -> Vec<f32> {
    let mut rng_state = seed as u64;
    (0..dim)
        .map(|_| {
            // Simple LCG for reproducibility
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

/// Heavy concurrent writes - 8 writers, 1000 vectors each
#[test]
fn stress_concurrent_writers() {
    let store = ThreadSafeVectorStore::new(128);
    let num_writers = 8;
    let vectors_per_writer = 1000;

    let handles: Vec<_> = (0..num_writers)
        .map(|writer_id| {
            let store = store.clone();
            thread::spawn(move || {
                for i in 0..vectors_per_writer {
                    let id = format!("w{writer_id}_v{i}");
                    let vec = Vector::new(random_vector(writer_id * 10000 + i, 128));
                    store
                        .set(id, vec, serde_json::json!({"writer": writer_id, "idx": i}))
                        .unwrap();
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let expected = num_writers * vectors_per_writer;
    assert_eq!(store.len(), expected, "All vectors should be inserted");

    // Verify all vectors exist
    for writer_id in 0..num_writers {
        for i in 0..vectors_per_writer {
            let id = format!("w{writer_id}_v{i}");
            assert!(store.contains(&id), "Vector {id} should exist");
        }
    }
}

/// Mixed readers and writers - 4 writers + 16 readers, sustained load
#[test]
fn stress_mixed_read_write() {
    let store = ThreadSafeVectorStore::new(128);
    let running = Arc::new(AtomicBool::new(true));
    let write_count = Arc::new(AtomicUsize::new(0));
    let read_count = Arc::new(AtomicUsize::new(0));

    // Seed some data first
    for i in 0..100 {
        store
            .set(
                format!("seed{i}"),
                Vector::new(random_vector(i, 128)),
                serde_json::json!({}),
            )
            .unwrap();
    }

    // Writer threads
    let writer_handles: Vec<_> = (0..4)
        .map(|writer_id| {
            let store = store.clone();
            let running = running.clone();
            let write_count = write_count.clone();
            thread::spawn(move || {
                let mut i = 0;
                while running.load(Ordering::Relaxed) {
                    let id = format!("live_w{writer_id}_v{i}");
                    let vec = Vector::new(random_vector(writer_id * 100000 + i, 128));
                    if store.set(id, vec, serde_json::json!({})).is_ok() {
                        write_count.fetch_add(1, Ordering::Relaxed);
                    }
                    i += 1;
                }
            })
        })
        .collect();

    // Reader threads
    let reader_handles: Vec<_> = (0..16)
        .map(|_| {
            let store = store.clone();
            let running = running.clone();
            let read_count = read_count.clone();
            thread::spawn(move || {
                let mut i = 0;
                while running.load(Ordering::Relaxed) {
                    let query = Vector::new(random_vector(i, 128));
                    if store.search(&query, 10).is_ok() {
                        read_count.fetch_add(1, Ordering::Relaxed);
                    }
                    i += 1;
                }
            })
        })
        .collect();

    // Run for 2 seconds
    thread::sleep(Duration::from_secs(2));
    running.store(false, Ordering::Relaxed);

    for handle in writer_handles {
        handle.join().unwrap();
    }
    for handle in reader_handles {
        handle.join().unwrap();
    }

    let writes = write_count.load(Ordering::Relaxed);
    let reads = read_count.load(Ordering::Relaxed);
    println!("Mixed read/write: {writes} writes, {reads} reads in 2s");
    // Lower threshold for CI machines which can be significantly slower
    assert!(writes > 50, "Should complete many writes");
    assert!(reads > 50, "Should complete many reads");
}

/// Rapid delete/re-insert same IDs - tests tombstone handling
#[test]
fn stress_delete_reinsert_cycle() {
    let store = ThreadSafeVectorStore::new(64);
    let num_ids = 100;
    let cycles = 50;

    // Initial insert
    for i in 0..num_ids {
        store
            .set(
                format!("cycle{i}"),
                Vector::new(random_vector(i, 64)),
                serde_json::json!({"cycle": 0}),
            )
            .unwrap();
    }

    // Delete/reinsert cycles
    for cycle in 1..=cycles {
        for i in 0..num_ids {
            store.delete(&format!("cycle{i}")).unwrap();
        }
        assert_eq!(store.len(), 0, "All should be deleted");

        for i in 0..num_ids {
            store
                .set(
                    format!("cycle{i}"),
                    Vector::new(random_vector(cycle * 1000 + i, 64)),
                    serde_json::json!({"cycle": cycle}),
                )
                .unwrap();
        }
        assert_eq!(store.len(), num_ids, "All should be reinserted");
    }

    // Verify final state
    for i in 0..num_ids {
        let (_, meta) = store.get(&format!("cycle{i}")).unwrap();
        assert_eq!(meta["cycle"].as_u64().unwrap(), cycles as u64);
    }
}

/// Concurrent deletes - multiple threads deleting overlapping ranges
#[test]
fn stress_concurrent_deletes() {
    let store = ThreadSafeVectorStore::new(64);
    let num_vectors = 1000;

    // Insert all vectors
    for i in 0..num_vectors {
        store
            .set(
                format!("del{i}"),
                Vector::new(random_vector(i, 64)),
                serde_json::json!({}),
            )
            .unwrap();
    }

    // Concurrent delete from multiple threads (overlapping ranges)
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let store = store.clone();
            thread::spawn(move || {
                for i in (t * 200)..(t * 200 + 400).min(num_vectors) {
                    let _ = store.delete(&format!("del{i}"));
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all were deleted (ranges 0-400, 200-600, 400-800, 600-1000 = all)
    assert_eq!(store.len(), 0, "All should be deleted");
}

// ============================================================================
// Memory Pressure Tests
// ============================================================================

/// Large vector count - 50K vectors, 128D
#[test]
#[ignore] // Run with --ignored for slow tests
fn stress_large_vector_count() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large_count.omen");

    let mut store = VectorStoreOptions::default()
        .dimensions(128)
        .open(&path)
        .unwrap();

    let num_vectors = 50_000;
    let start = Instant::now();

    // Batch insert for speed
    let batch: Vec<_> = (0..num_vectors)
        .map(|i| {
            (
                format!("v{i}"),
                Vector::new(random_vector(i, 128)),
                serde_json::json!({"idx": i}),
            )
        })
        .collect();

    store.set_batch(batch).unwrap();
    let insert_time = start.elapsed();
    println!("Inserted {num_vectors} vectors in {insert_time:?}");

    // Flush and reopen
    store.flush().unwrap();
    drop(store);

    let store = VectorStore::open(&path).unwrap();
    assert_eq!(store.len(), num_vectors);

    // Search performance
    let start = Instant::now();
    let num_searches = 100;
    for i in 0..num_searches {
        let query = Vector::new(random_vector(i + num_vectors, 128));
        let results = store.search(&query, 10, None).unwrap();
        assert_eq!(results.len(), 10);
    }
    let search_time = start.elapsed();
    println!(
        "{num_searches} searches in {search_time:?} ({:.0} QPS)",
        num_searches as f64 / search_time.as_secs_f64()
    );
}

/// High dimensions - 10K vectors, 768D (typical embedding size)
#[test]
fn stress_high_dimensions() {
    let store = ThreadSafeVectorStore::new(768);
    let num_vectors = 10_000;

    let start = Instant::now();
    let batch: Vec<_> = (0..num_vectors)
        .map(|i| {
            (
                format!("hd{i}"),
                Vector::new(random_vector(i, 768)),
                serde_json::json!({}),
            )
        })
        .collect();

    store.set_batch(batch).unwrap();
    let insert_time = start.elapsed();
    println!("Inserted {num_vectors} 768D vectors in {insert_time:?}");

    assert_eq!(store.len(), num_vectors);

    // Search
    let query = Vector::new(random_vector(0, 768));
    let results = store.search(&query, 10).unwrap();
    assert_eq!(results.len(), 10);
}

// ============================================================================
// Crash Recovery Tests
// ============================================================================

/// Simulated crash mid-batch - verifies WAL recovery
#[test]
fn stress_crash_mid_batch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("crash_batch.omen");

    let checkpoint_count = 500;
    let wal_count = 200;

    // Phase 1: Insert, flush (checkpoint), insert more (WAL only)
    {
        let mut store = VectorStoreOptions::default()
            .dimensions(64)
            .open(&path)
            .unwrap();

        // Checkpointed data
        for i in 0..checkpoint_count {
            store
                .set(
                    format!("cp{i}"),
                    Vector::new(random_vector(i, 64)),
                    serde_json::json!({}),
                )
                .unwrap();
        }
        store.flush().unwrap();

        // WAL-only data (simulates crash before next flush)
        for i in 0..wal_count {
            store
                .set(
                    format!("wal{i}"),
                    Vector::new(random_vector(i + 10000, 64)),
                    serde_json::json!({}),
                )
                .unwrap();
        }
        // Drop without flush - simulates crash
    }

    // Phase 2: Reopen and verify WAL recovery
    {
        let store = VectorStore::open(&path).unwrap();
        assert_eq!(
            store.len(),
            checkpoint_count + wal_count,
            "WAL recovery should restore all data"
        );

        // Verify checkpointed data
        for i in 0..checkpoint_count {
            assert!(store.contains(&format!("cp{i}")), "Checkpoint data missing");
        }

        // Verify WAL data
        for i in 0..wal_count {
            assert!(store.contains(&format!("wal{i}")), "WAL data missing");
        }
    }
}

/// Crash after deletes - verifies delete tombstones in WAL
#[test]
fn stress_crash_after_deletes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("crash_delete.omen");

    let initial_count = 500;
    let delete_count = 200;

    // Phase 1: Insert, flush, delete, crash
    {
        let mut store = VectorStoreOptions::default()
            .dimensions(64)
            .open(&path)
            .unwrap();

        for i in 0..initial_count {
            store
                .set(
                    format!("d{i}"),
                    Vector::new(random_vector(i, 64)),
                    serde_json::json!({}),
                )
                .unwrap();
        }
        store.flush().unwrap();

        // Delete some (goes to WAL)
        for i in 0..delete_count {
            store.delete(&format!("d{i}")).unwrap();
        }
        // Crash without flush
    }

    // Phase 2: Verify deletes survived
    {
        let store = VectorStore::open(&path).unwrap();
        assert_eq!(
            store.len(),
            initial_count - delete_count,
            "Delete tombstones should be recovered"
        );

        // Deleted vectors should not exist
        for i in 0..delete_count {
            assert!(
                !store.contains(&format!("d{i}")),
                "Deleted vector should not exist"
            );
        }

        // Remaining should exist
        for i in delete_count..initial_count {
            assert!(store.contains(&format!("d{i}")), "Vector should exist");
        }
    }
}

/// Multiple crash/recovery cycles
#[test]
fn stress_repeated_crash_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("repeated_crash.omen");

    let vectors_per_cycle = 100;
    let num_cycles = 5;

    for cycle in 0..num_cycles {
        // Open, insert, crash
        {
            let mut store = if cycle == 0 {
                VectorStore::open_with_dimensions(&path, 64).unwrap()
            } else {
                VectorStore::open(&path).unwrap()
            };

            // Verify previous data
            let expected = cycle * vectors_per_cycle;
            assert_eq!(store.len(), expected, "Cycle {cycle}: wrong count");

            // Add more
            for i in 0..vectors_per_cycle {
                let id = format!("c{cycle}_v{i}");
                store
                    .set(
                        id,
                        Vector::new(random_vector(cycle * 1000 + i, 64)),
                        serde_json::json!({}),
                    )
                    .unwrap();
            }
            // Crash (no flush)
        }
    }

    // Final verification
    let store = VectorStore::open(&path).unwrap();
    assert_eq!(store.len(), num_cycles * vectors_per_cycle);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/// Very large metadata
#[test]
fn stress_large_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large_meta.omen");

    let mut store = VectorStoreOptions::default()
        .dimensions(64)
        .open(&path)
        .unwrap();

    // Create large metadata (100KB+)
    let large_text: String = (0..100_000)
        .map(|i| ((i % 26) as u8 + b'a') as char)
        .collect();
    let large_meta = serde_json::json!({
        "large_field": large_text,
        "array": (0..1000).collect::<Vec<_>>(),
    });

    store
        .set(
            "large".to_string(),
            Vector::new(random_vector(0, 64)),
            large_meta.clone(),
        )
        .unwrap();
    store.flush().unwrap();

    // Reopen and verify
    drop(store);
    let store = VectorStore::open(&path).unwrap();
    let (_, meta) = store.get("large").unwrap();
    assert_eq!(meta["large_field"].as_str().unwrap().len(), 100_000);
}

/// Empty and single-element edge cases
#[test]
fn stress_edge_cases() {
    let store = ThreadSafeVectorStore::new(64);

    // Search on empty store
    let query = Vector::new(random_vector(0, 64));
    let results = store.search(&query, 10).unwrap();
    assert!(results.is_empty());

    // Single element
    store
        .set(
            "single".to_string(),
            Vector::new(random_vector(1, 64)),
            serde_json::json!({}),
        )
        .unwrap();

    let results = store.search(&query, 10).unwrap();
    assert_eq!(results.len(), 1);

    // Delete single element
    store.delete("single").unwrap();
    assert!(store.is_empty());

    // Search on now-empty store
    let results = store.search(&query, 10).unwrap();
    assert!(results.is_empty());
}

/// ID collision handling - same ID inserted repeatedly
#[test]
fn stress_id_collision() {
    let store = ThreadSafeVectorStore::new(64);

    // Insert same ID 100 times with different data
    for i in 0..100 {
        store
            .set(
                "collision".to_string(),
                Vector::new(random_vector(i, 64)),
                serde_json::json!({"version": i}),
            )
            .unwrap();
    }

    // Should only have 1 entry (upsert behavior)
    assert_eq!(store.len(), 1);

    // Should have latest version
    let (_, meta) = store.get("collision").unwrap();
    assert_eq!(meta["version"].as_u64().unwrap(), 99);
}
