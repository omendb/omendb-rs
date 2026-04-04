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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
                        .set(&id, vec, serde_json::json!({"writer": writer_id, "idx": i}))
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
                &format!("seed{i}"),
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
                    if store.set(&id, vec, serde_json::json!({})).is_ok() {
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
                &format!("cycle{i}"),
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
                    &format!("cycle{i}"),
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
                &format!("del{i}"),
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

/// Large vector count - 50K vectors, 128D
#[test]
#[ignore] // Run with --ignored for slow tests
fn stress_large_vector_count() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large_count.omen");

    let store = VectorStoreOptions::default()
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

/// Simulated crash mid-batch - verifies WAL recovery
#[test]
fn stress_crash_mid_batch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("crash_batch.omen");

    let checkpoint_count = 500;
    let wal_count = 200;

    // Phase 1: Insert, flush (checkpoint), insert more (WAL only)
    {
        let store = VectorStoreOptions::default()
            .dimensions(64)
            .open(&path)
            .unwrap();

        // Checkpointed data
        for i in 0..checkpoint_count {
            store
                .set(
                    &format!("cp{i}"),
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
                    &format!("wal{i}"),
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
        let store = VectorStoreOptions::default()
            .dimensions(64)
            .open(&path)
            .unwrap();

        for i in 0..initial_count {
            store
                .set(
                    &format!("d{i}"),
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
            let store = if cycle == 0 {
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
                        &id,
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

/// Very large metadata
#[test]
fn stress_large_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large_meta.omen");

    let store = VectorStoreOptions::default()
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
            "large",
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
            "single",
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
                "collision",
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

// ============================================================
// Crash recovery tests for vector-only checkpoint path
// ============================================================

/// Recovery after vector-only auto-checkpoint.
///
/// Flushes a batch of 9K vectors (durable via full checkpoint), then inserts
/// 1.5K more via individual set() calls which are WAL'd and trigger an
/// auto-checkpoint at the 10K WAL threshold. Drops without explicit flush.
/// Verifies all data recovered on reopen.
#[test]
fn stress_crash_after_vector_only_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vec_only_ckpt.omen");

    let batch_size = 9_000;
    let individual_size = 1_500; // WAL exceeds 10K total → triggers auto-checkpoint
    let n = batch_size + individual_size;
    let dim = 16;

    // Phase 1: flush batch (durable), then individual inserts trigger auto-checkpoint
    {
        let store = VectorStoreOptions::default()
            .dimensions(dim)
            .open(&path)
            .unwrap();

        let batch: Vec<_> = (0..batch_size)
            .map(|i| {
                (
                    format!("v{i}"),
                    Vector::new(random_vector(i, dim)),
                    serde_json::json!({"i": i}),
                )
            })
            .collect();
        store.set_batch(batch).unwrap();
        store.flush().unwrap(); // commit batch

        // Individual inserts push WAL past 10K → triggers auto-checkpoint
        for i in batch_size..n {
            store
                .set(
                    &format!("v{i}"),
                    Vector::new(random_vector(i, dim)),
                    serde_json::json!({"i": i}),
                )
                .unwrap();
        }
        // Drop without flush — auto-checkpoint already fired
    }

    // Phase 2: reopen, verify all data recovered
    {
        let store = VectorStore::open(&path).unwrap();
        assert_eq!(store.len(), n, "All vectors should survive crash recovery");

        // Spot-check first, middle, last
        for i in [0, n / 2, n - 1] {
            let (vec, meta) = store.get(&format!("v{i}")).unwrap();
            assert_eq!(vec.data.len(), dim);
            assert_eq!(meta["i"].as_u64().unwrap(), i as u64);
        }
    }
}

/// Recovery with WAL entries after vector-only checkpoint.
///
/// Flushes a batch (durable), then uses individual set() calls to push WAL
/// past 10K and trigger an auto-checkpoint. Inserts 500 more (WAL-only) and
/// crashes. On recovery all vectors (checkpointed + WAL-tail) must be present.
#[test]
fn stress_crash_checkpoint_plus_wal_tail() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ckpt_wal_tail.omen");

    let batch_size = 9_000;
    let individual_to_checkpoint = 1_500; // pushes WAL past 10K
    let wal_tail = 500;
    let dim = 16;

    {
        let store = VectorStoreOptions::default()
            .dimensions(dim)
            .open(&path)
            .unwrap();

        let batch: Vec<_> = (0..batch_size)
            .map(|i| {
                (
                    format!("cp{i}"),
                    Vector::new(random_vector(i, dim)),
                    serde_json::json!({}),
                )
            })
            .collect();
        store.set_batch(batch).unwrap();
        store.flush().unwrap(); // commit batch

        // Individual inserts to trigger auto-checkpoint
        for i in batch_size..batch_size + individual_to_checkpoint {
            store
                .set(
                    &format!("cp{i}"),
                    Vector::new(random_vector(i, dim)),
                    serde_json::json!({}),
                )
                .unwrap();
        }

        // Additional WAL-only inserts after checkpoint
        for i in 0..wal_tail {
            store
                .set(
                    &format!("tail{i}"),
                    Vector::new(random_vector(i + 100_000, dim)),
                    serde_json::json!({}),
                )
                .unwrap();
        }
        // Crash
    }

    {
        let store = VectorStore::open(&path).unwrap();
        let expected = batch_size + individual_to_checkpoint + wal_tail;
        assert_eq!(store.len(), expected, "All vectors recovered");

        // Verify WAL-tail data
        for i in 0..wal_tail {
            assert!(
                store.contains(&format!("tail{i}")),
                "WAL tail vector tail{i} missing"
            );
        }
    }
}

/// Upsert recovery: update existing vectors, crash before flush.
///
/// Flushes initial data, then updates vectors (WAL-only), crashes.
/// On recovery the updated values must be present.
#[test]
fn stress_crash_after_upserts() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("upsert_crash.omen");

    let n = 200;
    let dim = 32;

    // Phase 1: insert + flush
    {
        let store = VectorStoreOptions::default()
            .dimensions(dim)
            .open(&path)
            .unwrap();

        for i in 0..n {
            store
                .set(
                    &format!("u{i}"),
                    Vector::new(random_vector(i, dim)),
                    serde_json::json!({"version": 1}),
                )
                .unwrap();
        }
        store.flush().unwrap();

        // Update first half with new vectors and metadata
        for i in 0..n / 2 {
            store
                .set(
                    &format!("u{i}"),
                    Vector::new(random_vector(i + 50_000, dim)),
                    serde_json::json!({"version": 2}),
                )
                .unwrap();
        }
        // Crash without flush
    }

    // Phase 2: verify updates recovered
    {
        let store = VectorStore::open(&path).unwrap();
        assert_eq!(store.len(), n, "Count should be unchanged (upserts)");

        // Updated vectors should have version 2
        for i in 0..n / 2 {
            let (_, meta) = store.get(&format!("u{i}")).unwrap();
            assert_eq!(
                meta["version"].as_u64().unwrap(),
                2,
                "Vector u{i} should have updated metadata"
            );
        }

        // Untouched vectors should have version 1
        for i in n / 2..n {
            let (_, meta) = store.get(&format!("u{i}")).unwrap();
            assert_eq!(
                meta["version"].as_u64().unwrap(),
                1,
                "Vector u{i} should retain original metadata"
            );
        }
    }
}

/// Search correctness after crash recovery.
///
/// Inserts known vectors, crashes, reopens, verifies nearest-neighbor
/// search returns the correct closest vector.
#[test]
fn stress_crash_search_correctness() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("search_recovery.omen");

    let dim = 32;

    // Phase 1: insert vectors at known positions, flush, add WAL vectors, crash
    {
        let store = VectorStoreOptions::default()
            .dimensions(dim)
            .open(&path)
            .unwrap();

        // Insert a well-separated set of vectors
        // Vector "origin" at all zeros
        store
            .set(
                "origin",
                Vector::new(vec![0.0; dim]),
                serde_json::json!({"name": "origin"}),
            )
            .unwrap();

        // Vector "ones" at all ones
        store
            .set(
                "ones",
                Vector::new(vec![1.0; dim]),
                serde_json::json!({"name": "ones"}),
            )
            .unwrap();

        // Vector "tens" at all tens (far away)
        store
            .set(
                "tens",
                Vector::new(vec![10.0; dim]),
                serde_json::json!({"name": "tens"}),
            )
            .unwrap();

        store.flush().unwrap();

        // Add a WAL-only vector close to origin
        let mut near_origin = vec![0.0; dim];
        near_origin[0] = 0.01;
        store
            .set(
                "near_origin",
                Vector::new(near_origin),
                serde_json::json!({"name": "near_origin"}),
            )
            .unwrap();
        // Crash
    }

    // Phase 2: verify search correctness
    {
        let store = VectorStore::open(&path).unwrap();
        assert_eq!(store.len(), 4);

        // Search near origin — should find "origin" or "near_origin" as top results
        let query = Vector::new(vec![0.0; dim]);
        let results = store.search(&query, 4, None).unwrap();
        assert_eq!(results.len(), 4);

        // First result should be "origin" (exact match, distance 0)
        assert_eq!(results[0].id, "origin");
        assert!(results[0].distance < 0.001, "Origin should be exact match");

        // Second should be "near_origin"
        assert_eq!(results[1].id, "near_origin");

        // "tens" should be last (farthest)
        assert_eq!(results[3].id, "tens");
    }
}

/// Mixed operations across checkpoint boundary: insert, delete, update, crash.
///
/// Flushes initial state, then performs a mix of inserts, deletes, and
/// updates in WAL, crashes, and verifies the exact expected state.
#[test]
fn stress_crash_mixed_operations() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed_ops_crash.omen");

    let dim = 32;
    let initial = 300;
    let delete_range = 0..100; // delete first 100
    let update_range = 100..200; // update next 100
    let new_inserts = 50;

    // Phase 1
    {
        let store = VectorStoreOptions::default()
            .dimensions(dim)
            .open(&path)
            .unwrap();

        for i in 0..initial {
            store
                .set(
                    &format!("m{i}"),
                    Vector::new(random_vector(i, dim)),
                    serde_json::json!({"v": 1}),
                )
                .unwrap();
        }
        store.flush().unwrap();

        // Delete first 100
        for i in delete_range.clone() {
            store.delete(&format!("m{i}")).unwrap();
        }

        // Update 100-200 with new metadata
        for i in update_range.clone() {
            store
                .set(
                    &format!("m{i}"),
                    Vector::new(random_vector(i + 80_000, dim)),
                    serde_json::json!({"v": 2}),
                )
                .unwrap();
        }

        // Insert new vectors
        for i in 0..new_inserts {
            store
                .set(
                    &format!("new{i}"),
                    Vector::new(random_vector(i + 90_000, dim)),
                    serde_json::json!({"v": 1}),
                )
                .unwrap();
        }
        // Crash
    }

    // Phase 2
    {
        let store = VectorStore::open(&path).unwrap();

        let expected = (initial - delete_range.len()) + new_inserts;
        assert_eq!(store.len(), expected, "Count after mixed ops recovery");

        // Deleted vectors should be gone
        for i in delete_range {
            assert!(!store.contains(&format!("m{i}")), "m{i} should be deleted");
        }

        // Updated vectors should have version 2
        for i in update_range {
            let (_, meta) = store.get(&format!("m{i}")).unwrap();
            assert_eq!(meta["v"].as_u64().unwrap(), 2);
        }

        // Untouched vectors should have version 1
        for i in 200..initial {
            let (_, meta) = store.get(&format!("m{i}")).unwrap();
            assert_eq!(meta["v"].as_u64().unwrap(), 1);
        }

        // New inserts should exist
        for i in 0..new_inserts {
            assert!(store.contains(&format!("new{i}")));
        }
    }
}

/// Flush then reopen produces identical state.
///
/// Verifies that flush + reopen is lossless: same count, same vectors,
/// same metadata, same search results.
#[test]
fn stress_flush_reopen_identity() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("flush_identity.omen");

    let dim = 32;
    let n = 500;

    // Build original store
    let store = VectorStoreOptions::default()
        .dimensions(dim)
        .open(&path)
        .unwrap();

    for i in 0..n {
        store
            .set(
                &format!("fi{i}"),
                Vector::new(random_vector(i, dim)),
                serde_json::json!({"idx": i}),
            )
            .unwrap();
    }
    store.flush().unwrap();

    // Capture pre-close search results
    let query = Vector::new(random_vector(999, dim));
    let pre_results = store.search(&query, 10, None).unwrap();
    drop(store);

    // Reopen
    let store = VectorStore::open(&path).unwrap();
    assert_eq!(store.len(), n);

    // Same search results
    let post_results = store.search(&query, 10, None).unwrap();
    assert_eq!(pre_results.len(), post_results.len());
    for (pre, post) in pre_results.iter().zip(post_results.iter()) {
        assert_eq!(pre.id, post.id, "Search result IDs should match");
        assert!(
            (pre.distance - post.distance).abs() < 1e-5,
            "Distances should match"
        );
    }
}

/// Test: metadata-only update survives simulated crash (WAL recovery)
#[test]
fn stress_metadata_crash_recovery() {
    use tempfile::tempdir;
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("meta_crash_test");

    {
        let store = VectorStore::open(&db_path).unwrap();
        store
            .set(
                "doc1",
                Vector::new(vec![1.0f32; 64]),
                serde_json::json!({"title": "original"}),
            )
            .unwrap();
        store.flush().unwrap();
    }

    {
        let store = VectorStore::open(&db_path).unwrap();
        store
            .update("doc1", None, Some(serde_json::json!({"title": "updated"})))
            .unwrap();
        // Simulate crash - no flush()
    }

    {
        let store = VectorStore::open(&db_path).unwrap();
        let meta = store.get_metadata_by_id("doc1").unwrap();
        assert_eq!(
            meta.get("title").and_then(|v| v.as_str()),
            Some("updated"),
            "Metadata update should survive crash via WAL recovery"
        );
    }
}

/// Test: set_batch() without flush + crash = data lost (expected semantics)
#[test]
fn stress_batch_without_flush_data_lost() {
    use tempfile::tempdir;
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("batch_no_flush_test");

    {
        let store = VectorStore::open(&db_path).unwrap();
        let batch: Vec<(String, Vector, serde_json::Value)> = (0..100)
            .map(|i| {
                (
                    format!("doc{i}"),
                    Vector::new(random_vector(i, 64)),
                    serde_json::json!({"idx": i}),
                )
            })
            .collect();
        store.set_batch(batch).unwrap();
        // Simulate crash - no flush()
    }

    {
        let store = VectorStore::open(&db_path).unwrap();
        assert_eq!(
            store.len(),
            0,
            "set_batch() without flush should result in data loss on crash"
        );
    }
}

/// Test: set_batch() + flush + crash = data present (correct semantics)
#[test]
fn stress_batch_with_flush_data_survives() {
    use tempfile::tempdir;
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("batch_with_flush_test");

    let count = 100;

    {
        let store = VectorStore::open(&db_path).unwrap();
        let batch: Vec<(String, Vector, serde_json::Value)> = (0..count)
            .map(|i| {
                (
                    format!("doc{i}"),
                    Vector::new(random_vector(i, 64)),
                    serde_json::json!({"idx": i}),
                )
            })
            .collect();
        store.set_batch(batch).unwrap();
        store.flush().unwrap();
    }

    {
        let store = VectorStore::open(&db_path).unwrap();
        assert_eq!(
            store.len(),
            count,
            "set_batch() + flush() data should survive crash"
        );
        for i in 0..count {
            assert!(
                store.get_metadata_by_id(&format!("doc{i}")).is_some(),
                "doc{i} should be present after flush"
            );
        }
    }
}

/// Test: delete_batch uses single fsync (verify it succeeds for many deletions)
#[test]
fn stress_delete_batch_single_sync() {
    use tempfile::tempdir;
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("delete_batch_test");

    let count = 500;

    {
        let store = VectorStore::open(&db_path).unwrap();
        let batch: Vec<(String, Vector, serde_json::Value)> = (0..count)
            .map(|i| {
                (
                    format!("doc{i}"),
                    Vector::new(random_vector(i, 64)),
                    serde_json::json!({"idx": i}),
                )
            })
            .collect();
        store.set_batch(batch).unwrap();
        store.flush().unwrap();
    }

    {
        let store = VectorStore::open(&db_path).unwrap();
        assert_eq!(store.len(), count);

        let ids: Vec<String> = (0..count).map(|i| format!("doc{i}")).collect();
        let deleted = store.delete_batch(&ids).unwrap();
        assert_eq!(deleted, count, "All vectors should be deleted");
        assert_eq!(store.len(), 0);

        store.flush().unwrap();
    }

    {
        let store = VectorStore::open(&db_path).unwrap();
        assert_eq!(store.len(), 0, "Deletions should persist after flush");
    }
}

// ============================================================
// Edge stress tests
// ============================================================

/// Crash recovery with edges at scale.
///
/// Insert 500 vectors + 1000 edges, flush, delete 200 vectors (cascade),
/// crash. Reopen and verify no orphaned edges, correct counts.
#[test]
fn stress_crash_recovery_edges_at_scale() {
    use crate::vector::store::edge_store::EdgeDirection;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edge_scale.omen");

    let num_vectors = 500;
    let num_edges = 1000;
    let num_deletes = 200;
    let dim = 16;

    // Phase 1: insert, add edges, flush, delete, crash
    {
        let mut store = VectorStoreOptions::default()
            .dimensions(dim)
            .open(&path)
            .unwrap();

        // Insert vectors
        for i in 0..num_vectors {
            store
                .set(
                    &format!("v{i}"),
                    Vector::new(random_vector(i, dim)),
                    serde_json::json!({}),
                )
                .unwrap();
        }

        // Add edges between deterministic pairs
        let edge_types = ["link", "ref", "parent", "sibling"];
        for i in 0..num_edges {
            let from_idx = i % num_vectors;
            let to_idx = (i * 7 + 3) % num_vectors;
            if from_idx == to_idx {
                continue;
            }
            let etype = edge_types[i % edge_types.len()];
            store
                .add_edge(
                    &format!("v{from_idx}"),
                    &format!("v{to_idx}"),
                    etype,
                    1.0,
                    None,
                )
                .unwrap();
        }

        store.flush().unwrap();
        let edges_before = store.edge_count();
        assert!(edges_before > 0, "Should have edges");

        // Delete first 200 vectors (cascade removes their edges)
        for i in 0..num_deletes {
            store.delete(&format!("v{i}")).unwrap();
        }
        // Crash
    }

    // Phase 2: verify recovery
    {
        let store = VectorStore::open(&path).unwrap();
        assert_eq!(
            store.len(),
            num_vectors - num_deletes,
            "Correct vector count after recovery"
        );

        // No orphaned edges — every edge must reference live nodes
        let deleted: rustc_hash::FxHashSet<String> =
            (0..num_deletes).map(|i| format!("v{i}")).collect();

        for i in num_deletes..num_vectors {
            let id = format!("v{i}");
            for edge in store.get_edges(&id, EdgeDirection::Both, None) {
                assert!(
                    !deleted.contains(&edge.from_id),
                    "Orphaned edge from deleted node {}",
                    edge.from_id
                );
                assert!(
                    !deleted.contains(&edge.to_id),
                    "Orphaned edge to deleted node {}",
                    edge.to_id
                );
            }
        }
    }
}

/// Large edge metadata: edges with 100KB+ JSON metadata.
///
/// Verifies serialization roundtrip and WAL recovery with large metadata.
#[test]
fn stress_large_edge_metadata() {
    use crate::vector::store::edge_store::EdgeDirection;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large_edge_meta.omen");

    let large_text: String = (0..100_000)
        .map(|i| ((i % 26) as u8 + b'a') as char)
        .collect();
    let large_meta = serde_json::json!({
        "large_field": large_text,
        "array": (0..1000).collect::<Vec<i32>>(),
        "nested": {"deep": {"key": "value"}},
    });

    // Phase 1: insert with large metadata, flush, add WAL-only edge, crash
    {
        let mut store = VectorStoreOptions::default()
            .dimensions(16)
            .open(&path)
            .unwrap();

        store
            .set(
                "a",
                Vector::new(random_vector(0, 16)),
                serde_json::json!({}),
            )
            .unwrap();
        store
            .set(
                "b",
                Vector::new(random_vector(1, 16)),
                serde_json::json!({}),
            )
            .unwrap();
        store
            .set(
                "c",
                Vector::new(random_vector(2, 16)),
                serde_json::json!({}),
            )
            .unwrap();

        // Flushed edge with large metadata
        store
            .add_edge("a", "b", "big", 0.5, Some(large_meta.clone()))
            .unwrap();
        store.flush().unwrap();

        // WAL-only edge with large metadata
        store
            .add_edge("b", "c", "big", 0.75, Some(large_meta.clone()))
            .unwrap();
        // Crash
    }

    // Phase 2: verify both edges recovered with metadata intact
    {
        let store = VectorStore::open(&path).unwrap();
        assert_eq!(store.edge_count(), 2);

        let ab = store.get_edges("a", EdgeDirection::Outgoing, None);
        assert_eq!(ab.len(), 1);
        let ab_meta = ab[0].metadata.as_ref().unwrap();
        assert_eq!(ab_meta["large_field"].as_str().unwrap().len(), 100_000);
        assert_eq!(ab_meta["array"].as_array().unwrap().len(), 1000);

        let bc = store.get_edges("b", EdgeDirection::Outgoing, None);
        assert_eq!(bc.len(), 1);
        let bc_meta = bc[0].metadata.as_ref().unwrap();
        assert_eq!(bc_meta["large_field"].as_str().unwrap().len(), 100_000);
    }
}

/// Repeated crash recovery with edges.
///
/// 5 cycles of: open → insert vectors + edges → crash → reopen.
/// Verify edge count grows correctly and no corruption.
#[test]
fn stress_repeated_crash_recovery_with_edges() {
    use crate::vector::store::edge_store::EdgeDirection;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("repeated_edge_crash.omen");

    let vectors_per_cycle = 20;
    let edges_per_cycle = 10;
    let num_cycles = 5;
    let dim = 16;

    for cycle in 0..num_cycles {
        {
            let mut store = if cycle == 0 {
                VectorStore::open_with_dimensions(&path, dim).unwrap()
            } else {
                VectorStore::open(&path).unwrap()
            };

            // Verify previous data
            let expected_vecs = cycle * vectors_per_cycle;
            assert_eq!(
                store.len(),
                expected_vecs,
                "Cycle {cycle}: wrong vector count"
            );

            // Verify edges from previous cycles are present
            let expected_edges = cycle * edges_per_cycle;
            assert_eq!(
                store.edge_count(),
                expected_edges,
                "Cycle {cycle}: wrong edge count"
            );

            // Add more vectors
            for i in 0..vectors_per_cycle {
                let id = format!("c{cycle}_v{i}");
                store
                    .set(
                        &id,
                        Vector::new(random_vector(cycle * 1000 + i, dim)),
                        serde_json::json!({}),
                    )
                    .unwrap();
            }

            // Add edges within this cycle's vectors
            for i in 0..edges_per_cycle {
                let from = format!("c{cycle}_v{}", i % vectors_per_cycle);
                let to = format!("c{cycle}_v{}", (i * 3 + 1) % vectors_per_cycle);
                if from == to {
                    // Still add an edge, just pick different target
                    let to = format!("c{cycle}_v{}", (i + 1) % vectors_per_cycle);
                    store.add_edge(&from, &to, "cycle_edge", 1.0, None).unwrap();
                } else {
                    store.add_edge(&from, &to, "cycle_edge", 1.0, None).unwrap();
                }
            }

            // Crash (no flush)
        }
    }

    // Final verification
    let store = VectorStore::open(&path).unwrap();
    assert_eq!(store.len(), num_cycles * vectors_per_cycle);
    assert_eq!(store.edge_count(), num_cycles * edges_per_cycle);

    // Verify edges from each cycle
    for cycle in 0..num_cycles {
        let first_vec = format!("c{cycle}_v0");
        let edges = store.get_edges(&first_vec, EdgeDirection::Both, None);
        assert!(
            !edges.is_empty(),
            "Cycle {cycle} first vector should have edges"
        );
    }
}

/// Multiple flush/reopen cycles maintain data integrity.
#[test]
fn stress_multiple_flush_reopen_cycles() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi_cycle.omen");

    let dim = 32;
    let per_cycle = 100;
    let cycles = 5;

    for cycle in 0..cycles {
        let store = if cycle == 0 {
            VectorStoreOptions::default()
                .dimensions(dim)
                .open(&path)
                .unwrap()
        } else {
            VectorStore::open(&path).unwrap()
        };

        // Verify data from previous cycles
        let expected = cycle * per_cycle;
        assert_eq!(store.len(), expected, "Cycle {cycle}: count mismatch");

        // Add more data
        for i in 0..per_cycle {
            let id = format!("c{cycle}_v{i}");
            store
                .set(
                    &id,
                    Vector::new(random_vector(cycle * 10_000 + i, dim)),
                    serde_json::json!({"cycle": cycle}),
                )
                .unwrap();
        }

        store.flush().unwrap();
    }

    // Final verification
    let store = VectorStore::open(&path).unwrap();
    assert_eq!(store.len(), cycles * per_cycle);

    // Check each cycle's data
    for cycle in 0..cycles {
        for i in [0, per_cycle / 2, per_cycle - 1] {
            let (_, meta) = store.get(&format!("c{cycle}_v{i}")).unwrap();
            assert_eq!(meta["cycle"].as_u64().unwrap(), cycle as u64);
        }
    }
}
