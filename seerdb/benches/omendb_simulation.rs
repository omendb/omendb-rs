//! Omendb Workload Simulation
//!
//! Simulates two primary GraphDB patterns:
//! 1. **Adjacency List (Merge Operator)**:
//!    - Keys are `SrcID` (8 bytes).
//!    - Values are appended `DstID` (8 bytes).
//!    - Read via `Get` (returns full list).
//!    - Heavy `Merge` contention on hot keys.
//!
//! 2. **Edge List (Prefix Scan)**:
//!    - Keys are `SrcID:DstID` (16 bytes).
//!    - Values are empty/small.
//!    - Read via `Prefix` scan of `SrcID`.
//!    - Heavy `Put` throughput, but potentially slower reads (Seek + Next).
//!
//! Run with: cargo bench --bench omendb_simulation

use byteorder::{BigEndian, ByteOrder};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use seerdb::{DBOptions, MergeOperator, SyncPolicy, DB};
use std::sync::Arc;
use std::thread;
use tempfile::tempdir;

const OPS_PER_ITERATION: u64 = 10_000;
const NUM_NODES: u64 = 1_000; // 1k "supernodes" receiving edges

// --- Merge Operator for Adjacency Lists ---
#[derive(Debug)]
struct AdjacencyMerge;

impl MergeOperator for AdjacencyMerge {
    fn full_merge(
        &self,
        _key: &[u8],
        existing_value: Option<&[u8]>,
        operands: &[&[u8]],
    ) -> Option<Vec<u8>> {
        // Calculate total size: existing + sum(operands)
        let mut total_len = existing_value.map(|v| v.len()).unwrap_or(0);
        for op in operands {
            total_len += op.len();
        }

        let mut result = Vec::with_capacity(total_len);
        if let Some(v) = existing_value {
            result.extend_from_slice(v);
        }
        for op in operands {
            result.extend_from_slice(op);
        }
        Some(result)
    }

    fn partial_merge(
        &self,
        _key: &[u8],
        left_operand: &[u8],
        right_operand: &[u8],
    ) -> Option<Vec<u8>> {
        let mut result = Vec::with_capacity(left_operand.len() + right_operand.len());
        result.extend_from_slice(left_operand);
        result.extend_from_slice(right_operand);
        Some(result)
    }

    fn name(&self) -> &str {
        "AdjacencyMerge"
    }
}

// --- Helper to encode u64 key ---
fn encode_key(k: u64) -> [u8; 8] {
    let mut buf = [0u8; 8];
    BigEndian::write_u64(&mut buf, k);
    buf
}

fn encode_edge_key(src: u64, dst: u64) -> [u8; 16] {
    let mut buf = [0u8; 16];
    BigEndian::write_u64(&mut buf[0..8], src);
    BigEndian::write_u64(&mut buf[8..16], dst);
    buf
}

fn bench_omendb_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("omendb_simulation");
    group.throughput(Throughput::Elements(OPS_PER_ITERATION));

    // 1. ADJACENCY LIST (Merge Operator)
    // Scenario: Massive parallel edge ingestion into common nodes
    for threads in [1, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("adjacency_ingest_merge", threads),
            &threads,
            |b, &num_threads| {
                b.iter(|| {
                    let dir = tempdir().unwrap();
                    let mut opts = DBOptions {
                        data_dir: dir.path().to_path_buf(),
                        memtable_capacity: 64 * 1024 * 1024,
                        wal_sync_policy: SyncPolicy::None, // Batch ingestion usually disables sync
                        background_compaction: true,
                        background_flush: true,
                        ..Default::default()
                    };
                    opts.merge_operator = Some(Arc::new(AdjacencyMerge));
                    let db = Arc::new(DB::open(opts).unwrap());

                    let ops_per_thread = OPS_PER_ITERATION / num_threads as u64;
                    let handles: Vec<_> = (0..num_threads)
                        .map(|_| {
                            let db = db.clone();
                            thread::spawn(move || {
                                let mut rng = rand::thread_rng();
                                let mut dst_buf = [0u8; 8];

                                for _ in 0..ops_per_thread {
                                    // Pick a random source node (Supernode)
                                    let src = rng.gen_range(0..NUM_NODES);
                                    let dst = rng.gen::<u64>(); // Random target

                                    BigEndian::write_u64(&mut dst_buf, dst);

                                    let key = encode_key(src);
                                    // Merge: Append dst to src's list
                                    let _ = black_box(db.merge(&key, &dst_buf));
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                })
            },
        );
    }

    // 2. EDGE LIST (Prefix Scan)
    // Scenario: Massive parallel edge ingestion via Put(Src:Dst)
    // Note: We expect this to be slower on Read (Scanning), but maybe faster on Write (No RMW/Merge cost)?
    // Actually, standard Put is O(1) blind write too. But it creates MORE keys (index pressure).
    for threads in [1, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("edgelist_ingest_put", threads),
            &threads,
            |b, &num_threads| {
                b.iter(|| {
                    let dir = tempdir().unwrap();
                    let opts = DBOptions {
                        data_dir: dir.path().to_path_buf(),
                        memtable_capacity: 64 * 1024 * 1024,
                        wal_sync_policy: SyncPolicy::None,
                        background_compaction: true,
                        background_flush: true,
                        ..Default::default()
                    };
                    let db = Arc::new(DB::open(opts).unwrap());

                    let ops_per_thread = OPS_PER_ITERATION / num_threads as u64;
                    let handles: Vec<_> = (0..num_threads)
                        .map(|_| {
                            let db = db.clone();
                            thread::spawn(move || {
                                let mut rng = rand::thread_rng();

                                for _ in 0..ops_per_thread {
                                    let src = rng.gen_range(0..NUM_NODES);
                                    let dst = rng.gen::<u64>();

                                    let key = encode_edge_key(src, dst);
                                    // Put: Key=Src:Dst, Val=Empty
                                    let _ = black_box(db.put(&key, &[]));
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                })
            },
        );
    }

    // 3. READ COMPARISON (Get vs Prefix Scan)
    // We preload the DB, then measure read performance.
    // This requires setting up the DB *outside* the measurement loop, which Criterion `iter_batched` handles.
    // However, due to Arc<DB> and thread spawning complexity, we'll simplify:
    // We'll measure "Ops/Sec" of a fixed duration read phase on a pre-loaded DB.

    // ... Actually, integrating complex setup in Criterion is tricky.
    // Let's stick to the write-heavy ingest for now, as that's the primary MergeOp differentiator.
    // We can add a separate "read_workload" group if needed.

    group.finish();
}

criterion_group!(benches, bench_omendb_simulation);
criterion_main!(benches);
