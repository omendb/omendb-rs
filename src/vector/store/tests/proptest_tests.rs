use super::super::*;
use proptest::prelude::*;

mod general {
    use super::*;

    proptest! {
        #[test]
        fn params_roundtrip(
            m in 16usize..64,
            ef_construction in 100usize..500,
            ef_search in 100usize..500,
            dimensions in 8usize..128
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("test.omen");

            // Create store with specific params using VectorStoreOptions
            {
                let mut store = VectorStoreOptions::default()
                    .dimensions(dimensions)
                    .m(m)
                    .ef_construction(ef_construction)
                    .ef_search(ef_search)
                    .open(&path)
                    .unwrap();

                // Insert some vectors to trigger HNSW creation
                for i in 0..10 {
                    let v = Vector::new((0..dimensions).map(|j| (i * j) as f32 * 0.1).collect());
                    let id = format!("vec_{i}");
                    store.set(id, v, serde_json::json!({})).unwrap();
                }

                store.flush().unwrap();
            }

            // Load and verify
            let loaded = VectorStore::open(&path).unwrap();
            prop_assert_eq!(loaded.hnsw_m, m);
            prop_assert_eq!(loaded.hnsw_ef_construction, ef_construction);
            prop_assert_eq!(loaded.hnsw_ef_search, ef_search);
            prop_assert_eq!(loaded.len(), 10);
        }

        #[test]
        fn id_mapping_consistency(
            num_inserts in 10usize..100,
            num_deletes in 0usize..10
        ) {
            let mut store = VectorStore::new(8);

            // Insert vectors
            let mut ids = Vec::new();
            for i in 0..num_inserts {
                let id = format!("vec_{i}");
                let v = Vector::new((0..8).map(|j| (i * j) as f32 * 0.1).collect());
                store.set(id.clone(), v, serde_json::json!({})).unwrap();
                ids.push(id);
            }

            // Delete some
            let to_delete = num_deletes.min(ids.len());
            for id in ids.iter().take(to_delete) {
                store.delete(id).unwrap();
            }

            // Verify consistency: count matches
            let expected_count = num_inserts - to_delete;
            prop_assert_eq!(store.len(), expected_count);

            // Verify remaining IDs are accessible
            for id in ids.iter().skip(to_delete) {
                prop_assert!(store.contains(id));
            }
        }

        #[test]
        fn non_quantized_roundtrip(
            num_vectors in 10usize..30
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("nonquant.omen");

            // Create non-quantized store
            {
                let mut store = VectorStoreOptions::default()
                    .dimensions(64)
                    .open(&path)
                    .unwrap();

                // Insert some vectors
                for i in 0..num_vectors {
                    let v = Vector::new((0..64).map(|j| (i * j) as f32 * 0.01).collect());
                    let id = format!("vec_{i}");
                    store.set(id, v, serde_json::json!({})).unwrap();
                }

                store.flush().unwrap();
            }

            // Load and verify
            let loaded = VectorStore::open(&path).unwrap();
            prop_assert_eq!(loaded.len(), num_vectors);
        }

        #[test]
        fn sq8_quantized_roundtrip(
            num_vectors in 10usize..30
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("sq8quant.omen");

            // Create SQ8 quantized store
            {
                let mut store = VectorStoreOptions::default()
                    .dimensions(64)
                    .quantization(true)
                    .open(&path)
                    .unwrap();

                // Insert vectors in batches to trigger the bug
                for batch in 0..3 {
                    let batch_vectors: Vec<_> = (0..num_vectors / 3)
                        .map(|i| {
                            let idx = batch * (num_vectors / 3) + i;
                            let v = Vector::new((0..64).map(|j| (idx * j + batch) as f32 * 0.01).collect());
                            let id = format!("vec_{idx}");
                            (id, v, serde_json::json!({"batch": batch}))
                        })
                        .collect();
                    store.set_batch(batch_vectors).unwrap();
                }

                store.flush().unwrap();

                // Verify count before close
                prop_assert!(store.len() > 0, "Store should not be empty");
            }

            // Load and verify
            let loaded = VectorStore::open(&path).unwrap();
            prop_assert!(loaded.len() > 0, "Loaded store should not be empty");

            // Verify all IDs are searchable
            for batch in 0..3 {
                for i in 0..(num_vectors / 3) {
                    let idx = batch * (num_vectors / 3) + i;
                    let id = format!("vec_{idx}");
                    prop_assert!(
                        loaded.contains(&id),
                        "ID '{}' not found after reload",
                        id
                    );
                }
            }
        }
    }
}

mod persistence {
    use super::*;

    fn arb_vector_data(dim: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-100.0f32..100.0f32, dim)
    }

    proptest! {
        #[test]
        fn wal_recovery_no_flush(
            num_vectors in 1usize..20,
            dim in 4usize..32
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("wal_recovery.omen");

            let mut expected_vectors = Vec::new();

            // Insert without flush
            {
                let mut store = VectorStore::open_with_dimensions(&path, dim).unwrap();

                for i in 0..num_vectors {
                    let data: Vec<f32> = (0..dim).map(|j| (i * 100 + j) as f32 * 0.01).collect();
                    let id = format!("v{}", i);
                    store.set(id.clone(), Vector::new(data.clone()), serde_json::json!({"i": i})).unwrap();
                    expected_vectors.push((id, data, i));
                }
                // NO flush - data should survive via WAL
            }

            // Reopen and verify WAL recovery
            {
                let store = VectorStore::open(&path).unwrap();
                prop_assert_eq!(store.len(), num_vectors, "WAL recovery should restore all vectors");

                for (id, expected_data, idx) in &expected_vectors {
                    prop_assert!(store.contains(id), "ID not found after WAL recovery");
                    let (vec, meta) = store.get(id).unwrap();
                    prop_assert_eq!(&vec.data, expected_data, "Vector data mismatch");
                    prop_assert_eq!(meta["i"].as_u64().unwrap() as usize, *idx, "Metadata mismatch");
                }
            }
        }

        #[test]
        fn wal_recovery_with_deletes(
            num_inserts in 5usize..30,
            delete_ratio in 0.1f64..0.5
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("wal_delete.omen");

            let dim = 8;
            let num_deletes = ((num_inserts as f64) * delete_ratio) as usize;

            // Insert then delete without flush
            {
                let mut store = VectorStore::open_with_dimensions(&path, dim).unwrap();

                for i in 0..num_inserts {
                    let data: Vec<f32> = (0..dim).map(|j| (i + j) as f32).collect();
                    store.set(format!("v{}", i), Vector::new(data), serde_json::json!({})).unwrap();
                }

                // Delete first N vectors
                for i in 0..num_deletes {
                    store.delete(&format!("v{}", i)).unwrap();
                }
                // NO flush
            }

            // Reopen and verify
            {
                let store = VectorStore::open(&path).unwrap();
                let expected_count = num_inserts - num_deletes;
                prop_assert_eq!(store.len(), expected_count);

                // Deleted vectors should not exist
                for i in 0..num_deletes {
                    let id = format!("v{}", i);
                    prop_assert!(!store.contains(&id), "Deleted vector should not exist");
                }

                // Remaining vectors should exist
                for i in num_deletes..num_inserts {
                    let id = format!("v{}", i);
                    prop_assert!(store.contains(&id), "Vector should exist");
                }
            }
        }

        #[test]
        fn checkpoint_plus_wal(
            checkpoint_count in 5usize..20,
            wal_count in 1usize..10
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("checkpoint_wal.omen");
            let dim = 8;

            // Insert, flush, insert more (no flush)
            {
                let mut store = VectorStore::open_with_dimensions(&path, dim).unwrap();

                // First batch - will be checkpointed
                for i in 0..checkpoint_count {
                    let data: Vec<f32> = (0..dim).map(|j| (i + j) as f32).collect();
                    store.set(format!("cp{}", i), Vector::new(data), serde_json::json!({})).unwrap();
                }
                store.flush().unwrap();

                // Second batch - only in WAL
                for i in 0..wal_count {
                    let data: Vec<f32> = (0..dim).map(|j| (i + j + 1000) as f32).collect();
                    store.set(format!("wal{}", i), Vector::new(data), serde_json::json!({})).unwrap();
                }
                // NO flush for second batch
            }

            // Reopen - should have both checkpoint and WAL data
            {
                let store = VectorStore::open(&path).unwrap();
                prop_assert_eq!(store.len(), checkpoint_count + wal_count);

                for i in 0..checkpoint_count {
                    let id = format!("cp{}", i);
                    prop_assert!(store.contains(&id));
                }
                for i in 0..wal_count {
                    let id = format!("wal{}", i);
                    prop_assert!(store.contains(&id));
                }
            }
        }

        #[test]
        fn vector_data_integrity(
            values in arb_vector_data(16)
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("data_integrity.omen");

            // Store exact values
            {
                let mut store = VectorStore::open_with_dimensions(&path, 16).unwrap();
                store.set("test".to_string(), Vector::new(values.clone()), serde_json::json!({})).unwrap();
                store.flush().unwrap();
            }

            // Verify exact match after reload
            {
                let store = VectorStore::open(&path).unwrap();
                let (vec, _) = store.get("test").unwrap();
                prop_assert_eq!(vec.data.len(), values.len());
                for (i, (got, expected)) in vec.data.iter().zip(values.iter()).enumerate() {
                    prop_assert!(
                        (got - expected).abs() < f32::EPSILON,
                        "Float mismatch at index {i}: got {got}, expected {expected}"
                    );
                }
            }
        }

        #[test]
        fn metadata_types_roundtrip(
            int_val in -1000i64..1000,
            float_val in -100.0f64..100.0,
            bool_val in proptest::bool::ANY,
            str_len in 1usize..20
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("meta_types.omen");

            let string_val: String = (0..str_len).map(|i| ((i % 26) as u8 + b'a') as char).collect();

            let metadata = serde_json::json!({
                "int": int_val,
                "float": float_val,
                "bool": bool_val,
                "string": string_val,
                "null": null,
                "array": [1, 2, 3],
                "nested": {"a": 1, "b": "two"}
            });

            // Store
            {
                let mut store = VectorStore::open_with_dimensions(&path, 4).unwrap();
                store.set("test".to_string(), Vector::new(vec![1.0, 2.0, 3.0, 4.0]), metadata.clone()).unwrap();
                store.flush().unwrap();
            }

            // Verify
            {
                let store = VectorStore::open(&path).unwrap();
                let (_, loaded_meta) = store.get("test").unwrap();
                prop_assert_eq!(loaded_meta["int"].as_i64().unwrap(), int_val);
                prop_assert_eq!(loaded_meta["bool"].as_bool().unwrap(), bool_val);
                prop_assert_eq!(loaded_meta["string"].as_str().unwrap(), string_val);
                prop_assert!(loaded_meta["null"].is_null());
                prop_assert_eq!(loaded_meta["array"].as_array().unwrap().len(), 3);
                prop_assert_eq!(loaded_meta["nested"]["a"].as_i64().unwrap(), 1);
                prop_assert_eq!(loaded_meta["nested"]["b"].as_str().unwrap(), "two");
            }
        }

        #[test]
        fn upsert_persistence(
            num_updates in 2usize..5
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("upsert.omen");
            let dim = 4;
            let last_update = num_updates - 1;

            // Expected: last update's data (update index = num_updates - 1)
            let final_data: Vec<f32> = (0..dim).map(|i| (last_update * 100 + i) as f32).collect();

            // Insert then update multiple times
            {
                let mut store = VectorStore::open_with_dimensions(&path, dim).unwrap();

                for update in 0..num_updates {
                    let data: Vec<f32> = (0..dim).map(|i| (update * 100 + i) as f32).collect();
                    store.set("same_id".to_string(), Vector::new(data), serde_json::json!({"version": update})).unwrap();
                }
                store.flush().unwrap();
            }

            // Should have latest version
            {
                let store = VectorStore::open(&path).unwrap();
                prop_assert_eq!(store.len(), 1);
                let (vec, meta) = store.get("same_id").unwrap();
                prop_assert_eq!(vec.data, final_data);
                prop_assert_eq!(meta["version"].as_u64().unwrap() as usize, last_update);
            }
        }

        #[test]
        fn batch_persistence(
            num_batches in 2usize..5,
            batch_size in 5usize..15
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("batch.omen");
            let dim = 8;

            let total = num_batches * batch_size;

            // Insert in batches
            {
                let mut store = VectorStore::open_with_dimensions(&path, dim).unwrap();

                for batch_idx in 0..num_batches {
                    let items: Vec<_> = (0..batch_size).map(|i| {
                        let idx = batch_idx * batch_size + i;
                        let data: Vec<f32> = (0..dim).map(|j| (idx * 10 + j) as f32).collect();
                        (format!("b{}_i{}", batch_idx, i), Vector::new(data), serde_json::json!({"batch": batch_idx, "i": i}))
                    }).collect();
                    store.set_batch(items).unwrap();
                }
                store.flush().unwrap();
            }

            // Verify all
            {
                let store = VectorStore::open(&path).unwrap();
                prop_assert_eq!(store.len(), total);

                for batch_idx in 0..num_batches {
                    for i in 0..batch_size {
                        let id = format!("b{}_i{}", batch_idx, i);
                        prop_assert!(store.contains(&id), "Missing ID");
                    }
                }
            }
        }
    }
}
