"""Concurrent access tests for OmenDB - P0 production hardening"""

import math
import os
import random
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

import omendb


def generate_random_vector(dim: int, seed: int = None) -> list:
    """Generate a random normalized vector"""
    if seed is not None:
        random.seed(seed)
    embedding = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in embedding))
    return [x / norm for x in embedding]


def generate_random_vectors(n: int, dim: int, seed: int = 42) -> list:
    """Generate random normalized vectors"""
    random.seed(seed)
    vectors = []
    for i in range(n):
        embedding = generate_random_vector(dim)
        vectors.append(
            {
                "id": f"vec_{i}",
                "vector": embedding,
                "metadata": {"index": i, "group": i % 10},
            }
        )
    return vectors


class TestConcurrentReaders:
    """Test multiple threads reading simultaneously"""

    def test_concurrent_search_threads(self):
        """Multiple threads searching the same db concurrently"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            # Insert test data
            vectors = generate_random_vectors(1000, 64)
            db.set(vectors)

            errors = []
            results_count = []
            lock = threading.Lock()

            def search_worker(thread_id: int, num_searches: int):
                """Worker that performs multiple searches"""
                try:
                    for i in range(num_searches):
                        query = generate_random_vector(64, seed=thread_id * 1000 + i)
                        results = db.search(query, k=10)
                        with lock:
                            results_count.append(len(results))
                        if len(results) != 10:
                            with lock:
                                errors.append(f"Thread {thread_id} got {len(results)} results")
                except Exception as e:
                    with lock:
                        errors.append(f"Thread {thread_id} error: {e}")

            # Run 10 threads each doing 50 searches
            threads = []
            for i in range(10):
                t = threading.Thread(target=search_worker, args=(i, 50))
                threads.append(t)

            start = time.time()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            elapsed = time.time() - start

            assert len(errors) == 0, f"Errors: {errors}"
            assert len(results_count) == 500  # 10 threads * 50 searches
            assert all(r == 10 for r in results_count)
            print(f"\n10 threads, 500 searches: {elapsed:.2f}s ({500 / elapsed:.0f} QPS)")

    def test_concurrent_get_threads(self):
        """Multiple threads calling get() concurrently"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            vectors = generate_random_vectors(1000, 64)
            db.set(vectors)

            errors = []
            successful_gets = []
            lock = threading.Lock()

            def get_worker(thread_id: int, num_gets: int):
                try:
                    for i in range(num_gets):
                        vec_id = f"vec_{(thread_id * num_gets + i) % 1000}"
                        result = db.get(vec_id)
                        with lock:
                            successful_gets.append(result is not None)
                except Exception as e:
                    with lock:
                        errors.append(f"Thread {thread_id} error: {e}")

            threads = []
            for i in range(10):
                t = threading.Thread(target=get_worker, args=(i, 100))
                threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors: {errors}"
            assert len(successful_gets) == 1000
            assert all(successful_gets)

    def test_concurrent_filtered_search(self):
        """Multiple threads doing filtered searches"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            vectors = generate_random_vectors(1000, 64)
            db.set(vectors)

            errors = []
            lock = threading.Lock()

            def filtered_search_worker(thread_id: int, num_searches: int):
                try:
                    for i in range(num_searches):
                        query = generate_random_vector(64, seed=thread_id * 1000 + i)
                        group = thread_id % 10
                        results = db.search(query, k=10, filter={"group": group})
                        # Verify filter is respected
                        for r in results:
                            if r["metadata"]["group"] != group:
                                with lock:
                                    errors.append(
                                        f"Filter not respected: got {r['metadata']['group']}, expected {group}"
                                    )
                except Exception as e:
                    with lock:
                        errors.append(f"Thread {thread_id} error: {e}")

            threads = []
            for i in range(10):
                t = threading.Thread(target=filtered_search_worker, args=(i, 20))
                threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors: {errors}"


class TestConcurrentWriters:
    """Test multiple threads writing simultaneously"""

    def test_concurrent_set_threads(self):
        """Multiple threads setting to the same db"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            errors = []
            lock = threading.Lock()

            def set_worker(thread_id: int, num_vectors: int):
                try:
                    for i in range(num_vectors):
                        vec_id = f"thread_{thread_id}_vec_{i}"
                        embedding = generate_random_vector(64, seed=thread_id * 10000 + i)
                        db.set(
                            [
                                {
                                    "id": vec_id,
                                    "vector": embedding,
                                    "metadata": {"thread": thread_id, "index": i},
                                }
                            ]
                        )
                except Exception as e:
                    with lock:
                        errors.append(f"Thread {thread_id} error: {e}")

            threads = []
            num_threads = 5
            vectors_per_thread = 100

            start = time.time()
            for i in range(num_threads):
                t = threading.Thread(target=set_worker, args=(i, vectors_per_thread))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()
            elapsed = time.time() - start

            assert len(errors) == 0, f"Errors: {errors}"

            # Verify all vectors were inserted
            expected = num_threads * vectors_per_thread
            actual = len(db)
            assert actual == expected, f"Expected {expected}, got {actual}"
            print(f"\n{num_threads} threads, {expected} inserts: {elapsed:.2f}s")

    @pytest.mark.slow
    def test_concurrent_delete_threads(self):
        """Multiple threads deleting simultaneously"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            # Insert initial data
            vectors = generate_random_vectors(1000, 64)
            db.set(vectors)
            assert len(db) == 1000

            errors = []
            lock = threading.Lock()

            def delete_worker(thread_id: int, start_idx: int, end_idx: int):
                try:
                    for i in range(start_idx, end_idx):
                        db.delete([f"vec_{i}"])
                except Exception as e:
                    with lock:
                        errors.append(f"Thread {thread_id} error: {e}")

            # Each thread deletes a distinct range
            threads = []
            chunk_size = 100
            for i in range(10):
                t = threading.Thread(
                    target=delete_worker, args=(i, i * chunk_size, (i + 1) * chunk_size)
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors: {errors}"
            assert len(db) == 0, f"Expected 0, got {len(db)}"

    def test_concurrent_update_same_vector(self):
        """Multiple threads updating the same vector (race condition test)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            # Insert single vector
            initial = generate_random_vector(64, seed=0)
            db.set([{"id": "shared", "vector": initial, "metadata": {"version": 0}}])

            errors = []
            lock = threading.Lock()
            updates_completed = [0]

            def update_worker(thread_id: int, num_updates: int):
                try:
                    for i in range(num_updates):
                        embedding = generate_random_vector(64, seed=thread_id * 1000 + i)
                        db.update(
                            "shared",
                            embedding,
                            metadata={"version": thread_id * 1000 + i},
                        )
                        with lock:
                            updates_completed[0] += 1
                except Exception as e:
                    with lock:
                        errors.append(f"Thread {thread_id} error: {e}")

            threads = []
            for i in range(5):
                t = threading.Thread(target=update_worker, args=(i, 20))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # No errors should occur (even if final state is non-deterministic)
            assert len(errors) == 0, f"Errors: {errors}"
            assert updates_completed[0] == 100
            assert len(db) == 1  # Still only one vector


class TestMixedReadWrite:
    """Test concurrent reads and writes"""

    def test_read_while_writing(self):
        """Search while other threads are inserting"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            # Insert initial data
            vectors = generate_random_vectors(500, 64)
            db.set(vectors)

            errors = []
            search_results = []
            lock = threading.Lock()
            stop_flag = threading.Event()

            def writer_thread(thread_id: int):
                """Insert vectors continuously"""
                try:
                    i = 0
                    while not stop_flag.is_set():
                        vec_id = f"new_thread_{thread_id}_vec_{i}"
                        embedding = generate_random_vector(64, seed=thread_id * 100000 + i)
                        db.set(
                            [
                                {
                                    "id": vec_id,
                                    "vector": embedding,
                                    "metadata": {"thread": thread_id},
                                }
                            ]
                        )
                        i += 1
                        time.sleep(0.001)  # Small delay
                except Exception as e:
                    with lock:
                        errors.append(f"Writer {thread_id} error: {e}")

            def reader_thread(thread_id: int, num_searches: int):
                """Search continuously"""
                try:
                    for i in range(num_searches):
                        query = generate_random_vector(64, seed=thread_id * 10000 + i)
                        results = db.search(query, k=10)
                        with lock:
                            search_results.append(len(results))
                except Exception as e:
                    with lock:
                        errors.append(f"Reader {thread_id} error: {e}")

            # Start 2 writers
            writers = []
            for i in range(2):
                t = threading.Thread(target=writer_thread, args=(i,))
                writers.append(t)
                t.start()

            # Start 5 readers doing 50 searches each
            readers = []
            for i in range(5):
                t = threading.Thread(target=reader_thread, args=(i, 50))
                readers.append(t)
                t.start()

            # Wait for readers to finish
            for t in readers:
                t.join()

            # Stop writers
            stop_flag.set()
            for t in writers:
                t.join()

            assert len(errors) == 0, f"Errors: {errors}"
            assert len(search_results) == 250  # 5 readers * 50 searches
            # All searches should return results (may be < k if db is small)
            assert all(r > 0 for r in search_results)

    def test_delete_while_searching(self):
        """Delete vectors while other threads are searching"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            vectors = generate_random_vectors(1000, 64)
            db.set(vectors)

            errors = []
            search_results = []
            lock = threading.Lock()
            stop_flag = threading.Event()

            def deleter_thread():
                """Delete vectors one by one"""
                try:
                    for i in range(500):  # Delete half
                        if stop_flag.is_set():
                            break
                        db.delete([f"vec_{i}"])
                        time.sleep(0.002)
                except Exception as e:
                    with lock:
                        errors.append(f"Deleter error: {e}")

            def reader_thread(thread_id: int, num_searches: int):
                try:
                    for i in range(num_searches):
                        query = generate_random_vector(64, seed=thread_id * 10000 + i)
                        results = db.search(query, k=10)
                        with lock:
                            search_results.append(len(results))
                except Exception as e:
                    with lock:
                        errors.append(f"Reader {thread_id} error: {e}")

            # Start deleter
            deleter = threading.Thread(target=deleter_thread)
            deleter.start()

            # Start readers
            readers = []
            for i in range(3):
                t = threading.Thread(target=reader_thread, args=(i, 100))
                readers.append(t)
                t.start()

            for t in readers:
                t.join()

            stop_flag.set()
            deleter.join()

            assert len(errors) == 0, f"Errors: {errors}"
            assert len(search_results) == 300


class TestThreadPoolExecutor:
    """Test using ThreadPoolExecutor for parallel operations"""

    def test_parallel_search_batch(self):
        """Use ThreadPoolExecutor for parallel batch search"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            vectors = generate_random_vectors(5000, 64)
            db.set(vectors)

            queries = [generate_random_vector(64, seed=i) for i in range(100)]

            def do_search(query):
                return db.search(query, k=10)

            start = time.time()
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(do_search, q) for q in queries]
                results = [f.result() for f in as_completed(futures)]
            elapsed = time.time() - start

            assert len(results) == 100
            assert all(len(r) == 10 for r in results)
            print(
                f"\nThreadPoolExecutor 8 workers, 100 queries: {elapsed:.2f}s ({100 / elapsed:.0f} QPS)"
            )

    def test_parallel_set_batches(self):
        """Use ThreadPoolExecutor for parallel batch sets"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            # Create batches
            batches = []
            for batch_id in range(10):
                batch = []
                for i in range(100):
                    vec_id = f"batch_{batch_id}_vec_{i}"
                    embedding = generate_random_vector(64, seed=batch_id * 1000 + i)
                    batch.append(
                        {
                            "id": vec_id,
                            "vector": embedding,
                            "metadata": {"batch": batch_id},
                        }
                    )
                batches.append(batch)

            def do_set(batch):
                db.set(batch)
                return len(batch)

            start = time.time()
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(do_set, b) for b in batches]
                counts = [f.result() for f in as_completed(futures)]
            elapsed = time.time() - start

            assert sum(counts) == 1000
            assert len(db) == 1000
            print(f"\nThreadPoolExecutor 4 workers, 1000 vectors: {elapsed:.2f}s")


class TestStressConditions:
    """Stress tests for edge cases in concurrency"""

    @pytest.mark.slow
    def test_high_contention_single_key(self):
        """Many threads repeatedly updating the same key"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            # Insert initial vector
            db.set(
                [
                    {
                        "id": "hot_key",
                        "vector": generate_random_vector(64, seed=0),
                        "metadata": {"counter": 0},
                    }
                ]
            )

            errors = []
            lock = threading.Lock()
            total_ops = [0]

            def hammer_key(thread_id: int, num_ops: int):
                try:
                    for i in range(num_ops):
                        # Randomly read or write
                        if random.random() < 0.5:
                            db.get("hot_key")
                        else:
                            # Generate new embedding for update
                            embedding = generate_random_vector(64, seed=thread_id * 1000 + i)
                            db.update(
                                "hot_key",
                                embedding,
                                metadata={"counter": thread_id * 1000 + i},
                            )
                        with lock:
                            total_ops[0] += 1
                except Exception as e:
                    with lock:
                        errors.append(f"Thread {thread_id} error: {e}")

            threads = []
            for i in range(20):
                t = threading.Thread(target=hammer_key, args=(i, 50))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors: {errors}"
            assert total_ops[0] == 1000
            assert len(db) == 1

    @pytest.mark.slow
    @pytest.mark.skip(
        reason="Multi-process concurrent access needs file locking investigation - not thread safety issue"
    )
    def test_rapid_open_close(self):
        """Rapidly open and close database connections"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")  # Persistent

            # Initial population
            db = omendb.open(db_path, dimensions=64)
            vectors = generate_random_vectors(100, 64)
            db.set(vectors)
            db.flush()

            errors = []
            lock = threading.Lock()

            def open_close_worker(thread_id: int, num_cycles: int):
                try:
                    for i in range(num_cycles):
                        db_local = omendb.open(db_path, dimensions=64)
                        # Do a quick operation
                        query = generate_random_vector(64, seed=thread_id * 100 + i)
                        results = db_local.search(query, k=5)
                        assert len(results) <= 5
                        # Close by letting it go out of scope
                except Exception as e:
                    with lock:
                        errors.append(f"Thread {thread_id} error: {e}")

            threads = []
            for i in range(5):
                t = threading.Thread(target=open_close_worker, args=(i, 10))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors: {errors}"

            # Verify data integrity after all the open/close cycles
            db_final = omendb.open(db_path, dimensions=64)
            assert len(db_final) == 100


class TestWriteStress:
    """Heavy write stress tests for production verification"""

    @pytest.mark.slow
    def test_concurrent_write_stress(self):
        """Stress test: many threads writing large batches with verification"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "stress_db")
            db = omendb.open(db_path, dimensions=128)

            errors = []
            lock = threading.Lock()
            num_threads = 10
            vectors_per_thread = 500
            expected_total = num_threads * vectors_per_thread

            def write_worker(thread_id: int):
                """Each thread writes vectors and verifies immediately"""
                try:
                    for batch_num in range(10):  # 10 batches of 50
                        batch = []
                        for i in range(50):
                            idx = thread_id * 500 + batch_num * 50 + i
                            vec_id = f"stress_{idx}"
                            embedding = generate_random_vector(128, seed=idx)
                            batch.append(
                                {
                                    "id": vec_id,
                                    "vector": embedding,
                                    "metadata": {"thread": thread_id, "batch": batch_num, "idx": i},
                                }
                            )
                        db.set(batch)

                        # Verify one vector from this batch
                        verify_id = batch[25]["id"]
                        result = db.get(verify_id)
                        if result is None:
                            with lock:
                                errors.append(
                                    f"Thread {thread_id}: {verify_id} missing after insert"
                                )
                except Exception as e:
                    with lock:
                        errors.append(f"Thread {thread_id} error: {e}")

            start = time.time()
            threads = []
            for i in range(num_threads):
                t = threading.Thread(target=write_worker, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()
            elapsed = time.time() - start

            assert len(errors) == 0, f"Errors during write stress: {errors}"
            assert len(db) == expected_total, f"Expected {expected_total}, got {len(db)}"

            # Verify random samples
            for _ in range(100):
                idx = random.randint(0, expected_total - 1)
                vec_id = f"stress_{idx}"
                result = db.get(vec_id)
                assert result is not None, f"Missing vector: {vec_id}"

            print(
                f"\nWrite stress: {num_threads} threads, {expected_total} vectors: {elapsed:.2f}s ({expected_total / elapsed:.0f} vec/s)"
            )

    @pytest.mark.slow
    def test_mixed_write_search_stress(self):
        """Stress test: concurrent writes and searches, verify no corruption"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "mixed_stress_db")
            db = omendb.open(db_path, dimensions=128)

            # Seed with initial data
            initial = generate_random_vectors(1000, 128)
            for v in initial:
                v["id"] = f"initial_{v['id']}"
            db.set(initial)

            errors = []
            search_results = []
            lock = threading.Lock()
            stop_flag = threading.Event()

            def writer_thread(thread_id: int):
                """Continuously write new vectors"""
                try:
                    batch_num = 0
                    while not stop_flag.is_set():
                        batch = []
                        for i in range(25):
                            idx = thread_id * 100000 + batch_num * 25 + i
                            batch.append(
                                {
                                    "id": f"writer_{thread_id}_vec_{idx}",
                                    "vector": generate_random_vector(128, seed=idx),
                                    "metadata": {"writer": thread_id},
                                }
                            )
                        db.set(batch)
                        batch_num += 1
                        time.sleep(0.005)
                except Exception as e:
                    with lock:
                        errors.append(f"Writer {thread_id} error: {e}")

            def searcher_thread(thread_id: int, num_searches: int):
                """Search and verify results make sense"""
                try:
                    for i in range(num_searches):
                        query = generate_random_vector(128, seed=thread_id * 10000 + i)
                        results = db.search(query, k=10)

                        # Results should have valid IDs
                        for r in results:
                            if "id" not in r:
                                with lock:
                                    errors.append(f"Searcher {thread_id}: result missing id")
                            if "distance" not in r:
                                with lock:
                                    errors.append(f"Searcher {thread_id}: result missing distance")

                        with lock:
                            search_results.append(len(results))
                except Exception as e:
                    with lock:
                        errors.append(f"Searcher {thread_id} error: {e}")

            # Start 4 writers
            writers = []
            for i in range(4):
                t = threading.Thread(target=writer_thread, args=(i,))
                writers.append(t)
                t.start()

            # Start 8 searchers doing 100 searches each
            searchers = []
            for i in range(8):
                t = threading.Thread(target=searcher_thread, args=(i, 100))
                searchers.append(t)
                t.start()

            # Wait for searchers
            for t in searchers:
                t.join()

            # Stop writers
            stop_flag.set()
            for t in writers:
                t.join()

            assert len(errors) == 0, f"Errors during mixed stress: {errors}"
            assert len(search_results) == 800  # 8 searchers * 100 searches
            # All searches should return results
            assert all(r > 0 for r in search_results), "Some searches returned no results"
            assert len(db) > 1000, f"Expected more than 1000, got {len(db)}"
            print(f"\nMixed stress complete: {len(db)} vectors after write+search")
