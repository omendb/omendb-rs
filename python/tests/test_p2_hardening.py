"""P2 Production Hardening Tests

These tests verify edge cases and potential failure modes:
1. GIL deadlock scenarios
2. Large metadata handling (1MB+)
3. File handle leaks (rapid open/close)
4. Long-running soak test

Run with: pytest python/tests/test_p2_hardening.py -v
Run soak test: pytest python/tests/test_p2_hardening.py -v -m soak --timeout=600
"""

import gc
import math
import os
import random
import tempfile
import threading
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from concurrent.futures import (
    TimeoutError as FuturesTimeoutError,
)

import pytest

import omendb


def generate_random_vector(dim: int, seed: int = None) -> list:
    """Generate a random normalized vector"""
    if seed is not None:
        random.seed(seed)
    embedding = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in embedding))
    return [x / norm for x in embedding]


class TestGILDeadlock:
    """Test scenarios that could trigger GIL deadlock.

    PyO3 + RwLock interaction can deadlock if:
    - GIL is held while waiting for Rust lock
    - Rust code holds lock while calling back to Python
    - Thread starvation under high contention
    """

    def test_rapid_concurrent_calls(self):
        """Many threads calling OmenDB concurrently - stress GIL/lock interaction"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test")
            db = omendb.open(db_path, dimensions=64)

            # Pre-populate
            vectors = [
                {
                    "id": f"v{i}",
                    "vector": generate_random_vector(64, i),
                    "metadata": {"i": i},
                }
                for i in range(100)
            ]
            db.set(vectors)

            errors = []
            lock = threading.Lock()
            operations_completed = [0]

            def rapid_operations(thread_id: int, num_ops: int):
                """Rapidly switch between different operation types"""
                try:
                    for i in range(num_ops):
                        op = (thread_id + i) % 4
                        if op == 0:
                            db.search(generate_random_vector(64, thread_id * 1000 + i), k=5)
                        elif op == 1:
                            db.get(f"v{i % 100}")
                        elif op == 2:
                            len(db)
                        elif op == 3:
                            db.get_ef_search()
                        with lock:
                            operations_completed[0] += 1
                except Exception as e:
                    with lock:
                        errors.append(f"Thread {thread_id} error: {e}")

            # Use many threads to maximize contention
            num_threads = 20
            ops_per_thread = 100

            threads = []
            start = time.time()
            for i in range(num_threads):
                t = threading.Thread(target=rapid_operations, args=(i, ops_per_thread))
                threads.append(t)
                t.start()

            # Set a timeout to detect deadlocks
            deadline = start + 30  # 30 second timeout
            for t in threads:
                remaining = max(0.1, deadline - time.time())
                t.join(timeout=remaining)
                if t.is_alive():
                    errors.append("Thread timed out - possible deadlock")

            elapsed = time.time() - start

            # Check for errors
            assert len(errors) == 0, f"Errors (possible deadlock): {errors}"
            assert operations_completed[0] == num_threads * ops_per_thread
            print(
                f"\n{operations_completed[0]} ops in {elapsed:.2f}s ({operations_completed[0] / elapsed:.0f} ops/s)"
            )

    def test_threadpool_high_contention(self):
        """ThreadPoolExecutor with many workers hammering same db"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test")
            db = omendb.open(db_path, dimensions=64)

            vectors = [
                {
                    "id": f"v{i}",
                    "vector": generate_random_vector(64, i),
                    "metadata": {"i": i},
                }
                for i in range(500)
            ]
            db.set(vectors)

            def do_work(task_id):
                """Mix of read/write operations"""
                for _ in range(10):
                    query = generate_random_vector(64, task_id)
                    _ = db.search(query, k=5)
                    db.get(f"v{task_id % 500}")
                return task_id

            # Use more workers than CPU cores to stress scheduler
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = [executor.submit(do_work, i) for i in range(100)]

                completed = []
                try:
                    for f in as_completed(futures, timeout=30):
                        completed.append(f.result())
                except FuturesTimeoutError:
                    pytest.fail("ThreadPool timeout - possible deadlock")

            assert len(completed) == 100

    def test_mixed_batch_and_single_ops(self):
        """Interleave search_batch with single operations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test")
            db = omendb.open(db_path, dimensions=64)

            vectors = [
                {
                    "id": f"v{i}",
                    "vector": generate_random_vector(64, i),
                    "metadata": {"i": i},
                }
                for i in range(1000)
            ]
            db.set(vectors)

            errors = []
            lock = threading.Lock()

            def search_batcher(thread_id: int):
                try:
                    for _ in range(5):
                        queries = [
                            generate_random_vector(64, thread_id * 100 + j) for j in range(20)
                        ]
                        results = db.search_batch(queries, k=5)
                        assert len(results) == 20
                except Exception as e:
                    with lock:
                        errors.append(f"Batch searcher {thread_id}: {e}")

            def single_searcher(thread_id: int):
                try:
                    for i in range(100):
                        db.search(generate_random_vector(64, thread_id * 1000 + i), k=5)
                except Exception as e:
                    with lock:
                        errors.append(f"Single searcher {thread_id}: {e}")

            threads = []
            for i in range(5):
                threads.append(threading.Thread(target=search_batcher, args=(i,)))
            for i in range(10):
                threads.append(threading.Thread(target=single_searcher, args=(i + 5,)))

            start = time.time()
            for t in threads:
                t.start()

            for t in threads:
                t.join(timeout=30)
                if t.is_alive():
                    errors.append("Thread timed out")

            assert len(errors) == 0, f"Errors: {errors}"
            print(f"\nMixed batch/single: {time.time() - start:.2f}s")


class TestLargeMetadata:
    """Test handling of large metadata values (1MB+)"""

    def test_1mb_metadata_string(self):
        """Store and retrieve 1MB string in metadata"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test")
            db = omendb.open(db_path, dimensions=64)

            # Create 1MB string
            large_string = "x" * (1024 * 1024)  # 1MB

            db.set(
                [
                    {
                        "id": "large1",
                        "vector": generate_random_vector(64, 0),
                        "metadata": {
                            "content": large_string,
                            "size": len(large_string),
                        },
                    }
                ]
            )

            result = db.get("large1")
            assert result is not None
            assert len(result["metadata"]["content"]) == 1024 * 1024
            assert result["metadata"]["size"] == 1024 * 1024

            # Search should still work
            results = db.search(generate_random_vector(64, 0), k=1)
            assert len(results) == 1
            assert results[0]["id"] == "large1"

    @pytest.mark.slow
    def test_10mb_metadata_string(self):
        """Store and retrieve 10MB string in metadata"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test")
            db = omendb.open(db_path, dimensions=64)

            # Create 10MB string
            large_string = "y" * (10 * 1024 * 1024)  # 10MB

            db.set(
                [
                    {
                        "id": "huge1",
                        "vector": generate_random_vector(64, 0),
                        "metadata": {"content": large_string},
                    }
                ]
            )

            result = db.get("huge1")
            assert result is not None
            assert len(result["metadata"]["content"]) == 10 * 1024 * 1024

    def test_large_metadata_array(self):
        """Store large array in metadata"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test")
            db = omendb.open(db_path, dimensions=64)

            # Create array with 100K elements
            large_array = list(range(100_000))

            db.set(
                [
                    {
                        "id": "array1",
                        "vector": generate_random_vector(64, 0),
                        "metadata": {"data": large_array},
                    }
                ]
            )

            result = db.get("array1")
            assert result is not None
            assert len(result["metadata"]["data"]) == 100_000
            assert result["metadata"]["data"][50_000] == 50_000

    def test_deep_nested_metadata(self):
        """Deeply nested metadata structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test")
            db = omendb.open(db_path, dimensions=64)

            # Create deeply nested dict (20 levels)
            def create_nested(depth, value):
                if depth == 0:
                    return value
                return {"level": depth, "child": create_nested(depth - 1, value)}

            nested = create_nested(20, "deep_value")

            db.set(
                [
                    {
                        "id": "nested1",
                        "vector": generate_random_vector(64, 0),
                        "metadata": nested,
                    }
                ]
            )

            result = db.get("nested1")
            assert result is not None

            # Traverse to verify
            current = result["metadata"]
            for i in range(20, 0, -1):
                assert current["level"] == i
                current = current["child"]
            assert current == "deep_value"

    def test_many_large_metadata_vectors(self):
        """Multiple vectors with large metadata"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test")
            db = omendb.open(db_path, dimensions=64)

            # 100 vectors with 100KB metadata each = ~10MB total
            vectors = []
            for i in range(100):
                vectors.append(
                    {
                        "id": f"v{i}",
                        "vector": generate_random_vector(64, i),
                        "metadata": {"content": "x" * 100_000, "idx": i},
                    }
                )

            start = time.time()
            db.set(vectors)
            insert_time = time.time() - start

            assert len(db) == 100

            # Verify retrieval
            result = db.get("v50")
            assert len(result["metadata"]["content"]) == 100_000

            print(
                f"\n100 vectors with 100KB metadata: {insert_time:.2f}s ({100 / insert_time:.0f} vec/s)"
            )


class TestFileHandleLeaks:
    """Test for file handle leaks during rapid open/close cycles"""

    def test_rapid_open_close_sequential(self):
        """Sequential rapid open/close cycles"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test")

            # Initial population
            db = omendb.open(db_path, dimensions=64)
            vectors = [
                {
                    "id": f"v{i}",
                    "vector": generate_random_vector(64, i),
                    "metadata": {"i": i},
                }
                for i in range(100)
            ]
            db.set(vectors)
            db.flush()  # Required for persistence
            del db
            gc.collect()

            # Rapid open/close cycles
            errors = []
            for cycle in range(50):
                try:
                    db = omendb.open(db_path, dimensions=64)
                    # Quick operation
                    results = db.search(generate_random_vector(64, cycle), k=5)
                    assert len(results) <= 5
                    count = len(db)
                    assert count == 100
                    del db
                    gc.collect()
                except Exception as e:
                    errors.append(f"Cycle {cycle}: {e}")

            assert len(errors) == 0, f"Errors: {errors}"
            print("\n50 sequential open/close cycles: OK")

    def test_rapid_open_close_with_writes(self):
        """Open/close with write operations each time"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test")

            errors = []
            expected_count = 0

            for cycle in range(30):
                try:
                    db = omendb.open(db_path, dimensions=64)

                    # Add some vectors
                    vectors = [
                        {
                            "id": f"c{cycle}_v{i}",
                            "vector": generate_random_vector(64, cycle * 10 + i),
                        }
                        for i in range(10)
                    ]
                    db.set(vectors)
                    expected_count += 10

                    # Verify count
                    assert len(db) == expected_count, f"Expected {expected_count}, got {len(db)}"

                    db.flush()  # Required for persistence
                    del db
                    gc.collect()
                except Exception as e:
                    errors.append(f"Cycle {cycle}: {e}")

            assert len(errors) == 0, f"Errors: {errors}"

            # Final verification
            db = omendb.open(db_path, dimensions=64)
            assert len(db) == expected_count
            print(f"\n30 open/close cycles with writes: {expected_count} vectors persisted")

    def test_concurrent_open_different_paths(self):
        """Multiple databases open concurrently (different paths)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            num_dbs = 10
            dbs = []

            # Open multiple databases concurrently
            for i in range(num_dbs):
                db_path = os.path.join(tmpdir, f"db{i}")
                db = omendb.open(db_path, dimensions=64)
                db.set(
                    [
                        {
                            "id": f"v{i}",
                            "vector": generate_random_vector(64, i),
                            "metadata": {"db": i},
                        }
                    ]
                )
                dbs.append(db)

            # Verify all work
            for i, db in enumerate(dbs):
                result = db.get(f"v{i}")
                assert result is not None
                assert result["metadata"]["db"] == i

            # Flush and clean up
            for db in dbs:
                db.flush()  # Required for persistence
            dbs.clear()  # Release all references
            gc.collect()

            # Verify all still accessible after reopening
            for i in range(num_dbs):
                db_path = os.path.join(tmpdir, f"db{i}")
                db = omendb.open(db_path, dimensions=64)
                assert len(db) == 1
                del db

            print(f"\n{num_dbs} concurrent databases: OK")


class TestSoakTest:
    """Long-running soak test for memory leaks and stability"""

    @pytest.mark.soak
    @pytest.mark.timeout(600)  # 10 minute timeout
    def test_soak_5_minutes(self):
        """5-minute soak test with continuous operations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "soak")
            db = omendb.open(db_path, dimensions=64)

            # Initial population
            vectors = [
                {
                    "id": f"v{i}",
                    "vector": generate_random_vector(64, i),
                    "metadata": {"i": i},
                }
                for i in range(1000)
            ]
            db.set(vectors)

            duration = 300  # 5 minutes
            start = time.time()
            ops = {"search": 0, "set": 0, "delete": 0, "get": 0}
            errors = []
            vector_id_counter = 1000

            print("\nStarting 5-minute soak test...")

            while time.time() - start < duration:
                elapsed = time.time() - start

                # Progress indicator every 30 seconds
                if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                    total_ops = sum(ops.values())
                    print(f"  {int(elapsed)}s: {total_ops} ops, {len(db)} vectors")

                try:
                    # Random operation mix
                    op = random.choice(["search", "search", "search", "set", "delete", "get"])

                    if op == "search":
                        query = generate_random_vector(64)
                        _ = db.search(query, k=10)
                        ops["search"] += 1

                    elif op == "set":
                        # Add a new vector
                        db.set(
                            [
                                {
                                    "id": f"soak_{vector_id_counter}",
                                    "vector": generate_random_vector(64),
                                    "metadata": {"soak": True, "ts": time.time()},
                                }
                            ]
                        )
                        vector_id_counter += 1
                        ops["set"] += 1

                    elif op == "delete":
                        # Delete a random soak vector (if any exist)
                        if vector_id_counter > 1001:
                            delete_id = random.randint(1000, vector_id_counter - 1)
                            db.delete([f"soak_{delete_id}"])
                        ops["delete"] += 1

                    elif op == "get":
                        # Get a random vector
                        get_id = f"v{random.randint(0, 999)}"
                        db.get(get_id)
                        ops["get"] += 1

                except Exception as e:
                    errors.append(f"At {elapsed:.1f}s: {e}")
                    if len(errors) > 100:
                        pytest.fail(f"Too many errors: {errors[:10]}...")

                # Small delay to prevent CPU saturation
                time.sleep(0.001)

            total_time = time.time() - start
            total_ops = sum(ops.values())

            print("\nSoak test complete:")
            print(f"  Duration: {total_time:.1f}s")
            print(f"  Total ops: {total_ops} ({total_ops / total_time:.0f} ops/s)")
            print(f"  Breakdown: {ops}")
            print(f"  Final count: {len(db)} vectors")
            print(f"  Errors: {len(errors)}")

            assert len(errors) < 10, f"Too many errors: {errors}"

    @pytest.mark.soak
    @pytest.mark.timeout(120)  # 2-minute timeout for 30-second test
    def test_soak_memory_stability(self):
        """30-second memory stability test"""
        import tracemalloc

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "mem")
            db = omendb.open(db_path, dimensions=64)

            # Initial population
            vectors = [{"id": f"v{i}", "vector": generate_random_vector(64, i)} for i in range(500)]
            db.set(vectors)

            # Force garbage collection to establish baseline
            gc.collect()
            tracemalloc.start()
            initial_snapshot = tracemalloc.take_snapshot()

            duration = 30  # 30 seconds
            start = time.time()
            ops = 0

            while time.time() - start < duration:
                # Continuous operations
                for _ in range(100):
                    db.search(generate_random_vector(64), k=10)
                    db.get(f"v{random.randint(0, 499)}")
                    ops += 2

                # Periodic GC
                gc.collect()

            # Take final snapshot
            final_snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()

            # Compare memory
            top_stats = final_snapshot.compare_to(initial_snapshot, "lineno")

            # Calculate total memory growth
            total_growth = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
            total_growth_mb = total_growth / (1024 * 1024)

            print("\nMemory stability test:")
            print(f"  Duration: {time.time() - start:.1f}s")
            print(f"  Operations: {ops}")
            print(f"  Memory growth: {total_growth_mb:.2f}MB")

            # Allow some memory growth but flag if excessive (>50MB)
            assert total_growth_mb < 50, f"Excessive memory growth: {total_growth_mb:.2f}MB"


# Custom marker for soak tests
def pytest_configure(config):
    config.addinivalue_line("markers", "soak: mark test as a soak test (long-running)")
