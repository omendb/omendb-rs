"""1M vector scale test for production hardening"""

import math
import os
import random
import tempfile
import time

import pytest

import omendb

pytestmark = pytest.mark.slow  # Skip in CI


def generate_random_vectors(n, dim, seed=42):
    random.seed(seed)
    vectors = []
    for i in range(n):
        embedding = [random.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in embedding))
        embedding = [x / norm for x in embedding]
        vectors.append({"id": f"vec_{i}", "vector": embedding, "metadata": {"index": i}})
    return vectors


def main():
    print("=" * 60)
    print("1M VECTOR SCALE TEST")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_1m")
        dims = 128

        print(f"\n1. Creating database with {dims} dimensions...")
        db = omendb.open(db_path, dimensions=dims)

        # Insert 1M vectors in batches
        total_vectors = 1_000_000
        batch_size = 10_000

        print(f"\n2. Inserting {total_vectors:,} vectors in batches of {batch_size:,}...")
        start = time.time()

        for batch_num in range(total_vectors // batch_size):
            batch_start = batch_num * batch_size
            vectors = generate_random_vectors(batch_size, dims, seed=batch_start)
            # Renumber IDs to be unique
            for i, v in enumerate(vectors):
                v["id"] = f"vec_{batch_start + i}"
            db.set(vectors)

            if (batch_num + 1) % 10 == 0:
                elapsed = time.time() - start
                inserted = (batch_num + 1) * batch_size
                rate = inserted / elapsed
                print(f"   {inserted:>8,} vectors ({rate:,.0f} vec/s)")

        insert_time = time.time() - start
        insert_rate = total_vectors / insert_time
        print(
            f"\n   INSERT COMPLETE: {total_vectors:,} in {insert_time:.1f}s ({insert_rate:,.0f} vec/s)"
        )

        # Test search performance
        print("\n3. Testing search performance...")
        db.set_ef_search(100)

        num_queries = 100
        start = time.time()
        for q in range(num_queries):
            query = generate_random_vectors(1, dims, seed=9000 + q)[0]["vector"]
            _ = db.search(query, k=10)
        search_time = time.time() - start
        qps = num_queries / search_time
        print(f"   SEARCH: {num_queries} queries in {search_time:.2f}s ({qps:,.0f} QPS)")

        # Test persistence
        print("\n4. Testing persistence (save/load)...")
        start = time.time()
        db.flush()
        save_time = time.time() - start
        print(f"   SAVE: {save_time:.1f}s")

        del db

        start = time.time()
        db2 = omendb.open(db_path, dimensions=dims)
        load_time = time.time() - start
        print(f"   LOAD: {load_time:.1f}s")

        # Verify count
        count = len(db2)
        print(f"   VERIFIED: {count:,} vectors loaded")
        assert count == total_vectors, f"Expected {total_vectors}, got {count}"

        # Test search after load
        start = time.time()
        for q in range(num_queries):
            query = generate_random_vectors(1, dims, seed=9000 + q)[0]["vector"]
            _ = db2.search(query, k=10)
        search_time_2 = time.time() - start
        qps_2 = num_queries / search_time_2
        print(f"   SEARCH AFTER LOAD: {qps_2:,.0f} QPS")

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Vectors:      {total_vectors:,}")
        print(f"Dimensions:   {dims}")
        print(f"Insert rate:  {insert_rate:,.0f} vec/s")
        print(f"Search QPS:   {qps:,.0f}")
        print(f"Save time:    {save_time:.1f}s")
        print(f"Load time:    {load_time:.1f}s")
        print("=" * 60)
        print("✅ 1M SCALE TEST PASSED")


if __name__ == "__main__":
    main()
