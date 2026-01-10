#!/usr/bin/env python3
"""
OmenDB Basic Operations

Demonstrates core CRUD operations:
- Creating a database
- Adding vectors with metadata
- Searching for nearest neighbors
- Retrieving, updating, and deleting vectors
- Persistence to disk
"""

import tempfile
from pathlib import Path

import omendb


def main():
    # Use a temp directory for this example
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "vectors"

        # Create database (3D vectors for readability)
        db = omendb.open(str(db_path), dimensions=3)

        # --- INSERT ---
        # Add vectors individually or in batches
        db.set(
            [
                {"id": "a", "vector": [1.0, 0.0, 0.0], "metadata": {"axis": "x"}},
                {"id": "b", "vector": [0.0, 1.0, 0.0], "metadata": {"axis": "y"}},
                {"id": "c", "vector": [0.0, 0.0, 1.0], "metadata": {"axis": "z"}},
                {"id": "d", "vector": [0.7, 0.7, 0.0], "metadata": {"axis": "xy"}},
            ]
        )
        print(f"Inserted {len(db)} vectors")

        # --- SEARCH ---
        # Find vectors similar to query
        results = db.search(query=[1.0, 0.0, 0.0], k=3)
        print("\nSearch results for [1, 0, 0]:")
        for r in results:
            print(f"  {r['id']}: distance={r['distance']:.3f}, axis={r['metadata']['axis']}")

        # --- GET ---
        # Retrieve by ID
        vec = db.get("a")
        print(f"\nGet 'a': vector={vec['vector']}, metadata={vec['metadata']}")

        # --- UPDATE ---
        # Replace embedding and/or metadata
        db.set(
            [
                {
                    "id": "a",
                    "vector": [0.9, 0.1, 0.0],
                    "metadata": {"axis": "x", "modified": True},
                }
            ]
        )
        updated = db.get("a")
        print(f"Updated 'a': {updated['metadata']}")

        # --- DELETE ---
        deleted = db.delete(["d"])
        print(f"\nDeleted {deleted} vector(s), {len(db)} remaining")

        # --- PERSISTENCE ---
        # Flush to disk (also auto-flushes on close)
        db.flush()

        # Release first db to release file lock before reopening
        del db

        # Reopen to verify persistence
        db2 = omendb.open(str(db_path), dimensions=3)
        print(f"\nReopened database: {len(db2)} vectors")

        # --- BATCH OPERATIONS ---
        # Efficient for large datasets
        import random

        batch = [
            {"id": f"rand_{i}", "vector": [random.random() for _ in range(3)]} for i in range(100)
        ]
        db2.set(batch)
        print(f"After batch insert: {len(db2)} vectors")

        # Batch search
        queries = [[random.random() for _ in range(3)] for _ in range(10)]
        results = db2.search_batch(queries, k=5)
        print(f"Batch search: {len(results)} result sets, {len(results[0])} results each")


if __name__ == "__main__":
    main()
