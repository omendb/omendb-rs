#!/usr/bin/env python3
"""
Basic OmenDB usage example

Demonstrates:
- Opening/creating a database
- Inserting vectors with metadata
- Searching for nearest neighbors
- Retrieving vectors by ID
- Updating and deleting vectors
- Persisting to disk
"""

import omendb
import random
import shutil
from pathlib import Path

def main():
    # Clean up any previous database
    db_path = "./example_db"
    if Path(db_path).exists():
        shutil.rmtree(db_path)

    print("=== OmenDB Basic Usage Example ===\n")

    # 1. Open or create database (3D vectors for this example)
    print("1. Opening database...")
    db = omendb.open(db_path, dimensions=3)
    print(f"   Database opened at: {db_path}")
    print(f"   Current size: {len(db)} vectors\n")

    # 2. Insert vectors with metadata
    print("2. Inserting vectors...")

    # Create some sample 3D vectors
    vectors = [
        {
            "id": "vec1",
            "embedding": [1.0, 0.0, 0.0],
            "metadata": {"label": "x-axis", "type": "unit"}
        },
        {
            "id": "vec2",
            "embedding": [0.0, 1.0, 0.0],
            "metadata": {"label": "y-axis", "type": "unit"}
        },
        {
            "id": "vec3",
            "embedding": [0.0, 0.0, 1.0],
            "metadata": {"label": "z-axis", "type": "unit"}
        },
        {
            "id": "vec4",
            "embedding": [0.707, 0.707, 0.0],
            "metadata": {"label": "diagonal xy", "type": "mixed"}
        },
        {
            "id": "vec5",
            "embedding": [0.577, 0.577, 0.577],
            "metadata": {"label": "diagonal xyz", "type": "mixed"}
        },
    ]

    indices = db.set(vectors)
    print(f"   Inserted {len(indices)} vectors")
    print(f"   Database size: {len(db)} vectors\n")

    # 3. Search for nearest neighbors
    print("3. Searching for nearest neighbors...")

    # Query: find vectors similar to [1, 0, 0] (x-axis)
    query = [1.0, 0.0, 0.0]
    results = db.search(query=query, k=3)

    print(f"   Query: {query}")
    print(f"   Top 3 results:")
    for i, result in enumerate(results, 1):
        print(f"      {i}. {result['id']}: distance={result['distance']:.3f}, metadata={result['metadata']}")
    print()

    # 4. Get vector by ID
    print("4. Retrieving vector by ID...")
    vec = db.get("vec1")
    print(f"   ID: vec1")
    print(f"   Embedding: {vec['embedding']}")
    print(f"   Metadata: {vec['metadata']}\n")

    # 5. Update vector
    print("5. Updating vector...")
    db.update("vec1", embedding=[0.9, 0.1, 0.0], metadata={"label": "slightly off x-axis", "type": "unit"})

    updated = db.get("vec1")
    print(f"   Updated embedding: {updated['embedding']}")
    print(f"   Updated metadata: {updated['metadata']}\n")

    # 6. Search with the updated vector
    print("6. Searching again after update...")
    results = db.search(query=query, k=3)
    print(f"   Query: {query}")
    print(f"   Top 3 results:")
    for i, result in enumerate(results, 1):
        print(f"      {i}. {result['id']}: distance={result['distance']:.3f}")
    print()

    # 7. Delete vectors
    print("7. Deleting vectors...")
    deleted_count = db.delete(["vec4", "vec5"])
    print(f"   Deleted {deleted_count} vectors")
    print(f"   Database size: {len(db)} vectors\n")

    # 8. Save to disk
    print("8. Persisting to disk...")
    db.save()
    print(f"   Database saved to: {db_path}\n")

    # 9. Re-open database to verify persistence
    print("9. Re-opening database to verify persistence...")
    db2 = omendb.open(db_path, dimensions=3)
    print(f"   Database size after re-open: {len(db2)} vectors")

    # Verify data persisted correctly
    vec_check = db2.get("vec1")
    print(f"   vec1 embedding: {vec_check['embedding']}")
    print(f"   vec1 metadata: {vec_check['metadata']}\n")

    # 10. Batch set (efficient for large datasets)
    print("10. Batch set (adding 100 random vectors)...")

    batch = []
    for i in range(100):
        batch.append({
            "id": f"random_{i}",
            "embedding": [random.random(), random.random(), random.random()],
            "metadata": {"batch": "random", "index": i}
        })

    db2.set(batch)
    print(f"    Database size: {len(db2)} vectors\n")

    # Search in larger database
    print("11. Searching in larger database...")
    results = db2.search(query=[0.5, 0.5, 0.5], k=5)
    print(f"    Query: [0.5, 0.5, 0.5]")
    print(f"    Top 5 results:")
    for i, result in enumerate(results, 1):
        print(f"       {i}. {result['id']}: distance={result['distance']:.3f}")

    print("\n=== Example complete! ===")
    print(f"Database location: {db_path}")
    print(f"Final size: {len(db2)} vectors")

if __name__ == "__main__":
    main()
