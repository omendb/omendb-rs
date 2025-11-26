#!/usr/bin/env python3
"""Basic test for OmenDB Python bindings"""

import omendb
import os
import tempfile

def test_basic_operations():
    """Test basic CRUD operations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        # Test 1: Create database
        print("Test 1: Creating database...")
        db = omendb.open(db_path, dimensions=128)
        print("  ✓ Database created")

        # Test 2: Insert vectors
        print("\nTest 2: Inserting vectors...")
        vectors = [
            {"id": "vec1", "embedding": [0.1] * 128, "metadata": {"label": "A"}},
            {"id": "vec2", "embedding": [0.2] * 128, "metadata": {"label": "B"}},
            {"id": "vec3", "embedding": [0.3] * 128, "metadata": {"label": "C"}},
        ]
        ids = db.set(vectors)
        print(f"  ✓ Inserted {len(ids)} vectors")

        # Test 3: Search
        print("\nTest 3: Searching...")
        query = [0.15] * 128
        results = db.search(query, k=2)
        print(f"  ✓ Found {len(results)} results")
        for i, result in enumerate(results):
            print(f"    {i+1}. id={result['id']}, distance={result['distance']:.4f}, metadata={result['metadata']}")

        # Test 4: Delete
        print("\nTest 4: Deleting vectors...")
        db.delete(["vec2"])
        print("  ✓ Deleted vec2")

        # Test 5: Search after delete
        print("\nTest 5: Searching after delete...")
        results = db.search(query, k=3)
        print(f"  ✓ Found {len(results)} results (should be 2)")
        for i, result in enumerate(results):
            print(f"    {i+1}. id={result['id']}, distance={result['distance']:.4f}")
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"

        # Test 6: Save and reload
        print("\nTest 6: Save and reload...")
        db.save()
        print("  ✓ Saved database")

        db2 = omendb.open(db_path, dimensions=128)
        print(f"  Database has {len(db2)} vectors")
        results = db2.search(query, k=3)
        print(f"  ✓ Reloaded database, found {len(results)} results")
        for i, result in enumerate(results):
            print(f"    {i+1}. id={result['id']}, distance={result['distance']:.4f}")
        assert len(results) == 2, f"Expected 2 results after reload, got {len(results)}"

        print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_basic_operations()
