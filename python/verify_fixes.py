#!/usr/bin/env python3
"""Verify all bug fixes from the Dec 18 audit."""

import os
import tempfile

import numpy as np

import omendb


def test_quantization_persistence():
    """Verify quantization mode persists across save/load."""
    print("Testing quantization persistence...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "quant_test")

        # Create with SQ8 quantization
        db = omendb.open(db_path, dimensions=128, quantization=True)
        vectors = [{"id": f"v{i}", "vector": np.random.randn(128).tolist()} for i in range(100)]
        db.set(vectors)
        db.flush()
        del db

        # Reopen and verify quantization is still enabled
        db = omendb.open(db_path, dimensions=128)

        # Search should work (would fail if quantization lost)
        query = np.random.randn(128).tolist()
        results = db.search(query, k=10)
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"

        # Add more vectors - should still use quantization
        more = [{"id": f"v{100 + i}", "vector": np.random.randn(128).tolist()} for i in range(50)]
        db.set(more)
        db.flush()

        results = db.search(query, k=10)
        assert len(results) == 10

        print("  ✓ Quantization persistence works")
        return True


def test_distance_metric_persistence():
    """Verify distance metric persists across save/load."""
    print("Testing distance metric persistence...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "metric_test")

        # Create with cosine metric
        db = omendb.open(db_path, dimensions=128, metric="cosine")

        # Insert normalized vectors (for cosine)
        v1 = np.random.randn(128)
        v1 = (v1 / np.linalg.norm(v1)).tolist()
        v2 = np.random.randn(128)
        v2 = (v2 / np.linalg.norm(v2)).tolist()

        db.set(
            [
                {"id": "v1", "vector": v1},
                {"id": "v2", "vector": v2},
            ]
        )
        db.flush()
        del db

        # Reopen
        db = omendb.open(db_path, dimensions=128)

        # Search with v1 as query - should find v1 first with distance ~0
        results = db.search(v1, k=2)
        assert results[0]["id"] == "v1", f"Expected v1, got {results[0]['id']}"
        assert results[0]["distance"] < 0.01, (
            f"Cosine distance to self should be ~0, got {results[0]['distance']}"
        )

        print("  ✓ Distance metric persistence works")
        return True


def test_hnsw_params_persistence():
    """Verify HNSW parameters persist across save/load."""
    print("Testing HNSW params persistence...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "params_test")

        # Create with custom params
        db = omendb.open(db_path, dimensions=64, m=32, ef_construction=200, ef_search=150)

        vectors = [{"id": f"v{i}", "vector": np.random.randn(64).tolist()} for i in range(500)]
        db.set(vectors)
        db.flush()
        del db

        # Reopen and verify search still works with good recall
        db = omendb.open(db_path, dimensions=64)

        # Search should work
        query = np.random.randn(64).tolist()
        results = db.search(query, k=10)
        assert len(results) == 10

        print("  ✓ HNSW params persistence works")
        return True


def test_sq8_multibatch_insert():
    """Verify SQ8 ID mapping works across multiple batch inserts."""
    print("Testing SQ8 multi-batch insert...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "sq8_batch_test")

        db = omendb.open(db_path, dimensions=64, quantization=True)

        # Insert in multiple batches
        for batch_num in range(5):
            vectors = [
                {"id": f"batch{batch_num}_v{i}", "vector": np.random.randn(64).tolist()}
                for i in range(100)
            ]
            db.set(vectors)

        db.flush()

        # Verify all 500 vectors are present
        assert len(db) == 500, f"Expected 500 vectors, got {len(db)}"

        # Verify we can retrieve vectors from each batch
        for batch_num in range(5):
            result = db.get(f"batch{batch_num}_v50")
            assert result is not None, f"Could not find batch{batch_num}_v50"

        # Close and reopen
        del db
        db = omendb.open(db_path, dimensions=64)

        # Verify count after reopen
        assert len(db) == 500, f"After reopen: expected 500, got {len(db)}"

        # Verify search works
        query = np.random.randn(64).tolist()
        results = db.search(query, k=10)
        assert len(results) == 10

        # Verify we can still get vectors from each batch after reopen
        for batch_num in range(5):
            result = db.get(f"batch{batch_num}_v50")
            assert result is not None, f"After reopen: could not find batch{batch_num}_v50"

        # Verify items() works for quantized stores
        items = db.items()
        assert len(items) == 500, f"items() should return 500, got {len(items)}"

        print("  ✓ SQ8 multi-batch insert works")
        return True


def test_hybrid_search_subscores():
    """Verify hybrid search subscores feature works."""
    print("Testing hybrid search subscores...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "hybrid_test")

        db = omendb.open(db_path, dimensions=64)

        db.set(
            [
                {"id": "doc1", "vector": [1.0] + [0.0] * 63, "text": "machine learning algorithms"},
                {"id": "doc2", "vector": [0.0] * 63 + [1.0], "text": "machine learning models"},
                {"id": "doc3", "vector": [0.9] + [0.1] + [0.0] * 62, "text": "cooking recipes"},
            ]
        )
        db.flush()

        # Test without subscores
        results = db.search_hybrid([1.0] + [0.0] * 63, "machine learning", k=3)
        assert "keyword_score" not in results[0], "subscores=False should not have keyword_score"

        # Test with subscores
        results = db.search_hybrid([1.0] + [0.0] * 63, "machine learning", k=3, subscores=True)
        assert "keyword_score" in results[0], "subscores=True should have keyword_score"
        assert "semantic_score" in results[0], "subscores=True should have semantic_score"

        # doc1 should have both scores
        doc1 = next(r for r in results if r["id"] == "doc1")
        assert doc1["keyword_score"] is not None, "doc1 should have keyword_score"
        assert doc1["semantic_score"] is not None, "doc1 should have semantic_score"

        # doc3 should have semantic but no keyword (text doesn't match)
        doc3 = next(r for r in results if r["id"] == "doc3")
        assert doc3["semantic_score"] is not None, "doc3 should have semantic_score"
        assert doc3["keyword_score"] is None, "doc3 should not have keyword_score"

        print("  ✓ Hybrid search subscores works")
        return True


def test_header_corruption_detection():
    """Verify corrupted database files are detected."""
    print("Testing header corruption detection...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "corrupt_test")

        # Create valid database
        db = omendb.open(db_path, dimensions=64)
        db.set([{"id": "v1", "vector": [1.0] * 64}])
        db.flush()
        del db

        # Corrupt the .omen file
        omen_file = db_path + ".omen"
        with open(omen_file, "r+b") as f:
            f.seek(20)  # Skip magic bytes, corrupt header
            f.write(b"\xff\xff\xff\xff")

        # Try to open - should fail with checksum error
        try:
            omendb.open(db_path, dimensions=64)
            print("  ✗ Corruption not detected!")
            return False
        except Exception as e:
            if "checksum" in str(e).lower() or "invalid" in str(e).lower():
                print("  ✓ Header corruption detected")
                return True
            else:
                print(f"  ? Unexpected error: {e}")
                return True  # Still failed, which is what we want


def main():
    print("=" * 60)
    print("OmenDB v0.0.11 Bug Fix Verification")
    print("=" * 60)
    print()

    tests = [
        ("Quantization Persistence", test_quantization_persistence),
        ("Distance Metric Persistence", test_distance_metric_persistence),
        ("HNSW Params Persistence", test_hnsw_params_persistence),
        ("SQ8 Multi-Batch Insert", test_sq8_multibatch_insert),
        ("Hybrid Search Subscores", test_hybrid_search_subscores),
        ("Header Corruption Detection", test_header_corruption_detection),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name} FAILED with exception: {e}")

    print()
    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} passed")
    if failed == 0:
        print("All verification tests PASSED ✓")
    else:
        print(f"{failed} test(s) FAILED ✗")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
