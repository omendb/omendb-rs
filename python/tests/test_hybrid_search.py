"""Tests for hybrid search (vector + text) functionality"""

import os
import tempfile

import omendb


def test_enable_text_search():
    """Test enabling text search on a database"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)

        assert not db.has_text_search()
        db.enable_text_search()
        assert db.has_text_search()
        del db  # Ensure cleanup before temp dir removal


def test_set_with_text_auto_enables():
    """Test that set() with text field auto-enables text search"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)

        assert not db.has_text_search()

        count = db.set(
            [
                {
                    "id": "doc1",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "text": "Machine learning is a subset of artificial intelligence",
                    "metadata": {"category": "tech"},
                },
                {
                    "id": "doc2",
                    "vector": [0.0, 1.0, 0.0, 0.0],
                    "text": "Deep learning uses neural networks for pattern recognition",
                    "metadata": {"category": "tech"},
                },
            ]
        )

        db.flush()

        assert count == 2
        assert len(db) == 2
        assert db.has_text_search()  # Auto-enabled


def test_text_search():
    """Test pure text (BM25) search"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)

        db.set(
            [
                {
                    "id": "doc1",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "text": "Python programming language",
                },
                {
                    "id": "doc2",
                    "vector": [0.0, 1.0, 0.0, 0.0],
                    "text": "JavaScript web development",
                },
                {
                    "id": "doc3",
                    "vector": [0.0, 0.0, 1.0, 0.0],
                    "text": "Python data science machine learning",
                },
            ]
        )
        db.flush()

        results = db.search_text("Python", k=10)

        assert len(results) == 2
        ids = [r["id"] for r in results]
        assert "doc1" in ids and "doc3" in ids

        for r in results:
            assert "id" in r
            assert "score" in r
            assert r["score"] > 0
            # search_text now returns metadata
            assert "metadata" in r
            assert "text" in r["metadata"]


def test_update_text():
    """Test updating text re-indexes for BM25 search"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)

        # Insert with text
        db.set(
            [
                {
                    "id": "doc1",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "text": "Python programming language",
                },
            ]
        )
        db.flush()

        # Verify initial search works
        results = db.search_text("Python", k=10)
        assert len(results) == 1
        assert results[0]["id"] == "doc1"

        # Update text
        db.update("doc1", text="JavaScript web development")
        db.flush()

        # Old text should not match
        results = db.search_text("Python", k=10)
        assert len(results) == 0

        # New text should match
        results = db.search_text("JavaScript", k=10)
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["metadata"]["text"] == "JavaScript web development"


def test_hybrid_search_basic():
    """Test basic hybrid search (vector + text)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)

        db.set(
            [
                {
                    "id": "doc1",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "text": "Machine learning algorithms",
                    "metadata": {"type": "ml"},
                },
                {
                    "id": "doc2",
                    "vector": [0.9, 0.1, 0.0, 0.0],
                    "text": "Deep learning neural networks",
                    "metadata": {"type": "dl"},
                },
                {
                    "id": "doc3",
                    "vector": [0.0, 1.0, 0.0, 0.0],
                    "text": "Web development frameworks",
                    "metadata": {"type": "web"},
                },
            ]
        )
        db.flush()

        results = db.search_hybrid(query_vector=[1.0, 0.0, 0.0, 0.0], query_text="learning", k=3)

        assert len(results) >= 1

        for r in results:
            assert "id" in r
            assert "score" in r
            assert "metadata" in r
            assert r["score"] > 0


def test_hybrid_search_with_alpha():
    """Test hybrid search with alpha weighting"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)

        db.set(
            [
                {"id": "vec_close", "vector": [1.0, 0.0, 0.0, 0.0], "text": "unrelated topic"},
                {
                    "id": "text_match",
                    "vector": [0.0, 0.0, 0.0, 1.0],
                    "text": "exact query match here",
                },
            ]
        )
        db.flush()

        # High alpha (favor vector) - vec_close should rank higher
        results_vector = db.search_hybrid(
            query_vector=[1.0, 0.0, 0.0, 0.0], query_text="query match", k=2, alpha=0.9
        )

        # Low alpha (favor text) - text_match should rank higher
        results_text = db.search_hybrid(
            query_vector=[1.0, 0.0, 0.0, 0.0], query_text="query match", k=2, alpha=0.1
        )

        # Both should return results
        assert len(results_vector) >= 1
        assert len(results_text) >= 1

        # Verify scores are present and positive
        for r in results_vector:
            assert "score" in r
            assert r["score"] > 0

        for r in results_text:
            assert "score" in r
            assert r["score"] > 0


def test_hybrid_search_with_rrf_k():
    """Test hybrid search with custom RRF k parameter"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)

        db.set(
            [
                {"id": "doc1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "test document one"},
                {"id": "doc2", "vector": [0.0, 1.0, 0.0, 0.0], "text": "test document two"},
            ]
        )
        db.flush()

        # Test with different rrf_k values
        results_default = db.search_hybrid(
            query_vector=[1.0, 0.0, 0.0, 0.0], query_text="test", k=2
        )

        results_custom = db.search_hybrid(
            query_vector=[1.0, 0.0, 0.0, 0.0], query_text="test", k=2, rrf_k=10
        )

        assert len(results_default) >= 1
        assert len(results_custom) >= 1


def test_hybrid_search_with_filter():
    """Test hybrid search with metadata filter"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)

        db.set(
            [
                {
                    "id": "doc1",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "text": "machine learning basics",
                    "metadata": {"year": 2024},
                },
                {
                    "id": "doc2",
                    "vector": [0.9, 0.1, 0.0, 0.0],
                    "text": "machine learning advanced",
                    "metadata": {"year": 2023},
                },
                {
                    "id": "doc3",
                    "vector": [0.8, 0.2, 0.0, 0.0],
                    "text": "machine learning tutorial",
                    "metadata": {"year": 2024},
                },
            ]
        )
        db.flush()

        results = db.search_hybrid(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            query_text="machine learning",
            k=10,
            filter={"year": 2024},
        )

        assert len(results) >= 1
        for r in results:
            assert r["metadata"]["year"] == 2024


def test_hybrid_search_metadata_in_results():
    """Test that hybrid search returns metadata"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)

        db.set(
            [
                {
                    "id": "doc1",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "text": "search test",
                    "metadata": {"title": "Test Doc", "tags": ["a", "b"], "count": 42},
                },
            ]
        )
        db.flush()

        results = db.search_hybrid(query_vector=[1.0, 0.0, 0.0, 0.0], query_text="search", k=1)

        assert len(results) == 1
        result = results[0]

        assert result["id"] == "doc1"
        assert "metadata" in result
        assert result["metadata"]["title"] == "Test Doc"
        assert result["metadata"]["tags"] == ["a", "b"]
        assert result["metadata"]["count"] == 42


def test_hybrid_search_empty_results():
    """Test hybrid search with no matching results"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)

        db.set(
            [
                {"id": "doc1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "python programming"},
            ]
        )
        db.flush()

        # Search for text that doesn't exist
        results = db.search_text("xyznonexistent", k=10)

        # Should return empty or no matches
        assert len(results) == 0


def test_hybrid_search_all_params():
    """Test hybrid search with all parameters specified"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)

        db.set(
            [
                {
                    "id": "doc1",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "text": "comprehensive test",
                    "metadata": {"score": 100},
                },
                {
                    "id": "doc2",
                    "vector": [0.0, 1.0, 0.0, 0.0],
                    "text": "comprehensive test",
                    "metadata": {"score": 50},
                },
            ]
        )
        db.flush()

        results = db.search_hybrid(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            query_text="comprehensive",
            k=2,
            filter={"score": {"$gte": 50}},
            alpha=0.5,
            rrf_k=60,
        )

        assert len(results) >= 1
        for r in results:
            assert r["metadata"]["score"] >= 50


def test_hybrid_search_with_subscores():
    """Test hybrid search with subscores=True returns keyword and semantic scores"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)

        db.set(
            [
                # doc1: matches both vector and text
                {
                    "id": "doc1",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "text": "machine learning algorithms",
                },
                # doc2: matches text only (very different vector)
                {
                    "id": "doc2",
                    "vector": [0.0, 0.0, 0.0, 1.0],
                    "text": "machine learning models",
                },
                # doc3: matches vector only (no matching text)
                {
                    "id": "doc3",
                    "vector": [0.9, 0.1, 0.0, 0.0],
                    "text": "cooking recipes",
                },
            ]
        )
        db.flush()

        # Test without subscores (default) - should NOT have keyword_score/semantic_score
        results_default = db.search_hybrid(
            query_vector=[1.0, 0.0, 0.0, 0.0], query_text="machine learning", k=3
        )
        assert len(results_default) == 3
        assert "keyword_score" not in results_default[0]
        assert "semantic_score" not in results_default[0]

        # Test with subscores=True - should have keyword_score and semantic_score
        results = db.search_hybrid(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            query_text="machine learning",
            k=3,
            subscores=True,
        )

        assert len(results) == 3

        # All results should have the subscores keys
        for r in results:
            assert "id" in r
            assert "score" in r
            assert "keyword_score" in r
            assert "semantic_score" in r
            assert "metadata" in r

        # doc1 should have both scores (matches both)
        doc1 = next(r for r in results if r["id"] == "doc1")
        assert doc1["keyword_score"] is not None, "doc1 should have keyword_score"
        assert doc1["semantic_score"] is not None, "doc1 should have semantic_score"

        # doc3 should have semantic but no keyword (text doesn't match "machine learning")
        doc3 = next(r for r in results if r["id"] == "doc3")
        assert doc3["semantic_score"] is not None, "doc3 should have semantic_score"
        assert doc3["keyword_score"] is None, "doc3 should not have keyword_score"

        # doc1 should rank highest (both vector similarity and text match)
        assert results[0]["id"] == "doc1"


def test_hybrid_search_subscores_with_filter():
    """Test hybrid search with subscores and filter"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)

        db.set(
            [
                {
                    "id": "doc1",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "text": "machine learning",
                    "metadata": {"year": 2024},
                },
                {
                    "id": "doc2",
                    "vector": [0.9, 0.1, 0.0, 0.0],
                    "text": "machine learning",
                    "metadata": {"year": 2023},
                },
            ]
        )
        db.flush()

        results = db.search_hybrid(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            query_text="machine learning",
            k=10,
            filter={"year": 2024},
            subscores=True,
        )

        # Only doc1 should match (year == 2024)
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["keyword_score"] is not None
        assert results[0]["semantic_score"] is not None


if __name__ == "__main__":
    test_enable_text_search()
    test_set_with_text_auto_enables()
    test_text_search()
    test_hybrid_search_basic()
    test_hybrid_search_with_alpha()
    test_hybrid_search_with_rrf_k()
    test_hybrid_search_with_filter()
    test_hybrid_search_metadata_in_results()
    test_hybrid_search_empty_results()
    test_hybrid_search_all_params()
    print("All hybrid search tests passed!")
