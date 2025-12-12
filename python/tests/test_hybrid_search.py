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


def test_set_with_text():
    """Test inserting documents with text content"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)
        db.enable_text_search()

        indices = db.set_with_text(
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

        assert len(indices) == 2
        assert len(db) == 2


def test_text_search():
    """Test pure text (BM25) search"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)
        db.enable_text_search()

        db.set_with_text(
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


def test_hybrid_search_basic():
    """Test basic hybrid search (vector + text)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)
        db.enable_text_search()

        db.set_with_text(
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
        db.enable_text_search()

        db.set_with_text(
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
        db.enable_text_search()

        db.set_with_text(
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
        db.enable_text_search()

        db.set_with_text(
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
        db.enable_text_search()

        db.set_with_text(
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
        db.enable_text_search()

        db.set_with_text(
            [
                {"id": "doc1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "python programming"},
            ]
        )
        db.flush()

        # Search for text that doesn't exist
        results = db.search_text("xyznonexistent", k=10)

        # Should return empty or no matches
        assert len(results) == 0


def test_hybrid_search_without_enable():
    """Test that hybrid search fails if text search not enabled"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)

        try:
            db.set_with_text([{"id": "doc1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "test"}])
            raise AssertionError("Should have raised an error")
        except RuntimeError as e:
            assert "not enabled" in str(e).lower()


def test_hybrid_search_all_params():
    """Test hybrid search with all parameters specified"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=4)
        db.enable_text_search()

        db.set_with_text(
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


if __name__ == "__main__":
    test_enable_text_search()
    test_set_with_text()
    test_text_search()
    test_hybrid_search_basic()
    test_hybrid_search_with_alpha()
    test_hybrid_search_with_rrf_k()
    test_hybrid_search_with_filter()
    test_hybrid_search_metadata_in_results()
    test_hybrid_search_empty_results()
    test_hybrid_search_without_enable()
    test_hybrid_search_all_params()
    print("✅ All hybrid search tests passed!")
