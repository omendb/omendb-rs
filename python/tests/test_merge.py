"""Tests for graph merge operations."""

import shutil
import tempfile

import pytest

import omendb


class TestMerge:
    """Test merge functionality."""

    @pytest.fixture
    def db_path1(self):
        """Create first temporary database path."""
        path = tempfile.mkdtemp(suffix="")
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def db_path2(self):
        """Create second temporary database path."""
        path = tempfile.mkdtemp(suffix="")
        yield path
        shutil.rmtree(path, ignore_errors=True)

    def test_merge_basic(self, db_path1, db_path2):
        """Test basic merge operation."""
        db1 = omendb.open(db_path1, dimensions=128)
        db2 = omendb.open(db_path2, dimensions=128)

        # Populate db1
        db1.set(
            [
                {"id": f"db1_vec{i}", "vector": [0.1 * i] * 128, "metadata": {"source": "db1"}}
                for i in range(10)
            ]
        )

        # Populate db2
        db2.set(
            [
                {"id": f"db2_vec{i}", "vector": [0.2 * i] * 128, "metadata": {"source": "db2"}}
                for i in range(20)
            ]
        )

        assert len(db1) == 10
        assert len(db2) == 20

        # Merge db2 into db1
        merged_count = db1.merge_from(db2)

        assert merged_count == 20
        assert len(db1) == 30

        # Verify db2 vectors are searchable in db1
        result = db1.get("db2_vec5")
        assert result is not None
        assert result["metadata"]["source"] == "db2"

    def test_merge_dimension_mismatch(self, db_path1, db_path2):
        """Test that merging databases with different dimensions fails."""
        db1 = omendb.open(db_path1, dimensions=128)
        db2 = omendb.open(db_path2, dimensions=64)  # Different dimensions

        # Populate both dbs
        db1.set([{"id": "vec1", "vector": [0.1] * 128, "metadata": {}}])
        db2.set([{"id": "vec2", "vector": [0.1] * 64, "metadata": {}}])

        # Different dimensions should raise an error
        with pytest.raises(RuntimeError, match="Dimension mismatch"):
            db1.merge_from(db2)

    def test_merge_into_empty(self, db_path1, db_path2):
        """Test merging into empty database."""
        db1 = omendb.open(db_path1, dimensions=128)
        db2 = omendb.open(db_path2, dimensions=128)

        # First establish dimensions in db1 by inserting and deleting
        db1.set([{"id": "temp", "vector": [0.0] * 128, "metadata": {}}])
        db1.delete(["temp"])

        # Populate only db2
        db2.set(
            [
                {"id": f"vec{i}", "vector": [i * 0.1] * 128, "metadata": {"index": i}}
                for i in range(50)
            ]
        )

        merged_count = db1.merge_from(db2)

        assert merged_count == 50
        assert len(db1) == 50

    def test_merge_id_conflict(self, db_path1, db_path2):
        """Test that conflicting IDs are skipped (existing wins)."""
        db1 = omendb.open(db_path1, dimensions=128)
        db2 = omendb.open(db_path2, dimensions=128)

        # Same ID in both databases
        db1.set(
            [{"id": "shared_id", "vector": [0.1] * 128, "metadata": {"source": "db1", "value": 1}}]
        )

        db2.set(
            [
                {
                    "id": "shared_id",
                    "vector": [0.2] * 128,
                    "metadata": {"source": "db2", "value": 2},
                },
                {"id": "unique_id", "vector": [0.3] * 128, "metadata": {"source": "db2"}},
            ]
        )

        merged_count = db1.merge_from(db2)

        # Only unique_id should be merged
        assert merged_count == 1
        assert len(db1) == 2

        # shared_id should retain db1's value
        result = db1.get("shared_id")
        assert result["metadata"]["source"] == "db1"
        assert result["metadata"]["value"] == 1

    def test_merge_preserves_metadata(self, db_path1, db_path2):
        """Test that metadata is preserved during merge."""
        db1 = omendb.open(db_path1, dimensions=128)
        db2 = omendb.open(db_path2, dimensions=128)

        # First establish dimensions in db1 by inserting a vector
        db1.set([{"id": "placeholder", "vector": [0.0] * 128, "metadata": {}}])

        # Complex metadata in db2
        db2.set(
            [
                {
                    "id": "complex_meta",
                    "vector": [0.5] * 128,
                    "metadata": {
                        "nested": {"key": "value"},
                        "array": [1, 2, 3],
                        "unicode": "Hello 世界",
                        "number": 42.5,
                    },
                }
            ]
        )

        db1.merge_from(db2)

        result = db1.get("complex_meta")
        assert result["metadata"]["nested"]["key"] == "value"
        assert result["metadata"]["array"] == [1, 2, 3]
        assert result["metadata"]["unicode"] == "Hello 世界"
        assert result["metadata"]["number"] == 42.5

    def test_merge_search_works(self, db_path1, db_path2):
        """Test that search works correctly after merge."""
        db1 = omendb.open(db_path1, dimensions=128)
        db2 = omendb.open(db_path2, dimensions=128)

        # Insert distinctive vectors
        db1.set(
            [
                {
                    "id": "db1_unique",
                    "vector": [1.0] * 128,  # All 1s
                    "metadata": {"source": "db1"},
                }
            ]
        )

        db2.set(
            [
                {
                    "id": "db2_unique",
                    "vector": [0.0] * 128,  # All 0s
                    "metadata": {"source": "db2"},
                }
            ]
        )

        db1.merge_from(db2)

        # Search for vector close to all 0s - should find db2_unique
        results = db1.search([0.0] * 128, k=1)
        assert len(results) == 1
        assert results[0]["id"] == "db2_unique"

        # Search for vector close to all 1s - should find db1_unique
        results = db1.search([1.0] * 128, k=1)
        assert len(results) == 1
        assert results[0]["id"] == "db1_unique"

    def test_merge_larger_scale(self, db_path1, db_path2):
        """Test merge at larger scale (1K vectors)."""
        db1 = omendb.open(db_path1, dimensions=128)
        db2 = omendb.open(db_path2, dimensions=128)

        # Insert 500 vectors into each
        db1.set(
            [
                {"id": f"db1_{i}", "vector": [i * 0.002] * 128, "metadata": {"idx": i}}
                for i in range(500)
            ]
        )

        db2.set(
            [
                {"id": f"db2_{i}", "vector": [1.0 + i * 0.002] * 128, "metadata": {"idx": i}}
                for i in range(500)
            ]
        )

        assert len(db1) == 500
        assert len(db2) == 500

        merged_count = db1.merge_from(db2)

        assert merged_count == 500
        assert len(db1) == 1000

        # Search should work
        results = db1.search([0.5] * 128, k=10)
        assert len(results) == 10
