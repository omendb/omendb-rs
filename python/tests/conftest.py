"""Pytest configuration and shared fixtures for OmenDB tests"""

import os
import tempfile

import pytest


@pytest.fixture
def temp_db_path():
    """Provide a temporary database path"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest.fixture
def db(temp_db_path):
    """Create a fresh database instance for each test"""
    import gc

    import omendb

    database = omendb.open(temp_db_path, dimensions=128)
    yield database
    # Explicitly drop database and force GC to release file handles before temp dir cleanup
    del database
    gc.collect()


@pytest.fixture
def db_with_vectors(temp_db_path):
    """Create a database with sample vectors"""
    import gc

    import omendb

    database = omendb.open(temp_db_path, dimensions=128)

    vectors = [
        {"id": "vec1", "vector": [0.1] * 128, "metadata": {"label": "A", "value": 1}},
        {"id": "vec2", "vector": [0.2] * 128, "metadata": {"label": "B", "value": 2}},
        {"id": "vec3", "vector": [0.3] * 128, "metadata": {"label": "C", "value": 3}},
        {"id": "vec4", "vector": [0.4] * 128, "metadata": {"label": "A", "value": 4}},
        {"id": "vec5", "vector": [0.5] * 128, "metadata": {"label": "B", "value": 5}},
    ]
    database.set(vectors)

    # Sanity check: verify all vectors were inserted
    assert len(database) == 5, f"Expected 5 vectors, got {len(database)}"

    yield database
    # Explicitly drop database and force GC to release file handles before temp dir cleanup
    del database
    gc.collect()


@pytest.fixture
def sample_vectors():
    """Sample vectors for testing"""
    return [
        {"id": "v1", "vector": [0.1] * 128, "metadata": {"category": "A"}},
        {"id": "v2", "vector": [0.2] * 128, "metadata": {"category": "B"}},
        {"id": "v3", "vector": [0.3] * 128, "metadata": {"category": "C"}},
    ]
