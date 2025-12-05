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
    import omendb

    return omendb.open(temp_db_path, dimensions=128)


@pytest.fixture
def db_with_vectors(temp_db_path):
    """Create a database with sample vectors"""
    import omendb

    db = omendb.open(temp_db_path, dimensions=128)

    vectors = [
        {"id": "vec1", "vector": [0.1] * 128, "metadata": {"label": "A", "value": 1}},
        {"id": "vec2", "vector": [0.2] * 128, "metadata": {"label": "B", "value": 2}},
        {"id": "vec3", "vector": [0.3] * 128, "metadata": {"label": "C", "value": 3}},
        {"id": "vec4", "vector": [0.4] * 128, "metadata": {"label": "A", "value": 4}},
        {"id": "vec5", "vector": [0.5] * 128, "metadata": {"label": "B", "value": 5}},
    ]
    db.set(vectors)
    return db


@pytest.fixture
def sample_vectors():
    """Sample vectors for testing"""
    return [
        {"id": "v1", "vector": [0.1] * 128, "metadata": {"category": "A"}},
        {"id": "v2", "vector": [0.2] * 128, "metadata": {"category": "B"}},
        {"id": "v3", "vector": [0.3] * 128, "metadata": {"category": "C"}},
    ]
