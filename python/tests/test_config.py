"""Tests for configuration options"""

import pytest

import omendb


def test_default_config(temp_db_path):
    """Test database with default configuration"""
    db = omendb.open(temp_db_path, dimensions=128)

    # Should work with defaults
    vectors = [{"id": "v1", "vector": [0.1] * 128, "metadata": {}}]
    db.set(vectors)

    results = db.search([0.1] * 128, k=1)
    assert len(results) == 1


def test_hnsw_config(temp_db_path):
    """Test configuring HNSW parameters"""
    config = {"hnsw": {"m": 16, "ef_construction": 200, "ef_search": 50}}

    db = omendb.open(temp_db_path, dimensions=128, config=config)

    # Should work with custom HNSW params
    vectors = [{"id": f"v{i}", "vector": [float(i)] * 128, "metadata": {}} for i in range(100)]
    db.set(vectors)

    results = db.search([50.0] * 128, k=10)
    assert len(results) == 10


def test_quantization_2bit(temp_db_path):
    """Test 2-bit quantization configuration"""
    config = {"quantization": {"bits": 2}}

    db = omendb.open(temp_db_path, dimensions=128, config=config)

    vectors = [{"id": f"v{i}", "vector": [float(i)] * 128, "metadata": {}} for i in range(50)]
    db.set(vectors)

    # Search should still work (may have lower recall)
    results = db.search([25.0] * 128, k=5)
    assert len(results) == 5


def test_quantization_4bit(temp_db_path):
    """Test 4-bit quantization configuration"""
    config = {"quantization": {"bits": 4}}

    db = omendb.open(temp_db_path, dimensions=128, config=config)

    vectors = [{"id": f"v{i}", "vector": [float(i)] * 128, "metadata": {}} for i in range(50)]
    db.set(vectors)

    results = db.search([25.0] * 128, k=5)
    assert len(results) == 5


def test_quantization_8bit(temp_db_path):
    """Test 8-bit quantization configuration"""
    config = {"quantization": {"bits": 8}}

    db = omendb.open(temp_db_path, dimensions=128, config=config)

    vectors = [{"id": f"v{i}", "vector": [float(i)] * 128, "metadata": {}} for i in range(50)]
    db.set(vectors)

    results = db.search([25.0] * 128, k=5)
    assert len(results) == 5


def test_dimensions_parameter(temp_db_path):
    """Test different dimension sizes"""
    for dims in [64, 128, 256, 384, 512, 768, 1024, 1536]:
        db = omendb.open(temp_db_path + f"_{dims}", dimensions=dims)

        vector = {"id": "v1", "vector": [0.1] * dims, "metadata": {}}
        db.set([vector])

        results = db.search([0.1] * dims, k=1)
        assert len(results) == 1


@pytest.mark.xfail(reason="Known persistence issue: data not loaded on reopen")
def test_config_persistence(temp_db_path):
    """Test that configuration is persisted"""
    config = {"hnsw": {"m": 32, "ef_construction": 400, "ef_search": 100}}

    # Create with config
    db = omendb.open(temp_db_path, dimensions=128, config=config)
    db.set([{"id": "v1", "vector": [0.1] * 128, "metadata": {}}])
    db.flush()
    del db

    # Reload (config comes from saved state)
    db2 = omendb.open(temp_db_path, dimensions=128)
    db2.set([{"id": "v2", "vector": [0.2] * 128, "metadata": {}}])

    results = db2.search([0.15] * 128, k=2)
    assert len(results) == 2


def test_empty_config(temp_db_path):
    """Test passing empty config dict"""
    db = omendb.open(temp_db_path, dimensions=128, config={})

    vectors = [{"id": "v1", "vector": [0.1] * 128, "metadata": {}}]
    db.set(vectors)

    results = db.search([0.1] * 128, k=1)
    assert len(results) == 1


def test_invalid_config_type(temp_db_path):
    """Test invalid config type"""
    with pytest.raises((TypeError, ValueError)):
        omendb.open(temp_db_path, dimensions=128, config="invalid")
