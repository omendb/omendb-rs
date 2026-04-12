"""Tests for open() configuration options."""

import tempfile

import pytest

import omendb


def test_default_config(temp_db_path):
    """Persistent creation should go through the schema-first contract."""
    db = omendb.create(temp_db_path, {"dense": {"dim": 128}})
    db.set([{"id": "v1", "vector": [0.1] * 128, "metadata": {}}])
    results = db.search([0.1] * 128, k=1)
    assert len(results) == 1


def test_hnsw_parameters(temp_db_path):
    """Top-level HNSW parameters should configure the store cleanly."""
    db = omendb.create(temp_db_path, {"dense": {"dim": 128}})
    db.ef_search = 50

    vectors = [{"id": f"v{i}", "vector": [float(i)] * 128, "metadata": {}} for i in range(100)]
    db.set(vectors)

    results = db.search([50.0] * 128, k=10)
    assert len(results) == 10


def test_quantization_bool_option(temp_db_path):
    """Boolean quantization should map to the supported SQ8 path."""
    db = omendb.create(temp_db_path, {"dense": {"dim": 128, "quantization": "sq8"}})
    vectors = [{"id": f"v{i}", "vector": [float(i)] * 128, "metadata": {}} for i in range(50)]
    db.set(vectors)
    results = db.search([25.0] * 128, k=5)
    assert len(results) == 5


def test_quantization_string_option(temp_db_path):
    """String quantization aliases should use the same supported SQ8 path."""
    db = omendb.create(temp_db_path, {"dense": {"dim": 128, "quantization": "sq8"}})
    vectors = [{"id": f"v{i}", "vector": [float(i)] * 128, "metadata": {}} for i in range(50)]
    db.set(vectors)
    results = db.search([25.0] * 128, k=5)
    assert len(results) == 5


def test_dimensions_parameter(temp_db_path):
    """Different dimensionalities should still work through top-level dimensions."""
    for dims in [64, 128, 256, 384, 512, 768, 1024, 1536]:
        db = omendb.create(temp_db_path + f"_{dims}", {"dense": {"dim": dims}})
        db.set([{"id": "v1", "vector": [0.1] * dims, "metadata": {}}])
        results = db.search([0.1] * dims, k=1)
        assert len(results) == 1


def test_dimensions_can_infer_from_first_insert():
    """Single-vector stores should preserve 0-dim inference at open time."""
    db = omendb.open(":memory:")
    assert db.dimensions == 0

    db.set([{"id": "v1", "vector": [0.1, 0.2, 0.3], "metadata": {}}])

    assert db.dimensions == 3
    results = db.search([0.1, 0.2, 0.3], k=1)
    assert len(results) == 1


def test_multi_vector_requires_dimensions():
    """Multi-vector stores should still require explicit token dimensions."""
    with pytest.raises((TypeError, ValueError)):
        omendb.open(":memory:", multi_vector=True)


def test_config_persistence(temp_db_path):
    """Configured HNSW settings should persist across reopen."""
    db = omendb.create(temp_db_path, {"dense": {"dim": 128}})
    db.ef_search = 100
    db.set([{"id": "v1", "vector": [0.1] * 128, "metadata": {}}])
    db.flush()
    del db

    db2 = omendb.open(temp_db_path)
    db2.set([{"id": "v2", "vector": [0.2] * 128, "metadata": {}}])

    results = db2.search([0.15] * 128, k=2)
    assert len(results) == 2


def test_text_search_open_option():
    """Persistent text configuration should be created through schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/typed-open"
        db = omendb.create(
            path,
            {"dense": {"dim": 4}, "text": {"writer_buffer_mb": 20, "tokenizer": "code"}},
        )
        assert db.has_text_search() is True
        db.close()


def test_open_missing_database_requires_create(temp_db_path):
    """Persistent open() should not create stores implicitly."""
    with pytest.raises((TypeError, ValueError)) as excinfo:
        omendb.open(temp_db_path, dimensions=128)
    assert "omendb.create" in str(excinfo.value)


def test_close_raises_clean_error_on_closed_handle():
    """Closed handles should raise a regular error instead of panicking."""
    db = omendb.create(":memory:", {"dense": {"dim": 4}})
    db.close()

    with pytest.raises(RuntimeError, match="database is closed"):
        len(db)


def test_invalid_text_search_type(temp_db_path):
    """Unsupported text_search values should fail fast."""
    with pytest.raises((TypeError, ValueError)):
        omendb.open(temp_db_path, dimensions=128, text_search="invalid")


def test_schema_and_info_report_authoritative_runtime_contract():
    """schema() and info()['schema'] should reflect the live store contract."""
    db = omendb.open(
        ":memory:",
        dimensions=4,
        text_search={"writer_buffer_mb": 20, "tokenizer": "code"},
    )

    schema = db.schema()
    info = db.info()

    assert schema["metric"] == "l2"
    assert schema["dense"]["dim"] == 4
    assert schema["dense"]["quantization"] == "none"
    assert schema["text"]["tokenizer"] == "code"
    assert schema["text"]["writer_buffer_mb"] == 20
    assert info["schema"] == schema


def test_create_uses_schema_first_contract():
    """create() should build directly from explicit schema."""
    db = omendb.create(
        ":memory:",
        {
            "name": "docs",
            "metric": "l2",
            "dense": {"dim": 4, "quantization": "sq8"},
            "text": {"tokenizer": "code", "writer_buffer_mb": 20},
        },
    )

    schema = db.schema()
    assert schema["name"] == "docs"
    assert schema["dense"]["dim"] == 4
    assert schema["dense"]["quantization"] == "sq8"
    assert schema["text"]["tokenizer"] == "code"
