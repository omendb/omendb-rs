"""Shared helpers for schema-first persistent test setup."""

from __future__ import annotations

from collections.abc import Mapping

import omendb


def create_dense_db(
    path: str,
    dim: int,
    *,
    metric: str | None = None,
    quantization: bool | str | None = None,
    text_search: bool | Mapping[str, object] | None = None,
    embedding_fn=None,
):
    schema: dict[str, object] = {"dense": {"dim": dim}}
    if metric is not None:
        schema["metric"] = metric
    if quantization not in (None, False):
        dense = schema["dense"]
        assert isinstance(dense, dict)
        dense["quantization"] = "sq8" if quantization is True else quantization
    if text_search:
        text_schema: dict[str, object] = {}
        if isinstance(text_search, Mapping):
            if "writer_buffer_mb" in text_search:
                text_schema["writer_buffer_mb"] = text_search["writer_buffer_mb"]
            elif "buffer_mb" in text_search:
                text_schema["writer_buffer_mb"] = text_search["buffer_mb"]
            if "tokenizer" in text_search:
                text_schema["tokenizer"] = text_search["tokenizer"]
        schema["text"] = text_schema
    if embedding_fn is None:
        return omendb.create(path, schema)
    return omendb.create(path, schema, embedding_fn=embedding_fn)


def create_multi_db(
    path: str,
    token_dim: int,
    *,
    multi_vector: bool | Mapping[str, object] | None = None,
    metric: str | None = None,
    embedding_fn=None,
):
    multi_cfg = {} if multi_vector in (None, True) else dict(multi_vector)
    schema: dict[str, object] = {
        "metric": metric or "ip",
        "multi": {"token_dim": token_dim},
    }
    multi_schema = schema["multi"]
    assert isinstance(multi_schema, dict)
    for src_key, dst_key in (
        ("encoder", "encoder"),
        ("repetitions", "repetitions"),
        ("partition_bits", "partition_bits"),
        ("partitionBits", "partition_bits"),
        ("d_proj", "d_proj"),
        ("dProj", "d_proj"),
        ("seed", "seed"),
        ("max_tokens", "max_tokens"),
        ("maxTokens", "max_tokens"),
        ("pool_factor", "pool_factor"),
        ("poolFactor", "pool_factor"),
    ):
        if src_key in multi_cfg and multi_cfg[src_key] is not None:
            multi_schema[dst_key] = multi_cfg[src_key]
    if embedding_fn is None:
        return omendb.create(path, schema)
    return omendb.create(path, schema, embedding_fn=embedding_fn)


def ensure_dense_db(
    path: str,
    dim: int,
    *,
    metric: str | None = None,
    quantization: bool | str | None = None,
    text_search: bool | Mapping[str, object] | None = None,
    embedding_fn=None,
):
    try:
        return omendb.open(path)
    except ValueError as exc:
        if "omendb.create" not in str(exc):
            raise
    return create_dense_db(
        path,
        dim,
        metric=metric,
        quantization=quantization,
        text_search=text_search,
        embedding_fn=embedding_fn,
    )


def ensure_multi_db(
    path: str,
    token_dim: int,
    *,
    multi_vector: bool | Mapping[str, object] | None = None,
    metric: str | None = None,
    embedding_fn=None,
):
    try:
        return omendb.open(path)
    except ValueError as exc:
        if "omendb.create" not in str(exc):
            raise
    return create_multi_db(
        path,
        token_dim,
        multi_vector=multi_vector,
        metric=metric,
        embedding_fn=embedding_fn,
    )
