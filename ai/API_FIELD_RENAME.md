# API Field Rename: `embedding` → `vector`

**Date**: December 4, 2025
**Status**: TODO
**Breaking**: Yes (alpha, acceptable)

---

## Change

Standardize on `vector` field name, remove `embedding` alias.

### Why

| Reason               | Detail                                     |
| -------------------- | ------------------------------------------ |
| Endpoint consistency | `POST /vectors` → `{vector: [...]}`        |
| Product name         | "Vector database" not "embedding database" |
| Competitor alignment | Qdrant, Pinecone use `vector`              |
| Simplicity           | One field name, no aliases                 |

---

## Files to Update

### 1. Python bindings (`python/src/lib.rs`)

Line ~1103-1108:

```rust
// BEFORE: Support "embedding", "vector", or "values" field names
let embedding: Vec<f32> = dict.get_item("embedding")?
    .or(dict.get_item("vector")?)
    .or(dict.get_item("values")?)

// AFTER: Use "vector" only
let vector: Vec<f32> = dict.get_item("vector")?
    .ok_or_else(|| PyValueError::new_err(
        format!("Item '{}' missing 'vector' field", id)
    ))?
    .extract()?;
```

Line ~175 (set signature):

```rust
// BEFORE
#[pyo3(name = "set", signature = (id_or_items=None, embedding=None, ...))]
fn set_vectors(..., embedding: Option<Vec<f32>>, ...)

// AFTER
#[pyo3(name = "set", signature = (id_or_items=None, vector=None, ...))]
fn set_vectors(..., vector: Option<Vec<f32>>, ...)
```

### 2. LangChain adapter (`python/omendb/langchain.py`)

Line ~137:

```python
# BEFORE
items.append({
    "id": id_,
    "embedding": embedding,
    "metadata": item_metadata,
})

# AFTER
items.append({
    "id": id_,
    "vector": embedding,  # variable name doesn't matter, field name does
    "metadata": item_metadata,
})
```

### 3. LlamaIndex adapter (`python/omendb/llamaindex.py`)

Line ~161:

```python
# BEFORE
items.append({
    "id": node_id,
    "embedding": embedding,
    "metadata": metadata,
})

# AFTER
items.append({
    "id": node_id,
    "vector": embedding,
    "metadata": metadata,
})
```

### 4. Examples and tests

Update all occurrences in:

- `python/examples/*.py`
- `python/tests/*.py`

```python
# BEFORE
db.set([{"id": "doc1", "embedding": [0.1, 0.2, 0.3]}])

# AFTER
db.set([{"id": "doc1", "vector": [0.1, 0.2, 0.3]}])
```

### 5. FFI (`src/ffi.rs`)

Line ~123, ~170, ~246, ~430:

```rust
// BEFORE
// items_json: [{"id": "...", "embedding": [...], ...}]
item.get("embedding")
"embedding": vector.data,

// AFTER
// items_json: [{"id": "...", "vector": [...], ...}]
item.get("vector")
"vector": vector.data,
```

### 6. Docstrings

Update all docstrings that mention `embedding` field to say `vector`.

---

## Return values

`db.get()` should also return `vector` not `embedding`:

```python
# BEFORE
{"id": "doc1", "embedding": [...], "metadata": {...}}

# AFTER
{"id": "doc1", "vector": [...], "metadata": {...}}
```

---

## Migration

None needed - alpha version, breaking changes allowed.

Users update: `embedding` → `vector` in their code.
