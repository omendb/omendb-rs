/**
 * OmenDB C FFI Header
 *
 * Embedded vector database with HNSW indexing.
 *
 * Build the library:
 *   cargo build --release -p omendb-ffi
 *
 * Link against:
 *   target/release/libomendb_ffi.a (static) or
 *   target/release/libomendb_ffi.so / .dylib (dynamic)
 *
 * Example:
 *   #include "omendb.h"
 *
 *   omendb_db_t* db = omendb_open("./vectors", 384, NULL);
 *   if (!db) {
 *       printf("Error: %s\n", omendb_last_error());
 *       return 1;
 *   }
 *
 *   // Insert vectors
 *   const char* items = "[{\"id\":\"doc1\",\"vector\":[0.1,...],\"metadata\":{}}]";
 *   int64_t count = omendb_set(db, items);
 *
 *   // Search
 *   float query[384] = {0.1, ...};
 *   char* results = NULL;
 *   if (omendb_search(db, query, 384, 10, NULL, &results) == 0) {
 *       printf("Results: %s\n", results);
 *       omendb_free_string(results);
 *   }
 *
 *   omendb_close(db);
 */

#ifndef OMENDB_H
#define OMENDB_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque database handle. */
typedef struct OmenDB omendb_db_t;

/* ── Lifecycle ──────────────────────────────────────────────────────── */

/**
 * Open or create a database at the given path.
 *
 * @param path        Path to database directory (UTF-8 null-terminated)
 * @param dimensions  Vector dimensionality
 * @param config_json Optional JSON config string (NULL for defaults)
 *                    Format: {"m": 16, "ef_construction": 100, "ef_search": 100}
 * @return Database handle on success, NULL on failure (check omendb_last_error)
 */
omendb_db_t* omendb_open(const char* path, size_t dimensions, const char* config_json);

/**
 * Close database and free resources.
 * @param db Database handle (may be NULL)
 */
void omendb_close(omendb_db_t* db);

/* ── CRUD ───────────────────────────────────────────────────────────── */

/**
 * Insert or replace vectors.
 *
 * @param db         Database handle
 * @param items_json JSON array: [{"id": "...", "vector": [...], "metadata": {...}}, ...]
 * @return Number of vectors inserted, or -1 on error
 */
int64_t omendb_set(omendb_db_t* db, const char* items_json);

/**
 * Get vectors by ID.
 *
 * @param db       Database handle
 * @param ids_json JSON array of IDs: ["id1", "id2", ...]
 * @param result   Output pointer for result JSON (caller must free with omendb_free_string)
 * @return 0 on success, -1 on error
 *
 * Result format: [{"id": "...", "vector": [...], "metadata": {...}}, ...]
 */
int32_t omendb_get(omendb_db_t* db, const char* ids_json, char** result);

/**
 * Delete vectors by ID.
 *
 * @param db       Database handle
 * @param ids_json JSON array of IDs: ["id1", "id2", ...]
 * @return Number of vectors deleted, or -1 on error
 */
int64_t omendb_delete(omendb_db_t* db, const char* ids_json);

/**
 * Update a vector's data and/or metadata.
 *
 * @param db            Database handle
 * @param id            Vector ID (null-terminated UTF-8)
 * @param vector        New vector data (NULL to keep existing)
 * @param vector_dim    Length of vector array (ignored if vector is NULL)
 * @param metadata_json New metadata JSON (NULL to keep existing)
 * @return 0 on success, -1 on error
 */
int32_t omendb_update(omendb_db_t* db, const char* id,
                      const float* vector, size_t vector_dim,
                      const char* metadata_json);

/**
 * Delete vectors matching a metadata filter.
 *
 * @param db          Database handle
 * @param filter_json JSON filter: {"key": "value"} or {"key": {"$gt": 5}}
 * @return Count of deleted vectors, or -1 on error
 */
int64_t omendb_delete_by_filter(omendb_db_t* db, const char* filter_json);

/**
 * Check if a vector ID exists.
 *
 * @param db Database handle
 * @param id Vector ID (null-terminated UTF-8)
 * @return 1 if exists, 0 if not, -1 on error
 */
int32_t omendb_exists(const omendb_db_t* db, const char* id);

/* ── Search ─────────────────────────────────────────────────────────── */

/**
 * Search for similar vectors.
 *
 * @param db          Database handle
 * @param query       Query vector (float array)
 * @param query_len   Length of query vector (must match db dimensions)
 * @param k           Number of results to return
 * @param filter_json Optional filter JSON (NULL for no filter)
 * @param result      Output pointer for result JSON (caller must free)
 * @return 0 on success, -1 on error
 *
 * Result format: [{"id": "...", "distance": 0.123, "vector": [...], "metadata": {...}}, ...]
 */
int32_t omendb_search(omendb_db_t* db, const float* query, size_t query_len,
                      size_t k, const char* filter_json, char** result);

/* ── Text & Hybrid Search ───────────────────────────────────────────── */

/**
 * Enable text search (BM25) for hybrid search.
 *
 * @param db Database handle
 * @return 0 on success, -1 on error
 */
int32_t omendb_enable_text_search(omendb_db_t* db);

/**
 * Check if text search is enabled.
 *
 * @param db Database handle
 * @return 1 if enabled, 0 if not, -1 on error
 */
int32_t omendb_has_text_search(const omendb_db_t* db);

/**
 * Insert or replace vectors with associated text for hybrid search.
 *
 * @param db         Database handle
 * @param items_json JSON array: [{"id": "...", "vector": [...], "text": "...", "metadata": {...}}, ...]
 * @return Number of vectors inserted, or -1 on error
 */
int64_t omendb_set_with_text(omendb_db_t* db, const char* items_json);

/**
 * Text-only search using BM25.
 *
 * @param db     Database handle
 * @param query  Text query string (null-terminated UTF-8)
 * @param k      Number of results to return
 * @param result Output pointer for result JSON (caller must free)
 * @return 0 on success, -1 on error
 *
 * Result format: [{"id": "...", "score": 0.5, "metadata": {...}}, ...]
 */
int32_t omendb_text_search(omendb_db_t* db, const char* query, size_t k, char** result);

/**
 * Hybrid search combining vector similarity and BM25 text search.
 *
 * @param db           Database handle
 * @param query_vector Query vector (float array)
 * @param query_len    Length of query vector
 * @param query_text   Text query string (null-terminated UTF-8)
 * @param k            Number of results to return
 * @param alpha        Vector vs text weight (0.0=text only, 1.0=vector only, <0 for default 0.5)
 * @param rrf_k        RRF constant (0 for default 60)
 * @param filter_json  Optional filter JSON (NULL for no filter)
 * @param result       Output pointer for result JSON (caller must free)
 * @return 0 on success, -1 on error
 *
 * Result format: [{"id": "...", "score": 0.5, "metadata": {...}}, ...]
 */
int32_t omendb_hybrid_search(omendb_db_t* db, const float* query_vector, size_t query_len,
                             const char* query_text, size_t k, float alpha, size_t rrf_k,
                             const char* filter_json, char** result);

/**
 * Flush pending changes (commits text index).
 *
 * @param db Database handle
 * @return 0 on success, -1 on error
 */
int32_t omendb_flush(omendb_db_t* db);

/* ── Maintenance ────────────────────────────────────────────────────── */

/**
 * Compact the database by removing deleted records.
 *
 * @param db Database handle
 * @return Count of compacted (removed) items, or -1 on error
 */
int64_t omendb_compact(omendb_db_t* db);

/**
 * Optimize index for cache-efficient search.
 * Reorders nodes for better memory locality.
 *
 * @param db Database handle
 * @return Count of reordered items, or -1 on error
 */
int64_t omendb_optimize(omendb_db_t* db);

/* ── Info ───────────────────────────────────────────────────────────── */

/** Get number of vectors in database. Returns -1 if db is NULL. */
int64_t omendb_count(const omendb_db_t* db);

/**
 * Save database to disk.
 *
 * @param db Database handle
 * @return 0 on success, -1 on error
 */
int32_t omendb_save(omendb_db_t* db);

/**
 * Get database statistics as JSON.
 *
 * @param db     Database handle
 * @param result Output pointer for stats JSON (caller must free with omendb_free_string)
 * @return 0 on success, -1 on error
 *
 * Result format: {"count": N, "dimensions": D, "quantized": false, "memory_bytes": N}
 */
int32_t omendb_stats(const omendb_db_t* db, char** result);

/** Get last error message. Returns NULL if no error. Valid until next FFI call. */
const char* omendb_last_error(void);

/** Free a string returned by OmenDB. */
void omendb_free_string(char* s);

/** Get OmenDB version string (e.g., "0.0.27"). */
const char* omendb_version(void);

#ifdef __cplusplus
}
#endif

#endif /* OMENDB_H */
