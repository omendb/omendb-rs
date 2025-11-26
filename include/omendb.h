/**
 * OmenDB - Fast Embedded Vector Database
 *
 * C API for OmenDB vector database with HNSW + ACORN-1 filtered search.
 *
 * Build with: cargo build --release --features ffi
 * Link with: -lomendb (or libomendb.a / libomendb.so / libomendb.dylib)
 *
 * Example:
 *   omendb_db_t* db = omendb_open("./vectors", 384, NULL);
 *   omendb_set(db, "[{\"id\":\"doc1\",\"embedding\":[...],\"metadata\":{}}]");
 *
 *   float query[384] = {...};
 *   char* results = NULL;
 *   omendb_search(db, query, 384, 10, NULL, &results);
 *   printf("%s\n", results);
 *   omendb_free_string(results);
 *
 *   omendb_close(db);
 */

#ifndef OMENDB_H
#define OMENDB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque database handle */
typedef struct OmenDB omendb_db_t;

/**
 * Open a database at the given path.
 *
 * @param path        Path to database directory (UTF-8, will be created if needed)
 * @param dimensions  Vector dimensionality (e.g., 384, 768, 1536)
 * @param config_json Optional JSON config string, or NULL for defaults
 * @return Database handle on success, NULL on failure (check omendb_last_error)
 */
omendb_db_t* omendb_open(const char* path, size_t dimensions, const char* config_json);

/**
 * Close database and free all resources.
 *
 * @param db Database handle (safe to pass NULL)
 */
void omendb_close(omendb_db_t* db);

/**
 * Insert or replace vectors.
 *
 * @param db         Database handle
 * @param items_json JSON array: [{"id":"...", "embedding":[...], "metadata":{...}}, ...]
 * @return Number of vectors inserted on success, -1 on error
 */
int64_t omendb_set(omendb_db_t* db, const char* items_json);

/**
 * Get vectors by ID.
 *
 * @param db       Database handle
 * @param ids_json JSON array of IDs: ["id1", "id2", ...]
 * @param result   Output: JSON array of items (caller must free with omendb_free_string)
 * @return 0 on success, -1 on error
 */
int32_t omendb_get(omendb_db_t* db, const char* ids_json, char** result);

/**
 * Delete vectors by ID.
 *
 * @param db       Database handle
 * @param ids_json JSON array of IDs: ["id1", "id2", ...]
 * @return Number of vectors deleted on success, -1 on error
 */
int64_t omendb_delete(omendb_db_t* db, const char* ids_json);

/**
 * Search for similar vectors.
 *
 * @param db          Database handle
 * @param query       Query vector (float array)
 * @param query_len   Length of query vector (must match database dimensions)
 * @param k           Number of results to return
 * @param filter_json Optional filter JSON, or NULL for no filter
 * @param result      Output: JSON array of results (caller must free with omendb_free_string)
 *                    Format: [{"id":"...", "distance":0.123, "metadata":{...}}, ...]
 * @return 0 on success, -1 on error
 */
int32_t omendb_search(
    omendb_db_t* db,
    const float* query,
    size_t query_len,
    size_t k,
    const char* filter_json,
    char** result
);

/**
 * Get number of vectors in database.
 *
 * @param db Database handle
 * @return Vector count, or -1 on error
 */
int64_t omendb_count(const omendb_db_t* db);

/**
 * Save database to disk.
 *
 * @param db Database handle
 * @return 0 on success, -1 on error
 */
int32_t omendb_save(const omendb_db_t* db);

/**
 * Get last error message.
 *
 * @return Error message (valid until next FFI call), or NULL if no error
 */
const char* omendb_last_error(void);

/**
 * Free a string returned by OmenDB functions.
 *
 * @param s String to free (safe to pass NULL)
 */
void omendb_free_string(char* s);

/**
 * Get OmenDB version string.
 *
 * @return Version string (e.g., "0.0.1")
 */
const char* omendb_version(void);

#ifdef __cplusplus
}
#endif

#endif /* OMENDB_H */
