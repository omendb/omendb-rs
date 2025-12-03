/**
 * OmenDB C FFI Header
 *
 * Embedded vector database with HNSW indexing.
 *
 * Build the library:
 *   cargo build --release --features ffi
 *
 * Link against:
 *   target/release/libomendb.a (static) or
 *   target/release/libomendb.so / .dylib (dynamic)
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
 *   const char* items = "[{\"id\":\"doc1\",\"embedding\":[0.1,...],\"metadata\":{}}]";
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

/**
 * Open or create a database at the given path.
 *
 * @param path        Path to database directory (UTF-8 null-terminated)
 * @param dimensions  Vector dimensionality
 * @param config_json Optional JSON config string (NULL for defaults)
 * @return Database handle on success, NULL on failure (check omendb_last_error)
 */
omendb_db_t* omendb_open(const char* path, size_t dimensions, const char* config_json);

/**
 * Close database and free resources.
 * @param db Database handle (may be NULL)
 */
void omendb_close(omendb_db_t* db);

/**
 * Insert or replace vectors.
 *
 * @param db         Database handle
 * @param items_json JSON array: [{"id": "...", "embedding": [...], "metadata": {...}}, ...]
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
 * Search for similar vectors.
 *
 * @param db          Database handle
 * @param query       Query vector (float array)
 * @param query_len   Length of query vector (must match db dimensions)
 * @param k           Number of results to return
 * @param filter_json Optional filter JSON (NULL for no filter, not yet implemented)
 * @param result      Output pointer for result JSON (caller must free)
 * @return 0 on success, -1 on error
 *
 * Result format: [{"id": "...", "distance": 0.123, "embedding": [...], "metadata": {...}}, ...]
 */
int32_t omendb_search(omendb_db_t* db, const float* query, size_t query_len,
                      size_t k, const char* filter_json, char** result);

/** Get number of vectors in database. Returns -1 if db is NULL. */
int64_t omendb_count(const omendb_db_t* db);

/** Save database to disk. Returns 0 on success, -1 on error. */
int32_t omendb_save(const omendb_db_t* db);

/** Get last error message. Returns NULL if no error. Valid until next FFI call. */
const char* omendb_last_error(void);

/** Free a string returned by OmenDB. */
void omendb_free_string(char* s);

/** Get OmenDB version string (e.g., "0.0.1"). */
const char* omendb_version(void);

#ifdef __cplusplus
}
#endif

#endif /* OMENDB_H */
