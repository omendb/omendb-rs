/**
 * OmenDB Basic Operations
 *
 * Demonstrates core CRUD operations:
 * - Creating a database
 * - Adding vectors with metadata
 * - Searching for nearest neighbors
 * - Retrieving, updating, and deleting vectors
 * - Persistence to disk
 */

import { create, open } from "../index.js";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

async function main() {
	// Use a temp directory for this example
	const tmpDir = mkdtempSync(join(tmpdir(), "omendb-"));
	const dbPath = join(tmpDir, "vectors");

	try {
		// Create database (3D vectors for readability)
		const db = create(dbPath, { dense: { dim: 3 } });

		// --- INSERT ---
		// Add vectors individually or in batches
		await db.set([
			{ id: "a", vector: new Float32Array([1.0, 0.0, 0.0]), metadata: { axis: "x" } },
			{ id: "b", vector: new Float32Array([0.0, 1.0, 0.0]), metadata: { axis: "y" } },
			{ id: "c", vector: new Float32Array([0.0, 0.0, 1.0]), metadata: { axis: "z" } },
			{ id: "d", vector: new Float32Array([0.7, 0.7, 0.0]), metadata: { axis: "xy" } },
		]);
		console.log(`Inserted ${db.length} vectors`);

		// --- SEARCH ---
		// Find vectors similar to query
		const results = await db.search([1.0, 0.0, 0.0], 3);
		console.log("\nSearch results for [1, 0, 0]:");
		for (const r of results) {
			console.log(`  ${r.id}: distance=${r.distance.toFixed(3)}, axis=${r.metadata.axis}`);
		}

		// --- GET ---
		// Retrieve by ID
		const vec = db.get("a");
		if (vec) {
			console.log(`\nGet 'a': vector=[${Array.from(vec.vector).join(", ")}], metadata=${JSON.stringify(vec.metadata)}`);
		}

		// --- UPDATE ---
		// Replace embedding and/or metadata
		await db.set([
			{ id: "a", vector: new Float32Array([0.9, 0.1, 0.0]), metadata: { axis: "x", modified: true } },
		]);
		const updated = db.get("a");
		console.log(`Updated 'a': ${JSON.stringify(updated?.metadata)}`);

		// --- DELETE ---
		const deleted = db.delete(["d"]);
		console.log(`\nDeleted ${deleted} vector(s), ${db.length} remaining`);

		// --- PERSISTENCE ---
		// Flush to disk (also auto-flushes on close)
		db.flush();
		db.close();

		// Reopen to verify persistence
		const db2 = open(dbPath);
		console.log(`\nReopened database: ${db2.length} vectors`);

		// --- BATCH OPERATIONS ---
		// Efficient for large datasets
		const batch = Array.from({ length: 100 }, (_, i) => ({
			id: `rand_${i}`,
			vector: new Float32Array([Math.random(), Math.random(), Math.random()]),
		}));
		await db2.set(batch);
		console.log(`After batch insert: ${db2.length} vectors`);

		// Batch search (async, runs in parallel on thread pool)
		const queries = Array.from({ length: 10 }, () => [
			Math.random(),
			Math.random(),
			Math.random(),
		]);
		const batchResults = await db2.searchBatch(queries, 5);
		console.log(`Batch search: ${batchResults.length} result sets, ${batchResults[0].length} results each`);
		db2.close();
	} finally {
		// Cleanup
		setTimeout(() => rmSync(tmpDir, { recursive: true, force: true }), 100);
	}
}

main();
