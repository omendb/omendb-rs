/**
 * OmenDB Multi-Vector (ColBERT-style) Search
 *
 * Demonstrates MUVERA encoding for late-interaction retrieval:
 * - Creating a multi-vector store
 * - Inserting documents with token embeddings
 * - Searching with MaxSim reranking
 * - Persistence across restarts
 *
 * Multi-vector search is ideal for ColBERT, SPLADE, and other
 * models that produce per-token embeddings.
 */

import { create, open } from "../index.js";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

// Simulate token embeddings (in practice, use a model like ColBERTv2)
function simulateTokens(text: string, dim: number): Float32Array[] {
	const words = text.toLowerCase().split(/\s+/);
	return words.map((word, i) => {
		const vec = new Float32Array(dim);
		// Simple hash-based embedding for demonstration
		for (let j = 0; j < dim; j++) {
			vec[j] = Math.sin(word.charCodeAt(j % word.length) + i * 0.1 + j * 0.01);
		}
		return vec;
	});
}

async function main() {
	console.log("=== In-Memory Multi-Vector Search ===\n");

	// Create multi-vector store with default config (good balance of speed/quality)
	const db = create(":memory:", { multi: { tokenDim: 64 } });

	// Sample documents with token embeddings
	const documents = [
		{ id: "doc1", text: "vector database for machine learning" },
		{ id: "doc2", text: "nearest neighbor search algorithms" },
		{ id: "doc3", text: "embedding models and retrieval" },
		{ id: "doc4", text: "graph based similarity search" },
		{ id: "doc5", text: "quantization techniques for vectors" },
	];

	// Insert with token embeddings
	await db.set(
		documents.map((doc) => ({
			id: doc.id,
			vectors: simulateTokens(doc.text, 64),
			metadata: { text: doc.text },
		}))
	);
	console.log(`Inserted ${db.length} documents (multi-vector: ${db.isMultiVector})`);

	// Search with query tokens
	const queryText = "vector similarity search";
	const queryTokens = simulateTokens(queryText, 64);
	console.log(`\nQuery: "${queryText}" (${queryTokens.length} tokens)`);

	// Default search (rerank=true for best quality)
	const results = await db.search(queryTokens, 3);
	console.log("\nTop 3 results (with MaxSim reranking):");
	for (const r of results) {
		console.log(`  ${r.id}: score=${r.distance.toFixed(4)} - "${r.metadata.text}"`);
	}

	// Fast search (rerank=false, ~2x faster, lower recall)
	const fastResults = await db.search(queryTokens, 3, { rerank: false });
	console.log("\nFast search (no reranking):");
	for (const r of fastResults) {
		console.log(`  ${r.id}: score=${r.distance.toFixed(4)} - "${r.metadata.text}"`);
	}

	// High-quality search (larger rerank factor)
	const qualityResults = await db.search(queryTokens, 3, { rerankFactor: 8 });
	console.log("\nHigh-quality search (rerankFactor=8):");
	for (const r of qualityResults) {
		console.log(`  ${r.id}: score=${r.distance.toFixed(4)} - "${r.metadata.text}"`);
	}

	// === Persistence ===
	console.log("\n=== Multi-Vector Persistence ===\n");

	const tmpDir = mkdtempSync(join(tmpdir(), "omendb-multivec-"));
	const dbPath = join(tmpDir, "multivec");

	try {
		// Create persistent multi-vector store
		const persistentDb = create(dbPath, { multi: { tokenDim: 32 } });

		// Insert documents
		await persistentDb.set([
			{
				id: "persistent1",
				vectors: [new Float32Array(32).fill(0.1), new Float32Array(32).fill(0.2)],
				metadata: { tag: "first" },
			},
			{
				id: "persistent2",
				vectors: [new Float32Array(32).fill(0.3)],
				metadata: { tag: "second" },
			},
		]);
		persistentDb.flush();
		console.log(`Created persistent store: ${persistentDb.length} documents`);
		persistentDb.close();

		// Reopen - auto-detects multi-vector from saved config
		const reopened = open(dbPath);
		console.log(`Reopened: ${reopened.length} documents, multiVector=${reopened.isMultiVector}`);

		// MaxSim reranking works after reload
		const persistedResults = await reopened.search(
			[new Float32Array(32).fill(0.1), new Float32Array(32).fill(0.15)],
			2,
			{ rerank: true }
		);
		console.log("Search after reload:");
		for (const r of persistedResults) {
			console.log(`  ${r.id}: ${r.metadata.tag}`);
		}
		reopened.close();
	} finally {
		rmSync(tmpDir, { recursive: true, force: true });
	}

	console.log("\n--- Multi-Vector Config Options ---");
	console.log("  multiVector: true                    - default config");
	console.log("  multiVector: { repetitions: 10 }     - higher quality");
	console.log("  multiVector: { partitionBits: 4 }    - finer partitioning");
}

main();
