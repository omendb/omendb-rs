import { describe, it, expect, beforeEach } from "vitest";
import { open, VectorDatabase } from "../index.js";

describe("Multi-Vector (MUVERA)", () => {
	describe("creation", () => {
		it("should create a multi-vector store with multiVector: true", () => {
			const db = open(":memory:", { dimensions: 128, multiVector: true });
			expect(db.isMultiVector).toBe(true);
			expect(db.dimensions).toBe(128);
		});

		it("should create a regular store by default", () => {
			const db = open(":memory:", { dimensions: 128 });
			expect(db.isMultiVector).toBe(false);
		});

		it("should create a multi-vector store with custom config", () => {
			const db = open(":memory:", {
				dimensions: 128,
				multiVector: { repetitions: 10, partitionBits: 4 },
			});
			expect(db.isMultiVector).toBe(true);
		});

		it("should reject multi-vector with quantization", () => {
			expect(() =>
				open(":memory:", { dimensions: 128, multiVector: true, quantization: true })
			).toThrow(/quantization/);
		});
	});

	describe("unified API - insert", () => {
		let db: VectorDatabase;

		beforeEach(() => {
			db = open(":memory:", { dimensions: 8, multiVector: { dProj: null } });
		});

		it("should insert using unified set() with vectors field", async () => {
			const vectors = [
				new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
				new Float32Array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
			];
			const count = await db.set([{ id: "doc1", vectors, metadata: { title: "Test" } }]);
			expect(count).toBe(1);
			expect(db.count()).toBe(1);
		});

		it("should insert multiple docs using unified set() with vectors field", async () => {
			const items = Array.from({ length: 10 }, (_, i) => ({
				id: `doc${i}`,
				vectors: Array.from({ length: 5 }, () =>
					new Float32Array(8).fill(i / 10)
				),
				metadata: { index: i },
			}));
			const count = await db.set(items);
			expect(count).toBe(10);
			expect(db.count()).toBe(10);
		});

		it("should reject empty vectors array with unified set()", async () => {
			await expect(
				db.set([{ id: "doc1", vectors: [], metadata: {} }])
			).rejects.toThrow(/must not be empty/);
		});

		it("should reject vectors field on regular store", () => {
			const regularDb = open(":memory:", { dimensions: 8 });
			expect(() =>
				regularDb.set([
					{ id: "doc1", vectors: [new Float32Array(8).fill(0.1)], metadata: {} },
				])
			).toThrow(/multiVector: true/);
		});
	});

	describe("unified API - search", () => {
		let db: VectorDatabase;

		beforeEach(async () => {
			db = open(":memory:", { dimensions: 8, multiVector: { dProj: null } });

			// Insert 100 docs with distinct patterns using unified API
			const items = Array.from({ length: 100 }, (_, i) => {
				const base = i / 100;
				const vectors = Array.from({ length: 5 }, (_, j) =>
					new Float32Array(8).fill(base + j * 0.01)
				);
				return { id: `doc${i}`, vectors, metadata: { index: i } };
			});
			await db.set(items);
		});

		it("should perform basic multi-vector search with unified search()", async () => {
			const query = [
				[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
				[0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51],
			];
			const results = await db.search(query, 5);

			expect(results).toHaveLength(5);
			expect(results.every((r) => "id" in r && "distance" in r && "metadata" in r)).toBe(true);
		});

		it("should search with Float32Array query using unified search()", async () => {
			const query = [
				new Float32Array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
				new Float32Array([0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51]),
			];
			const results = await db.search(query, 5);
			expect(results).toHaveLength(5);
		});

		it("should search with rerank disabled via options", async () => {
			const query = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]];
			const results = await db.search(query, 5, { rerank: false });
			expect(results).toHaveLength(5);
		});

		it("should search with custom rerank factor via options", async () => {
			const query = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]];
			const results = await db.search(query, 5, { rerank: true, rerankFactor: 8 });
			expect(results).toHaveLength(5);
		});

		it("should return metadata in results", async () => {
			const query = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]];
			const results = await db.search(query, 1);

			expect(results).toHaveLength(1);
			expect(results[0].metadata).toHaveProperty("index");
		});
	});

	describe("unified API - single-vector store", () => {
		it("should use unified search() with options on single-vector store", async () => {
			const db = open(":memory:", { dimensions: 8 });

			// Insert with unified API (vector field)
			await db.set([
				{ id: "doc1", vector: new Float32Array(8).fill(0.1), metadata: { category: "A" } },
				{ id: "doc2", vector: new Float32Array(8).fill(0.2), metadata: { category: "B" } },
				{ id: "doc3", vector: new Float32Array(8).fill(0.3), metadata: { category: "A" } },
			]);

			// Search with options object
			const results = await db.search([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 2, {
				ef: 100,
				filter: { category: "A" },
			});

			expect(results).toHaveLength(2);
			expect(results.every(r => r.metadata.category === "A")).toBe(true);
		});

		it("should maintain backward compatibility with positional args on single-vector store", async () => {
			const db = open(":memory:", { dimensions: 8 });

			await db.set([
				{ id: "doc1", vector: new Float32Array(8).fill(0.1) },
				{ id: "doc2", vector: new Float32Array(8).fill(0.2) },
			]);

			// Old positional args style: search(query, k, ef, filter, maxDistance)
			const results = await db.search([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 2, 100);

			expect(results).toHaveLength(2);
		});
	});

	describe("reranking quality", () => {
		let db: VectorDatabase;

		beforeEach(async () => {
			db = open(":memory:", { dimensions: 16, multiVector: { dProj: null } });

			// Create docs with overlapping patterns where reranking matters
			const items = Array.from({ length: 50 }, (_, i) => {
				const numTokens = 3 + (i % 5); // 3-7 tokens
				const vectors = Array.from({ length: numTokens }, (_, j) => {
					const vec = new Float32Array(16).fill(0);
					vec[i % 16] = 1.0; // One hot component
					vec[(i + j) % 16] += 0.5;
					return vec;
				});
				return { id: `doc${i}`, vectors, metadata: { numTokens } };
			});
			await db.set(items);
		});

		it("should return valid results with and without reranking", async () => {
			const query = [
				(() => {
					const v = new Float32Array(16).fill(0);
					v[0] = 1.0;
					return v;
				})(),
				(() => {
					const v = new Float32Array(16).fill(0);
					v[1] = 1.0;
					return v;
				})(),
			];

			const resultsNoRerank = await db.search(query, 10, { rerank: false });
			const resultsRerank = await db.search(query, 10, { rerank: true });

			expect(resultsNoRerank).toHaveLength(10);
			expect(resultsRerank).toHaveLength(10);

			// All results should have valid doc IDs
			expect(resultsNoRerank.every((r) => r.id.startsWith("doc"))).toBe(true);
			expect(resultsRerank.every((r) => r.id.startsWith("doc"))).toBe(true);
		});
	});

	describe("persistence (MUV-13)", () => {
		const fs = require("fs");
		const os = require("os");
		const path = require("path");

		function tempPath(): string {
			return path.join(os.tmpdir(), `omendb_test_${Date.now()}_${Math.random().toString(36).slice(2)}`);
		}

		it("should persist multi-vector data across close/reopen", async () => {
			const dbPath = tempPath();
			try {
				// Create and populate (dProj: null for small dim)
				const db1 = open(dbPath, { dimensions: 8, multiVector: { dProj: null } });
				await db1.set([
					{ id: "doc1", vectors: [new Float32Array(8).fill(0.1), new Float32Array(8).fill(0.2)], metadata: { title: "first" } },
					{ id: "doc2", vectors: [new Float32Array(8).fill(0.3)], metadata: { title: "second" } },
				]);
				db1.flush();
				expect(db1.count()).toBe(2);
				db1.close();

				// Reopen and verify
				const db2 = open(dbPath, { dimensions: 8 });
				expect(db2.isMultiVector).toBe(true);
				expect(db2.count()).toBe(2);

				// Verify search works
				const results = await db2.search([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], 2);
				expect(results.length).toBe(2);
				db2.close();
			} finally {
				// Cleanup
				try { fs.unlinkSync(dbPath + ".omen"); } catch {}
				try { fs.unlinkSync(dbPath + ".wal"); } catch {}
			}
		});

		it("should support reranking after reload", async () => {
			const dbPath = tempPath();
			try {
				// Create store with docs (dProj: null for small dim)
				const db1 = open(dbPath, { dimensions: 4, multiVector: { dProj: null } });
				await db1.set([
					{ id: "doc1", vectors: [[1, 0, 0, 0], [0, 1, 0, 0]], metadata: {} },
					{ id: "doc2", vectors: [[0.3, 0.3, 0.3, 0]], metadata: {} },
				]);
				db1.flush();
				db1.close();

				// Reopen and search with reranking
				const db2 = open(dbPath, { dimensions: 4 });
				expect(db2.isMultiVector).toBe(true);

				const query = [[1, 0, 0, 0], [0, 1, 0, 0]];
				const results = await db2.search(query, 2, { rerank: true });

				expect(results.length).toBe(2);
				// doc1 should be first (better MaxSim match)
				expect(results[0].id).toBe("doc1");
				db2.close();
			} finally {
				try { fs.unlinkSync(dbPath + ".omen"); } catch {}
				try { fs.unlinkSync(dbPath + ".wal"); } catch {}
			}
		});

		it("should handle large persistent multi-vector store", async () => {
			const dbPath = tempPath();
			try {
				// Create store with 100 docs
				const db1 = open(dbPath, { dimensions: 32, multiVector: true });
				const items = Array.from({ length: 100 }, (_, i) => ({
					id: `doc${i}`,
					vectors: Array.from({ length: (i % 5) + 1 }, () => {
						const vec = new Float32Array(32);
						for (let j = 0; j < 32; j++) vec[j] = Math.random();
						return vec;
					}),
					metadata: { idx: i },
				}));
				await db1.set(items);
				db1.flush();
				db1.close();

				// Reopen and verify
				const db2 = open(dbPath, { dimensions: 32 });
				expect(db2.isMultiVector).toBe(true);
				expect(db2.count()).toBe(100);

				// Search should work
				const query = Array.from({ length: 3 }, () => {
					const vec = new Float32Array(32);
					for (let j = 0; j < 32; j++) vec[j] = Math.random();
					return vec;
				});
				const results = await db2.search(query, 5);
				expect(results.length).toBe(5);
				db2.close();
			} finally {
				try { fs.unlinkSync(dbPath + ".omen"); } catch {}
				try { fs.unlinkSync(dbPath + ".wal"); } catch {}
			}
		}, 180_000);
	});

	describe("scale", () => {
		it("should handle 1000 documents with 10 tokens each", { timeout: 120_000 }, async () => {
			const db = open(":memory:", { dimensions: 32, multiVector: true });

			// Insert 1000 docs
			const items = Array.from({ length: 1000 }, (_, i) => ({
				id: `doc${i}`,
				vectors: Array.from({ length: 10 }, () => {
					const vec = new Float32Array(32);
					for (let j = 0; j < 32; j++) {
						vec[j] = Math.random();
					}
					return vec;
				}),
				metadata: { index: i },
			}));
			await db.set(items);
			expect(db.count()).toBe(1000);

			// Search
			const query = Array.from({ length: 5 }, () => {
				const vec = new Float32Array(32);
				for (let j = 0; j < 32; j++) {
					vec[j] = Math.random();
				}
				return vec;
			});
			const results = await db.search(query, 10);
			expect(results).toHaveLength(10);
		});

		it("should handle variable token counts", async () => {
			const db = open(":memory:", { dimensions: 16, multiVector: { dProj: null } });

			// 1 to 20 tokens per doc
			const items = Array.from({ length: 100 }, (_, i) => ({
				id: `doc${i}`,
				vectors: Array.from({ length: 1 + (i % 20) }, () => {
					const vec = new Float32Array(16);
					for (let j = 0; j < 16; j++) {
						vec[j] = Math.random();
					}
					return vec;
				}),
				metadata: { numTokens: 1 + (i % 20) },
			}));
			await db.set(items);
			expect(db.count()).toBe(100);

			// Search with varying query sizes
			for (const numQueryTokens of [1, 5, 10]) {
				const query = Array.from({ length: numQueryTokens }, () => {
					const vec = new Float32Array(16);
					for (let j = 0; j < 16; j++) {
						vec[j] = Math.random();
					}
					return vec;
				});
				const results = await db.search(query, 5);
				expect(results).toHaveLength(5);
			}
		});
	});
});
