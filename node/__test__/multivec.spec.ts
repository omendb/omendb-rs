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

		it("should reject multi-vector with persistence", () => {
			expect(() => open("./test_mv", { dimensions: 128, multiVector: true })).toThrow(
				/in-memory/
			);
		});

		it("should reject multi-vector with quantization", () => {
			expect(() =>
				open(":memory:", { dimensions: 128, multiVector: true, quantization: true })
			).toThrow(/quantization/);
		});
	});

	describe("insert", () => {
		let db: VectorDatabase;

		beforeEach(() => {
			db = open(":memory:", { dimensions: 8, multiVector: true });
		});

		it("should insert a single multi-vector document", () => {
			const vectors = [
				new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
				new Float32Array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
			];
			const indices = db.setMultiVec([{ id: "doc1", vectors, metadata: { title: "Test" } }]);
			expect(indices).toHaveLength(1);
			expect(db.count()).toBe(1);
		});

		it("should insert multiple multi-vector documents", () => {
			const items = Array.from({ length: 10 }, (_, i) => ({
				id: `doc${i}`,
				vectors: Array.from({ length: 5 }, () =>
					new Float32Array(8).fill(i / 10)
				),
				metadata: { index: i },
			}));
			const indices = db.setMultiVec(items);
			expect(indices).toHaveLength(10);
			expect(db.count()).toBe(10);
		});

		it("should reject empty vectors array", () => {
			expect(() =>
				db.setMultiVec([{ id: "doc1", vectors: [], metadata: {} }])
			).toThrow(/must not be empty/);
		});

		it("should reject setMultiVec on regular store", () => {
			const regularDb = open(":memory:", { dimensions: 8 });
			expect(() =>
				regularDb.setMultiVec([
					{ id: "doc1", vectors: [new Float32Array(8).fill(0.1)], metadata: {} },
				])
			).toThrow(/multi-vector store/);
		});
	});

	describe("search", () => {
		let db: VectorDatabase;

		beforeEach(() => {
			db = open(":memory:", { dimensions: 8, multiVector: true });

			// Insert 100 docs with distinct patterns
			const items = Array.from({ length: 100 }, (_, i) => {
				const base = i / 100;
				const vectors = Array.from({ length: 5 }, (_, j) =>
					new Float32Array(8).fill(base + j * 0.01)
				);
				return { id: `doc${i}`, vectors, metadata: { index: i } };
			});
			db.setMultiVec(items);
		});

		it("should perform basic multi-vector search", () => {
			const query = [
				[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
				[0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51],
			];
			const results = db.searchMultiVec(query, 5);

			expect(results).toHaveLength(5);
			expect(results.every((r) => "id" in r && "distance" in r && "metadata" in r)).toBe(true);
		});

		it("should search with Float32Array query", () => {
			const query = [
				new Float32Array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
				new Float32Array([0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51]),
			];
			const results = db.searchMultiVec(query, 5);
			expect(results).toHaveLength(5);
		});

		it("should search with rerank disabled", () => {
			const query = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]];
			const results = db.searchMultiVec(query, 5, false);
			expect(results).toHaveLength(5);
		});

		it("should search with custom rerank factor", () => {
			const query = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]];
			const results = db.searchMultiVec(query, 5, true, 8);
			expect(results).toHaveLength(5);
		});

		it("should return metadata in results", () => {
			const query = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]];
			const results = db.searchMultiVec(query, 1);

			expect(results).toHaveLength(1);
			expect(results[0].metadata).toHaveProperty("index");
		});

		it("should reject searchMultiVec on regular store", () => {
			const regularDb = open(":memory:", { dimensions: 8 });
			regularDb.set([{ id: "doc1", vector: new Float32Array(8).fill(0.1) }]);

			expect(() =>
				regularDb.searchMultiVec([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], 1)
			).toThrow(/multi-vector store/);
		});
	});

	describe("reranking quality", () => {
		let db: VectorDatabase;

		beforeEach(() => {
			db = open(":memory:", { dimensions: 16, multiVector: true });

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
			db.setMultiVec(items);
		});

		it("should return valid results with and without reranking", () => {
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

			const resultsNoRerank = db.searchMultiVec(query, 10, false);
			const resultsRerank = db.searchMultiVec(query, 10, true);

			expect(resultsNoRerank).toHaveLength(10);
			expect(resultsRerank).toHaveLength(10);

			// All results should have valid doc IDs
			expect(resultsNoRerank.every((r) => r.id.startsWith("doc"))).toBe(true);
			expect(resultsRerank.every((r) => r.id.startsWith("doc"))).toBe(true);
		});
	});

	describe("scale", () => {
		it("should handle 1000 documents with 10 tokens each", () => {
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
			db.setMultiVec(items);
			expect(db.count()).toBe(1000);

			// Search
			const query = Array.from({ length: 5 }, () => {
				const vec = new Float32Array(32);
				for (let j = 0; j < 32; j++) {
					vec[j] = Math.random();
				}
				return vec;
			});
			const results = db.searchMultiVec(query, 10);
			expect(results).toHaveLength(10);
		});

		it("should handle variable token counts", () => {
			const db = open(":memory:", { dimensions: 16, multiVector: true });

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
			db.setMultiVec(items);
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
				const results = db.searchMultiVec(query, 5);
				expect(results).toHaveLength(5);
			}
		});
	});
});
