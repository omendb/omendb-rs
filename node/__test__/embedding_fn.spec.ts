import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { create, open, VectorDatabase } from "../index.js";
import { mkdtemp, rm } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";

/**
 * Deterministic embedding function for testing.
 * Maps each text to a 4D vector based on simple hash.
 */
function fakeEmbedder(texts: string[]): Float32Array[] {
	return texts.map((text) => {
		let h = 0;
		for (let i = 0; i < text.length; i++) {
			h = (Math.imul(31, h) + text.charCodeAt(i)) | 0;
		}
		h = Math.abs(h) % 1000;
		const raw = [
			h / 1000,
			((h * 3) % 1000) / 1000,
			((h * 7) % 1000) / 1000,
			((h * 11) % 1000) / 1000,
		];
		const norm = Math.sqrt(raw.reduce((s, v) => s + v * v, 0));
		const vec = norm > 0 ? raw.map((v) => v / norm) : raw;
		return new Float32Array(vec);
	});
}

describe("Embedding Function", () => {
	let tempDir: string;
	let dbPath: string;

	beforeEach(async () => {
		tempDir = await mkdtemp(join(tmpdir(), "omendb-embfn-"));
		dbPath = join(tempDir, "testdb");
	});

	afterEach(async () => {
		await new Promise((r) => setTimeout(r, 100));
		try {
			await rm(tempDir, { recursive: true, force: true });
		} catch {}
	});

	describe("open with embeddingFn", () => {
		it("should accept embeddingFn parameter", () => {
			const db = create(dbPath, { dense: { dim: 4 } }, fakeEmbedder);
			expect(db.hasEmbeddingFn).toBe(true);
		});

		it("should default to no embeddingFn", () => {
			const db = create(dbPath, { dense: { dim: 4 } });
			expect(db.hasEmbeddingFn).toBe(false);
		});

		it("should work with in-memory database", () => {
			const db = open(":memory:", { dimensions: 4 }, fakeEmbedder);
			expect(db.hasEmbeddingFn).toBe(true);
		});
	});

	describe("set with document", () => {
		it("should auto-embed single document", async () => {
			const db = create(dbPath, { dense: { dim: 4 } }, fakeEmbedder);
			const count = await db.set([{ id: "d1", document: "hello world" }]);
			expect(count).toBe(1);
			expect(db.count()).toBe(1);
		});

		it("should auto-embed batch of documents", async () => {
			const db = create(dbPath, { dense: { dim: 4 } }, fakeEmbedder);
			const count = await db.set([
				{ id: "d1", document: "hello" },
				{ id: "d2", document: "world" },
				{ id: "d3", document: "foo" },
			]);
			expect(count).toBe(3);
			expect(db.count()).toBe(3);
		});

		it("should allow explicit vectors alongside documents", async () => {
			const db = create(dbPath, { dense: { dim: 4 } }, fakeEmbedder);
			const count = await db.set([
				{ id: "v1", vector: new Float32Array([1, 0, 0, 0]) },
				{ id: "d1", document: "hello world" },
			]);
			expect(count).toBe(2);
		});

		it("should error when document provided without embeddingFn", async () => {
			const db = create(dbPath, { dense: { dim: 4 } });
			await expect(
				db.set([{ id: "d1", document: "hello" }]),
			).rejects.toThrow(/embedding function/i);
		});

		it("should error when both vector and document provided", async () => {
			const db = create(dbPath, { dense: { dim: 4 } }, fakeEmbedder);
			await expect(
				db.set([
					{
						id: "d1",
						vector: new Float32Array([1, 0, 0, 0]),
						document: "hello",
					},
				]),
			).rejects.toThrow(/cannot have both/i);
		});
	});

	describe("search with string query", () => {
		let db: VectorDatabase;

		beforeEach(async () => {
			db = create(dbPath, { dense: { dim: 4 } }, fakeEmbedder);
			await db.set([
				{ id: "d1", document: "hello" },
				{ id: "d2", document: "world" },
				{ id: "d3", document: "foo" },
			]);
		});

		it("should search by string query", async () => {
			const results = await db.search("hello", 2);
			expect(results.length).toBe(2);
			// First result should be "hello" since query is identical
			expect(results[0].id).toBe("d1");
		});

		it("should error for string query without embeddingFn", async () => {
			const db2 = create(":memory:", { dense: { dim: 4 } });
			await db2.set([
				{ id: "v1", vector: new Float32Array([1, 0, 0, 0]) },
			]);
			await expect(db2.search("hello", 1)).rejects.toThrow(
				/embedding function/i,
			);
		});

		it("should still work with vector query", async () => {
			const results = await db.search([1, 0, 0, 0], 1);
			expect(results.length).toBe(1);
		});

		it("should still work with Float32Array query", async () => {
			const results = await db.search(new Float32Array([1, 0, 0, 0]), 1);
			expect(results.length).toBe(1);
		});
	});

	describe("searchHybrid with string query", () => {
		let db: VectorDatabase;

		beforeEach(async () => {
			db = create(dbPath, { dense: { dim: 4 } }, fakeEmbedder);
			await db.set([
				{
					id: "d1",
					document: "machine learning",
					text: "machine learning algorithms",
				},
				{
					id: "d2",
					document: "web development",
					text: "web development frameworks",
				},
			]);
			db.flush();
		});

		it("should search with string query (auto-embed + text)", async () => {
			const results = await db.searchHybrid("machine learning", null, 2);
			expect(results.length).toBeGreaterThanOrEqual(1);
		});

		it("should still work with vector + text args", async () => {
			const results = await db.searchHybrid(
				[1.0, 0.0, 0.0, 0.0],
				"machine",
				2,
			);
			expect(results.length).toBeGreaterThanOrEqual(1);
		});
	});

	describe("collection inheritance", () => {
		it("should inherit embeddingFn from parent", async () => {
			const db = create(dbPath, { dense: { dim: 4 } }, fakeEmbedder);
			const col = db.collection("test_col");
			expect(col.hasEmbeddingFn).toBe(true);

			const count = await col.set([{ id: "d1", document: "hello" }]);
			expect(count).toBe(1);
		});

		it("should allow embeddingFn override on collection", () => {
			const db = create(dbPath, { dense: { dim: 4 } }, fakeEmbedder);
			const otherEmbedder = (texts: string[]) =>
				texts.map(() => new Float32Array([0.5, 0.5, 0.5, 0.5]));
			const col = db.collection("test_col", otherEmbedder);
			expect(col.hasEmbeddingFn).toBe(true);
		});

		it("should not have embeddingFn when parent has none", () => {
			const db = create(dbPath, { dense: { dim: 4 } });
			const col = db.collection("test_col");
			expect(col.hasEmbeddingFn).toBe(false);
		});
	});
});
