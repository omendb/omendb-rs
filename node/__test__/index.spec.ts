import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { create, open, VectorDatabase } from "../index.js";
import { mkdtemp, rm } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";

describe("VectorDatabase", () => {
	describe("in-memory", () => {
		let db: VectorDatabase;

		beforeEach(() => {
			db = open(":memory:", { dimensions: 128 });
		});

		describe("set", () => {
			it("should insert single vector", async () => {
				const count = await db.set([{ id: "doc1", vector: Array(128).fill(0.1) }]);
				expect(count).toBe(1);
				expect(db.count()).toBe(1);
			});

			it("should insert batch of vectors", async () => {
				const items = Array.from({ length: 100 }, (_, i) => ({
					id: `doc${i}`,
					vector: Array(128).fill(i / 100),
					metadata: { index: i },
				}));
				const count = await db.set(items);
				expect(count).toBe(100);
				expect(db.count()).toBe(100);
			});

			it("should handle metadata", async () => {
				await db.set([
					{
						id: "doc1",
						vector: Array(128).fill(0.1),
						metadata: { title: "Test", tags: ["a", "b"], count: 42 },
					},
				]);
				const doc = db.get("doc1");
				expect(doc?.metadata).toEqual({
					title: "Test",
					tags: ["a", "b"],
					count: 42,
				});
			});

			it("should handle text field and auto-store in metadata", async () => {
				await db.set([
					{
						id: "doc1",
						vector: Array(128).fill(0.1),
						text: "Hello world",
					},
				]);
				const doc = db.get("doc1");
				expect(doc?.metadata).toEqual({ text: "Hello world" });
			});

			it("should replace existing vector with same id", async () => {
				await db.set([{ id: "doc1", vector: Array(128).fill(0.1) }]);
				await db.set([
					{
						id: "doc1",
						vector: Array(128).fill(0.9),
						metadata: { new: true },
					},
				]);
				expect(db.count()).toBe(1);
				const doc = db.get("doc1");
				expect(doc?.metadata).toEqual({ new: true });
			});
		});

		it("should create from explicit schema", async () => {
			const schemaDb = create(":memory:", {
				name: "docs",
				metric: "l2",
				dense: { dim: 8, quantization: "sq8" },
				text: { tokenizer: "code", writerBufferMb: 20 },
			});

			expect(schemaDb.schema()).toMatchObject({
				name: "docs",
				metric: "l2",
				dense: {
					dim: 8,
					quantization: "sq8",
					mutableIndex: "hnsw",
					frozenIndex: "hnsw",
				},
				text: { tokenizer: "code", writerBufferMb: 20 },
			});
			expect(schemaDb.info().schema).toEqual(schemaDb.schema());
			schemaDb.close();
		});

		describe("search", () => {
			beforeEach(async () => {
				await db.set([
					{ id: "a", vector: Array(128).fill(0.1) },
					{ id: "b", vector: Array(128).fill(0.5) },
					{ id: "c", vector: Array(128).fill(0.9) },
				]);
			});

			it("should return k nearest neighbors", async () => {
				const results = await db.search(Array(128).fill(0.5), 2);
				expect(results).toHaveLength(2);
				expect(results[0].id).toBe("b"); // Closest to 0.5
			});

			it("should return results with distance, score, and metadata", async () => {
				await db.set([
					{
						id: "d",
						vector: Array(128).fill(0.5),
						metadata: { key: "value" },
					},
				]);
				const results = await db.search(Array(128).fill(0.5), 1);
				expect(results[0]).toHaveProperty("id");
				expect(results[0]).toHaveProperty("distance");
				expect(results[0]).toHaveProperty("score");
				expect(results[0]).toHaveProperty("metadata");
				expect(typeof results[0].distance).toBe("number");
				expect(typeof results[0].score).toBe("number");
				expect(results[0].score).toBeGreaterThan(0);
				expect(results[0].score).toBeLessThanOrEqual(1);
			});

			it("should accept Float32Array", async () => {
				const query = new Float32Array(128).fill(0.5);
				const results = await db.search(query, 2);
				expect(results).toHaveLength(2);
			});

			it("should respect ef option", async () => {
				const results = await db.search(Array(128).fill(0.5), 2, { ef: 200 });
				expect(results).toHaveLength(2);
			});

			it("should filter results by metadata", async () => {
				// Add vectors with different categories
				await db.set([
					{
						id: "cat1",
						vector: Array(128).fill(0.4),
						metadata: { category: "A" },
					},
					{
						id: "cat2",
						vector: Array(128).fill(0.6),
						metadata: { category: "B" },
					},
				]);

				// Search with filter - should only return category A
				const filtered = await db.search(Array(128).fill(0.5), 10, {
					filter: { category: "A" },
				});
				expect(filtered.every((r) => r.metadata?.category === "A")).toBe(true);
			});

			it("should filter with comparison operators", async () => {
				await db.set([
					{ id: "n1", vector: Array(128).fill(0.4), metadata: { score: 10 } },
					{ id: "n2", vector: Array(128).fill(0.5), metadata: { score: 50 } },
					{ id: "n3", vector: Array(128).fill(0.6), metadata: { score: 90 } },
				]);

				// Filter for score > 40 (should match n2=50 and n3=90)
				const filtered = await db.search(Array(128).fill(0.5), 10, {
					filter: { score: { $gt: 40 } },
				});
				expect(filtered).toHaveLength(2);
				expect(filtered.every((r) => (r.metadata?.score as number) > 40)).toBe(
					true,
				);
			});

			it("should respect maxDistance option", async () => {
				const results = await db.search(Array(128).fill(0.5), 10, {
					maxDistance: 0.1,
				});
				// Should filter out distant results
				expect(results.every((r) => r.distance <= 0.1)).toBe(true);
			});

			it("should reject cosine zero-vector queries", async () => {
				const cosineDb = open(":memory:", { dimensions: 3, metric: "cosine" });
				await cosineDb.set([{ id: "doc", vector: [1, 0, 0] }]);

				await expect(cosineDb.search([0, 0, 0], 1)).rejects.toThrow(
					/zero vector/i,
				);
			});

			it("should filter with $not operator", async () => {
				await db.set([
					{ id: "a1", vector: Array(128).fill(0.3), metadata: { type: "A" } },
					{ id: "b1", vector: Array(128).fill(0.5), metadata: { type: "B" } },
					{ id: "c1", vector: Array(128).fill(0.7), metadata: { type: "C" } },
				]);

				// $not: exclude type A
				const filtered = await db.search(Array(128).fill(0.5), 10, {
					filter: { $not: { type: "A" } },
				});
				expect(filtered.every((r) => r.metadata?.type !== "A")).toBe(true);
				expect(filtered.some((r) => r.metadata?.type === "B")).toBe(true);
			});

			it("should filter with $not and compound conditions", async () => {
				await db.set([
					{ id: "x1", vector: Array(128).fill(0.3), metadata: { cat: "A", val: 10 } },
					{ id: "x2", vector: Array(128).fill(0.5), metadata: { cat: "A", val: 50 } },
					{ id: "x3", vector: Array(128).fill(0.7), metadata: { cat: "B", val: 30 } },
				]);

				// NOT (cat=A AND val>=50) => excludes x2
				const filtered = await db.search(Array(128).fill(0.5), 10, {
					filter: { $not: { $and: [{ cat: "A" }, { val: { $gte: 50 } }] } },
				});
				const ids = filtered.map((r) => r.id);
				expect(ids).not.toContain("x2");
				expect(ids).toContain("x1");
				expect(ids).toContain("x3");
			});
		});

		describe("searchBatch", () => {
			beforeEach(async () => {
				const items = Array.from({ length: 100 }, (_, i) => ({
					id: `doc${i}`,
					vector: Array(128).fill(i / 100),
				}));
				await db.set(items);
			});

			it("should search multiple queries in parallel", async () => {
				const queries = [
					Array(128).fill(0.0),
					Array(128).fill(0.5),
					Array(128).fill(1.0),
				];
				const results = await db.searchBatch(queries, 5);
				expect(results).toHaveLength(3);
				expect(results[0]).toHaveLength(5);
				expect(results[1]).toHaveLength(5);
				expect(results[2]).toHaveLength(5);
			});

			it("should return correct nearest neighbors for each query", async () => {
				const queries = [Array(128).fill(0.0), Array(128).fill(0.99)];
				const results = await db.searchBatch(queries, 1);
				expect(results[0][0].id).toBe("doc0");
				expect(results[1][0].id).toBe("doc99");
			});

			it("should accept Float32Array queries", async () => {
				const queries = [
					new Float32Array(128).fill(0.5),
					new Float32Array(128).fill(0.9),
				];
				const results = await db.searchBatch(queries, 3);
				expect(results).toHaveLength(2);
			});
		});

		describe("get", () => {
			it("should return vector by id", async () => {
				await db.set([
					{
						id: "doc1",
						vector: Array(128).fill(0.5),
						metadata: { key: "value" },
					},
				]);
				const doc = db.get("doc1");
				expect(doc).not.toBeNull();
				expect(doc?.id).toBe("doc1");
				expect(doc?.vector).toHaveLength(128);
				expect(doc?.metadata).toEqual({ key: "value" });
			});

			it("should return null for non-existent id", () => {
				const doc = db.get("nonexistent");
				expect(doc).toBeNull();
			});
		});

		describe("delete", () => {
			beforeEach(async () => {
				await db.set([
					{ id: "doc1", vector: Array(128).fill(0.1) },
					{ id: "doc2", vector: Array(128).fill(0.2) },
					{ id: "doc3", vector: Array(128).fill(0.3) },
				]);
			});

			it("should delete single vector with string", () => {
				const deleted = db.delete("doc1");
				expect(deleted).toBe(1);
				expect(db.count()).toBe(2);
				expect(db.get("doc1")).toBeNull();
			});

			it("should delete multiple vectors with array", () => {
				const deleted = db.delete(["doc1", "doc3"]);
				expect(deleted).toBe(2);
				expect(db.count()).toBe(1);
			});

			it("should return 0 for non-existent ids", () => {
				const deleted = db.delete("nonexistent");
				expect(deleted).toBe(0);
				expect(db.count()).toBe(3);
			});
		});

		describe("update", () => {
			beforeEach(async () => {
				await db.set([
					{
						id: "doc1",
						vector: Array(128).fill(0.1),
						metadata: { old: true },
					},
				]);
			});

			it("should update vector", () => {
				db.update("doc1", { vector: Array(128).fill(0.9) });
				const doc = db.get("doc1");
				expect(doc?.vector[0]).toBeCloseTo(0.9, 1);
			});

			it("should update metadata", () => {
				db.update("doc1", { metadata: { new: true, count: 42 } });
				const doc = db.get("doc1");
				expect(doc?.metadata).toEqual({ new: true, count: 42 });
			});

			it("should update both vector and metadata", () => {
				db.update("doc1", {
					vector: Array(128).fill(0.5),
					metadata: { updated: true },
				});
				const doc = db.get("doc1");
				expect(doc?.vector[0]).toBeCloseTo(0.5, 1);
				expect(doc?.metadata).toEqual({ updated: true });
			});
		});

		describe("efSearch", () => {
			it("should get and set efSearch", () => {
				const initial = db.efSearch;
				expect(typeof initial).toBe("number");

				db.efSearch = 200;
				expect(db.efSearch).toBe(200);

				db.efSearch = 50;
				expect(db.efSearch).toBe(50);
			});
		});

		describe("count", () => {
			it("should return 0 for empty database", () => {
				expect(db.count()).toBe(0);
			});

			it("should return correct count after inserts", async () => {
				await db.set([{ id: "doc1", vector: Array(128).fill(0.1) }]);
				expect(db.count()).toBe(1);

				await db.set([{ id: "doc2", vector: Array(128).fill(0.2) }]);
				expect(db.count()).toBe(2);
			});
		});

		describe("ids", () => {
			it("should return all vector IDs", async () => {
				await db.set([
					{ id: "a", vector: Array(128).fill(0.1) },
					{ id: "b", vector: Array(128).fill(0.2) },
					{ id: "c", vector: Array(128).fill(0.3) },
				]);

				const ids = db.ids();
				expect(ids).toHaveLength(3);
				expect(new Set(ids)).toEqual(new Set(["a", "b", "c"]));
			});

			it("should exclude deleted vectors", async () => {
				await db.set([
					{ id: "a", vector: Array(128).fill(0.1) },
					{ id: "b", vector: Array(128).fill(0.2) },
				]);
				db.delete("a");

				const ids = db.ids();
				expect(ids).toEqual(["b"]);
			});

			it("should return empty array for empty database", () => {
				expect(db.ids()).toEqual([]);
			});
		});

		describe("items", () => {
			it("should return all items with metadata", async () => {
				await db.set([
					{ id: "a", vector: Array(128).fill(0.1), metadata: { x: 1 } },
					{ id: "b", vector: Array(128).fill(0.2), metadata: { x: 2 } },
				]);

				const items = db.items();
				expect(items).toHaveLength(2);

				const byId = Object.fromEntries(items.map((i) => [i.id, i]));
				expect(byId.a.metadata).toEqual({ x: 1 });
				expect(byId.b.metadata).toEqual({ x: 2 });
			});

			it("should exclude deleted vectors", async () => {
				await db.set([
					{ id: "a", vector: Array(128).fill(0.1) },
					{ id: "b", vector: Array(128).fill(0.2) },
				]);
				db.delete("a");

				const items = db.items();
				expect(items).toHaveLength(1);
				expect(items[0].id).toBe("b");
			});
		});

		describe("exists", () => {
			it("should return true for existing ID", async () => {
				await db.set([{ id: "a", vector: Array(128).fill(0.1) }]);
				expect(db.exists("a")).toBe(true);
			});

			it("should return false for non-existent ID", () => {
				expect(db.exists("nonexistent")).toBe(false);
			});

			it("should return false for deleted ID", async () => {
				await db.set([{ id: "a", vector: Array(128).fill(0.1) }]);
				db.delete("a");
				expect(db.exists("a")).toBe(false);
			});
		});

		describe("getBatch", () => {
			beforeEach(async () => {
				await db.set([
					{ id: "a", vector: Array(128).fill(0.1), metadata: { x: 1 } },
					{ id: "b", vector: Array(128).fill(0.2), metadata: { x: 2 } },
				]);
			});

			it("should return multiple vectors by ID", () => {
				const results = db.getBatch(["a", "b", "c"]); // c doesn't exist

				expect(results).toHaveLength(3);
				expect(results[0]?.id).toBe("a");
				expect(results[1]?.id).toBe("b");
				expect(results[2]).toBeNull();
			});

			it("should preserve input order", () => {
				const results = db.getBatch(["b", "a"]);
				expect(results[0]?.id).toBe("b");
				expect(results[1]?.id).toBe("a");
			});

			it("should return empty array for empty input", () => {
				expect(db.getBatch([])).toEqual([]);
			});

			it("should return all null for missing IDs", () => {
				const results = db.getBatch(["x", "y", "z"]);
				expect(results).toEqual([null, null, null]);
			});
		});

		describe("deleteByFilter", () => {
			it("should delete by equality filter", async () => {
				await db.set([
					{
						id: "a",
						vector: Array(128).fill(0.1),
						metadata: { status: "active" },
					},
					{
						id: "b",
						vector: Array(128).fill(0.2),
						metadata: { status: "archived" },
					},
					{
						id: "c",
						vector: Array(128).fill(0.3),
						metadata: { status: "archived" },
					},
				]);

				const count = db.deleteByFilter({ status: "archived" });
				expect(count).toBe(2);
				expect(new Set(db.ids())).toEqual(new Set(["a"]));
			});

			it("should delete with comparison operators", async () => {
				await db.set([
					{ id: "a", vector: Array(128).fill(0.1), metadata: { score: 0.3 } },
					{ id: "b", vector: Array(128).fill(0.2), metadata: { score: 0.7 } },
					{ id: "c", vector: Array(128).fill(0.3), metadata: { score: 0.9 } },
				]);

				const count = db.deleteByFilter({ score: { $lt: 0.5 } });
				expect(count).toBe(1);
				expect(new Set(db.ids())).toEqual(new Set(["b", "c"]));
			});

			it("should return 0 when no match", async () => {
				await db.set([{ id: "a", vector: Array(128).fill(0.1), metadata: { x: 1 } }]);
				const count = db.deleteByFilter({ x: 999 });
				expect(count).toBe(0);
				expect(db.ids()).toEqual(["a"]);
			});

			it("should delete with complex filter", async () => {
				await db.set([
					{
						id: "a",
						vector: Array(128).fill(0.1),
						metadata: { type: "doc", score: 0.5 },
					},
					{
						id: "b",
						vector: Array(128).fill(0.2),
						metadata: { type: "doc", score: 0.9 },
					},
					{
						id: "c",
						vector: Array(128).fill(0.3),
						metadata: { type: "image", score: 0.3 },
					},
				]);

				const count = db.deleteByFilter({
					$and: [{ type: "doc" }, { score: { $lt: 0.8 } }],
				});
				expect(count).toBe(1);
				expect(new Set(db.ids())).toEqual(new Set(["b", "c"]));
			});
		});
	});

	describe("persistent", () => {
		let tempDir: string;
		let dbPath: string;

		beforeEach(async () => {
			tempDir = await mkdtemp(join(tmpdir(), "omendb-test-"));
			dbPath = join(tempDir, "testdb");
		});

		afterEach(async () => {
			// Allow file handles to be released
			await new Promise((r) => setTimeout(r, 100));
			try {
				await rm(tempDir, { recursive: true, force: true });
			} catch {
				// Ignore cleanup errors
			}
		});

		it("should persist and reload data", async () => {
			// Create and save
			const db1 = open(dbPath, { dimensions: 64 });
			await db1.set([
				{
					id: "persist1",
					vector: Array(64).fill(0.5),
					metadata: { saved: true },
				},
				{ id: "persist2", vector: Array(64).fill(0.9) },
			]);
			db1.close(); // Release file lock

			// Reopen and verify
			const db2 = open(dbPath, { dimensions: 64 });
			expect(db2.count()).toBe(2);

			const doc = db2.get("persist1");
			expect(doc?.id).toBe("persist1");
			expect(doc?.metadata).toEqual({ saved: true });
		});

		it("should support collections", async () => {
			const db = open(dbPath, { dimensions: 64 });

			const users = db.collection("users");
			const products = db.collection("products");

			await users.set([{ id: "user1", vector: Array(64).fill(0.1) }]);
			await products.set([{ id: "prod1", vector: Array(64).fill(0.2) }]);

			expect(users.count()).toBe(1);
			expect(products.count()).toBe(1);

			// IDs are independent
			await users.set([{ id: "item1", vector: Array(64).fill(0.3) }]);
			await products.set([{ id: "item1", vector: Array(64).fill(0.4) }]);

			expect(users.count()).toBe(2);
			expect(products.count()).toBe(2);
		});

		it("should preserve multi-vector mode in collections", async () => {
			const db = open(dbPath, { dimensions: 8, multiVector: { dProj: null } });
			const docs = db.collection("docs");

			expect(docs.isMultiVector).toBe(true);

			await docs.set([
				{
					id: "doc1",
					vectors: [Array(8).fill(0.1), Array(8).fill(0.2)],
					metadata: { kind: "multi" },
				},
			]);

			const results = await docs.search([Array(8).fill(0.1)], 1);
			expect(results).toHaveLength(1);
			expect(results[0]?.id).toBe("doc1");
			expect(results[0]?.metadata?.kind).toBe("multi");
		});

		it("should list collections", () => {
			const db = open(dbPath, { dimensions: 64 });
			db.collection("alpha");
			db.collection("beta");
			db.collection("gamma");

			const names = db.collections();
			expect(names).toEqual(["alpha", "beta", "gamma"]);
		});

		it("should delete collections", async () => {
			const db = open(dbPath, { dimensions: 64 });
			const col = db.collection("todelete");
			await col.set([{ id: "doc1", vector: Array(64).fill(0.1) }]);

			db.deleteCollection("todelete");
			expect(db.collections()).not.toContain("todelete");
		});
	});

	describe("info", () => {
		it("should return comprehensive diagnostics", async () => {
			const db = open(":memory:", { dimensions: 4 });
			await db.set([{ id: "a", vector: [1, 0, 0, 0] }]);
			const info = db.info();
			expect(info.vectorCount).toBe(1);
			expect(info.deletedCount).toBe(0);
			expect(info.dimensions).toBe(4);
			expect(typeof info.metric).toBe("string");
			expect(typeof info.totalMemoryBytes).toBe("number");
			expect(info.totalMemoryBytes).toBeGreaterThan(0);
			expect(info.isPersistent).toBe(false);
			expect(info.hnswM).toBe(16);
			expect(info.quantization).toBe(false);
			expect(info.schema.metric).toBe("l2");
			expect(info.schema.dense?.dim).toBe(4);
			db.close();
		});

		it("should expose the authoritative schema directly", () => {
			const db = open(":memory:", {
				dimensions: 4,
				textSearch: { writerBufferMb: 18, tokenizer: "code" },
			});
			const schema = db.schema();

			expect(schema.metric).toBe("l2");
			expect(schema.dense?.dim).toBe(4);
			expect(schema.text?.tokenizer).toBe("code");
			expect(schema.text?.writerBufferMb).toBe(18);
			expect(db.info().schema).toEqual(schema);
			db.close();
		});
	});

	describe("mergeFrom", () => {
		it("should merge without prefix", async () => {
			const db1 = open(":memory:", { dimensions: 4 });
			const db2 = open(":memory:", { dimensions: 4 });
			await db1.set([{ id: "a", vector: [1, 0, 0, 0] }]);
			await db2.set([{ id: "b", vector: [0, 1, 0, 0] }]);
			const count = db1.mergeFrom(db2);
			expect(count).toBe(1);
			expect(new Set(db1.ids())).toEqual(new Set(["a", "b"]));
			db1.close();
			db2.close();
		});

		it("should merge with key prefix", async () => {
			const db1 = open(":memory:", { dimensions: 4 });
			const db2 = open(":memory:", { dimensions: 4 });
			await db1.set([{ id: "a", vector: [1, 0, 0, 0] }]);
			await db2.set([{ id: "b", vector: [0, 1, 0, 0] }]);
			const count = db1.mergeFrom(db2, "src_");
			expect(count).toBe(1);
			expect(new Set(db1.ids())).toEqual(new Set(["a", "src_b"]));
			db1.close();
			db2.close();
		});
	});

	describe("error handling", () => {
		it("should reject invalid m parameter", () => {
			expect(() => open(":memory:", { dimensions: 128, m: 2 })).toThrow();
			expect(() => open(":memory:", { dimensions: 128, m: 100 })).toThrow();
		});

		it("should reject numeric quantization modes", () => {
			expect(() =>
				open(":memory:", { dimensions: 128, quantization: 8 as never })
			).toThrow(/quantization/);
		});

		it("should accept the scalar quantization alias", () => {
			const db = open(":memory:", { dimensions: 128, quantization: "scalar" });
			expect(db.info().quantization).toBe(true);
			db.close();
		});

		it("should reject empty collection name", () => {
			const db = open(":memory:", { dimensions: 128 });
			// Collections require persistent storage
			expect(() => db.collection("test")).toThrow();
		});
	});

	describe("dimension inference", () => {
		it("should infer dimensions from first insert for regular stores", async () => {
			const db = open(":memory:");
			expect(db.dimensions).toBe(0);

			await db.set([{ id: "v1", vector: [0.1, 0.2, 0.3] }]);

			expect(db.dimensions).toBe(3);
			const results = await db.search([0.1, 0.2, 0.3], 1);
			expect(results).toHaveLength(1);
		});

		it("should require explicit dimensions for multi-vector stores", () => {
			expect(() => open(":memory:", { multiVector: true })).toThrow(/dimensions/);
		});
	});
});
