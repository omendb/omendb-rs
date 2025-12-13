import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { open, VectorDatabase } from "../index.js";
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
			it("should insert single vector", () => {
				const indices = db.set([{ id: "doc1", vector: Array(128).fill(0.1) }]);
				expect(indices).toHaveLength(1);
				expect(db.count).toBe(1);
			});

			it("should insert batch of vectors", () => {
				const items = Array.from({ length: 100 }, (_, i) => ({
					id: `doc${i}`,
					vector: Array(128).fill(i / 100),
					metadata: { index: i },
				}));
				const indices = db.set(items);
				expect(indices).toHaveLength(100);
				expect(db.count).toBe(100);
			});

			it("should handle metadata", () => {
				db.set([
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

			it("should handle document field", () => {
				db.set([
					{
						id: "doc1",
						vector: Array(128).fill(0.1),
						document: "Hello world",
					},
				]);
				const doc = db.get("doc1");
				expect(doc?.metadata).toEqual({ document: "Hello world" });
			});

			it("should replace existing vector with same id", () => {
				db.set([{ id: "doc1", vector: Array(128).fill(0.1) }]);
				db.set([
					{
						id: "doc1",
						vector: Array(128).fill(0.9),
						metadata: { new: true },
					},
				]);
				expect(db.count).toBe(1);
				const doc = db.get("doc1");
				expect(doc?.metadata).toEqual({ new: true });
			});
		});

		describe("search", () => {
			beforeEach(() => {
				db.set([
					{ id: "a", vector: Array(128).fill(0.1) },
					{ id: "b", vector: Array(128).fill(0.5) },
					{ id: "c", vector: Array(128).fill(0.9) },
				]);
			});

			it("should return k nearest neighbors", () => {
				const results = db.search(Array(128).fill(0.5), 2);
				expect(results).toHaveLength(2);
				expect(results[0].id).toBe("b"); // Closest to 0.5
			});

			it("should return results with distance and metadata", () => {
				db.set([
					{
						id: "d",
						vector: Array(128).fill(0.5),
						metadata: { key: "value" },
					},
				]);
				const results = db.search(Array(128).fill(0.5), 1);
				expect(results[0]).toHaveProperty("id");
				expect(results[0]).toHaveProperty("distance");
				expect(results[0]).toHaveProperty("metadata");
				expect(typeof results[0].distance).toBe("number");
			});

			it("should accept Float32Array", () => {
				const query = new Float32Array(128).fill(0.5);
				const results = db.search(query, 2);
				expect(results).toHaveLength(2);
			});

			it("should respect ef parameter", () => {
				const results = db.search(Array(128).fill(0.5), 2, 200);
				expect(results).toHaveLength(2);
			});

			it("should filter results by metadata", () => {
				// Add vectors with different categories
				db.set([
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
				const filtered = db.search(Array(128).fill(0.5), 10, undefined, {
					category: "A",
				});
				expect(filtered.every((r) => r.metadata?.category === "A")).toBe(true);
			});

			it("should filter with comparison operators", () => {
				db.set([
					{ id: "n1", vector: Array(128).fill(0.4), metadata: { score: 10 } },
					{ id: "n2", vector: Array(128).fill(0.5), metadata: { score: 50 } },
					{ id: "n3", vector: Array(128).fill(0.6), metadata: { score: 90 } },
				]);

				// Filter for score > 40 (should match n2=50 and n3=90)
				const filtered = db.search(Array(128).fill(0.5), 10, undefined, {
					score: { $gt: 40 },
				});
				expect(filtered).toHaveLength(2);
				expect(filtered.every((r) => (r.metadata?.score as number) > 40)).toBe(
					true,
				);
			});
		});

		describe("searchBatch", () => {
			beforeEach(() => {
				const items = Array.from({ length: 100 }, (_, i) => ({
					id: `doc${i}`,
					vector: Array(128).fill(i / 100),
				}));
				db.set(items);
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
			it("should return vector by id", () => {
				db.set([
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
			beforeEach(() => {
				db.set([
					{ id: "doc1", vector: Array(128).fill(0.1) },
					{ id: "doc2", vector: Array(128).fill(0.2) },
					{ id: "doc3", vector: Array(128).fill(0.3) },
				]);
			});

			it("should delete single vector", () => {
				const deleted = db.delete(["doc1"]);
				expect(deleted).toBe(1);
				expect(db.count).toBe(2);
				expect(db.get("doc1")).toBeNull();
			});

			it("should delete multiple vectors", () => {
				const deleted = db.delete(["doc1", "doc3"]);
				expect(deleted).toBe(2);
				expect(db.count).toBe(1);
			});

			it("should return 0 for non-existent ids", () => {
				const deleted = db.delete(["nonexistent"]);
				expect(deleted).toBe(0);
				expect(db.count).toBe(3);
			});
		});

		describe("update", () => {
			beforeEach(() => {
				db.set([
					{
						id: "doc1",
						vector: Array(128).fill(0.1),
						metadata: { old: true },
					},
				]);
			});

			it("should update vector", () => {
				db.update("doc1", Array(128).fill(0.9));
				const doc = db.get("doc1");
				expect(doc?.vector[0]).toBeCloseTo(0.9, 1);
			});

			it("should update metadata", () => {
				db.update("doc1", Array(128).fill(0.1), { new: true, count: 42 });
				const doc = db.get("doc1");
				expect(doc?.metadata).toEqual({ new: true, count: 42 });
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
				expect(db.count).toBe(0);
			});

			it("should return correct count after inserts", () => {
				db.set([{ id: "doc1", vector: Array(128).fill(0.1) }]);
				expect(db.count).toBe(1);

				db.set([{ id: "doc2", vector: Array(128).fill(0.2) }]);
				expect(db.count).toBe(2);
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

		it("should persist and reload data", () => {
			// Create and save
			const db1 = open(dbPath, { dimensions: 64 });
			db1.set([
				{
					id: "persist1",
					vector: Array(64).fill(0.5),
					metadata: { saved: true },
				},
				{ id: "persist2", vector: Array(64).fill(0.9) },
			]);
			db1.flush();

			// Reopen and verify
			const db2 = open(dbPath, { dimensions: 64 });
			expect(db2.count).toBe(2);

			const doc = db2.get("persist1");
			expect(doc?.id).toBe("persist1");
			expect(doc?.metadata).toEqual({ saved: true });
		});

		it("should support collections", () => {
			const db = open(dbPath, { dimensions: 64 });

			const users = db.collection("users");
			const products = db.collection("products");

			users.set([{ id: "user1", vector: Array(64).fill(0.1) }]);
			products.set([{ id: "prod1", vector: Array(64).fill(0.2) }]);

			expect(users.count).toBe(1);
			expect(products.count).toBe(1);

			// IDs are independent
			users.set([{ id: "item1", vector: Array(64).fill(0.3) }]);
			products.set([{ id: "item1", vector: Array(64).fill(0.4) }]);

			expect(users.count).toBe(2);
			expect(products.count).toBe(2);
		});

		it("should list collections", () => {
			const db = open(dbPath, { dimensions: 64 });
			db.collection("alpha");
			db.collection("beta");
			db.collection("gamma");

			const names = db.collections();
			expect(names).toEqual(["alpha", "beta", "gamma"]);
		});

		it("should delete collections", () => {
			const db = open(dbPath, { dimensions: 64 });
			const col = db.collection("todelete");
			col.set([{ id: "doc1", vector: Array(64).fill(0.1) }]);

			db.deleteCollection("todelete");
			expect(db.collections()).not.toContain("todelete");
		});
	});

	describe("error handling", () => {
		it("should reject invalid m parameter", () => {
			expect(() => open(":memory:", { dimensions: 128, m: 2 })).toThrow();
			expect(() => open(":memory:", { dimensions: 128, m: 100 })).toThrow();
		});

		it("should reject empty collection name", () => {
			const db = open(":memory:", { dimensions: 128 });
			// Collections require persistent storage
			expect(() => db.collection("test")).toThrow();
		});
	});
});
