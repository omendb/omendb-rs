import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { open, VectorDatabase } from "../index.js";
import { mkdtemp, rm } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";

describe("Hybrid Search", () => {
	let tempDir: string;
	let dbPath: string;
	let db: VectorDatabase;

	beforeEach(async () => {
		tempDir = await mkdtemp(join(tmpdir(), "omendb-hybrid-"));
		dbPath = join(tempDir, "testdb");
		db = open(dbPath, { dimensions: 4 });
	});

	afterEach(async () => {
		await new Promise((r) => setTimeout(r, 100));
		try {
			await rm(tempDir, { recursive: true, force: true });
		} catch {
			// Ignore cleanup errors
		}
	});

	describe("auto-enable text search", () => {
		it("should allow text search config at open time", async () => {
			const configuredPath = join(tempDir, "configured");
			const configured = open(configuredPath, {
				dimensions: 4,
				textSearch: { bufferMb: 20, tokenizer: "code" },
			});

			expect(configured.hasTextSearch).toBe(true);

			await configured.set([
				{
					id: "doc1",
					vector: [1.0, 0.0, 0.0, 0.0],
					text: "HTTPClient handles user_id",
				},
			]);
			configured.flush();

			const results = configured.searchText("client", 10);
			expect(results).toHaveLength(1);
			expect(results[0]?.id).toBe("doc1");
			configured.close();
		});

		it("should auto-enable text search when using text field", async () => {
			expect(db.hasTextSearch).toBe(false);

			await db.set([
				{
					id: "doc1",
					vector: [1.0, 0.0, 0.0, 0.0],
					text: "Machine learning is a subset of AI",
				},
			]);

			expect(db.hasTextSearch).toBe(true);
		});

		it("should not enable text search for items without text", async () => {
			await db.set([{ id: "doc1", vector: [1.0, 0.0, 0.0, 0.0] }]);
			expect(db.hasTextSearch).toBe(false);
		});
	});

	describe("set with text", () => {
		it("should insert documents with text", async () => {
			const count = await db.set([
				{
					id: "doc1",
					vector: [1.0, 0.0, 0.0, 0.0],
					text: "Machine learning is a subset of AI",
				},
				{
					id: "doc2",
					vector: [0.0, 1.0, 0.0, 0.0],
					text: "Deep learning uses neural networks",
					metadata: { category: "tech" },
				},
			]);

			db.flush();

			expect(count).toBe(2);
			expect(db.count()).toBe(2);
		});

		it("should store text in metadata for retrieval", async () => {
			await db.set([
				{
					id: "doc1",
					vector: [1.0, 0.0, 0.0, 0.0],
					text: "Stored text content",
				},
			]);

			const doc = db.get("doc1");
			expect(doc?.metadata?.text).toBe("Stored text content");
		});
	});

	describe("searchText", () => {
		beforeEach(async () => {
			await db.set([
				{
					id: "doc1",
					vector: [1.0, 0.0, 0.0, 0.0],
					text: "Python programming language",
				},
				{
					id: "doc2",
					vector: [0.0, 1.0, 0.0, 0.0],
					text: "JavaScript web development",
				},
				{
					id: "doc3",
					vector: [0.0, 0.0, 1.0, 0.0],
					text: "Python data science machine learning",
				},
			]);
			db.flush();
		});

		it("should find documents by text", () => {
			const results = db.searchText("Python", 10);

			expect(results.length).toBeGreaterThanOrEqual(1);

			const ids = results.map((r) => r.id);
			expect(ids.some((id) => id === "doc1" || id === "doc3")).toBe(true);
		});

		it("should return score and metadata", () => {
			const results = db.searchText("Python", 10);

			for (const r of results) {
				expect(r).toHaveProperty("id");
				expect(r).toHaveProperty("score");
				expect(r).toHaveProperty("metadata");
				expect(r.score).toBeGreaterThan(0);
			}
		});

		it("should return empty for non-matching query", () => {
			const results = db.searchText("xyznonexistent", 10);
			expect(results).toHaveLength(0);
		});
	});

	describe("searchHybrid", () => {
		beforeEach(async () => {
			await db.set([
				{
					id: "doc1",
					vector: [1.0, 0.0, 0.0, 0.0],
					text: "Machine learning algorithms",
					metadata: { type: "ml" },
				},
				{
					id: "doc2",
					vector: [0.9, 0.1, 0.0, 0.0],
					text: "Deep learning neural networks",
					metadata: { type: "dl" },
				},
				{
					id: "doc3",
					vector: [0.0, 1.0, 0.0, 0.0],
					text: "Web development frameworks",
					metadata: { type: "web" },
				},
			]);
			db.flush();
		});

		it("should combine vector and text search", async () => {
			const results = await db.searchHybrid([1.0, 0.0, 0.0, 0.0], "learning", 3);

			expect(results.length).toBeGreaterThanOrEqual(1);

			for (const r of results) {
				expect(r).toHaveProperty("id");
				expect(r).toHaveProperty("score");
				expect(r).toHaveProperty("metadata");
				expect(r.score).toBeGreaterThan(0);
			}
		});

		it("should return metadata in results", async () => {
			const results = await db.searchHybrid([1.0, 0.0, 0.0, 0.0], "machine", 2);

			expect(results.length).toBeGreaterThanOrEqual(1);
			expect(results[0].metadata).toBeDefined();
			expect(typeof results[0].metadata).toBe("object");
		});

		it("should accept Float32Array", async () => {
			const query = new Float32Array([1.0, 0.0, 0.0, 0.0]);
			const results = await db.searchHybrid(query, "learning", 3);
			expect(results.length).toBeGreaterThanOrEqual(1);
		});

		it("should support alpha option", async () => {
			// High alpha (favor vector)
			const vectorResults = await db.searchHybrid(
				[1.0, 0.0, 0.0, 0.0],
				"web",
				3,
				{ alpha: 0.9 },
			);

			// Low alpha (favor text)
			const textResults = await db.searchHybrid(
				[1.0, 0.0, 0.0, 0.0],
				"web",
				3,
				{ alpha: 0.1 },
			);

			expect(vectorResults.length).toBeGreaterThanOrEqual(1);
			expect(textResults.length).toBeGreaterThanOrEqual(1);
		});

		it("should support rrfK option", async () => {
			const defaultResults = await db.searchHybrid(
				[1.0, 0.0, 0.0, 0.0],
				"learning",
				2,
			);

			const customResults = await db.searchHybrid(
				[1.0, 0.0, 0.0, 0.0],
				"learning",
				2,
				{ rrfK: 10 },
			);

			expect(defaultResults.length).toBeGreaterThanOrEqual(1);
			expect(customResults.length).toBeGreaterThanOrEqual(1);
		});

		it("should support filter option", async () => {
			const results = await db.searchHybrid([1.0, 0.0, 0.0, 0.0], "learning", 10, {
				filter: { type: "ml" },
			});

			expect(results.length).toBeGreaterThanOrEqual(1);
			for (const r of results) {
				expect(r.metadata?.type).toBe("ml");
			}
		});

		it("should support subscores option", async () => {
			const results = await db.searchHybrid([1.0, 0.0, 0.0, 0.0], "learning", 2, {
				subscores: true,
			});

			expect(results.length).toBeGreaterThanOrEqual(1);
			// When subscores enabled, should have keywordScore and semanticScore
			for (const r of results) {
				expect(r).toHaveProperty("keywordScore");
				expect(r).toHaveProperty("semanticScore");
			}
		});

		it("should work with all options", async () => {
			const results = await db.searchHybrid(
				[1.0, 0.0, 0.0, 0.0],
				"learning",
				2,
				{
					filter: { type: { $in: ["ml", "dl"] } },
					alpha: 0.7,
					rrfK: 60,
				},
			);

			expect(results.length).toBeGreaterThanOrEqual(1);
			for (const r of results) {
				expect(["ml", "dl"]).toContain(r.metadata?.type);
			}
		});
	});

	describe("flush", () => {
		it("should commit text index changes", async () => {
			await db.set([
				{ id: "doc1", vector: [1.0, 0.0, 0.0, 0.0], text: "test document" },
			]);

			// Before flush, text may not be searchable
			db.flush();

			// After flush, text should be searchable
			const results = db.searchText("test", 10);
			expect(results.length).toBeGreaterThanOrEqual(1);
		});
	});
});
