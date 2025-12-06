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
		db.enableTextSearch();
	});

	afterEach(async () => {
		await new Promise((r) => setTimeout(r, 100));
		try {
			await rm(tempDir, { recursive: true, force: true });
		} catch {
			// Ignore cleanup errors
		}
	});

	describe("enableTextSearch", () => {
		it("should enable text search on database", () => {
			const db2 = open(join(tempDir, "testdb2"), { dimensions: 4 });
			expect(db2.hasTextSearch).toBe(false);
			db2.enableTextSearch();
			expect(db2.hasTextSearch).toBe(true);
		});
	});

	describe("setWithText", () => {
		it("should insert documents with text", () => {
			const indices = db.setWithText([
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

			expect(indices).toHaveLength(2);
			expect(db.count).toBe(2);
		});

		it("should fail without enabling text search", () => {
			const db2 = open(join(tempDir, "testdb2"), { dimensions: 4 });
			expect(() => {
				db2.setWithText([
					{ id: "doc1", vector: [1.0, 0.0, 0.0, 0.0], text: "test" },
				]);
			}).toThrow(/not enabled/i);
		});
	});

	describe("textSearch", () => {
		beforeEach(() => {
			db.setWithText([
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
			const results = db.textSearch("Python", 10);

			expect(results.length).toBeGreaterThanOrEqual(1);

			const ids = results.map((r) => r.id);
			expect(ids.some((id) => id === "doc1" || id === "doc3")).toBe(true);
		});

		it("should return score and metadata", () => {
			const results = db.textSearch("Python", 10);

			for (const r of results) {
				expect(r).toHaveProperty("id");
				expect(r).toHaveProperty("score");
				expect(r).toHaveProperty("metadata");
				expect(r.score).toBeGreaterThan(0);
			}
		});

		it("should return empty for non-matching query", () => {
			const results = db.textSearch("xyznonexistent", 10);
			expect(results).toHaveLength(0);
		});
	});

	describe("hybridSearch", () => {
		beforeEach(() => {
			db.setWithText([
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

		it("should combine vector and text search", () => {
			const results = db.hybridSearch([1.0, 0.0, 0.0, 0.0], "learning", 3);

			expect(results.length).toBeGreaterThanOrEqual(1);

			for (const r of results) {
				expect(r).toHaveProperty("id");
				expect(r).toHaveProperty("score");
				expect(r).toHaveProperty("metadata");
				expect(r.score).toBeGreaterThan(0);
			}
		});

		it("should return metadata in results", () => {
			const results = db.hybridSearch([1.0, 0.0, 0.0, 0.0], "machine", 2);

			expect(results.length).toBeGreaterThanOrEqual(1);
			expect(results[0].metadata).toBeDefined();
			expect(typeof results[0].metadata).toBe("object");
		});

		it("should accept Float32Array", () => {
			const query = new Float32Array([1.0, 0.0, 0.0, 0.0]);
			const results = db.hybridSearch(query, "learning", 3);
			expect(results.length).toBeGreaterThanOrEqual(1);
		});

		it("should support alpha parameter", () => {
			// High alpha (favor vector)
			const vectorResults = db.hybridSearch(
				[1.0, 0.0, 0.0, 0.0],
				"web",
				3,
				undefined,
				0.9,
			);

			// Low alpha (favor text)
			const textResults = db.hybridSearch(
				[1.0, 0.0, 0.0, 0.0],
				"web",
				3,
				undefined,
				0.1,
			);

			expect(vectorResults.length).toBeGreaterThanOrEqual(1);
			expect(textResults.length).toBeGreaterThanOrEqual(1);
		});

		it("should support rrf_k parameter", () => {
			const defaultResults = db.hybridSearch(
				[1.0, 0.0, 0.0, 0.0],
				"learning",
				2,
			);

			const customResults = db.hybridSearch(
				[1.0, 0.0, 0.0, 0.0],
				"learning",
				2,
				undefined,
				undefined,
				10,
			);

			expect(defaultResults.length).toBeGreaterThanOrEqual(1);
			expect(customResults.length).toBeGreaterThanOrEqual(1);
		});

		it("should support metadata filter", () => {
			const results = db.hybridSearch([1.0, 0.0, 0.0, 0.0], "learning", 10, {
				type: "ml",
			});

			expect(results.length).toBeGreaterThanOrEqual(1);
			for (const r of results) {
				expect(r.metadata?.type).toBe("ml");
			}
		});

		it("should work with all parameters", () => {
			const results = db.hybridSearch(
				[1.0, 0.0, 0.0, 0.0],
				"learning",
				2,
				{ type: { $in: ["ml", "dl"] } },
				0.7,
				60,
			);

			expect(results.length).toBeGreaterThanOrEqual(1);
			for (const r of results) {
				expect(["ml", "dl"]).toContain(r.metadata?.type);
			}
		});
	});

	describe("flush", () => {
		it("should commit text index changes", () => {
			db.setWithText([
				{ id: "doc1", vector: [1.0, 0.0, 0.0, 0.0], text: "test document" },
			]);

			// Before flush, text may not be searchable
			db.flush();

			// After flush, text should be searchable
			const results = db.textSearch("test", 10);
			expect(results.length).toBeGreaterThanOrEqual(1);
		});
	});
});
