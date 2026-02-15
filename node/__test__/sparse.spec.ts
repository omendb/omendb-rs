import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { open, VectorDatabase } from "../index.js";
import { mkdtemp, rm } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";

describe("Sparse vectors", () => {
	it("enableSparse / hasSparse", () => {
		const db = open(":memory:", { dimensions: 3 });
		expect(db.hasSparse).toBe(false);
		db.enableSparse();
		expect(db.hasSparse).toBe(true);
	});

	it("setSparse with indices/values format", () => {
		const db = open(":memory:");
		db.setSparse("doc1", { indices: [10, 42, 100], values: [0.5, 1.2, 0.8] }, { title: "Hello" });
		expect(db.hasSparse).toBe(true);
	});

	it("setSparse with dict format", () => {
		const db = open(":memory:");
		db.setSparse("doc1", { "10": 0.5, "42": 1.2, "100": 0.8 }, { title: "Hello" });
		expect(db.hasSparse).toBe(true);
	});

	it("sparseSearch ordering", () => {
		const db = open(":memory:");

		db.setSparse("doc1", { "10": 1.0, "20": 0.5 });
		db.setSparse("doc2", { "10": 0.5, "30": 1.0 });
		db.setSparse("doc3", { "10": 0.1, "20": 0.1 });

		const results = db.sparseSearch({ "10": 1.0, "20": 1.0 }, 3);

		expect(results.length).toBe(3);
		expect(results[0].id).toBe("doc1");
		expect(results[0].score).toBeGreaterThan(results[1].score);
	});

	it("setHybridSparse + hybridSparseSearch", () => {
		const db = open(":memory:", { dimensions: 3 });

		db.setHybridSparse("doc1", [1, 0, 0], { "10": 1.0, "20": 0.5 });
		db.setHybridSparse("doc2", [0, 1, 0], { "10": 0.5, "30": 1.0 });
		db.setHybridSparse("doc3", [0, 0, 1], { "10": 0.1 });

		const results = db.hybridSparseSearch([1, 0, 0], { "10": 1.0, "20": 1.0 }, 3, {
			alpha: 0.5,
		});

		expect(results.length).toBe(3);
		expect(results[0].id).toBe("doc1");
	});

	it("filtered sparse search", () => {
		const db = open(":memory:");

		db.setSparse("doc1", { "10": 1.0 }, { category: "A" });
		db.setSparse("doc2", { "10": 0.8 }, { category: "B" });
		db.setSparse("doc3", { "10": 0.2 }, { category: "A" });

		const results = db.sparseSearch({ "10": 1.0 }, 10, { filter: { category: "A" } });

		expect(results.length).toBe(2);
		const ids = new Set(results.map((r) => r.id));
		expect(ids.has("doc1")).toBe(true);
		expect(ids.has("doc3")).toBe(true);
	});

	it("delete removes from sparse results", () => {
		const db = open(":memory:");

		db.setSparse("doc1", { "10": 1.0 });
		db.setSparse("doc2", { "10": 0.5 });

		let results = db.sparseSearch({ "10": 1.0 }, 10);
		expect(results.length).toBe(2);

		db.delete(["doc1"]);

		results = db.sparseSearch({ "10": 1.0 }, 10);
		expect(results.length).toBe(1);
		expect(results[0].id).toBe("doc2");
	});
});
