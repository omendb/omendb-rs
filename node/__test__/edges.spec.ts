import { describe, it, expect, beforeEach } from "vitest";
import { open, VectorDatabase } from "../index.js";

function makeDb(): VectorDatabase {
	const db = open(":memory:", { dimensions: 4 });
	return db;
}

async function populate(db: VectorDatabase): Promise<void> {
	await db.set([
		{ id: "a", vector: [1, 0, 0, 0] },
		{ id: "b", vector: [0, 1, 0, 0] },
		{ id: "c", vector: [0, 0, 1, 0] },
		{ id: "d", vector: [0, 0, 0, 1] },
	]);
}

describe("EdgeStore", () => {
	let db: VectorDatabase;

	beforeEach(async () => {
		db = makeDb();
		await populate(db);
	});

	it("addEdge and getEdges basic", () => {
		db.addEdge("a", "b", "link");
		expect(db.edgeCount).toBe(1);
		expect(db.schema()).toMatchObject({
			graph: { enabled: true, temporal: "none", provenance: false },
		});

		const out = db.getEdges("a", "outgoing");
		expect(out).toHaveLength(1);
		expect(out[0].fromId).toBe("a");
		expect(out[0].toId).toBe("b");
		expect(out[0].edgeType).toBe("link");
		expect(out[0].weight).toBeCloseTo(1.0, 5);

		const inc = db.getEdges("b", "incoming");
		expect(inc).toHaveLength(1);
		expect(inc[0].fromId).toBe("a");
	});

	it("addEdge with weight and metadata", () => {
		db.addEdge("a", "b", "related", 0.75, { score: 42 });
		const edges = db.getEdges("a", "outgoing");
		expect(edges[0].weight).toBeCloseTo(0.75, 5);
		expect(edges[0].metadata).toEqual({ score: 42 });
	});

	it("addEdge replaces same type", () => {
		db.addEdge("a", "b", "link", 0.5);
		db.addEdge("a", "b", "link", 0.9, { key: "val" });
		expect(db.edgeCount).toBe(1);
		const edges = db.getEdges("a", "outgoing");
		expect(edges[0].weight).toBeCloseTo(0.9, 5);
		expect(edges[0].metadata).toEqual({ key: "val" });
	});

	it("removeEdge", () => {
		db.addEdge("a", "b", "link");
		expect(db.removeEdge("a", "b", "link")).toBe(true);
		expect(db.edgeCount).toBe(0);
		expect(db.removeEdge("a", "b", "link")).toBe(false);
	});

	it("getEdges both directions", () => {
		db.addEdge("a", "b", "link");
		db.addEdge("c", "a", "ref");
		expect(db.getEdges("a", "both")).toHaveLength(2);
	});

	it("traverse depth 2", () => {
		db.addEdge("a", "b", "next");
		db.addEdge("b", "c", "next");
		const reachable = db.traverse("a", "outgoing", 2);
		expect(reachable).toContain("b");
		expect(reachable).toContain("c");
		expect(reachable).toHaveLength(2);
	});

	it("traverse depth 1 default", () => {
		db.addEdge("a", "b", "next");
		db.addEdge("b", "c", "next");
		const reachable = db.traverse("a");
		expect(reachable).toEqual(["b"]);
	});

	it("traverse with edge type filter", () => {
		db.addEdge("a", "b", "link");
		db.addEdge("a", "c", "ref");
		expect(db.traverse("a", "outgoing", 1, "link")).toEqual(["b"]);
		expect(db.traverse("a", "outgoing", 1, "ref")).toEqual(["c"]);
	});

	it("expand", () => {
		db.addEdge("a", "c", "rel");
		db.addEdge("b", "d", "rel");
		const expanded = db.expand(["a", "b"]);
		expect(expanded.sort()).toEqual(["a", "b", "c", "d"]);
	});

	it("delete cascades edges", async () => {
		db.addEdge("a", "b", "link");
		await db.delete("a");
		expect(db.edgeCount).toBe(0);
	});

	it("invalid direction throws", () => {
		db.addEdge("a", "b", "link");
		expect(() => db.getEdges("a", "sideways" as any)).toThrow();
	});

	it("getEdge", () => {
		db.addEdge("a", "b", "link", 0.5, { k: 1 });
		const edge = db.getEdge("a", "b", "link");
		expect(edge).not.toBeNull();
		expect(edge!.weight).toBeCloseTo(0.5, 5);
		expect(edge!.metadata).toEqual({ k: 1 });
		expect(db.getEdge("a", "b", "nope")).toBeNull();
		expect(db.getEdge("b", "a", "link")).toBeNull();
	});

	it("neighbors", () => {
		db.addEdge("a", "b", "link");
		db.addEdge("a", "c", "ref");
		db.addEdge("d", "a", "link");
		expect(db.neighbors("a", "outgoing")!.sort()).toEqual(["b", "c"]);
		expect(db.neighbors("a", "incoming")).toEqual(["d"]);
		expect(db.neighbors("a", "outgoing", "link")).toEqual(["b"]);
	});

	it("nodeDegree", () => {
		db.addEdge("a", "b", "link");
		db.addEdge("a", "c", "ref");
		db.addEdge("d", "a", "link");
		expect(db.nodeDegree("a", "outgoing")).toBe(2);
		expect(db.nodeDegree("a", "incoming")).toBe(1);
		expect(db.nodeDegree("a", "both")).toBe(3);
		expect(db.nodeDegree("a", "outgoing", "link")).toBe(1);
	});

	it("hasPath", () => {
		db.addEdge("a", "b", "next");
		db.addEdge("b", "c", "next");
		expect(db.hasPath("a", "c")).toBe(true);
		expect(db.hasPath("c", "a")).toBe(false);
		expect(db.hasPath("a", "c", "outgoing", 1)).toBe(false);
		expect(db.hasPath("a", "a")).toBe(true);
	});

	it("shortestPath", () => {
		db.addEdge("a", "b", "next");
		db.addEdge("b", "c", "next");
		db.addEdge("c", "d", "next");
		expect(db.shortestPath("a", "d")).toEqual(["a", "b", "c", "d"]);
		expect(db.shortestPath("d", "a")).toBeNull();
		expect(db.shortestPath("a", "a")).toEqual(["a"]);
	});

	it("traverseEdges", () => {
		db.addEdge("a", "b", "next");
		db.addEdge("b", "c", "next");
		const hits = db.traverseEdges("a", "outgoing", 2);
		expect(hits).toHaveLength(2);
		const hitB = hits.find((h) => h.id === "b")!;
		expect(hitB.depth).toBe(1);
		expect(hitB.edge.fromId).toBe("a");
		expect(hitB.edge.toId).toBe("b");
	});

	it("subgraph", () => {
		db.addEdge("a", "b", "link");
		db.addEdge("b", "c", "link");
		db.addEdge("d", "a", "link");
		const sg = db.subgraph("a", 2, "outgoing");
		expect(sg.nodeIds.sort()).toEqual(["a", "b", "c"]);
		expect(sg.edges).toHaveLength(2);
	});

	it("addEdges batch", () => {
		const added = db.addEdges([
			{ fromId: "a", toId: "b", edgeType: "link" },
			{ fromId: "b", toId: "c", edgeType: "link", weight: 0.5 },
		]);
		expect(added).toBe(2);
		expect(db.edgeCount).toBe(2);
		const edge = db.getEdge("b", "c", "link");
		expect(edge!.weight).toBeCloseTo(0.5, 5);
	});

	it("edgeTypes", () => {
		db.addEdge("a", "b", "link");
		db.addEdge("a", "c", "ref");
		expect(db.edgeTypes().sort()).toEqual(["link", "ref"]);
	});

	it("nodeIds", () => {
		db.addEdge("a", "b", "link");
		db.addEdge("c", "a", "ref");
		expect(db.nodeIds().sort()).toEqual(["a", "b", "c"]);
	});
});
