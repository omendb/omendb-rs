const { open } = require("../index.js");

const db = open(":memory:", { dimensions: 128 });

db.set([
	{ id: "doc1", vector: Array(128).fill(0.1), metadata: { title: "First doc" } },
	{ id: "doc2", vector: Array(128).fill(0.2), metadata: { title: "Second doc" } },
	{ id: "doc3", vector: Array(128).fill(0.15), metadata: { title: "Third doc" } },
]);

const results = db.search(Array(128).fill(0.1), 2);

for (const r of results) {
	console.log(`${r.id}: ${r.metadata?.title} (distance: ${r.distance.toFixed(4)})`);
}
