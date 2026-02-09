/**
 * OmenDB Metadata Filtering
 *
 * Demonstrates MongoDB-style filter operators:
 * - Equality: $eq, implicit
 * - Comparison: $gt, $gte, $lt, $lte
 * - Set membership: $in
 * - Negation: $ne
 * - Logical: $and, $or
 */

import { open } from "../index.js";

// Sample dataset: research papers
const PAPERS = [
	{
		id: "hnsw",
		vector: new Float32Array([0.1, 0.2, 0.3]),
		metadata: { title: "HNSW", year: 2018, venue: "PAMI", citations: 1500, seminal: true },
	},
	{
		id: "scann",
		vector: new Float32Array([0.2, 0.3, 0.4]),
		metadata: { title: "ScaNN", year: 2020, venue: "ICML", citations: 500, seminal: false },
	},
	{
		id: "diskann",
		vector: new Float32Array([0.3, 0.4, 0.5]),
		metadata: { title: "DiskANN", year: 2019, venue: "NeurIPS", citations: 800, seminal: true },
	},
	{
		id: "acorn",
		vector: new Float32Array([0.4, 0.5, 0.6]),
		metadata: { title: "ACORN", year: 2023, venue: "VLDB", citations: 120, seminal: false },
	},
	{
		id: "lsmvec",
		vector: new Float32Array([0.5, 0.6, 0.7]),
		metadata: { title: "LSM-VEC", year: 2024, venue: "VLDB", citations: 30, seminal: false },
	},
	{
		id: "faiss",
		vector: new Float32Array([0.6, 0.7, 0.8]),
		metadata: { title: "Faiss", year: 2017, venue: "arXiv", citations: 2000, seminal: true },
	},
];

function main() {
	// 3D vectors for simple demonstration
	const db = open(":memory:", { dimensions: 3 });
	db.set(PAPERS);

	const query = [0.3, 0.4, 0.5];

	// --- EQUALITY ---
	// Implicit (shorthand)
	let results = db.search(query, 10, { filter: { venue: "VLDB" } });
	console.log("venue = 'VLDB':", results.map((r) => r.id));

	// Explicit $eq
	results = db.search(query, 10, { filter: { seminal: { $eq: true } } });
	console.log("seminal = true:", results.map((r) => r.id));

	// --- COMPARISON ---
	results = db.search(query, 10, { filter: { citations: { $gt: 500 } } });
	console.log("citations > 500:", results.map((r) => r.id));

	results = db.search(query, 10, { filter: { year: { $gte: 2020 } } });
	console.log("year >= 2020:", results.map((r) => r.id));

	// --- SET MEMBERSHIP ---
	results = db.search(query, 10, { filter: { venue: { $in: ["VLDB", "SIGMOD"] } } });
	console.log("venue in [VLDB, SIGMOD]:", results.map((r) => r.id));

	// --- NEGATION ---
	results = db.search(query, 10, { filter: { venue: { $ne: "arXiv" } } });
	console.log("venue != 'arXiv':", results.map((r) => r.id));

	// --- LOGICAL AND ---
	results = db.search(query, 10, {
		filter: { $and: [{ year: { $gte: 2020 } }, { venue: { $in: ["VLDB", "SIGMOD"] } }] },
	});
	console.log("year >= 2020 AND venue in DB confs:", results.map((r) => r.id));

	// --- LOGICAL OR ---
	results = db.search(query, 10, {
		filter: { $or: [{ citations: { $gt: 1000 } }, { year: { $gte: 2024 } }] },
	});
	console.log("citations > 1000 OR year >= 2024:", results.map((r) => r.id));

	// --- COMBINED RANGE ---
	results = db.search(query, 10, { filter: { year: { $gte: 2018, $lte: 2020 } } });
	console.log("2018 <= year <= 2020:", results.map((r) => r.id));

	console.log("\n--- Supported Operators ---");
	console.log("  $eq, $ne             - equality, negation");
	console.log("  $gt, $gte, $lt, $lte - comparison");
	console.log("  $in                  - set membership");
	console.log("  $and, $or            - logical operators");
}

main();
