/**
 * OmenDB embeddingFn example — auto-embed documents and queries.
 */

const { create } = require("../index.js");

// Mock embedding function. Replace with your model.
function fakeEmbedder(texts) {
  return texts.map((text) => {
    const padded = text.padEnd(4).slice(0, 4);
    return new Float32Array(Array.from(padded).map((c) => (c.charCodeAt(0) % 10) / 10.0));
  });
}

async function main() {
  const db = create(":memory:", { dense: { dim: 4 } }, fakeEmbedder);

  // Add documents — vectors computed automatically
  await db.set([
    { id: "doc1", document: "Paris is the capital of France", metadata: { topic: "geography" } },
    { id: "doc2", document: "The mitochondria is the powerhouse of the cell", metadata: { topic: "biology" } },
    { id: "doc3", document: "JavaScript was created by Brendan Eich", metadata: { topic: "programming" } },
  ]);

  console.log(`Added ${db.count()} documents`);

  // Search with text — query auto-embedded
  const results = await db.search("capital city", 2);
  for (const r of results) {
    console.log(`  ${r.id}: distance=${r.distance.toFixed(4)}`);
  }

  db.close();
}

main();
