#!/usr/bin/env python3
"""
Retrieval-Augmented Generation (RAG) example with OmenDB

Demonstrates:
- Document chunking and embedding
- Vector storage with metadata
- Semantic search for context retrieval
- RAG workflow pattern

Note: This example uses mock embeddings for simplicity.
In production, use a real embedding model like:
- OpenAI embeddings (openai.Embedding.create)
- Sentence Transformers (sentence-transformers library)
- Cohere embeddings (cohere.embed)
"""

import omendb
import shutil
from pathlib import Path

# Sample documents (in practice, these would come from your knowledge base)
DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Introduction to Vector Databases",
        "text": "Vector databases are specialized databases designed to store and search high-dimensional vectors. They enable semantic search by finding vectors that are similar to a query vector.",
        "source": "technical_guide.pdf",
        "page": 1,
    },
    {
        "id": "doc2",
        "title": "HNSW Algorithm",
        "text": "Hierarchical Navigable Small World (HNSW) is a graph-based algorithm for approximate nearest neighbor search. It builds a multi-layer graph structure that allows for efficient traversal during search.",
        "source": "technical_guide.pdf",
        "page": 5,
    },
    {
        "id": "doc3",
        "title": "Quantization Techniques",
        "text": "Quantization reduces memory usage by storing vectors in lower precision. RaBitQ is a state-of-the-art quantization method that achieves 8x compression with 100% recall using 4-bit quantization.",
        "source": "technical_guide.pdf",
        "page": 12,
    },
    {
        "id": "doc4",
        "title": "Python Installation",
        "text": "To install OmenDB, simply run: pip install omendb. This will install the pre-built binary wheel for your platform. Python 3.8 or later is required.",
        "source": "quickstart.md",
        "page": 1,
    },
    {
        "id": "doc5",
        "title": "Basic Usage",
        "text": "Start by opening a database with omendb.open('./data'). Then use db.set() to add vectors and db.search() to find similar vectors. All operations are persisted to disk automatically.",
        "source": "quickstart.md",
        "page": 2,
    },
]

def mock_embed(text: str) -> list[float]:
    """
    Mock embedding function (returns random-ish embeddings based on text hash).

    In production, replace with real embedding model:
    - OpenAI: openai.Embedding.create(input=text, model="text-embedding-ada-002")
    - Sentence Transformers: model.encode(text)
    - Cohere: co.embed(texts=[text])
    """
    # Simple hash-based mock (deterministic for same text)
    import hashlib
    hash_bytes = hashlib.sha256(text.encode()).digest()

    # Convert to 128-dimensional embedding (normalized)
    embedding = []
    for i in range(0, 128):
        byte_idx = i % len(hash_bytes)
        val = (hash_bytes[byte_idx] / 255.0) * 2 - 1  # Normalize to [-1, 1]
        embedding.append(val)

    # L2 normalize
    magnitude = sum(x*x for x in embedding) ** 0.5
    return [x / magnitude for x in embedding]

def chunk_document(doc: dict, chunk_size: int = 100) -> list[dict]:
    """
    Split document into chunks for better retrieval granularity.

    In production, use proper chunking strategies:
    - Sentence-based chunking (nltk, spacy)
    - Semantic chunking (split on topics)
    - Overlapping chunks (for context preservation)
    """
    text = doc["text"]
    words = text.split()

    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i + chunk_size])
        chunks.append({
            "id": f"{doc['id']}_chunk_{i // chunk_size}",
            "text": chunk_text,
            "embedding": mock_embed(chunk_text),
            "metadata": {
                "doc_id": doc["id"],
                "title": doc["title"],
                "source": doc["source"],
                "page": doc["page"],
                "chunk_index": i // chunk_size,
            }
        })

    return chunks

def retrieve_context(db: omendb.VectorDatabase, query: str, k: int = 3) -> list[dict]:
    """
    Retrieve relevant context for a query using semantic search.
    """
    # Embed the query
    query_embedding = mock_embed(query)

    # Search for similar chunks
    results = db.search(query=query_embedding, k=k)

    return results

def generate_answer(query: str, context: list[dict]) -> str:
    """
    Generate answer using retrieved context.

    In production, this would call an LLM:
    - OpenAI: openai.ChatCompletion.create(messages=[...])
    - Anthropic: anthropic.completion(prompt=...)
    - Open source: llama.cpp, vLLM, etc.
    """
    # Mock answer generation (in production, use LLM)
    context_text = "\n\n".join([
        f"[{c['metadata']['source']}, p.{c['metadata']['page']}] {c['metadata']['title']}"
        for c in context
    ])

    return f"""Based on the retrieved context:

{context_text}

Query: {query}

[In production, this would be an LLM-generated answer based on the context]
"""

def main():
    # Clean up any previous database
    db_path = "./rag_example_db"
    if Path(db_path).exists():
        shutil.rmtree(db_path)

    print("=== RAG Example with OmenDB ===\n")

    # 1. Create vector database
    print("1. Creating vector database...")
    db = omendb.open(db_path)
    print(f"   Database created at: {db_path}\n")

    # 2. Process and store documents
    print("2. Processing and storing documents...")
    all_chunks = []

    for doc in DOCUMENTS:
        # For simplicity, we're not chunking in this example
        # In production, use chunk_document(doc)
        chunks = [{
            "id": doc["id"],
            "embedding": mock_embed(doc["text"]),
            "metadata": {
                "title": doc["title"],
                "text": doc["text"],  # Store original text in metadata
                "source": doc["source"],
                "page": doc["page"],
            }
        }]
        all_chunks.extend(chunks)

    # Batch set for efficiency
    db.set(all_chunks)
    print(f"   Stored {len(all_chunks)} document chunks")
    print(f"   Database size: {len(db)} vectors\n")

    # 3. RAG workflow: Query → Retrieve → Generate
    queries = [
        "How do I install OmenDB?",
        "What is HNSW?",
        "How does quantization work?",
    ]

    for query in queries:
        print(f"Query: {query}")
        print("-" * 60)

        # Retrieve relevant context
        context = retrieve_context(db, query, k=2)

        print("\nRetrieved context:")
        for i, ctx in enumerate(context, 1):
            print(f"  {i}. [{ctx['metadata']['source']}, p.{ctx['metadata']['page']}] {ctx['metadata']['title']}")
            print(f"     Distance: {ctx['distance']:.3f}")
            print(f"     Text: {ctx['metadata']['text'][:100]}...")

        # Generate answer (mock)
        answer = generate_answer(query, context)
        print(f"\n{answer}\n")
        print("=" * 60)
        print()

    # 4. Advanced: Search with metadata filters
    print("4. Advanced: Searching with metadata filters...")
    query = "Tell me about the technical details"
    query_embedding = mock_embed(query)

    # Only search in technical_guide.pdf
    results = db.search(
        query=query_embedding,
        k=3,
        filter={"source": "technical_guide.pdf"}
    )

    print(f"   Query: {query}")
    print(f"   Filter: source=technical_guide.pdf")
    print(f"   Results:")
    for i, result in enumerate(results, 1):
        print(f"      {i}. {result['metadata']['title']} (distance={result['distance']:.3f})")

    print("\n=== RAG Example Complete ===")
    print("\nNext steps:")
    print("1. Replace mock_embed() with a real embedding model")
    print("2. Implement proper document chunking")
    print("3. Connect to an LLM for answer generation")
    print("4. Add re-ranking for better retrieval quality")
    print("5. Implement hybrid search (vector + keyword)")

if __name__ == "__main__":
    main()
