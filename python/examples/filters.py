#!/usr/bin/env python3
"""
Metadata Filtering Example with OmenDB

Demonstrates:
- MongoDB-style filter operators
- Equality, comparison, and set operators
- Logical operators ($and, $or)
- Combining filters with vector search
- Filter performance characteristics
"""

import omendb
import shutil
from pathlib import Path

def create_sample_data() -> list[dict]:
    """
    Create sample research papers dataset with rich metadata.
    """
    papers = [
        {
            "id": "paper1",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {
                "title": "HNSW: Efficient Graph-Based ANN Search",
                "authors": ["Malkov", "Yashunin"],
                "year": 2018,
                "venue": "PAMI",
                "citations": 1500,
                "topics": ["graph-based", "ann", "hnsw"],
                "is_seminal": True,
            }
        },
        {
            "id": "paper2",
            "embedding": [0.2, 0.3, 0.4],
            "metadata": {
                "title": "RaBitQ: Quantization for Vector Search",
                "authors": ["Chen", "Zhang"],
                "year": 2024,
                "venue": "SIGMOD",
                "citations": 50,
                "topics": ["quantization", "compression", "rabitq"],
                "is_seminal": False,
            }
        },
        {
            "id": "paper3",
            "embedding": [0.3, 0.4, 0.5],
            "metadata": {
                "title": "DiskANN: Fast Billion-Scale Vector Search",
                "authors": ["Subramanya", "Kadekodi", "Simhadri"],
                "year": 2019,
                "venue": "NeurIPS",
                "citations": 800,
                "topics": ["graph-based", "disk", "billion-scale"],
                "is_seminal": True,
            }
        },
        {
            "id": "paper4",
            "embedding": [0.4, 0.5, 0.6],
            "metadata": {
                "title": "ACORN: Filtered Vector Search",
                "authors": ["Li", "Wang"],
                "year": 2023,
                "venue": "VLDB",
                "citations": 120,
                "topics": ["filtered-search", "hnsw", "metadata"],
                "is_seminal": False,
            }
        },
        {
            "id": "paper5",
            "embedding": [0.5, 0.6, 0.7],
            "metadata": {
                "title": "LSM-VEC: Vector Search on LSM-Trees",
                "authors": ["Zhang", "Liu"],
                "year": 2024,
                "venue": "VLDB",
                "citations": 30,
                "topics": ["lsm-tree", "disk", "streaming"],
                "is_seminal": False,
            }
        },
        {
            "id": "paper6",
            "embedding": [0.6, 0.7, 0.8],
            "metadata": {
                "title": "Faiss: Library for Vector Search",
                "authors": ["Johnson", "Douze", "Jégou"],
                "year": 2017,
                "venue": "arXiv",
                "citations": 2000,
                "topics": ["library", "gpu", "quantization"],
                "is_seminal": True,
            }
        },
    ]

    return papers

def main():
    # Clean up any previous database
    db_path = "./filters_example_db"
    if Path(db_path).exists():
        shutil.rmtree(db_path)

    print("=== Metadata Filtering Example ===\n")

    # 1. Create database and load sample data (3D vectors for simplicity)
    print("1. Creating database with sample research papers...")
    db = omendb.open(db_path, dimensions=3)

    papers = create_sample_data()
    db.set(papers)

    print(f"   Loaded {len(papers)} research papers")
    print(f"   Database size: {len(db)} vectors\n")

    # 2. Equality filters ($eq)
    print("2. Equality Filters ($eq)")
    print("-" * 60)

    # Implicit equality (shorthand)
    results = db.search(
        query=[0.3, 0.4, 0.5],
        k=10,
        filter={"venue": "VLDB"}
    )
    print(f"Papers from VLDB venue: {len(results)} results")
    for r in results:
        print(f"  - {r['metadata']['title']} ({r['metadata']['year']})")
    print()

    # Explicit equality
    results = db.search(
        query=[0.3, 0.4, 0.5],
        k=10,
        filter={"is_seminal": {"$eq": True}}
    )
    print(f"Seminal papers: {len(results)} results")
    for r in results:
        print(f"  - {r['metadata']['title']} (citations: {r['metadata']['citations']})")
    print()

    # 3. Comparison operators ($gt, $gte, $lt, $lte)
    print("3. Comparison Operators")
    print("-" * 60)

    # Greater than
    results = db.search(
        query=[0.3, 0.4, 0.5],
        k=10,
        filter={"citations": {"$gt": 500}}
    )
    print(f"Highly cited papers (>500 citations): {len(results)} results")
    for r in results:
        print(f"  - {r['metadata']['title']}: {r['metadata']['citations']} citations")
    print()

    # Range query
    results = db.search(
        query=[0.3, 0.4, 0.5],
        k=10,
        filter={"year": {"$gte": 2020, "$lte": 2024}}
    )
    print(f"Recent papers (2020-2024): {len(results)} results")
    for r in results:
        print(f"  - {r['metadata']['title']} ({r['metadata']['year']})")
    print()

    # 4. Set operator ($in)
    print("4. Set Operator ($in)")
    print("-" * 60)

    # In set
    results = db.search(
        query=[0.3, 0.4, 0.5],
        k=10,
        filter={"venue": {"$in": ["VLDB", "SIGMOD", "ICDE"]}}
    )
    print(f"Database conference papers: {len(results)} results")
    for r in results:
        print(f"  - {r['metadata']['title']} @ {r['metadata']['venue']}")
    print()

    # NOTE: Array filtering (checking if array contains element) is not yet supported
    # The $in operator checks if the field VALUE is in the provided list,
    # not if any element of the field array is in the list
    print("NOTE: Array element filtering ($in on arrays) not yet supported")
    print("      $in checks if field value is in list, not if array contains element")
    print()

    # 5. Logical operators ($and, $or)
    print("5. Logical Operators ($and, $or)")
    print("-" * 60)

    # AND: Recent papers from top venues
    results = db.search(
        query=[0.3, 0.4, 0.5],
        k=10,
        filter={
            "$and": [
                {"year": {"$gte": 2020}},
                {"venue": {"$in": ["VLDB", "SIGMOD", "NeurIPS"]}}
            ]
        }
    )
    print(f"Recent papers from top venues ($and): {len(results)} results")
    for r in results:
        print(f"  - {r['metadata']['title']} @ {r['metadata']['venue']} ({r['metadata']['year']})")
    print()

    # OR: Either highly cited OR recent
    results = db.search(
        query=[0.3, 0.4, 0.5],
        k=10,
        filter={
            "$or": [
                {"citations": {"$gt": 1000}},
                {"year": {"$gte": 2023}}
            ]
        }
    )
    print(f"Highly cited OR recent papers ($or): {len(results)} results")
    for r in results:
        print(f"  - {r['metadata']['title']}: {r['metadata']['citations']} citations ({r['metadata']['year']})")
    print()

    # 6. Combining multiple filters with $and
    print("6. Combining Multiple Filters")
    print("-" * 60)

    # Find: Recent AND highly cited papers
    results = db.search(
        query=[0.3, 0.4, 0.5],
        k=10,
        filter={
            "$and": [
                {"year": {"$gte": 2020}},
                {"citations": {"$gt": 100}}
            ]
        }
    )
    print(f"Recent AND highly cited papers: {len(results)} results")
    for r in results:
        print(f"  - {r['metadata']['title']}")
        print(f"    Citations: {r['metadata']['citations']}, Year: {r['metadata']['year']}")
    print()

    # 7. Negation operator ($ne)
    print("7. Negation Operator ($ne)")
    print("-" * 60)

    # Not equal
    results = db.search(
        query=[0.3, 0.4, 0.5],
        k=10,
        filter={"venue": {"$ne": "arXiv"}}
    )
    print(f"Papers NOT from arXiv: {len(results)} results")
    for r in results:
        print(f"  - {r['metadata']['title']} @ {r['metadata']['venue']}")
    print()

    # NOTE: $nin operator not yet implemented
    print("NOTE: $nin (not in) operator not yet implemented")
    print()

    # 8. Performance note: Filters vs brute force
    print("8. Performance Characteristics")
    print("-" * 60)
    print("Filters are applied AFTER vector search (post-filtering).")
    print("This means:")
    print("  ✓ Efficient for selective filters (returns few results)")
    print("  ✗ Less efficient for broad filters (filters most results)")
    print()
    print("For highly selective queries, consider:")
    print("  - Using larger k value to ensure enough results after filtering")
    print("  - ACORN-1 algorithm (metadata-aware graph construction) - future work")
    print()

    # Example: Selective filter may need larger k
    print("Example: Searching for seminal papers")

    # With k=2, might not get enough results
    results = db.search(
        query=[0.3, 0.4, 0.5],
        k=2,
        filter={"is_seminal": True}
    )
    print(f"  k=2, filter seminal: {len(results)} results (may miss some)")

    # With k=10, more likely to get all seminal papers
    results = db.search(
        query=[0.3, 0.4, 0.5],
        k=10,
        filter={"is_seminal": True}
    )
    print(f"  k=10, filter seminal: {len(results)} results (better coverage)")

    print("\n=== Filtering Example Complete ===")
    print("\nSupported operators:")
    print("  ✓ Equality: $eq (or implicit)")
    print("  ✓ Comparison: $gt, $gte, $lt, $lte")
    print("  ✓ Set: $in")
    print("  ✓ Logical: $and, $or")
    print("  ✓ Negation: $ne")
    print("\nNot yet implemented:")
    print("  ✗ $nin (not in set)")
    print("  ✗ Array element filtering (checking if array contains value)")
    print("\nNext steps:")
    print("  1. Try combining filters with your own data")
    print("  2. Experiment with different k values for selective filters")
    print("  3. Monitor performance for very selective queries")

if __name__ == "__main__":
    main()
