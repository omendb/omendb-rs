#!/usr/bin/env python3
"""
OmenDB Metadata Filtering

Demonstrates MongoDB-style filter operators:
- Equality: $eq, implicit
- Comparison: $gt, $gte, $lt, $lte
- Set membership: $in
- Negation: $ne
- Logical: $and, $or
"""

import omendb

# Sample dataset: research papers
PAPERS = [
    {
        "id": "hnsw",
        "vector": [0.1, 0.2, 0.3],
        "metadata": {
            "title": "HNSW",
            "year": 2018,
            "venue": "PAMI",
            "citations": 1500,
            "seminal": True,
        },
    },
    {
        "id": "rabitq",
        "vector": [0.2, 0.3, 0.4],
        "metadata": {
            "title": "RaBitQ",
            "year": 2024,
            "venue": "SIGMOD",
            "citations": 50,
            "seminal": False,
        },
    },
    {
        "id": "diskann",
        "vector": [0.3, 0.4, 0.5],
        "metadata": {
            "title": "DiskANN",
            "year": 2019,
            "venue": "NeurIPS",
            "citations": 800,
            "seminal": True,
        },
    },
    {
        "id": "acorn",
        "vector": [0.4, 0.5, 0.6],
        "metadata": {
            "title": "ACORN",
            "year": 2023,
            "venue": "VLDB",
            "citations": 120,
            "seminal": False,
        },
    },
    {
        "id": "lsmvec",
        "vector": [0.5, 0.6, 0.7],
        "metadata": {
            "title": "LSM-VEC",
            "year": 2024,
            "venue": "VLDB",
            "citations": 30,
            "seminal": False,
        },
    },
    {
        "id": "faiss",
        "vector": [0.6, 0.7, 0.8],
        "metadata": {
            "title": "Faiss",
            "year": 2017,
            "venue": "arXiv",
            "citations": 2000,
            "seminal": True,
        },
    },
]


def main():
    # 3D vectors for simple demonstration
    db = omendb.open(":memory:", dimensions=3)
    db.set(PAPERS)

    query = [0.3, 0.4, 0.5]

    # --- EQUALITY ---
    # Implicit (shorthand)
    results = db.search(query, k=10, filter={"venue": "VLDB"})
    print("venue = 'VLDB':", [r["id"] for r in results])

    # Explicit $eq
    results = db.search(query, k=10, filter={"seminal": {"$eq": True}})
    print("seminal = True:", [r["id"] for r in results])

    # --- COMPARISON ---
    results = db.search(query, k=10, filter={"citations": {"$gt": 500}})
    print("citations > 500:", [r["id"] for r in results])

    results = db.search(query, k=10, filter={"year": {"$gte": 2020}})
    print("year >= 2020:", [r["id"] for r in results])

    # --- SET MEMBERSHIP ---
    results = db.search(query, k=10, filter={"venue": {"$in": ["VLDB", "SIGMOD"]}})
    print("venue in [VLDB, SIGMOD]:", [r["id"] for r in results])

    # --- NEGATION ---
    results = db.search(query, k=10, filter={"venue": {"$ne": "arXiv"}})
    print("venue != 'arXiv':", [r["id"] for r in results])

    # --- LOGICAL AND ---
    results = db.search(
        query,
        k=10,
        filter={"$and": [{"year": {"$gte": 2020}}, {"venue": {"$in": ["VLDB", "SIGMOD"]}}]},
    )
    print("year >= 2020 AND venue in DB confs:", [r["id"] for r in results])

    # --- LOGICAL OR ---
    results = db.search(
        query, k=10, filter={"$or": [{"citations": {"$gt": 1000}}, {"year": {"$gte": 2024}}]}
    )
    print("citations > 1000 OR year >= 2024:", [r["id"] for r in results])

    # --- COMBINED RANGE ---
    results = db.search(query, k=10, filter={"year": {"$gte": 2018, "$lte": 2020}})
    print("2018 <= year <= 2020:", [r["id"] for r in results])

    print("\n--- Supported Operators ---")
    print("  $eq, $ne           - equality, negation")
    print("  $gt, $gte, $lt, $lte - comparison")
    print("  $in                - set membership")
    print("  $and, $or          - logical operators")


if __name__ == "__main__":
    main()
