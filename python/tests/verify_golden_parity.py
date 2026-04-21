import math

import omendb


def golden_vector(id_num: int, dim: int) -> list[float]:
    data = [0.0] * dim
    for i in range(dim):
        data[i] = math.sin((float(id_num + i)) * 0.1337)
    return data


def test_verify_golden_hybrid_ranking():
    db = omendb.open(":memory:", dimensions=128, text_search=True)

    docs = [
        (0, "the quick brown fox jumps over the lazy dog [animal]", {"tag": "animal"}),
        (1, "machine learning algorithms for vector databases [tech]", {"tag": "tech"}),
        (2, "rust programming language is fast and safe [tech]", {"tag": "tech"}),
        (3, "cooking recipes for delicious apple pie [food]", {"tag": "food"}),
        (4, "deep learning models and neural networks [tech]", {"tag": "tech"}),
    ]

    items = []
    for id_num, text, metadata in docs:
        doc_id = f"doc_{id_num}"
        vec = golden_vector(id_num, 128)
        items.append({"id": doc_id, "vector": vec, "text": text, "metadata": metadata})
    db.set(items)

    db.flush()

    # Hybrid Search: Vector (similar to doc 1) + Text ("learning")
    query_vec = golden_vector(1, 128)
    results = db.search_hybrid(query_vec, "learning", k=5, alpha=0.5)

    print(f"Hybrid Results: {[r['id'] for r in results]}")
    assert results[0]["id"] == "doc_1", "Doc 1 should rank first"
    assert results[1]["id"] == "doc_4", "Doc 4 should rank second"

    # Pure Text Search Stability
    text_results = db.search_text("tech", k=10)
    print(f"Text Results: {[r['id'] for r in text_results]}")
    ids = [r["id"] for r in text_results]
    assert "doc_1" in ids
    assert "doc_2" in ids
    assert "doc_4" in ids

    print("Golden Parity: PASS")


if __name__ == "__main__":
    test_verify_golden_hybrid_ranking()
