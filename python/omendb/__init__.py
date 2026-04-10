"""OmenDB - Fast embedded vector database with HNSW + ACORN-1 filtered search.

Example (standalone):
    >>> import omendb
    >>> db = omendb.create("./my_vectors", {"dense": {"dim": 128}})
    >>> db.set([{"id": "doc1", "vector": [0.1] * 128, "metadata": {"title": "Hello"}}])
    >>> results = db.search(query=[0.1] * 128, k=5)

Example (LangChain):
    >>> from omendb.langchain import OmenDBVectorStore
    >>> from langchain_openai import OpenAIEmbeddings
    >>> store = OmenDBVectorStore.from_texts(
    ...     texts=["Hello world"],
    ...     embedding=OpenAIEmbeddings(),
    ...     path="./my_vectors",
    ... )
    >>> docs = store.similarity_search("greeting", k=1)
"""

# Re-export everything from native Rust module
from omendb.omendb import VectorDatabase, create, open

__all__ = ["open", "create", "VectorDatabase"]
__version__ = "0.0.34"
