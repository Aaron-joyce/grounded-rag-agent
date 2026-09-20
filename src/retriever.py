"""Retriever module for vector storage and semantic search."""

import chromadb
from sentence_transformers import SentenceTransformer
from src.config import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME

# Initialize ChromaDB client & collection
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine", "description": "Document store for RAG agent"}
)

# Initialize local embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def search_documents(query: str, n_results: int = 5) -> list[dict]:
    """Search vector collection using dense embeddings.

    Args:
        query: User search string.
        n_results: Max number of results to return.

    Returns:
        List of result dictionaries containing content, source, and relevance_score.
    """
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    formatted = []
    if results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            formatted.append(
                {
                    "content": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "relevance_score": 1 - results["distances"][0][i],
                }
            )

    return formatted


def get_indexed_sources() -> set[str]:
    """Retrieve unique source names indexed in the collection."""
    all_data = collection.get(include=["metadatas"])
    if not all_data["metadatas"]:
        return set()
    return {metadata["source"] for metadata in all_data["metadatas"]}
