import re
import chromadb
from rank_bm25 import BM25Okapi
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

# In-memory BM25 index state
_bm25_index: BM25Okapi | None = None
_bm25_documents: list[str] = []
_bm25_metadatas: list[dict] = []


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def init_bm25_index(force_rebuild: bool = False):
    global _bm25_index, _bm25_documents, _bm25_metadatas

    if _bm25_index is not None and not force_rebuild:
        return

    all_data = collection.get(include=["documents", "metadatas"])
    _bm25_documents = all_data.get("documents") or []
    _bm25_metadatas = all_data.get("metadatas") or []

    if _bm25_documents:
        tokenized_corpus = [tokenize_text(doc) for doc in _bm25_documents]
        _bm25_index = BM25Okapi(tokenized_corpus)
    else:
        _bm25_index = None


def search_documents(query: str, n_results: int = 5) -> list[dict]:
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


def search_bm25(query: str, n_results: int = 10) -> list[dict]:
    init_bm25_index()

    if _bm25_index is None or not _bm25_documents:
        return []

    tokenized_query = tokenize_text(query)
    scores = _bm25_index.get_scores(tokenized_query)

    # Pair indices with scores and filter positive matches
    indexed_scores = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )

    results = []
    for idx, score in indexed_scores[:n_results]:
        if score > 0:
            results.append(
                {
                    "content": _bm25_documents[idx],
                    "source": _bm25_metadatas[idx]["source"],
                    "bm25_score": float(score),
                }
            )

    return results


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    top_n: int = 5,
) -> list[dict]:
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    # Helper to generate unique key per chunk
    def get_key(doc: dict) -> str:
        return f"{doc['source']}::{doc['content']}"

    # Score Dense results
    for rank, item in enumerate(dense_results, start=1):
        key = get_key(item)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))
        if key not in doc_map:
            doc_map[key] = item.copy()
        doc_map[key]["dense_rank"] = rank

    # Score BM25 results
    for rank, item in enumerate(bm25_results, start=1):
        key = get_key(item)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))
        if key not in doc_map:
            doc_map[key] = item.copy()
        doc_map[key]["bm25_rank"] = rank

    # Sort candidates by combined RRF score descending
    sorted_keys = sorted(
        rrf_scores.keys(), key=lambda k_: rrf_scores[k_], reverse=True
    )

    fused_results = []
    for key in sorted_keys[:top_n]:
        res = doc_map[key]
        res["rrf_score"] = rrf_scores[key]
        fused_results.append(res)

    return fused_results


def search_hybrid(query: str, n_results: int = 5) -> list[dict]:
    dense_results = search_documents(query, n_results=10)
    bm25_results = search_bm25(query, n_results=10)

    # Fallback to pure dense if BM25 has no matches
    if not bm25_results:
        return dense_results[:n_results]

    return reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_n=n_results)


def get_indexed_sources() -> set[str]:
    all_data = collection.get(include=["metadatas"])
    if not all_data["metadatas"]:
        return set()
    return {metadata["source"] for metadata in all_data["metadatas"]}
