"""Document indexing module."""

from pathlib import Path
from src.chunker import chunk_text
from src.retriever import collection, embedding_model


def load_and_index_document(docs_path: str | Path):
    """Parses and indexes documents from a target directory into ChromaDB.

    Args:
        docs_path: Directory containing documents to index.
    """
    docs_dir = Path(docs_path)

    if not docs_dir.exists():
        print(f"Creating Docs Directory: {docs_dir}")
        docs_dir.mkdir(parents=True, exist_ok=True)
        return

    indexed_count = 0

    for file_path in docs_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in [".txt", ".md", ".py", ".js", ".ts"]:
            try:
                content = file_path.read_text(encoding="utf-8")

                if len(content) < 50:
                    continue

                chunks = chunk_text(content)
                embeddings = embedding_model.encode(chunks).tolist()

                ids = [f"{file_path.name}_{i}" for i in range(len(chunks))]
                metadatas = [
                    {"source": str(file_path), "chunk_index": i}
                    for i in range(len(chunks))
                ]

                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=metadatas,
                )

                indexed_count += len(chunks)
                print(f"Indexed: {file_path.name} ({len(chunks)} chunks)")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"\nTotal Chunks indexed: {indexed_count}")
