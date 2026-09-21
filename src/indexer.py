from pathlib import Path
from src.chunker import chunk_text
from src.retriever import collection, embedding_model, init_bm25_index
from pypdf import PdfReader

def extract_from_file(file_path: Path) -> str:
    if file_path.suffix in [".txt", ".md", ".py", ".js", ".ts"]:
        return file_path.read_text(encoding="utf-8")
    elif file_path.suffix == ".pdf":
        reader = PdfReader(file_path)
        # Page-by-page extraction (efficient for multi-page PDFs)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages_text)
    return ""

def load_and_index_document(docs_path: str | Path):
    docs_dir = Path(docs_path)

    if not docs_dir.exists():
        print(f"Creating Docs Directory: {docs_dir}")
        docs_dir.mkdir(parents=True, exist_ok=True)
        return

    indexed_count = 0

    for file_path in docs_dir.rglob("*"):
        try:
            if file_path.is_file():
                content = extract_from_file(file_path)

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

    # Refresh in-memory BM25 index with newly indexed documents
    init_bm25_index(force_rebuild=True)

    print(f"\nTotal Chunks indexed: {indexed_count}")
