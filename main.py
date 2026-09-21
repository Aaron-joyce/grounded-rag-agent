import sys
from src.config import DEFAULT_DOCS_DIR
from src.indexer import load_and_index_document
from src.agent import run_rag_agent


def main():
    docs_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOCS_DIR

    print(f"Loading and indexing documents from: {docs_path}")
    load_and_index_document(docs_path)

    questions = [
        # "How do I install this project?",
        "Give 5 line description about providence"
    ]

    for q in questions:
        run_rag_agent(q)
        print("\n" + "#" * 80 + "\n")


if __name__ == "__main__":
    main()
