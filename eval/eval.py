import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import run_rag_agent
from src.config import DEFAULT_DOCS_DIR
from src.indexer import load_and_index_document
from src.retriever import search_hybrid
from google.genai.errors import APIError

def run_agent_with_retry(question: str, max_retries: int = 3, initial_delay: int = 15) -> str:
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return run_rag_agent(question)
        except APIError as e:
            if ("429" in str(e) or "503" in str(e)) and attempt < max_retries:
                print(f"\nAPI Rate limit hit (Attempt {attempt}/{max_retries}). Waiting {delay}s before retry...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise e


EXACT_REFUSAL_TEXT = "I could not find information about that in your uploaded notebook documents."
def run_evaluation():
    eval_dir = Path(__file__).parent
    queries_file = eval_dir / "test_queries.json"
    results_file = eval_dir / "results.md"
    if not queries_file.exists():
        print(f"Error: {queries_file} not found!")
        return
    print("=== Ensuring Documents are Indexed ===")
    load_and_index_document(DEFAULT_DOCS_DIR)
    with open(queries_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    results = []
    recalled_count = 0
    refused_count = 0
    total_answerable = 0
    total_tricks = 0
    total_latency_ms = 0.0
    print("\n=== Running Benchmark Suite ===")
    for test in test_cases:
        qid = test["id"]
        q = test["question"]
        is_trick = test["is_trick"]
        expected_src = test["expected_source"]
        print(f"\nEvaluating Q{qid}: '{q}'")
        start_time = time.time()
        # 1. Test Retrieval Recall @ 5
        retrieved_chunks = search_hybrid(q, n_results=5)
        retrieved_sources = [Path(c["source"]).name for c in retrieved_chunks]
        recall_success = False
        if not is_trick:
            total_answerable += 1
            recall_success = any(expected_src in src for src in retrieved_sources)
            if recall_success:
                recalled_count += 1
        # 2. Test Agent Response & Refusal Compliance
        # Pause briefly to respect Gemini Free Tier RPM pacing
        time.sleep(2)
        response_text = run_agent_with_retry(q)
        elapsed_ms = (time.time() - start_time - 2.0) * 1000
        total_latency_ms += max(elapsed_ms, 0)
        refusal_success = False
        if is_trick:
            total_tricks += 1
            refusal_success = EXACT_REFUSAL_TEXT.lower() in response_text.lower()
            if refusal_success:
                refused_count += 1
        status = "PASS" if (recall_success if not is_trick else refusal_success) else "FAIL"
        results.append({
            "id": qid,
            "question": q,
            "is_trick": is_trick,
            "expected": expected_src or "REFUSAL",
            "retrieved_sources": ", ".join(set(retrieved_sources)) or "None",
            "latency_ms": round(elapsed_ms, 2),
            "status": status,
        })
    # Calculate overall metrics
    recall_pct = (recalled_count / total_answerable * 100) if total_answerable > 0 else 0.0
    refusal_pct = (refused_count / total_tricks * 100) if total_tricks > 0 else 0.0
    avg_latency_ms = (total_latency_ms / len(test_cases)) if test_cases else 0.0
    # Format Markdown Results Summary
    markdown_report = f"""# RAG Engine Evaluation Results
## Benchmark Summary
| Metric | Score | Target |
| :--- | :--- | :--- |
| **Retrieval Recall @ 5** | **{recall_pct:.1f}%** ({recalled_count}/{total_answerable}) | ≥ 85.0% |
| **Refusal Accuracy** | **{refusal_pct:.1f}%** ({refused_count}/{total_tricks}) | 100.0% |
| **Avg Query Latency** | **{avg_latency_ms:.2f} ms** | N/A |
---
## Detailed Test Case Results
| ID | Type | Question | Expected Source / Action | Retrieved Sources | Latency (ms) | Status |
|---|---|---|---|---|---|---|
"""
    for r in results:
        qtype = "Trick" if r["is_trick"] else "Standard"
        markdown_report += f"| {r['id']} | {qtype} | {r['question']} | `{r['expected']}` | `{r['retrieved_sources']}` | {r['latency_ms']} ms | **{r['status']}** |\n"
    # Save to eval/results.md
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(markdown_report)
    print(f"\nResults saved to: {results_file}")
if __name__ == "__main__":
    run_evaluation()
