"""RAG Agent module using Gemini tool calling."""

from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, LLM_MODEL_NAME
from src.retriever import search_hybrid, get_indexed_sources

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


def search_knowledge_base(query: str, num_results: int = 5) -> str:
    num_results = min(num_results, 10)
    results = search_hybrid(query, n_results=num_results)

    if not results:
        return "No relevant results found. Try another query."

    formatted = f"Found {len(results)} relevant document sections (Hybrid RRF Search):\n\n"
    for i, r in enumerate(results, 1):
        score_info = f"rrf_score: {r['rrf_score']:.4f}" if "rrf_score" in r else f"relevance: {r.get('relevance_score', 0):.2f}"
        formatted += f"---- Result {i} ({score_info}) ---\n"
        formatted += f"source: {r['source']}\n"
        formatted += f"content:\n{r['content']}\n\n"

    return formatted


def list_available_documents() -> str:
    sources = get_indexed_sources()

    if not sources:
        return "No documents have been indexed yet."

    result = f"Indexed Documents ({len(sources)} files):\n\n"
    for source in sorted(sources):
        result += f" - {source}\n"

    return result


def run_rag_agent(question: str) -> str:
    print("\n" + "=" * 30)
    print("RAG AGENT")
    print("=" * 30)
    print(f"\nQuestion: {question}\n")
    print("-" * 60)

    system_prompt = """You are a knowledgeable assistant with access to a document knowledge base.

When answering questions:
1. SEARCH FIRST - Always search the knowledge base before answering questions.
2. BE THOROUGH - If initial results are weak, try alternate search terms.
3. CITE SOURCES - Reference which documents your information comes from.
4. BE HONEST - State clearly if the requested information is absent.
5. USE ONLY THE RESOURCES PROVIDED: use only the documents provided to answer and queries. If a question does not have a relevant resource, respond with "I could not find information about that in your uploaded notebook documents."

Format your response as:
**Answer:** [Your comprehensive answer]
**Sources:**
- [List source documents used]"""

    response = client.models.generate_content(
        model=LLM_MODEL_NAME,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[search_knowledge_base, list_available_documents],
            temperature=0.2,
        ),
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(f"\n{response.text}\n")

    return response.text
