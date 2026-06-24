import json
import os

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

from src.config.config import CHROMA_DB_DIR

# Global variable to cache the database in memory
_VECTOR_DB = None


def get_vector_db():
    """
    Load the database only once. In subsequent calls,
    the cached version in RAM is returned directly, which significantly speeds up execution.
    """
    global _VECTOR_DB
    if _VECTOR_DB is None:
        try:
            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma

            embeddings = OpenAIEmbeddings()
            _VECTOR_DB = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Chroma DB: {e}")

    return _VECTOR_DB


@tool
def company_pdf_rag(company_name: str, question: str, k: int = 5) -> str:
    """
    DOCUMENT AGENT (persistent RAG):
    Search a pre-built Chroma vector store of all company PDFs and retrieve the
    most relevant chunks for a given company and question.

    IMPORTANT FOR EVALUATION:
    This tool returns a JSON string with both:
    - display: human-readable excerpts for the agent to use in its final answer
    - retrieved_contexts: structured chunks for RAGAS/evaluator faithfulness metrics
    """
    if not os.path.exists(CHROMA_DB_DIR):
        return json.dumps({
            "tool_name": "company_pdf_rag",
            "error": (
                f"Chroma vector database not found at '{CHROMA_DB_DIR}'. "
                "Run the PDF ingestion script (ingest_pdfs.py) first to build it."
            ),
            "retrieved_contexts": [],
            "display": (
                f"ERROR: Chroma vector database not found at '{CHROMA_DB_DIR}'. "
                "Run the PDF ingestion script (ingest_pdfs.py) first to build it."
            ),
        }, ensure_ascii=False)

    try:
        vector_db = get_vector_db()

        # Search with metadata filter first.
        try:
            results = vector_db.similarity_search(
                question,
                k=k,
                filter={"company": company_name}
            )
        except TypeError:
            results = vector_db.similarity_search(question, k=k)

        # Fallback if metadata filter is too strict.
        if not results:
            enriched_query = f"{company_name}: {question}"
            results = vector_db.similarity_search(enriched_query, k=k)

        if not results:
            return json.dumps({
                "tool_name": "company_pdf_rag",
                "company_name": company_name,
                "question": question,
                "retrieved_contexts": [],
                "display": f"No relevant information found for '{company_name}' regarding '{question}'.",
            }, ensure_ascii=False)

        retrieved_contexts = []
        display_lines = [
            f"Top {len(results)} excerpts for company '{company_name}' related to question: {question}",
            ""
        ]

        for i, doc in enumerate(results, start=1):
            meta = doc.metadata or {}
            src = meta.get("source", "Unknown source")
            page = meta.get("page", None)
            page_number = page + 1 if isinstance(page, int) else None
            content = doc.page_content.strip()

            retrieved_contexts.append({
                "content": content,
                "source": src,
                "page": page_number,
                "company": meta.get("company", company_name),
                "rank": i,
                "retriever": "Chroma.similarity_search",
            })

            text_snippet = content[:800] + "..." if len(content) > 800 else content
            page_info = f"(page {page_number})" if page_number is not None else ""
            display_lines.append(f"[{i}] {src} {page_info}\n{text_snippet}\n")

        payload = {
            "tool_name": "company_pdf_rag",
            "company_name": company_name,
            "question": question,
            "k": k,
            "retrieved_contexts": retrieved_contexts,
            "display": "\n".join(display_lines),
        }
        return json.dumps(payload, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "tool_name": "company_pdf_rag",
            "company_name": company_name,
            "question": question,
            "error": str(e),
            "retrieved_contexts": [],
            "display": f"ERROR accessing or querying the Chroma vector DB: {e}",
        }, ensure_ascii=False)