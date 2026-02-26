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

    This expects that you have already run `ingest_pdfs.py` to populate
    the Chroma DB at CHROMA_DB_DIR.

    Parameters:
    - company_name: canonical or approximate company name (e.g., "Amazon.com Inc.", "Samsung")
    - question: natural-language question (e.g., "high-risk sourcing countries", "opportunities for improvement")
    - k: number of top chunks to return (default 5)
    """
    if not os.path.exists(CHROMA_DB_DIR):
        return (
            f"ERROR: Chroma vector database not found at '{CHROMA_DB_DIR}'. "
            "Run the PDF ingestion script (ingest_pdfs.py) first to build it."
        )

    try:
        # Get the database
        vector_db = get_vector_db()
        
        # 1. Search with metadata filter
        try:
            results = vector_db.similarity_search(
                question,
                k=k,
                filter={"company": company_name}
            )
        except TypeError:
            results = vector_db.similarity_search(question, k=k)

        # 2. Fallback to a combined string search if no results are found via filtering
        if not results:
            enriched_query = f"{company_name}: {question}"
            results = vector_db.similarity_search(enriched_query, k=k)

        if not results:
            return f"No relevant information found for '{company_name}' regarding '{question}'."

        lines = [
            f"Top {len(results)} excerpts for company '{company_name}' "
            f"related to question: {question}",
            ""
        ]
        
        for i, doc in enumerate(results, start=1):
            meta = doc.metadata or {}
            src = meta.get("source", "Unknown source")
            page = meta.get("page", None)
            page_info = f"(page {page + 1})" if isinstance(page, int) else ""
            
            text_snippet = doc.page_content.strip()
            if len(text_snippet) > 800:
                text_snippet = text_snippet[:800] + "..."
                
            lines.append(f"[{i}] {src} {page_info}\n{text_snippet}\n")

        return "\n".join(lines)
        
    except Exception as e:
        return f"ERROR accessing or querying the Chroma vector DB: {e}"