import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.config import CHROMA_DB_DIR


@tool
def extract_pdf_text(path: str, max_pages: int = 5) -> str:
    """
    DOCUMENT AGENT:
    Extract raw text from the first 'max_pages' pages of a local PDF file.

    Parameters:
    - path: local filesystem path to the PDF file.
    - max_pages: maximum number of pages to extract (default 5).
    """
    if not os.path.exists(path):
        return f"ERROR: PDF file not found at path: {path}"

    try:
        from pypdf import PdfReader
    except ImportError:
        return (
            "ERROR: The 'pypdf' package is not installed. "
            "Install it via 'pip install pypdf' to enable PDF text extraction."
        )

    try:
        reader = PdfReader(path)
        pages_text: List[str] = []
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            txt = page.extract_text() or ""
            pages_text.append(txt)
        if not pages_text:
            return f"No text extracted from the first {max_pages} pages of {path}."
        return f"Extracted text from the first {len(pages_text)} pages of '{path}':\n\n" + "\n\n".join(pages_text)
    except Exception as e:
        return f"ERROR while reading PDF: {e}"


@tool
def semantic_pdf_search(path: str, question: str, k: int = 5, max_pages: int = 50) -> str:
    """
    DOCUMENT AGENT (ad-hoc RAG):
    Perform a semantic search over a single local PDF file using embeddings and a FAISS
    in-memory vector store, and return the most relevant chunks of text for the question.

    Parameters:
    - path: local filesystem path to the PDF file.
    - question: natural-language question to search for in the PDF.
    - k: number of top chunks to return (default 5).
    - max_pages: maximum number of pages to load from the PDF (default 50).
    """
    if not os.path.exists(path):
        return f"ERROR: PDF file not found at path: {path}"

    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        # Optionally limit pages
        if max_pages is not None and max_pages > 0:
            docs = [d for d in docs if (d.metadata.get("page", 0) + 1) <= max_pages]

        if not docs:
            return f"No pages loaded from '{path}'."

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        if not chunks:
            return f"No text chunks available after splitting '{path}'."

        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.from_documents(chunks, embeddings)

        results = vectordb.similarity_search(question, k=k)
        if not results:
            return f"No relevant text found in PDF for question: {question}"

        lines = [
            f"Top {len(results)} relevant excerpts from '{path}' for question: {question}",
            ""
        ]
        for i, doc in enumerate(results, start=1):
            page = doc.metadata.get("page", None)
            page_info = f"(page {page + 1})" if page is not None else ""
            text_snippet = doc.page_content.strip()
            if len(text_snippet) > 800:
                text_snippet = text_snippet[:800] + "..."
            lines.append(f"[{i}] {page_info}\n{text_snippet}\n")

        return "\n".join(lines)
    except Exception as e:
        return f"ERROR during semantic PDF search: {e}"


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
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma
    except ImportError:
        return (
            "ERROR: Could not import Chroma vector store. "
            "Install it via 'pip install langchain-chroma chromadb' or "
            "'pip install langchain-community chromadb'."
        )

    try:
        embeddings = OpenAIEmbeddings()
        vector_db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
        )
        try:
            results = vector_db.similarity_search(
                question,
                k=k,
                filter={"company": company_name}
            )
        except TypeError:
            results = vector_db.similarity_search(question, k=k)

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
            lines.append(
                f"[{i}] {src} {page_info}\n{text_snippet}\n"
            )

        return "\n".join(lines)
    except Exception as e:
        return f"ERROR accessing or querying the Chroma vector DB: {e}"
