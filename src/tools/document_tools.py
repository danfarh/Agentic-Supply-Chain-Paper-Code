import json
import math
import os
import re

from collections import Counter
from difflib import SequenceMatcher
from itertools import zip_longest

from langchain_core.documents import Document
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


# Workbook names that differ from the short names on the PDF scorecards
_COMPANY_ALIASES = {
    "amazon com": "amazon",
    "hp enterprise": "hpe",
    "amd": "advanced micro devices",
    "boe technology group": "boe technology",
    "cisco systems": "cisco",
    "dell technologies": "dell",
    "foxconn": "hon hai",
    "fujifilm holdings": "fujifilm",
    "hewlett packard enterprise co hpe": "hpe",
    "hewlett packard enterprise": "hpe",
    "hon hai precision industry co ltd foxconn": "hon hai",
    "logitech international": "logitech",
    "luxshare precision industry": "luxshare precision",
    "micron technology": "micron",
    "motorola solutions": "motorola",
    "samsung electronics": "samsung",
    "taiwan semiconductor manufacturing": "tsmc",
    "telefonaktiebolaget lm ericsson publ": "ericsson",
}


def _company_name(value):
    name = re.sub(r"^(?:2025\s+)?KTC ICT Scorecard_", "", value.strip(), flags=re.I)
    name = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
    name = re.sub(r"(?: (?:inc|corp|co|ltd|plc|nv|ag|oyj|s a))+$", "", name)
    return _COMPANY_ALIASES.get(name, name)


def _company_filter(vector_db, company_name):
    wanted = _company_name(company_name)
    if not wanted:
        return None
    metadata = vector_db.get(include=["metadatas"])["metadatas"]
    keys = {m["company"] for m in metadata if m and isinstance(m.get("company"), str)
            and _company_name(m["company"]) == wanted}
    if company_name in keys:
        return company_name
    return keys.pop() if len(keys) == 1 else None


# Leave semantic search in charge unless the question contains terms from the PDF.
_QUERY_FILLER = set("a an the in of from for on by and or is are what which does did how "
                    "its it to s according pdf scorecard ktc knowthechain ict 2025".split())


def _terms(text):
    return re.findall(r"[^\W\d_]+|\d+", text.casefold())


def _passage_key(doc):
    meta = doc.metadata
    return meta.get("source"), meta.get("page"), " ".join(doc.page_content.split())


def _company_passages(vector_db, company_key, question, k):
    if k < 1:
        raise ValueError("k must be positive")
    where = {"company": company_key}
    data = vector_db.get(where=where, include=["documents", "metadatas"])
    passages = {}
    for content, meta in zip(data["documents"], data["metadatas"], strict=True):
        if not content or not meta or meta.get("company") != company_key:
            continue
        filename = str(meta.get("source") or "").replace("\\", "/").split("/")[-1]
        stem, extension = os.path.splitext(filename)
        if extension.lower() != ".pdf" or _company_name(stem) != _company_name(company_key):
            continue
        doc = Document(page_content=content, metadata=meta)
        passages.setdefault(_passage_key(doc), doc)
    if not passages:
        return []

    # Retrieve within the company. Over-fetch to allow for repeated indexed chunks.
    semantic = vector_db.similarity_search(question, k=min(len(data["documents"]), 3 * k), filter=where)
    vector_keys = list(dict.fromkeys(_passage_key(d) for d in semantic if _passage_key(d) in passages))

    ignored = _QUERY_FILLER.union(_terms(_company_name(company_key)))
    query = [w for w in _terms(question) if w not in ignored]
    counts = Counter(query)
    words = {key: [w for w in _terms(doc.page_content) if w not in ignored]
             for key, doc in passages.items()}
    frequency = Counter(term for tokens in words.values() for term in set(tokens))
    lexical = []
    for key, tokens in words.items():
        shared = counts.keys() & set(tokens)
        if not shared:
            continue
        phrase = SequenceMatcher(None, query, tokens, autojunk=False).find_longest_match().size
        weight = sum(math.log1p(len(words) / frequency[w]) for w in sorted(shared))
        lexical.append((phrase, weight, key))
    lexical.sort(key=lambda item: (item[0], item[1]), reverse=True)
    text_keys = [key for _, _, key in lexical]

    explicit = lexical and (lexical[0][0] >= 2 or len(query) == 1)
    first, second = (text_keys, vector_keys) if explicit else (vector_keys, text_keys)
    selected = {}
    for pair in zip_longest(first, second):
        for key in pair:
            if key is not None:
                selected.setdefault(key, passages[key])
            if len(selected) == k:
                return list(selected.values())
    return list(selected.values())


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

        company_key = _company_filter(vector_db, company_name)
        results = _company_passages(vector_db, company_key, question, k) if company_key else []

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
                "retriever": "company-filtered semantic and lexical search",
            })

            page_info = f"(page {page_number})" if page_number is not None else ""
            display_lines.append(f"[{i}] {src} {page_info}\n{content}\n")

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