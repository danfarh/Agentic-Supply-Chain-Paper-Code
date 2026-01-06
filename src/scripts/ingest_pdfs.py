import os
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PDF_DIR = os.path.join(PROJECT_ROOT, "data", "pdfs")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_db")

print(f"📂 Looking for PDFs in: {PDF_DIR}")
print(f"💾 Saving DB to: {CHROMA_DB_DIR}")


def guess_company_name(filename: str) -> str:
    """
    Heuristic to guess a company name from a PDF filename.

    You should adapt this to your real naming convention if needed.
    Examples:
      "Amazon_KTC_Report_2025.pdf"      -> "Amazon"
      "Samsung Electronics_KTC.pdf"     -> "Samsung Electronics"
      "01_Amazon.com Inc._Report.pdf"   -> "Amazon.com Inc."
    """
    base = os.path.splitext(os.path.basename(filename))[0]

    # Remove leading index like "01_" or "1-"
    for sep in ["_", "-", " "]:
        parts = base.split(sep, 1)
        if parts[0].isdigit() and len(parts) > 1:
            base = parts[1]
            break

    return base.strip()


def ingest_data():
    if not os.path.exists(PDF_DIR):
        print(f"❌ PDF directory not found: {PDF_DIR}")
        return

    pdf_files: List[str] = [
        f for f in os.listdir(PDF_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"❌ No PDF files found in {PDF_DIR}")
        return

    print(f"🔄 Found {len(pdf_files)} PDF files in {PDF_DIR}. Starting ingestion...")

    all_docs = []

    for fname in pdf_files:
        path = os.path.join(PDF_DIR, fname)
        company_name = guess_company_name(fname)

        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
        except Exception as e:
            print(f"❌ Error loading {fname}: {e}")
            continue

        for d in docs:
            d.metadata["company"] = company_name
            d.metadata["source"] = fname

        all_docs.extend(docs)
        print(f"✅ Loaded {len(docs)} pages from {fname} (company: {company_name})")

    if not all_docs:
        print("❌ No documents loaded; nothing to index.")
        return

    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    print(f"📊 Total chunks to index: {len(chunks)}")

    if not chunks:
        print("❌ No chunks produced; aborting.")
        return

    print("🧠 Computing embeddings and creating Chroma DB...")
    embeddings = OpenAIEmbeddings()

    # Create / overwrite Chroma DB
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    vector_store.persist()

    print(f"🎉 Ingestion complete. Chroma DB saved to: {CHROMA_DB_DIR}")


if __name__ == "__main__":
    ingest_data()
