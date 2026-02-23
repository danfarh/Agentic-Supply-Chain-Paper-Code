import os

from dotenv import load_dotenv

load_dotenv()

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CONFIG_DIR))
EXCEL_PATH = os.path.join(BASE_DIR, "data", "KTC-2025-ICT-benchmark-data.xlsx")
SCORING_SHEET = "1) Scoring"
DETAILED_SHEET = "2) Detailed scoring & research"
NON_SCORED_SHEET = "3) Non-scored research"

# Log Path
LOG_DIR = os.path.join(BASE_DIR, "logs")

# PDF / RAG Paths
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

# OpenAI Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
