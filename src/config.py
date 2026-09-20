import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
DEFAULT_DOCS_DIR = BASE_DIR / "docs"

# Model configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-3.6-flash"

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
