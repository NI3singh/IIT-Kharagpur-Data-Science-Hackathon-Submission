# src/config.py
import os
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# 2. API Configuration (Nebius)
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL")

# 3. Model Identifiers
LLM_MODEL_REASONING = os.getenv("LLM_MODEL_REASONING")
LLM_MODEL_EXTRACTION = os.getenv("LLM_MODEL_EXTRACTION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# 4. Project Paths (Dynamic Resolution)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BOOKS_DIR = os.path.join(DATA_DIR, "books")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "results.csv")

# 5. Pathway Server Configuration
PATHWAY_HOST = os.getenv("PATHWAY_HOST", "localhost")
PATHWAY_PORT = int(os.getenv("PATHWAY_PORT", 8000))

# 6. Indexing & Retrieval Settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval parameters
TOP_K_RETRIEVAL = 5  # Number of evidence chunks to retrieve per fact

# 7. Generation Parameters
TEMPERATURE_EXTRACTION = 0.0
TEMPERATURE_REASONING = 0.1

# 8. OpenAI Client Helper
def get_openai_client():
    """
    Returns a pre-configured OpenAI client connected to Nebius API.
    This ensures every script uses the exact same connection logic.
    """
    from openai import OpenAI
    
    return OpenAI(
        base_url=NEBIUS_BASE_URL,
        api_key=NEBIUS_API_KEY
    )

# 9. Validation
def validate_environment():
    """Quick check to ensure all necessary variables are set."""
    if not NEBIUS_API_KEY:
        raise ValueError("ERROR: NEBIUS_API_KEY not found in environment variables.")
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"ERROR: Data directory not found at {DATA_DIR}")
    
    print("✓ Configuration loaded successfully.")
    print(f"  - Reasoning Model: {LLM_MODEL_REASONING}")
    print(f"  - Extraction Model: {LLM_MODEL_EXTRACTION}")
    print(f"  - Embedding Model: {EMBEDDING_MODEL}")
    print(f"  - API Base: {NEBIUS_BASE_URL}")

if __name__ == "__main__":
    validate_environment()