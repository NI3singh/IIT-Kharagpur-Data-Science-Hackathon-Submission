#  pathway_app/app.py

# import pathway as pw
# from pathway.xpacks.llm import embedders, splitters, vector_store
# from src.config import *

# # --- 1. DATA INGESTION ---
# # Connect to the 'data/books' folder to read .txt files
# # mode="static" means we read the files once at startup (books don't change during the contest)
# # with_metadata=True gives us the filename, which is crucial to know which book the chunk came from
# documents = pw.io.fs.read(
#     BOOKS_DIR,
#     format="binary",
#     mode="static",
#     with_metadata=True
# )

# # --- 2. TEXT PARSING ---
# # Convert raw bytes to string
# @pw.udf
# def parse_text(data: bytes) -> str:
#     """Decode bytes to UTF-8 string."""
#     return data.decode("utf-8")

# # Apply parsing
# parsed_docs = documents.select(
#     text=parse_text(pw.this.data),
#     path=pw.this.metadata["path"]
# )

# # --- 3. TEXT CHUNKING ---
# # We split long books into smaller chunks (e.g., 800 characters) so the embedding model can capture specific details.
# # We use CharCountSplitter to strictly follow the CHUNK_SIZE defined in config.
# # min_characters=100 ensures we don't create tiny, useless fragments.
# splitter = splitters.CharCountSplitter(
#     min_characters=100,
#     chunk_size=CHUNK_SIZE,
#     overlap=CHUNK_OVERLAP
# )

# # Apply chunking
# chunks = splitter.split_text(parsed_docs)

# # --- 4. EMBEDDING (NEBIUS API INTEGRATION) ---
# # We use OpenAIEmbedder because Nebius is OpenAI-compatible.
# # This configures Pathway to call the Nebius API for embeddings.
# embedder = embedders.OpenAIEmbedder(
#     capacity=10,  # Number of concurrent API calls to Nebius
#     model=EMBEDDING_MODEL,
#     api_key=NEBIUS_API_KEY,
#     base_url=NEBIUS_BASE_URL,
#     dimensions=1024  # Qwen3-Embedding-8B outputs 1024 dimensions
# )

# # Generate embeddings for every chunk
# embedded_data = embedder.embed(chunks, text=pw.this.text)

# # --- 5. VECTOR STORE INDEXING ---
# # We build the Vector Store Server.
# # This creates an in-memory index that can be queried via REST API.
# vector_store_server = embedded_data + vector_store.VectorStoreServer(
#     embedder=embedder, 
#     # dimensions=1024 # Optional: usually inferred from the embedder
# )

# # --- 6. SERVE THE API ---
# # This starts the server. It listens on http://0.0.0.0:8000 (by default)
# # The main.py script will send HTTP requests to this address to get evidence.
# print(f"🚀 Pathway Indexer starting on {PATHWAY_HOST}:{PATHWAY_PORT}")
# print(f"📚 Indexing books from: {BOOKS_DIR}")

# # Run the computation graph
# pw.run()


import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.xpacks.llm import splitters
from pathway.xpacks.llm import embedders
import requests
import os
from src.config import *
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION
# ============================================================================

# Nebius API Configuration
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Chunking Configuration
CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 100  # characters overlap

# Server Configuration
HOST = "0.0.0.0"
PORT = 8000

# Data Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BOOKS_DIR = os.path.join(DATA_DIR, "books")


# ============================================================================
# CUSTOM NEBIUS EMBEDDER
# ============================================================================

class NebiusEmbedder:
    """Custom embedder using Nebius API"""
    
    def __init__(self, api_key: str, model: str = EMBEDDING_MODEL):
        self.api_key = api_key
        self.model = model
        self.base_url = NEBIUS_BASE_URL
        
    def __call__(self, text: str) -> list[float]:
        """
        Embed a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": [text]  # API expects a list
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract first embedding
            return result["data"][0]["embedding"]
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            raise


# ============================================================================
# CUSTOM CHARACTER SPLITTER
# ============================================================================

class CharacterSplitter:
    """
    Custom splitter that chunks text by character count with overlap.
    Returns list of tuples (text, metadata) as expected by Pathway.
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def __call__(self, text: str, metadata: dict = None) -> list[tuple[str, dict]]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Input text to split
            metadata: Optional metadata dict to attach to each chunk
            
        Returns:
            List of tuples (chunk_text, metadata)
        """
        if metadata is None:
            metadata = {}
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append((chunk, metadata.copy()))
            
            # Move start position (accounting for overlap)
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= text_length:
                break
        
        return chunks if chunks else [("", metadata)]


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def run_vector_server():
    """
    Main function to run the RAG vector store server
    """
    
    print("🚀 Starting Pathway RAG Vector Store Server...\n")
    
    # Step 1: Read text files from the books directory
    print(f"📚 Reading books from: {BOOKS_DIR}")
    text_files = pw.io.fs.read(
        path=BOOKS_DIR,
        format="binary",
        mode="static",  # Use "streaming" for live updates
        with_metadata=True
    )
    
    # Step 2: Parse binary data to text
    # CRITICAL FIX: Access JSON metadata with bracket notation []
    parsed_docs = text_files.select(
        data=pw.apply(lambda x: x.decode('utf-8'), pw.this.data),
        _metadata=pw.this._metadata
  # ✅ CORRECT: bracket notation for JSON
    )
    
    # Step 3 & 4: Initialize embedder and splitter
    print(f"🔍 Embedding model: {EMBEDDING_MODEL}")
    print(f"📦 Chunk size: {CHUNK_SIZE} chars with {CHUNK_OVERLAP} overlap")
    
    embedder = embedders.LiteLLMEmbedder(
    model=EMBEDDING_MODEL,
    api_base=NEBIUS_BASE_URL,
    api_key=NEBIUS_API_KEY,
    custom_llm_provider="openai",
    )

    text_splitter = CharacterSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Step 5: Create and run the vector store server
    # VectorStoreServer handles embedding, chunking, and indexing automatically
    vector_server = VectorStoreServer(
        parsed_docs,  # Pass the parsed documents table
        embedder=embedder,
        splitter=text_splitter,
    )
    
    print(f"🌐 Server starting on http://{HOST}:{PORT}")
    print(f"\n💡 API Endpoints:")
    print(f"   POST http://localhost:{PORT}/v1/retrieve")
    print(f"   Body: {{'query': 'your search query', 'k': 5}}")
    print(f"\n✨ Ready to serve queries!\n")
    
    # Run the server
    vector_server.run_server(
        host=HOST,
        port=PORT,
        with_cache=False,  # Set to True to enable caching
        threaded=False  # Set to True if running in background
    )


# ============================================================================
# VALIDATION & MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Validate API key
    if not NEBIUS_API_KEY:
        raise ValueError(
            "❌ NEBIUS_API_KEY not found!\n"
            "Set it with: export NEBIUS_API_KEY='your-key-here'"
        )
    
    # Validate books directory
    if not os.path.exists(BOOKS_DIR):
        raise ValueError(
            f"❌ Books directory not found: {BOOKS_DIR}\n"
            f"Create it with: mkdir -p {BOOKS_DIR}"
        )
    
    # Check for .txt files
    txt_files = [f for f in os.listdir(BOOKS_DIR) if f.endswith('.txt')]
    if not txt_files:
        raise ValueError(
            f"❌ No .txt files found in {BOOKS_DIR}\n"
            "Add your book files there!"
        )
    
    print(f"✅ Found {len(txt_files)} book(s): {', '.join(txt_files)}\n")
    
    # Run the server
    run_vector_server()