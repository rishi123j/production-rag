import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data/documents"
LOGS_DIR = "logs"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_STRATEGY = "recursive"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSIONS = 768

TOP_K = 3
SIMILARITY_THRESHOLD = 0.5

LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0
MAX_TOKENS = 1000

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
