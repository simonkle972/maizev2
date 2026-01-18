import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o"
    
    ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "maize-admin-2024")
    ADMIN_USERNAME = os.getenv("admin_id", "")
    ADMIN_PASSWORD = os.getenv("admin_pw", "")
    
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    TOP_K_RETRIEVAL = 20
    TOP_K_RERANK = 8
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    
    SESSION_TTL_HOURS = 24
    MAX_CONVERSATION_TURNS = 10
    
    CHROMA_DB_PATH = "./chroma_db"
    
    METADATA_SCHEMA_VERSION = "1.0"
