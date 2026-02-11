import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-5.2"
    LLM_REASONING_HIGH = "high"
    LLM_REASONING_MEDIUM = "medium"
    LLM_MAX_COMPLETION_TOKENS = 16000
    
    ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "maize-admin-2024")
    ADMIN_USERNAME = os.getenv("admin_id", "")
    ADMIN_PASSWORD = os.getenv("admin_pw", "")
    
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    TOP_K_RETRIEVAL = 20
    TOP_K_RERANK = 8
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 200
    
    SESSION_TTL_HOURS = 24
    MAX_CONVERSATION_TURNS = 10
    
    CHROMA_DB_PATH = "./chroma_db"
    
    METADATA_SCHEMA_VERSION = "1.0"
    
    QA_LOG_SHEET_ID = os.getenv("qa_log_googlesheet", "")
    QA_LOG_TAB_NAME = "qa_logs_v2"
    INDEX_LOG_TAB_NAME = "index_logs_v2"
    
    HYBRID_RETRIEVAL_ENABLED = True
    HYBRID_CONFIDENCE_THRESHOLD = 6
    HYBRID_MAX_DOC_TOKENS = 80000
    HYBRID_SCORE_SPREAD_THRESHOLD = 2
