import os
from dotenv import load_dotenv

# Load environment file (defaults to .env, can override with DOTENV_PATH)
# When DOTENV_PATH is explicitly set (e.g. .env.local for dev), use override=True
# so it WINS over any vars already loaded by Flask CLI's auto .env preload.
# Without this, dev runs against whatever DATABASE_URL is in .env (often prod) — which has caused
# accidental migrations against production. Do not change without understanding this.
_explicit_dotenv = os.getenv('DOTENV_PATH')
if _explicit_dotenv:
    load_dotenv(_explicit_dotenv, override=True)
else:
    load_dotenv('.env')

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-5.2"
    LLM_REASONING_HIGH = "high"
    LLM_REASONING_MEDIUM = "medium"
    LLM_REASONING_LOW = "low"
    LLM_MAX_COMPLETION_TOKENS = 16000
    VISION_MODEL = "gpt-4o"  # Image description tasks — no reasoning needed, far cheaper than gpt-5.2

    # Stripe Configuration
    USE_STRIPE_TEST_MODE = os.getenv('USE_STRIPE_TEST_MODE', 'True') == 'True'

    if USE_STRIPE_TEST_MODE:
        STRIPE_PUBLIC_KEY = os.getenv('STRIPE_PUBLIC_KEY_TEST', '')
        STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY_TEST')
        STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET_TEST', '')
    else:
        STRIPE_PUBLIC_KEY = os.getenv('STRIPE_PUBLIC_KEY_LIVE', '')
        STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY_LIVE')
        STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET_LIVE', '')

    # Billing tiers with Stripe price IDs
    BILLING_TIERS = {
        'tier1': {
            'name': 'Small Course',
            'max_students': 50,
            'price_monthly': 9.99,
            'stripe_price_id': os.getenv('STRIPE_PRICE_TIER1')
        },
        'tier2': {
            'name': 'Medium Course',
            'max_students': 100,
            'price_monthly': 19.99,
            'stripe_price_id': os.getenv('STRIPE_PRICE_TIER2')
        },
        'tier3': {
            'name': 'Large Course',
            'max_students': 250,
            'price_monthly': 29.99,
            'stripe_price_id': os.getenv('STRIPE_PRICE_TIER3')
        },
    }

    # Email validation settings
    REQUIRE_EDU_EMAIL = True
    
    ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY")  # Required — no default
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
    
    _raw_sheet = os.getenv("qa_log_googlesheet", "")
    import re as _re
    _sheet_match = _re.search(r'/spreadsheets/d/([a-zA-Z0-9_-]+)', _raw_sheet)
    QA_LOG_SHEET_ID = _sheet_match.group(1) if _sheet_match else _raw_sheet
    QA_LOG_TAB_NAME = "qa_logs_v2"
    INDEX_LOG_TAB_NAME = "index_logs_v2"
    
    HYBRID_RETRIEVAL_ENABLED = True
    HYBRID_CONFIDENCE_THRESHOLD = 6
    HYBRID_MAX_DOC_TOKENS = 80000
    HYBRID_SCORE_SPREAD_THRESHOLD = 2

    CONTEXTUALIZER_ENABLED = os.getenv('CONTEXTUALIZER_ENABLED', 'true').lower() == 'true'
    CONTEXTUALIZER_MODEL = os.getenv('CONTEXTUALIZER_MODEL', 'gpt-4o-mini')
    CONTEXTUALIZER_MAX_HISTORY = 6

    # Pre-retrieval adversarial / off-topic filter. When True, queries the contextualizer
    # classifies as `off_topic` short-circuit before retrieval/generation and get a brief
    # canned redirect. Easy kill switch if classification accuracy drops.
    ADVERSARIAL_FILTER_ENABLED = os.getenv('ADVERSARIAL_FILTER_ENABLED', 'true').lower() == 'true'

    # Days to retain student-uploaded image_data on ChatMessage rows before the
    # `flask cleanup-images` CLI command zeroes them out. Keeps storage bounded
    # and limits the privacy footprint of student work.
    IMAGE_RETENTION_DAYS = int(os.getenv('IMAGE_RETENTION_DAYS', '30'))

    # Auth0 Configuration (professor app)
    AUTH0_DOMAIN = os.getenv('AUTH0_DOMAIN')
    AUTH0_CLIENT_ID = os.getenv('AUTH0_CLIENT_ID')
    AUTH0_CLIENT_SECRET = os.getenv('AUTH0_CLIENT_SECRET')
    # Auth0 student app (separate to allow role-specific login pages)
    AUTH0_STUDENT_CLIENT_ID = os.getenv('AUTH0_STUDENT_CLIENT_ID')
    AUTH0_STUDENT_CLIENT_SECRET = os.getenv('AUTH0_STUDENT_CLIENT_SECRET')

    # Auth0 M2M app (for Management API calls like resending verification emails)
    AUTH0_M2M_CLIENT_ID = os.getenv('AUTH0_M2M_CLIENT_ID')
    AUTH0_M2M_CLIENT_SECRET = os.getenv('AUTH0_M2M_CLIENT_SECRET')
    # Canonical Auth0 domain for Management API (custom domains don't support client_credentials)
    AUTH0_CANONICAL_DOMAIN = os.getenv('AUTH0_CANONICAL_DOMAIN', '')
