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

    # pdf2image needs to find `pdftoppm` from the poppler distribution. On the VPS,
    # systemd restricts PATH so we must point at it explicitly (`POPPLER_PATH=/usr/bin`
    # in the unit's Environment=). On macOS/Homebrew, leaving this unset makes
    # pdf2image scan PATH and find /usr/local/bin/pdftoppm (Intel) or
    # /opt/homebrew/bin/pdftoppm (Apple Silicon). Hardcoding /usr/bin broke local dev.
    POPPLER_PATH = os.getenv("POPPLER_PATH") or None

    # Phase A retrieval refactor (gap analysis 2026-05-22). When True, retrieve_context
    # uses the new pre-stage query rewriter + hybrid Stage 1 + intent classifier path.
    # When False, falls back to the legacy analyze_query() regex + fuzzy filename matcher.
    # Default False; flip to True after verifying via the eval harness.
    RETRIEVAL_V2_ENABLED = os.getenv("RETRIEVAL_V2_ENABLED", "false").lower() == "true"

    # Phase A retrieval refactor. RRF k constant for reciprocal rank fusion in hybrid
    # Stage 1. Standard default is 60; lower values bias toward top-ranked items in
    # each list. See https://blog.serghei.pl/posts/reciprocal-rank-fusion-explained/.
    RRF_K = int(os.getenv("RRF_K", "60"))

    # Phase A retrieval refactor. Top-K candidate documents Stage 1 returns. Stage 2
    # (chunk retrieval) is constrained to chunks from these documents.
    STAGE_1_TOP_K_DOCS = int(os.getenv("STAGE_1_TOP_K_DOCS", "5"))

    # Phase A Stage 5. Pre-fusion direct-match short-circuit in hybrid_doc_search.
    # When the query has a high-confidence singular filename match (e.g. names
    # a specific lecture or pset clearly), skip RRF and return that doc only.
    # See attached_assets/maize-retrieval-residual-failures-research-2026-05-22.md
    # for the design rationale (Q1+Q2 verdicts). Threshold + margin env-tunable.
    FILENAME_DIRECT_MATCH_THRESHOLD = float(os.getenv("FILENAME_DIRECT_MATCH_THRESHOLD", "0.7"))
    FILENAME_DIRECT_MATCH_MARGIN = float(os.getenv("FILENAME_DIRECT_MATCH_MARGIN", "0.15"))

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
    # Tab name is env-configurable so local dev can write to a separate tab
    # (e.g. "dev_logs") in the same shared sheet without polluting the prod
    # qa_logs_v2 tab. Production leaves it unset → defaults to qa_logs_v2.
    QA_LOG_TAB_NAME = os.getenv("qa_log_tab_name", "qa_logs_v2")
    INDEX_LOG_TAB_NAME = "index_logs_v2"
    
    HYBRID_RETRIEVAL_ENABLED = True
    HYBRID_CONFIDENCE_THRESHOLD = 6
    HYBRID_MAX_DOC_TOKENS = 80000
    HYBRID_SCORE_SPREAD_THRESHOLD = 2

    # --- Reranker vendor -------------------------------------------------------
    # "gpt-5.2" (default, incumbent) | "cohere". Defaults to the incumbent so the
    # Cohere code ships dormant and a rollback is one env var, not a deploy.
    #
    # Why swap at all: gpt-5.2 rerank measures 11-19s per fresh-retrieval turn and
    # dominates the 29s T1 median that drives adoption erosion. Cohere measured
    # 161-556ms on the same shape of call.
    #
    # Second reason, measured 2026-08-17: gpt-5.2's reranking is markedly
    # NONDETERMINISTIC. Two identical passes over 58 queries agreed at RBO 0.772
    # with the top result differing on ~24% of them, and disabling the query
    # rewriter changed that by 0.018 — so the reranker, not the contextualizer,
    # owns it. A cross-encoder is deterministic, which also stabilises
    # assess_retrieval_confidence, since that thresholds directly on these scores.
    RERANKER_VENDOR = os.getenv("RERANKER_VENDOR", "gpt-5.2")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
    COHERE_TIMEOUT_S = float(os.getenv("COHERE_TIMEOUT_S", "10"))

    # Cohere returns relevance in [0,1]; gpt-5.2 returns 0-10. Scores are scaled
    # x10 onto the incumbent scale so assess_retrieval_confidence and every
    # downstream consumer keep working unchanged.
    #
    # SCALING IS NOT CALIBRATION. A Cohere 0.6 does not mean what a gpt-5.2 6
    # means, and the two Cohere models differ from each other too — on one probe
    # rerank-v3.5 scored a correct answer 0.82 where rerank-english-v3.0 scored it
    # 0.9995. So HYBRID_CONFIDENCE_THRESHOLD must be re-derived from real score
    # distributions before Cohere is enabled in prod; this override exists to hold
    # that value once measured.
    HYBRID_CONFIDENCE_THRESHOLD_COHERE = float(
        os.getenv("HYBRID_CONFIDENCE_THRESHOLD_COHERE", str(HYBRID_CONFIDENCE_THRESHOLD)))

    # Ceiling on the context assembled for a session-cache follow-up turn. That
    # path concatenates up to four sources (cached document + solution doc +
    # cached supplementary + fresh concept-lookup chunks) and historically
    # bounded none of them jointly — only the cached document was size-checked,
    # and only at cache-write time. Deliberately equal to HYBRID_MAX_DOC_TOKENS
    # (an assembled context should not exceed what a single full document may
    # be) but named separately so the two can diverge. At 80000 no current
    # corpus is affected; it bounds the pathological case, e.g. a textbook
    # solutions manual matching the `'solution' in doc_name` detector.
    # Env-tunable so the drop path can be exercised without a code edit:
    #   SESSION_CONTEXT_MAX_TOKENS=500 python eval/run_eval.py ...
    SESSION_CONTEXT_MAX_TOKENS = int(os.getenv("SESSION_CONTEXT_MAX_TOKENS", "80000"))

    # --- PDF figure-supplement vision pass -------------------------------------
    # The supplement used to render EVERY page at 200 DPI and send EVERY page to
    # gpt-4o. On a 356-page textbook that is 356 in-memory JPEGs before the first
    # API call, plus 356 calls — which blew past the 5-minute indexing watchdog and
    # made large PDFs impossible to index at all.
    #
    # Pages are now pre-screened with pdfplumber, which reports embedded rasters
    # and vector drawings for free. Measured on two econometrics textbooks
    # (356pp and 246pp): median curves per page = 0, so figure pages stand out
    # sharply. `curves >= 10 or raster or no-text` selects 17% and 11% of pages —
    # roughly a 7x cut that keeps the figures, unlike a blanket page cap which
    # would discard every figure in a long book.
    VISION_FIGURE_CURVE_THRESHOLD = int(os.getenv("VISION_FIGURE_CURVE_THRESHOLD", "10"))
    # Backstop so a pathological file (vector-heavy on every page) cannot explode.
    VISION_MAX_PAGES_PER_DOC = int(os.getenv("VISION_MAX_PAGES_PER_DOC", "120"))
    # How often the extraction heartbeat fires, in candidate pages. The indexing
    # watchdog fails a job after 5 minutes without a progress update, so this must
    # stay well below that in wall-clock terms.
    VISION_HEARTBEAT_EVERY = int(os.getenv("VISION_HEARTBEAT_EVERY", "5"))

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
