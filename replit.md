# Maize - AI Teaching Assistant Platform

## Overview
Maize is a multi-tenant AI-powered teaching assistant platform. Instructors create TAs for their courses, upload course materials (PDFs, DOCX, spreadsheets), and students interact with a chat interface to get help understanding concepts without receiving direct answers.

## Project Structure
```
maize/
├── app.py                    # Main Flask app with all routes
├── config.py                 # Configuration and environment variables
├── models.py                 # SQLAlchemy models (TeachingAssistant, Document, ChatSession, ChatMessage)
├── src/
│   ├── document_processor.py # Document ingestion, text extraction, chunking, LLM metadata extraction
│   ├── retriever.py          # Vector search with ChromaDB, pre-retrieval filtering
│   ├── response_generator.py # GPT-4o response generation with context
│   └── query_analyzer.py     # Query understanding and classification
├── templates/
│   ├── landing.html          # Home page at /
│   ├── admin.html            # Admin panel at /admin
│   ├── chat.html             # Student chat interface at /<slug>
│   └── 404.html              # Not found page
├── static/css/style.css      # All CSS styles
├── data/courses/             # Per-TA document storage
└── chroma_db/                # Vector indices per TA
```

## Key URLs
- `/` - Landing page
- `/admin` - Admin panel (requires admin key)
- `/<slug>` - Student chat interface for a TA (e.g., `/financecourse`)

## Technology Stack
- **Backend**: Python 3.11, Flask
- **Database**: PostgreSQL (via SQLAlchemy)
- **Vector Store**: PostgreSQL with pgvector extension (persistent, uses ivfflat indexing)
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **Chunking**: chunk_size=800, overlap=200 (increased from 512/50 to reduce sub-problem boundary splits)
- **LLM**: OpenAI GPT-4o
- **File Parsing**: PyPDF2, python-docx, openpyxl/pandas

## Key Design Decisions (Lessons from Prior Build)
1. **One Retrieval Path**: Unified pipeline with parameterized behavior, not multiple conditional code paths
2. **Pre-retrieval Filtering**: PostgreSQL metadata filters applied BEFORE vector search, not after
3. **Full LLM Metadata Extraction**: Consistent approach using GPT-4o for document classification
4. **Instructional Unit Normalization**: "Lecture 5", "Class 5", "Week 5" all map to unit_number=5
5. **PostgreSQL for Persistence**: TA configs, documents, sessions, AND vector embeddings all in PostgreSQL
6. **Per-TA Isolation**: Each TA has separate document collection filtered by ta_id

## Environment Variables
- `OPENAI_API_KEY` - Required for embeddings and LLM
- `DATABASE_URL` - PostgreSQL connection (auto-provided)
- `SESSION_SECRET` - Flask session secret
- `admin_id` - Admin username for login (required)
- `admin_pw` - Admin password for login (required)
- `qa_log_googlesheet` - Google Sheet ID for QA logging (optional)

## How to Use
1. Go to `/admin` and login with the admin key
2. Create a new TA with a slug (URL path), name, and course name
3. Upload course documents (PDF, DOCX, XLSX, TXT, PPTX)
4. Click "Index Documents" to process and create the vector index
5. Students can access the TA at `/<slug>` and ask questions

## Document Types Supported
- PDF (lectures, assignments, exams)
- DOCX (readings, notes)
- XLSX (data, spreadsheets)
- TXT (plain text)
- PPTX (lecture slides)

## Recent Changes
- Hybrid Retrieval with Full-Document Fallback (Jan 2026)
  - Addresses core retrieval issue: chunks for distant subproblems (e.g., "problem 2f") often missed
  - When LLM reranking shows low confidence (top score < 6 or low score spread), triggers full-doc fallback
  - Identifies target document from query analysis or most-frequent document in retrieved chunks
  - Extracts full document text and passes to LLM instead of chunks
  - Configurable thresholds: HYBRID_CONFIDENCE_THRESHOLD=6, HYBRID_SCORE_SPREAD_THRESHOLD=2, HYBRID_MAX_DOC_TOKENS=80000
  - New diagnostic fields in QA logs: hybrid_fallback_triggered, hybrid_fallback_reason, hybrid_doc_filename, hybrid_doc_tokens
  - Key insight: ChatGPT with full document reliably answers any question; this replicates that for low-confidence retrievals
  - Preserves chunk-based retrieval for high-confidence queries (cheaper, faster)
- Boundary-Aware Chunking (Jan 2026)
  - Fixed critical bug where all chunks were labeled "Problem 1" even when containing Problem 2 content
  - Root cause: chunks spanning section boundaries inherited context from the PREVIOUS section
  - Solution: Force chunk breaks at section headers (Problem X, Question X, etc.)
  - When a header is detected within a chunk window, end chunk at boundary and start next chunk exactly at header
  - No overlap applied across section boundaries to prevent context contamination
  - Chunks now correctly labeled with the section they belong to
- Index Logging to Google Sheets (Jan 2026)
  - Added comprehensive indexing diagnostics to `index_logs` tab in Google Sheets
  - Logs 16 fields per chunk: timestamp, ta_id, ta_slug, file_name, doc_type, total_pages, raw_text_length, chunk_index, total_chunks, chunk_text_length, chunk_context, chunk_text_preview (300 chars), enriched_text_preview (300 chars), has_embedding, status, error_message
  - Enables diagnosis of text extraction and chunking issues
  - Batch logging for efficiency (all chunks logged at end of indexing)
- Context Injection for Chunking (Jan 2026)
  - Added structural context to chunk embeddings to improve retrieval
  - Chunks now prefixed with document name and detected problem/section headers
  - Example: "[ProblemSet1.pdf > Problem 2: Airline Tickets] c) (0.5 points)..."
  - Fixes vocabulary mismatch where "problem 2c" query didn't match chunks containing just "c)"
  - Simple regex patterns detect Problem/Question/Section/Part/Exercise headers
- Phase 2: LLM-Based Reranking (Jan 2026)
  - Replaced keyword reranking with GPT-4o-mini semantic reranking
  - Retrieves 20 chunks initially, LLM scores each for relevance 0-10
  - LLM understands specific problem references (e.g., "problem 2f" vs "3d")
  - Returns top 8 most relevant chunks based on LLM judgment
  - Includes reasoning for each chunk score for observability
  - QA logging: rerank_applied, rerank_method, rerank_latency_ms, llm_score_top1, llm_score_top8, vector_score_top1, top_reasons, pre_rerank_candidates (33 columns, A-AG)
  - Robust fallback: pads with vector-order chunks if LLM returns incomplete results
- Document-Aware Query Matching (Jan 2026)
  - Added fallback mechanism when regex patterns don't detect structured queries
  - Tokenizes query and document filenames, scores overlap to find matches
  - Handles unusual document names like "Grow Co. 1" that regex can't anticipate
  - Applies filename filter pre-retrieval (same unified pipeline, new parameter)
  - Logs match source, score, and matched tokens for observability
- Text sanitization for indexing (Jan 2026)
  - Added sanitize_text() to strip null bytes and control characters from PDFs
  - Fixes PostgreSQL "string literal cannot contain NUL" errors
- Phase 1 Retrieval Observability (Jan 2026)
  - Added 11 new diagnostic fields to QA logging (now 25 columns total, A-Y)
  - New fields: total_chunks_in_ta, filters_applied, filter_match_count, retrieval_method, is_conceptual, score_top1, score_top8, score_mean, score_spread, chunk_scores, chunk_sources_detail
  - Both streaming and non-streaming endpoints now log QA entries
  - Enables diagnosis of retrieval issues (precision vs. extraction problems)
- QA logging to Google Sheets with timing metrics (Jan 2026)
  - Async logging via background threads for non-blocking operation
- Streaming chat responses with SSE (Jan 2026)
- Dynamic status indicators during query processing (Jan 2026)
- Improved LaTeX rendering with delimiter conversion (Jan 2026)
- Fixed Flask app context issue in streaming endpoint (Jan 2026)
- Migrated from ChromaDB to PostgreSQL pgvector for persistent vector storage (Jan 2026)
- Added DocumentChunk model with embeddings stored in PostgreSQL (Jan 2026)
- Added admin endpoint to reset stuck indexing status (Jan 2026)
- Document content stored in PostgreSQL (file_content column) for persistence across deployments (Jan 2026)
- Admin panel session expiration handling with user-friendly redirect (Jan 2026)
- Background indexing with progress tracking (Jan 2026)
- In-page notification bar replacing browser alerts (Jan 2026)
- Session-based admin login with username/password (Jan 2026)
- LaTeX math rendering in chat using KaTeX (Jan 2026)
- Drag-and-drop multi-file upload in admin panel (Jan 2026)
- Editable URL slugs for TAs (Jan 2026)
- Initial MVP build (Jan 2026)
- Multi-tenant architecture with PostgreSQL
- LLM-based metadata extraction
- Session-based conversation history

## User Preferences
- Cost is not a major consideration - prioritize reliability
- Full LLM extraction preferred over regex
- Persistence of extracted data is crucial
- All query types (structured, conceptual, coverage) should work together
