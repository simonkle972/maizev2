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
- Phase 2 Retrieval Improvement (Jan 2026)
  - Retrieve more chunks (20) and rerank with LLM to select top 8
  - Added GPT-4o-based reranker that scores chunk relevance to query
  - Added content_identifier metadata field for document name recognition (e.g., "Grow Co. I", "Extra Problem Set")
  - Query analyzer now checks known content_identifiers and applies pre-retrieval filtering
  - QA logging expanded to 28 columns (A-AB) with rerank metrics: rerank_applied, pre_rerank_scores, rerank_scores
  - Config: TOP_K_RETRIEVAL=20, TOP_K_RERANK=8
- Phase 1 Retrieval Observability (Jan 2026)
  - Added 11 new diagnostic fields to QA logging
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
