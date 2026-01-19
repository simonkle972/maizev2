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
- **Vector Store**: ChromaDB (persistent, per-TA collections)
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: OpenAI GPT-4o
- **File Parsing**: PyPDF2, python-docx, openpyxl/pandas

## Key Design Decisions (Lessons from Prior Build)
1. **One Retrieval Path**: Unified pipeline with parameterized behavior, not multiple conditional code paths
2. **Pre-retrieval Filtering**: ChromaDB metadata filters applied BEFORE vector search, not after
3. **Full LLM Metadata Extraction**: Consistent approach using GPT-4o for document classification
4. **Instructional Unit Normalization**: "Lecture 5", "Class 5", "Week 5" all map to unit_number=5
5. **PostgreSQL for Persistence**: TA configs, documents, sessions all in PostgreSQL
6. **Per-TA Isolation**: Each TA has separate ChromaDB collection and document folder

## Environment Variables
- `OPENAI_API_KEY` - Required for embeddings and LLM
- `DATABASE_URL` - PostgreSQL connection (auto-provided)
- `SESSION_SECRET` - Flask session secret
- `admin_id` - Admin username for login (required)
- `admin_pw` - Admin password for login (required)

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
- Background indexing with progress tracking (Jan 2026)
- In-page notification bar replacing browser alerts (Jan 2026)
- ChromaDB batch size fix for large document collections (Jan 2026)
- Session-based admin login with username/password (Jan 2026)
- LaTeX math rendering in chat using KaTeX (Jan 2026)
- Drag-and-drop multi-file upload in admin panel (Jan 2026)
- Editable URL slugs for TAs (Jan 2026)
- Initial MVP build (Jan 2026)
- Multi-tenant architecture with PostgreSQL
- LLM-based metadata extraction
- Pre-retrieval filtering in ChromaDB
- Session-based conversation history

## User Preferences
- Cost is not a major consideration - prioritize reliability
- Full LLM extraction preferred over regex
- Persistence of extracted data is crucial
- All query types (structured, conceptual, coverage) should work together
