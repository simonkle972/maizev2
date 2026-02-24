# RAG Pipeline Architecture

## Overview

Maize TA uses Retrieval-Augmented Generation (RAG) to answer student questions by retrieving relevant course material and generating pedagogically sound responses.

## Pipeline Stages

### 1. Document Processing
**File**: [src/document_processor.py](../src/document_processor.py)

- Extracts text from PDF, Word, Excel files
- Supports complex PDFs with OCR fallback via pdf2image/poppler
- Chunks text at section boundaries (Problem 1, Question 2, etc.)
- Enriches chunks with context headers: `[filename > Problem 1] actual text...`
- Chunk size: 800 chars, overlap: 200 chars (configurable in config.py)

### 2. Indexing
**File**: [app.py](../app.py) route `/api/teaching-assistants/<ta_id>/index`

- Runs asynchronously in background thread
- Creates embeddings using OpenAI `text-embedding-3-small`
- Stores in `document_chunks` table with pgvector column
- Updates `indexing_jobs` for resumption on crashes
- Progress tracked via `IndexingJob.docs_processed` and `IndexingJob.total_docs`

### 3. Retrieval
**File**: [src/retriever.py](../src/retriever.py)

**Hybrid Mode** (default):
- Uses both vector similarity AND keyword matching
- Enabled when: `HYBRID_RETRIEVAL_ENABLED=True` AND confidence score < threshold
- Falls back to full document text when RAG results are weak
- Detects solution documents automatically and excludes until student attempts

**Vector-Only Mode**:
- Pure cosine similarity search on embeddings
- Top-K retrieval: 20 chunks initially, reranked to 8 best (configurable)

### 4. Response Generation
**File**: [src/response_generator.py](../src/response_generator.py)

- Uses OpenAI `gpt-5.2` with extended thinking mode
- System prompt enforces pedagogical guidelines:
  - Never reveal answers before student attempts
  - Use specific data from documents (grounded responses)
  - Direct, concise tone (no "Sure, I'd be happy to help...")
  - Guide without giving direct answers
- Escapes `#` characters in LaTeX math mode (KaTeX compatibility)
- Streams responses using Server-Sent Events (SSE)

## Solution Document Detection

The retriever automatically identifies solution documents by matching patterns:
- "Solution to Problem Set 1" ↔ "Problem Set 1"
- "Homework 3 Solutions" ↔ "Homework 3"
- Extracts document numbers for flexible matching (PS1, Problem Set 1, etc.)

**CRITICAL**: Solution content is NEVER revealed until student submits an attempt. Used only for answer verification.

## Configuration

**Key Settings** (config.py):
- `LLM_MODEL = "gpt-5.2"` - Extended thinking model
- `LLM_MAX_COMPLETION_TOKENS = 16000` - Response length limit
- `EMBEDDING_MODEL = "text-embedding-3-small"` - 1536 dimensions
- `TOP_K_RETRIEVAL = 20` / `TOP_K_RERANK = 8` - RAG retrieval
- `CHUNK_SIZE = 800` / `CHUNK_OVERLAP = 200` - Document chunking
- `HYBRID_RETRIEVAL_ENABLED = True` - Enables smart fallback
- `HYBRID_CONFIDENCE_THRESHOLD = 6` - Triggers full doc mode
- `HYBRID_MAX_DOC_TOKENS = 80000` - Token limit for full docs

## Performance Considerations

- Vector similarity search is fast due to pgvector indexing
- Full document retrieval (hybrid mode) can be expensive for large docs
- Indexing runs asynchronously to avoid blocking main thread
- Streaming responses improve perceived performance

## Changing RAG Behavior

- **Retrieval logic**: Edit `src/retriever.py` → `hybrid_retrieve()` or `vector_retrieve()`
- **Response guidelines**: Edit `src/response_generator.py` → `BASE_INSTRUCTIONS`
- **Chunking strategy**: Edit `src/document_processor.py` → `chunk_text_with_context()`
- **Reindex TAs after chunking changes** (old chunks remain until reindex)

## Debugging RAG Issues

1. Check indexing completed: `IndexingJob` table or TA admin panel
2. Verify embeddings exist: `SELECT COUNT(*) FROM document_chunks WHERE ta_id='...'`
3. Test retrieval: Check `/api/chat/<session_id>/stream` logs for retrieved chunks
4. Check confidence scores: Hybrid mode triggers when confidence < 6
5. Solution detection: Look for "solution" keyword in document names
