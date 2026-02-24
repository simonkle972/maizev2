# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Maize TA is an AI-powered teaching assistant platform built with Flask and PostgreSQL+pgvector. It uses RAG (Retrieval-Augmented Generation) to answer student questions based on course materials.

## Quick Start

### Local Development
```bash
./run_local.sh  # Starts Docker PostgreSQL + Flask dev server
# Access at http://localhost:5000
```

### Database Migrations
```bash
export DOTENV_PATH=.env.local
flask db migrate -m "Description"  # After modifying models.py
flask db upgrade                   # Apply migrations
```

**CRITICAL**: Never use `db.create_all()` - always use migrations.

## Architecture

### Core Systems
- **RAG Pipeline**: OpenAI embeddings + pgvector retrieval → pedagogical responses
- **Parallel Sessions**: Students use `session['_student_id']`, professors use `session['_user_id']`
- **Async Indexing**: Background document processing with resumption support

### Data Model
```
Institution → TeachingAssistant → Documents/DocumentChunks/ChatSessions
```

**Key Detail**: TA IDs are **strings** (not integers!)
- Routes: Use `<ta_id>`, never `<int:ta_id>`
- Database: `TeachingAssistant.id = db.String(32)`

## Critical Patterns

### Flask Imports
Always import `session` when using session storage:
```python
from flask import Blueprint, ..., session
```

### Migrations Workflow
1. Modify `models.py`
2. `flask db migrate -m "Description"`
3. Review generated file
4. `flask db upgrade` locally
5. Commit both `models.py` and migration file

### LaTeX Rendering
- Frontend: KaTeX (`$...$` inline, `$$...$$` display)
- Backend: Escape `#` in LaTeX → `$x\#1$` (see `escape_hash_in_latex`)

### Streaming Responses
```python
def generate():
    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
return Response(stream_with_context(generate()), mimetype='text/event-stream')
```

## File Structure

```
app.py                    # Flask routes, indexing, chat endpoints
models.py                 # SQLAlchemy models (students, TAs, docs, chunks)
auth.py                   # Authentication (role-specific login routes)
auth_student.py           # Parallel session system for students
student.py                # Student dashboard and chat routes
professor.py              # Professor TA management routes
src/
  ├── document_processor.py  # PDF/Word/Excel extraction, chunking
  ├── retriever.py          # RAG retrieval (hybrid + vector modes)
  ├── response_generator.py # OpenAI streaming chat
  └── qa_logger.py          # Optional Google Sheets logging
```

## Detailed Documentation

For comprehensive information, see:
- **[docs/RAG.md](docs/RAG.md)** - RAG pipeline architecture, configuration, debugging
- **[docs/DATABASE.md](docs/DATABASE.md)** - Migration workflow, data model, connection pooling
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Production deployment, git workflow, security

## Common Tasks

**Add a model field**:
1. Edit `models.py`
2. `flask db migrate -m "Add field_name"`
3. Test: `flask db upgrade`

**Change RAG behavior**:
- Retrieval: `src/retriever.py`
- Response guidelines: `src/response_generator.py → BASE_INSTRUCTIONS`
- Chunking: `src/document_processor.py → chunk_text_with_context()`

**Debug RAG**:
- Check indexing: TA admin panel or `IndexingJob` table
- Verify chunks: `SELECT COUNT(*) FROM document_chunks WHERE ta_id='...'`
- Check logs: Look for confidence scores and retrieval diagnostics

## Configuration

**Key Settings** (config.py):
- `LLM_MODEL = "gpt-5.2"` - Extended thinking model
- `EMBEDDING_MODEL = "text-embedding-3-small"`
- `TOP_K_RETRIEVAL = 20` / `TOP_K_RERANK = 8`
- `HYBRID_RETRIEVAL_ENABLED = True` - Smart fallback to full docs

## Deployment

```bash
# Production (on VPS)
cd /opt/maize
sudo -u maize git pull
sudo -u maize ./venv/bin/flask db upgrade
systemctl restart maize
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for full details.

## Security

- Never commit `.env` files (gitignored)
- Use app-specific passwords for SMTP
- Session secrets auto-generated if not provided
- Admin auth: Environment variables (TODO: migrate to User model)
