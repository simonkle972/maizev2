# Maize - AI Teaching Assistant Platform

## Overview
Maize is a multi-tenant AI-powered teaching assistant platform designed to help students understand course concepts without providing direct answers. Instructors create AI TAs for their courses, upload various course materials (PDFs, DOCX, spreadsheets, etc.), and students interact with a chat interface to receive guidance. The platform aims to reduce instructor workload and enhance student learning by providing 24/7 AI support. The project's ambition is to become a leading AI educational tool, offering reliable and insightful assistance across a wide range of academic subjects.

## User Preferences
- Cost is not a major consideration - prioritize reliability
- Full LLM extraction preferred over regex
- Persistence of extracted data is crucial
- All query types (structured, conceptual, coverage) should work together

## System Architecture
Maize utilizes a Python Flask backend with PostgreSQL for all persistent data, including SQLAlchemy models for TAs, documents, chat sessions, and messages. Vector embeddings are also stored in PostgreSQL using the `pgvector` extension for efficient and persistent vector search.

Key architectural decisions include:
- **Unified Retrieval Pipeline**: A single, parameterized pipeline handles all document retrieval, eliminating multiple conditional code paths.
- **Pre-retrieval Filtering**: Metadata filters (e.g., by document type, year) are applied in PostgreSQL *before* vector search to narrow down the search space.
- **LLM for Metadata Extraction**: GPT-4o is used for comprehensive and consistent document classification and metadata extraction at upload time, including identifying instructional units and content titles. Admins can review and edit these values (display_name, doc_type, unit_number) before indexing.
- **Human-in-the-Loop Document Categorization**: Documents get LLM-extracted metadata at upload, which admins can correct before indexing. The display_name field allows admins to rename documents for better retrieval matching.
- **Instructional Unit Normalization**: The system normalizes various ways of referring to instructional units (e.g., "Lecture 5", "Week 5") into a consistent `unit_number` for filtering.
- **Per-TA Isolation**: Each AI teaching assistant operates with its own isolated document collection and vector index, filtered by `ta_id`.
- **Hybrid Retrieval Strategy**: The system employs a hybrid retrieval approach, combining initial vector search with LLM-based reranking. For queries requiring deep context or when confidence is low, it intelligently falls back to processing the full document with the LLM. This includes specific handling for queries that reference sub-parts of problems, directly leveraging full document context to ensure accuracy and reduce latency.
- **Conversational Context Enrichment**: The system detects follow-up queries and enriches them with context from conversation history. Enrichment is applied only to the semantic search (embeddings) while preserving the original query for filter extraction, ensuring accurate document matching. A guard ensures enrichment only occurs when history contains useful context (problem/document references).
- **Session Context Caching**: When early hybrid routing successfully retrieves a document (e.g., for "problem 2a from PS1"), the full document content is cached in the session. On ANY subsequent query with conversation history, the cached document is reused directly without re-searching (NOT gated on regex follow-up detection). Solution documents are automatically fetched and included once the student has had at least one Q&A exchange (2+ student messages), allowing the LLM to naturally validate student answers from conversational context. The cache is cleared when users switch topics (mention a different problem or document). Security: ta_id validation prevents cross-tenant context leakage.
- **LLM-Native Answer Validation**: Instead of relying on rigid regex patterns to detect "answer submissions," the system trusts the LLM to naturally understand from conversation context when a student is sharing an answer (e.g., "I got -1", "yes I still get the same thing", "it's 3"). Solution documents are always available during cached sessions, and the prompt instructs the LLM to validate definitively whenever a student communicates a result.
- **Pedagogical Constraints**: The TA is designed to help students learn, not give them answers. The core principle is: NEVER reveal any answer (numerical OR qualitative) before the student has made their own attempt. Solution documents are treated as a secret answer key for validation only - never quoted or revealed proactively. For problem help, the TA provides grounded setup using specific data from documents (actual equations, values, parameters) but does NOT perform calculations or reveal conclusions. Responses avoid filler language and jump straight to substance.
- **Patience System**: Graduated response based on conversation exchange count (not regex-based attempt detection):
  - Early conversation (1st exchange): Brief "Not quite. Try again!" with no hints
  - Moderate (2-3 exchanges): Provide a helpful hint (formula/method or specific error)
  - Extended (4+ exchanges): Walk through the approach step-by-step (still without final answer)
  - Exchange counts are tracked per problem in the session cache and reset when switching problems.
- **Boundary-Aware Chunking**: Documents are chunked with a focus on preserving semantic boundaries (e.g., problem statements, section headers) to ensure chunks maintain relevant context and are correctly attributed.
- **Context Injection**: Structural context (e.g., document name, problem/section headers) is injected into chunks before embedding to improve retrieval accuracy for specific queries.
- **Post-Retrieval Validation**: An explicit validation step checks if the retrieved chunks align with the specific problem references in the user's query, triggering a full-document fallback if discrepancies are found, even with high LLM reranking scores.
- **UI/UX**: The platform features an admin panel for TA creation and document management, and a student chat interface. It includes dynamic status indicators, in-page notifications, LaTeX math rendering with KaTeX, and drag-and-drop multi-file upload. LaTeX/markdown rendering is protected by prompt-level formatting rules (preventing asterisk/math mixing), frontend sanitization of problematic patterns, and graceful KaTeX error fallbacks.
- **Institution Management**: The admin panel supports multi-tenant institutions with a dedicated "Institutions" tab. Each institution has a name, customer_id (for billing), and notes. TAs can optionally be associated with an institution. The admin UI features tabbed navigation, table views with search/filter/sort/pagination, and popup modals for editing.
- **Intelligent Solution Document Fetching**: When a session cache is active (any conversation about a specific problem), the system automatically searches for the corresponding solution document (e.g., "Practice Problems Set 1" → "Solution to Practice Problems Set 1") using name containment and number matching strategies. The solution is always included alongside the problem context, enabling the LLM to validate student answers naturally from conversation context.
- **Technology Stack**: Python 3.11, Flask, SQLAlchemy, PostgreSQL with pgvector, OpenAI `text-embedding-3-small` for embeddings (1536 dimensions), OpenAI GPT-4o for LLM tasks, PyPDF2, docx2python (for DOCX with list numbering preservation), antiword/catdoc (for legacy .doc files), openpyxl for file parsing.
- **Document Extraction**: Uses docx2python library for DOCX files to preserve list numbering (a, b, c, 1, 2, 3), tables, footnotes, and document structure. This ensures sub-part labels in problem sets are correctly indexed for accurate LLM navigation.
- **Vision-Based PDF Extraction**: For image-heavy or handwritten PDFs where text extraction fails (PyPDF2 and pdfplumber both yield <100 chars), the system automatically falls back to GPT-4o vision. Each page is converted to a JPEG image (200 DPI via pdf2image/poppler), sent to GPT-4o with instructions to transcribe handwriting, equations (LaTeX), diagrams (described in [DIAGRAM:] tags), and tables. Supports up to 50 pages per document. This enables indexing of handwritten lecture notes and scanned documents.

## Brand Identity
- **Name**: Maize (named after the creator's tortoiseshell cat)
- **Personality**: Thoughtful, calm, friendly, academically serious
- **Mascot**: Stylized tortoiseshell cat face logo (flat geometric patches, no outlines)
- **Color Palette**: Deep Charcoal (#2B1F1A), Dark Cocoa (#4A3328), Warm Amber (#F2A93B), Burnt Orange (#D9772A), Soft Cream (#F7EBD8), Muted Rose (#D97A8A)
- **Typography**: Inter (humanist sans-serif)
- **Landing Page**: Hero with mascot, features section, comparison table (Maize vs ChatGPT), demo request form
- **Demo Requests**: Submitted via /api/demo-request, emails sent to simon.kleffner@yale.edu if SMTP configured

## External Dependencies
- **OpenAI API**: Used for generating embeddings (`text-embedding-3-small`) and all Large Language Model (LLM) operations (`GPT-4o`).
- **PostgreSQL**: Primary database for all persistent data, including SQLAlchemy models and vector embeddings (via `pgvector` extension).
- **Google Sheets API**: Used for logging QA and indexing diagnostics.