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
- **Conversational Context Enrichment**: The system detects follow-up queries (answer submissions like "I got 5", clarifications like "what do you mean?", pronoun references like "can you explain that?") and enriches them with context from conversation history. Enrichment is applied only to the semantic search (embeddings) while preserving the original query for filter extraction, ensuring accurate document matching. A guard ensures enrichment only occurs when history contains useful context (problem/document references).
- **Session Context Caching**: When early hybrid routing successfully retrieves a document (e.g., for "problem 2a from PS1"), the full document content is cached in the session. On follow-up queries like answer submissions ("I got 5") or clarifications, the cached document is reused directly without re-searching. This ensures consistent context throughout a conversation about a specific problem. The cache is cleared when users switch topics (mention a different problem or document). Security: ta_id validation prevents cross-tenant context leakage.
- **Pedagogical Constraints**: The TA is designed to help students learn, not give them answers. For problem help, the TA provides only setup (context, relevant formulas, defining equations) but does NOT perform calculations. Students are encouraged to try the work themselves and can validate their answers through dialogue - the TA will confirm if correct or hint at errors without revealing the answer.
- **Patience System for Answer Validation**: When students submit wrong answers, the TA uses a graduated response based on attempt count:
  - 1st wrong attempt: Brief "Not quite. Try again!" with no hints - let them think more
  - 2nd wrong attempt: Provide a helpful hint (formula/method or specific error)
  - 3rd+ wrong attempts: Offer to walk through the approach step-by-step (still without final answer)
  - Attempt counts are tracked per problem in the session cache and reset when switching to a different problem. This mimics patient human tutoring where help escalates as the student struggles.
- **Boundary-Aware Chunking**: Documents are chunked with a focus on preserving semantic boundaries (e.g., problem statements, section headers) to ensure chunks maintain relevant context and are correctly attributed.
- **Context Injection**: Structural context (e.g., document name, problem/section headers) is injected into chunks before embedding to improve retrieval accuracy for specific queries.
- **Post-Retrieval Validation**: An explicit validation step checks if the retrieved chunks align with the specific problem references in the user's query, triggering a full-document fallback if discrepancies are found, even with high LLM reranking scores.
- **UI/UX**: The platform features an admin panel for TA creation and document management, and a student chat interface. It includes dynamic status indicators, in-page notifications, LaTeX math rendering with KaTeX, and drag-and-drop multi-file upload. LaTeX/markdown rendering is protected by prompt-level formatting rules (preventing asterisk/math mixing), frontend sanitization of problematic patterns, and graceful KaTeX error fallbacks.
- **Technology Stack**: Python 3.11, Flask, SQLAlchemy, PostgreSQL with pgvector, OpenAI `text-embedding-3-small` for embeddings (1536 dimensions), OpenAI GPT-4o for LLM tasks, PyPDF2, docx2python (for DOCX with list numbering preservation), antiword/catdoc (for legacy .doc files), openpyxl for file parsing.
- **Document Extraction**: Uses docx2python library for DOCX files to preserve list numbering (a, b, c, 1, 2, 3), tables, footnotes, and document structure. This ensures sub-part labels in problem sets are correctly indexed for accurate LLM navigation.

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