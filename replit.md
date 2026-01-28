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
- **LLM for Metadata Extraction**: GPT-4o is used for comprehensive and consistent document classification and metadata extraction during ingestion, including identifying instructional units and content titles.
- **Instructional Unit Normalization**: The system normalizes various ways of referring to instructional units (e.g., "Lecture 5", "Week 5") into a consistent `unit_number` for filtering.
- **Per-TA Isolation**: Each AI teaching assistant operates with its own isolated document collection and vector index, filtered by `ta_id`.
- **Hybrid Retrieval Strategy**: The system employs a hybrid retrieval approach, combining initial vector search with LLM-based reranking. For queries requiring deep context or when confidence is low, it intelligently falls back to processing the full document with the LLM. This includes specific handling for queries that reference sub-parts of problems, directly leveraging full document context to ensure accuracy and reduce latency.
- **Boundary-Aware Chunking**: Documents are chunked with a focus on preserving semantic boundaries (e.g., problem statements, section headers) to ensure chunks maintain relevant context and are correctly attributed.
- **Context Injection**: Structural context (e.g., document name, problem/section headers) is injected into chunks before embedding to improve retrieval accuracy for specific queries.
- **Post-Retrieval Validation**: An explicit validation step checks if the retrieved chunks align with the specific problem references in the user's query, triggering a full-document fallback if discrepancies are found, even with high LLM reranking scores.
- **UI/UX**: The platform features an admin panel for TA creation and document management, and a student chat interface. It includes dynamic status indicators, in-page notifications, LaTeX math rendering with KaTeX, and drag-and-drop multi-file upload.
- **Technology Stack**: Python 3.11, Flask, SQLAlchemy, PostgreSQL with pgvector, OpenAI `text-embedding-3-small` for embeddings (1536 dimensions), OpenAI GPT-4o for LLM tasks, PyPDF2, python-docx, openpyxl for file parsing.

## External Dependencies
- **OpenAI API**: Used for generating embeddings (`text-embedding-3-small`) and all Large Language Model (LLM) operations (`GPT-4o`).
- **PostgreSQL**: Primary database for all persistent data, including SQLAlchemy models and vector embeddings (via `pgvector` extension).
- **Google Sheets API**: Used for logging QA and indexing diagnostics.