# Maize Architecture Review — Indexing + Retrieval

**Created:** 2026-05-23 · **Status:** in flight (scaffold only — sections 3.1, 3.2, 3.3 populate as the work lands) · **Grounded in:** [Phase A implementation plan](maize-retrieval-phase-a-implementation-plan-2026-05-22.md), [residual-failures research](maize-retrieval-residual-failures-research-2026-05-22.md), [top-of-funnel research](maize-retrieval-top-of-funnel-research-2026-05-23.md), the 27-query manual battery from 2026-05-23.

## Context

By 2026-05-23, the Phase A retrieval refactor had shipped Stages 1, 2, 2B, 3, 4, plus Stage 5 Step 1 (the filename short-circuit + structural-injection scope fix, commit `b402d5d`). Net of all of that on the ECON S1117 corpus vs the legacy V1 path: Type A failures 0% → 100%, Type C 14% → 57%, Type D 40% → 60% with 0% solution-leak, working cases 71% → 59% (partial recovery from the 41% mid-refactor low).

User flagged a meta-pattern after this work landed: **we have been reactive.** V2 was reactive to the ECON S1117 evaluation failures. The residual-failures research was reactive to V2's regressions. The top-of-funnel research was reactive to doc-switching failures observed in the manual battery. Each piece is individually defensible and grounded in research, but the assembly is uncoordinated. The metaphor the user offered: we keep adding fancy parts — a researched short-circuit, a researched intent classifier — without checking whether the underlying car is the right car.

V1 works well for every TA except ECON S1117, which tells us the architecture isn't catastrophically wrong — but it also means we have only negative evidence (specific failures fixed) and no positive evidence for which parts are *right*. This document is the audit that fills that gap. It is read-only with respect to code; downstream implementation plans depend on what we find here.

## Scope

**In scope:** indexing pipeline (file upload → DocumentChunk rows written) and retrieval pipeline (student query enters Flask → chunks handed to response generator).

**Out of scope:** response generation (`src/response_generator.py`). The "chunks → LLM prompt → student response" stage is a separate concern; auditing it would balloon this review.

User-stated rationale for including indexing: "If we do a better job indexing, we not only make retrieval easier, we also make it quicker and more reliable." Upstream changes can reduce downstream complexity. Examples of leverage we want to evaluate: precomputed section-path tags on chunks could simplify within-doc structural disambiguation (Type B failures); contextual-prefix embeddings at ingest could shrink the rerank's job; smarter chunking could reduce the need for the late-fallback hybrid_full_doc path.

## 3.1 Architecture map

**Status:** populated 2026-05-23. Mapped manually (Explore subagents bailed on phantom permission issues twice). All `file:line` citations verified against HEAD at commit `b402d5d`.

This section is a structured walkthrough of two pipelines: indexing (file upload → `DocumentChunk` rows in DB) and retrieval (student query → chunks handed to response generator). For each step:
- **Entry point** — function + file:line range
- **Inputs** — args + relevant state read
- **Outputs** — return values + side effects
- **Cost signals** — LLM calls, DB queries, expensive Python ops, latency markers
- **Failure modes** — what breaks if this misbehaves
- **Callers / dependents** — who invokes + who consumes

### 3.1.A Indexing pipeline (file upload → DocumentChunk rows)

#### A1. Upload Flask route — admin path

- **Entry point**: `upload_document` ([app.py:737-867](../app.py#L737-L867)). Admin-only; analogous professor route exists.
- **Inputs**: Flask `request.files['file']`, `ta_id` from URL path.
- **Outputs**: HTTP 200 JSON `{success, document_id, filename, display_name, doc_type, ...}`. Side effects: writes a `Document` row (DB) and persists `file_content` as bytes on the row + an on-disk path at `data/courses/<ta_id>/docs/<safe_filename>`. Launches a background thread for LLM metadata enrichment.
- **Cost signals**: 1 DB insert + 1 DB commit + Python file-read in-memory. No LLM call on the request path. Latency: ~50-200ms for the response itself; the background thread takes 10-30s.
- **Failure modes**: file ext not in `{pdf, docx, doc, xlsx, xls, txt, pptx, ppt}` → 400. Background thread failure logged but never aborts the response.
- **Callers / dependents**: admin UI (`templates/admin_manage_ta.html` upload widget). Downstream: `run_indexing_task` is NOT triggered here — indexing is triggered by a separate POST to `/admin/api/tas/<ta_id>/reindex`. Background metadata thread populates `Document.doc_type`, `assignment_number`, `instructional_unit_number`, `instructional_unit_label`, `content_title`, `metadata_extracted`, `extraction_metadata` via `extract_metadata_from_file_content`.

#### A2. Synchronous filename-heuristic metadata

- **Entry point**: `infer_doc_metadata_from_filename` ([src/document_processor.py:1173-1253](../src/document_processor.py#L1173-L1253)).
- **Inputs**: filename string.
- **Outputs**: dict with optional `doc_type`, `assignment_number`, `instructional_unit_number`. Empty dict when no pattern matches.
- **Cost signals**: pure regex, sub-millisecond.
- **Failure modes**: regex match returning the wrong number (e.g. "Quiz 2025" matching year as assignment number) — historically caused upload-time mis-tagging. Wrapped in try/except so it can't break the upload.
- **Callers / dependents**: called once at upload time ([app.py:789](../app.py#L789)). Sets initial `doc_type`/`assignment`/`unit` on the `Document` row. These get overwritten later if the LLM extractor returns non-null values.

#### A3. Background LLM metadata thread (per-doc, post-upload)

- **Entry point**: `extract_metadata_bg` ([app.py:827-849](../app.py#L827-L849)) wrapping `extract_metadata_from_file_content` ([src/document_processor.py:1255-1299](../src/document_processor.py#L1255-L1299)) → `extract_text_from_file` → `extract_metadata_with_llm`.
- **Inputs**: file bytes, file_ext, original_filename. Loads `Document` by id, re-extracts text, calls LLM.
- **Outputs**: updates `Document.doc_type`/`assignment_number`/`instructional_unit_number`/`instructional_unit_label`/`content_title`/`metadata_extracted`/`extraction_metadata`. Commits.
- **Cost signals**: 1 text extraction (could be 5-30s for PDF vision fallback), 1 OpenAI chat-completion call (gpt-4o, ~1-2s). Runs in a daemon thread so the upload response returns fast.
- **Failure modes**: any exception caught at [app.py:847-848](../app.py#L847-L848); logged but silent to the user. Result: doc's `metadata_extracted` stays False until next indexing run. Note: this thread runs BEFORE indexing — so its results inform the synchronous metadata that the indexing pipeline later overwrites in `process_and_index_documents_resumable`. There's a double LLM-metadata call (here + in the indexing pipeline) that may be redundant.
- **Callers / dependents**: only the upload route. The doc's `metadata_extracted=True` flag is what the admin UI polls for to show "metadata loaded".

#### A4. Background indexing job orchestration

- **Entry point**: `run_indexing_task` ([app.py:1034-1149](../app.py#L1034-L1149)).
- **Inputs**: `ta_id`, optional `job_id`, `is_resume` flag (True = incremental, False = full rebuild).
- **Outputs**: `IndexingJob` row updates (status, progress, chunks_created). TA-level fields: `is_indexed`, `indexing_status`, `indexing_progress`, `indexing_warnings` (per-doc failure summary).
- **Cost signals**: invokes `process_and_index_documents_resumable` which dominates. Job-orchestration overhead is small (~50ms of DB writes per progress update).
- **Failure modes**: bare `except Exception` at [app.py:1126-1149](../app.py#L1126-L1149) catches any error, logs traceback, marks `IndexingJob.status='failed'`, updates TA status. Result: partial indexing state can be left in the DB. Resumable design relies on `Document.last_indexed_at IS NULL` to detect "needs processing".
- **Callers / dependents**: triggered by POST `/admin/api/tas/<ta_id>/reindex` ([app.py:~1190](../app.py#L1190)) and equivalent professor route. Threading launched at [app.py:1190, 1218, 1242](../app.py#L1190-L1242).

#### A5. Main indexing loop — resumable

- **Entry point**: `process_and_index_documents_resumable` ([src/document_processor.py:1625-1947](../src/document_processor.py#L1625-L1947)). Note: there's also a non-resumable variant `process_and_index_documents` ([line 1334](../src/document_processor.py#L1334)) — appears to be **dead code** (no live callers; the indexing task always uses the resumable variant). Candidate for elimination.
- **Inputs**: `ta_id`, `progress_callback`, `resume_from_doc_id` flag.
- **Outputs**: `{chunks_indexed, docs_succeeded, docs_failed}`. Side effects: writes `DocumentChunk` rows, updates `Document` fields (`bm25_tsvector`, `doc_role`, `doc_role_provenance`, `doc_category`, `extraction_metadata`, `last_indexed_at`).
- **Cost signals**: per doc: 1 text extraction (5-30s), 1 LLM metadata call (1-2s), 1 LLM `classify_doc_role` call (1-2s — DEPRECATED, see A8), 1 LLM `classify_doc_category` call (1-2s), batched embedding calls (1 per ~100 chunks, ~500ms each), N `DocumentChunk` inserts. For the 75-doc ECON S1117 corpus, end-to-end indexing is ~10-15 min.
- **Failure modes**: per-doc try/except catches extraction failures and records them in `docs_failed` list. A doc with `extraction_metadata._indexing_status='extraction_failed'` is skipped on retries until resolved. Last_indexed_at semantics: NULL → needs processing; non-NULL → skip on resume.
- **Callers / dependents**: only `run_indexing_task`.

#### A6. Text extraction dispatch

- **Entry point**: `extract_text_from_file` ([src/document_processor.py:297-334](../src/document_processor.py#L297-L334)). Dispatches on file extension to per-format extractors.
- **Inputs**: file path on disk.
- **Outputs**: `(text, page_count)` tuple.
- **Cost signals**: dispatches; cost is in the per-format extractor (A7).
- **Failure modes**: unsupported extension → empty text. Downstream caller marks the doc `_indexing_status='extraction_failed'`.
- **Callers / dependents**: `process_and_index_documents_resumable` per doc. Also called by `extract_metadata_from_file_content` for the upload-time metadata thread.

#### A7. Per-format extractors

Each format has its own routine in `src/document_processor.py`. Common pattern: text extraction first, vision fallback for image-only / unparseable content.

**PDF — three-tier extraction** ([src/document_processor.py:336-571](../src/document_processor.py#L336-L571)):
- Tier 1: `_extract_pdf_pdfplumber` ([line 537](../src/document_processor.py#L537)) — text + tables.
- Tier 2: `_extract_pdf_pypdf2` ([line 572](../src/document_processor.py#L572)) — fallback for files pdfplumber rejects.
- Tier 3: `_extract_pdf_vision` ([line 444](../src/document_processor.py#L444)) — `pdf2image` + GPT-4o vision per page when both text-extractors return empty or insufficient. Calls `Config.VISION_MODEL` (gpt-4o) per page; expensive (5-30s for a 10-page doc).
- Supplement: `_supplement_pdf_with_figures` ([line 357](../src/document_processor.py#L357)) — runs after text extraction succeeds, uses vision per page to extract `[FIGURE: ...]` blocks and splices under page markers. Adds 5-15s per doc.
- **Cost signals**: vision passes are the dominant cost. PDF with figures triggers 2 vision passes (figure supplement + possibly vision fallback if text was empty). All page renders happen at 200 DPI via `pdf2image.convert_from_path` (env-driven `Config.POPPLER_PATH`).
- **Failure modes**: poppler-not-found returns empty. Single-page vision failure logged but doesn't abort the doc.

**DOCX** ([src/document_processor.py:589-714](../src/document_processor.py#L589-L714)):
- Primary: `python-docx` for structured walk (paragraphs + tables + list-level prefix detection via `_get_paragraph_list_prefix`).
- Fallback: `docx2python` for content that python-docx misses, flattened via `_flatten_docx2python_content`.
- **Cost signals**: pure CPU. Fast (~100-500ms per doc).
- **Failure modes**: nested tables can produce malformed text. No vision fallback.

**PPTX** ([src/document_processor.py:766-877](../src/document_processor.py#L766-L877)):
- Iterates slides, recurses into `_iter_shapes` for group shapes.
- TEXT_FRAME shapes → text extraction.
- PICTURE shapes → vision via `Config.VISION_MODEL`.
- CHART shapes → structured chart data via `python-pptx` chart API.
- Slide placeholder emitted even on extraction failure to preserve slide-N alignment for downstream structural injection (this is load-bearing — see B14).
- **Cost signals**: vision per picture shape; can be 30+ vision calls for a slide-heavy deck.
- **Failure modes**: missing chart parts, malformed shapes — caught per-shape, doc continues. Group-shape recursion is recent and may miss exotic layouts ("needs verification" — line 766+ has complex shape-type dispatch I didn't fully trace).

**Other formats**: XLSX via `extract_excel` ([line 756](../src/document_processor.py#L756)) — openpyxl-based, sheet-by-sheet. TXT/MD/code → raw read. JSON, CSV, Jupyter notebooks have dedicated handlers ([lines 232-296](../src/document_processor.py#L232-L296)). Images (PNG/JPG/etc.) → `extract_image` ([line 180](../src/document_processor.py#L180)) → direct vision call.

#### A8. LLM metadata extraction (per-doc, during indexing)

- **Entry point**: `extract_metadata_with_llm` ([src/document_processor.py:879-977](../src/document_processor.py#L879-L977)) called from `process_and_index_documents_resumable` ([line ~1413, ~1727](../src/document_processor.py#L1413)).
- **Inputs**: extracted text (first 3000 chars), filename.
- **Outputs**: dict `{doc_type, assignment_number, instructional_unit_number, instructional_unit_label, content_title, ...}`.
- **Cost signals**: 1 gpt-4o-mini chat-completion call per doc (~500-1000ms, ~$0.0005). Note: this is the SECOND LLM metadata call per doc (the first ran in the upload-time background thread A3). For a freshly-uploaded doc that's then indexed, the LLM is called twice on the same content. The pipeline only overwrites empty fields, so values are preserved, but the duplicated cost is a candidate audit finding.
- **Failure modes**: 3 retries, then defaults to empty dict. Doesn't abort the doc.
- **Callers / dependents**: only the indexing pipeline. Output stored in `Document.extraction_metadata` JSON + denormalized to typed columns.

#### A9. doc_role classification (DEPRECATED but still runs)

- **Entry point**: `classify_doc_role` ([src/document_processor.py:1096-1171](../src/document_processor.py#L1096-L1171)). Called from `process_and_index_documents_resumable` at lines [~1438](../src/document_processor.py#L1438) and [~1751](../src/document_processor.py#L1751).
- **Inputs**: text + filename.
- **Outputs**: `(role, confidence, rationale)` triple. Role is one of `{problem, solution, lecture, syllabus, reference, other}`. Persisted to `Document.doc_role` + `DocumentChunk.doc_role` (denormalized) + `Document.doc_role_provenance` JSON.
- **Cost signals**: 1 gpt-4o-mini chat-completion call per doc (~500-1000ms, ~$0.0005). 3 retries on failure.
- **Failure modes**: invalid role → defaults to "other". Skipped if `doc_role_provenance.source == 'professor'` (manual override).
- **Callers / dependents**: NO LIVE READERS in retrieval code. `retriever.py` has zero references to `doc_role`. This is a Stage-2-era column that Stage 2B superseded with `doc_category` + a per-TA configurable vocabulary. **Audit finding candidate: classify_doc_role is dead-weight at index time** (~10-15 min wasted across the 75-doc corpus, ~$0.04 wasted in LLM cost). Safe to remove.

#### A10. doc_category classification (Stage 2B — load-bearing)

- **Entry point**: `classify_doc_category` ([src/document_processor.py:1000-1094](../src/document_processor.py#L1000-L1094)). Called from `process_and_index_documents_resumable` at lines [~1455-1469](../src/document_processor.py#L1455-L1469) and [~1775-1799](../src/document_processor.py#L1775-L1799).
- **Inputs**: text + filename + the parent TA's `doc_categories` list (Stage 2B per-TA controlled vocab — JSON column on `TeachingAssistant`).
- **Outputs**: `(slug, confidence, rationale)` triple. Slug must be one of the TA's configured categories. Persisted to `Document.doc_category` + `DocumentChunk.doc_category` (denormalized).
- **Cost signals**: 1 gpt-4o-mini chat-completion call per doc (~500-1000ms, ~$0.0005). 3 retries.
- **Failure modes**: invalid slug → defaults to "other" or last entry. Skipped if `doc_category` is already set (treated as authoritative — admin UI or backfill).
- **Callers / dependents**: LIVE READERS in retrieval:
  - `hybrid_doc_search` Stage 5 short-circuit Path A (category+number DB lookup).
  - `llm_rerank` prompt template (surfaces `[category: <slug>]` per chunk).
  - Stored on `DocumentChunk.doc_category` for the latter.

#### A11. BM25 tsvector population

- **Entry point**: inline in `process_and_index_documents_resumable` at line [~1470](../src/document_processor.py#L1470) and [~1802](../src/document_processor.py#L1802). `doc.bm25_tsvector = db.func.to_tsvector('english', sanitize_text(text))`.
- **Inputs**: full extracted text after `sanitize_text` strips control chars.
- **Outputs**: PostgreSQL `tsvector` value persisted to `Document.bm25_tsvector`. GIN index `ix_documents_bm25_tsvector` enables fast lookup.
- **Cost signals**: PostgreSQL-side computation, <100ms per doc. No external API.
- **Failure modes**: would silently produce empty tsvector if text is empty. Stored as NULL when no text extracted.
- **Callers / dependents**: read by `hybrid_doc_search` BM25 ranking step in V2 mode (B12.V2).

#### A12. Chunking with section context

- **Entry point**: `chunk_text_with_context` ([src/document_processor.py:65-154](../src/document_processor.py#L65-L154)). Uses `extract_section_headers` ([line 12](../src/document_processor.py#L12)) and `get_context_for_position` ([line 52](../src/document_processor.py#L52)).
- **Inputs**: full text, chunk_size (default 800 tokens), overlap (default 200), filename.
- **Outputs**: list of `{original_text, text, context}` dicts. `context` is the section header for the chunk (e.g. "Slide 3:", "Problem 2:", "--- Page 7 ---"). Used downstream as `DocumentChunk.chunk_context`, which structural injection (B15) matches against.
- **Cost signals**: pure CPU. ~50-200ms per doc. Header extraction is regex-based, supporting "Problem N", "Question N", "Section I/II/III", "Part A", "Exercise N", "Slide N", "--- Page N ---" patterns.
- **Failure modes**: docs without any headers → chunks all share `context=""` and structural injection can't help. Multi-layer structure ("Section II Part b") is NOT captured — chunk_context records only the most-recent single header.
- **Callers / dependents**: `process_and_index_documents_resumable`. Output drives downstream embedding + `chunk_context` denormalization.

#### A13. Embedding generation

- **Entry point**: inline in `process_and_index_documents_resumable` ([lines ~1540-1560](../src/document_processor.py#L1540-L1560) for fresh, [~1857-1865](../src/document_processor.py#L1857-L1865) for resumable). Calls OpenAI `embeddings.create(model=Config.EMBEDDING_MODEL, input=batch_texts)`.
- **Inputs**: list of enriched chunk texts (chunk_text + context). Batched to 100 chunks per API call.
- **Outputs**: list of 1536-dim embedding vectors. Persisted to `DocumentChunk.embedding` (pgvector column).
- **Cost signals**: ~500ms per batch of 100. `Config.EMBEDDING_MODEL = "text-embedding-3-small"`. Cost ~$0.00002 per chunk.
- **Failure modes**: API failure → exception propagates, aborts current doc, marks doc failed. No retry at the embedding step.
- **Callers / dependents**: only the indexing pipeline. Downstream: pgvector cosine_distance in `hybrid_doc_search` dense ranking and main chunk vector search (B12, B13).

#### A14. DocumentChunk persistence + denormalization

- **Entry point**: inline at [src/document_processor.py:~1567-1581](../src/document_processor.py#L1567-L1581) and [~1867-1882](../src/document_processor.py#L1867-L1882). Each chunk gets a `DocumentChunk` row with: `ta_id`, `document_id`, `chunk_index`, `chunk_text`, `chunk_context`, `doc_type`, `assignment_number`, `instructional_unit_number`, `instructional_unit_label`, `file_name`, `doc_role`, `doc_category`, `embedding`.
- **Inputs**: chunks list + embeddings list (parallel arrays).
- **Outputs**: N DocumentChunk INSERTs per doc.
- **Cost signals**: DB INSERTs batched. ~50-100ms per batch.
- **Failure modes**: SQLAlchemy commit failure → wrapped retry via `db_commit_with_retry`. Stale chunks from prior failed runs are cleared at the start of fresh indexing.
- **Callers / dependents**: only the indexing pipeline. Downstream: chunks are the retrieval surface.

---

### 3.1.B Retrieval pipeline (student query → chunks → response generator)

#### B1. Flask route entry points

- **Entry points**:
  - Student authenticated: `chat_stream` ([student.py:175-207](../student.py#L175-L207)) — delegates to `stream_chat_response` ([src/chat_streaming.py:113-380+](../src/chat_streaming.py#L113)).
  - Anonymous-slug variant in `app.py` (`/<slug>/api/chat/stream`) — uses the same `stream_chat_response` helper.
  - Professor test-chat: `test_chat_stream` ([professor.py:482-597](../professor.py#L482-L597)) — inlines the retrieval+generation logic (does NOT call `stream_chat_response`).
- **Inputs**: query string, conversation history, session_id.
- **Outputs**: SSE stream of `{type: status|sources|token|done|error}` events.
- **Cost signals**: route overhead is small; real cost is in `retrieve_context` + `generate_response_stream`.
- **Failure modes**: route-level exceptions → 500 / JSON error. Streaming errors emit `{type: error}` and end the stream.
- **Callers / dependents**: student dashboard chat UI, professor test-chat UI, anonymous slug chat page.

#### B2. ChatSession + cache load

- **Entry point**: `retrieve_context` lines [~1745-1761](../src/retriever.py#L1745-L1761). `ChatSession.query.get(session_id)` → `session.active_context` JSON.
- **Inputs**: `session_id`.
- **Outputs**: in-memory `session_context` dict OR None. Cross-tenant validation: session_context only used if `ChatSession.ta_id == ta_id`.
- **Cost signals**: 1 DB query (PK lookup).
- **Failure modes**: any exception caught; logs warning and proceeds with `session_context=None`.
- **Callers / dependents**: read by downstream cache-short-circuit decision (B10) and contextualizer (B5). Note: only `student.py` chat persists ChatSession rows; the professor test-chat creates a synthetic in-memory `session_id` ([professor.py:512](../professor.py#L512)) that has no DB row — so cache logic never fires for test-chat queries.

#### B3. Moderation pre-filter

- **Entry point**: inline at [src/retriever.py:1763-1780](../src/retriever.py#L1763-L1780). Calls `moderation_check` ([line 1586](../src/retriever.py#L1586)) which hits OpenAI Moderations API.
- **Inputs**: query string.
- **Outputs**: dict `{flagged, categories, latency_ms}`. If flagged → caller short-circuits to `draft_off_topic_redirect` ([retriever.py:1614](../src/retriever.py#L1614)) and returns `[], diagnostics`.
- **Cost signals**: 1 free Moderations API call (~50-100ms).
- **Failure modes**: API failure → caught, returns `flagged=False` (fail-open). Adversarial query might slip through.
- **Callers / dependents**: only `retrieve_context`. Independent of the LLM-based off_topic classifier (B5) — different threat surface.

#### B4. `contextualize_query` — bucketed LLM intent classifier

- **Entry point**: `contextualize_query` ([src/retriever.py:1606-1760](../src/retriever.py#L1606-L1760)). Called from `retrieve_context` line [~1787](../src/retriever.py#L1787).
- **Inputs**: query, conversation_history, session_context (cached doc info), ta_id.
- **Outputs**: dict `{rewritten_query, intent, current_focus, reason, latency_ms, fallback}`. `intent` ∈ `{continuation, concept_lookup, pivot, clarification, new, off_topic}`.
- **Cost signals**: 1 gpt-4o-mini chat-completion call (~100-200ms). Skipped if no prior context AND adversarial filter disabled.
- **Failure modes**: API failure → `fallback=True`, returns raw query unchanged. Downstream code falls back to heuristic topic-switch detection.
- **Callers / dependents**: `retrieve_context` uses `intent` to drive the cache short-circuit decision (B10) and the off_topic short-circuit (B6). **Audit finding candidate: 6-bucket taxonomy critique** — see top-of-funnel research; provider consensus is structured action emission over label classification.

#### B5. Adversarial / off_topic short-circuit

- **Entry point**: inline at [src/retriever.py:1796-1809](../src/retriever.py#L1796-L1809).
- **Inputs**: contextualizer's `intent` field.
- **Outputs**: returns `[], diagnostics` early if `intent == 'off_topic'`. Triggers `draft_off_topic_redirect` to generate a polite redirect message.
- **Cost signals**: 1 additional gpt-4o-mini call for the redirect text (~500ms-1s). Total saving: ~10-30s of downstream retrieval + generation.
- **Failure modes**: false-positive (real student incorrectly classified off_topic) → frustrated user, no retrieval. The contextualizer prompt has explicit "bias toward NOT off_topic" guidance, but it can still mis-classify.
- **Callers / dependents**: only `retrieve_context`.

#### B6. `analyze_query` — regex/filename feature extraction

- **Entry point**: `analyze_query` ([src/retriever.py:1299-~1410](../src/retriever.py#L1299)). Called from `retrieve_context` line [~1816](../src/retriever.py#L1816) with the contextualized query.
- **Inputs**: query string.
- **Outputs**: dict with `doc_type_filter`, `assignment_filter`, `unit_filter`, `year_filter`, `filename_filter` (via `find_matching_documents`), `structural_reference` (slide/page detected via regex), `problem_reference` (problem/question patterns), `requires_early_hybrid` flag, `is_conceptual`.
- **Cost signals**: pure regex + a DB query in `find_matching_documents` (read all `Document` rows for the TA, Python loop tokenizing filenames). For 75-doc corpus, ~100-300ms. Scales linearly with corpus size.
- **Failure modes**: regex misclassification (e.g. "Quiz 2025" matching 2025 as assignment_number) historically caused Type A failures. Largely mitigated by V2 dropping hard filters from chunk search.
- **Callers / dependents**: `retrieve_context` (cache topic-switch detection, V2's `hybrid_doc_search`, legacy hard-filter cascade, structural injection, identify_target_documents). **Audit finding candidate: runs parallel to `contextualize_query`** — significant feature overlap (filename, doc_type detection). The 2-LLM-then-regex stack at the top of retrieval is the redundancy the user flagged.

#### B7. Follow-up detection + query enrichment

- **Entry point**: `detect_followup_query` ([src/retriever.py:1417-~1500](../src/retriever.py#L1417)). Followed by `extract_context_from_history` + `enrich_query_with_context` ([lines 1505-1585](../src/retriever.py#L1505-L1585)).
- **Inputs**: query + conversation_history.
- **Outputs**: dict `{is_followup, followup_type, needs_context_enrichment}`. If needs enrichment AND no contextualizer rewrite was produced, the query is enriched with topic_summary + document_reference + problem_reference from recent messages.
- **Cost signals**: pure regex + Python iteration over history. Sub-millisecond.
- **Failure modes**: false-positive enrichment → query gets bloated with old context that's no longer relevant. The contextualizer's rewrite takes precedence when available, so this path is mostly dead when CONTEXTUALIZER_ENABLED=True.
- **Callers / dependents**: `retrieve_context`. **Audit finding candidate: largely subsumed by `contextualize_query`** — kept as a fallback for when the LLM call fails. Could be eliminated if we accept that fallback path's behavior degrades.

#### B8. Session-cache short-circuit decision (topic-switch)

- **Entry point**: inline at [src/retriever.py:~1876-~2050](../src/retriever.py#L1876). Compares `query_analysis` filters against cached `session_context` values; uses contextualizer's `intent` as an override (B4).
- **Inputs**: session_context (B2), query_analysis (B6), contextualizer result (B4).
- **Outputs**: `is_topic_switch` boolean. If False → reuses cached chunks + returns early. If True → proceeds with fresh retrieval.
- **Cost signals**: pure Python. No external calls.
- **Failure modes**: contextualizer classifies "pivot" but heuristic sees no filter change → contextualizer wins (cache flushed). Contextualizer classifies "continuation" but heuristic sees filter change → contextualizer wins (cache preserved). The bias is toward preserving cache. Doc-switching failures (rows 7, 25, 27 in the 2026-05-23 CSV) happen when contextualizer mis-classifies pronoun-prefixed pivots as continuations.
- **Callers / dependents**: only `retrieve_context`. **Audit finding candidate: cache + contextualizer can disagree** — this is one of the three explicit anti-patterns the audit (3.2) needs to evaluate.

#### B9. Query embedding generation

- **Entry point**: inline at [src/retriever.py:~2058-2062](../src/retriever.py#L2058-L2062). `client.embeddings.create(model=Config.EMBEDDING_MODEL, input=effective_query)`.
- **Inputs**: rewritten/enriched query.
- **Outputs**: 1536-dim vector.
- **Cost signals**: 1 OpenAI embeddings call (~100-200ms, ~$0.00002).
- **Failure modes**: API failure → exception, retrieval aborts. No retry.
- **Callers / dependents**: feeds chunk vector search (B13) and V2 `hybrid_doc_search` dense ranking (B12.V2).

#### B10. Early hybrid routing

- **Entry point**: inline at [src/retriever.py:~2067-2149](../src/retriever.py#L2067-L2149). Triggered when `query_analysis['requires_early_hybrid']=True` (specific problem reference like "section 1 question a") AND `Config.HYBRID_RETRIEVAL_ENABLED`.
- **Inputs**: query_analysis (specifically problem_reference + filters).
- **Outputs**: if `identify_target_documents` finds a doc AND its full text fits in `Config.HYBRID_MAX_DOC_TOKENS` (default 80000), returns a single "chunk" containing the full doc text with score=10.0. Short-circuits the rest of retrieval.
- **Cost signals**: 1 DB query to fetch doc + on-disk file read + text extraction (could be 5-30s if PDF vision). Caches result in `session.active_context` for follow-up queries.
- **Failure modes**: target doc not identifiable (regex misses) → falls through to chunk retrieval. Doc too large → falls through. Both paths logged.
- **Callers / dependents**: only `retrieve_context`. **Audit finding candidate: legacy regex-driven; doesn't use V2's hybrid_doc_search even when V2 flag is on** — pre-Stage-3 code path. Might be a candidate for unification.

#### B11. Vector chunk base_query construction

- **Entry point**: inline at [src/retriever.py:~2150-2160](../src/retriever.py#L2150-L2160). Builds SQLAlchemy `base_query` selecting chunk_text + denormalized fields + cosine_distance score.
- **Inputs**: query_embedding.
- **Outputs**: SQLAlchemy Query object (lazy).
- **Cost signals**: nothing until executed.
- **Failure modes**: pgvector cosine_distance failure → caught downstream.
- **Callers / dependents**: filtered by either V2 (B12.V2) or legacy cascade (B12.L) before execution.

#### B12. V2 hybrid Stage 1 (RETRIEVAL_V2_ENABLED=True) — flag-gated path

- **Entry point**: `hybrid_doc_search` ([src/retriever.py:1120-1416](../src/retriever.py#L1120-L1416)). Called from `retrieve_context` line [~2363](../src/retriever.py#L2363) with `query_analysis` passed through.
- **Inputs**: query, query_embedding, ta_id, optional query_analysis dict.
- **Outputs**: `(doc_ids, diagnostics)`. doc_ids is a single-element list when Stage 5 short-circuit fires; otherwise top-K (default 5) by RRF.
- **Cost signals**:
  - Stage 5 short-circuit (Path A: category+number DB lookup) — 1 lightweight DB query, ~5-10ms.
  - Stage 5 short-circuit (Path B: filename+number margin) — pure Python.
  - Stage 5 short-circuit (Path C: margin-only) — pure Python.
  - Filename overlap (Python loop over all Documents in TA): ~50-900ms scaling with corpus size.
  - BM25 ranking (PostgreSQL plainto_tsquery + ts_rank, top-K=20): ~1-20ms.
  - Dense ranking (per-doc mean-pooled top-5 chunks, pgvector cosine): ~10-25ms.
  - RRF fusion: pure Python, <1ms.
- **Failure modes**: any of BM25/dense/filename can fail individually; fusion runs on whatever's available. If all three fail → empty `doc_ids`, caller falls through to unfiltered chunk search.
- **Callers / dependents**: only `retrieve_context`. Output drives constrained chunk vector search (B13) and the V2-scoped structural injection (B15).

#### B12.L Legacy regex hard-filter cascade (RETRIEVAL_V2_ENABLED=False)

- **Entry point**: inline at [src/retriever.py:~2402-~2425](../src/retriever.py#L2402-L2425). Cascade of `if query_analysis['doc_type_filter'] and ...` filters.
- **Inputs**: query_analysis.
- **Outputs**: `filtered_query` (SQLAlchemy) + `has_filters` flag.
- **Cost signals**: pure SQLAlchemy filter chaining; no execution yet.
- **Failure modes**: empty result set → falls back to base_query (unfiltered). The Type A failure (PS2 query returning quiz 2) originated here because both docs had `doc_type='homework'`.
- **Callers / dependents**: only `retrieve_context`. Skipped entirely in V2 mode.

#### B13. Chunk vector search

- **Entry point**: inline at [src/retriever.py:~2389-~2402](../src/retriever.py#L2389-L2402) (V2 branch) and [~2434-~2445](../src/retriever.py#L2434-L2445) (legacy branch).
- **Inputs**: filtered_query (constrained to candidate_doc_ids in V2, or doc_type/filename filters in legacy).
- **Outputs**: `results` = top-INITIAL_RETRIEVAL_K (default 20) chunks ordered by cosine distance.
- **Cost signals**: 1 pgvector cosine_distance query with ORDER BY + LIMIT 20. ~50-150ms depending on corpus.
- **Failure modes**: pgvector exception caught; falls back to unfiltered base_query.
- **Callers / dependents**: produces `initial_chunks` consumed by structural injection (B15), paste detection (B16), and `llm_rerank` (B17).

#### B14. Initial chunks dict assembly

- **Entry point**: inline at [src/retriever.py:~2465-~2479](../src/retriever.py#L2465-L2479).
- **Inputs**: SQLAlchemy results from B13.
- **Outputs**: `initial_chunks` list of dicts `{text, score, file_name, doc_type, doc_category, metadata}`.
- **Cost signals**: pure Python.
- **Failure modes**: row.score being None or unparseable → defaults to 0.0.
- **Callers / dependents**: consumed by all downstream retrieval steps.

#### B15. Structural injection (slide/page boost)

- **Entry point**: inline at [src/retriever.py:~2480-~2540](../src/retriever.py#L2480-L2540). Fires when `query_analysis['structural_reference']` matches "slide N" or "page N".
- **Inputs**: query_analysis + initial_chunks + candidate_doc_ids (V2 mode).
- **Outputs**: chunks with `chunk_context LIKE 'Slide N:%'` or `'--- Page N ---%'` are inserted at the top of initial_chunks with score=10.0.
- **Cost signals**: 1 DB query on `DocumentChunk.chunk_context`. ~20-100ms.
- **Failure modes**: in V2 mode, scoped to candidate_doc_ids (Stage 5 fix) — if candidate_doc_ids is empty/wrong, scope is wrong. In legacy mode, scoped to `query_analysis` doc_type/unit filters when set, else unfiltered (was the source of Midterm 2022A solutions hijacking slide queries pre-Stage-5).
- **Callers / dependents**: only `retrieve_context`. Chunks get score=10.0 to survive `llm_rerank` (B17).

#### B16. Paste detection

- **Entry point**: `detect_pasted_question` ([src/retriever.py:~981](../src/retriever.py#L981)). Called at [retriever.py:~2548](../src/retriever.py#L2548).
- **Inputs**: query + initial_chunks (top-20).
- **Outputs**: if any doc in top-20 has high k-gram containment with the query → promotes that doc's best-containing chunk to top with score=9.5.
- **Cost signals**: Python k-gram computation. ~50-200ms.
- **Failure modes**: false-positive promotion → unrelated chunk gets ranked high. Mitigated by `llm_rerank` (B17) downstream.
- **Callers / dependents**: only `retrieve_context`. Affects which chunks survive reranking.

#### B17. LLM rerank

- **Entry point**: `llm_rerank` ([src/retriever.py:674-808](../src/retriever.py#L674-L808)). Called at [retriever.py:~2580](../src/retriever.py#L2580).
- **Inputs**: query + initial_chunks (up to 20). Each chunk presented to the LLM as `[N] [category: <doc_category>] <file_name>: <text_preview>`.
- **Outputs**: top-FINAL_K (default 8) chunks reordered by LLM `llm_relevance_score`.
- **Cost signals**: 1 `Config.LLM_MODEL` (gpt-5.2 with reasoning_effort=MEDIUM) chat-completion call (~5-15s). Dominant latency in retrieval.
- **Failure modes**: API failure → caught; returns initial_chunks unchanged with `reranked=False`. JSON parsing failure same. Stage 4 added doc_category to the prompt — see top-of-funnel research for the design critique.
- **Callers / dependents**: only `retrieve_context`. Output feeds confidence assessment (B18).

#### B18. Confidence assessment + hybrid_full_doc fallback

- **Entry points**: `assess_retrieval_confidence` ([src/retriever.py:257-322](../src/retriever.py#L257-L322)) + `validate_chunks_contain_reference` ([line 909](../src/retriever.py#L909)). Fallback block at [retriever.py:~2570-~2640](../src/retriever.py#L2570-L2640).
- **Inputs**: reranked chunks + rerank_info.
- **Outputs**: `confidence` dict + boolean `should_trigger_hybrid`. If triggered: `identify_target_documents` runs → full-doc fetched (text extracted from disk again) → returned as single high-score chunk.
- **Cost signals**: confidence assessment is pure Python. If triggered, fallback path adds 1 doc fetch + text extraction (5-30s if PDF vision). Stage 3 fix: passes empty `query_analysis` in V2 mode so `identify_target_documents` uses `chunk_frequency` (rerank-derived) instead of regex.
- **Failure modes**: false-trigger → retrieves a full doc when chunks would have been better. False-skip → low-quality chunks survive. Score thresholds are heuristic and not tuned to V2's scoring profile (`hybrid_rrf_3signal` produces different score distributions than legacy).
- **Callers / dependents**: only `retrieve_context`. Caches the chosen doc in `session.active_context` for follow-up queries.

#### B19. Supplementary teaching material retrieval

- **Entry point**: `retrieve_supplementary_teaching_material` ([src/retriever.py:534-672](../src/retriever.py#L534-L672)). Called at [retriever.py:~2640+](../src/retriever.py#L2640).
- **Inputs**: ta_id, primary_chunks, query_analysis, diagnostics.
- **Outputs**: 0-4 extra chunks labeled `[TEACHING MATERIAL — From: ...]`. Targets "concept reinforcement" cases where the primary chunk is a problem and the LLM needs additional lecture/reference content to help the student.
- **Cost signals**: 1 LLM concept-extraction call (`_extract_concepts_via_llm`, ~1-2s) + 1 supplementary vector search (~50ms).
- **Failure modes**: pulls noise into context — eval has shown supplementary often pulls the same source (e.g., syllabus appearing 5x with score 0.5-0.57) when the concept query is generic. Logged at INFO level so reviewable.
- **Callers / dependents**: only `retrieve_context`. Adds chunks to the final return list. **Audit finding candidate: deserves a hard look — its concept-extractor LLM call adds latency and the chunks it returns are often noise.**

#### B20. Return + session-cache update

- **Entry point**: `retrieve_context` returns at multiple points: [line 2531](../src/retriever.py#L2531) (normal), [line 2467](../src/retriever.py#L2467) (early hybrid), [line 2142](../src/retriever.py#L2142) (cache hit), [line 1780, 1809](../src/retriever.py#L1780) (moderation/adversarial short-circuit). Cache updates inline before each return.
- **Inputs**: final chunks + diagnostics.
- **Outputs**: `(chunks, diagnostics)` tuple to caller. `ChatSession.active_context` updated with document_filename + document_content + problem_reference + supplementary_content for follow-up queries.
- **Cost signals**: 1 DB UPDATE per cache update. ~10-30ms.
- **Failure modes**: cache update can fail silently — logged but doesn't abort retrieval.
- **Callers / dependents**: chat-stream Flask routes consume the chunks; `qa_logger.log_qa_entry` consumes diagnostics; downstream `generate_response_stream` consumes the chunks for prompt assembly.

---

**Notes for the auditor.** Three things to flag explicitly when 3.2 lands:

1. **The two-LLM-then-regex top-of-funnel.** B4 (`contextualize_query`) + B6 (`analyze_query`) + B7 (`detect_followup_query`) + B8 (cache topic-switch heuristic) all do overlapping work to decide "what is this query about + does the cache apply?" This is the redundancy the user flagged. The top-of-funnel research (parked Step 2) addresses this directly.
2. **Indexing-side dead-weight: `classify_doc_role` (A9).** Runs at index time, costs LLM calls, has zero readers in retrieval. Safe to eliminate.
3. **Duplicate LLM metadata extraction (A3 + A8).** The upload-time background thread runs `extract_metadata_with_llm`, then `process_and_index_documents_resumable` re-runs the same LLM call. Fields are only overwritten when non-empty, so values are preserved, but the duplicate cost is unjustified.

## 3.2 Architecture audit

**Status:** populated 2026-05-23 by research subagent. Citations + per-question verdicts inline. Step IDs reference 3.1 (A1-A14 indexing, B1-B20 retrieval).

**Framing reminder.** The user's "shitty old car" critique is the audit's null hypothesis: we may have been bolting fancy parts (V2 hybrid_doc_search, Stage 5 short-circuit, doc_category, contextualizer) onto an architecture that is no longer the right shape for production conversational RAG in 2026. This audit answers at the architectural level — "should this component exist at this stage at all" — not the component level.

### 3.2 Indexing track

#### Q1. Modern chunking strategies

**What the literature says.**

The 2024-2026 chunking landscape has split decisively in two directions, neither of which is "smarter fixed-size chunks." Both directions have strong empirical support; they apply to different document classes.

1. **Anthropic Contextual Retrieval** (Sept 2024, [anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval)): prepend a 1-2 sentence LLM-generated chunk-specific context blob to each chunk *before embedding*. Reduces retrieval failures 49% on its own, 67% combined with reranking. Cost: ~$1.02 per million document tokens at ingest with prompt caching; **zero query-time overhead**. Reproduced widely ([AWS / Bedrock](https://aws.amazon.com/blogs/machine-learning/contextual-retrieval-in-anthropic-using-amazon-bedrock-knowledge-bases/), [Unstructured](https://unstructured.io/blog/contextual-chunking-in-unstructured-platform-boost-your-rag-retrieval-accuracy), [DataCamp](https://www.datacamp.com/tutorial/contextual-retrieval-anthropic)).

2. **RAPTOR hierarchical chunking** ([arXiv 2401.18059](https://arxiv.org/abs/2401.18059), ICLR 2024): recursively cluster + summarise chunks into a tree, retrieve at multiple abstraction levels. Strongest empirical wins on textbooks and multi-hop questions. The 2025 follow-ups ([Frontiers](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1710121/full), [TreeRAG ACL 2025](https://aclanthology.org/2025.findings-acl.20.pdf)) explicitly fix RAPTOR's biggest weakness — its fixed-token leaf chunks — by combining with semantic chunking. **The dominant educational-content win.**

3. **SF-RAG (Structure-Fidelity RAG)** ([arXiv 2602.13647](https://arxiv.org/pdf/2602.13647)): preserves native outline as a path-indexed structure; retrieval is path-guided. Section path is the primary key, embeddings are secondary. Closest in spirit to what Maize's `chunk_context` is gesturing at — but Maize stores only the most recent single header, while SF-RAG stores the full path.

4. **Late chunking** (Jina 2024, [jina.ai/news/late-chunking-in-long-context-embedding-models](https://jina.ai/news/late-chunking-in-long-context-embedding-models/), [Weaviate](https://weaviate.io/blog/late-chunking)): embed the whole document with a long-context encoder first, then pool token embeddings into chunk vectors. Each chunk vector sees the full document context. Best for narrative docs (financial filings, reports); less useful for slides where pages are already discrete.

5. **LlamaIndex SemanticSplitter / SemanticDoubleMerging** ([docs](https://developers.llamaindex.ai/python/examples/node_parsers/semantic_double_merging_chunking/)): adaptive breakpoints by embedding similarity. The **2025 Max-Min benchmark study** ([Springer](https://link.springer.com/article/10.1007/s10791-025-09638-7)) found Semantic Splitter under-performed fixed-size baselines on benchmark corpora. Mixed verdict; not the default choice for production.

6. **Layout-aware chunking** (Reducto, LlamaParse `extract_layout`, Unstructured) ([Reducto](https://llms.reducto.ai/chunking-api), [Firecrawl 2026](https://www.firecrawl.dev/blog/best-pdf-parsers)): a vision-first parser produces bounding-boxed layout regions (table, figure, title, list, text) and chunks are aligned to layout boundaries. Reducto reports ~20% higher parsing accuracy on real-world docs. Strongest fit for structured but visually-complex docs (research papers, financial reports).

7. **Small-to-big / parent-child / `HierarchicalNodeParser`** ([LlamaIndex](https://medium.com/data-science/advanced-rag-01-small-to-big-retrieval-172181b396d4)): embed small chunks for retrieval precision, return larger parent chunks to the LLM for synthesis context. Compatible with all of the above.

**Per-Maize-step verdict (step A12 — `chunk_text_with_context`).**

- **Refactor.** A12 is doing a single-layer version of #3 (SF-RAG section path) but storing only `section_header` not `section_path`. The residual-failures research already flagged this for Type B. **Replace `chunk_context: str` with `section_path: list[str]` and a doc-relative position field**; this generalises the slide/page handling rather than replacing it.
- **Add Anthropic Contextual Retrieval (#1) to the ingest pipeline.** This is the single highest-leverage indexing change in this audit. 49% retrieval-failure reduction at one-time ingest cost is the best precision/$ ratio in print, and the cost (~$1 per million doc tokens with prompt caching) is negligible for the 75-doc ECON S1117 corpus (~$0.20). **Crucially**, the contextual prefix is what reranker-as-text wants — it converts our current "chunks lose their identity once vectorised" failure mode into "every chunk knows what doc-section it came from."
- **RAPTOR (#2) deferred.** RAPTOR helps multi-hop QA, but Maize's failure modes are predominantly single-doc lookup ("find the problem-set 2 solutions") and within-doc structural navigation ("Section II part b"). The single-doc bias means RAPTOR's cross-doc tree summaries solve a problem we don't have. **Defer to v2** unless cross-doc synthesis becomes a primary failure class.
- **Layout-aware chunking (#6) — consider for PDF only.** Maize's PDF pipeline already does layout-ish work via `pdfplumber` + `[FIGURE: ...]` splice but doesn't get the boundaries it needs to align chunks with layout. **Defer unless** we see a class of PDF parsing failures the audit doesn't already attribute to other causes.
- **Late chunking (#4) and SemanticSplitter (#5) — reject for v1.** Slides and problem sets are page/problem-discrete; late chunking's win comes from continuous narrative.
- **Small-to-big (#7) — already partially present.** B18's `hybrid_full_doc` fallback is a one-way version of this (small → biggest). A cleaner implementation would keep parent-child node references at ingest so retrieval can step up granularity without re-extracting from disk.

**Could reduce retrieval-side complexity?** Yes, strongly:
- Section-path metadata at ingest **eliminates B15 structural injection** as a distinct retrieval step — it becomes a metadata filter on the same chunk search.
- Contextual Retrieval prefixes **make B17 (`llm_rerank`) more accurate** by giving each chunk self-describing context, which reduces the need for B19 (supplementary teaching material).
- Parent-child references **eliminate the on-disk re-extraction in B18** (currently 5-30s of PDF vision in the worst case).

---

#### Q2. Multi-modal indexing

**What the literature says.**

The 2024-2026 multi-modal indexing field has split into two distinct schools, with the choice driven by document type and budget.

1. **Vision-only "page as image" retrieval — ColPali / ColQwen / ColSmol** ([arXiv 2407.01449](https://arxiv.org/abs/2407.01449), [illuin-tech/colpali](https://github.com/illuin-tech/colpali), [Together AI](https://www.together.ai/blog/multimodal-document-rag-with-llama-3-2-vision-and-colqwen2)). Treats each page as an image, patchifies (e.g. 32×32 = 1024 patches), generates ColBERT-style multi-vector embeddings per patch. Strongest on slide decks, posters, charts, hand-drawn diagrams — exactly the educational use case. **No text extraction needed at all.** Production platforms ([Mixpeek](https://medium.com/@intuitivedl/rag-with-colpali-everything-you-need-to-know-46b7bd50901b)) wrap this with the operational concerns (storage, A/B, explainability).

2. **Hybrid text-extraction + vision supplementation** — what Maize currently does (A7). Three-tier PDF (pdfplumber → pypdf2 → GPT-4o vision), `_supplement_pdf_with_figures` per-page vision pass, PPTX per-picture vision. This is the **conservative, lower-storage** path. Reducto's vision-first multi-pass pipeline ([llms.reducto.ai](https://llms.reducto.ai/document-parser-comparison)) is the productionised version: layout-aware models break the doc into regions first, then per-region extractors specialise.

3. **Multi-granularity multi-modal RAG (MHier-RAG, MMRAG-DocQA, MG-RAG)** ([arXiv 2508.00579](https://arxiv.org/abs/2508.00579), [arXiv 2510.15253](https://arxiv.org/html/2510.15253v1)): combine page-level vision embeddings with text chunks via hierarchical indexing. The 2025-2026 academic consensus for visually-rich documents.

**Practitioner consensus.** For **educational slides specifically**, the ColPali family is clearly winning the conversation — multiple production write-ups in 2025-2026 ([DEV Community](https://dev.to/aws/beyond-text-building-intelligent-document-agents-with-vision-language-models-and-colpali-and-oc), [decoding-ai](https://www.decodingai.com/p/the-king-of-multi-modal-rag-colpali)) treat ColPali/ColQwen as the default. For **mixed corpora** (Maize's reality: PDFs + slides + Word + notebooks + Excel), the practical answer is still hybrid: text-first where text exists cleanly, vision-supplement where it doesn't.

**Per-Maize-step verdict (step A7).**

- **Keep the hybrid approach as the default.** Maize's three-tier PDF + per-page figure supplement is well-architected and matches Reducto's published pattern (vision + OCR + VLM). The recent fixes (POPPLER_PATH env-driven, group-shape recursion in PPTX) show the path is maturing rather than misaligned.
- **Consider adding ColPali/ColQwen as a parallel indexing track for slide-heavy TAs.** The Maize use case where this most clearly wins is the ECON S1117 lecture decks — slides with formulas, diagrams, and minimal text, where today's PPTX path drops a lot of semantic content per slide on the floor. A per-doc-type flag (`use_vision_index: bool`) on `Document` lets us A/B without ripping out the text path.
- **Refactor `_supplement_pdf_with_figures`.** Today it runs unconditionally per page after text extraction succeeds — that's expensive (~5-15s per doc) and produces `[FIGURE: ...]` blocks that often duplicate text already extracted. Gate it on a heuristic: "skip if pdfplumber returned >N words per page AND no embedded image annotations detected."
- **The unsolved-problem class: math notation.** Formulas in PDFs are an active research gap. ColPali handles formula visual matching reasonably, but neither ColPali nor Reducto cleanly extract LaTeX from rendered formulas. Maize's KaTeX rendering on the response side means we want formula text, not formula images, in retrieval — but no production system reliably does this. **First-principles call: live with the gap and rely on figure vision blobs to give the reranker enough signal.**

**Could reduce retrieval-side complexity?** Indirectly — better extraction at ingest means fewer "the chunk doesn't say what the doc actually says" failures downstream, which reduces B18 fallback firing.

---

#### Q3. Metadata enrichment at ingest

**What the literature says.**

The 2025 production consensus is that ingest-time enrichment is the cheapest retrieval-quality dollar. The empirical results are unusually consistent across providers and benchmarks:

- **Anthropic Contextual Retrieval** ([anthropic.com](https://www.anthropic.com/news/contextual-retrieval)): 49% failure reduction from prepending a chunk-specific context blob.
- **MetaRAG / Systematic Framework for Enterprise Knowledge Retrieval** ([arXiv 2512.05411](https://arxiv.org/pdf/2512.05411), already cited in doc-classification research): 12% precision gain (0.825 vs 0.733) from metadata fused into chunk text.
- **Two-tier indexing (document-summary index + chunk index)** ([Ragie](https://www.ragie.ai/blog/advanced-rag-with-document-summarization)): document-level summary embeddings (often 3072-dim) live alongside chunk embeddings; retrieval first picks the doc by summary, then chunks within doc. Maize's V2 `hybrid_doc_search` (B12.V2) is doing a version of this with BM25 + dense + filename instead of summary embeddings — same shape, different signals.
- **Hypothetical Questions / Reverse-QA** ([PIXION](https://pixion.co/blog/rag-strategies-hypothetical-questions-hyde)): at ingest, an LLM generates 3-5 questions that each chunk could answer; embed the questions and store as chunk metadata. Query → question-embedding match is more reliable than query → chunk-text match because both are short. **Pushes HyDE's per-query cost into one-time ingest cost.**
- **Section-path metadata** ([Towards AI](https://towardsai.net/p/machine-learning/production-rag-the-chunking-retrieval-and-evaluation-strategies-that-actually-work), already cited in residual-failures research): `full_path: "Chapter 3 > Section 3.2 > (b)"` becomes a routable filter.

**What Maize precomputes today (A2, A3, A8, A10, A11, A12, A13, A14):** filename heuristics, LLM doc-level metadata (twice — A3 and A8), `doc_category` (Stage 2B), BM25 tsvector, single section header per chunk, embedding. Notably absent: per-doc summary, section path, hypothetical questions, contextual prefix.

**Per-Maize verdict (across A2-A14).**

- **Add: contextual prefix per chunk (Anthropic-style).** Already argued in Q1. Single highest-leverage indexing addition.
- **Add: `section_path: list[str]` on every chunk.** Already argued in Q1 and the residual-failures research. Costs one extra column + ~10 lines in the chunker.
- **Add: per-doc summary stored alongside the chunk index.** Stored on `Document.summary` (text) and `Document.summary_embedding` (vector). At retrieval, the doc-routing layer (today's B12.V2) can use summary cosine as a 4th signal in the RRF — and the reranker can be told "this chunk is from a doc whose summary is: ..." to disambiguate when two chunks score similarly. ~1 LLM call per doc at ingest, ~$0.001 per doc.
- **Defer: hypothetical-questions reverse-index.** Real research evidence of the lift, but the cost is 3-5 extra LLM calls per chunk. For a 75-doc corpus that's manageable; for 1000+ chunks per TA at scale, it's the largest single ingest cost. **Revisit when we have direct evidence chunk-text → query matching is the dominant failure mode.** The contextual-prefix change should be tested first.
- **Eliminate: duplicate LLM metadata call (A3 + A8).** Already flagged in 3.1. The upload-time background thread (A3) and the indexing pipeline (A8) both run `extract_metadata_with_llm` on the same content. Pick one — keep A8 (in the indexing pipeline) since that's where downstream consumers live; delete A3, replace with a "metadata pending — will populate on next index" state in the UI.
- **Eliminate: `classify_doc_role` (A9).** Dead-weight. Already flagged in 3.1. Zero readers in retrieval. ~$0.04 wasted per 75-doc index.
- **Keep: BM25 tsvector (A11), embedding (A13), `doc_category` (A10).** All have live readers and are well-architected.

**Could reduce retrieval-side complexity?** Yes — this is where the cross-cutting bias bites hardest:
- Section-path → **eliminates B15 structural injection as a distinct step**.
- Contextual prefix → **reduces or eliminates B19 supplementary teaching material retrieval** (chunks self-describe; reranker doesn't need extra cited-context).
- Per-doc summaries → **simplifies B12.V2's 3-signal RRF to a 2-signal blend** (summary-cosine + filename-overlap), eliminating BM25 as a primary doc-routing signal at the cost of a small recall hit that the reranker mops up.

---

#### Q4. Indexing as observability surface

**What the literature says.**

The 2025-2026 RAG observability ecosystem has standardised on chunk-level span tracing. The reference shape:

- **Span-level retrieval traces** with chunk-level scores attached — Galileo, Langfuse, Phoenix, Maxim, Braintrust all converge on this ([FutureAGI 2026](https://futureagi.com/blog/what-is-rag-observability-2026), [Maxim](https://www.getmaxim.ai/articles/top-5-rag-observability-platforms-in-2025/), [Braintrust](https://www.braintrust.dev/articles/best-rag-evaluation-tools)).
- **RAG-specific eval metrics at span**: Context Adherence, Chunk Attribution, Completeness, Context Relevance, Faithfulness ([Maxim guide](https://www.getmaxim.ai/articles/articles/rag-evaluation-a-complete-guide-for-2025/)).
- **CI/CD quality gates on golden datasets** ([Dextralabs](https://dextralabs.com/blog/production-rag-in-2025-evaluation-cicd-observability/)): chunks-as-data tests fail PRs if precision drops on the golden set.
- **Operator-facing chunk inspectors**: ability to (a) browse "which chunks ended up in which doc with what context tags," (b) re-classify a doc by hand and re-index just that doc, (c) flag a chunk as "wrong" and have it excluded.

**What Maize has today (A4 + the admin UI + qa_logger).**
- `IndexingJob` table tracks status / progress / chunks-created.
- TA-level `indexing_warnings` summarises per-doc failures.
- Admin UI shows "metadata loaded" badge per doc.
- `qa_logger` writes one row per QA event with retrieval diagnostics.
- **No chunk-level inspection UI.** No way for an operator to ask "what chunks does this doc have, in what order, with what context tags."
- **No golden-set eval harness.** The 27-query manual battery is run by hand.

**Per-Maize verdict.**

- **Add: chunk-level admin inspector.** Single most-leveraged observability change. Per-doc page in the admin UI shows the chunk list: `chunk_index`, `chunk_context`, `doc_category`, first 200 chars of `chunk_text`, embedding-presence indicator. Lets operators verify chunking quality without dropping into the DB.
- **Add: a small CI golden set per TA.** ~20 queries per TA with expected `doc_id`s in the top-5. Run on every merge to `main`. Reuse the existing `qa_logger` schema for storage. This is the single biggest unblocker for the "we have only negative evidence" problem the 3.0 context flagged.
- **Add: per-doc re-index button.** Today re-index is all-or-nothing on a TA. A doc-level button means an operator can fix one bad ingest without re-processing the whole 75-doc corpus.
- **Refactor: `indexing_warnings` is a string blob.** Should be structured: `{doc_id, failure_reason, retry_count, last_attempt_at}`. Then the UI can surface "3 docs failed extraction" with clickable detail.
- **Keep: `IndexingJob` orchestration, `qa_logger`.** Both work. The qa_logger's `is_preview` column added for professor test-chat is good architecture.

**Could reduce retrieval-side complexity?** Indirectly — but in the highest-leverage way. Today, retrieval bugs surface only at the query layer (the 27-query battery). A chunk inspector + golden-set CI surfaces ingest-side bugs at ingest, which is where the cheapest fix lives. The "shitty old car" critique is partly enabled by lack of observability — we can't tell whether V2 short-circuits fired correctly without combing logs, so the team papers over visible failures rather than fixing root causes.

---

### 3.2 Retrieval track

#### Q5. Reference architectures for production conversational RAG

**Canonical step-graph in 2026.**

The cleanest published expression of the modern shape is the **agentic RAG** pattern that Anthropic ([Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)), OpenAI ([orchestrating_agents cookbook](https://developers.openai.com/cookbook/examples/orchestrating_agents)), Google ([Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling)), and the 2026 productionised stacks ([LangGraph](https://medium.com/@vinodkrane/next-generation-agentic-rag-with-langgraph-2026-edition-d1c4c068d2b8), [SoK Agentic RAG survey arXiv 2603.07379](https://arxiv.org/abs/2603.07379)) all converge on. The canonical step-graph:

1. **Receive turn** (raw user query + conversation history + session state).
2. **Single LLM tool-call decision**: emits a structured action (`retrieve`, `use_cache`, `answer_directly`, `clarify`, `redirect_off_topic`) *and* the parameters needed to execute it (rewritten query, filter hints, target tool). This is the "tool call as routing" pattern; the action is *the dispatch*, not a label downstream code branches on.
3. **Tool execution** — if `retrieve`, the retrieval tool runs:
   - **Hybrid candidate generation** (BM25 + dense + optional metadata filters).
   - **Reranking** (cross-encoder or LLM rerank, typically top-20 → top-8).
   - **Optional small-to-big expansion** (return parent context for retrieved children).
4. **Self-critique / refinement loop** (optional, agentic): if confidence is low, the agent emits a refined query / different tool call and loops back.
5. **Generation** with grounded context.
6. **Session-state update** (what was retrieved, what was answered, any user preferences extracted).

**Variants by maturity.**

- **Pipeline RAG** (most legacy systems including current Maize): fixed sequence, no agentic loop. Cheaper, more predictable, but blind to its own failures.
- **Agentic RAG** (canonical 2026): decision-making agent at every step, loops on low confidence. ~3-10× token cost, justified for multi-hop / high-stakes ([MarsDevs 2026 guide](https://www.marsdevs.com/guides/agentic-rag-2026-guide)).
- **Hybrid** (the practical sweet spot most production teams ship): single agentic decision at the top of the funnel (Q5 step 2 above), pipeline downstream, possibly one critique loop on confidence-low results.

**Educational-RAG specifics.** No major ed-tech RAG publishes its architecture in detail. **Khanmigo** ([learningmate](https://learningmate.com/ep-11-building-khanmigo-an-ai-powered-personal-tutor-and-teaching-assistant/), [skywork.ai](https://skywork.ai/skypage/en/Khanmigo-Deep-Dive:-How-Khan-Academy's-AI-is-Shaping-the-Future-of-Education/1972857707881885696)) is reported to use "dual RAG + knowledge graph" with the curriculum ingested ahead of time, but no step-graph is public. The pedagogical layer (Socratic method) sits on the *generation* side, not retrieval. **CS50.ai** is hand-tuned for one course. No published reference architecture for the multi-tenant ed-RAG case.

**Maize's deviations from canonical.**

- **Maize has 4 top-of-funnel stages (B4, B6, B7, B8) where canonical has 1.** Already flagged in the 3.1 notes and top-of-funnel research. This is the biggest single deviation.
- **Maize's "intent" is a label, not an action.** Six labels (B4: `continuation`, `concept_lookup`, `pivot`, `clarification`, `new`, `off_topic`) → if/else dispatch. Canonical: tool call where action *is* the routing.
- **Maize has no self-critique loop.** B18 confidence assessment is the closest analog but it's one-shot — there's no "if confidence is still low after fallback, re-route." For the high-stakes failure modes in the residual-failures research, this is a real gap.
- **Maize's retrieval is essentially pipeline-RAG with agentic top-of-funnel half-built.** The contextualizer (B4) is the agentic seed; everything else after B4 is fixed pipeline. This is the "hybrid" sweet spot most teams ship — Maize is just on the more-pipeline side of that.
- **Maize has no doc-summary index.** Canonical 2026 (two-tier indexing) has one; B12.V2 is doing approximately the same job with BM25 + dense + filename-overlap, which is a defensible substitute but is more moving parts than necessary.

---

#### Q6. Per-step comparison

| Step | Maize today | Canonical 2026 | Verdict |
|---|---|---|---|
| B1 — Flask route entry | 3 entry points (auth student, anonymous slug, prof test-chat) | Single entry, same downstream | **Refactor** — collapse to one helper (test-chat already breaks this; flagged in 3.1) |
| B2 — ChatSession + cache load | DB PK lookup; cache used only in B8 + B10 | Session state passed *into* the routing LLM call as context | **Refactor** — make session-state a first-class input to the routing tool call, not a parallel heuristic |
| B3 — Moderation pre-filter | OpenAI Moderations API call | Same in most production systems (OpenAI / Anthropic both publish this pattern) | **Keep** |
| B4 — `contextualize_query` | gpt-4o-mini, 6-label intent + rewrite | Single routing tool call emitting action + params | **Refactor** — top-of-funnel research's Option B |
| B5 — Adversarial / off_topic short-circuit | Branches on B4's `intent='off_topic'` | Becomes the `redirect_off_topic` action emitted by the routing tool call | **Refactor** — folds into B4's redesign, not a separate step |
| B6 — `analyze_query` regex | Regex + DB scan, ~100-300ms | Subsumed by routing LLM call's `filter_hints` field | **Eliminate** — the routing tool call should emit filter hints |
| B7 — `detect_followup_query` | Regex + enrichment, mostly dead | Subsumed by routing LLM call (rewrite already handles this) | **Eliminate** — flagged in 3.1 as already mostly subsumed |
| B8 — Topic-switch heuristic | Compares analysis to cached context | Subsumed by routing tool call emitting `cache_action: preserve | invalidate` | **Eliminate** — see anti-pattern B below |
| B9 — Query embedding | OpenAI embeddings, ~100-200ms | Same | **Keep** |
| B10 — Early hybrid routing | Triggered by regex `requires_early_hybrid` | Redundant with B12.V2 Stage 5 short-circuit | **Eliminate / merge into B12** — flagged in 3.1 as legacy regex-driven |
| B11 — Vector base_query construction | SQLAlchemy lazy query | Same | **Keep** |
| B12.V2 — `hybrid_doc_search` | RRF(BM25 + dense + filename), Stage 5 short-circuit | Doc-routing layer; canonical uses doc-summary cosine as primary signal | **Refactor** — see anti-pattern C below |
| B12.L — Legacy hard-filter cascade | Old code path | Not present in modern designs | **Eliminate** — already deprecated; remove the flag |
| B13 — Chunk vector search | pgvector top-20 | Same | **Keep** |
| B14 — Initial chunks dict assembly | Python row → dict | Implementation detail | **Keep** |
| B15 — Structural injection | DB query + score=10.0 promotion | Replaced by section-path metadata filter at the chunk-vector stage | **Eliminate** if Q1/Q3 section_path lands; **keep until then** |
| B16 — Paste detection | k-gram containment, ~50-200ms | Genuinely useful; no canonical equivalent (Maize-specific) | **Keep** — well-architected, narrow purpose |
| B17 — LLM rerank | gpt-5.2 reasoning_effort=MEDIUM, top-20 → top-8 | Canonical 2026 uses cross-encoders (e.g. Cohere, Contextual AI) OR LLM rerank with smaller model; both common | **Refactor model choice** — gpt-5.2 reasoning at MEDIUM is heavy for reranking (~5-15s dominates retrieval latency). Drop to gpt-4o-mini or move to Cohere v3 reranker; reserve gpt-5.2 for generation |
| B18 — Confidence assessment + hybrid_full_doc fallback | Heuristic thresholds tuned to legacy scores | Canonical: agentic self-critique loop OR small-to-big parent-child expansion | **Refactor** — the agentic loop is over-budget; small-to-big parent-child (Q1 #7) is the right shape and replaces both the assessment heuristic and the on-disk re-extraction |
| B19 — Supplementary teaching material | Concept-extraction LLM call + extra vector search | No canonical equivalent; literature warns against retrieval-padding ([arXiv 2401.14887 "Power of Noise"](https://arxiv.org/pdf/2401.14887)) | **Eliminate** — see Q8 below |
| B20 — Return + cache update | DB UPDATE on `active_context` | Canonical: session state updated as part of the agentic loop's output | **Refactor** — once B2 becomes first-class input to routing, cache update becomes part of the routing tool's output |

**Net.** Of B1-B20, ~5 steps eliminate cleanly, ~6 refactor, ~7 keep. The architecture isn't catastrophically wrong, but it has substantially more components than the canonical shape.

---

#### Q7. Anti-patterns A, B, C

**Anti-pattern A — Multiple LLM calls for what could be one (B4 + B6 + B7 + B8).**

The top-of-funnel research already made the literature case at length (CHIQ, Alhena, Anthropic/OpenAI/Google primary docs). The architecture-level synthesis: **this is the canonical anti-pattern of "label classification + dispatch + parallel regex feature extraction + parallel heuristic state comparison" that the 2024-2026 shift to tool-call routing was designed to eliminate.** Anthropic's "Building Effective Agents" advice is "start simple, optimize single LLM calls with retrieval and in-context examples" before adding orchestration; Maize has added orchestration *without* first collapsing the redundant stages.

The hidden cost of the 4-stage stack is **error compounding**. Each of B4, B6, B7, B8 has its own failure surface; a query goes wrong if any one mis-fires. With a single routing tool call, there's one failure surface, the model can self-correct mid-emission (reasoning + action + params in one structured output), and the action *is* the dispatch — no label/dispatch mismatch.

**Maize-specific recommendation.** Adopt the top-of-funnel research's **Option B** (single `route_student_turn` tool call). The parked research is now upgraded from "parked, may revive" to "recommended, gated on a small offline eval on the existing 27-query battery." This is the most important single retrieval-side change in this audit.

**Anti-pattern B — Cache layers that don't expose themselves to the routing decision (B2 vs B8).**

The pattern: `session.active_context` is loaded in B2, then consulted in B8 *via comparison against `query_analysis` heuristic filters*. The contextualizer's intent verdict (B4) can override the heuristic. Net effect: two systems vote on "is this a topic switch" with different inputs (B8 looks at filter changes; B4 looks at the rewritten query and history), and they can disagree.

The 2025 conversational-RAG literature ([RAGFlow 2025 year-end review](https://ragflow.io/blog/rag-review-2025-from-rag-to-context), [chatnexus](https://articles.chatnexus.io/knowledge-base/conversational-rag-maintaining-context-across-mult/)) is unanimous: **session state is an input to the routing decision, not a parallel signal that competes with it.** The Alhena production write-up explicitly treats cached state as a feature the routing call sees, not a separate decision layer.

**Maize-specific recommendation.** When Option B above lands, the session state becomes part of the routing tool's input prompt: "Here's the cached context: {...}. Decide action + cache_action." The routing tool emits `cache_action: preserve | invalidate` as part of its single output. **B8 disappears.** This is mechanically a 50-line change to `retrieve_context` once Option B is in.

**Anti-pattern C — Hybrid Stage-1 retrieval at the wrong layer (B12.V2).**

The framing from the user: should `hybrid_doc_search` exist at this stage of the pipeline at all? The top-of-funnel research's verdict ("LLM should be making routing decisions instead of RRF") implies it shouldn't — at least not as the *primary* doc-routing signal.

The literature: RRF is a fusion technique for *similar-style* signals (sparse + dense rankers over the same corpus). Using it to fuse three semantically-different signals (filename overlap, BM25, dense) is **stretching RRF outside its design** ([apxml.com](https://apxml.com/courses/advanced-vector-search-llms/chapter-3-hybrid-search-approaches/rrf-fusion-algorithms), [emergentmind RRF overview](https://www.emergentmind.com/topics/reciprocal-rank-fusion-rrf)): "RRF is agnostic to the confidence of the retriever, which could otherwise provide useful signals for weighting documents." When one signal is far more confident than the others (the Stage 5 short-circuit case), RRF dilutes its information. The Stage 5 short-circuit was an admission that RRF doesn't handle this case — so the fix was bolted on outside RRF.

**The canonical 2026 shape for Maize's doc-routing problem.** The doc-routing decision ("which docs are relevant to this query before we look at chunks") is a *classification* problem, not a *ranking* problem. The 2026 stack would solve it with:
- A doc-summary index (per Q3) where the query embeds against summary vectors and the top-K docs are picked by cosine.
- A small LLM tool call ("Given the query and these doc summaries, which 1-3 docs do we need? Respond with doc_ids.") for the hard cases — gated behind a confidence threshold on the cosine ranking.
- Filename + numeric direct-match remains as a **soft boost** (Elasticsearch `constant_score` recipe per residual-failures research), not a primary signal.

**Maize-specific recommendation.** This is a Phase-B-sized change, not a Phase-A patch. Path forward:
1. Land Q3's per-doc summary embeddings at ingest first.
2. A/B `hybrid_doc_search` (RRF) vs. summary-cosine + LLM-tiebreaker on the existing 27-query battery.
3. If summary-cosine wins, delete `hybrid_doc_search` entirely; B12.V2 becomes "summary-cosine top-K → optional LLM tiebreaker."
4. Stage 5 short-circuit folds into the LLM tiebreaker's tool call (the LLM emits "doc_id=X is a clear winner; don't fuse" as a structured action).

**This is the largest single retrieval-side simplification in the audit** — properly executed, it eliminates ~300 lines of fusion + short-circuit code.

---

#### Q8. Historical residue

**Already flagged in 3.1.**

| Component | Reason | Verdict |
|---|---|---|
| `classify_doc_role` (A9) | Dead-weight; no readers in retrieval | **Eliminate.** Single-PR change; saves ~$0.04 per 75-doc index + 10-15 min of indexing time |
| Duplicate LLM metadata extraction (A3 + A8) | A3 runs at upload, A8 re-runs at index on the same content | **Eliminate A3.** Replace with "metadata pending" UI state; A8 is the canonical home |
| Legacy hard-filter cascade (B12.L, `RETRIEVAL_V2_ENABLED=False`) | Pre-V2 code path; V2 is shipped and dominant | **Eliminate the flag.** Make V2 the only path; delete B12.L |
| `process_and_index_documents` (non-resumable variant, [src/document_processor.py:1334](../src/document_processor.py#L1334)) | Identified as dead code in 3.1 (no live callers) | **Eliminate.** ~200 lines deletable |

**Additional candidates surfaced by the per-step comparison.**

| Component | Reason | Verdict |
|---|---|---|
| `analyze_query` regex (B6) | Subsumed by the routing tool call's `filter_hints` field once Option B lands | **Eliminate** (gated on Option B) |
| `detect_followup_query` + `enrich_query_with_context` (B7) | Already mostly dead when CONTEXTUALIZER_ENABLED=True; rewrite handles this | **Eliminate** (gated on Option B) |
| B8 topic-switch heuristic | Conflicts with B4's intent verdict (anti-pattern B) | **Eliminate** (gated on Option B) |
| B10 early hybrid routing | Pre-Stage-3 regex path that V2 short-circuit subsumes | **Eliminate now** — unify with B12.V2 |
| B19 supplementary teaching material | Concept-extraction LLM call + extra vector search; literature warns against retrieval-padding noise; 3.1 author already flagged "deserves a hard look" | **Eliminate.** The chunks it returns are demonstrably noise (syllabus appearing 5× with score 0.5-0.57 in eval logs per 3.1). The latency it adds (~1-2s LLM + ~50ms search) is unjustified. Replace its job (pedagogical concept reinforcement) with prompt-level guidance to the generator: "If the chunk is a problem statement, identify the relevant concept and explain it inline." That's a generation-side fix, out of audit scope but the right home. |
| B18 confidence-assessment thresholds | Heuristic thresholds tuned to pre-V2 score profiles per 3.1 | **Refactor** — recalibrate to V2's score profile; medium-term, replace with parent-child small-to-big expansion (Q1 #7) which sidesteps the calibration problem |
| B17 `llm_rerank` model choice (gpt-5.2 reasoning MEDIUM) | Over-spec'd for reranking; ~5-15s dominates retrieval latency | **Refactor** — drop to gpt-4o-mini or cross-encoder. Reserve gpt-5.2 for generation. The Iternal.ai 2026 rule of thumb: "rerank, route, classify — none of these need a premium model." |

---

### 3.2 Cross-cutting findings — highest-leverage indexing changes

Per the user's framing, the highest-leverage moves are ingest-side changes that delete retrieval-side complexity. Ranked:

**1. Section-path metadata on every chunk (Q1, Q3).** Replaces `chunk_context: str` with `section_path: list[str]`. **Eliminates B15 structural injection entirely** — slide/page becomes a metadata filter at the chunk-vector stage. Also addresses Type B failures from residual-failures research. Cost: one column + ~10 lines in `chunk_text_with_context` + a backfill migration. **Estimated retrieval-code deletion: ~80 lines.**

**2. Anthropic Contextual Retrieval prefix per chunk (Q1, Q3).** 49% retrieval-failure reduction at ~$0.20 ingest cost for the ECON S1117 corpus. **Reduces B17 reranker's job** (chunks self-describe) and **eliminates B19 supplementary teaching material's reason for existing** (the reranker no longer needs extra context to understand a chunk's purpose). Cost: ~1 LLM call per chunk at ingest, prompt-cached. **Estimated retrieval-code deletion if B19 goes: ~140 lines.**

**3. Per-doc summary + summary embedding at ingest (Q3, anti-pattern C).** Unlocks the Phase-B refactor of B12.V2: doc-routing becomes summary-cosine + small LLM tiebreaker instead of 3-signal RRF + Stage 5 short-circuit. **Largest single retrieval simplification available**: eliminates ~300 lines of fusion + short-circuit code. Gated on landing the summary index first.

**4. Eliminate `classify_doc_role` (A9) + duplicate metadata call (A3).** Saves ~$0.05 and ~15 min per re-index, plus deletes ~100 lines. Pure simplification, no retrieval impact.

**5. Chunk-level admin inspector + per-TA golden set (Q4).** Doesn't delete code, but turns the "shitty old car" problem from an opinion into a measurement. Without (5), we can't tell whether any of (1)-(4) actually improved things in production. **This is the prerequisite to everything else** — without it we're back to the reactive cycle the user flagged.

---

### 3.2 What the literature doesn't tell us

Gaps where 3.3 will need first-principles calls:

- **The right cardinality for `section_path`** on Maize's heterogeneous corpus. RAPTOR / SF-RAG assume textbooks with clean Chapter > Section > Subsection hierarchies. Maize has mixed slide decks (slide N), problem sets (Section II Part b), syllabi (week N), and Word docs with arbitrary heading levels. We may end up with paths of length 1-5 depending on doc type. No literature guides this; a quick survey of the existing ECON S1117 corpus structure should drive it.
- **Whether contextual prefix should use gpt-4o-mini or gpt-5.2 at ingest.** Anthropic's published numbers used Claude Haiku-class models. The 49% number is robust to model choice within reason, but the absolute quality of the prefix matters for downstream reranking. Run both on a 20-doc sample, eyeball, decide.
- **Whether to keep the LLM reranker (B17) or move to a cross-encoder.** The literature has converged on cross-encoders for production reranking ([apxml hybrid search](https://apxml.com/courses/advanced-vector-search-llms/chapter-3-hybrid-search-approaches/rrf-fusion-algorithms)), but Cohere's reranker pricing for Maize's volume isn't published-favorable, and an LLM rerank gives us natural-language rationales for debugging. Practical call, not a research call.
- **Whether to ship Option B (single routing tool call) as a clean refactor or behind a feature flag with rollback.** The top-of-funnel research recommends an A/B; the audit-level concern is that a feature flag preserves the very technical debt we're trying to clean up. Suggest: ship clean, but keep the previous `contextualize_query` callable in the codebase for one release cycle in case rollback is needed.
- **What replaces B19's pedagogical concept-reinforcement role.** B19 is bad retrieval architecture, but it was solving a real generation-side problem ("the chunk is a problem statement, the student needs concept context"). The audit says delete it; the generation prompt needs to absorb that work. Out of audit scope but flagged for the response_generator owner.
- **Whether the agentic self-critique loop (Q5 step 4) is worth its 3-10× token cost** for Maize's specific failure modes. The residual failures look like single-shot retrieval problems, not multi-hop reasoning problems. Lean: no, but worth one explicit decision in 3.3.

These are the calls 3.3 will need to make based on the user's risk tolerance, the existing prod QA logs (where the user wants ≥20-30 rows per change), and which Phase-B sequencing the team can absorb.

## 3.3 Decisions + path-dependent next move

**Status:** TODO — populated after 3.1 and 3.2 land.

Will record:
- Per-finding decision (apply now / apply later / reject) with rationale
- Whether the verdict is "architecture matches best practices, optimize components" (proceed component-by-component with parked plans like top-of-funnel Option B) or "architecture has structural gaps" (refactor structure first)
- Links to any follow-up implementation plans (e.g., a hypothetical Phase B)
- Explicit decision about what to do with the parked top-of-funnel research

## Status + audit trail

| Date | Event | Ref |
|---|---|---|
| 2026-05-23 | Scaffold created (Step 3.0) | this commit |
| 2026-05-23 | 3.1 map populated (manual; Explore subagents bailed on phantom permissions) | uncommitted |
| _TBD_ | 3.2 audit populated by research subagent | _TBD_ |
| _TBD_ | 3.3 decisions logged | _TBD_ |
| _TBD_ | Architecture review concluded; follow-up plan(s) created | _TBD_ |

## Related documents

- [Phase A implementation plan (Stages 1-5 of V2 retrieval refactor)](maize-retrieval-phase-a-implementation-plan-2026-05-22.md)
- [Gap analysis (the failure-mode discovery that drove Phase A)](maize-retrieval-gap-analysis-2026-05-22.md)
- [Doc-classification research (Stage 2B basis)](maize-retrieval-doc-classification-research-2026-05-22.md)
- [Residual-failures research (Step 1 basis)](maize-retrieval-residual-failures-research-2026-05-22.md)
- [Top-of-funnel research (parked, may revive in 3.3)](maize-retrieval-top-of-funnel-research-2026-05-23.md)
