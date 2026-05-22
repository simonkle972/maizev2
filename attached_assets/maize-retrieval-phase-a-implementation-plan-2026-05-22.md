# Maize TA — Phase A Retrieval Refactor Implementation Plan

**Created:** 2026-05-22 · **Last updated:** 2026-05-22 (Stage 2B added — design correction from enum to per-TA configurable categories) · **Grounded in:** [gap analysis 2026-05-22](maize-retrieval-gap-analysis-2026-05-22.md) which is grounded in [parked retrieval plan](maize-retrieval-redesign-plan.md). Read those first if you arrived here cold.

## Context

The Phase A v1 scope is fixed (per the gap analysis): drop hard SQL filter + hybrid Stage 1, per-turn query rewriter, structured doc classification + solutions-as-supplementary, and the doc-classification field becomes the primary semantic axis for retrieval (replacing the legacy doc_type enum). CoRank Stage 2.5 coarse rerank + CDE embeddings are deferred. This document is the step-by-step plan to actually build it.

**Important revision (Stage 2B, 2026-05-22).** The original plan committed to a fixed 6-value `doc_role` enum (problem/solution/lecture/syllabus/reference/other) as the new classification axis. User correctly pushed back: that's the same enum-rigidity that caused the Type A failure with `doc_type`. Stage 2 shipped with the enum (commit `67873cd`), but Stage 2B reverses the SHAPE of that decision before Stage 3 lands. The new model is per-TA configurable categories — see Stage 2B below. The PLUMBING from Stage 2 (LLM call shape, provenance tracking, backfill script structure, PATCH route hooks) carries over; only the schema specifics change.

Goal: ship the refactor in a way that's (a) staged across discrete sessions so each lands cleanly, (b) feature-flagged so the legacy retrieval path remains available, (c) verifiable via the existing eval harness, and (d) rollback-able if something regresses.

## Scope reminder (one paragraph, post-Stage-2B revision)

**Ship in v1:** new `TeachingAssistant.doc_categories` JSON column (per-TA configurable list, seeded with sensible defaults at TA creation) + free-form `Document.doc_category` field auto-classified at upload from the TA's list (replaces `doc_type` as the primary semantic axis for retrieval — see Stage 2B for why this differs from the original Stage 2 doc_role enum); new `bm25_tsvector` column for hybrid Stage 1; new pre-stage `rewrite_query()` (per-turn decontextualizer); new `hybrid_doc_search()` replacing the regex+fuzzy-filename matcher in `analyze_query()`; updated `llm_rerank()` prompt surfacing `doc_category` text; updated `BASE_INSTRUCTIONS` in `response_generator.py` for the existing solutions-handling pedagogy rule (kept as-is — no new boolean flags).

**Dropped from original plan (Stage 2B revision):** `is_solution` / `is_lecture` booleans (pedagogy is prompt-level, not retrieval-level — solutions-leak isn't a real failure mode per user feedback). The `classify_intent()` per-turn helper is also dropped from the v1 scope; the LLM at response time handles intent in-context.

**Defer to v2:** CoRank Stage 2.5 coarse rerank, CDE embeddings.

## Implementation stages

The work is staged into SIX discrete sessions (was five — Stage 2B added 2026-05-22 to correct the schema-shape decision committed too early in Stage 2). Each stage lands a checkpoint that's individually shippable (or at least feature-flag-disable-able) without breaking the existing app. Order matters — later stages depend on earlier ones.

| Stage | What | Risk | Status | Estimated effort |
|---|---|---|---|---|
| 1 | Schema + migration + Config flags | Low | ✅ shipped (commit `4e4abf7`) | ~0.5 session |
| 2 | Indexing pipeline: doc_role auto-classification + BM25 tsvector | Medium | ✅ shipped (commit `67873cd`) — but schema-shape revised in 2B | ~1 session |
| **2B** | **Design correction: research + rework from enum to per-TA configurable categories** | Low (additive schema; rename existing) | ☐ NEXT | ~3-4.5h |
| 3 | Retriever: pre-stage rewriter + hybrid Stage 1 (behind feature flag) | High (the load-bearing change) | ☐ | ~1-1.5 sessions |
| 4 | Retriever: solutions-as-supplementary integration + reranker surfaces doc_category | Medium | ☐ | ~0.5 session |
| 5 | Response generator prompt rule confirmation + final integration + eval re-run | Low | ☐ | ~0.5 session |

Total: 4-6 sessions. Below, each stage in detail.

---

## Stage 1 — Schema, migration, feature flags

**Goal:** Add the database columns we'll need + the Config-driven feature flag that gates the new retrieval path. Lowest-risk piece; lands first so subsequent stages have something to populate / consume.

### Changes

#### 1.1 — Model additions ([models.py](../models.py))

Add to `Document` model:

```python
# Phase A retrieval refactor (gap analysis 2026-05-22).
# doc_role replaces doc_type as the primary semantic axis for retrieval.
# doc_type stays for backward-compat / UI continuity but is no longer
# load-bearing in the retriever.
doc_role = db.Column(db.String(32), nullable=True)
# {'source': 'auto'|'professor', 'confidence': float, 'classified_at': str}
doc_role_provenance = db.Column(db.JSON, nullable=True)
# BM25 over chunk_text content for hybrid Stage 1 retrieval.
# Indexed via GIN for fast text-search queries.
bm25_tsvector = db.Column(TSVECTOR, nullable=True)
```

Add to `DocumentChunk` model:

```python
# Denormalized copy of Document.doc_role so the retriever can filter
# chunks without joining back to documents. Synced via the same path
# as the other denormalized metadata (display_name, doc_type, etc.).
doc_role = db.Column(db.String(32), nullable=True)
```

**Grounding:** `doc_role` enum + provenance is the ACORD-style document-role typology pattern (see gap analysis decision 3). `bm25_tsvector` is the BM25 substrate for the hybrid Stage 1 (gap analysis decision 1).

#### 1.2 — Migration

Run `flask db migrate -m "Phase A retrieval refactor: doc_role + doc_role_provenance + bm25_tsvector"` to generate a migration file. Review it. Strip any unrelated autogeneration noise (we've been bitten by this before, see [migration 509f7774eea9](../migrations/versions/509f7774eea9_add_indexing_warnings_json_column_to_ta.py) where Alembic also tried to drop a CASCADE on `chat_message_images`). Add a GIN index on `bm25_tsvector` manually in the migration:

```python
op.create_index('ix_documents_bm25_tsvector', 'documents', ['bm25_tsvector'], postgresql_using='gin')
```

#### 1.3 — Config flags ([config.py](../config.py))

Add:

```python
# Phase A retrieval refactor (gap analysis 2026-05-22). When True, retrieve_context
# uses the new pre-stage query rewriter + hybrid Stage 1 + intent classifier path.
# When False, falls back to the legacy analyze_query() regex + fuzzy filename matcher.
# Default False; flip to True after verifying via the eval harness.
RETRIEVAL_V2_ENABLED = os.getenv("RETRIEVAL_V2_ENABLED", "false").lower() == "true"

# Phase A retrieval refactor. RRF k constant for reciprocal rank fusion.
# Standard default is 60; lower values bias toward top-ranked items in each list.
RRF_K = int(os.getenv("RRF_K", "60"))

# Phase A retrieval refactor. Top-K candidate documents Stage 1 returns.
# Stage 2 (chunk retrieval) is constrained to chunks from these documents.
STAGE_1_TOP_K_DOCS = int(os.getenv("STAGE_1_TOP_K_DOCS", "5"))
```

**Grounding:** Feature-flagging the new path lets us A/B test locally before any blast-radius change. RRF_K=60 is the standard from [Serghei blog](https://blog.serghei.pl/posts/reciprocal-rank-fusion-explained/).

### Verification (Stage 1)

1. `flask db upgrade` applies cleanly locally.
2. `psql ... -c "\d documents"` shows the three new columns + GIN index.
3. `python -c "from app import app; from models import Document; app.app_context().push(); print(Document.query.first().doc_role)"` returns None (column exists, no data yet).
4. App starts; existing chat queries work unchanged (RETRIEVAL_V2_ENABLED defaults to False → legacy path).

### Risks + rollback (Stage 1)

- Risk: Alembic autogenerates noise. Mitigation: hand-edit the migration file before applying, as we've done before.
- Rollback: `flask db downgrade -1` drops the new columns. Safe — no data to lose.

---

## Stage 2 — Indexing pipeline: doc_role auto-classification + BM25 tsvector

**Goal:** populate the new columns. Existing docs get a backfill pass; new uploads use the updated path.

### Changes

#### 2.1 — Doc-role auto-classifier ([src/document_processor.py](../src/document_processor.py))

Add a function near `extract_metadata_with_llm` (~line 1463 area):

```python
def classify_doc_role(text: str, filename: str) -> tuple[str, float]:
    """Classify the document's pedagogical role using gpt-4o-mini.

    Returns (role, confidence) where role is one of:
      'problem' | 'solution' | 'lecture' | 'syllabus' | 'reference' | 'other'

    Used by the retriever to determine PRIMARY citation eligibility and
    to tag solution chunks as supplementary reference in the prompt.

    Grounded in: ACORD's clause-role pattern (gap analysis decision 3).
    """
    # gpt-4o-mini call with a tight few-shot prompt that includes:
    # - the 6 role definitions with concrete examples
    # - first ~2000 chars of doc content + filename
    # - JSON response: {"role": "problem", "confidence": 0.92, "rationale": "..."}
```

Wire it into `process_and_index_documents_resumable` (~line 1463-1480) alongside `extract_metadata_with_llm`. Set `doc.doc_role` + `doc.doc_role_provenance = {"source": "auto", "confidence": ..., "classified_at": ..., "rationale": ...}`.

**Behavior on user override:** the existing PATCH route for metadata edit will need a `doc_role` field added (admin route in app.py:884, professor route in professor.py:1043). When professor sets `doc_role` manually, update provenance to `{"source": "professor", ...}`. Auto-classification skips if `doc.doc_role` is already set and provenance is `professor` (don't overwrite human input).

#### 2.2 — BM25 tsvector population

In `process_and_index_documents_resumable` after chunk extraction (~line 1487 area where chunks are built):

```python
# Build BM25 tsvector over the doc's full extracted text for Stage 1
# hybrid retrieval. We use the full text rather than per-chunk because
# Stage 1 routes at the DOCUMENT level (Stage 2 routes at chunk level).
doc.bm25_tsvector = db.func.to_tsvector('english', sanitize_text(text))
```

Run on every doc-indexed flow (fresh + incremental). Existing already-indexed docs get covered by the backfill (Stage 2.3).

**Grounding:** BM25 over content (not filename) is explicit in the gap analysis to avoid re-introducing Type C false positives. Cited: [Serghei RRF blog](https://blog.serghei.pl/posts/reciprocal-rank-fusion-explained/), [OpenSearch RRF blog](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/).

#### 2.3 — Backfill script (`scripts/backfill_doc_role_and_bm25.py`)

One-shot script that iterates all existing `Document` rows:
- For each: re-extract text (use the same path as the worker), call `classify_doc_role`, set `doc_role` + `doc_role_provenance`.
- Compute and set `bm25_tsvector`.
- Update denormalized `DocumentChunk.doc_role` to match.
- Commit per-batch.

Run once on local DB. Will need to run on prod after we ship.

#### 2.4 — Chunk-metadata sync update

`DocumentChunk.doc_role` needs to be synced when `Document.doc_role` changes. Add `doc_role` to the sync blocks in:
- [professor.py:1072-1078](../professor.py#L1072) (professor PATCH route)
- [app.py:884-919](../app.py#L884) (admin PATCH route)

### Verification (Stage 2)

1. Backfill script runs successfully on local ECON S1117 (69 docs). All docs have `doc_role` set with confidence ≥ 0.7 (LLM should be confident on academic doc types).
2. Spot-check: `econometrics quiz 1-5` all classify as `'problem'` (NOT `'homework'`). `econ117_pset03- 2025, solutions` classifies as `'solution'`. `Interactive Lecture 3-1-1` classifies as `'lecture'`. `ECON 117 syllabus, session B, 2025` classifies as `'syllabus'`.
3. `psql ... -c "SELECT count(*) FROM documents WHERE bm25_tsvector IS NOT NULL;"` returns 69.
4. Existing chat queries still work unchanged (we haven't flipped the feature flag).
5. Re-running the baseline eval against the CURRENT retriever still produces the same scorecard (no behavior change — we only added data).

### Risks + rollback (Stage 2)

- Risk: LLM mis-classifies some docs. Mitigation: the `confidence` field + per-doc UI to override (later v2 polish). For now, professors override via existing PATCH route once we surface `doc_role` in the UI.
- Risk: Backfill takes long on large corpora. Mitigation: batched + idempotent — can resume if interrupted.
- Rollback: clear the new columns (`UPDATE documents SET doc_role = NULL, bm25_tsvector = NULL`). Data isn't lost — the source text is unchanged, just re-run the backfill.

---

## Stage 2B — Design correction: from doc_role enum to per-TA configurable categories

**Inserted 2026-05-22, AFTER Stage 2 had already shipped (commit `67873cd`).** This stage exists because the original Stage 2 committed to a wrong-shape schema. Stage 2B reworks the schema before Stage 3 lands.

### Why this section exists

Stage 2 shipped a 6-value `doc_role` enum (`problem` | `solution` | `lecture` | `syllabus` | `reference` | `other`) as the new PRIMARY semantic axis for retrieval. The choice was grounded in the gap analysis, which had cited ACORD-style document-role typology + Metadata-Driven RAG (see [maize-retrieval-gap-analysis-2026-05-22.md](maize-retrieval-gap-analysis-2026-05-22.md) decisions 3 + 4). But during Stage 2 review the user correctly pushed back on two points:

1. **The enum is the same kind of forced consolidation that broke us with `doc_type`.** A quiz, a homework, a lab, and "extra problems" are all conceptually distinct in the professor's mental model, but the enum smashes them all into `problem`. That's exactly the rigidity that caused Type A in the original retrieval (LLM mapped "econometrics quiz N" → `doc_type=homework, assignment=N`, colliding with PSets). Replacing one rigid enum with another rigid enum doesn't address the root cause.

2. **`is_solution` and `is_lecture` booleans (which Stages 3/4 would have added) are solving a non-problem.** Solutions-leak has not been a real failure mode in prod — the response_generator's prompt rules already prevent the LLM from divulging answers. My eval's `forbidden_hit = 40%` metric for Type D was measuring the wrong thing: solutions appearing IN retrieved chunks is fine; what would be bad is solutions getting DIVULGED to the student, which doesn't happen because of the prompt. The real retrieval problem is FINDING the right doc when a student asks ("help with quiz 2" → return quiz 2), not preventing solutions from appearing in context.

### The revised design (Stage 2B target state)

**Two schema changes, both more flexible than the original Stage 2 design:**

| Field | Type | Editable by | Used by |
|---|---|---|---|
| `TeachingAssistant.doc_categories` | JSON array of strings | Professor — at TA creation (pre-seeded with sensible defaults) + editable in manage_ta UI | UI dropdown population; classifier candidate list |
| `Document.doc_category` (renamed from `doc_role`) | String, free-form but validated to be in the TA's `doc_categories` array | Professor (UI dropdown). LLM auto-suggests at upload from the TA's list. | UI display + Stage 3 reranker as text context (NOT as filter or boolean tag) |
| `DocumentChunk.doc_category` (renamed from `doc_role`) | String — denormalized copy | Synced via existing chunk-sync paths | Retriever consults via chunk row |

**Default seed list at TA creation:** `["lectures", "readings", "homeworks", "problem sets", "quizzes", "labs", "exams", "syllabus", "reference materials", "extra problems", "other"]`. Professor can add/remove/rename.

**Booleans NOT introduced:** no `is_solution`, no `is_lecture`. Pedagogy stays in the response_generator prompt (already works there). Retrieval gets `doc_category` as a single signal at the reranker stage.

**`classify_intent()` per-turn helper dropped from v1.** The LLM at response time handles intent in-context.

### Sub-phase 2B.1 — Research (literature mining BEFORE rework)

To honor the project's data → research → validation → implementation discipline (which we'd previously applied at the gap-analysis level but skipped at this sub-decision level), Stage 2B starts with a focused literature pass.

Output: `attached_assets/maize-retrieval-doc-classification-research-2026-05-22.md` answering these 5 questions:

1. **Per-corpus / per-tenant configurable taxonomies in RAG.** Is this a known production pattern? Multi-tenant SaaS RAG systems (Glean, Zendesk, Notion AI), LangChain / LlamaIndex docs on custom metadata schemas.
2. **User-defined doc labels as a retrieval signal.** Empirical evidence — does retrieval quality improve when the LABEL is user-controlled but the reranker consumes it as text context, vs a fixed-enum baseline?
3. **Educational-RAG specifically.** Do Khanmigo, Coursera AI tutor, Quizlet Q-Chat, Carnegie Learning, etc. surface "tell me what kinds of docs you have" setup flows? Public writeups / talks.
4. **Cold-start defaults for user-defined taxonomies.** What do production systems do when no user-defined taxonomy exists yet?
5. **Failure modes of free-form labels.** Where do they break? Guardrails the literature recommends?

Expected outcome: either evidence supports the user's design (proceed), or surfaces caveats (refine first), or literature is silent on this specific UX (proceed on first-principles reasoning grounded in already-cited general patterns from the gap analysis). All three outcomes are acceptable; we just want the audit trail.

### Sub-phase 2B.2 — Rework (gated on research)

**Schema:**
- Add `TeachingAssistant.doc_categories` (JSON array nullable). Migration includes a backfill step to populate the field on existing TAs with the default seed list.
- Rename `Document.doc_role` → `Document.doc_category` (or drop+add — TBD by migration safety check). Stays a String(32).
- Rename `DocumentChunk.doc_role` → `DocumentChunk.doc_category`.

**Code:**
- `classify_doc_role()` → `classify_doc_category(text, filename, ta_categories)`. LLM picks from the passed-in TA category list. Falls back to last element (typically "other") if no good match.
- Backfill re-runs on existing 69 ECON S1117 docs with the new TA-scoped classifier. Existing role values from the original Stage 2 backfill can stay as initial values where the labels overlap (e.g., "lecture" stays "lecture"); the LLM re-picks where they don't (e.g., "problem" gets refined into "homeworks" or "quizzes" or "labs" depending on filename).
- PATCH routes (admin app.py:884 + professor professor.py:1043): accept `doc_category` string; validate against the doc's TA's `doc_categories` array; reject with 400 if not in list.
- Stage 3 (retriever, when it lands): reranker prompt surfaces `chunk.doc_category` as text context per chunk. Hybrid Stage 1 BM25+dense unchanged (content-driven, no metadata filter).

**UI (TA creation + manage_ta):**
- New "Document categories" section on manage_ta (collapsed-by-default panel near the top). Lists current categories with add/remove/rename controls.
- Per-doc-row `<select>` populated from the TA's `doc_categories`. Replaces the existing `doc_type` dropdown. Provenance badge: `auto · 0.92` or `professor`.
- TA creation flow: existing flow lands the professor on manage_ta after TA creation; the categories panel is pre-populated with the default seed list. No new step required at creation; defaults work out of the box.

### What survives from Stage 2 unchanged

The plumbing carries over even though the schema specifics change:
- LLM call shape (gpt-4o-mini, few-shot prompt, JSON output, error handling).
- Provenance tracking pattern (`{source: 'auto'|'professor', confidence, ...}`).
- Backfill script structure (idempotent, per-doc, chunks reconstructed from existing chunk_text).
- PATCH route handler hooks (chunk-metadata sync block).
- `Document.bm25_tsvector` + GIN index — completely orthogonal, fully preserved.
- All Config flags (`RETRIEVAL_V2_ENABLED`, `RRF_K`, `STAGE_1_TOP_K_DOCS`).

### What's deprecated / dropped from prior decisions

- The 6-value `doc_role` enum constraint — column becomes free-form text.
- The `is_solution` / `is_lecture` booleans we'd planned for Stages 3-4 — never added.
- The `classify_intent()` per-turn helper — dropped from v1 scope.
- The ACORD-style document-role-typology framing in the gap analysis. ACORD's general pattern (separate retrieval-load-bearing metadata from filtering) still informs us; the specific 5-role enum proposal does not.

### Effort estimate

| Sub-phase | Time |
|---|---|
| Research phase + writeup | ~1-2h |
| Schema migration + classifier refactor | ~30 min |
| Backfill re-run on local ECON S1117 | ~2 min |
| PATCH route adjustments | ~15 min |
| UI: categories panel on manage_ta + per-doc dropdown | ~45-60 min |
| Verification + commit | ~15 min |
| **Total** | **~3-4.5h** |

### Verification (Stage 2B acceptance criteria)

1. All TAs (existing + new) have `doc_categories` populated (default seed for existing; professor-editable for new).
2. All 69 ECON S1117 docs have `doc_category` matching one of the TA's `doc_categories`.
3. PATCH route rejects (HTTP 400) attempts to set `doc_category` to a value not in the TA's list.
4. Professor can add a new category (e.g., "case studies") via UI; can immediately assign docs to it.
5. Spot-check: `econometrics quiz N` docs now classify as "quizzes" (not "problem"), `econ117_pset02_2025B_with_table` as "homeworks" or "problem sets" (professor's choice of label), `Lab2` as "labs". Reflects real-world distinctions the user wanted preserved.
6. Stage 3 implementation is unblocked — retriever's rerank prompt has a clean `doc_category` field to consume.

### Risks + rollback

- **Risk:** Some existing docs may classify to "other" if their old `doc_role` value doesn't map cleanly to any default seed entry. Mitigation: include "solutions" in the default seed list (and other commonly-needed values); professors can rename / remove later.
- **Risk:** Per-TA categories add UX friction at TA creation (small new panel). Mitigation: defaults work out of the box; no new mandatory step.

### Backwards compatibility strategy

The schema rework is **additive, never rename**. This is critical for safe rollout + rollback.

| Concern | Strategy |
|---|---|
| Schema migration | **ADD** `doc_categories` JSON on TA, `doc_category` String(64) on Document, `doc_category` String(64) on DocumentChunk. **KEEP** existing `doc_role`, `doc_role_provenance`, `doc_type` columns. Nothing dropped in this migration. |
| Backfill at deploy | All existing TAs get default `doc_categories` seed. All existing docs get `doc_category` populated via new classifier using the TA's list (~1-2s per doc). Chunks get denormalized sync. |
| Code reads | Prefer `doc_category`; fall back to `doc_role` if `doc_category` is NULL. Means code works cleanly during/after partial backfill. |
| Code writes | Only `doc_category` going forward. `doc_role` column survives but isn't updated. |
| PATCH routes | Accept `doc_category` (slug or label, normalize to slug, validate against TA's list). Drop `doc_role` from accepted payload. Tolerates docs that have only legacy `doc_role` set. |
| UI | Show `doc_category` dropdown only. Hide both `doc_type` and `doc_role` from manage_ta. Defensive fallback: if `ta.doc_categories` is empty, render the default seed list inline. |
| Rollback (code revert) | Old code reads `doc_role` — still populated for pre-v1 docs, still works. Loss: docs added during v2 period have only `doc_category` set; old code sees them as unclassified. Acceptable — retrieval still works without the hint. |
| Rollback (migration downgrade) | Drops the three new columns (`doc_categories`, both `doc_category`). Standard. |
| Future cleanup (v2 polish, NOT v1) | After 2+ weeks of stable v1, separate cleanup migration drops `doc_role`, `doc_role_provenance`, `doc_type` from all tables. Need confidence nothing still reads them. |

### Slug ↔ label representation (Refinement #1 from research)

Categories are stored as `{slug, label}` objects, not plain strings:

```json
[
  {"slug": "lectures", "label": "Lectures"},
  {"slug": "homeworks", "label": "Homeworks"},
  {"slug": "problem_sets", "label": "Problem Sets"},
  ...
]
```

- **`Document.doc_category` persists the SLUG**, not the label.
- Slugs are auto-generated from the label at creation: `"Problem Sets"` → `slug: "problem_sets"`. Lowercased, internal whitespace → underscores, non-alphanumeric chars stripped.
- Renaming `"Problem Sets"` → `"PSets"` updates only the label; the slug stays `"problem_sets"`. All existing docs with `doc_category="problem_sets"` still point at the same category.
- Adding a new category generates a fresh slug (case-insensitive uniqueness within the TA).

This is the Library Drift (arXiv 2605.19576) mitigation: renames don't orphan classified documents.

### Input normalization rules (Refinement #2 from research)

At the API boundary (PATCH route + TA category-edit endpoint):
- Trim leading/trailing whitespace
- Collapse internal whitespace to single spaces
- Reject empty strings
- Max 64 chars (column constraint matches)
- Allow letters, numbers, spaces, hyphens (CJK characters too — don't restrict to ASCII)
- Case-preserved for display label; case-insensitive uniqueness for slug
- Reject duplicate slugs within a TA (case-insensitive)

### Default seed list at TA creation

```json
[
  {"slug": "lectures", "label": "Lectures"},
  {"slug": "readings", "label": "Readings"},
  {"slug": "homeworks", "label": "Homeworks"},
  {"slug": "problem_sets", "label": "Problem Sets"},
  {"slug": "quizzes", "label": "Quizzes"},
  {"slug": "labs", "label": "Labs"},
  {"slug": "exams", "label": "Exams"},
  {"slug": "solutions", "label": "Solutions"},
  {"slug": "syllabus", "label": "Syllabus"},
  {"slug": "reference_materials", "label": "Reference Materials"},
  {"slug": "extra_problems", "label": "Extra Problems"},
  {"slug": "other", "label": "Other"}
]
```

12 entries (research flagged 11 might be too many for cognitive load — added "solutions" explicitly per the original gap analysis's solutions-handling concern, even though pedagogy stays in the prompt). Professor edits this list freely post-creation. If we see "Other" overuse in analytics, that's signal to reduce/tune defaults.

### Audit trail update for refinements

| Decision | Source |
|---|---|
| Slug + label representation (not plain strings) | Research finding Q5; Library Drift (arXiv 2605.19576) |
| Input normalization rules | Research finding Q5; LLM Failure Modes (arXiv 2511.19933) |
| Per-TA scoping (not global) | Research findings Q1 + Q5 |
| Default 12-category seed | Research finding Q4 (defaults + JIT pattern); informed by gap analysis's failure type analysis |

### Updated audit trail (Stage 2 + Stage 2B together)

| Decision | First documented | Status |
|---|---|---|
| `doc_role` enum as primary semantic axis (gap analysis decision 4) | gap analysis 2026-05-22 | Reversed in Stage 2B — replaced with per-TA configurable `doc_category` |
| 6-value enum (problem/solution/lecture/syllabus/reference/other) | gap analysis 2026-05-22 | Dropped in Stage 2B — string field validated against per-TA list |
| ACORD-style role typology framing | gap analysis decision 3 + 4 | General pattern still informs us; specific 5-role enum dropped |
| `is_solution` / `is_lecture` booleans (Stage 4) | implementation plan 2026-05-22 | Dropped in Stage 2B — pedagogy stays prompt-level |
| `classify_intent()` per-turn helper | implementation plan 2026-05-22 | Dropped in Stage 2B — LLM handles intent in-context |
| BM25 + dense + RRF hybrid Stage 1 | gap analysis decision 1 | Unchanged. Survives Stage 2B. |
| Per-turn query rewriter | gap analysis decision 2 | Unchanged. Survives Stage 2B. |
| Drop hard SQL filter | gap analysis decision 1 | Unchanged. Survives Stage 2B. |

---

## Stage 3 — Retriever: pre-stage rewriter + hybrid Stage 1 (BEHIND FEATURE FLAG)

**Goal:** the load-bearing change. New retrieval path that the feature flag gates. Legacy path remains intact until we verify.

### Changes

All changes in [src/retriever.py](../src/retriever.py).

#### 3.1 — New per-turn query rewriter

New function `rewrite_query()` placed near `contextualize_query()` (~line 1417):

```python
def rewrite_query(query: str, prior_turns: list, prior_retrieved_doc: str | None) -> dict:
    """Decontextualize the student's query into a self-contained form.

    If the query is already standalone (no coreferences, no follow-up references),
    return it unchanged. Otherwise produce a rewritten version that resolves
    pronouns / "this" / "that" / "the previous question" against prior_turns
    AND can override a stuck cache via prior_retrieved_doc context.

    Returns {"rewritten": str, "was_standalone": bool, "rationale": str}.

    Grounded in: H-RAG (arxiv 2605.00631) — lightweight LLM rewriter that
    returns query unchanged when standalone, decontextualizes when coreferent.
    """
    # gpt-4o-mini call. ~100-200ms.
```

**Grounding:** Per-turn rewriter is decision 2 of the gap analysis (research: H-RAG, RAGAboutIt, Alhena). Source URLs in gap analysis.

#### 3.2 — New hybrid document search (Stage 1)

New function `hybrid_doc_search()` placed near `analyze_query()` (~line 1110):

```python
def hybrid_doc_search(query: str, ta_id: str, top_k: int = Config.STAGE_1_TOP_K_DOCS) -> list[dict]:
    """Stage 1 hybrid document-level retrieval.

    BM25 over Document.bm25_tsvector + dense over Document chunk embeddings,
    fused via Reciprocal Rank Fusion (RRF). Returns top-K document IDs +
    diagnostic info.

    Replaces the regex + fuzzy filename matcher + hard SQL filter that
    used to live in analyze_query(). doc_type / assignment_number are
    NO LONGER used as hard filters here.

    Grounded in: VersionRAG (arxiv 2510.08109), Serghei + OpenSearch RRF
    docs, OptyxStack (hard-filter anti-pattern). See gap analysis decision 1.
    """
    # 1. BM25 query: pgvector's to_tsquery + tsvector @@ tsquery, ranked by ts_rank.
    # 2. Dense query: embed query, average doc-chunk embeddings OR use a doc-level
    #    embedding if we add one later. For v1, mean-pool the doc's top-N chunk
    #    embeddings as a doc-level signal.
    # 3. Convert each ranking to RRF score: 1 / (k + rank), k = Config.RRF_K.
    # 4. Sum the two RRF scores per doc. Return top_k.
```

**Important details:**
- The dense half mean-pools chunk embeddings per-document at query time (cheap with pgvector) rather than maintaining a separate doc-level embedding column. Reason: keeps the migration smaller; we can add a dedicated doc_embedding column in v2 if mean-pool is empirically insufficient.
- BM25 uses PostgreSQL's full-text search with `english` config — same dictionary used by `to_tsvector('english', ...)` in Stage 2.

#### 3.3 — Refactor `retrieve_context()` to use the new path

[retrieve_context (line 1668)](../src/retriever.py#L1668) gets a top-level branch:

```python
if Config.RETRIEVAL_V2_ENABLED:
    return _retrieve_v2(...)  # new path
else:
    # existing legacy path unchanged
```

`_retrieve_v2` flow:
1. Call `rewrite_query()` with conversation_history + prior_retrieved_doc from session cache.
2. Call `hybrid_doc_search()` to get top-K candidate doc IDs.
3. Vector search over chunks WHERE document_id IN (candidate doc IDs). NO hard filter on doc_type/assignment_number/etc.
4. Pass to existing `llm_rerank()` (Stage 3) with the prompt update from Stage 4.
5. Apply hybrid fallback path (existing logic) if needed.

### Verification (Stage 3)

1. `RETRIEVAL_V2_ENABLED=false` (default): all baseline eval rows produce IDENTICAL results to today's baseline scorecard. No behavior change.
2. `RETRIEVAL_V2_ENABLED=true`: rerun the eval harness. Expect Type A correct_hit@5 to jump from 50% to ≥80%, Type C from 0% to ≥70%. Working cases stay at or above 65%.
3. Spot-check one row manually: "I need help with problem set 2" → top-1 should now be `econ117_pset02_2025B_with_table`, NOT `econometrics quiz 2`. Verify the pre-stage rewriter doesn't mangle the query when it's already standalone.

### Risks + rollback (Stage 3)

- Risk: Hybrid Stage 1 over-retrieves or under-retrieves at the doc level. Mitigation: STAGE_1_TOP_K_DOCS is env-configurable; start at 5, tune via eval.
- Risk: Mean-pooled doc embeddings perform poorly compared to a dedicated doc-level embedding. Mitigation: if Type B residuals appear post-Stage 3, add a `doc_embedding VECTOR(1536)` column + populate via the same backfill path in v2.
- Rollback: `RETRIEVAL_V2_ENABLED=false`. The legacy path is untouched. Zero data risk.

---

## Stage 4 — Reranker surfaces doc_category as text context (REVISED per Stage 2B)

**Goal:** wire `doc_category` into the reranker so the LLM can use it as one signal among many when picking primary chunks for a query. This is what's left of the "Type D fix" after Stage 2B dropped the intent classifier and solutions-as-supplementary booleans.

> **What changed from the original Stage 4 plan (pre-Stage-2B):** The original Stage 4 added a per-turn `classify_intent()` helper + tagged solution chunks as `'solution_reference'` retrieval-role + extended the `chat_streaming.py` context assembly with a `[REFERENCE / DO NOT DIVULGE]` block. Stage 2B dropped all of that — solutions-leak isn't a real failure mode in prod; the response_generator prompt already prevents the LLM from divulging answers. What remains is the simpler: surface `doc_category` to the reranker as context, let the LLM do the right thing.

### Changes

#### 4.1 — Reranker prompt surfaces `chunk['doc_category']`

In `llm_rerank()` (~line 674 of [src/retriever.py](../src/retriever.py)): update the chunk-summary template to include `chunk.doc_category` alongside `file_name` and the text preview. Reranker is implicitly category-aware — given "help me with quiz 2", chunks tagged "quizzes" in the right TA get a natural relevance boost; chunks tagged "homeworks" don't.

No boolean filtering. No `'solution_reference'` retrieval role. The text label IS the signal.

#### 4.2 — Source-line citation behavior

`chat_streaming.py:346-355` already lists `chunks[:8]` deduplicated to top 3 in the Sources line. No change needed — the reranker's category-aware ordering puts the right primary doc top-1, which is what gets cited. If a solution doc appears in the retrieved set, it may appear in Sources too; we accept that because (a) the LLM doesn't divulge from solutions per existing pedagogy prompt, and (b) source attribution showing both the problem and its solutions doc is arguably useful for the student to know.

### Verification (Stage 4)

1. With `RETRIEVAL_V2_ENABLED=true`: rerun the eval. Type C correct_hit@5 should be ≥70% (was 0% in the baseline) — primarily driven by hybrid Stage 1, but doc_category in the rerank prompt helps too.
2. Spot-check: "I need help with quiz 2" → top-1 PRIMARY chunk is from `econometrics quiz 2` (doc_category='quizzes'), not from PS2 or Lab2 even though those share the same legacy `doc_type='homework'`.
3. Spot-check: "I need help with PS2 question 1" → top-1 PRIMARY is `econ117_pset02_2025B_with_table` (doc_category='homeworks' or 'problem sets'), not a quiz doc.

### Risks + rollback (Stage 4)

- Risk: LLM rerank ignores the `doc_category` field. Mitigation: explicit prompt language directing the reranker to consider the category alongside content. Test empirically.
- Rollback: `RETRIEVAL_V2_ENABLED=false`.

---

## Stage 5 — Response generator prompt rule + final integration + eval re-run

**Goal:** wire up the LLM-level "don't divulge" rule and run the post-refactor scorecard.

### Changes

#### 5.1 — Update `BASE_INSTRUCTIONS` ([src/response_generator.py:37](../src/response_generator.py#L37))

Add a new section (before or after the existing pedagogy rules) explaining the `[REFERENCE / DO NOT DIVULGE]` tag:

```
## REFERENCE-ONLY CONTEXT

You may sometimes see context tagged as
"[REFERENCE / DO NOT DIVULGE — solution material for your guidance only]"
followed by passages from a solutions document. Treat that material as
INTERNAL REFERENCE ONLY. You may use it to:
  - Validate whether the student's work is correct.
  - Construct better Socratic questions and hints.
  - Identify the conceptual step the student is missing.

You MUST NOT reveal the solution's answer, final numerical result, or
step-by-step worked answer directly to the student. The student is solving
the problem; your job is to guide them, not to give them the answer.

Citation rule: do not cite the solutions document in your "Sources" line.
Only the PRIMARY document (the problem the student is solving) gets cited.
```

**Grounding:** This is the prompt-level pedagogy rule per the gap analysis's solutions-as-supplementary design (decision 3, refined by user input).

#### 5.2 — Post-refactor scorecard

Set `RETRIEVAL_V2_ENABLED=true`, run:

```bash
DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 venv/bin/python eval/run_eval.py --out eval/post_refactor_scorecard.md
```

Side-by-side compare to `eval/baseline_scorecard_pre_refinement.md`:

**Pass criteria (all must hold):**
- Type A correct_hit@5: ≥75% (baseline 50%).
- Type B correct_hit@5: ≥90% (baseline 100% — keep parity).
- Type C correct_hit@5: ≥70% (baseline 0%).
- Type D `primary_is_forbidden`: ≤10% (baseline 40% on forbidden_hit). Adds a new metric to `run_eval.py` — `primary_is_forbidden` = top-1 doc in forbidden_doc_ids.
- Type E correct_hit@5: ≥85% (baseline 86% — keep parity; cache anchoring fix isn't measurable here).
- Working cases correct_hit@5: ≥65% (baseline 65% — must not regress).
- Errors: 0.

If any criterion fails: investigate. Don't flip the prod flag.

#### 5.3 — Add `primary_is_forbidden` metric to `run_eval.py`

Small change to `eval/run_eval.py`'s `RowResult` + aggregation: add `primary_is_forbidden` field = top-1 doc in forbidden_doc_ids. Per the gap analysis's solutions-handling refinement: `forbidden_hit` is too strict (solutions ARE allowed in context), but `primary_is_forbidden` cleanly measures "did a forbidden doc get cited as the primary source."

### Verification (Stage 5)

1. Post-refactor scorecard meets all pass criteria above.
2. Manual chat in a real browser session (RETRIEVAL_V2_ENABLED=true) — ask a problem-solving question on an indexed TA, verify the system DOES NOT divulge the solution even when one is in the context.
3. Same scenario but explicit "show me the answer" — verify intent classifier flips and the answer comes through.
4. Multi-turn cache-anchoring test (the prod failure that the eval can't reproduce): T1 ask about PS2, get correct answer. T2 ask follow-up. T3 ask "actually I meant PS3" — verify the system re-retrieves PS3 instead of staying anchored to PS2.

### Ship-to-prod sequence (after Stage 5 passes verification locally)

1. Commit all stages + push to main.
2. On VPS:
   - `cd /opt/maize`
   - Add `RETRIEVAL_V2_ENABLED=false` to systemd unit Environment= (DEFAULT OFF initially even on prod).
   - `sudo systemctl daemon-reload`
   - `sudo -u maize git pull`
   - `sudo -u maize ./venv/bin/flask db upgrade` (applies the new columns migration)
   - `sudo -u maize ./venv/bin/python scripts/backfill_doc_role_and_bm25.py` (populates doc_role + bm25 for all prod TAs; ~30 min for typical corpora)
   - `sudo systemctl restart maize` (picks up code + migration + backfilled data; STILL on legacy retrieval)
3. Verify clean startup: `sudo journalctl -u maize -n 100 --no-pager`.
4. Smoke-test prod chat — should be identical to pre-deploy behavior (legacy path).
5. Flip `RETRIEVAL_V2_ENABLED=true` in systemd unit, daemon-reload, restart.
6. Smoke-test prod chat — should now use the new path.
7. If anything regresses: flip back to false. Investigate.

## Rollback playbook

In order of escalation:

1. **Bad query response in prod**: flip `RETRIEVAL_V2_ENABLED=false`, restart. Legacy path resumes immediately.
2. **Bad data from backfill**: clear new columns (`UPDATE documents SET doc_role = NULL, bm25_tsvector = NULL`). Legacy path doesn't read these, so chat is unaffected.
3. **Migration itself causing issues**: `flask db downgrade -1`. Drops the columns. Data wasn't there to begin with, no loss.
4. **Code itself unstable**: `git revert <deploy commit>` + redeploy. Full restoration.

## Out of scope (Phase A v2 — DEFERRED with empirical triggers)

| Item | Add when |
|---|---|
| CoRank Stage 2.5 coarse rerank | If post-v1 scorecard shows Stage 3 precision is the new bottleneck. |
| CDE embeddings | If post-v1 Type B residuals appear in prod. |
| Force-rebuild after professor changes `doc_role` | If we see prod cases where doc_role override doesn't propagate cleanly. |
| UI surfacing of `doc_role` in admin + professor manage_ta | After v1 ships and we want professors to easily override auto-classification. |

## Critical files (all stages, updated post-Stage-2B)

| Stage | File | Purpose |
|---|---|---|
| 1 | [models.py](../models.py) | Document + DocumentChunk model additions (shipped) |
| 1 | [migrations/versions/](../migrations/versions/) | New migration (shipped: `6616a6d2eb0a`) |
| 1 | [config.py](../config.py) | Feature flag + RRF_K + STAGE_1_TOP_K_DOCS (shipped) |
| 2 | [src/document_processor.py](../src/document_processor.py) | `classify_doc_role` + bm25_tsvector population (shipped; classifier function gets repurposed in 2B) |
| 2 | scripts/backfill_doc_role_and_bm25.py | NEW one-shot backfill (shipped; structure reused in 2B) |
| 2 | [professor.py](../professor.py) + [app.py](../app.py) | Add doc_role to PATCH metadata sync (shipped; field renames in 2B) |
| **2B** | `attached_assets/maize-retrieval-doc-classification-research-2026-05-22.md` | NEW research output |
| **2B** | [models.py](../models.py) | Add `TeachingAssistant.doc_categories` JSON; rename `Document.doc_role` → `doc_category` + same for chunks |
| **2B** | new migration | Schema rename + add `doc_categories` column with default seed for existing TAs |
| **2B** | [src/document_processor.py](../src/document_processor.py) | `classify_doc_role` → `classify_doc_category(text, filename, ta_categories)` |
| **2B** | scripts/backfill_doc_role_and_bm25.py | Renamed + reworked to pass TA categories list |
| **2B** | [professor.py](../professor.py) + [app.py](../app.py) | PATCH route accepts `doc_category` string; validates against TA's `doc_categories` |
| **2B** | [templates/admin_manage_ta.html](../templates/admin_manage_ta.html) + [templates/professor/manage_ta.html](../templates/professor/manage_ta.html) | Categories panel + per-doc dropdown |
| 3 | [src/retriever.py](../src/retriever.py) | `rewrite_query` + `hybrid_doc_search` + `_retrieve_v2` branch |
| 4 | [src/retriever.py](../src/retriever.py) | `llm_rerank` prompt update to surface `chunk.doc_category` |
| 5 | [src/response_generator.py](../src/response_generator.py) | `BASE_INSTRUCTIONS` confirmation (existing pedagogy stays; no new section needed per Stage 2B) |
| 5 | [eval/run_eval.py](../eval/run_eval.py) | `primary_is_forbidden` metric (still useful as a diagnostic even though we're not chasing it as a target) |

## What this plan explicitly does NOT include (updated post-Stage-2B)

- Touching Phases B or C of the parked plan. They're parked because Phase A's per-turn rewriter likely subsumes them.
- Implementing CoRank Stage 2.5 or CDE. Deferred per the gap analysis with explicit triggers for when to revisit.
- `is_solution` / `is_lecture` booleans on Document. Dropped in Stage 2B — pedagogy stays prompt-level.
- `classify_intent()` per-turn helper. Dropped in Stage 2B — LLM handles intent in-context at response generation.
- Re-validating against PROD before code lands. The eval is local; prod validation happens post-deploy via the smoke test.
- Force-rebuilding existing chunks. The backfill script populates / updates columns on existing rows without re-creating chunks.

## Audit trail (updated post-Stage-2B)

| Decision | Gap analysis source | External research | Status post-Stage-2B |
|---|---|---|---|
| Hybrid Stage 1 BM25+dense+RRF | [decision 1](maize-retrieval-gap-analysis-2026-05-22.md#refinement-1) | OptyxStack, VersionRAG, CoRank, RRF docs | Unchanged |
| Per-turn query rewriter | [decision 2](maize-retrieval-gap-analysis-2026-05-22.md#refinement-2) | H-RAG, RAGAboutIt, Alhena | Unchanged |
| Replaces `doc_type` as primary semantic axis | [decision 4](maize-retrieval-gap-analysis-2026-05-22.md#refinement-4-adjacent) | OptyxStack, ACORD | Direction stays; SHAPE revised in Stage 2B (per-TA configurable categories, not enum) |
| Structured doc classification | gap analysis decision 3 | ACORD, REIC, NVIDIA AI-Q | Per-TA-configurable `doc_category` string (Stage 2B) instead of 6-value enum or booleans |
| Solutions-as-supplementary + prompt rule | gap analysis "refined solutions-handling design" | User-specified pedagogy + literature-consistent mechanism | DROPPED in Stage 2B — solutions-leak isn't a real failure mode; pedagogy already in prompt |
| Intent classifier (`is_solution_request`, etc.) | gap analysis decision 3 | REIC, NVIDIA AI-Q | DROPPED in Stage 2B — LLM handles intent in-context |
| Defer CoRank Stage 2.5 | [decision 5](maize-retrieval-gap-analysis-2026-05-22.md#defer-1) | Empirical-not-supported (yet) | Unchanged |
| Defer CDE | [decision 6](maize-retrieval-gap-analysis-2026-05-22.md#defer-2) | Research's own staging recommendation | Unchanged |
