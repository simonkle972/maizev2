# Maize TA — Phase A Retrieval Refactor Implementation Plan

**Created:** 2026-05-22 · **Grounded in:** [gap analysis 2026-05-22](maize-retrieval-gap-analysis-2026-05-22.md) which is grounded in [parked retrieval plan](maize-retrieval-redesign-plan.md). Read those first if you arrived here cold.

## Context

The Phase A v1 scope is fixed (per the gap analysis): drop hard SQL filter + hybrid Stage 1, per-turn query rewriter, `doc_role` + intent classifier + solutions-as-supplementary, and `doc_role` becomes the primary semantic axis for retrieval. CoRank Stage 2.5 coarse rerank + CDE embeddings are deferred. This document is the step-by-step plan to actually build it.

Goal: ship the refactor in a way that's (a) staged across discrete sessions so each lands cleanly, (b) feature-flagged so the legacy retrieval path remains available, (c) verifiable via the existing eval harness, and (d) rollback-able if something regresses.

## Scope reminder (one paragraph)

**Ship in v1:** new `doc_role` enum column + auto-classification at upload (role replaces `doc_type` as the primary semantic axis for retrieval); new `bm25_tsvector` column for hybrid Stage 1; new pre-stage `rewrite_query()` (per-turn decontextualizer); new `hybrid_doc_search()` replacing the regex+fuzzy-filename matcher in `analyze_query()`; updated `llm_rerank()` prompt surfacing `doc_role`; new per-turn `classify_intent()` upstream of retrieval; updated `BASE_INSTRUCTIONS` in `response_generator.py` for the solutions-as-reference rule.

**Defer to v2:** CoRank Stage 2.5 coarse rerank, CDE embeddings.

## Implementation stages

The work is staged into FIVE discrete sessions. Each lands a checkpoint that's individually shippable (or at least individually feature-flag-disable-able) without breaking the existing app. Order matters — later stages depend on earlier ones.

| Stage | What | Risk | Estimated effort |
|---|---|---|---|
| 1 | Schema + migration + Config flags | Low (additive, nullable columns + new env flag) | ~0.5 session |
| 2 | Indexing pipeline: doc_role auto-classification + BM25 tsvector | Medium (touches upload + reindex flow) | ~1 session |
| 3 | Retriever: pre-stage rewriter + hybrid Stage 1 (behind feature flag) | High (the load-bearing change) | ~1-1.5 sessions |
| 4 | Retriever: intent classifier + solutions-as-supplementary integration | Medium (depends on stage 3 + response_generator) | ~0.5-1 session |
| 5 | Response generator prompt rule + final integration + eval re-run | Low (mostly prompt + verification) | ~0.5 session |

Total: 3-5 sessions. Below, each stage in detail.

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

## Stage 4 — Intent classifier + solutions-as-supplementary integration

**Goal:** wire the doc_role-aware solutions handling into the new path. This is the Type D fix.

### Changes

#### 4.1 — New per-turn intent classifier

New function `classify_intent()` in [src/retriever.py](../src/retriever.py):

```python
def classify_intent(query: str, prior_turns: list) -> dict:
    """Cheap structured-intent classifier (gpt-4o-mini, ~100ms).

    Produces:
      - is_solution_request (bool): student explicitly asking for the answer
      - concept_or_problem ('concept' | 'problem' | 'both')
      - document_corrected_from_prior_turn (bool): explicit correction-pivot

    Grounded in: REIC (arxiv 2506.00210) + NVIDIA AI-Q intent-classifier
    pattern. See gap analysis decision 3.
    """
```

Called per turn in `_retrieve_v2` upstream of Stage 3. Passes result downstream as features (not as routing gates — the LLM rerank stage uses them in its prompt).

#### 4.2 — Solutions-as-supplementary

In `_retrieve_v2`'s chunk retrieval (Stage 2):
- Get top-20 chunks from the candidate docs as before.
- Tag chunks where `chunk.doc_role == 'solution'` with `retrieval_role = 'solution_reference'`.
- If `intent['is_solution_request']` is True, also let solution chunks be PRIMARY (no tag).

In `llm_rerank()` (~line 674) — update the prompt to surface `chunk['doc_role']` AND prefer `doc_role='problem'` chunks as top-1 unless `is_solution_request=true`.

In `chat_streaming.py:346-355` — extend the context-assembly block to handle the new `'solution_reference'` retrieval_role:

```python
primary = [c for c in chunks if not c.get('retrieval_role')]
teaching = [c for c in chunks if c.get('retrieval_role') == 'teaching_material']
solution_ref = [c for c in chunks if c.get('retrieval_role') == 'solution_reference']

parts = [f"[From: {c['file_name']}]\n{c['text']}" for c in primary]
if teaching:
    parts.append("[RELEVANT TEACHING MATERIAL FROM COURSE LECTURES]")
    parts.extend(f"[From: {c['file_name']}]\n{c['text']}" for c in teaching)
if solution_ref:
    parts.append("[REFERENCE / DO NOT DIVULGE — solution material for your guidance only]")
    parts.extend(f"[From: {c['file_name']}]\n{c['text']}" for c in solution_ref)

# sources only includes primary chunks (the chat citation list)
sources = list(dict.fromkeys(c['file_name'] for c in primary[:8]))[:3]
```

**Grounding:** Mirror the existing `teaching_material` retrieval-role pattern (already proven to work). The `solution_reference` role is conceptually identical — supplementary chunks with a special prompt-context tag. Per gap analysis decision 3 + the user's refined solutions-handling design.

### Verification (Stage 4)

1. With `RETRIEVAL_V2_ENABLED=true`: rerun the eval. Type D `forbidden_hit` (interpreted as "primary_is_forbidden" — was top-1 a solution doc when student is solving?) should drop from 40% to ≤10%.
2. Spot-check: "I need help with problem 2 from PS3" → top-1 PRIMARY is `econ117_pset03` (problem doc; or in our local case, also `not_in_corpus` → solutions doc may appear as supplementary but not as PRIMARY). For a row where the problem doc IS in the corpus, primary should never be a solution.
3. Spot-check: "Show me the answer key for PS2 question 1" → intent classifier flips `is_solution_request=true` → solutions doc allowed as primary.

### Risks + rollback (Stage 4)

- Risk: Intent classifier mis-classifies a legitimate help request as `is_solution_request=true`, leaking solutions. Mitigation: design the few-shot prompt CONSERVATIVELY — only flip to true on explicit signals ("show me the answer", "what's the solution"). Default is false.
- Rollback: flip `RETRIEVAL_V2_ENABLED=false`. Legacy path doesn't have the intent classifier or solutions-as-supplementary plumbing — it's lossy but safe.

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

## Critical files (all stages)

| Stage | File | Purpose |
|---|---|---|
| 1 | [models.py](../models.py) | Document + DocumentChunk model additions |
| 1 | [migrations/versions/](../migrations/versions/) | New migration |
| 1 | [config.py](../config.py) | Feature flag + RRF_K + STAGE_1_TOP_K_DOCS |
| 2 | [src/document_processor.py](../src/document_processor.py) | classify_doc_role + bm25_tsvector population |
| 2 | scripts/backfill_doc_role_and_bm25.py | NEW one-shot backfill |
| 2 | [professor.py](../professor.py) + [app.py](../app.py) | Add doc_role to PATCH metadata sync |
| 3 | [src/retriever.py](../src/retriever.py) | rewrite_query + hybrid_doc_search + _retrieve_v2 branch |
| 4 | [src/retriever.py](../src/retriever.py) | classify_intent + solutions-as-supplementary in chunk retrieval |
| 4 | [src/chat_streaming.py](../src/chat_streaming.py) | Context-assembly extension for solution_reference role |
| 5 | [src/response_generator.py](../src/response_generator.py) | BASE_INSTRUCTIONS new section |
| 5 | [eval/run_eval.py](../eval/run_eval.py) | primary_is_forbidden metric |

## What this plan explicitly does NOT include

- Touching Phases B or C of the parked plan. They're parked because Phase A's per-turn rewriter likely subsumes them.
- Implementing CoRank Stage 2.5 or CDE. Deferred per the gap analysis with explicit triggers for when to revisit.
- Adding `doc_role` UI controls in the admin or professor manage_ta pages. Auto-classification + provenance is enough for v1; UI surfacing comes in v2 if needed.
- Re-validating against PROD before code lands. The eval is local; prod validation happens post-deploy via the smoke test.
- Force-rebuilding existing chunks. The backfill script populates new columns on existing rows without re-creating chunks — much faster + safer.

## Audit trail

| Decision | Gap analysis source | External research |
|---|---|---|
| Hybrid Stage 1 BM25+dense+RRF | [decision 1](maize-retrieval-gap-analysis-2026-05-22.md#refinement-1) | OptyxStack, VersionRAG, CoRank, RRF docs |
| Per-turn query rewriter | [decision 2](maize-retrieval-gap-analysis-2026-05-22.md#refinement-2) | H-RAG, RAGAboutIt, Alhena |
| doc_role + intent classifier | [decision 3](maize-retrieval-gap-analysis-2026-05-22.md#refinement-3) | ACORD, REIC, NVIDIA AI-Q |
| doc_role replaces doc_type as primary | [decision 4](maize-retrieval-gap-analysis-2026-05-22.md#refinement-4-adjacent) | OptyxStack, ACORD |
| Solutions-as-supplementary + prompt rule | gap analysis "refined solutions-handling design" | User-specified pedagogy + literature-consistent mechanism |
| Defer CoRank Stage 2.5 | [decision 5](maize-retrieval-gap-analysis-2026-05-22.md#defer-1) | Empirical-not-supported (yet) |
| Defer CDE | [decision 6](maize-retrieval-gap-analysis-2026-05-22.md#defer-2) | Research's own staging recommendation |
