# Retrieval Refactor — Gap Analysis (2026-05-22)

**Purpose.** Before writing the Phase A implementation plan, formally cross-check the 6 research-derived refinements against (a) the empirical baseline scorecard and (b) the current code state. Decide which refinements to actually ship in Phase A v1 vs defer or drop.

**Inputs.**
- Research output: [maize-retrieval-redesign-plan.md](maize-retrieval-redesign-plan.md) (the parked plan, with 6 refinements).
- Empirical baseline: [eval/baseline_scorecard_pre_refinement.md](../eval/baseline_scorecard_pre_refinement.md) (64 rows on local ECON S1117).
- Per-row pass/miss data: `eval/baseline_run.log` (gitignored; quoted inline below).
- Current code: `src/retriever.py` (analyze_query @ line 1110, llm_rerank @ line 674, retrieve_context @ line 1668, contextualize_query @ line 1417).

---

## Key empirical finding the abstract scorecard hides

**`econometrics quiz 2` wins top-1 for the majority of "problem set 2" / discrimination-style queries** — including 3 of 3 Type A rows, all 3 working_pset rows, multiple Type E and synthetic Type D rows. This wasn't visible in the aggregated 50% / 40% / etc. numbers; it's only visible per-row in the log.

DB state explains why:
| Doc | doc.doc_type | doc.assignment_number | chunk.doc_type | chunk.assignment |
|---|---|---|---|---|
| `econ117_pset02_2025B_with_table` | homework | 2 | homework | 2 |
| `econometrics quiz 1` | homework | 1 | homework | 1 |
| `econometrics quiz 2` | homework | 2 | **other** | 2 |
| `Lab2` | other | 2 | other | 2 |
| `pset2_solutions-1` | homework | 2 | homework | 2 |

The LLM auto-classifier labeled quizzes as `doc_type='homework', assignment_number=N` (quiz N → "homework N"). For quiz 1 the chunk metadata is now sync'd (the user must have edited it via admin UI post-Bug-2-fix); for quiz 2-5 the chunks still carry the legacy `doc_type='other'`. Either way, the docs are misclassified and the retrieval system pulls them into the wrong candidate pools — sometimes via the SQL filter on chunks (when sync'd), sometimes via the hybrid fallback's doc-level metadata lookup (always).

**This is a Type A failure mode that's worse than the prod logs suggested.** Quiz N ≈ Lab N ≈ PS N in the system's eyes. Dropping the hard filter doesn't fully fix it because doc-level metadata is still consulted by the hybrid fallback path.

The gap analysis below treats this as the dominant signal.

---

## Matrix 1 — Per-refinement decision

| # | Refinement | Failure type(s) addressed | Empirical signal | Current code state | Cost / risk | Decision |
|---|---|---|---|---|---|---|
| 1 | **Hybrid Stage 1: BM25 + dense + RRF, replacing the fuzzy-filename-and-doc_type SQL pre-filter** | A, C, partial B | A: strong (per-row econ-quiz-2 dominance); C: strong (0% hit, fuzzy filename picks Quiz 2 solutions for "quiz 1"); B: weak (100% locally) | `analyze_query` (`src/retriever.py:1110-1209`) extracts doc_type + assignment via regex + fuzzy filename match → hard SQL filter. Reranker prompt (`llm_rerank` @ line 674) sees only `file_name` + text preview — no `doc_type` / `doc_role` / `assignment_number` surfaced | Medium: new tsvector column + GIN index, query path change, BM25 weight tuning, RRF code. Roughly 200-300 LOC + migration | **SHIP v1.** Type A and Type C together account for 13 in-corpus rows; the current SQL+fuzzy approach is responsible for the worst empirical pattern in the baseline. |
| 2 | **Per-turn query rewriter (decontextualizer) + per-turn re-retrieval** | E (cache anchor), multi-turn working continuations | Weak locally (Type E at 86% under cold-cache; multi-turn working at ~50%); strong from prod logs (the rewriter is what eliminates cache anchoring) | No standalone rewriter. `contextualize_query` (`src/retriever.py:1417`) classifies intent + rewrites query, but cache anchoring (session-level) uses turn-1's hybrid_doc as the cache key | Low-medium: one new LLM call per turn (~100-200ms), small prompt to write, hook into existing per-turn flow. ~50-100 LOC | **SHIP v1.** Even though local eval can't validate (cold cache), the prod failure mode is real and this is the smallest fix. Side benefit: makes multi-turn working cases more robust. |
| 3 | **`doc_role` field (problem/solution/lecture/syllabus/reference) + structured intent classifier + solutions-as-supplementary** | D, partial E | D: strong (40% leak; per-row shows solutions-doc top-1 on most D rows); E: weak | No `doc_role` exists. Reranker prompt doesn't distinguish problem vs solution chunks. Response_generator has no "don't divulge" rule for solutions | Medium: new schema column (+migration), auto-classification at upload (LLM call), intent classifier (LLM call per turn, ~100ms), updated rerank prompt, updated `BASE_INSTRUCTIONS` prompt rule. ~200-300 LOC | **SHIP v1.** This is the load-bearing fix for Type D and aligns with the user's refined design (solutions used as reference, never divulged). |
| 4 | **CoRank-style Stage 2.5 coarse rerank on compact doc-features** | All (noise reduction before fine rerank) | None directly — derived from literature, not from per-row data | Currently single rerank (LLM-rerank on top-20 chunks). Rerank latency is the dominant cost: 11-17s per query in the baseline. Adding another rerank stage could double that | Medium-high: new prompt, another LLM call (~80-200ms), tuning. Mostly compounds latency, which is already the slow leg | **DEFER.** The empirical baseline doesn't justify the latency cost. Watch the post-refactor scorecard after refinements 1+3 land — if precision is still low we can add this in v2. |
| 5 | **Contextual Document Embeddings (CDE) replacing text-embedding-3-small** | B (Roman num siblings) | None — Type B is 100% locally | Embeds with text-embedding-3-small via OpenAI. Stored in pgvector | High: replace embedding model, re-embed entire corpus (one-off but heavy), swap pgvector dim if different, manage a new model dependency | **DEFER.** Research itself recommended deferral; local eval confirms. Adopt only if hybrid leaves Type B residuals in prod. |
| 6 | **Eval methodology before implementation** | All (verification) | This IS the eval | Done | Zero | **✅ DONE.** Baseline scorecard exists; we'll re-run after the refactor and compare. |

---

## Matrix 2 — Per-failure-type fix mapping (sanity check)

For each failure type from the baseline, work backwards: which refinement(s) actually fix it?

| Failure | Baseline | Per-row dominant failure pattern | Fix(es) from above | Smallest viable fix? |
|---|---|---|---|---|
| **A — Lab vs PS** | 50% correct | `econ-quiz-2` wins top-1 for "problem set 2" queries — the LLM-mis-classified quiz docs share the homework+assignment metadata. Reranker can't disambiguate from text alone | Refinement 1 (drop hard filter + hybrid Stage 1) | YES if we also fix the metadata mis-classification. Adding `doc_role` (refinement 3) makes the disambiguation explicit. |
| **B — Roman num siblings** | 100% (passes locally) | n/a locally; prod showed "extra problems I" vs "II" confusion driven by fuzzy filename tokenization | Refinement 1 (hybrid) addresses this in theory | Local doesn't validate; ship 1 anyway since it addresses other types. CDE (5) deferred. |
| **C — Lookalike unrelated** | 0% correct, 57% forbidden | `Quiz 2 solutions` returned for "quiz 1" queries — fuzzy filename matcher picks lexically similar wrong files | Refinement 1 (hybrid + content-anchored Stage 1 replaces fuzzy filename matcher) | YES. The fuzzy filename matcher in `analyze_query` is the proximate cause. Hybrid Stage 1 replaces it cleanly. |
| **D — Solutions leak** | 40% correct, 40% forbidden | reranker treats solution chunks as candidates equally with problem chunks. Synthetic Type D rows show same pattern with `econ-quiz-2` (which also has the auto-classified `doc_type=homework`) | Refinement 3 (doc_role + intent + supplementary tagging) | YES. Doc_role is the cleanest discriminator. |
| **E — Cache anchor** | 86% misleading | Local eval doesn't trigger cache anchoring (cold cache per row). Prod failures are real and well-documented | Refinement 2 (per-turn rewriter + re-retrieval) | YES. Validated by prod, not the eval. |
| **Working** | 65% | Multi-turn continuations miss because cold cache loses T1's context. Some single-turn working rows also drift (e.g., `working_pset_*` hit by the quiz-2 dominance) | Refinement 2 helps multi-turn. Refinements 1 + 3 help single-turn working cases that share the doc-classification bug | Together. |

---

## Adjacent decision needed (not in the original 6 refinements but surfaced by the per-row data)

**The auto-classified `doc_type` is wrong for quizzes.** The LLM at upload time labels `econometrics quiz N` as `doc_type='homework', assignment_number=N`. This is the proximate cause of the worst Type A pattern. Refinements 1+3 work AROUND it (drop reliance on this metadata for retrieval), but the LLM auto-classification is still emitting bad labels that the UI surfaces.

Two options:
- **Option α (cheap):** Tighten the auto-classification prompt to distinguish quizzes from homework. Add "quiz" as a possible doc_type, OR more explicitly tell the LLM "if the document is a quiz, label it as `exam` not `homework`."
- **Option β (cleaner, aligned with refinement 3):** Replace `doc_type` as the primary semantic axis with `doc_role` (problem/solution/lecture/syllabus/reference). `doc_type` stays around for backward-compat but the auto-classifier emits `doc_role` and that's what retrieval consults. Quizzes get `doc_role='problem'` (which is correct).

**Recommendation:** Option β, bundled with refinement 3. We're already adding `doc_role`; making it the primary semantic axis for retrieval (rather than `doc_type`) is the obvious move.

---

## Phase A v1 scope (the decision)

**Ship in v1 (3 of 6 refinements + 1 adjacent fix):**

1. **Drop hard SQL filter on doc_type + assignment_number** (the "fuzzy classifier wins for the wrong document" failure mode). Filter becomes a soft signal at the reranker level, not a hard pre-filter.
2. **Hybrid Stage 1: BM25 (over chunk_text) + dense (existing pgvector) with RRF.** Replaces both the hard filter and the fuzzy filename matcher in `analyze_query` as the document-routing layer.
3. **Per-turn query rewriter** (decontextualizer) + per-turn re-retrieval. Eliminates cache anchoring by construction.
4. **`doc_role` field + intent classifier + solutions-as-supplementary** with the response_generator prompt rule that solutions tagged `[REFERENCE / DO NOT DIVULGE]` inform but never get divulged.
5. **(Adjacent) `doc_role` becomes the primary semantic axis in retrieval.** Auto-classified at upload; `doc_type` stays for backward-compat but isn't load-bearing.

**Defer to v2 (after v1 verification):**

6. **CoRank Stage 2.5 coarse rerank.** Add only if v1's rerank precision is insufficient on the post-refactor scorecard.
7. **CDE embeddings.** Add only if Type B residuals appear in prod after v1 ships.

**Out of scope entirely (drop):**

- None. Both deferred items have a path back if empirically motivated.

## Estimated implementation surface for v1

- **Schema:** new `Document.doc_role` + `doc_role_provenance` + `bm25_tsvector` columns. One migration.
- **Indexing pipeline (`src/document_processor.py`):** auto-classify `doc_role` at upload (replace or supplement existing metadata extraction); build BM25 tsvector over content.
- **Retriever (`src/retriever.py`):** new pre-stage `rewrite_query()`; replace `analyze_query()`'s hard-filter regex + fuzzy filename matcher with `hybrid_doc_search()` (BM25 + dense + RRF); update `llm_rerank()` prompt to surface `doc_role`; new `classify_intent()` per turn; remove the hard-filter SQL path in `retrieve_context()`.
- **Response generator (`src/response_generator.py`):** new section in `BASE_INSTRUCTIONS` for the solutions-as-reference rule.
- **Migration of existing data:** auto-classify `doc_role` for already-indexed docs as a backfill task (one-shot).

Rough total: 600-900 LOC across 4-5 files + 1 migration + 1 backfill script. Probably 2-3 implementation sessions.

## What this analysis explicitly does NOT include

- Writing the implementation plan itself. That's the next session.
- Validating these decisions against PROD data. We're operating on local data; the eval may not catch all prod failures (e.g., Type B locally passes).
- Re-running the eval. The baseline stands.
- Touching Phases B or C of the parked plan.

## Verification of this analysis

After commit:
1. Document exists at `attached_assets/maize-retrieval-gap-analysis-2026-05-22.md` with all matrices populated.
2. Each refinement has an explicit ship/defer/drop decision with rationale.
3. The Phase A v1 scope is narrower than the original 6 refinements (3 of 6 + 1 adjacent), justified by data not assumption.
4. The "econ-quiz-2 dominance" finding is captured as the load-bearing empirical signal it actually is.
