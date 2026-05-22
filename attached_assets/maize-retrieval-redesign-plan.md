# Maize TA — Retrieval Redesign Plan (principled, not band-aid)

**Created:** 2026-05-14 · **Last updated:** 2026-05-22 (baseline scorecard landed + solutions-handling refined) · **Canonical reference** for the retrieval redesign work triggered by the ECON S1117 TA review.

This is the guiding document. It supersedes any working-plan churn at [.claude/plans/humble-mapping-biscuit.md](../../../.claude/plans/humble-mapping-biscuit.md), which gets overwritten between sessions. All substantive context — diagnosis, data, research, recommended architecture, execution state — lives here.

---

## Status snapshot

| Phase | What | Data | Research | Deepening | Eval baseline | Implementation plan | Implementation | Verification |
|---|---|---|---|---|---|---|---|---|
| **A** | **Retrieval redesign** (doc selection + multi-turn handling) | ✅ 44 rows | ✅ | ✅ REFINED (6 changes) | ✅ [scorecard](../eval/baseline_scorecard_pre_refinement.md) | ☐ **NEXT** | ☐ | ☐ |
| **B** | Doc switching / correction-pivots | — | — | — — likely absorbed by Phase A's query rewriter; re-evaluate post-A | — | — | — | — |
| **C** | Paste/cache mismatch invalidation | — | — | — — likely obsolete once per-turn retrieval re-runs; re-evaluate post-A | — | — | — | — |

### Baseline (CURRENT retriever) — captured 2026-05-22, 64 rows on local ECON S1117

| Failure type | n | correct_hit@5 | hard_neg_top1 | forbidden_hit |
|---|---:|---:|---:|---:|
| A (Lab vs PS) | 6 | 50% | 0% | 0% |
| B (Roman numeral) | 5 | 100% ⚠ | 0% | 20% |
| C (lookalike-unrelated) | 7 | **0%** | 57% | 57% |
| D (problem-vs-solutions) | 5 | 40% | 40% | 40% ⚠ |
| E (cache anchoring) | 21 | 86% ⚠ | 0% | 5% |
| working cases | 17 | 65% | 6% | 6% |

⚠ flagged numbers have interpretation caveats — see [scorecard interpretation notes](../eval/baseline_scorecard_pre_refinement.md):
- Type B 100%: local re-index produced cleaner metadata than prod's; prod may still exhibit Type B failures the local eval doesn't catch.
- Type E 86%: harness uses cold cache per row; doesn't reproduce the actual cache-anchoring failure mode (which the per-turn rewriter eliminates by construction anyway).
- Type D 40% forbidden_hit: under the refined solutions-handling design (below), this metric is too strict — solutions ARE allowed to retrieve, just not be cited as primary. Cleaner replacement metric `primary_is_forbidden` to be added in the post-refactor scorecard.

**Current next action:** write the Phase A implementation plan (separate planning session). Code changes touch: schema (`doc_role`, `doc_embedding`, `bm25_tsvector` cols on Document), indexing pipeline (doc-level embedding + BM25 tsvector + auto-classify doc_role), retriever (per-turn query rewriter + hybrid Stage 1 + Stage 2.5 coarse rerank + Stage 3 prompt-refinement + LLM-router escape hatch + structured intent classifier), and response_generator prompt (the "don't divulge solutions" rule, see refined design below).

**Why the order changed:** Phase A's research deepening (2026-05-21) surfaced six material refinements beyond the original recommended architecture. Most importantly, the per-turn query rewriter that came out of the deepening likely absorbs Phase B's mandate, and per-turn re-retrieval makes Phase C largely obsolete. So Phases B and C aren't just "deferred until after Phase A" anymore — they may be subsumed by Phase A's refined architecture. We'll know after verification.

### Solutions-handling design — refined 2026-05-22

The original Phase A design treated solutions docs as wholly forbidden from retrieval (Stage 3 default-excluded `doc_role=solution`). The user pushed back: **solutions docs are valuable as supplementary reference material the LLM can use to construct higher-quality Socratic guidance** — they shouldn't be wholesale filtered out. The pedagogical concern ("don't divulge the answer") is a PROMPT-LEVEL rule, not a retrieval-level filter.

Refined design:
- Stage 1/2/3 retrieval is allowed to bring solutions chunks into context alongside problem chunks.
- Solutions chunks are tagged `[REFERENCE / DO NOT DIVULGE]` in the prompt context.
- Source attribution (the visible "Sources" list) shows the PRIMARY doc only — the problem doc the student is solving. Solutions inform the response but don't get cited.
- The `response_generator` system prompt gets a new explicit rule: "You may have access to solution chunks tagged `[REFERENCE / DO NOT DIVULGE]`. Use them to guide your Socratic questions, validate the student's work, and construct higher-quality hints. Do not reveal the answer directly to the student."
- This mirrors the existing `supplementary_teaching` pattern in `src/retriever.py` — solutions become another category of supplementary context.

What this changes in the rest of this document:
- The Stage 3 reranker no longer "default-excludes doc_role=solution" — it default-prioritizes problem docs as PRIMARY while allowing solution chunks as supplementary.
- The intent classifier's `is_solution_request` signal becomes used to (a) decide whether solutions chunks should be promoted to primary (for legitimate verification requests like "let me check my answer") and (b) tune the prompt instruction wording. Default behavior — solutions stay supplementary, never primary.
- Doc-role metadata (problem / solution / lecture / syllabus / reference) is STILL required for this to work. The field stays in the model.

---

## Why this plan exists

The 15-row prod test on the ECON S1117 TA ([Maize QA Master [PROD] - ECON S1117 test.csv](Maize%20QA%20Master%20[PROD]%20-%20ECON%20S1117%20test.csv)) — our largest and most diverse TA so far (63 documents across interactive lecture slides, async lecture slides, problem sets, labs, quizzes, extra problems) — surfaced 9 problem rows out of 15. A 29-row follow-up test (also on ECON S1117) confirmed the failure modes generalize. The first draft of a triage plan proposed enumeration-of-special-cases fixes (add `lab` to a `doc_type` enum, special-case Roman numerals in filename matching, list negation patterns for correction-pivots).

The user correctly pushed back: that's exactly the brittle pattern the project has explicitly moved away from in prior work (substring → k-gram containment for paste detection, rigid intent rules → LLM-classified intent, regex problem-detection → contextualizer-driven). The right question isn't "which patterns do we add for the next collision" — it's "what's the principled sub-system that doesn't need patterns at all."

A second push-back came on 2026-05-21: **validation by walking the 44 prod rows on paper is itself enumeration.** If the architecture is genuinely cutting-edge research applied to a less-than-cutting-edge problem (course-document retrieval), it should be self-evidently sufficient — and the need to validate by instance-walking signals that the research isn't deep enough. So step 3 of the cycle was reframed as a research deepening pass.

This plan documents the diagnosis, the principled redesigns, the supporting data, the research (initial + deepened), the refined architecture, the eval methodology, and the execution state.

## The unifying observation

Most failures share a common architectural property: **the system is doing rigid symbolic/rule-based work where semantic/LLM-driven work would generalize naturally.**

- "Lab 2" vs "Problem Set 2" being conflated is a *rigid metadata enumeration* problem — both got tagged as `doc_type=homework, assignment=2`, and the SQL filter has no way to distinguish them. The principled answer isn't "add `lab` to the enum" (a new band-aid for the next collision), it's "stop using rigid metadata filters as the document-selection mechanism."
- "extra problems I" matching "extra problems II" is a *rigid filename tokenization* problem — the matcher strips Roman numerals as noise. The principled answer isn't "add Roman numerals to the tokenizer's not-noise list" (next collision: years? versions? letters?), it's "stop using rule-based filename matching when we have embeddings and LLMs sitting right there."
- Correction-pivots being missed is a *rigid intent classifier inputs* problem — the contextualizer doesn't see what was retrieved last turn, so it can't reason about "the system retrieved X and the student is now correcting it." The principled answer isn't "add 'not X' to a list of negation patterns," it's "give the contextualizer richer input so it can recognize correction-of-prior-retrieval as a first-class signal."

These three failure modes account for 8 of 9 problem rows. They share a single root cause: **we are coding around the LLM instead of letting the LLM be the LLM.** Every new edge case adds another rule; the next TA exposes another edge case. The cycle repeats.

## The execution discipline

For each Phase (A first, then B and C re-evaluated after A ships), we run a five-step cycle:

1. **Data-gathering** — ≥20-30 prod rows characterizing the target failure mode AND its regression risk. Non-negotiable.
2. **Research** — mine 2025-2026 RAG / retrieval / agent literature + mature production patterns. Identify the principled sub-system.
3. **Research deepening (replaces row-by-row "validation")** — go deeper on (a) failure-mode-specific production patterns, not generic best practices, and (b) evaluation methodology for retrieval improvements as its own field. Validation-by-enumeration is the same anti-pattern as fixing-by-enumeration. The prod rows belong in step 5, not as a pre-implementation gate.
4. **Eval set + implementation plan** — build the eval harness FIRST, then propose specific code changes. Smallest change that delivers the principled redesign.
5. **Implementation + verification** — re-run the actual prod test cases + eval set against the implementation before declaring done.

This matches the discipline used successfully for paste detection (substring → k-gram containment), topic switching (concept_lookup tightening), and Stage 2A reasoning_effort tuning.

**Durable behavioral framing for this cycle** lives in the auto-memory:
- [feedback_principled_redesigns.md](../../../.claude/projects/-Users-simonkleffner-Desktop-Maize-TA-Master-App-Dev-Maize-Blueprint-V2/memory/feedback_principled_redesigns.md) — default to principled redesigns over enumeration; run the five-phase cycle; validation deepens research, doesn't walk instances.
- [feedback_sample_size.md](../../../.claude/projects/-Users-simonkleffner-Desktop-Maize-TA-Master-App-Dev-Maize-Blueprint-V2/memory/feedback_sample_size.md) — ≥20-30 rows across multiple TAs/subjects before approving major changes.

---

## PHASE A — Retrieval redesign (FIRST)

**Target:** the document-selection sub-system *and* the multi-turn handling that surrounds it. **Addresses Failure Types A–E from the 44-row data.**

### Failure modes characterized from the 44-row data

Two test batches on ECON S1117 (15 initial + 29 doc-switching) yielded five durable failure types plus a working-case baseline:

- **Type A — "Lab N" vs "PS N" conflation** (Lab2 retrieved for "PS2"): session fdkCLBi616 T1, plus rows 7/8/9/12 from the first batch.
- **Type B — Filename matcher confuses sibling docs** ("extra problems I" vs "II"): session SiCBkT49QV T3, rows 13/14 from the first batch.
- **Type C — Filename matcher picks unrelated wrong file**: sessions 1QJt2q1OPN, fhBpYEX7vp ("quiz 1" → "Quiz 2 solutions"), GwDG1cavQr ("econometrics quiz 1" → "econ117-final-fall-2018-1").
- **Type D — Solutions doc retrieved instead of problem doc**: sessions CYsgc7lmpO, uKTUrCh-aQ, y81Fu8Rop6 ("PS3" → "econ117_pset03- 2025, solutions").
- **Type E — Once cached wrong, can't recover** (cascading from A–D): sessions SiCBkT49QV T2–T6, y81Fu8Rop6 T2–T4, x2MBKNO7uo T2–T4, 1QJt2q1OPN T2.

**Working cases (must NOT regress):** sessions wbYlK0EP_l (interactive lecture 3 → correct), x2MBKNO7uo T1 (pre-recorded lecture 3 → correct), qID1qBDQqV (lab 4 → correct), _WE1UMJhel (quiz 3 → correct), plus rows 2/5/6/10/11 from the first batch.

### Current mechanism + what's wrong

`analyze_query()` in [src/retriever.py](../src/retriever.py) extracts structured filters (`doc_type`, `assignment_number`, `filename_filter` via fuzzy match) from the student's query. Those become a SQL `WHERE` clause on pgvector retrieval. Filename matching uses a tokenized-fuzzy comparison that strips noise.

The structured extraction is brittle (Lab 2 vs PSet 2 collide on `doc_type='homework', assignment_number=2`), the filter is binary (pass/fail with no graceful degradation), the filename matcher fails on edge cases (Roman numerals, similar names, abbreviations, typos), and the conversational cache anchors to whichever document was retrieved on turn 1 — so a mistake on turn 1 cascades through the entire session.

### Three candidate Options (historical — the principled redesigns we evaluated)

1. **Drop hard filters entirely; trust the reranker.** Cheap; relies on Stage 3 alone for disambiguation.
2. **LLM document router as a pre-step.** gpt-4o-mini decides which 1-3 documents are the intended source; constrain vector search to those.
3. **Embedding-based document matching.** Pre-compute doc-level embeddings; retrieve candidate docs by similarity; chunk-level retrieval within.

The initial research output (preserved below) recommended a hybrid of Option 3 + Option 1 + Option 2 in roles. The 2026-05-21 research deepening **refined** that recommendation — see the next section for the post-deepening architecture.

### Initial research output — preserved (literature converges on hierarchical two-stage)

The seven initial research questions all point in the same direction: **hierarchical two-stage retrieval (document-level coarse → chunk-level fine) is the dominant production pattern for multi-document corpora at our scale.** Frameworks like HiChunk, MacRAG, and the broader "Hierarchical RAG" pattern all decompose retrieval into:

1. **Stage 1 — Document-level (coarse).** Identify the right document(s) using doc-level embeddings or LLM routing.
2. **Stage 2 — Chunk-level (fine).** Vector search within the selected documents.
3. **Stage 3 — Reranking.** Cross-encoder LLM rerank over the fine results.

This shape was the basis for the original recommended architecture. The deepening pass kept the shape and refined the components.

---

### Research deepening output (2026-05-21) — REFINED architecture

The deepening pass targeted two dimensions: (1) failure-mode-specific production patterns (not generic RAG best practices) and (2) evaluation methodology as its own discipline. Both dimensions are summarized below; details + citations are in the [Sources](#sources) section.

#### Dimension 1 — Failure-mode-specific findings

- **Type A (sibling-bucket collisions).** Hard metadata filters are a recognized RAG-reliability anti-pattern; hard filters should be reserved for access control / tenant isolation / forbidden scopes. Business-dimension metadata (doc_type, assignment_number) should "bias ranking first and relax only when coverage is weak." CoRank's three-stage pipeline (coarse metadata-aware rerank → fine full-text rerank) is the canonical refinement. → **CONFIRMS "drop hard filters; metadata as soft signal"** + suggests adopting CoRank-style three-stage explicitly.
- **Type B (Roman-numeral / version-letter sibling discrimination).** **Pure dense embeddings collapse** because near-duplicate docs share most semantic content — discriminative signal is in rare tokens (numerals, version letters). VersionRAG (Oct 2025) explicitly identifies this as an underexplored RAG failure mode. Canonical solution: **hybrid BM25 + dense with RRF**, OR ColBERT-style late interaction, OR Contextual Document Embeddings (CDE, ICLR 2025) which train against hard negatives from corpus context. → **REFINES Stage 1 from pure dense to hybrid BM25(content) + dense with RRF.**
- **Type C (lookalike-but-unrelated filename matches).** Filename overlap produces literal false positives; semantic overlap produces over-generalized ones. Resolution: filename/title must be a *feature among many*, never dominant. Content-based Stage 1 doc embeddings defuse the failure naturally. → **CONFIRMS content-based Stage 1**; **REFINES** the BM25 half of the hybrid to run over content, not filename (otherwise we'd re-introduce Type C while fixing Type B).
- **Type D (problem-vs-solutions corpora).** No paper directly solves "retrieve from problems but never from solutions based on student intent" as a first-class concern. Adjacent literature (REIC for intent classification, ACORD for legal clause-role retrieval, metadata-driven RAG for finance) converges on: (i) explicit `doc_role` field per document (`problem` / `solution` / `lecture` / `syllabus` / `reference`), (ii) cheap intent classifier on every query producing `is_solution_request` and similar signals, (iii) reranker default-excludes solutions unless intent explicitly opts in. → **REFINES** the LLM-router escape hatch into structured intent classification on every turn.
- **Type E (cache anchoring across turns).** Recognized failure pattern ("context collapse" / "anchor problem"). Canonical fix is **per-turn query rewriting / decontextualization** producing a self-contained query, NOT cached-doc anchoring. H-RAG's pattern is canonical: lightweight LLM rewriter returns query unchanged when standalone, decontextualizes when coreferent. → **REFINES** the architecture by adding a required pre-Stage-1 query rewriter and making per-turn re-retrieval the default; **likely absorbs Phase B** and **likely obsoletes Phase C**.
- **Educational-RAG-specific writeups** (Khanmigo, CS50.ai, etc.) are thin on retrieval-architecture detail. Academic ed-RAG papers (GraphRAG on math textbooks, entity-linking for educational platforms) converge on graph/hierarchical approaches consistent with our shape. AI-TA (arxiv 2311.02775) is the closest published predecessor and worth a direct read for implementation specifics. → **No additional refinement from ed-tech sources**; general retrieval literature is load-bearing.

#### Dimension 2 — Evaluation methodology findings

- **BEIR methodology** (NDCG@10, Recall@k, zero-shot, hard-negative-mined test sets) is the standard. Not the BEIR datasets themselves (general-domain) but the methodology. Adapt as: a Maize-internal mini-BEIR built from prod queries.
- **RAGAS** — context precision + context recall + faithfulness + answer relevancy. Lowest-friction integration; LLM-as-judge under the hood. → **Adopt for retrieval-stage evaluation in CI.**
- **DeepEval** — 50+ metrics, native pytest/CI integration. → **Adopt for regression-test wiring** (fail builds on regressions to known-good rows).
- **RAGChecker** (Amazon Science 2024) — claim-level entailment checking; correlates better with human judgments than RAGAS in their evaluation. Heavier integration. → **Defer to post-Phase-A**; layer in when investigating generation-side regressions.
- **TruLens** — feedback functions + tracing. Useful for unified eval+observability. → **Optional**; RAGAS+DeepEval cover the core need.
- **Hard-negative mining.** The canonical pattern for sibling-collision regression coverage: for each correct (query, doc) pair, explicitly enumerate hard negatives that look superficially relevant. For Maize: for query "ECON1117 PS3 Problem 2," the correct doc is PS3 and the hard negatives are PS3-solutions (Type D), Lab 3 (Type A), Extra Problems III (Type B), and the 2018 final (Type C).
- **Regression test corpus design.** Production consensus: 30-50 real production queries from logs, each labeled with correct doc(s), 2-5 hard-negative confusables, expected pedagogical response style, and forbidden retrievals. Re-run on every retrieval change in CI; fail builds on regressions.

→ **Recommendation:** build **Maize Eval Set v1** (the 44 ECON S1117 prod rows + ~50 hand-constructed hard-negative rows targeting Types A–E) and wire **RAGAS** (retrieval scoring) + **DeepEval** (CI integration) **before** the architecture change ships. Per-failure-type Recall@5 is the headline metric.

#### Refined recommended architecture

**Pre-Stage — Per-turn query rewriter (NEW, required, addresses Type E).** Cheap LLM call (gpt-4o-mini, ~100ms) decontextualizes the current student turn into a self-contained query using prior turn(s) + the prior turn's retrieved doc. If the query is already standalone, return it unchanged. This runs every turn; there is no Stage-1 caching.

**Stage 1 — Hybrid document-level pre-retrieval (REFINED).**
- **Sparse half:** BM25 over each document's *content* (concatenated chunk text or doc summary). Captures rare-token signal — numerals, version letters, assignment numbers, course codes — that dense embeddings collapse.
- **Dense half:** doc-level embeddings of `display_name + content_title + first ~500 chars`. Captures semantic content.
- **Fusion:** Reciprocal Rank Fusion (RRF) merges sparse + dense rankings. Documents both retrievers like rise; documents only one likes get downweighted.
- **Optional future:** swap the dense half for [Contextual Document Embeddings (CDE, ICLR 2025)](https://arxiv.org/html/2410.02525v1) — drop-in replacement explicitly trained on corpus-internal hard negatives. Phased: ship hybrid first, adopt CDE only if Type B residuals persist.
- **Confidence handling:** if top-1 RRF score significantly exceeds top-2, proceed with top-1. Otherwise top-5 advance to Stage 2.

**Stage 2 — Chunk retrieval (existing, but constrained).** pgvector cosine over chunks of the candidate docs only. Top-20 chunks. NO hard SQL filters on `doc_type` / `assignment_number` — they drop from this layer entirely.

**Stage 2.5 — Coarse metadata-aware rerank (NEW, CoRank-style).** Compact representation per candidate chunk: filename + section heading + doc_type + doc_role + first sentence. Fast cross-encoder or small-LLM rerank to top-8. Reduces noise into Stage 3.

**Stage 3 — Fine-grained LLM rerank (existing, prompt-refined).** gpt-5.2 rerank, with the prompt explicitly surfacing `file_name`, `display_name`, `doc_type`, and **`doc_role`** for each chunk. Reranker is **intent-aware**: it prioritizes the problem doc as PRIMARY (the doc that gets cited in "Sources"). Solution chunks are NOT excluded — they're allowed into context as supplementary reference, tagged `[REFERENCE / DO NOT DIVULGE]` so the response_generator's prompt rule knows to use them for guidance without revealing them. If the intent classifier produces `is_solution_request=true` (rare, e.g., "let me check my answer"), the rule relaxes and solutions can be promoted to primary.

**Structured intent classifier (NEW, runs per turn).** Cheap classification (few-shot prompt to gpt-4o-mini or fine-tuned small model) producing structured signals: `is_solution_request` (rare, pedagogically dangerous), `concept_vs_problem`, `document_corrected_from_prior_turn` (subsumes Phase B's concern), and similar. Feeds Stage 3 as features, not as a routing gate.

**LLM document router (Option 2, narrowed escape hatch).** Still fires when Stage 1 RRF returns ambiguous results (multiple docs above similarity threshold, no clear winner). Less load-bearing than originally framed — most disambiguation now happens via hybrid + intent + reranker — but useful for the genuinely-ambiguous cases.

#### How the refined architecture handles each Failure Type

- **Type A (Lab vs PS conflation).** Hard filters dropped. Stage 1 hybrid: BM25 picks up "lab" vs "PS" lexically, dense picks up content differences (LLN simulation vs discrimination). RRF resolves cleanly. ✓
- **Type B (Roman numeral siblings).** Hybrid's BM25 half preserves "II" vs "I" as rare-token signal. Dense alone would collapse; hybrid + RRF does not. ✓
- **Type C (lookalike-but-unrelated).** BM25 runs over content (not filename), so "econometrics quiz 1" doesn't lexically match "econ117-final-fall-2018-1" on filename tokens; dense embedding sees content divergence. RRF rewards consensus, both retrievers agree. ✓
- **Type D (problem vs solutions).** Doc-role field is required. Intent classifier produces `is_solution_request=false` by default. Stage 3 reranker promotes the problem doc as PRIMARY for citation while solution chunks ride along as supplementary reference (tagged `[REFERENCE / DO NOT DIVULGE]` in prompt context). Student is cited the problem doc; the LLM uses solutions to construct higher-quality Socratic guidance WITHOUT revealing the answer. If they explicitly ask "show me the solution," intent flips and solutions can be promoted to primary. ✓
- **Type E (cache anchoring).** Per-turn query rewriter + per-turn Stage 1 re-retrieval. There is no turn-1 cache to anchor to. If turn N's rewritten query is "Problem Set 2 problem 1," Stage 1 picks PS2 regardless of what was retrieved on turn 1. ✓

#### Question 7 — user-input metadata: KEEP, downgrade to soft signal (CONFIRMED)

The deepening confirmed the original verdict. Metadata is useful as a reranker feature, harmful as a hard pre-filter. UI control stays; the SQL filter that uses it gets dropped. The deepening adds one thing: **doc_role is required, not optional** — needed for Type D resolution. Auto-classified at upload (cheap LLM call against document content), professor override available, **provenance tracked** (auto-confidence vs professor-override) so debugging wrong role labels is tractable.

#### Cost / latency profile (refined)

Compared to the original "pure dense Stage 1" proposal:

| Component | Latency | Notes |
|---|---|---|
| Pre-Stage query rewriter | +100ms | gpt-4o-mini, runs every turn. Cached when query is already standalone. |
| Stage 1 hybrid (BM25 + dense + RRF) | ~30ms | BM25 index lookup + pgvector lookup + RRF merge. |
| Stage 2 chunk retrieval | unchanged | constrained to candidate docs. |
| Stage 2.5 coarse rerank | +80ms | small-LLM or cross-encoder over top-20 chunks. |
| Stage 3 fine rerank | unchanged | gpt-5.2, prompt-refined. |
| Intent classifier | +60ms | gpt-4o-mini few-shot, runs every turn. |
| LLM router (escape hatch) | ~250ms when fires | only on ambiguous Stage 1 results, ~10% of queries. |
| **Net added per query** | **~250-300ms** | hybrid Stage 1 is the cheapest piece; the rewriter + intent + coarse rerank account for most of it. |

Storage: Document gets a `doc_embedding VECTOR(1536)`, `doc_role VARCHAR`, `doc_role_provenance JSONB`, and a `bm25_content_tsvector` column (or external BM25 index — TBD at implementation). ~10KB/doc all-in. Trivial.

#### Implementation footprint preview (refined)

- **Schema** ([models.py](../models.py)):
  - `Document.doc_embedding VECTOR(1536)` — for the dense half of Stage 1.
  - `Document.doc_role VARCHAR` — enum: problem / solution / lecture / syllabus / reference / other.
  - `Document.doc_role_provenance JSONB` — `{source: auto|professor, confidence: 0.92, classified_at: ...}`.
  - `Document.bm25_tsvector TSVECTOR` — for BM25 over content, OR external BM25 index (decide at implementation).
  - Migration covering all of the above.
- **Indexing pipeline** ([src/document_processor.py](../src/document_processor.py)): after chunk extraction, (a) compute doc-level summary (display_name + content_title + first ~500 chars), (b) embed it, (c) compute BM25 tsvector over full content, (d) auto-classify doc_role with confidence + provenance.
- **Retriever** ([src/retriever.py](../src/retriever.py)):
  - Add pre-Stage query rewriter.
  - Replace hard-filter SQL with Stage 1 hybrid BM25+dense+RRF.
  - Add Stage 2.5 coarse rerank.
  - Refine Stage 3 prompt: surface filename + display_name + doc_type + doc_role.
  - Remove `doc_type`/`assignment_number` filtering from `analyze_query()`.
- **Intent classifier** (new function in retriever.py or new module): per-turn structured classification.
- **LLM router** (new function): conditional, narrowed to genuinely-ambiguous Stage 1.
- **Eval harness** (new dir, e.g. `eval/`): Maize Eval Set v1 JSONL + RAGAS integration + DeepEval pytest hookup.

Meaningful change but contained — touches retrieval architecture, indexing, schema, and adds a new eval harness. Reversible by feature flag if needed. Roughly 2x the implementation surface of the original (pre-deepening) plan, because the per-turn rewriter + intent classifier + hybrid + coarse rerank + eval harness are all new pieces. But the alternative is shipping an architecture the literature flags as insufficient for our specific failure types.

### Phase A — Execution log

- **Data-gathering:** ✅ Complete (44 rows across two ECON S1117 test batches; Failure Types A–E characterized; 9 working-case sessions identified).
- **Research (initial):** ✅ Complete. Hierarchical two-stage shape established.
- **Research deepening (2026-05-21):** ✅ Complete. Verdict: REFINED. Six material refinements integrated above. Architecture is research-complete; ready for eval-set construction.
- **Maize Eval Set v1:** ☐ **NEXT** — JSONL with 44 prod rows + ~50 hand-constructed hard-negative rows targeting Types A–E. Each row: (query, prior_turns, correct_doc_ids, hard_negative_doc_ids, forbidden_doc_ids, expected_intent_signals).
- **RAGAS + DeepEval wiring:** ☐ pending eval set.
- **Implementation plan:** ☐ pending eval-set + framework wiring.
- **Implementation:** ☐ pending implementation plan.
- **Verification:** ☐ pending implementation. 44 prod rows + Eval Set v1 become the regression test, gated in CI by DeepEval.

---

## PHASE B — Doc switching / correction-pivots (likely absorbed by Phase A)

**Originally:** add the prior turn's retrieved doc to the contextualizer input so it can reason about correction-pivots ("not asking about Lab 2, asking about PS 2").

**Status after Phase A deepening:** the per-turn query rewriter is given prior turns AND the prior turn's retrieved doc as context. When the student says "not asking about Lab 2, asking about PS 2," the rewriter produces a standalone query "Problem Set 2 [explicit correction from earlier wrong retrieval of Lab 2]." Stage 1 re-runs and picks PS 2. The structured intent classifier additionally produces `document_corrected_from_prior_turn=true`, which the reranker can use.

So Phase B's mandate is structurally absorbed by Phase A's refined architecture. We'll confirm in verification — if correction-pivots still fail after Phase A ships, Phase B re-opens with fresh data-gathering.

### Phase B — Execution log

- **Status:** likely absorbed by Phase A's query rewriter + structured intent classification. Re-evaluate post-Phase A using fresh prod data.

---

## PHASE C — Paste/cache mismatch invalidation (likely obsolete)

**Originally:** when paste detection identifies a different document than the cached one, invalidate the cache.

**Status after Phase A deepening:** the refined architecture has **no turn-1 cache to invalidate**. Stage 1 re-runs every turn on the rewritten query. Paste detection's doc identification can flow into Stage 1's candidate set (or directly into the reranker) but no cross-check against a stale cache is needed — there is no stale cache.

### Phase C — Execution log

- **Status:** likely obsolete. Re-evaluate post-Phase A — if there's a residual misattribution pattern the per-turn architecture doesn't solve, Phase C re-opens with fresh data-gathering.

---

## Adjacent observation — Quant SETUP variance (no Phase; need more data)

Rows 1 vs 2 (same query, different answers) and row 4 (filter matched but no chunks survived) each have N=1 evidence. Per [feedback_sample_size.md](../../../.claude/projects/-Users-simonkleffner-Desktop-Maize-TA-Master-App-Dev-Maize-Blueprint-V2/memory/feedback_sample_size.md), don't ship from tiny samples. Defer until the pattern recurs across more TAs or sessions.

---

## Cross-references

**Sibling parked plans** (other in-progress workstreams; all stable):
- [maize-latency-optimization-plan.md](maize-latency-optimization-plan.md) — staged latency program (Stages 1A/1D/2A; instrumentation-first discipline).
- [maize-socratic-dialogue-plan.md](maize-socratic-dialogue-plan.md) — pedagogy prompts for quant + qualitative SETUP, validated against 31 prod rows.
- [maize-topic-switching-plan.md](maize-topic-switching-plan.md) — contextualizer's `concept_lookup` tightening so structured→conceptual switches classify correctly.

**Durable behavioral memory:**
- [feedback_principled_redesigns.md](../../../.claude/projects/-Users-simonkleffner-Desktop-Maize-TA-Master-App-Dev-Maize-Blueprint-V2/memory/feedback_principled_redesigns.md)
- [feedback_sample_size.md](../../../.claude/projects/-Users-simonkleffner-Desktop-Maize-TA-Master-App-Dev-Maize-Blueprint-V2/memory/feedback_sample_size.md)

**Source data:**
- [Maize QA Master [PROD] - ECON S1117 test.csv](Maize%20QA%20Master%20[PROD]%20-%20ECON%20S1117%20test.csv) — initial 15-row test.
- The 29-row doc-switching follow-up batch (also ECON S1117) appended into Phase A's 44-row dataset.

**Code anchors** (for when Phase A enters implementation):
- [src/retriever.py](../src/retriever.py) — `analyze_query()`, `retrieve_context()`, `llm_rerank()`, `contextualize_query()`.
- [src/document_processor.py](../src/document_processor.py) — indexing pipeline (where doc-level embedding + BM25 tsvector + doc-role classification get added).
- [models.py](../models.py) — `Document` model (where `doc_embedding`, `doc_role`, `doc_role_provenance`, `bm25_tsvector` columns are added).

**Eval frameworks** (to be wired before implementation):
- [RAGAS docs](https://docs.ragas.io/) — retrieval scoring (context precision + recall).
- [DeepEval docs](https://www.confident-ai.com/) — pytest/CI integration.
- [BEIR paper](https://arxiv.org/pdf/2104.08663) — methodology reference.
- [RAGChecker paper](https://arxiv.org/pdf/2408.08067) — deferred; layer in post-Phase-A if needed.

---

## Open questions worth flagging (from the deepening)

1. **CDE vs hybrid BM25+dense.** Hybrid RRF is the recommended starting point — proven, lower-lift integration. CDE is the more cutting-edge dense-only option but requires changing embedding models and re-embedding the corpus. Phased decision: ship hybrid first, measure on the eval set, only adopt CDE if hybrid leaves >X% Type B failures.
2. **Doc-role auto-classification reliability.** Auto-classifying uploaded documents as problem/solution/lecture/syllabus is itself an ML task. Professor override must carry explicit provenance (`auto-confidence-0.92 vs professor-override`) — otherwise wrong role labels become a new silent failure mode that's hard to debug.
3. **LLM-router-on-every-query vs only-on-ambiguous.** GPT-4o-mini per-query routing adds ~250ms latency. With hybrid + intent classifier + reranker doing most of the disambiguation, the router can stay narrowly scoped to ambiguous Stage 1 results (~10% of queries). Worth validating against the eval set.
4. **Multi-modal content (figures, equations) is unaddressed.** Our PDFs include figures, and vision-supplemented extraction puts `[FIGURE: ...]` markers in the chunk stream. None of the retrieval architecture above explicitly handles "the student is asking about Figure 3." Worth flagging as a separate future phase, not blocking Phase A.
5. **AI-TA (arxiv 2311.02775)** is the closest published predecessor to our exact use case. Worth a direct read during implementation planning.

---

## Out of scope

- **No code changes** flow from this plan until the eval set is built, RAGAS+DeepEval are wired, and the implementation plan is written.
- **No band-aid fixes** for individual symptoms (Lab/PSet enum addition, Roman numeral special-casing, negation-pattern lists). The principled redesign is the alternative; enumeration regresses to the failure mode we're trying to escape.
- **No major UI changes** without their own data + research cycle. The `doc_type` UI control stays as-is for now; the underlying filter is what changes. The new `doc_role` field is auto-classified at upload, with optional professor override — minimally intrusive on the upload flow.
- **No Phase B or C implementation** before Phase A ships and we observe whether their symptoms persist.
- **No row-by-row "validation" pass.** That was reframed during deepening — see the execution-discipline section. The 44 rows live in the verification step (post-implementation regression test), not as a pre-implementation gate.

---

## Sources

### Research informing the initial Phase A architecture (hierarchical retrieval, reranking, doc routing, metadata)

- [Reranking Explained: Why It Matters for RAG Systems — Chatbase](https://www.chatbase.co/blog/reranking)
- [Top 7 Rerankers for RAG — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2025/06/top-rerankers-for-rag/)
- [RAGRouter: Learning to Route Queries to Multiple Retrieval-Augmented Language Models — arxiv 2505.23052](https://arxiv.org/pdf/2505.23052)
- [Hierarchical RAG: Multi-level Retrieval — Emergent Mind](https://www.emergentmind.com/topics/hierarchical-retrieval-augmented-generation-hierarchical-rag)
- [HiChunk: Evaluating and Enhancing RAG with Hierarchical Chunking — arxiv 2509.11552](https://arxiv.org/pdf/2509.11552)
- [MacRAG: Compress, Slice, and Scale-up for Multi-Scale Adaptive Context RAG — arxiv 2505.06569](https://arxiv.org/pdf/2505.06569)
- [MODE: Mixture of Document Experts for RAG — arxiv 2509.00100v1](https://arxiv.org/html/2509.00100v1)
- [Anthropic's Contextual Retrieval: A Guide With Implementation — DataCamp](https://www.datacamp.com/tutorial/contextual-retrieval-anthropic)
- [My RAG Stack for Code Retrieval — Dev.to](https://dev.to/daniel_romitelli_44e77dc6/my-rag-stack-for-code-retrieval-pgvector-hnsw-metadata-filters-reranking-and-the-parts-i-54en)
- [Understanding Metadata in RAG — Vectorize](https://docs.vectorize.io/build-deploy/data-pipelines/understanding-metadata/)
- [Metadata-Based Filtering in RAG Systems — CodeSignal](https://codesignal.com/learn/courses/scaling-up-rag-with-vector-databases/lessons/metadata-based-filtering-in-rag-systems)
- [Best Practices for Implementing RAG Systems in Production — Unstructured](https://unstructured.io/insights/rag-systems-best-practices-unstructured-data-pipeline)

### Research deepening (2026-05-21) — failure-mode-specific patterns

- [Metadata filters in RAG: why good documents disappear before retrieval starts — OptyxStack](https://optyxstack.com/rag-reliability/metadata-filters-in-rag-why-good-documents-disappear-before-retrieval-starts) (Type A — hard-filter anti-pattern)
- [Metadata-Driven RAG for Financial Question Answering — arxiv 2510.24402](https://arxiv.org/pdf/2510.24402) (Type A + D — metadata-as-feature, not filter)
- [CoRank: three-stage retrieval with metadata-aware coarse rerank — arxiv 2505.13757](https://arxiv.org/pdf/2505.13757) (Type A — three-stage refinement pattern)
- [VersionRAG: Hierarchical Graph for Version Discrimination — arxiv 2510.08109](https://arxiv.org/pdf/2510.08109) (Type B — version-sibling discrimination as a recognized RAG failure)
- [Reciprocal Rank Fusion explained — Serghei blog](https://blog.serghei.pl/posts/reciprocal-rank-fusion-explained/) (Type B + C — hybrid retrieval fusion)
- [Introducing Reciprocal Rank Fusion Hybrid Search — OpenSearch](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/) (Type B + C — production RRF)
- [ColBERTv2 — arxiv 2112.01488](https://arxiv.org/abs/2112.01488) and [PLAID — arxiv 2205.09707](https://arxiv.org/abs/2205.09707) (Type B — late-interaction option)
- [Late Interaction Overview — Weaviate](https://weaviate.io/blog/late-interaction-overview) (Type B — near-duplicate discrimination)
- [Contextual Document Embeddings (CDE, ICLR 2025) — arxiv 2410.02525](https://arxiv.org/html/2410.02525v1) (Type B — corpus-context-conditioned embeddings)
- [Reducing False Positives in Retrieval-Augmented Generation — InfoQ](https://www.infoq.com/articles/reducing-false-positives-retrieval-augmented-generation/) (Type C)
- [Full-Text Search vs Semantic Search — SingleStore](https://www.singlestore.com/blog/full-text-search-vs-semantic-search/) (Type C)
- [REIC: Retrieval-Enhanced Intent Classification — arxiv 2506.00210](https://arxiv.org/abs/2506.00210) (Type D — intent classification)
- [ACORD: Clause-Role Retrieval for Legal Contracts — arxiv 2501.06582v2](https://arxiv.org/html/2501.06582v2) (Type D — document-role metadata pattern)
- [NVIDIA AI-Q intent-classifier blueprint](https://docs.nvidia.com/aiq-blueprint/2.0.0/architecture/agents/intent-classifier.html) (Type D — intent + routing)
- [The Context Collapse Crisis — RAGAboutIt](https://ragaboutit.com/the-context-collapse-crisis-why-your-multi-turn-rag-system-loses-track-after-5-questions/) (Type E — anchor problem)
- [Query Rewriting Before Retrieval — Alhena](https://alhena.ai/blog/query-rewriting-before-retrieval-multi-turn-rag/) (Type E)
- [H-RAG: Lightweight LLM-based query rewriting — arxiv 2605.00631](https://arxiv.org/html/2605.00631) (Type E — canonical per-turn rewriter pattern)
- [Surprisingly Simple yet Effective Multi-Query Rewriting — HuggingFace 2406.18960](https://huggingface.co/papers/2406.18960) (Type E)
- [Comparing RAG and GraphRAG for Page-Level Retrieval on Math Textbook — arxiv 2509.16780](https://arxiv.org/pdf/2509.16780) (educational RAG)
- [Enhancing RAG with Entity Linking for Educational Platforms — arxiv 2512.05967](https://arxiv.org/pdf/2512.05967) (educational RAG)
- [AI-TA: AI Teaching Assistant for university courses — arxiv 2311.02775](https://arxiv.org/pdf/2311.02775) (closest published predecessor)

### Research deepening (2026-05-21) — evaluation methodology

- [BEIR: Heterogeneous Zero-Shot IR Benchmark — arxiv 2104.08663](https://arxiv.org/pdf/2104.08663)
- [RAGAS metrics docs](https://docs.ragas.io/en/v0.1.21/references/metrics.html)
- [RAGChecker: Fine-Grained RAG Diagnosis — arxiv 2408.08067](https://arxiv.org/pdf/2408.08067) · [Amazon Science page](https://www.amazon.science/publications/ragchecker-a-fine-grained-framework-for-diagnosing-retrieval-augmented-generation) · [GitHub](https://github.com/amazon-science/RAGChecker)
- [TruLens for RAG evaluation](https://www.trulens.org/)
- [DeepEval / Confident AI](https://www.confident-ai.com/)
- [Hard-negative mining for retrieval — arxiv 2511.08029](https://arxiv.org/pdf/2511.08029) and [arxiv 2505.18366](https://arxiv.org/pdf/2505.18366)
- [What is RAG Evaluation — Braintrust](https://www.braintrust.dev/articles/what-is-rag-evaluation)
- [Golden Datasets and Evaluation Standards — Statsig](https://www.statsig.com/perspectives/golden-datasets-evaluation-standards)
