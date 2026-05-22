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

## Research grounding per decision

Each Matrix 1 ship/defer/drop call is backed by specific external literature surfaced in the research deepening pass (2026-05-21). This section makes the link explicit. URLs resolve to the parked plan's [Sources section](maize-retrieval-redesign-plan.md#sources).

### Refinement 1 — Drop hard SQL filter + hybrid Stage 1 (BM25 + dense + RRF) → SHIP v1

- **[OptyxStack: "Metadata filters in RAG: why good documents disappear before retrieval starts"](https://optyxstack.com/rag-reliability/metadata-filters-in-rag-why-good-documents-disappear-before-retrieval-starts)** — treats hard metadata filters as a recall-killer: "the system can exclude the correct document before vector search, lexical search, or reranking ever start... the final answer comes from an incomplete candidate pool, which often looks like weak retrieval or hallucination even though the real problem is filtering." Canonical recommendation: hard filters are appropriate ONLY for access control, tenant isolation, and forbidden scopes — business-dimension metadata should "bias ranking first and relax only when coverage is weak." Direct match for our Type A failure (LLM-misclassified quizzes pulled into PS candidate pool by the homework+assignment_number filter).

- **[Metadata-Driven RAG for Financial Question Answering (arxiv 2510.24402)](https://arxiv.org/pdf/2510.24402)** — explicitly uses metadata as a relevance feature inside the reranker rather than a hard pre-filter. Same pattern we'd be adopting.

- **[CoRank (arxiv 2505.13757)](https://arxiv.org/pdf/2505.13757)** — three-stage pipeline of (i) offline extraction of document-level features (categories, sections, keywords), (ii) coarse reranking with compact metadata-aware representations, (iii) fine-grained full-text reranking on top candidates. The hybrid Stage 1 we're shipping is the first two stages of CoRank, light-weighted. (We're DEFERRING the full Stage 2.5 coarse rerank — see below — but the broader CoRank pattern still grounds the "metadata stops being a hard gate" decision.)

- **[VersionRAG (arxiv 2510.08109)](https://arxiv.org/pdf/2510.08109)** — Oct 2025; identifies "version discrimination" as an underexplored RAG failure mode: "standard RAG systems frequently fail at version discrimination, retrieving content from incorrect document iterations when multiple versions exist." Same shape as our Type B (Roman numeral siblings), and the canonical fix they recommend is hybrid BM25 + dense.

- **[Reciprocal Rank Fusion explained (Serghei blog)](https://blog.serghei.pl/posts/reciprocal-rank-fusion-explained/)** and **[Introducing RRF Hybrid Search (OpenSearch)](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/)** — production deployment patterns. The OpenSearch blog confirms RRF as the default fusion in OpenSearch, Elasticsearch, Azure AI Search, MongoDB Atlas, and Weaviate. Standard, battle-tested, not novel.

**Empirical vs research convergence:** Strong agreement. Both lines of evidence point at hybrid + drop hard filter as the right call for Types A and C. Type B local-100%-but-prod-failing is an interesting wrinkle (research says hybrid is the fix; local doesn't validate, prod does); we ship the fix anyway because both signals at least don't contradict.

### Refinement 2 — Per-turn query rewriter + per-turn re-retrieval → SHIP v1

- **[H-RAG: Lightweight LLM-based query rewriting (arxiv 2605.00631)](https://arxiv.org/html/2605.00631)** — canonical per-turn rewriter pattern: lightweight LLM rewriter returns query unchanged when already standalone, decontextualizes otherwise. Exactly what we'd be implementing.

- **[The Context Collapse Crisis — RAGAboutIt](https://ragaboutit.com/the-context-collapse-crisis-why-your-multi-turn-rag-system-loses-track-after-5-questions/)** — describes the failure pattern in plain terms: "By the fifth or sixth turn... multi-turn RAG systems may retrieve documents that seem relevant but contradict information from earlier turns." Maps directly to our Type E cache anchoring observation.

- **[Query Rewriting Before Retrieval — Alhena](https://alhena.ai/blog/query-rewriting-before-retrieval-multi-turn-rag/)** — quantifies the problem: "Over 60% of follow-up messages contain unresolved coreferences or implicit context that depend entirely on prior turns." Underscores the per-turn rewriter is load-bearing, not optional.

- **[Surprisingly Simple yet Effective Multi-Query Rewriting (HuggingFace 2406.18960)](https://huggingface.co/papers/2406.18960)** — confirms multi-query rewriting at retrieval time as a robust technique.

**Empirical vs research convergence:** Research-strong, empirical-weak (locally). Local eval can't reproduce cache anchoring (cold cache per row); the case for shipping is anchored on prod logs + literature. Confidence: high based on literature consensus; local eval is silent here, not contradictory.

### Refinement 3 — doc_role + structured intent classifier + solutions-as-supplementary → SHIP v1

- **[REIC: Retrieval-Enhanced Intent Classification (arxiv 2506.00210)](https://arxiv.org/abs/2506.00210)** — useful as a primitive for per-turn intent classification. Shows intent classification benefits from retrieving (query, intent) exemplars before classifying — relevant for the cheap-LLM-call-per-turn classifier we'd add.

- **[ACORD: Clause-Role Retrieval for Legal Contracts (arxiv 2501.06582v2)](https://arxiv.org/html/2501.06582v2)** — directly demonstrates the "explicit role label per document + role-aware reranking" pattern in legal contracts. The mechanism transfers cleanly: in their setting, contract-clause role (representation/warranty/covenant) determines retrieval priority; in ours, doc role (problem/solution/lecture/syllabus/reference) does.

- **[NVIDIA AI-Q intent classifier blueprint](https://docs.nvidia.com/aiq-blueprint/2.0.0/architecture/agents/intent-classifier.html)** — production pattern for intent + routing upstream of retrieval. Validates the architectural shape.

- **[Metadata-Driven RAG (arxiv 2510.24402)](https://arxiv.org/pdf/2510.24402)** — also covers Type D: metadata-driven retrieval with role-aware reranking as the canonical pattern.

**Empirical vs research convergence:** Strong agreement. Per-row data shows 40% solutions-leak; the literature converges on doc-role + intent as the canonical fix. The user-refined "solutions-as-supplementary, not forbidden" framing is an OUR-design choice consistent with this literature (the literature doesn't prescribe pedagogy; we're applying its mechanisms to our specific pedagogical goal).

### Refinement 4 (adjacent) — `doc_role` becomes the primary semantic axis (replaces `doc_type` as the retrieval-load-bearing field) → SHIP v1

- **[OptyxStack](https://optyxstack.com/rag-reliability/metadata-filters-in-rag-why-good-documents-disappear-before-retrieval-starts)** (same source as Refinement 1) — explicit warning about "hidden-default metadata propagation" (e.g., filters inherited from session context without explicit evidence). Calls out user-applied metadata reliability as a known unbenchmarked weak spot. Justifies treating `doc_type` as a label whose accuracy can't be assumed.

- **[ACORD (arxiv 2501.06582v2)](https://arxiv.org/html/2501.06582v2)** — the pattern of "document-role typology drives retrieval" is the load-bearing design. ACORD's 5-role schema (representation, warranty, covenant, etc.) maps to our planned 5-role schema (problem, solution, lecture, syllabus, reference) at the structural level.

**Empirical vs research convergence:** Strong. The per-row data — `econometrics quiz N` mis-classified as `doc_type='homework', assignment_number=N` causing the cascade of Type A failures — is the smoking gun the literature would have predicted. Both confirm `doc_type` is not a reliable enough field to be retrieval-load-bearing, and a cleaner orthogonal axis (role) is needed.

### Defer 1 — CoRank-style Stage 2.5 coarse rerank

- **[CoRank (arxiv 2505.13757)](https://arxiv.org/pdf/2505.13757)** — the source that recommends this stage. Paper is well-cited and the pattern is canonical, BUT empirical signal in our baseline is absent: the current single-rerank-stage achieves the failures we see for reasons OTHER than rerank precision (mostly upstream filter/routing). Adding Stage 2.5 mostly compounds latency without addressing identified failures.

- **Justification for deferral:** evidence-based. We're not contradicting the paper; we're saying "shipping it without empirical motivation would be cargo-culting." Watch the post-refactor scorecard — if Stage 3 precision is the new bottleneck, Stage 2.5 lands in v2.

### Defer 2 — Contextual Document Embeddings (CDE)

- **[Contextual Document Embeddings (CDE, ICLR 2025 — arxiv 2410.02525)](https://arxiv.org/html/2410.02525v1)** — the source. Paper describes drop-in replacement for OpenAI text-embedding-3-small, trained explicitly on hard negatives from corpus context. cde-small-v1 was the top sub-400M model on MTEB at time of research.

- **Justification for deferral:** the parked plan ITSELF explicitly recommended deferral in its [research output section](maize-retrieval-redesign-plan.md): "Hybrid RRF is the recommended starting point — proven, lower-lift integration. CDE is the more cutting-edge dense-only option but requires changing embedding models and re-embedding the corpus. Phased decision: ship hybrid first, measure on the eval set, only adopt CDE if hybrid leaves >X% Type B failures." We're following the research's own staging guidance.

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
5. Each decision cites 1-3 specific external sources with URLs + paraphrases of what the source claims.

---

## Convergence and divergence: research vs empirical signal

This table makes explicit, per decision, whether research and empirical baseline agree — important for confidence calibration before implementation.

| Decision | Research says | Local empirical says | Convergence? | Confidence |
|---|---|---|---|---|
| 1. Drop hard SQL filter + hybrid Stage 1 | Hybrid BM25+dense+RRF is the canonical fix for hard-filter cascades and version/sibling discrimination. Hard filters are an anti-pattern outside access control. (OptyxStack, VersionRAG, CoRank, Metadata-Driven RAG, RRF docs.) | Per-row data: Type A dominated by `econ-quiz-2` winning for PS2 queries — exactly the hard-filter pathology literature describes. Type C at 0% is the fuzzy-filename matcher misfiring. Both textbook research-predicted failures. | **STRONG CONVERGENCE** | High |
| 2. Per-turn query rewriter + per-turn re-retrieval | Per-turn rewriting is the canonical solution to multi-turn "context collapse"; 60%+ of follow-up turns have unresolved coreferences without it. (H-RAG, RAGAboutIt, Alhena, HuggingFace 2406.18960.) | Local eval can't reproduce cache anchoring (cold cache per row → Type E 86%, multi-turn working 65% partly artifact). Prod logs DO show the failure. | **Research-strong, eval-silent** (not contradictory — inconclusive locally) | High (literature + prod observation) |
| 3. doc_role + intent classifier + solutions-as-supplementary | Role-aware retrieval is canonical in dual-document corpora; intent classification upstream of retrieval is a known production technique. (ACORD, REIC, NVIDIA AI-Q, Metadata-Driven RAG.) | Per-row data: 40% solutions leak on Type D; multiple synthetic Type D rows return solutions as top-1. Direct match for the research-predicted failure. | **STRONG CONVERGENCE** | High |
| 4 (adj). doc_role as primary axis (replaces doc_type) | User-applied metadata reliability isn't well-benchmarked; ACORD-style role typology is the canonical workaround. (OptyxStack, ACORD.) | Per-row data: LLM-auto-classified `doc_type` is demonstrably broken for ECON S1117 quizzes — root cause of the Type A cascade. | **STRONG CONVERGENCE** (empirical reveals the very weakness research warned about) | High |
| 5. (Defer) CoRank Stage 2.5 coarse rerank | Coarse rerank is documented SOTA, but the paper doesn't claim it's load-bearing for every architecture. | No empirical signal in baseline that another rerank stage would help — rerank precision isn't currently the bottleneck (filter/routing is). | **Research-recommended, eval-unsupported** | Medium (defer correct, but watch post-v1) |
| 6. (Defer) CDE embeddings | Strong literature support, but research explicitly recommended staging: "ship hybrid first, adopt CDE only if hybrid leaves Type B residuals." | Type B passes 100% locally — no residuals to address. | **Research-aligned deferral** (following the source's own guidance) | High |

### Where to be vigilant after v1 ships

Two decisions where research+empirical aren't fully unanimous, requiring active follow-up:

- **Refinement 2 (per-turn rewriter)** — empirical is SILENT (cold-cache artifact). After v1 ships, manually test cache-anchoring scenarios in a real browser session (not the eval harness) to confirm the rewriter actually eliminates the prod-observed failure mode.

- **Refinement 1 prod gap** — Type B passes locally but research + prod logs say it's a real failure mode. After v1 ships locally, plan to validate against a fresh prod-corpus snapshot to confirm hybrid actually fixes Type B in prod (not just papers over a locally-clean state).

### What this convergence analysis is NOT

- An assertion that research-aligned decisions are guaranteed correct. They're well-grounded; empirical verification post-v1 is still required.
- A replacement for the post-refactor scorecard. The scorecard is the actual proof; this is the pre-implementation justification.
- A peer-reviewed academic claim. It's an engineering decision audit trail with citations to surface external defensibility.
