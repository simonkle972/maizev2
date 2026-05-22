# Stage 2B research findings — per-TA configurable doc classification

**Date:** 2026-05-22 · **Purpose:** literature validation of the per-TA configurable document-category design before committing the Stage 2B schema rework. See [implementation plan Stage 2B](maize-retrieval-phase-a-implementation-plan-2026-05-22.md#stage-2b--design-correction-from-doc_role-enum-to-per-ta-configurable-categories) for context.

## Q1 — Per-corpus / per-tenant configurable taxonomies in RAG

**Finding:** Per-tenant configurable metadata schemas are a recognized, mature pattern in production multi-tenant RAG and enterprise search. The canonical implementation is "tenant defines its own schema/properties," not "tenant customizes one global enum." Closest production analogs to the proposed Maize design:

- **Microsoft 365 Copilot Graph connectors** — each connection registers a flat schema of properties (data type, label, aliases, search behavior flags); these properties get surfaced to Copilot's retrieval layer.
- **Glean** — "group schema" defined per datasource; metadata values attached per document. No global cross-tenant enum.
- **LlamaIndex / LangChain** — `Document.metadata` is an open dict; `AttributeInfo` / self-query retrievers declare per-corpus filterable attributes.
- **Pinecone / Qdrant** — explicitly favor schemaless JSON metadata so new fields can appear without re-indexing. (Weaviate / Milvus are the schema-first counterexample.)

**Sources:**
- [Microsoft 365 Copilot connectors — register and update schema](https://learn.microsoft.com/en-us/graph/connecting-external-content-manage-schema)
- [Glean custom data source schemas](https://docs.glean.com/connectors/custom/about) + [Glean indexing API overview](https://developers.glean.com/api-info/indexing/getting-started/overview)
- [Multi-tenant RAG with Amazon Bedrock Knowledge Bases](https://aws.amazon.com/blogs/machine-learning/multi-tenant-rag-with-amazon-bedrock-knowledge-bases/)
- [Building successful multi-tenant RAG applications (Nile)](https://www.thenile.dev/blog/multi-tenant-rag)
- [Multi-tenant RAG with LlamaIndex](https://www.llamaindex.ai/blog/building-multi-tenancy-rag-system-with-llamaindex-0d6ab4e0c44b)
- [Metadata filtering in vector search — engineering leaders guide](https://www.saumilsrivastava.ai/blog/metadata-filtering-in-vector-search-a-comprehensive-guide-for-engineering-leaders)

**Implication for our design:** Per-TA `doc_categories` JSON array maps cleanly to Microsoft's "per-connection schema" and Glean's "group schema." We are NOT inventing — we are picking the well-trodden side of a real industry split (schemaless/per-tenant flexible vs. global enum). One nuance to borrow: these systems treat the schema as a *registered* artifact (labels, aliases, types), not just a free string list. Worth tracking display-label AND a stable internal slug so renames don't break already-classified documents (this becomes Refinement #1 below).

## Q2 — User-defined doc labels as retrieval signal (text context, not filter)

**Finding:** Empirical evidence strongly supports **metadata-as-text-context** as a robust retrieval booster, independent of whether labels are fixed or user-defined. Key 2025-2026 studies:

- **Metadata-Driven RAG for Financial QA** (arXiv 2510.24402, Oct 2025) — contextual chunks (metadata prepended as text before embedding) consistently improved F1 across all configurations; metadata-aware reranker hit F1 43.2 vs. 44.4 for Cohere's commercial reranker. Pre-retrieval filtering on metadata showed mixed results and sometimes REDUCED recall — exactly the failure mode we're avoiding.
- **MetaRAG / Systematic Framework for Enterprise Knowledge Retrieval** (arXiv 2512.05411) — up to **12% precision gain for metadata-enriched retrieval** (0.825 vs. 0.733 content-only) when metadata is fused into chunk text.
- **Anthropic Contextual Retrieval** — prepending a chunk-situating context string reduces retrieval failures by 35% on its own; combined with reranking, 67%. Same architectural shape Maize is using.
- **Contextual AI's instruction-following reranker** — explicitly accepts `document_type` and metadata fields as instructions or per-document metadata, treated as soft preference signals, not hard filters.

**No head-to-head study compares FREE-FORM vs ENUM-CONSTRAINED labels as reranker text inputs.** That specific A/B is a research gap. Closest adjacent evidence: classical IR literature on folksonomies vs. controlled vocabularies — folksonomies match user mental models better; controlled vocabularies are consistent but expensive to maintain. For reranker-as-text consumption, the LLM's semantic understanding makes synonymy a much smaller problem than in classical IR.

**Sources:**
- [Metadata-Driven RAG for Financial QA (arXiv 2510.24402)](https://arxiv.org/html/2510.24402v1)
- [Systematic Framework for Enterprise Knowledge Retrieval (arXiv 2512.05411)](https://arxiv.org/pdf/2512.05411)
- [Anthropic Contextual Retrieval guide (DataCamp)](https://www.datacamp.com/tutorial/contextual-retrieval-anthropic)
- [Contextual AI instruction-following reranker](https://contextual.ai/blog/introducing-instruction-following-reranker)
- [MongoDB on instruction-following rerankers as context engineering](https://www.mongodb.com/company/blog/technical/instruction-following-rerankers-an-unsung-context-engineering-tool)

**Implication for our design:** "Metadata as text context for the reranker, not a filter" is well-supported. Literature is actively warning against pre-retrieval metadata filtering as the PRIMARY mechanism — it cuts recall when labels are imperfect (which user-defined labels reliably will be). The unanswered question (free-form vs. enum) is unlikely to be load-bearing once a capable reranker reads the label as text. **Direction confirmed.**

## Q3 — Educational-RAG specifically

**Finding:** No major production educational-RAG system publicly documents a "tell us what kinds of docs you have" corpus-setup flow with a configurable taxonomy. Available technical writeups suggest most ship with implicit or fixed structure:

- **CS50.ai (Harvard SIGCSE 2024 paper)** — RAG knowledge base auto-updates from the existing curriculum; document-type taxonomy not a configurable concept. Hand-tuned by CS50 staff for one course's structure.
- **Khanmigo** — RAG + SFT + RL; no public detail on per-document classification. Khan Academy's content is heavily structured by their internal taxonomy (units, lessons, exercises) so the "doc category" problem doesn't surface the same way.
- **Coursera Coach** — operates over uploaded course materials. No public mention of professor-facing document categorization at course-creation; categorization is implicit from Coursera's existing schema.
- **MathGPT.ai, TutorFlow, CircleIn, UC San Diego bespoke tutor** — emphasize uploads and behavior controls. CircleIn describes 15-minute setup with "syllabus, policies, due dates" — closest production analog but a fixed three-bucket model, not configurable.
- **Workleap onboarding wizard, Learnster Content Creation Wizard** — interestingly, the closest analog exists in CORPORATE onboarding/L&D, where wizards run LLM categorization with predefined-but-editable format types. Exactly the "sensible default list, configurable" shape.

**Sources:**
- [Teaching CS50 with AI (SIGCSE 2024)](https://cs.harvard.edu/malan/publications/V1fp0567-liu.pdf)
- [CS50.ai docs](https://cs50.readthedocs.io/cs50.ai/)
- [Coursera Coach / U-M launch](https://news.umich.edu/u-m-launches-ai-powered-coursera-coach-for-interactive-instruction/)
- [CircleIn for Professors](https://www.circleinapp.com/professors/aitutor)
- [MathGPT.ai features](https://www.mathgpt.ai/product/features)
- [Learnster Content Creation Wizard](https://helpcenter.learnster.com/en/articles/317005-content-creation-wizard)
- [Workleap AI Onboarding Wizard](https://workleap.com/blog/introducing-the-worlds-first-ai-employee-onboarding)

**Implication for our design:** The canonical pattern in ed-tech RAG doesn't exist for our exact UX. We'd be the first ed-tech RAG to adopt the per-tenant configurable-taxonomy pattern explicitly. **Not reckless** — it's a port of well-validated Workleap/Learnster L&D + Glean/M365 enterprise patterns into ed-tech. But it's differentiation, not best-practice copying. We own the failure surface; we don't have a CS50.ai blog post to debug against.

## Q4 — Cold-start defaults for user-defined taxonomies

**Finding:** Literature converges on **three complementary patterns**, often combined:

1. **Sensible defaults** (UX): UserOnboard, Candu, Chameleon all converge that defaults should "shorten the path to value" — pre-fill the most common shape, let the user edit.
2. **Auto-discover via clustering + LLM labeling**: TELEClass (arXiv 2403.00165), TaxoAdapt (arXiv 2506.10737), Microsoft Data Science Medium piece — embed docs, cluster, ask an LLM to name each cluster, present the resulting taxonomy as suggestions. Most useful when a tenant has many existing documents.
3. **Just-in-time labeling at upload time**: LlamaIndex's metadata extractor, "auto-tagging" patterns — at ingest, the LLM proposes a label from the current taxonomy; can also propose a new category if confidence is low.

**Sources:**
- [Sensible Defaults (UserOnboard)](https://www.useronboard.com/onboarding-ux-patterns/sensible-defaults/)
- [Onboarding wizard patterns (UserGuiding)](https://userguiding.com/blog/what-is-an-onboarding-wizard-with-examples)
- [TELEClass — LLM-enhanced hierarchical text classification (arXiv 2403.00165)](https://arxiv.org/pdf/2403.00165)
- [TaxoAdapt — taxonomy construction aligned to evolving corpora (arXiv 2506.10737)](https://arxiv.org/pdf/2506.10737)
- [Building taxonomies from unstructured text using LLMs (Microsoft Data Science)](https://medium.com/data-science-at-microsoft/from-chaos-to-clarity-building-taxonomies-from-unstructured-text-using-large-language-models-c1303db3adb1)
- [Using LLMs for automating taxonomy tagging (More Awesome)](https://moreawesome.co/insights/using-llms-for-automating-taxonomy-tagging/)

**Implication for our design:** The proposed UX (default list at creation, professor edits, JIT LLM suggests at upload) is the well-supported synthesis of all three patterns. One enhancement for a later iteration (NOT v1): periodic background check that clusters the corpus and surfaces "we noticed X documents that don't fit your current categories; want to add a new one?" — the TELEClass/TaxoAdapt pattern adapted to a small course corpus. **Defer to v2.**

## Q5 — Failure modes of free-form labels

**Finding:** IA literature + recent LLM-systems papers + enterprise-tagging blogs surface a consistent set of failure modes — most of which our design already addresses or can address cheaply:

1. **Schema explosion**: Every user picks slightly different labels ("Homework" vs "Hw" vs "Problem Sets" vs "PSets"). arXiv 2511.19933 flags "enum explosion" as highest-priority data-integrity alert. *Mitigated in our design:* labels are **per-TA, not global**, so explosion is bounded to one course corpus.
2. **LLM drift when candidate set isn't constrained**: Open-vocabulary classifiers drift across calls (Flip-Flop Consistency, arXiv 2510.14242; Semantic Clustering of QA, arXiv 2410.15440). *Mitigated:* our LLM auto-classifier picks from the TA's CURRENT list — constrained vocabulary at classification time, not open generation.
3. **Inconsistency across labelers**: Folksonomy synonymy/polysemy. *Mitigated:* one labeler per TA (the LLM + the professor's accept/edit). Multi-user inconsistency doesn't apply.
4. **Multi-word vs single-word, case, punctuation**: Library Drift (arXiv 2605.19576) flags unbounded skill-name growth as causing retrieval degradation. *Mitigation NEEDED:* explicit input normalization at storage (trim/collapse whitespace, max length, case-preserve display but lowercase for matching; reject empty strings).
5. **Renames break previously-classified docs**: If "Problem Sets" is renamed to "PSets", existing docs with `doc_category="Problem Sets"` either orphan or need backfill. *Mitigation NEEDED:* store internal stable slug per category; rename only changes the display label. (This becomes Refinement #1.)
6. **Empty / "Other" overuse**: Free-form systems often see one bucket dominate. *Future mitigation:* monitor distribution analytics; if >50% goes to "other" for any TA, surface a hint. **Defer to v2.**

**Sources:**
- [Failure Modes in LLM Systems (arXiv 2511.19933)](https://arxiv.org/pdf/2511.19933)
- [Library Drift in self-evolving LLM skill libraries (arXiv 2605.19576)](https://arxiv.org/html/2605.19576)
- [Flip-Flop Consistency (arXiv 2510.14242)](https://arxiv.org/pdf/2510.14242)
- [Folksonomy challenges (ScienceDirect)](https://www.sciencedirect.com/topics/computer-science/folksonomies)
- [Error analysis for LLMs](https://futureagi.com/blog/what-is-error-analysis-llm-2026)

**Implication for our design:** The two most underweighted concerns in the original draft are **rename robustness** and **input normalization**. Both are 30-min additions to the spec that prevent most documented failure modes.

## Convergence and divergence summary

| Question | Verdict |
|---|---|
| Q1: Per-tenant configurable taxonomy a known pattern? | **Supports design.** Strong precedent in Glean, M365 Copilot, LlamaIndex, Pinecone. Maize would be first ed-tech RAG to adopt explicitly. |
| Q2: User-defined labels as reranker text context? | **Supports design.** Metadata-as-text strongly evidenced (12%+ precision gain). Free-form-vs-enum is a research gap; unlikely to matter once reranker reads label as text. |
| Q3: Ed-tech precedent? | **Silent / divergent.** No major ed-tech RAG ships this UX. Closest analogs are corporate L&D. Differentiation, not copying. |
| Q4: Cold-start defaults pattern? | **Supports design.** Defaults + JIT labeling = synthesis of recommended literature. Auto-discovery via clustering is v2 enhancement, not v1 blocker. |
| Q5: Free-form-label failure modes? | **Refinement needed.** Most modes mitigated by per-TA scoping and LLM-picks-from-list. Two gaps: rename robustness + explicit input normalization. |

## Recommendation: REFINED

Research supports the per-TA configurable category design overall. **Two specific adjustments before rework:**

1. **Rename robustness via internal slug.** Store each category as `{slug: string, label: string}`. The slug is what's persisted on `Document.doc_category`. Renaming a category updates only the label across the TA and its documents; the slug stays immutable. Costs ~20 lines of code now; saves a painful migration later when professors inevitably rename categories.

2. **Explicit input normalization rules** for category labels:
   - Trim whitespace
   - Collapse internal whitespace to single spaces
   - Max 64 chars
   - Non-empty
   - Allow letters / numbers / spaces / hyphens (and CJK if internationalization matters later)
   - Apply at the API boundary
   - Don't enforce case
   - Reject duplicate slugs (case-insensitive) within a TA

**Optional / defer to v2:**
- Periodic clustering-based "suggest a new category" pass (TELEClass/TaxoAdapt pattern).
- "Other" distribution monitoring with a soft hint when one bucket exceeds 50%.

The core decisions — per-TA scope, LLM picks from current list (constrained vocabulary, not open generation), reranker consumes label as text (not filter), defaults at creation — are all well-supported by the 2024–2026 literature. **Proceed with Stage 2B rework after the two refinements above are in the spec.**

## Open questions / gaps

- **No direct A/B exists for free-form vs. enum-constrained labels as reranker-text inputs.** Supporting evidence in Q2 is by analogy and architectural similarity, not direct empirical comparison. If we want hard validation, this is a small in-house experiment for later (once we have ~50 prod queries per TA across 3+ TAs).
- **No published precedent for the exact ed-tech UX.** We own the failure surface — no CS50.ai blog post to debug against. Mitigate by instrumenting category usage and reranker score distributions from day 1.
- **Multi-language category labels not researched.** If a TA is non-English, does the LLM auto-classifier handle non-English labels well? gpt-4o handles this, but the JIT classifier prompt should be language-agnostic. Worth a spot check before shipping.
- **11-category default is on the upper end.** Hick's law / cognitive load: a 5–7 category default with "add more" might convert better at onboarding than 11. UX call, not a research call.

## Cross-references

- Implementation plan: [maize-retrieval-phase-a-implementation-plan-2026-05-22.md](maize-retrieval-phase-a-implementation-plan-2026-05-22.md) — Stage 2B section
- Gap analysis: [maize-retrieval-gap-analysis-2026-05-22.md](maize-retrieval-gap-analysis-2026-05-22.md)
- Parked retrieval plan: [maize-retrieval-redesign-plan.md](maize-retrieval-redesign-plan.md)
