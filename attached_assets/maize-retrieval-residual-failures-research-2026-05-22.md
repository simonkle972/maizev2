# Maize Retrieval — Residual Failures Research

**Date:** 2026-05-22
**Context:** Post-hybrid-refactor (BM25 + dense + filename-overlap fused via RRF; doc_category injected into the LLM reranker). Two open failure classes remain. This memo grounds three design questions before we cut the next ticket.

---

## Q1. Adaptive routing / confidence-based fusion in RAG retrieval

### What the literature says

Equal-weight RRF is a known foot-gun when one signal is far more informative than the others on a given query. The recent literature has converged on three distinct approaches that all reject "static blend for every query":

1. **Question-complexity routing (Adaptive-RAG, Jeong et al., 2024).** A small classifier LM predicts whether a query is "simple" (no retrieval), "moderate" (single-shot retrieval), or "complex" (multi-hop) and routes accordingly. Labels are auto-collected from downstream outcomes, not hand-labeled. ([arxiv.org/abs/2403.14403](https://arxiv.org/abs/2403.14403)) This is the closest formal analogue of "direct lookup vs. conceptual."
2. **Self-RAG (Asai et al., 2023).** Instead of an external router, the generator emits "reflection tokens" that decide on-the-fly whether and how to retrieve. ([arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511)) More expensive — requires fine-tuned generator — and probably overkill for our case.
3. **Dynamic Alpha Tuning (DAT, Hsu & Tzeng, 2025).** An LLM scores the top-1 result from each retriever (BM25 vs. dense) for each query, normalises the two scores, and uses the ratio as the fusion weight α. Beats fixed-α hybrid across the standard benchmarks. ([arxiv.org/abs/2503.23013](https://arxiv.org/abs/2503.23013), [arxiv.org/html/2503.23013v1](https://arxiv.org/html/2503.23013v1)) This is the cleanest "downweight rankers when another one is confident" mechanism in print.

There are also "weighted RRF" variants that multiply the 1/(k+rank) term by a normalised confidence score per result (uregina.ca paper on WRRF, [uregina.ca/~nss373/papers/Rag-CCNC2026.pdf](https://uregina.ca/~nss373/papers/Rag-CCNC2026.pdf)), but these still treat each retriever symmetrically — they don't short-circuit when one is dominant.

### Practitioner consensus

- **Weaviate** explicitly recommends per-query α tuning: "high-intent, brand-heavy queries have higher BM25 weight, while vague, exploratory queries get higher vector weight." ([weaviate.io/blog/hybrid-search-fusion-algorithms](https://weaviate.io/blog/hybrid-search-fusion-algorithms), [docs.weaviate.io/weaviate/concepts/search/hybrid-search](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search))
- **LlamaIndex** ships first-class `RouterQueryEngine` with LLM and Pydantic selectors over a small set of choices. ([developers.llamaindex.ai/python/framework/module_guides/querying/router/](https://developers.llamaindex.ai/python/framework/module_guides/querying/router/)) The pattern is "one cheap LLM call → pick a retrieval tool" — exactly the LLM-router shape the question asks about.
- **Pinecone** treats α as a single global dial but documents that α≈0 (pure BM25) is the right setting for exact-lookup workloads and α≈1 (pure dense) for natural-language. ([pinecone.io/learn/hybrid-search-intro](https://www.pinecone.io/learn/hybrid-search-intro/), [docs.pinecone.io/guides/search/hybrid-search](https://docs.pinecone.io/guides/search/hybrid-search))
- **Glean** describes ranking as "field boosts, freshness weights, authority scores, popularity signals, and hybrid fusion between lexical and semantic matching all interact" — i.e. it is fundamentally a multi-signal scorer, not a fixed fusion. ([glean.com/perspectives/how-to-debug-enterprise-search-relevancy-issues](https://www.glean.com/perspectives/how-to-debug-enterprise-search-relevancy-issues))
- **Helixiora's "Building an Efficient Router for RAG"** is explicit about the cost tradeoff: keyword-rule routers are fast and free but rigid; LLM classifiers are accurate but add cost/latency; lightweight distilled classifiers are the practical middle. ([helixiora.com/building-an-efficient-router-for-rag-applications](https://helixiora.com/building-an-efficient-router-for-rag-applications/))

On **calibration thresholds for "high enough" filename overlap**: nothing in the literature gives a magic number. The closest concrete advice is Elasticsearch's `constant_score` recipe of giving exact title matches a score ~3× the fuzzy match (e.g. 15 vs. 5) so they dominate without entirely silencing other signals. ([discuss.elastic.co/t/how-to-increase-score-for-exact-word-phrase-match-in-elastic-search/184901](https://discuss.elastic.co/t/how-to-increase-score-for-exact-word-phrase-match-in-elastic-search/184901), [forloop.co.uk/blog/favouring-exact-matches-in-elasticsearch](https://forloop.co.uk/blog/favouring-exact-matches-in-elasticsearch))

### Verdict — apply

Apply, with the cheap variant. The full DAT pattern (one LLM call per query to score top-1 from each retriever) is principled but doubles latency. The pragmatic path:

- **Tier 1 — rule-based short-circuit.** If the filename-overlap signal produces a unique top doc whose normalised score exceeds a threshold AND the query contains a token Maize already classifies as a filename feature (e.g. "Lecture 5", "Pset 2"), bypass dense/BM25 fusion and pin that doc as the retrieval anchor. This is the Elasticsearch `constant_score` idea applied to our filename signal.
- **Tier 2 — LLM router only if Tier 1 misses.** A single cheap classifier call ("direct lookup / conceptual / ambiguous") gated behind a token-overlap pre-filter, in the LlamaIndex selector mould. Do not pay for it on every query.

Reject the full DAT/Self-RAG architectures for now — too much surface area for the marginal lift over a rule + cheap classifier.

---

## Q2. Numeric / sequential filename metadata as RAG signal

### What the literature says

This is the question with the **thinnest formal literature**. There is no canonical paper on "ordinal identifiers in filenames as a retrieval signal" — it sits in the practitioner space.

What exists:

- **Multi-Meta-RAG (Poliakov & Shvai, 2024)** extracts metadata fields from queries via an LLM and uses them as database filters — directly applicable when the user query mentions a number that matches a metadata field. ([arxiv.org/abs/2406.13213](https://arxiv.org/abs/2406.13213))
- **Haystack's "Extract Metadata from Queries"** is the cleanest practitioner write-up: an LLM extracts structured fields from the query and converts them into hard filters on the metadata store. ([haystack.deepset.ai/blog/extracting-metadata-filter](https://haystack.deepset.ai/blog/extracting-metadata-filter))
- **Self-query retrievers (LangChain pattern)** do the same: parse the natural-language query into a (semantic_query, metadata_filter) tuple. Examples in the wild include extracting year/country from climate-report filenames. ([medium.com/@lorevanoudenhove/enhancing-rag-performance-with-metadata-the-power-of-self-query-retrievers-e29d4eecdb73](https://medium.com/@lorevanoudenhove/enhancing-rag-performance-with-metadata-the-power-of-self-query-retrievers-e29d4eecdb73))
- **Pinecone metadata filtering** and **Azure AI Search filters** are the production substrate for this — sparse metadata index alongside the dense index. ([techcommunity.microsoft.com/blog/azure-ai-foundry-blog/boost-rag-performance-enhance-vector-search-with-metadata-filters-in-azure-ai-search/4208985](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/boost-rag-performance-enhance-vector-search-with-metadata-filters-in-azure-ai-search/4208985))

### Practitioner consensus

Enterprise search products mostly treat ordinal/sequential filename tokens as **soft boost signals, not hard filters**. From Glean: "Consistent metadata gives every piece of content a set of reliable signals — signals that power filtering, ranking, and cross-repository discovery at scale." ([docs.glean.com/connectors/connectors-power-glean](https://docs.glean.com/connectors/connectors-power-glean)) But the practical recipe for "user mentions N, exactly one doc has N in its title" is the Elasticsearch one: a high `constant_score` boost on the exact match while still leaving the other signals in play. ([discuss.elastic.co/t/how-to-increase-score-for-exact-word-phrase-match-in-elastic-search/184901](https://discuss.elastic.co/t/how-to-increase-score-for-exact-word-phrase-match-in-elastic-search/184901))

Where this loops back into Q1: in our domain (educational materials), "user mentions N AND exactly one corpus doc has N in its filename" is empirically a near-deterministic direct match. This is the cleanest "high-confidence direct-match signal" the router in Q1 needs.

### Verdict — apply (carefully)

Apply as a **soft routing key, not a hard filter** — burned once already by hard filters built on unreliable doc_type labels. Two specific moves the research supports:

1. Add an explicit numeric-token feature to the filename-overlap signal. When the query contains a number (extracted via regex) AND exactly one doc has that number in its filename, set the filename-overlap score for that doc to a constant-score boost (~3× the next-highest filename-overlap score) — i.e. the Elasticsearch recipe. This makes the signal a routing key for Q1's short-circuit.
2. Do **not** use it as a hard SQL filter. If the query says "Lecture 6" and no doc has 6, fall back to the existing fusion — never zero out the result set on a soft signal.

This is consistent with the broader practitioner message (Glean, Haystack, Pinecone): metadata is a ranking and routing signal, hard filters require absolute confidence in the metadata's correctness.

---

## Q3. LLM-based intra-document structural navigation

### What the literature says

This question has the **most active recent research**, all converging on hierarchical/structural representations:

- **RAPTOR (Sarthi et al., ICLR 2024).** Recursively clusters and summarises chunks into a tree; retrieval traverses the tree. The summaries at each node act as a learned table-of-contents. ([arxiv.org/abs/2401.18059](https://arxiv.org/abs/2401.18059), [deeplearning.ai/the-batch/raptor](https://www.deeplearning.ai/the-batch/raptor-a-recursive-summarizer-captures-more-relevant-context-for-llm-inputs))
- **HiChunk (2025).** Explicitly evaluates hierarchical chunking for RAG; preserves section structure as first-class. ([arxiv.org/pdf/2509.11552](https://arxiv.org/pdf/2509.11552))
- **SF-RAG (Structure-Fidelity RAG, 2026).** Directly on-point: "current RAG methods flatten documents into unordered chunks, losing section signals; SF-RAG constructs a structure-fidelity index that preserves the native outline and performs path-guided retrieval." ([arxiv.org/pdf/2602.13647](https://arxiv.org/pdf/2602.13647))
- **PAGER / Structured Knowledge Representation (2026).** LLM constructs a structured cognitive outline of the question, then retrieves to fill each slot. ([arxiv.org/pdf/2601.09402](https://arxiv.org/pdf/2601.09402))
- **Small-to-big / parent-child retrieval (LlamaIndex).** Index small precise chunks but return their parent (section or page) at generation time. ([medium.com/data-science/advanced-rag-01-small-to-big-retrieval-172181b396d4](https://medium.com/data-science/advanced-rag-01-small-to-big-retrieval-172181b396d4), [developers.llamaindex.ai/python/framework-api-reference/node_parsers/hierarchical/](https://developers.llamaindex.ai/python/framework-api-reference/node_parsers/hierarchical/))
- **Anthropic Contextual Retrieval.** Prepends LLM-generated context (often section/structural) to each chunk *at ingest time*, not query time. 49% reduction in retrieval failure on its own, 67% with reranking. ([anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval))

### Practitioner consensus

The split between per-query LLM resolution vs. precomputed structure is real, and the dominant practitioner answer is **precompute the structure, resolve at query time with a small step**:

- Databricks, Weaviate, and Firecrawl all advocate hierarchical chunking for documents with native outlines (textbooks, contracts, technical specs). ([community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089), [weaviate.io/blog/chunking-strategies-for-rag](https://weaviate.io/blog/chunking-strategies-for-rag), [firecrawl.dev/blog/best-chunking-strategies-rag](https://www.firecrawl.dev/blog/best-chunking-strategies-rag))
- **Anthropic's empirical result is the strongest single argument** for moving structure into ingestion rather than query time: precomputed structural context at ingest gets ~half of the failure reduction "for free" at retrieval, before any cleverness at query time. ([anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval))
- **Cost analysis from the SF-RAG and RAPTOR write-ups is consistent**: per-query LLM calls to navigate a long document do not scale; precomputed indexes do.

### Verdict — apply, but with the precompute-first bias

Apply, but reject the "per-query LLM reads the whole doc to find Section II part b" framing. The research is clear: do the structural extraction **once at ingest**, store section paths as chunk metadata (`section_path: ["Chapter 7", "II", "(b)"]`), and at query time use a tiny LLM call (or even regex) to translate "Section II part b" into a path-filter over the existing metadata.

This is the same pattern as our current `chunk_context` for slides/pages — just generalised to the outline tree.

---

## Recommended design changes to `src/retriever.py`

Tying the three findings into concrete moves, in priority order:

### 1. Numeric / filename direct-match short-circuit (Q1 + Q2)

Add a pre-fusion stage in `retrieve_context`:

- Extract numeric tokens from the query (regex `\b\d{1,3}\b` + ordinal words: "first", "second", …).
- For each corpus doc, compute a `filename_match_score` that combines the existing token overlap with a large constant-score bonus when a query numeric token matches a numeric token in the filename.
- If exactly one doc scores above threshold T (calibrate empirically; start at "3× the median filename-overlap score across the corpus") AND no other doc is within 0.5× of it, **bypass RRF**: pin that doc as the retrieval anchor and pull its top-k chunks via dense alone.
- Otherwise, fall through to the existing RRF fusion unchanged.

This implements the Q1 "high-confidence short-circuit" using Q2's numeric signal as the trigger. Cost: zero additional LLM calls.

### 2. Optional LLM router behind a feature flag (Q1)

For queries where Tier 1 doesn't fire but classification is ambiguous (e.g. token overlap exists but is not dominant), gate a single cheap classifier call ("direct lookup / conceptual / mixed") behind a flag. Use the LlamaIndex Pydantic-selector shape — function-calling for reliability. Default the flag OFF until we have data showing Tier 1 leaves a meaningful gap.

### 3. Section-path metadata at ingest, path-resolution at query (Q3)

In `src/document_processor.py`'s chunker, when a document has detectable structure (numbered sections, parts, questions), store `section_path` as a list field on each `DocumentChunk`. At query time in `retriever.py`:

- Extract structural references from the query ("Section II part b", "Question 3a") via the same LLM router from change #2, or a regex first-pass.
- If extraction returns a path, apply it as a soft filter (boost matching chunks ~3×, do not exclude non-matching).
- This generalises today's slide/page handling rather than replacing it.

### 4. Adopt Anthropic-style contextual prefixing at ingest (Q3 supporting)

Independently of the above, the Anthropic result suggests prepending an LLM-generated 1-2 sentence "this chunk is from Section II part b of the final, which is about X" to each chunk before embedding. This is an ingest-time change in `document_processor.py`, not retriever logic, but it directly attacks the Q3 failure class (chunks that lack structural context lose their section identity once vectorised). One-time cost, persistent recall improvement.

---

## What the literature doesn't tell us

Concrete gaps where we'll be making judgement calls:

- **Calibration thresholds for the filename-overlap signal.** No paper gives "X% normalised overlap = treat as direct match." The Elasticsearch "3× boost" heuristic is the strongest signal but isn't validated on educational-material naming conventions. We'll need to backtest against the existing prod QA logs.
- **The interaction between filename-numeric short-circuit and student queries that reference the wrong number.** If a student says "Lecture 6" but means "Lecture 5" (because the syllabus shifted), a hard short-circuit will return the wrong doc with high confidence. Literature doesn't address this. Suggest: in the LLM reranker step after short-circuit, allow it to reject the anchor doc and fall back to fusion if context doesn't match.
- **Whether educational documents have reliable enough structure for path-based filtering.** RAPTOR/SF-RAG assume legal contracts or textbooks with clean hierarchies. Lecture slides, problem sets, and instructor notes vary wildly in how cleanly numbered they are. We may need a per-doc-type capability flag ("this doc has clean structure → path filtering enabled").
- **The exact prompt for the LLM router.** LlamaIndex's selectors are generic. The taxonomy that matters for Maize ("direct doc lookup / conceptual question / cross-doc synthesis / structural navigation within a known doc") is domain-specific and will need empirical tuning.

These are first-principles calls we'll make from the QA logs once the Q1+Q2 short-circuit is in.
