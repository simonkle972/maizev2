# Maize TA — Latency Optimization Plan

**Created:** 2026-04-30 · **Status:** proposed, not yet started · **Goal:** ≥20% latency reduction on the typical student turn while holding answer quality constant.

## Context

We've focused on precision, accuracy, and in-session persistence so far. The result is a high-quality but slow pipeline (~25-30s per typical turn). This plan stages cuts so each layer can be measured before adding the next, and so we can stop at the first stage that delivers the goal if subsequent stages prove risky.

## Execution order (decided 2026-04-30)

Don't tackle stages strictly left-to-right. Sequence by leverage and risk:

1. **Stage 1D first** — instrumentation is mandatory. Without per-stage latency in the QA log we can't measure anything that follows.
2. **Stage 2A next** — single-line change, biggest single lever (3-6s), trivial revert. OpenAI's own GPT-5 guide treats `medium` as the default.
3. **Measure for ~50 real turns**, then decide:
   - If we're already at ≥25-30% reduction with quality holding, **stop** — 1A/1B/1C may not be worth the spend.
   - Otherwise, add **1A** (prompt-cache reorder, free quality-wise) and **1C** (skip rerank on session-cache follow-ups, big if follow-ups are common in our traffic).
   - **Skip 1B** unless we end up doing a broader async refactor anyway — 800ms isn't worth introducing asyncio/threading into an otherwise-sync Flask path.
4. **Stage 3** is the last lever — only pull it if Stages 1 + 2 don't get us there. Adds an external dependency.

## Baseline (typical non-short-circuit turn, ~25-30s)

| # | Step | Model / API | Latency | File:line |
|---|------|------------|---------|-----------|
| 1 | Moderation pre-filter | OpenAI omni-moderation | ~100ms | [src/retriever.py:1620](src/retriever.py#L1620) |
| 2 | Contextualizer | gpt-4o-mini, T=0 | ~1.0s | [src/retriever.py:1638](src/retriever.py#L1638) (defined [:1288](src/retriever.py#L1288)) |
| 3 | Initial vector retrieval (k=20) | text-embedding-3-small + pgvector | ~200ms | — |
| 4 | **LLM rerank 20→8** | **gpt-5.2 reasoning=medium** | **~2-4s** | [src/retriever.py:674](src/retriever.py#L674), call site [:729](src/retriever.py#L729), `reasoning_effort` at [:734](src/retriever.py#L734) |
| 5 | Supplementary concept extraction | gpt-4o-mini | ~800ms | conditional |
| 6 | Supplementary embedding + search | embedding + pgvector | ~250ms | conditional |
| 7 | **Generation** | **gpt-5.2 reasoning=high, ~10K-token prompt** | **~10-15s** | [src/response_generator.py:406, 465](src/response_generator.py#L406) |

Two gpt-5.2 calls dominate the wall clock; everything else combined is <2s.

## Stage 1 — Zero-risk wins (no quality test needed)

These are pure orchestration / prompt-shape changes. Estimated combined: **3-5s saved (~12-20%).**

### 1A. Reorder generation prompt to enable OpenAI prompt caching
- **Where:** [`src/response_generator.py`](src/response_generator.py) — wherever the generation prompt is assembled before the gpt-5.2 call (around lines 406/465).
- **Change:** Restructure the prompt so the *stable* portions come first (system prompt → BASE_INSTRUCTIONS → per-TA persona/course → conversation history) and the *dynamic* portions last (retrieved chunks → user query). OpenAI auto-caches stable prefixes ≥1024 tokens.
- **Expected:** 1-3s TTFT improvement on second-and-onward turns within a session. Up to 67% TTFT reduction on long prompts per OpenAI docs.
- **Quality risk:** None — output is identical, only the prompt order changes.

### 1B. Parallelize moderation + contextualizer
- **Where:** [src/retriever.py:1614-1660](src/retriever.py#L1614). Currently sequential (moderation runs first, then contextualizer).
- **Change:** Run both concurrently with `asyncio.gather` (or `concurrent.futures.ThreadPoolExecutor` if we don't want to async-ify the whole call site). Both look at the raw query; neither depends on the other.
- **Expected:** ~800ms saved per turn.
- **Quality risk:** None.

### 1C. Skip rerank on session-cache follow-ups
- **Where:** [src/retriever.py:729](src/retriever.py#L729) (rerank call). The session_context cache already holds the prior turn's reranked chunks.
- **Change:** When `session_context` is populated AND contextualizer intent ∈ {`continuation`, `clarification`}, reuse cached rerank scores instead of re-running rerank.
- **Expected:** 2-4s saved on follow-up turns (which are the majority of turns once a conversation gets going).
- **Quality risk:** None — cached chunks are by definition the ones that just satisfied the prior turn on the same problem.

### 1D. Instrument per-stage latency in QA logger
- **Where:** [`src/qa_logger.py`](src/qa_logger.py) headers + row construction. Diagnostics dict already carries some fields.
- **Change:** Add columns `moderation_latency_ms`, `contextualizer_latency_ms`, `rerank_latency_ms`, `generation_ttft_ms`, `generation_total_ms`, `prompt_cache_hit` (if exposed). Most of these need only a stopwatch around existing calls.
- **Why this is mandatory:** Without per-stage instrumentation, we can't tell which stage is responsible for any regression in Stage 2/3. Do this *before* Stage 2.

## Stage 2 — Reasoning effort A/B (quantifiable quality test)

### 2A. Drop generation `reasoning_effort` from `high` → `medium`
- **Where:** [src/response_generator.py:406](src/response_generator.py#L406) and [:465](src/response_generator.py#L465). Note: a `medium` fallback already exists at lines 421/480, so the model variant is exercised.
- **Procedure:**
  1. Pick 20-30 real student questions from the QA log spanning easy/medium/hard and conceptual/computational.
  2. Run each through both `high` and `medium` (use a feature flag or branch).
  3. Have the user grade pairwise (blind preference + obvious-quality-loss flag).
  4. If `medium` wins or ties on ≥80% of cases AND no "obvious quality loss" flags → ship.
- **Expected if shipped:** 3-6s saved per turn.
- **Quality risk:** Real but quantifiable. The OpenAI 2026 GPT-5 guide explicitly says `medium` is the default and `high` should be reserved for multi-step planning / tool-heavy workflows — teaching responses don't fit that profile.
- **Revert plan:** Single line change. Trivial.

## Stage 3 — Reranker swap (architectural, biggest single win)

### 3A. Replace gpt-5.2 LLM rerank with a specialized reranker
- **Where:** [src/retriever.py:674-770](src/retriever.py#L674) (`llm_rerank` function and call site).
- **Vendor candidates** (decide after Stage 2 numbers come in):
  - **Cohere Rerank 3.5** — ~600ms, hosted API, ~$2/1k queries. Easiest swap.
  - **Voyage rerank-2.5** — competitive accuracy, hosted API.
  - **BGE reranker v2** (self-hosted) — cheapest at scale, requires GPU infra.
- **Expected:** 2-3s saved on turns that rerank.
- **Quality risk:** ~5% NDCG@10 gap to LLM rerank on adversarial benchmarks per 2026 reports; for our domain (homework retrieval, well-curated course material) likely much smaller. The 2026 consensus across multiple sources is that LLM reranking is the wrong tool for this candidate-set size and stakes profile.
- **Validation procedure:** Same A/B pairwise grading as Stage 2A on a held-out set of student questions, plus offline NDCG@10 on a labeled set if we want to be thorough.
- **Side benefit:** Removes one gpt-5.2 dependency, lowering cost-per-turn meaningfully.

## Combined target

| Stage | Cumulative savings | % of 25-30s baseline |
|-------|-------------------|---------------------|
| Stage 1 alone | 3-5s | ~12-20% |
| Stage 1+2 | 6-9s | ~25-35% |
| Stage 1+2+3 | 9-13s | ~35-50% |

Stage 1 hits the ≥20% goal on its own under favorable conditions. Stage 2 is the bigger lever; Stage 3 is the highest single win but adds an external dependency.

## Verification (per stage)

- **Stage 1:** Compare median per-turn latency in QA log over 50 turns before vs. after. No quality grading needed.
- **Stage 2:** Pairwise blind quality grading on 20-30 questions + median latency comparison.
- **Stage 3:** Same as Stage 2 + offline NDCG@10 on labeled retrieval set if available.

## Open questions

- Does our OpenAI account tier allow prompt caching? (Default for paid accounts since mid-2025; verify in dashboard.)
- Do we want the asyncio refactor (Stage 1B) to ripple beyond the moderation/contextualizer pair, or keep it surgical via a thread pool? Surgical is faster to ship; full async unlocks more parallelism (e.g. Stage 1B + supplementary concept extraction overlap).
- Stage 3 vendor decision — defer until we've measured Stage 1+2 in prod. We may not need Stage 3 if we're already at ≥30%.

## Sources informing this plan

- [Reranking in RAG: Cross-Encoders, Cohere Rerank & FlashRank — Mar 2026](https://medium.com/@vaibhav-p-dixit/reranking-in-rag-cross-encoders-cohere-rerank-flashrank-c7d40c685f6a)
- [Ultimate Guide to Choosing the Best Reranking Model in 2026 — ZeroEntropy](https://zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025/)
- [Should You Use an LLM as a Reranker? Pros, Cons, and Benchmarks — ZeroEntropy](https://www.zeroentropy.dev/articles/llm-as-reranker-guide)
- [Best Reranker Models for RAG — BSWEN, Feb 2026](https://docs.bswen.com/blog/2026-02-25-best-reranker-models/)
- [Does Adding a Reranker to RAG Increase Latency? — BSWEN, Feb 2026](https://docs.bswen.com/blog/2026-02-25-reranker-latency-impact/)
- [OpenAI Prompt Caching Guide](https://developers.openai.com/api/docs/guides/prompt-caching)
- [OpenAI GPT-5 Prompting Guide (cookbook)](https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide)
- [Using GPT-5.2 — OpenAI](https://platform.openai.com/docs/guides/latest-model)
- [RAG Latency Without the Usual Trade-Offs — Mar 2026](https://medium.com/@Nexumo_/rag-latency-without-the-usual-trade-offs-34a52107f0ca)
