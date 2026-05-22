# Baseline (CURRENT retriever)

**Total rows:** 64 (61 in-corpus + 3 not-in-corpus). **Errors:** 0.

| Failure type | n | correct_hit@5 | hard_negative_top1 | forbidden_hit | avg_latency_ms | errors |
|---|---:|---:|---:|---:|---:|---:|
| A (Lab vs PS) | 6 | 50% | 0% | 0% | 29062 | 0 |
| B (Roman numeral siblings) | 5 | 100% | 0% | 20% | 50410 | 0 |
| C (lookalike-unrelated) | 7 | 0% | 57% | 57% | 18837 | 0 |
| D (problem-vs-solutions) | 5 | 40% | 40% | 40% | 45534 | 0 |
| E (cache anchoring) | 21 | 86% | 0% | 5% | 27692 | 0 |
| working cases | 17 | 65% | 6% | 6% | 27622 | 0 |

**Not-in-corpus bucket:** 3 rows. Retrieved a labeled-correct doc on 0/3 (expected 0 — these docs are unverified in the corpus). Forbidden-hit rate: 100%.

## Metric definitions
- **correct_hit@5** — fraction of rows where at least one `correct_doc_ids` appeared in retrieved top-5.
- **hard_negative_top1** — fraction of rows where the retrieved top-1 doc matched a known hard negative (i.e., the current bad retrieval pattern fired).
- **forbidden_hit** — fraction of rows where ANY retrieved doc was on the forbidden list (e.g., solutions doc returned when student is solving). Lower is better; ideal = 0%.

## Interpretation caveats (added post-run)

This scorecard captures the CURRENT retriever's state against 64 labeled rows on the local ECON S1117 TA. Several numbers need contextual framing before being compared against the post-refactor scorecard.

### 1. Cold-cache caveat for Type E + multi-turn working cases

The harness uses a unique `session_id` per row, so the conversational cache (`session_context["document_filename"]` etc.) is always empty when each row's query runs. This means:

- **Type E (cache anchoring)** at 86% correct_hit@5 is misleadingly high. The actual prod failure requires a turn-1-populated cache that anchors to the wrong doc and refuses to release on subsequent corrections. The harness can't reproduce that without cache state. What it DOES measure: "given the conversation history alone (no cache), can the contextualizer + retriever recover the right doc?" The answer is mostly yes (86%) — but that's not the same as "cache anchoring is solved."
- **Multi-turn working cases** are penalized by this. Working continuations like "now help with problem 3" rely on T1's cache to know what doc the student is on. With cold cache, T2 retrieval has only the history to go on — sometimes that's enough, sometimes not. This contributes to the working-cases 65% rate.

**Implication for the refactor:** Phase A's per-turn query rewriter + per-turn re-retrieval structurally eliminates the cache-anchoring failure mode regardless of cache state. So Type E's misleading high will continue to look high after refactor (now legitimately so). Verify cache-anchoring fix in a real browser session, not via this harness.

### 2. Type B local-vs-prod difference

Type B (Roman-numeral siblings — "extra problems I" vs "extra problems II") scores 100% correct_hit@5 locally, but the prod logs that motivated this failure type showed reliable misretrieval. Likely explanation: the local re-index after the poppler fix produced cleaner chunk metadata than prod's older index. The local TA may not exhibit the failure mode the eval was designed to catch.

**Implication:** the eval set should be re-validated against prod once the refactor ships; prod may still exhibit Type B failures the local eval doesn't catch.

### 3. Solutions docs — `forbidden_hit` metric is too strict for the refined design

The original Phase A design treated solutions docs as wholly forbidden from retrieval. After discussion (2026-05-22), the design changed: solutions are USEFUL as supplementary reference material the LLM can use to construct higher-quality Socratic guidance — they just shouldn't be DIVULGED to the student directly. That pedagogy is enforced via the LLM system prompt, not via retrieval filtering.

Under the refined design:
- Retrieval is allowed to bring solutions chunks into context.
- Source attribution / citation should NOT list the solutions doc as primary.
- LLM prompt instruction: solutions chunks tagged `[REFERENCE / DO NOT DIVULGE]`; use them to inform Socratic responses, don't reveal them.

The `forbidden_hit` metric in this scorecard counts ANY appearance of a forbidden doc in retrieved top-K — which now mis-measures the desired behavior. A cleaner metric for the refined design:

- **`primary_is_forbidden`** — was the TOP-1 retrieved doc a forbidden doc? (e.g., did the system cite a solutions doc for a problem-solving query?)

When we generate the post-refactor scorecard, we'll add this metric. The current baseline numbers stand; only the interpretation of `forbidden_hit` changes — it's now best read as "how often does ANY solutions-leak happen," and the goal becomes "primary attribution stays correct" rather than "no solutions present in context."

### 4. Type C is the cleanest failure signal

Type C (lookalike-but-unrelated) at 0% correct_hit / 57% forbidden_hit is the most reliable signal — the current retriever cannot find Quiz 1 for "quiz 1" queries and instead returns Quiz 2 solutions or final exams. This is the load-bearing failure the Phase A refactor must fix. Watch this row carefully in the post-refactor scorecard.
