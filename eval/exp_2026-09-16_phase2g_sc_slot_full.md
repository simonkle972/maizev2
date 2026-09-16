# Retrieval scorecard

**Total rows:** 250 (250 in-corpus + 0 not-in-corpus). **Errors:** 0.
**Reranker:** `cohere` · **low-confidence action:** `widen` · **cache reuse:** `True` · **rerank query:** `raw` · **cache as prior:** `True` · **contextualizer v2:** `True`
**Config overrides (--set):** `SHORT_CIRCUIT_AS_SLOT=True`
**TAs in this run:** 4 — `EgZ14pvqEYzfQRTM`, `WBNtFkfPGZaJVQIk`, `iDYis09JtNUkyEJJ`, `z_B4fFY6jD1mhy9K`. Cross-TA aggregate scoring; re-run with `--ta-id <id>` to scope to one TA.

## Openers vs follow-ups (bucket_hit)

| Bucket | openers n | hit | collapsed | p50 / p95 ms | follow-ups n | hit | collapsed | p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **all in-corpus** | 113 | 83% | 19% | 2308 / 3735 | 137 | 78% | 9% | 2402 / 4339 |
| A | 16 | 88% | 62% | 2568 / 4951 | 0 | — | — | — / — |
| B | 14 | 57% | 36% | 2361 / 7449 | 7 | 100% | 0% | 2849 / 2869 |
| C | 15 | 87% | 20% | 2583 / 3477 | 3 | 100% | 33% | 3823 / 4376 |
| E | 0 | — | — | — / — | 22 | 91% | 9% | 2384 / 4261 |
| F1 | 0 | — | — | — / — | 15 | 67% | 0% | 2020 / 4253 |
| F2 | 0 | — | — | — / — | 36 | 69% | 17% | 2687 / 4002 |
| H | 15 | 67% | 0% | 2319 / 4032 | 0 | — | — | — / — |
| I | 0 | — | — | — / — | 15 | 87% | 13% | 2422 / 4529 |
| K | 0 | — | — | — / — | 15 | 87% | 0% | 1462 / 4252 |
| L | 17 | 82% | 0% | 1456 / 3157 | 0 | — | — | — / — |
| M | 12 | 100% | 0% | 2139 / 2853 | 0 | — | — | — / — |
| working | 24 | 96% | 17% | 2577 / 3647 | 24 | 67% | 4% | 2139 / 5420 |

## Doc-routing buckets (Wave 1)

| Failure type | n | hit@5 pre→post (lift) | collapsed_to_full_doc | hard_neg_top1 | forbidden_hit | avg_latency_ms | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (Lab vs PS) | 16 | 75%→88% (+12%) | 62% | 0% | 0% | 2890 | 0 |
| B (Roman numeral siblings, cross-doc) | 21 | 67%→71% (+5%) | 24% | 5% | 0% | 2688 | 0 |
| C (lookalike-unrelated) | 18 | 83%→89% (+6%) | 22% | 11% | 0% | 2754 | 0 |
| E (cache anchoring) | 22 | 50%→91% (+41%) | 9% | 5% | 0% | 2627 | 0 |
| F1 (explicit conceptual switch) | 15 | 73%→67% (-7%) | 0% | 7% | 0% | 2131 | 0 |
| F2 (explicit document switch) | 36 | 67%→81% (+14%) | 17% | 0% | 0% | 2756 | 0 |
| working cases | 48 | 69%→81% (+12%) | 10% | 4% | 0% | 3043 | 0 |

## Intent-classification buckets (Wave 2)

| Failure type | n | bucket_hit | hit@5 (doc-routing) | all_correct_in_top_5 (H only) | intent_class_match | redirect_fired (L only) | avg_chunks_returned | avg_latency_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H (multi-document intent) | 15 | 67% | 100% | 67% | — | — | 8.0 | 2632 |
| I (document correction) | 15 | 87% | 87% | — | — | — | 7.9 | 2740 |
| K (followup/clarification) | 15 | 87% | 0% | — | 80% | — | 8.0 | 2013 |
| L (off-topic / redirect) | 17 | 82% | 0% | — | 82% | 82% | 1.6 | 1624 |
| M (coverage gap / acknowledge) | 12 | 100% | 0% | — | 90% | — | 8.0 | 2294 |

**Answer-good-without-direct-source:** 4 rows the labeler marked as having produced a good production answer even though no document directly answered them. 1/4 score a retrieval hit here — the remainder are rows where our metrics say MISS but the student was served correctly. Treat them as a ceiling on how much retrieval improvement is actually available, not as failures to fix.

**Not-in-corpus bucket:** 0 rows. Retrieved a labeled-correct doc on 0/0 (expected 0 — these docs are unverified in the corpus). Forbidden-hit rate: 0%.

## Metric definitions
- **hit@5 pre→post (lift)** — pre-rerank hit-rate → post-rerank hit-rate, with the rerank's contribution as a `(±X%)` delta. Positive = rerank moved correct chunks into top-5 that weren't there before. Negative = rerank pushed correct chunks out. Zero = rerank didn't affect top-5 membership.
- **correct_hit@5** — fraction of rows where at least one `correct_doc_ids` appeared in retrieved top-5 (post-rerank).
- **hard_negative_top1** — fraction of rows where the retrieved top-1 doc matched a known hard negative (i.e., the current bad retrieval pattern fired).
- **forbidden_hit** — fraction of rows where ANY retrieved doc was on the forbidden list (e.g., solutions doc returned when student is solving). Lower is better; ideal = 0%.
- **bucket_hit** (Wave 2) — the primary HIT signal for a row, depends on its `expected_action`: for `retrieve` rows it equals hit@5; for `redirect` rows it requires `adversarial_short_circuit` fired AND zero chunks returned; for `no_retrieval` rows (K) it passes when nothing was retrieved or every top-5 document is one the session was already working in -- pulling an UNRELATED document is the failure; with no recorded prior it falls back to `intent == 'clarification'`.
- **all_correct_in_top_5** (H bucket) — stricter than hit@5: requires EVERY `correct_doc_ids` entry to appear in top-5, not just one. Tests whether multi-doc intent surfaces ALL needed docs.
- **intent_class_match** — fraction of rows (with `expected_intent.intent_class` labeled) where the contextualizer's classification matches the label. Measures intent-classification accuracy independently of retrieval — Q1+Q2 deep-research flagged this as a literature gap; doing this puts Maize ahead of published practice.
- **collapsed_to_full_doc** — fraction of rows where `hybrid_fallback_triggered` fired: the router gave up on its shortlist and expanded ONE whole document to full text. This is the Path 4 collapse. Lower is better; for M (coverage-gap) rows it IS the bucket_hit rule inverted.
- **redirect_fired** (L bucket) — fraction of rows where `adversarial_short_circuit` fired in diagnostics, regardless of whether chunks were also returned.
- **avg_chunks_returned** — average number of chunks the retriever returned. For K/L rows the IDEAL value is 0 (system should skip retrieval). Useful as a smoke check that the skip-gate is firing.
