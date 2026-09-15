# Retrieval scorecard

**Total rows:** 250 (250 in-corpus + 0 not-in-corpus). **Errors:** 0.
**Reranker:** `cohere` · **low-confidence action:** `collapse` · **cache reuse:** `True` · **rerank query:** `raw`
**TAs in this run:** 4 — `EgZ14pvqEYzfQRTM`, `WBNtFkfPGZaJVQIk`, `iDYis09JtNUkyEJJ`, `z_B4fFY6jD1mhy9K`. Cross-TA aggregate scoring; re-run with `--ta-id <id>` to scope to one TA.

## Openers vs follow-ups (bucket_hit)

| Bucket | openers n | hit | collapsed | p50 / p95 ms | follow-ups n | hit | collapsed | p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **all in-corpus** | 113 | 56% | 63% | 3456 / 4885 | 137 | 49% | 82% | 2508 / 4458 |
| A | 16 | 88% | 100% | 3680 / 7572 | 0 | — | — | — / — |
| B | 14 | 57% | 57% | 3732 / 4386 | 7 | 100% | 100% | 2228 / 2489 |
| C | 15 | 80% | 53% | 3968 / 4640 | 3 | 0% | 67% | 2343 / 4266 |
| E | 0 | — | — | — / — | 22 | 45% | 86% | 2887 / 4117 |
| F1 | 0 | — | — | — / — | 15 | 47% | 47% | 2688 / 4477 |
| F2 | 0 | — | — | — / — | 36 | 36% | 83% | 2979 / 4646 |
| H | 15 | 20% | 53% | 3000 / 5242 | 0 | — | — | — / — |
| I | 0 | — | — | — / — | 15 | 27% | 93% | 2230 / 5076 |
| K | 0 | — | — | — / — | 15 | 67% | 93% | 2062 / 6238 |
| L | 16 | 12% | 75% | 2689 / 4885 | 0 | — | — | — / — |
| M | 12 | 8% | 92% | 2534 / 4156 | 0 | — | — | — / — |
| working | 25 | 92% | 32% | 4027 / 5105 | 24 | 67% | 79% | 2408 / 4316 |

## Doc-routing buckets (Wave 1)

| Failure type | n | hit@5 pre→post (lift) | collapsed_to_full_doc | hard_neg_top1 | forbidden_hit | avg_latency_ms | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (Lab vs PS) | 16 | 81%→88% (+6%) | 100% | 6% | 0% | 3880 | 0 |
| B (Roman numeral siblings, cross-doc) | 21 | 43%→71% (+29%) | 71% | 0% | 0% | 3041 | 0 |
| C (lookalike-unrelated) | 18 | 72%→67% (-6%) | 56% | 0% | 0% | 3585 | 0 |
| E (cache anchoring) | 22 | 14%→45% (+32%) | 86% | 14% | 0% | 2822 | 0 |
| F1 (explicit conceptual switch) | 15 | 40%→47% (+7%) | 47% | 40% | 0% | 2753 | 0 |
| F2 (explicit document switch) | 36 | 33%→47% (+14%) | 83% | 8% | 0% | 3091 | 0 |
| working cases | 49 | 55%→80% (+24%) | 55% | 8% | 0% | 3271 | 0 |

## Intent-classification buckets (Wave 2)

| Failure type | n | bucket_hit | hit@5 (doc-routing) | all_correct_in_top_5 (H only) | intent_class_match | redirect_fired (L only) | avg_chunks_returned | avg_latency_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H (multi-document intent) | 15 | 20% | 87% | 20% | — | — | 4.5 | 3097 |
| I (document correction) | 15 | 27% | 27% | — | — | — | 2.3 | 2496 |
| K (followup/clarification) | 15 | 67% | 0% | — | 67% | — | 1.5 | 2257 |
| L (off-topic / redirect) | 16 | 12% | 0% | — | 12% | 12% | 2.0 | 2894 |
| M (coverage gap / acknowledge) | 12 | 8% | 0% | — | 20% | — | 1.6 | 2504 |

**Answer-good-without-direct-source:** 4 rows the labeler marked as having produced a good production answer even though no document directly answered them. 1/4 score a retrieval hit here — the remainder are rows where our metrics say MISS but the student was served correctly. Treat them as a ceiling on how much retrieval improvement is actually available, not as failures to fix.

**Not-in-corpus bucket:** 0 rows. Retrieved a labeled-correct doc on 0/0 (expected 0 — these docs are unverified in the corpus). Forbidden-hit rate: 0%.

## Metric definitions
- **hit@5 pre→post (lift)** — pre-rerank hit-rate → post-rerank hit-rate, with the rerank's contribution as a `(±X%)` delta. Positive = rerank moved correct chunks into top-5 that weren't there before. Negative = rerank pushed correct chunks out. Zero = rerank didn't affect top-5 membership.
- **correct_hit@5** — fraction of rows where at least one `correct_doc_ids` appeared in retrieved top-5 (post-rerank).
- **hard_negative_top1** — fraction of rows where the retrieved top-1 doc matched a known hard negative (i.e., the current bad retrieval pattern fired).
- **forbidden_hit** — fraction of rows where ANY retrieved doc was on the forbidden list (e.g., solutions doc returned when student is solving). Lower is better; ideal = 0%.
- **bucket_hit** (Wave 2) — the primary HIT signal for a row, depends on its `expected_action`: for `retrieve` rows it equals hit@5; for `redirect` rows it requires `adversarial_short_circuit` fired AND zero chunks returned; for `no_retrieval` rows it's a proxy via `intent == 'clarification'` today (becomes a true skip-gate metric post-LangGraph adaptation).
- **all_correct_in_top_5** (H bucket) — stricter than hit@5: requires EVERY `correct_doc_ids` entry to appear in top-5, not just one. Tests whether multi-doc intent surfaces ALL needed docs.
- **intent_class_match** — fraction of rows (with `expected_intent.intent_class` labeled) where the contextualizer's classification matches the label. Measures intent-classification accuracy independently of retrieval — Q1+Q2 deep-research flagged this as a literature gap; doing this puts Maize ahead of published practice.
- **collapsed_to_full_doc** — fraction of rows where `hybrid_fallback_triggered` fired: the router gave up on its shortlist and expanded ONE whole document to full text. This is the Path 4 collapse. Lower is better; for M (coverage-gap) rows it IS the bucket_hit rule inverted.
- **redirect_fired** (L bucket) — fraction of rows where `adversarial_short_circuit` fired in diagnostics, regardless of whether chunks were also returned.
- **avg_chunks_returned** — average number of chunks the retriever returned. For K/L rows the IDEAL value is 0 (system should skip retrieval). Useful as a smoke check that the skip-gate is firing.
