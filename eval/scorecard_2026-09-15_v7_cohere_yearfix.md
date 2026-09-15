# Retrieval scorecard

**Total rows:** 250 (250 in-corpus + 0 not-in-corpus). **Errors:** 0.
**Reranker:** `cohere` · **low-confidence action:** `collapse` · **cache reuse:** `True` · **rerank query:** `raw`
**TAs in this run:** 4 — `EgZ14pvqEYzfQRTM`, `WBNtFkfPGZaJVQIk`, `iDYis09JtNUkyEJJ`, `z_B4fFY6jD1mhy9K`. Cross-TA aggregate scoring; re-run with `--ta-id <id>` to scope to one TA.

## Openers vs follow-ups (bucket_hit)

| Bucket | openers n | hit | collapsed | p50 / p95 ms | follow-ups n | hit | collapsed | p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **all in-corpus** | 113 | 58% | 60% | 3552 / 5389 | 137 | 53% | 82% | 2810 / 5262 |
| A | 16 | 88% | 100% | 3783 / 6127 | 0 | — | — | — / — |
| B | 14 | 57% | 50% | 4296 / 5212 | 7 | 100% | 100% | 2357 / 2810 |
| C | 15 | 80% | 53% | 4127 / 5196 | 3 | 100% | 100% | 1703 / 2216 |
| E | 0 | — | — | — / — | 22 | 45% | 86% | 2951 / 4335 |
| F1 | 0 | — | — | — / — | 15 | 47% | 47% | 2812 / 5623 |
| F2 | 0 | — | — | — / — | 36 | 39% | 81% | 3230 / 6019 |
| H | 15 | 20% | 53% | 3130 / 5077 | 0 | — | — | — / — |
| I | 0 | — | — | — / — | 15 | 27% | 93% | 2579 / 4778 |
| K | 0 | — | — | — / — | 15 | 60% | 93% | 1815 / 5145 |
| L | 16 | 12% | 75% | 2691 / 6014 | 0 | — | — | — / — |
| M | 12 | 8% | 92% | 2869 / 4505 | 0 | — | — | — / — |
| working | 25 | 100% | 24% | 4257 / 5353 | 24 | 75% | 79% | 2940 / 5319 |

## Doc-routing buckets (Wave 1)

| Failure type | n | hit@5 pre→post (lift) | collapsed_to_full_doc | hard_neg_top1 | forbidden_hit | avg_latency_ms | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (Lab vs PS) | 16 | 81%→88% (+6%) | 100% | 6% | 0% | 4138 | 0 |
| B (Roman numeral siblings, cross-doc) | 21 | 48%→71% (+24%) | 67% | 0% | 0% | 3363 | 0 |
| C (lookalike-unrelated) | 18 | 72%→83% (+11%) | 61% | 0% | 0% | 3717 | 0 |
| E (cache anchoring) | 22 | 14%→45% (+32%) | 86% | 14% | 0% | 3039 | 0 |
| F1 (explicit conceptual switch) | 15 | 40%→47% (+7%) | 47% | 40% | 0% | 3010 | 0 |
| F2 (explicit document switch) | 36 | 36%→50% (+14%) | 81% | 6% | 0% | 3387 | 0 |
| working cases | 49 | 59%→88% (+29%) | 51% | 8% | 0% | 3649 | 0 |

## Intent-classification buckets (Wave 2)

| Failure type | n | bucket_hit | hit@5 (doc-routing) | all_correct_in_top_5 (H only) | intent_class_match | redirect_fired (L only) | avg_chunks_returned | avg_latency_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H (multi-document intent) | 15 | 20% | 87% | 20% | — | — | 4.5 | 3149 |
| I (document correction) | 15 | 27% | 27% | — | — | — | 2.3 | 2535 |
| K (followup/clarification) | 15 | 60% | 0% | — | 60% | — | 1.5 | 2169 |
| L (off-topic / redirect) | 16 | 12% | 0% | — | 12% | 12% | 2.0 | 2970 |
| M (coverage gap / acknowledge) | 12 | 8% | 0% | — | 20% | — | 1.6 | 2740 |

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
