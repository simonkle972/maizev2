# Retrieval scorecard

**Total rows:** 250 (250 in-corpus + 0 not-in-corpus). **Errors:** 4.
**Reranker:** `cohere` · **low-confidence action:** `collapse` · **cache reuse:** `True` · **rerank query:** `raw` · **cache as prior:** `True`
**Config overrides (--set):** `CACHE_AS_PRIOR_ENABLED=True`
**TAs in this run:** 4 — `EgZ14pvqEYzfQRTM`, `WBNtFkfPGZaJVQIk`, `iDYis09JtNUkyEJJ`, `z_B4fFY6jD1mhy9K`. Cross-TA aggregate scoring; re-run with `--ta-id <id>` to scope to one TA.

## Openers vs follow-ups (bucket_hit)

| Bucket | openers n | hit | collapsed | p50 / p95 ms | follow-ups n | hit | collapsed | p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **all in-corpus** | 113 | 58% | 60% | 4502 / 7927 | 137 | 63% | 39% | 5808 / 18103 |
| A | 16 | 88% | 100% | 4025 / 7176 | 0 | — | — | — / — |
| B | 14 | 57% | 50% | 4414 / 5247 | 7 | 71% | 29% | 3492 / 4807 |
| C | 15 | 80% | 53% | 4552 / 8061 | 3 | 67% | 33% | 3736 / 4791 |
| E | 0 | — | — | — / — | 22 | 77% | 50% | 5802 / 7790 |
| F1 | 0 | — | — | — / — | 15 | 53% | 13% | 11520 / 603957 |
| F2 | 0 | — | — | — / — | 36 | 58% | 47% | 7515 / 14051 |
| H | 15 | 20% | 53% | 6465 / 10458 | 0 | — | — | — / — |
| I | 0 | — | — | — / — | 15 | 53% | 53% | 6352 / 12985 |
| K | 0 | — | — | — / — | 15 | 60% | 47% | 5716 / 9471 |
| L | 16 | 12% | 75% | 3227 / 9702 | 0 | — | — | — / — |
| M | 12 | 8% | 92% | 5146 / 9714 | 0 | — | — | — / — |
| working | 25 | 100% | 24% | 4502 / 7015 | 24 | 67% | 25% | 4819 / 6991 |

## Doc-routing buckets (Wave 1)

| Failure type | n | hit@5 pre→post (lift) | collapsed_to_full_doc | hard_neg_top1 | forbidden_hit | avg_latency_ms | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (Lab vs PS) | 16 | 81%→88% (+6%) | 100% | 6% | 0% | 4100 | 0 |
| B (Roman numeral siblings, cross-doc) | 21 | 71%→62% (-10%) | 43% | 0% | 0% | 3837 | 0 |
| C (lookalike-unrelated) | 18 | 89%→78% (-11%) | 50% | 0% | 0% | 4529 | 0 |
| E (cache anchoring) | 22 | 77%→77% (+0%) | 50% | 14% | 14% | 6714 | 0 |
| F1 (explicit conceptual switch) | 15 | 53%→53% (+0%) | 13% | 7% | 0% | 71941 | 4 |
| F2 (explicit document switch) | 36 | 69%→69% (+0%) | 47% | 6% | 0% | 8156 | 0 |
| working cases | 49 | 88%→84% (-4%) | 24% | 6% | 0% | 4698 | 0 |

## Intent-classification buckets (Wave 2)

| Failure type | n | bucket_hit | hit@5 (doc-routing) | all_correct_in_top_5 (H only) | intent_class_match | redirect_fired (L only) | avg_chunks_returned | avg_latency_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H (multi-document intent) | 15 | 20% | 87% | 20% | — | — | 4.5 | 5691 |
| I (document correction) | 15 | 53% | 53% | — | — | — | 5.1 | 6588 |
| K (followup/clarification) | 15 | 60% | 0% | — | 60% | — | 4.7 | 5488 |
| L (off-topic / redirect) | 16 | 12% | 0% | — | 12% | 12% | 2.0 | 3963 |
| M (coverage gap / acknowledge) | 12 | 8% | 0% | — | 20% | — | 1.6 | 5257 |

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
