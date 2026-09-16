# Retrieval scorecard

**Total rows:** 250 (250 in-corpus + 0 not-in-corpus). **Errors:** 0.
**Reranker:** `cohere` · **low-confidence action:** `collapse` · **cache reuse:** `True` · **rerank query:** `raw` · **cache as prior:** `True` · **contextualizer v2:** `True`
**Config overrides (--set):** `CACHE_AS_PRIOR_ENABLED=True`, `CONTEXTUALIZER_V2_ENABLED=True`
**TAs in this run:** 4 — `EgZ14pvqEYzfQRTM`, `WBNtFkfPGZaJVQIk`, `iDYis09JtNUkyEJJ`, `z_B4fFY6jD1mhy9K`. Cross-TA aggregate scoring; re-run with `--ta-id <id>` to scope to one TA.

## Openers vs follow-ups (bucket_hit)

| Bucket | openers n | hit | collapsed | p50 / p95 ms | follow-ups n | hit | collapsed | p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **all in-corpus** | 113 | 70% | 47% | 4498 / 7096 | 137 | 72% | 32% | 5135 / 7798 |
| A | 16 | 94% | 94% | 5161 / 7778 | 0 | — | — | — / — |
| B | 14 | 57% | 50% | 5443 / 6303 | 7 | 100% | 29% | 3745 / 4872 |
| C | 15 | 73% | 47% | 5253 / 6407 | 3 | 100% | 0% | 4286 / 5911 |
| E | 0 | — | — | — / — | 22 | 91% | 41% | 6202 / 7427 |
| F1 | 0 | — | — | — / — | 15 | 73% | 7% | 3525 / 7094 |
| F2 | 0 | — | — | — / — | 36 | 67% | 33% | 5469 / 7798 |
| H | 15 | 33% | 47% | 4016 / 7096 | 0 | — | — | — / — |
| I | 0 | — | — | — / — | 15 | 93% | 33% | 4140 / 5606 |
| K | 0 | — | — | — / — | 15 | 0% | 67% | 3674 / 8982 |
| L | 16 | 100% | 0% | 2530 / 3447 | 0 | — | — | — / — |
| M | 12 | 8% | 92% | 2885 / 6070 | 0 | — | — | — / — |
| working | 25 | 92% | 24% | 5077 / 9163 | 24 | 79% | 21% | 4682 / 9140 |

## Doc-routing buckets (Wave 1)

| Failure type | n | hit@5 pre→post (lift) | collapsed_to_full_doc | hard_neg_top1 | forbidden_hit | avg_latency_ms | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (Lab vs PS) | 16 | 94%→94% (+0%) | 94% | 0% | 0% | 4964 | 0 |
| B (Roman numeral siblings, cross-doc) | 21 | 67%→71% (+5%) | 43% | 0% | 0% | 4486 | 0 |
| C (lookalike-unrelated) | 18 | 72%→78% (+6%) | 39% | 11% | 0% | 4872 | 0 |
| E (cache anchoring) | 22 | 86%→91% (+5%) | 41% | 5% | 0% | 6368 | 0 |
| F1 (explicit conceptual switch) | 15 | 73%→73% (+0%) | 7% | 0% | 0% | 4067 | 0 |
| F2 (explicit document switch) | 36 | 75%→78% (+3%) | 33% | 6% | 0% | 5263 | 0 |
| working cases | 49 | 71%→86% (+14%) | 22% | 6% | 0% | 5340 | 0 |

## Intent-classification buckets (Wave 2)

| Failure type | n | bucket_hit | hit@5 (doc-routing) | all_correct_in_top_5 (H only) | intent_class_match | redirect_fired (L only) | avg_chunks_returned | avg_latency_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H (multi-document intent) | 15 | 33% | 100% | 33% | — | — | 5.3 | 4472 |
| I (document correction) | 15 | 93% | 93% | — | — | — | 6.7 | 4021 |
| K (followup/clarification) | 15 | 0% | 0% | — | 0% | — | 3.9 | 4470 |
| L (off-topic / redirect) | 16 | 100% | 0% | — | 100% | 100% | 0.0 | 2568 |
| M (coverage gap / acknowledge) | 12 | 8% | 0% | — | 80% | — | 1.9 | 3251 |

**Answer-good-without-direct-source:** 4 rows the labeler marked as having produced a good production answer even though no document directly answered them. 2/4 score a retrieval hit here — the remainder are rows where our metrics say MISS but the student was served correctly. Treat them as a ceiling on how much retrieval improvement is actually available, not as failures to fix.

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
