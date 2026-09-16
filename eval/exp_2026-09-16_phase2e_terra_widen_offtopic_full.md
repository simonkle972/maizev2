# Retrieval scorecard

**Total rows:** 250 (250 in-corpus + 0 not-in-corpus). **Errors:** 0.
**Reranker:** `cohere` · **low-confidence action:** `widen` · **cache reuse:** `True` · **rerank query:** `raw` · **cache as prior:** `True` · **contextualizer v2:** `True`
**Config overrides (--set):** `LOW_CONFIDENCE_ACTION='widen'`, `CONTEXTUALIZER_MODEL='gpt-5.6-terra'`, `OFFTOPIC_COURSE_SUMMARY_ENABLED=True`
**TAs in this run:** 4 — `EgZ14pvqEYzfQRTM`, `WBNtFkfPGZaJVQIk`, `iDYis09JtNUkyEJJ`, `z_B4fFY6jD1mhy9K`. Cross-TA aggregate scoring; re-run with `--ta-id <id>` to scope to one TA.

## Openers vs follow-ups (bucket_hit)

| Bucket | openers n | hit | collapsed | p50 / p95 ms | follow-ups n | hit | collapsed | p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **all in-corpus** | 113 | 84% | 22% | 3427 / 8088 | 137 | 79% | 15% | 3627 / 8726 |
| A | 16 | 94% | 75% | 3541 / 11163 | 0 | — | — | — / — |
| B | 14 | 57% | 29% | 3851 / 8075 | 7 | 100% | 0% | 4237 / 6809 |
| C | 15 | 87% | 33% | 3176 / 18005 | 3 | 100% | 0% | 4933 / 5824 |
| E | 0 | — | — | — / — | 22 | 91% | 27% | 4019 / 7330 |
| F1 | 0 | — | — | — / — | 15 | 67% | 13% | 2897 / 11485 |
| F2 | 0 | — | — | — / — | 36 | 69% | 19% | 3706 / 11597 |
| H | 15 | 53% | 0% | 3544 / 7871 | 0 | — | — | — / — |
| I | 0 | — | — | — / — | 15 | 87% | 13% | 3185 / 9948 |
| K | 0 | — | — | — / — | 15 | 87% | 0% | 2422 / 8860 |
| L | 16 | 94% | 0% | 2129 / 6721 | 0 | — | — | — / — |
| M | 12 | 100% | 0% | 3603 / 5879 | 0 | — | — | — / — |
| working | 25 | 96% | 16% | 3813 / 8590 | 24 | 71% | 17% | 3138 / 8726 |

## Doc-routing buckets (Wave 1)

| Failure type | n | hit@5 pre→post (lift) | collapsed_to_full_doc | hard_neg_top1 | forbidden_hit | avg_latency_ms | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (Lab vs PS) | 16 | 88%→94% (+6%) | 75% | 0% | 0% | 4956 | 0 |
| B (Roman numeral siblings, cross-doc) | 21 | 76%→71% (-5%) | 19% | 5% | 0% | 4461 | 0 |
| C (lookalike-unrelated) | 18 | 83%→89% (+6%) | 28% | 0% | 0% | 4485 | 0 |
| E (cache anchoring) | 22 | 82%→91% (+9%) | 27% | 5% | 0% | 4577 | 0 |
| F1 (explicit conceptual switch) | 15 | 67%→67% (+0%) | 13% | 7% | 0% | 3539 | 0 |
| F2 (explicit document switch) | 36 | 83%→81% (-3%) | 19% | 0% | 0% | 4232 | 0 |
| working cases | 49 | 82%→84% (+2%) | 16% | 6% | 0% | 4908 | 0 |

## Intent-classification buckets (Wave 2)

| Failure type | n | bucket_hit | hit@5 (doc-routing) | all_correct_in_top_5 (H only) | intent_class_match | redirect_fired (L only) | avg_chunks_returned | avg_latency_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H (multi-document intent) | 15 | 53% | 93% | 53% | — | — | 8.5 | 4279 |
| I (document correction) | 15 | 87% | 87% | — | — | — | 8.1 | 3964 |
| K (followup/clarification) | 15 | 87% | 0% | — | 73% | — | 8.0 | 3405 |
| L (off-topic / redirect) | 16 | 94% | 0% | — | 94% | 94% | 0.5 | 2748 |
| M (coverage gap / acknowledge) | 12 | 100% | 0% | — | 60% | — | 8.0 | 3705 |

**Answer-good-without-direct-source:** 4 rows the labeler marked as having produced a good production answer even though no document directly answered them. 0/4 score a retrieval hit here — the remainder are rows where our metrics say MISS but the student was served correctly. Treat them as a ceiling on how much retrieval improvement is actually available, not as failures to fix.

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
