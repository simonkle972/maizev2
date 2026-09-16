# Retrieval scorecard

**Total rows:** 250 (250 in-corpus + 0 not-in-corpus). **Errors:** 0.
**Reranker:** `cohere` · **low-confidence action:** `collapse` · **cache reuse:** `True` · **rerank query:** `raw` · **cache as prior:** `True` · **contextualizer v2:** `True`
**Config overrides (--set):** `CACHE_AS_PRIOR_ENABLED=True`, `CONTEXTUALIZER_V2_ENABLED=True`
**TAs in this run:** 4 — `EgZ14pvqEYzfQRTM`, `WBNtFkfPGZaJVQIk`, `iDYis09JtNUkyEJJ`, `z_B4fFY6jD1mhy9K`. Cross-TA aggregate scoring; re-run with `--ta-id <id>` to scope to one TA.

## Openers vs follow-ups (bucket_hit)

| Bucket | openers n | hit | collapsed | p50 / p95 ms | follow-ups n | hit | collapsed | p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **all in-corpus** | 113 | 68% | 50% | 5850 / 8915 | 137 | 75% | 33% | 5603 / 10522 |
| A | 16 | 94% | 94% | 6646 / 9549 | 0 | — | — | — / — |
| B | 14 | 64% | 43% | 7285 / 8915 | 7 | 86% | 14% | 2835 / 5175 |
| C | 15 | 80% | 53% | 6611 / 8926 | 3 | 100% | 0% | 4376 / 5408 |
| E | 0 | — | — | — / — | 22 | 86% | 50% | 8551 / 11689 |
| F1 | 0 | — | — | — / — | 15 | 73% | 13% | 3669 / 7659 |
| F2 | 0 | — | — | — / — | 36 | 67% | 33% | 6028 / 10070 |
| H | 15 | 27% | 67% | 4971 / 8630 | 0 | — | — | — / — |
| I | 0 | — | — | — / — | 15 | 73% | 33% | 4007 / 15657 |
| K | 0 | — | — | — / — | 15 | 73% | 67% | 5739 / 10522 |
| L | 16 | 94% | 0% | 2693 / 7968 | 0 | — | — | — / — |
| M | 12 | 8% | 92% | 3467 / 7057 | 0 | — | — | — / — |
| working | 25 | 84% | 28% | 6582 / 9585 | 24 | 75% | 17% | 4869 / 8397 |

## Doc-routing buckets (Wave 1)

| Failure type | n | hit@5 pre→post (lift) | collapsed_to_full_doc | hard_neg_top1 | forbidden_hit | avg_latency_ms | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (Lab vs PS) | 16 | 88%→94% (+6%) | 94% | 0% | 0% | 6313 | 0 |
| B (Roman numeral siblings, cross-doc) | 21 | 57%→71% (+14%) | 33% | 0% | 0% | 5509 | 0 |
| C (lookalike-unrelated) | 18 | 78%→83% (+6%) | 44% | 11% | 0% | 6130 | 0 |
| E (cache anchoring) | 22 | 77%→86% (+9%) | 50% | 9% | 5% | 8595 | 0 |
| F1 (explicit conceptual switch) | 15 | 73%→73% (+0%) | 13% | 0% | 0% | 4551 | 0 |
| F2 (explicit document switch) | 36 | 69%→78% (+8%) | 33% | 6% | 0% | 6311 | 0 |
| working cases | 49 | 69%→80% (+10%) | 22% | 6% | 0% | 6001 | 0 |

## Intent-classification buckets (Wave 2)

| Failure type | n | bucket_hit | hit@5 (doc-routing) | all_correct_in_top_5 (H only) | intent_class_match | redirect_fired (L only) | avg_chunks_returned | avg_latency_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H (multi-document intent) | 15 | 27% | 87% | 27% | — | — | 3.6 | 5158 |
| I (document correction) | 15 | 73% | 73% | — | — | — | 6.4 | 4820 |
| K (followup/clarification) | 15 | 73% | 0% | — | 0% | — | 3.6 | 6161 |
| L (off-topic / redirect) | 16 | 94% | 0% | — | 94% | 94% | 0.5 | 2950 |
| M (coverage gap / acknowledge) | 12 | 8% | 0% | — | 90% | — | 1.9 | 3716 |

**Answer-good-without-direct-source:** 4 rows the labeler marked as having produced a good production answer even though no document directly answered them. 2/4 score a retrieval hit here — the remainder are rows where our metrics say MISS but the student was served correctly. Treat them as a ceiling on how much retrieval improvement is actually available, not as failures to fix.

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
