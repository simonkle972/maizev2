# Retrieval scorecard

**Total rows:** 250 (250 in-corpus + 0 not-in-corpus). **Errors:** 0.
**Reranker:** `cohere` · **low-confidence action:** `collapse` · **cache reuse:** `True` · **rerank query:** `raw` · **cache as prior:** `True` · **contextualizer v2:** `True`
**TAs in this run:** 4 — `EgZ14pvqEYzfQRTM`, `WBNtFkfPGZaJVQIk`, `iDYis09JtNUkyEJJ`, `z_B4fFY6jD1mhy9K`. Cross-TA aggregate scoring; re-run with `--ta-id <id>` to scope to one TA.

## Openers vs follow-ups (bucket_hit)

| Bucket | openers n | hit | collapsed | p50 / p95 ms | follow-ups n | hit | collapsed | p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **all in-corpus** | 113 | 68% | 50% | 2417 / 3989 | 137 | 74% | 33% | 2544 / 4459 |
| A | 16 | 94% | 94% | 2453 / 3989 | 0 | — | — | — / — |
| B | 14 | 64% | 43% | 2785 / 5131 | 7 | 86% | 29% | 2557 / 5371 |
| C | 15 | 80% | 53% | 2385 / 4186 | 3 | 100% | 0% | 3190 / 4937 |
| E | 0 | — | — | — / — | 22 | 82% | 45% | 2835 / 5175 |
| F1 | 0 | — | — | — / — | 15 | 73% | 7% | 2544 / 4042 |
| F2 | 0 | — | — | — / — | 36 | 67% | 33% | 3048 / 6017 |
| H | 15 | 27% | 67% | 2535 / 4228 | 0 | — | — | — / — |
| I | 0 | — | — | — / — | 15 | 67% | 40% | 2150 / 3268 |
| K | 0 | — | — | — / — | 15 | 80% | 67% | 2370 / 3137 |
| L | 16 | 94% | 0% | 2115 / 2800 | 0 | — | — | — / — |
| M | 12 | 8% | 92% | 2073 / 4419 | 0 | — | — | — / — |
| working | 25 | 84% | 28% | 2673 / 3634 | 24 | 75% | 17% | 2197 / 3837 |

## Doc-routing buckets (Wave 1)

| Failure type | n | hit@5 pre→post (lift) | collapsed_to_full_doc | hard_neg_top1 | forbidden_hit | avg_latency_ms | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (Lab vs PS) | 16 | 88%→94% (+6%) | 94% | 0% | 0% | 2722 | 0 |
| B (Roman numeral siblings, cross-doc) | 21 | 62%→71% (+10%) | 38% | 0% | 0% | 2887 | 0 |
| C (lookalike-unrelated) | 18 | 78%→83% (+6%) | 44% | 11% | 0% | 2684 | 0 |
| E (cache anchoring) | 22 | 73%→82% (+9%) | 45% | 9% | 5% | 3087 | 0 |
| F1 (explicit conceptual switch) | 15 | 73%→73% (+0%) | 7% | 0% | 0% | 2592 | 0 |
| F2 (explicit document switch) | 36 | 72%→78% (+6%) | 33% | 6% | 0% | 3168 | 0 |
| working cases | 49 | 67%→80% (+12%) | 22% | 6% | 0% | 2576 | 0 |

## Intent-classification buckets (Wave 2)

| Failure type | n | bucket_hit | hit@5 (doc-routing) | all_correct_in_top_5 (H only) | intent_class_match | redirect_fired (L only) | avg_chunks_returned | avg_latency_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H (multi-document intent) | 15 | 27% | 87% | 27% | — | — | 3.6 | 2847 |
| I (document correction) | 15 | 67% | 67% | — | — | — | 6.0 | 2219 |
| K (followup/clarification) | 15 | 80% | 0% | — | 0% | — | 3.6 | 2450 |
| L (off-topic / redirect) | 16 | 94% | 0% | — | 94% | 94% | 0.5 | 2087 |
| M (coverage gap / acknowledge) | 12 | 8% | 0% | — | 90% | — | 1.9 | 2323 |

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
