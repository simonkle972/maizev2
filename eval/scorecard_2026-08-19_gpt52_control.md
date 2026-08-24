# Retrieval scorecard

**Total rows:** 92 (89 in-corpus + 3 not-in-corpus). **Errors:** 0.
**TAs in this run:** 3 — `EgZ14pvqEYzfQRTM`, `WBNtFkfPGZaJVQIk`, `z_B4fFY6jD1mhy9K`. Cross-TA aggregate scoring; re-run with `--ta-id <id>` to scope to one TA.

## Doc-routing buckets (Wave 1)

| Failure type | n | hit@5 pre→post (lift) | hard_neg_top1 | forbidden_hit | forbidden_text_hit | avg_latency_ms | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (Lab vs PS) | 6 | 100%→100% (+0%) | 0% | 0% | 0% | 22061 | 0 |
| B (Roman numeral siblings, cross-doc) | 5 | 100%→80% (-20%) | 0% | 0% | 0% | 27825 | 0 |
| C (lookalike-unrelated) | 7 | 71%→71% (+0%) | 14% | 14% | 0% | 17250 | 0 |
| D (problem-vs-solutions) | 5 | 60%→60% (+0%) | 20% | 0% | 0% | 44968 | 0 |
| E (cache anchoring) | 22 | 14%→41% (+27%) | 14% | 0% | 0% | 10524 | 0 |
| F1 (explicit conceptual switch) | 0 | — | — | — | — | — | — |
| F2 (explicit document switch) | 1 | 0%→0% (+0%) | 100% | 0% | 0% | 1517 | 0 |
| G1 (intra-doc section confusion) | 1 | 0%→0% (+0%) | 100% | 0% | 100% | 3283 | 0 |
| G2 (intra-doc Roman/numeric sibling) | 2 | 0%→0% (+0%) | 100% | 0% | 100% | 3660 | 0 |
| working cases | 40 | 30%→68% (+38%) | 5% | 2% | 0% | 10090 | 0 |

## Intent-classification buckets (Wave 2)

| Failure type | n | bucket_hit | hit@5 (doc-routing) | all_correct_in_top_5 (H only) | intent_class_match | redirect_fired (L only) | avg_chunks_returned | avg_latency_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H (multi-document intent) | 0 | — | — | — | — | — | — | — |
| I (document correction) | 0 | — | — | — | — | — | — | — |
| J (concept-vs-problem) | 0 | — | — | — | — | — | — | — |
| K (followup/clarification) | 0 | — | — | — | — | — | — | — |
| L (off-topic / redirect) | 0 | — | — | — | — | — | — | — |

**Not-in-corpus bucket:** 3 rows. Retrieved a labeled-correct doc on 0/3 (expected 0 — these docs are unverified in the corpus). Forbidden-hit rate: 0%.

## Metric definitions
- **hit@5 pre→post (lift)** — pre-rerank hit-rate → post-rerank hit-rate, with the rerank's contribution as a `(±X%)` delta. Positive = rerank moved correct chunks into top-5 that weren't there before. Negative = rerank pushed correct chunks out. Zero = rerank didn't affect top-5 membership.
- **correct_hit@5** — fraction of rows where at least one `correct_doc_ids` appeared in retrieved top-5 (post-rerank).
- **hard_negative_top1** — fraction of rows where the retrieved top-1 doc matched a known hard negative (i.e., the current bad retrieval pattern fired).
- **forbidden_hit** — fraction of rows where ANY retrieved doc was on the forbidden list (e.g., solutions doc returned when student is solving). Lower is better; ideal = 0%.
- **forbidden_text_hit** — fraction of rows where ANY retrieved chunk's text contained a forbidden substring (e.g., "Part II" when the row tests Part I retrieval). Detects G failures — right doc, wrong section/part. Lower is better; ideal = 0%.
- **bucket_hit** (Wave 2) — the primary HIT signal for a row, depends on its `expected_action`: for `retrieve` rows it equals hit@5; for `redirect` rows it requires `adversarial_short_circuit` fired AND zero chunks returned; for `no_retrieval` rows it's a proxy via `intent == 'clarification'` today (becomes a true skip-gate metric post-LangGraph adaptation).
- **all_correct_in_top_5** (H bucket) — stricter than hit@5: requires EVERY `correct_doc_ids` entry to appear in top-5, not just one. Tests whether multi-doc intent surfaces ALL needed docs.
- **intent_class_match** — fraction of rows (with `expected_intent.intent_class` labeled) where the contextualizer's classification matches the label. Measures intent-classification accuracy independently of retrieval — Q1+Q2 deep-research flagged this as a literature gap; doing this puts Maize ahead of published practice.
- **redirect_fired** (L bucket) — fraction of rows where `adversarial_short_circuit` fired in diagnostics, regardless of whether chunks were also returned.
- **avg_chunks_returned** — average number of chunks the retriever returned. For K/L rows the IDEAL value is 0 (system should skip retrieval). Useful as a smoke check that the skip-gate is firing.
