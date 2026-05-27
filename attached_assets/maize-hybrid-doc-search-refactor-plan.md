# `hybrid_doc_search` refactor — summary-cosine + LLM tiebreaker (Phase B retrieval-side overhaul)

**Status: ATTEMPTED, ROLLED BACK 2026-05-26. DO NOT RE-ATTEMPT WITHOUT NEW EVIDENCE.** The audit's recommended refactor of `hybrid_doc_search` was implemented end-to-end (function + LLM tiebreaker + flag + branch in `retrieve_context`), eval-tested against the 92-row body, and reverted. See "Result — did not ship" at the bottom of this doc for the full diagnosis. The plan body below preserves the original intent so future contributors can see what was tried and why; **the code is no longer in the repo**.

---

## Goal

Replace the current Stage 1 doc-routing stack (Stage 5 filename short-circuit + 3-signal RRF over BM25 + dense + filename) with a single summary-cosine match against `Document.summary_embedding`, with an LLM tiebreaker for ambiguous cases. **Largest single retrieval-side simplification in the audit** ([maize-architecture-review-2026-05-23.md line 802](maize-architecture-review-2026-05-23.md)): ~300 lines deletable in a follow-up session.

## Why now

1. **Both indexing-side gates are satisfied.** B10 populated `Document.summary` + `summary_embedding` across every doc in the indexed corpus (100% coverage on the 3 eval TAs). B9 made each chunk self-describing. D12 multi-TA eval infrastructure is in place.
2. **The audit ranks this as the highest-deletion-to-risk retrieval-side cleanup.** ~300 deletable lines, flag-gated A/B path, no schema change, no re-indexing required.
3. **Logical sequencing.** B19 deletion is also unblocked but requires generation-side prompt work to absorb B19's pedagogical role. Group C is larger and pairs naturally with this refactor for an end-to-end retrieval overhaul. Doing `hybrid_doc_search` first is the cleanest atomic move.

## What today's `hybrid_doc_search` does (380 lines, `src/retriever.py:1171-1550`)

Two-phase routing:

1. **Stage 5 short-circuit** (~290 lines, `:1171-1460`). Filename overlap + number-matching:
   - Path A: category + number DB lookup ("pset 3" + doc_category="pset")
   - Path B: filename-overlap + number match
   - Path C: margin-only filename overlap (no number)
   - Plus multi-doc detection and various guards
2. **RRF fusion** (~90 lines, `:1461-1550`). Three rankings fused via Reciprocal Rank Fusion:
   - BM25 over `Document.bm25_tsvector`
   - Dense cosine of query embedding vs top-N chunks per doc (mean-pooled)
   - Filename token overlap

Returns `(doc_ids, diagnostics)`. Single call site at `src/retriever.py:2681` (inside `retrieve_context`), where the returned `doc_ids` constrain the downstream chunk vector search.

## Scope (this session)

### 1. New function — `summary_doc_search(query, query_embedding, ta_id, top_k=None, query_analysis=None) -> tuple`

Location: `src/retriever.py`, immediately above the existing `hybrid_doc_search`.

Body:
1. **Cosine match** query_embedding vs `Document.summary_embedding`, ORDER BY cosine ASC, LIMIT `PER_SIDE_K`. Single SQL call. Returns `(doc_id, cosine_score)` per doc.
2. **LLM tiebreaker** (fires only when top-1 minus top-2 cosine difference < `SUMMARY_TIEBREAKER_MARGIN`). Asks gpt-4o-mini: "Given query X and these N summaries, which doc is the student asking about?" Returns refined ordering of top-3.
3. **Same return shape** as `hybrid_doc_search` — `(top_k doc_ids, diagnostics)`. Diagnostics carries `method="summary_cosine"`, cosine scores, tiebreaker fired/not, latency.

~30-line SQL + ~50-line LLM tiebreaker. Total ~80 lines vs the 380 it replaces.

### 2. Config flag

`Config.SUMMARY_DOC_ROUTING_ENABLED` (default `False`). Env-driven so we can flip it in prod without a code deploy.

Also add: `Config.SUMMARY_TIEBREAKER_MARGIN` (default `0.03`) — the cosine-score gap below which the LLM tiebreaker fires.

### 3. Wire into `retrieve_context`

At [src/retriever.py:2681](src/retriever.py#L2681):
```python
if Config.SUMMARY_DOC_ROUTING_ENABLED:
    candidate_doc_ids, hybrid_diag = summary_doc_search(
        effective_query, query_embedding, ta_id, query_analysis=query_analysis
    )
else:
    candidate_doc_ids, hybrid_diag = hybrid_doc_search(
        effective_query, query_embedding, ta_id, query_analysis=query_analysis
    )
```

Both functions have the same signature + return shape, so downstream code (chunk filter, structural narrowing, B15 structural injection, B17 rerank, B18 hybrid fallback) is unchanged.

### 4. NO deletion

`hybrid_doc_search` body stays. Flag default is OFF in this session. A/B happens in this session (locally, on the eval); flipping default-on in prod and deleting the old body is a separate future-session decision once we have prod data.

## What's NOT in scope

- Deleting `hybrid_doc_search` (deferred to follow-up after prod A/B)
- Deleting the Stage 5 short-circuit code path (same)
- Deleting filename + BM25 + dense fusion code (same)
- Touching the chunk-level vector search downstream
- Group C top-of-funnel redesign (separate session)
- B19 supplementary-teaching deletion (separate session, requires generation-side prompt work)

## Validation gate

Same eval rigor as B9 — cross-TA + warm-cache.

1. **Smoke test** — call `summary_doc_search` directly on ~5 sample queries per TA. Check the top doc is sensible, that ambiguous queries trigger the tiebreaker, that latency is in the expected range (~50-200ms vs 1-3s for current hybrid).
2. **Eval with flag OFF** — run 92-row eval against today's `hybrid_doc_search` (this is the post-B9 state, scorecard we already have as `/tmp/eval_scorecard_post_b9.md`). Effectively reuses that scorecard.
3. **Eval with flag ON** — flip `SUMMARY_DOC_ROUTING_ENABLED=true`, re-run 92-row eval, save to `/tmp/eval_scorecard_summary_routing.md`.
4. **Compare** the two scorecards. Bucket-level focus:
   - **Type A (Lab vs PS)** — was the Stage 5 short-circuit's main job. Most likely to regress. Acceptance threshold: 100% → ≥83% (1 row regression max).
   - **Type B (Roman cross-doc)** — was 80% post-B9. Summary-routing should ideally improve.
   - **Type C (lookalike-unrelated)** — summary-routing's strong suit. Should equal or beat 71%.
   - **F2 (explicit doc switch)** — should be a clear win since the summary captures topic explicitly.
   - **Working cases (n=40)** — non-negotiable: must not drop more than 2 rows (5pp).
5. **Latency** — should drop substantially (1 SQL call + maybe 1 LLM call vs 3 SQL calls + filename loop + RRF). Useful for the "is it production-ready" call but not the validation gate itself.

**Pass criterion**: overall hit@5 within ±2pp of post-B9 baseline, no individual bucket regresses by more than the row-count-determined fragility threshold (e.g., n=2 buckets can flip 1 row; n=40 working cases can lose 2 rows max).

## Sequencing

1. Write `summary_doc_search` function — SQL cosine + diagnostics scaffold.
2. Write LLM tiebreaker (gpt-4o-mini, top-3 summaries as input, returns ordered doc IDs).
3. Add `Config.SUMMARY_DOC_ROUTING_ENABLED` + `Config.SUMMARY_TIEBREAKER_MARGIN`.
4. Wire branch into `retrieve_context`.
5. Smoke test on 5 queries per TA — verify SQL works, tiebreaker fires when expected, return shape is correct.
6. Run eval with flag OFF (use existing `/tmp/eval_scorecard_post_b9.md` if it's still relevant — it represents the post-B9 baseline this refactor is being compared against).
7. Run eval with flag ON (`SUMMARY_DOC_ROUTING_ENABLED=true`), 92 rows, warm cache, → `/tmp/eval_scorecard_summary_routing.md`.
8. Compare. Decide commit / iterate / revert.
9. If pass: commit with flag default OFF. Audit-doc 3.3 entry.
10. Log B12 (or wherever this slots) in audit trail with "shipped behind flag, A/B with prod data is the next gate before flipping default-on + deletion."

## Risks + rollback

- **Risk: Type A (Lab vs PS) regresses** because short queries like "pset 2" lack enough lexical-vs-summary signal. **Mitigation**: if it does, add a numeric-aware filter as a Stage 2 narrowing — query analyzer extracts the number, summary-cosine returns top-K, filter to docs whose `assignment_number` matches. Same logic as the existing Stage 5 short-circuit, just cleaner and without the filename-overlap dance.
- **Risk: multi-doc queries** ("compare lecture 3 and 4") don't surface both docs. **Mitigation**: summary-cosine top-K naturally returns multiple high-scoring docs. If a multi-doc row fails in eval, examine its top-K output to see whether both expected docs were in the top-K but in the wrong order — could be a tiebreaker prompt tweak rather than an architectural problem.
- **Risk: LLM tiebreaker adds latency** that erodes the speed win. **Mitigation**: the tiebreaker only fires when scores are close; if it fires on too many queries, raise `SUMMARY_TIEBREAKER_MARGIN` or skip the tiebreaker entirely for first ship.
- **Risk: summary embedding quality** turns out lower than expected on some docs (e.g., docs with terse content). **Mitigation**: spot-check the docs that fail in eval; if their summary is thin, re-run `summarize_doc` for that doc — fixable at the data layer without touching retrieval code.
- **Rollback**: flag defaults to False. If eval regresses badly, the new function still ships but disabled. Reverting is a one-line `Config.SUMMARY_DOC_ROUTING_ENABLED = False` change.

## Effort estimate

~3-4h of implementation + smoke. ~2h of eval (one new full 92-row warm-cache run at ~90 min). Total ~5-6h session.

## Path forward after this ship

If A/B is parity-or-win locally:
1. Wait for the next prod push window (this commit can batch with the queued B10/B9/eval-expansion commits or wait for a UX change).
2. Once in prod with the flag default OFF, monitor qa_logs.
3. Flip `SUMMARY_DOC_ROUTING_ENABLED=true` in prod for a A/B period (~1-2 weeks).
4. After prod A/B win is confirmed: **delete `hybrid_doc_search` body** (drop ~290 lines of Stage 5 short-circuit + ~90 lines of RRF fusion), **drop BM25 from doc routing entirely** (`Document.bm25_tsvector` column may stay for now since it's used by other paths or as a fallback). Filename overlap routine survives if anything else still needs it.

That second commit is the ~300-line deletion the audit promised. **This session ships the foundation; the deletion is a clean follow-up that can be scheduled after we have confidence the new path holds in prod.**

---

## Result — did not ship (2026-05-26)

The plan above describes what was implemented. The implementation went smoothly: `summary_doc_search` + `_summary_routing_tiebreaker` (~180 lines) + 2 config flags + a one-line branch in `retrieve_context`. Smoke tests on 9 representative queries already surfaced the failure mode: numeric/structural queries like "pset 2 question 1" missed the correct doc entirely (returned `[39, 76, 33]` — extra problems II, week 1 practice, pset 1) when `hybrid_doc_search` nailed it via the Stage 5 short-circuit. The 92-row eval confirmed this was systemic, not isolated.

### Side-by-side scorecard

| Bucket | n | post-B9 hit@5 | summary-routing hit@5 | Δ |
|---|---:|---:|---:|---:|
| A (Lab vs PS) | 6 | 100% | 67% | **−33pp (−2 rows)** |
| B (Roman cross-doc) | 5 | 80% | 20% | **−60pp (−3 rows)** |
| C (lookalike) | 7 | 71% | 14% | **−57pp (−4 rows)** |
| D (problem-vs-solutions) | 5 | 60% | 80% | +20pp (+1 row), but `forbidden_hit` 0% → **60%** |
| E (cache anchoring) | 22 | 73% | 55% | **−18pp (−4 rows)** |
| F2 (doc switch) | 1 | 100% | 100% | 0 |
| G1 (intra-doc section) | 1 | 0% | 0% | 0 |
| G2 (intra-doc sibling) | 2 | 50% | 0% | −50pp (−1 row) |
| Working cases | 40 | 78% | 78% | 0 |
| **Overall weighted** | **89** | **75.5%** | **61.0%** | **−14.5pp** |

Pass criterion in the plan was ±2pp. We failed by ~7x.

### Diagnosis

The audit ([maize-architecture-review-2026-05-23.md](maize-architecture-review-2026-05-23.md) line 802) ranked this as "the largest single retrieval simplification" based on Anthropic's published 49% retrieval-failure-reduction figure. Three things about Maize's specific situation make that figure not transfer:

1. **`hybrid_doc_search` already has a Stage 5 short-circuit** that explicitly handles numeric/filename matching. It's not a generic "BM25 + dense" baseline like Anthropic's benchmark — it's a 3-signal RRF *plus* a high-precision direct-match path that fires on most structural queries. Summary-cosine doesn't replicate the direct-match path, so queries that used to short-circuit cleanly now have to go through a noisier dense-only retrieval.

2. **Numeric/structural queries dominate Maize's query distribution.** "pset 2", "extra problems II", "lecture 5", "question 14 from PS3" — these are the dominant query class in the eval body AND in prod qa_logs. Dense summary embeddings dilute the exact-token signal ("pset", "2") into broader topical similarity, which can't distinguish "pset 2" from "pset 1" reliably. BM25 + filename overlap are SPARSE signals that match exact tokens — exactly the right tool for that query class.

3. **Anthropic's 49% figure was on text-corpus benchmarks** (FinanceBench, ChunkRAG, etc.) where queries are full-sentence semantic questions, not short structural identifiers. The retrieval-lift mechanism (contextualizing each chunk's role in its parent doc) only helps when the query's semantic content matches the chunk's contextualized content. Numeric identifiers don't participate in that mechanism.

D's apparent +20pp "win" was misleading: `forbidden_hit` jumped from 0% to 60%, meaning the retriever was *more* likely to surface solutions docs the student shouldn't see. That's a regression, not a win.

### What stayed (the artifacts of this experiment)

- This doc (preserved as historical record).
- A defensive comment above `hybrid_doc_search` in `src/retriever.py` flagging the negative result so future contributors don't try the same thing.
- An entry in [maize-architecture-review-2026-05-23.md](maize-architecture-review-2026-05-23.md) section 3.3 + the audit trail.
- A project memory note (`feedback_summary_routing_did_not_work.md`).

### What this means for the master plan

The audit's broader claim — that B10 (per-doc summary + summary_embedding) unlocks a major retrieval-side simplification — **does not hold for Maize's query distribution.** B10's indexing-side cost (~$0.001/doc + 1-3s per doc) was paid; the retrieval-side payoff doesn't materialize. The `Document.summary_embedding` column is still useful for explainability (debugging "what is this doc about" via SQL), but not as a primary doc-routing signal.

The hybrid_doc_search refactor is removed from the roadmap. The remaining retrieval-side cleanup items per [maize-architecture-review-2026-05-23.md](maize-architecture-review-2026-05-23.md) — **B19 supplementary-teaching deletion** (gated on generation-side prompt work), **Group C top-of-funnel redesign**, and the **B17 reranker decision** (parked, gated on prod multi-TA qa_logs) — are unaffected by this negative result and remain in the queue.
