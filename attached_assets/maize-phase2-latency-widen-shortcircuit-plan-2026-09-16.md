# Phase 2 — latency back, widen instead of collapse, short-circuit as a slot

**2026-09-16. Follows Phase 1 (accepted, both flags default on, commit f0660ed). Baseline for
Phase 2 = `eval/exp_2026-09-16_phase1b_full.*` (250 rows, Cohere, 180/250, K under the new rule).**

Priorities, in the user's words: document retrieval accuracy first, then retrieval speed.

## 1. Latency (first, because it is the cost Phase 1 left)

Phase 1 p50: openers 3.6 → 5.9 s, follow-ups 2.8 → 5.6 s. Instrumented stages sum to ~3.2 s;
~2 s per turn is unattributed.

1. **Instrument, then cut.** Add a stage timer to `retrieve_context` (`diagnostics["stage_ms"]`)
   covering every segment: session load, moderation, contextualizer, query analysis, prior
   resolution + solutions lookup, embedding(s), doc search (+ raw union), chunk search, prior
   re-materialisation, structural injection, paste detection, rerank, validation/confidence,
   collapse fetch, supplementary, cache write. Run ~30 rows, read the medians.
2. **Remove the raw-query union** unless the timers say it is free: on the 34 openers it fired on
   it was accuracy-neutral (16/34 vs v7 15/34) and costs an embedding + a doc search each time.
3. Cut whatever else dominates (candidates: `find_solution_document` loads the full solutions text
   every follow-up just to learn its id; `_doc_id_for_filename` chains; the v2 prompt size).
4. Target: p50 within ~1 s of v7 on both halves, accuracy unchanged (±2 rows on the same set).

## 2. Widen as the default low-confidence arm

Measured 2026-09-13 on 32 low-confidence rows with a real document: widen 29 vs collapse 17,
reproduced. Under Phase 1 most remaining regressions are the fresh path's collapse choosing a
sibling. Flip `LOW_CONFIDENCE_ACTION=widen`; the prior-aware collapse stays only for
validation-failed references; `LOW_CONFIDENCE_INSTRUCTIONS` go live; `HYBRID_FULL_DOC_INSTRUCTIONS`
already course-level. Measure on the full set. Also decide the dead thin-material guard here.

## 3. Filename short-circuit: guaranteed slot, not a bypass

Evidence (Phase 1 run and v7): openers where it fired hit 87–89 %, where it did not 44–59 %. It is
the most precise routing signal, not a latency trick (fusion costs ~30 ms). Its risk is the FORM:
returns one document, forecloses multi-document questions and any recovery. Change: the
short-circuit document becomes the first guaranteed member of a normal fused shortlist (same
mechanism as the Phase 1 prior/hint slots); the reranker decides. Flag
`SHORT_CIRCUIT_AS_SLOT`, measured on the full set; watch A/B/C openers and H.

## 4. Deploy

VPS latency check on a handful of real queries (prod hardware differs; never project from the
laptop), then the Phase 1 + Phase 2 push as one deploy. Rollback = the flags.

## Not in scope
Chunk-level lexical index and the document card (Phase 3); retrieval as a generator tool call.
