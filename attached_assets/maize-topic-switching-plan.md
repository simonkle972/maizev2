# Maize TA — Topic-switch handling (structured → conceptual)

**Created:** 2026-05-09 · **Status:** Option A shipping today; Options B and C documented as fallbacks if Option A's prompt refinement doesn't fully resolve the misattribution pattern in prod.

## Why this plan exists

Prod log [Maize QA Master [PROD] - Topic switching.csv](Maize QA Master [PROD] - Topic switching.csv) (16 rows, 4 sessions, 3 TAs) surfaced a clear failure mode: when a student is mid-conversation on a structured problem (homework, problem set) and asks a broader conceptual question **without explicitly referencing a new document**, the contextualizer classifies the turn as `concept_lookup` and the session cache stays anchored to the original document. The model still produces a passable answer (because the supplementary teaching fetch pulls in the right concept material), but the `sources` line shown to the student misattributes the answer to the original homework, and the cache continues to anchor subsequent turns to that wrong document.

Conversely, structured → structured pivots ("now help me with homework 2 question 1") work correctly — those produce explicit document references that the contextualizer reliably classifies as `pivot`.

## What the data shows

### Switches that work (structured → structured)

- **Row 8** (ec112): "Ok now I need help with question 1 from homework 2" after working on Homework 1 → `intent=pivot`, `cache_action=invalidated_by_contextualizer_pivot`, hybrid_doc cleanly switches to Homework 2. ✓
- **Row 16** (mgt403prob): "ok now i need help with question 1a from problem set 2" after working on Pset3 → `intent=pivot`, cache invalidated, hybrid_doc switches to Pset2. ✓

The common signal is an explicit reference to a new document.

### Switches that don't work (structured → conceptual)

Five rows in three different sessions:

| Row | Session | Prior turns on | Concept query | Cache action | hybrid_doc stays at |
|---|---|---|---|---|---|
| 5 | QcbEnmuujI | Homework 1 (Nominal GDP) | "ok can you now explain circular flow to me?" | `preserved_by_contextualizer_concept_lookup` | 2026 Homework 1 |
| 7 | iPsRwMZ4Yf | Homework 1 (Nominal GDP) | "ok can you explain the circular flow to me?" | `preserved_by_contextualizer_concept_lookup` | 2026 Homework 1 |
| 12 | skRm-31FNf | Problem Set 1 (Porter forces on BYND) | "how do I build a DCF?" | `preserved_by_contextualizer_concept_lookup` | Problem Set 1 |
| 14 | vI7Nb5Zejw | Pset3 6a (joint normal P(X>Y)) | "can you explain how covariance works?" | `preserved_by_contextualizer_concept_lookup` | MGT_403_2025_Pset3 |
| 15 | vI7Nb5Zejw | Pset3 6a | "what is Bayes theorem?" | `preserved_by_contextualizer_concept_lookup` | MGT_403_2025_Pset3 |

Note Row 14 is genuinely borderline — covariance directly underlies the joint-normal problem, so preserving cache is defensible. Rows 5, 7, 12, 15 are clear misattributions: the concept asked about has no meaningful connection to the problem the cache is anchored on.

## Root cause

The contextualizer's intent definition in `src/retriever.py`:

> `"concept_lookup"`: student asks about a **related** concept that isn't itself a new problem — stay on the current problem but pull in supporting material

The classifier is too lenient on the word "related." From the model's perspective, anything from the same course is "related." What we actually want: *the concept is in service of solving the current problem* (student is stuck on a covariance calc and asks "what's covariance again?"), not *the concept is on the course menu somewhere*.

Downstream consequence in the cache-hit path of `retrieve_context()`:
- We call `_fetch_concept_supplementary_chunks()` and append the teaching material to the cached chunk's text.
- The `chunks` list returned to the caller still has only the single cached chunk with `file_name = <homework doc>`.
- The model receives the right concept material in its prompt (good — explains "passable" answers).
- The `sources` line lists only the homework (bad — misattribution).
- The session cache stays anchored to the homework on subsequent turns (compounding).

## Three options

### Option A — Tighten the contextualizer prompt (SHIPPED FIRST)

Add explicit examples + a sharper rule to the contextualizer prompt distinguishing "concept in service of current problem" from "broader concept question." Reclassify the latter as `pivot` so the existing cache-invalidation + fresh-retrieval path (verified working in rows 8, 16) handles it.

**Pros:** Prompt-only; no retrieval refactor; trivially reversible. We've successfully sharpened the contextualizer before (off-topic intent, paste detection) and it worked.
**Cons:** LLM classifier is judgement-based. May not draw the line crisply enough on edge cases (row 14 should preserve; row 5 should pivot — these may not be as obvious to the model as to us).

### Option B — Programmatic relatedness check (fallback if A is insufficient)

Keep `concept_lookup` semantics in the contextualizer, but add a programmatic guard in `retrieve_context()`: when `intent == concept_lookup`, check whether the concept terms appear in the cached `document_content`. If not, treat as `pivot`.

**Implementation sketch:**
- Tokenize the concept query (already done via the contextualizer's `current_focus` field or via simple noun extraction)
- Substring-check those terms against `session_context["document_content"]` (already loaded)
- If 0/1 matches → invalidate cache, fall through to fresh retrieval
- Else preserve cache as today

**Pros:** Deterministic, no LLM dependency for this decision, cheap (regex over already-loaded text, ~5ms).
**Cons:** Heuristic — "loanable funds" vs "savings/investment market" are the same concept with different words; substring may miss. Could over-invalidate on subtle relationships.

### Option C — Always refresh retrieval on `concept_lookup` (most invasive fallback)

Don't try to classify the concept as in-vs-out of current problem. Always do fresh retrieval when intent is `concept_lookup`, return the fresh teaching chunks as the PRIMARY chunks (so `sources` reflects them), AND keep the cached problem document available in the session as background context (model knows what problem the student is "on" but cites the right materials for the concept).

**Implementation sketch:**
- In the cache-hit path, when intent is `concept_lookup`, call `_fetch_concept_supplementary_chunks()` (already done today) but ALSO return those chunks as primary chunks
- The cached doc text can be appended to one of the chunks as "Current problem context: ..." but doesn't drive the `chunks[0].file_name` for source attribution
- Sources line now reflects concept-relevant docs

**Pros:** Best UX — sources always accurate. No edge-case classification needed.
**Cons:** More invasive change. Refactor of cache-hit return shape. Extra retrieval latency on every `concept_lookup` (~300-600ms — vector search + rerank). Defer unless A and B together don't suffice.

## Recommendation

**Option A first, with Option B as a layered safety net if needed.** Option C only if architectural simplification becomes attractive for other reasons.

Reasoning:
- Option A is the smallest change with the highest leverage. We've successfully sharpened the contextualizer prompt before and it worked.
- The clear cases in the log (circular flow vs nominal GDP, DCF vs Porter forces, Bayes vs joint-normal) are plainly different topics — a sharper prompt with explicit examples should catch them.
- Row 14 (covariance while working on covariance) genuinely should preserve cache, so we don't want a blanket "all concept questions are pivots" rule.
- If Option A leaves cases on the table, Option B's substring check can layer on cleanly without a refactor.

## Concrete change plan for Option A

In `src/retriever.py`'s `contextualize_query()` prompt:

1. **Sharpen the `concept_lookup` definition** to require that the concept is DIRECTLY USED in the current problem (not just same-course). Cite examples.
2. **Sharpen the `pivot` definition** to explicitly include "asking about a concept that isn't part of the current problem's mechanics."
3. **Add 3-4 worked examples** drawn directly from this prod log:
   - GOOD `concept_lookup`: "what's covariance again?" while working on a covariance computation in Pset3 → preserve cache
   - BAD (should be `pivot`): "explain circular flow" while working on Homework 1 nominal GDP arithmetic → invalidate cache
   - BAD (should be `pivot`): "how do I build a DCF?" while working on Porter forces in Problem Set 1 → invalidate cache
   - BAD (should be `pivot`): "what is Bayes theorem?" while working on a joint-normal probability problem (Pset3 6a) → invalidate cache
4. **Reinforce the existing "when in doubt, prefer continuation/concept_lookup" guardrail** with a sentence: "but if the concept is plainly a different topic from what the cached problem is testing, classify as pivot to allow fresh retrieval on the new concept."

That's the entirety of Option A — additions to the contextualizer prompt, no other code touched.

## Verification

1. **Local: re-run the 4 prod scenarios** that failed:
   - Start a session on Homework 1, after 2-3 turns ask "explain circular flow." Should classify as `pivot`, invalidate cache, do fresh retrieval — sources should list a lecture/reading on circular flow.
   - Same shape: start on mgt423 Problem Set 1, then ask "how do I build a DCF?". Sources should list DCF-relevant lecture material.
   - Same shape: start on mgt403prob Pset3, then ask "what is Bayes theorem?". Sources should list the probability lecture.
2. **Local: confirm the borderline case still preserves cache.** Start on Pset3 6a (covariance computation). Ask "how does covariance work?" → should classify as `concept_lookup`, preserve cache.
3. **Local: confirm structured→structured pivots still work.** Start on Homework 1, then "I need help with homework 2 question 1" → still classified as `pivot`.
4. **Local: confirm `continuation` is unaffected.** Multi-turn validation on the same problem should still preserve cache.
5. **Prod observation for 24-48h post-deploy.** Watch `sources` column for new misattribution after concept questions. If Option A leaves cases on the table, escalate to Option B.

## Out of scope (deferred)

- Option B (programmatic substring check) — defer unless Option A's prompt change doesn't catch enough cases in prod observation.
- Option C (always-refresh retrieval on concept_lookup) — most invasive; defer unless A and B together don't suffice or we end up wanting to refactor the cache path for other reasons.
- Conceptual → structured switches — user confirms these already work; no change needed.
- Adding a new `concept_pivot` intent — keep the intent enum compact; express the distinction via prompt tightening alone.
