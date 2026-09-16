# Phase 3 — one conversation for every stage

**2026-09-16. Defined after a user-supplied assessment was verified against the code. Starts after
Phase 2 closes (model decision, widen default, short-circuit as slot, deploy).**

## The finding

The generator never sees the end of the TA's last answer. `src/chat_streaming.py` (history block
around lines 255–276) sends the six most recent messages cut to their first 300 characters, as a
prose block inside the user message. The structured `history_for_llm` message list only activates
when a session contains an image, and it cuts to 600. A TA answer runs 1–3k characters and the step
the student did not follow, or the question the TA asked, sits at the END. So "I did not follow your
last step" reaches gpt-5.2 without the step. Contextualizer v2 (2026-09-15) was given the full last
turn for exactly this reason; the generator was not. Three stages each reason about a different,
lossy version of the same conversation.

No retrieval metric can see this; it needs an answer-quality measurement.

## Items

1. **Generator sees the transcript.** Always the structured message list (role/content messages),
   no per-message cut; cap by a token budget (`HISTORY_MAX_TOKENS`, default ~8000) keeping the most
   recent turns; the last assistant turn is never cut. Images keep working (that path already
   carries them). Stable system prefix first (prompt cache unchanged); retrieved material stays in
   its own system message after history. Files: `src/chat_streaming.py`
   (`_build_history_for_llm`), `src/response_generator.py` (`build_messages` already supports the
   structured form), `professor.py` test chat, `eval/run_eval.py::_generate_answer` (mirrors the
   cut by hand — must change in lockstep).
2. **Paired answer judge** (`eval/judge_pairs.py`, new): two `--generate` runs (old vs new history);
   each pair judged by gpt-5.2 in BOTH orders (August rule: one-order pairwise judgements flipped
   8/20 verdicts); report agreed wins / losses / ties and disagreements separately. Rows: K (15),
   I (15), the 20 reply rows, the 4 canaries. Report generation tokens and latency per turn too.
3. **Solutions gating as a judgement.** `student_stated_answer` (bool) in the v2 classifier output;
   the solutions document joins the shortlist only when true — replaces the "two student messages"
   proxy in the Phase 1 prior block and the legacy cache path. No `solution_request` field: a bare
   request is already off-topic rule (e); a mid-conversation request is declined by the generator
   by policy (user rule 2026-09-16: a student requesting solutions is declined; the retriever may
   use solutions to enrich a response while the student works). Verify on the BR / writeup
   canaries; re-run the August leak check (0 leaks expected).

## Where I differ from the assessment that prompted this

"The model upgrade can wait": our data says the classifier model is what makes the skip gate work
(gpt-4o-mini 0/15 "no retrieval" on clarification prompts; terra 11–13/15). Transcript fixes the
answer; the classifier decides whether to search at all (latency + drift). Complementary, not
sequential — the model decision closes in Phase 2.

## Phase 4 (was Phase 3)
Chunk-level lexical index + HNSW, document card / alias layer, flat-vs-routed test.
