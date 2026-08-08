# Labeling guide — Wave 1 + Wave 2 prod-log expansion

This doc tells you how to format prod-conversation logs so I can convert them
straight into eval rows. It covers BOTH Wave 1 (doc-routing failures) and
Wave 2 (intent-classification failures) — labeled the same way, in one pass.

Per the 2026-05-29 pivot ([audit doc 3.4](../attached_assets/maize-architecture-review-2026-05-23.md)),
doc-switching/intent-understanding is the focal next failure mode. The 92-row
eval body has heavy E coverage (the only well-measured bucket on prod data,
~20-25% real failure rate) but thin coverage on most other intent dimensions.
This expansion is the validation gate for the LangGraph upstream-gate adaptation
that's the next implementation step.

## Background — what the buckets test

| Bucket | What it tests | Ground truth signal |
|---|---|---|
| **A** | Lab vs PS sibling docs (doc-routing) | `correct_doc_ids` is the right doc |
| **B** | Roman numeral siblings across docs (doc-routing) | `correct_doc_ids` is the right doc |
| **C** | Lookalike-unrelated topic confusion (doc-routing) | `correct_doc_ids` is the right doc |
| **D** | Problem-vs-solutions intent (don't return solutions) | `correct_doc_ids` is the right doc; `forbidden_doc_ids` has the solutions doc |
| **E** | Cache anchoring — multi-turn doc stickiness | `correct_doc_ids` is the right doc; `prior_turns` has the prior context |
| **F1** | Explicit conceptual switch within session | `correct_doc_ids` is the new-topic doc; `prior_turns` has the prior context |
| **F2** | Explicit document switch within session | `correct_doc_ids` is the new doc; `prior_turns` has the prior context |
| **G1** | Intra-doc section confusion (right doc, wrong section) | `correct_doc_ids` is the right doc; `forbidden_text_fragments` has the wrong-section markers |
| **G2** | Intra-doc Roman/numeric sibling within multi-part doc | Same as G1 |
| **H** | **Multi-document intent** — query needs BOTH docs | `correct_doc_ids` has 2+ entries |
| **I** | **Document correction from prior turn** | `correct_doc_ids` is the post-correction doc; `prior_turns` shows the wrong→corrected exchange |
| **J** | **Concept-vs-problem disambiguation** | `correct_doc_ids` is the right doc; label `concept_or_problem` explicitly |
| **K** | **Followup / clarification — no retrieval needed** | `expected_action = "no_retrieval"`; `correct_doc_ids` empty |
| **L** | **Off-topic / out-of-scope — redirect needed** | `expected_action = "redirect"`; `correct_doc_ids` empty |

## What to send me

For each prod conversation you want to add to the eval, send me a block per
target turn (not per session — one row tests one "target turn" + its prior
conversation context). Format doesn't matter; pasted markdown is fine and I'll
convert. Each block needs the following information:

### Required for every row

1. **Conversation snippet** — the messages leading up to the target turn (can
   be empty for first-turn rows), plus the target turn itself. Format like:
   ```
   T1 (user): "I need help with problem set 2 question 1"
   T1 (assistant): "Sure, problem set 2 question 1 asks you to..."
   T2 (user): "wait, I meant problem set 3"     ← target turn
   ```
2. **Which bucket** (one of A/B/C/D/E/F1/F2/G1/G2/H/I/J/K/L) — your best read
   on what failure mode this exercises. If you're unsure, say so; I'll classify.
3. **Ground truth** — exactly what the right action SHOULD have been:
   - For A–J: the document(s) that should have been retrieved (filename or
     display_name as it appears in the TA's docs). For H, list ALL the docs
     that should both surface (e.g. `["lecture 3", "lecture 4"]`).
   - For K: just write "no retrieval — followup to prior assistant turn"
   - For L: just write "redirect — off-topic" + optionally a category hint
4. **TA / course** — which TA this conversation is from (slug, name, or ta_id).

### Optional (helpful but not required)

- **Notes**: anything I should know about why this is a failure mode, what
  the current system does wrong, or what makes the row interesting.
- **For J rows specifically**: whether the student was looking for a **concept
  explanation** vs **help solving a problem** vs **both**. This is the key
  ground truth signal for J — the bucket exists because the same query string
  can mean either, and the contextualizer's classification determines what
  gets retrieved.
- **For I rows**: confirm the correction is explicit ("no I meant…", "wait,
  not pset 3, I meant pset 2") vs implicit (just naming a different doc with
  no acknowledgment of the prior turn). Explicit corrections are what I rows
  test.

### Example blocks (Wave 1 + Wave 2 mixed)

```
=== Row 1 — bucket E (cache anchoring) ===
TA: ECON S1117
Conversation:
  T1 (user): "I need help with problem 2 from problem set 1"
  T1 (assistant): "Problem 2 from problem set 1 is about regression discontinuity..."
  T2 (user): "what does the bar over x mean?"  ← target turn
Right answer: should retrieve from problem set 1 (continuation — same doc)
Notes: prior system has been failing on this; the cache should stay on pset 1.

=== Row 2 — bucket F2 (explicit document switch) ===
TA: ECON S1117
Conversation:
  T1 (user): "help me with problem 2 from problem set 1"
  T1 (assistant): "Problem 2 from problem set 1 is about..."
  T2 (user): "actually, let me ask about pset 3 question 2 instead"  ← target turn
Right answer: should retrieve from problem set 3 (pivot — new doc)
Notes: explicit doc-switch, cache should invalidate.

=== Row 3 — bucket H (multi-document intent) ===
TA: ECON S1117
Conversation:
  T1 (user): "compare problem set 2 question 1 and problem set 3 question 1"  ← target turn
Right answer: should retrieve from BOTH problem set 2 AND problem set 3
Notes: multi-doc query; we want both docs in top-5.

=== Row 4 — bucket I (document correction) ===
TA: ECON S1117
Conversation:
  T1 (user): "help me with pset 3"
  T1 (assistant): "Pset 3 covers..." (assume system retrieved pset 3 here)
  T2 (user): "wait no, I meant pset 2, not pset 3"  ← target turn
Right answer: should retrieve from pset 2 (correction overrides cache)
Notes: explicit correction phrasing.

=== Row 5 — bucket J (concept-vs-problem) ===
TA: ECON S1117
Conversation:
  T1 (user): "what is Bayes' theorem?"  ← target turn
Right answer: should retrieve from lecture material on Bayes (concept lookup)
concept_or_problem: concept
Notes: query phrasing signals concept lookup, not problem-solving. Test row.

=== Row 6 — bucket K (followup, no retrieval) ===
TA: ECON S1117
Conversation:
  T1 (user): "explain the OLS estimator"
  T1 (assistant): "The OLS estimator minimizes the sum of squared residuals..."
  T2 (user): "wait, what do you mean by 'residuals'?"  ← target turn
Right answer: NO RETRIEVAL — the student is asking about the assistant's prior message, not new course material.
Notes: today's system probably still retrieves; we want a HIT once the upstream
skip-gate ships.

=== Row 7 — bucket L (off-topic, redirect) ===
TA: ECON S1117
Conversation:
  T1 (user): "what's the capital of France?"  ← target turn
Right answer: REDIRECT — completely off-topic, not course material.
Notes: should trigger the off-topic short-circuit and return a redirect message,
no chunks.
```

That's it. Send blocks like the above and I'll batch-convert them into eval
JSONL rows with auto-derived hard negatives, forbidden docs, and the right
schema fields per bucket.

## How many rows per bucket?

The principle (per [feedback_sample_size](../.claude/projects/-Users-simonkleffner-Desktop-Maize-TA-Master-App-Dev-Maize-Blueprint-V2/memory/feedback_sample_size.md))
is ~20 prod-data rows per major failure mode for failure-rate reliability —
the buckets where rates are already reliable (E n=22, working n=40) tell us
that's the right floor.

**Target counts to reach this floor**:

| Bucket | Current prod rows | Target prod rows | Gap |
|---|---:|---:|---:|
| A | 3 | 15-20 | +~12 |
| B | 1 | 15-20 | +~14 |
| C | 4 | 15-20 | +~11 |
| D | 5 | 15-20 | +~10 |
| E | 19 | 15-20 | already there |
| F1 | 0 | 10-15 | +~10 |
| F2 | 1 | 10-15 | +~10 |
| G1 | 1 | 10-15 | +~10 |
| G2 | 2 | 10-15 | +~10 |
| H | 0 | 10-15 | +~10 (new) |
| I | 0 | 10-15 | +~10 (new) |
| J | 0 | 10-15 | +~10 (new) |
| K | 0 | 10-15 | +~10 (new) |
| L | 0 | 10-15 | +~10 (new) |

Roughly **130-160 new rows total** to bring everything up. That's a lot, but
the labeling per row is light. You don't need to hit every bucket at once —
**E expansion, F1/F2 (the doc-switching family), and the new H/I/J/K/L are
the highest-priority for the upcoming LangGraph adaptation validation**.
A/B/C/D/G can stay at current coverage until a future arc focuses on them.

## What happens after you send logs

1. I batch-convert blocks → JSONL rows via the extended `csv_to_jsonl.py`.
2. Validator runs to catch schema issues (incompatible action/bucket combos,
   missing fields per bucket type).
3. The expanded eval body becomes the validation gate for the LangGraph
   upstream-gate adaptation.
4. Pre-implementation baseline run captures today's behavior across all buckets.
5. Adaptation lands → re-run → compare scorecards bucket-by-bucket.

Pass criterion will be defined in the LangGraph adaptation plan (next working
doc in attached_assets/) — driven by the expanded eval's bucket-specific
metrics, not just overall hit@5.
