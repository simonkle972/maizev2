# Labelling guide — eval set v2 (target ~120 rows)

You fill **two columns on most rows.** Everything else is either derived from the QA Master
export or filled by me. This file is the spec for what to fill and, more importantly, **which
rows to pick**.

## Decisions made 2026-09-07 (apply these globally, don't re-litigate per row)

**1. Solutions docs are CORRECT, never forbidden.** Retrieval's job is to find the material; the
generator decides what to withhold, and it already does so by content. So `label_forbidden_docs`
is now almost always blank — reserve it for genuinely wrong material (a sibling course's doc),
not for answer keys.

*My side of this:* `csv_to_jsonl.py` currently auto-populates `forbidden_doc_ids` with the
matching solutions doc by pattern — that has to come out, and the existing Type D rows in the
92-row set need revisiting under the new policy. Some of them will flip from fail to pass. That
is a real change to historical numbers and I'll report it as one rather than quietly rebaselining.

**2. Supporting docs are a first-class label, not an optional extra.** The product promise is that
the TA teaches with *the lecturer's own methodology*. That means on a problem-set query the
must-have doc is the problem set, and the lecture / book chapter that teaches the underlying
method is a nice-to-have the generator genuinely needs. Today nothing measures whether it arrives
— and the mechanism that would supply it is gated off on every conceptual query. Filling this
column is what makes that measurable.

## The one habit to change

In the ec112 sheet, `Correct doc` was filled **only when retrieval got it wrong**. That is the
natural way to review, and it is the wrong input for this harness.

`eval/candidate_ceiling.py` drops any row with no label (`load_labelled_openers`, line 107), and
the four gates (L0–L3) are all computed *against the label* — not against whether the answer read
well. A blank label on a working row means:

- the row is invisible to the ceiling measurement, and
- **we cannot detect a regression on it**, which is the entire point of keeping working cases.

**So: fill `label_correct_docs` on every row, including the ones that worked.** On a working row
you can usually copy it straight out of the `sources` column.

## Columns to add

Keep the QA Master export exactly as it is and add these to the left of it (the ec112 sheet
already does this with `Human notes` / `Correct doc`).

| Column | Fill it? | What it means |
|---|---|---|
| `label_correct_docs` | **every row** | The doc(s) the answer should be built from. Pipe-delimited: `docA\|docB`. Blank **only** when `label_expected_action` is set. |
| `label_verdict` | **every row** | Your read of what actually happened: `good`, `wrong_doc`, `right_doc_wrong_section`, or `unsure`. |
| `label_mode` | only when >1 doc | `any` = either is acceptable. `all` = the answer genuinely needs all of them. Default is `any`. See below — this is a real gap in today's harness. |
| `label_supporting_docs` | **structured/problem rows** | Your "nice-to-have" column. The lecture, book chapter, or notes that teach the method behind the question. Pipe-delimited. **Never scored as a miss** — it gets its own rate, so an empty cell costs nothing but a filled one is signal. |
| `label_forbidden_docs` | almost never | Genuinely wrong material only (e.g. a sibling course's doc). **Not** answer keys — see decision 1. |
| `label_expected_action` | ~8 rows | `no_retrieval` (query references the prior turn — "thanks", "what do you mean?") or `redirect` (off-topic). Leave blank otherwise. |
| `label_note` | when unsure | Free text. Flag ambiguity here rather than guessing — an ambiguous row I know about is useful; one I don't is contamination. |

### `label_mode` — why it exists

26 of today's 92 rows already carry 2+ labels, and they mean two different things that nothing
currently distinguishes:

- **alternatives** (18 rows) — `[exam_part_2_solutions, exam_part_2]`, either is fine → `any`
- **a mix** (6 rows) — homework 1 *plus* the four lectures that teach it → `all`

`all_correct_in_top_5` is already computed for every row but reported for none. It's correct for
`all` rows and nonsense for `any` rows, which is why it can't be switched on until this column
exists.

## What I derive — do not fill these

From the export's own columns: `row_id`, `source`, `prior_turns` (walked from `session_id` +
`timestamp`), `hard_negative_doc_ids` (from `sources` minus your label), `failure_type_target`,
`expected_intent`, `not_in_corpus`, and which of the four routing paths the row exercises (from
`retrieval_method`). You don't need to guess at path coverage — I can measure it after the fact
and tell you where the set is thin.

## Which rows to pick — the composition that matters

Aim for **~120 rows to land ≥100 usable** after corpus pruning. Rough targets:

| Slice | Target | Why |
|---|---:|---|
| **Turn-1 openers** | **≥60** | `candidate_ceiling.py` runs on session openers *only*. Today there are 49. This slice is the ceiling measurement — everything else in the funnel work depends on it. |
| Working cases (verdict `good`) | ~40% of total | Without these the harness measures failure only and cannot catch a regression. |
| Spread across ≥3 TAs | ≥30 non-econ-s1117 | Every routing number to date comes from one corpus in one subject. |
| Multi-turn doc switches | ~15 | Turn 2+ names a new doc. Cache anchoring, currently the least measured behaviour. |
| Conceptual queries | ~10 | Supplementary teaching material is skipped on *every* conceptual query today (50 prod turns). Entirely unmeasured. |
| Contentless / off-topic | ~8 | Blank `label_correct_docs` + `label_expected_action`. These get their own bucket so they stop scoring as routing failures. |
| Known-hard shapes | ~10 | Roman-numeral siblings (*extra problems I* vs *II*), year siblings (*final-fall-2018* vs *2019*), abbreviations (`ps1` vs `pset1`). These are the Path 4 collapse rows. |

Slices overlap — a turn-1 conceptual working case counts in three of them.

## Filename format

Strings must match the canonical file name (`display_name or original_filename`). Safest source,
in order: the `sources` / `hybrid_doc_filename` columns of the export, or the TA's document list
in admin. **Paste, don't retype** — `Central Tendencies-2-1` has already cost us a row by being a
truncated version of `week 1 practice problems - Random Variable, Central Tendencies-2-1`.

I'll fuzzy-resolve and **report** anything that doesn't match rather than silently dropping it, so
a near-miss costs a line in a report, not a row.

## Prerequisite I need to confirm

The harness runs against the **local Docker postgres**, not prod. Rows for a TA that isn't indexed
locally cannot execute. `econ-s1117-local` is there; ec112 and mgt410 need checking before you
invest labelling time in them.
