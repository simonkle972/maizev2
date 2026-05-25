# Bootstrapping eval rows for a new TA

The eval harness (`eval/run_eval.py`) supports multiple TAs via the `ta_id` field on each row. The default ECON S1117 corpus (`EgZ14pvqEYzfQRTM`) has 64 hand-authored rows covering Failure Types A-E plus working cases. This document is the procedure for adding eval coverage for any other TA.

## When to add eval rows for a new TA

Add eval coverage for a TA when **any** of these is true:
- The TA is on the priority-monitor list for V2 retrieval changes (e.g. high-traffic, high-stakes, frequent failure reports)
- Production `qa_logs_v2` shows ≥1 retrieval failure or hard-negative top-1 for that TA in the last 1-2 weeks
- A Phase B / Phase C change (e.g. Anthropic Contextual Retrieval, Cohere reranker, top-of-funnel Option B) is about to ship and we need that TA in the regression battery

If none of the above applies — wait. Don't pre-author rows speculatively.

## Minimal-input mode (recommended for human-in-the-loop)

The lowest-effort workflow when Simon (or another labeler) provides the inputs and Claude fills the rest:

**Human provides:**
1. `ta_id` + TA name + 1-line subject description
2. TA indexed in the local Docker postgres (`maize_postgres_dev`) so the live retriever can be exercised against it
3. ~10-20 rows in `eval/new_ta_input_template.csv` format, filling only:
   - `row_id` (use the convention `{ta_slug}_{source}_{type}_{NN}`)
   - `source` (`prod_log` for queries lifted from qa_logs; `synthetic_working_case` for working cases)
   - `ta_id`
   - `query` (verbatim from qa_logs)
   - `correct_doc_ids` (pipe-delimited; the doc(s) the student was *actually* asking about — this is the one piece only the human can label confidently)
   - `notes` (free-form, e.g. qa_logs session_id + timestamp for provenance)

**Claude fills the remaining columns:**
- `prior_turns_json` — reconstructed by walking the qa_logs `session_id` (or `[]` for first-turn)
- `hard_negative_doc_ids` — by running the live V2 retriever against each query and subtracting `correct_doc_ids` from what it returns
- `forbidden_doc_ids` — pattern-based (solutions docs when the student is solving)
- `failure_type_target` — classified A/B/C/D/E from the retrieval pattern
- `expected_intent` (3 sub-fields) — derived from query wording
- `not_in_corpus` — verified against the TA's actual indexed docs
- JSONL conversion + validation (`eval/validate_schema.py`) + smoke-test (`run_eval.py --ta-id <new_ta_id> --limit 3`)

Hand the CSV to Claude; receive validated JSONL appended to `maize_eval_v1.jsonl` + a smoke-tested first 3 rows + a scorecard slice for the new TA.

## Two bootstrap paths

### Path 1: from production qa_logs (fastest, ~20-30 min per TA)

Best when prod has been running V2 for ≥1 week and `qa_logs_v2` has accumulated representative queries for the target TA.

1. **Sample queries.** Open the prod `qa_logs_v2` Google Sheet, filter by the target `ta_id`, select 8-12 queries that:
   - Cover the failure-type spread you care about (problem-set lookups, doc switches, conceptual queries, year-specified exams, etc.)
   - Include at least 2-3 "working" cases (queries that returned a sensible top-1 — these are the no-regression rows)
   - Include any rows the professor flagged as wrong via the in-app feedback mechanism (when that exists)

2. **For each sampled query, fill in the row fields:**
   - `row_id`: `<ta_slug>_<source>_<type>_<NN>` (e.g. `hbs2_real_typeA_01`, `mushroom_syn_typeC_01`).
   - `source`: `"prod_log"` for queries lifted from qa_logs, `"synthetic"` for ones you authored from scratch.
   - `ta_id`: the prod TA's id from the `teaching_assistants` table.
   - `query`: the verbatim query from qa_logs (`query` column).
   - `prior_turns`: reconstruct from the qa_logs `session_id` if it's a follow-up; otherwise `[]`. The Sheet's `session_id` column lets you walk back across rows.
   - `correct_doc_ids`: identify by reading the prod corpus + manually verifying. Use the TA's admin manage page to inspect doc list + content if needed.
   - `hard_negative_doc_ids`: docs that the current retriever would PLAUSIBLY return but that *look* right and aren't. The qa_logs `sources` column shows what V2 actually returned — if that's wrong, those are your hard negatives.
   - `forbidden_doc_ids`: docs that should NEVER appear. Solutions docs when the student is solving (Type D pattern). Cross-course docs from a sibling TA.
   - `failure_type_target`: one of `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, or `null` (for working cases). See `eval/schema.md` for definitions.
   - `expected_intent`: fill per `eval/schema.md`.
   - `notes`: include the original qa_logs row reference (`session_id` + timestamp), plus any context.

3. **Validate.** Run `python eval/validate_schema.py` — confirms every row parses and references valid corpus filenames.

4. **Smoke-test.** Run `python eval/run_eval.py --ta-id <new_ta_id> --limit 3` to confirm the new rows execute against the new TA's corpus.

### Path 2: hand-authored from corpus inspection (slowest, ~45-60 min per TA)

Use when prod traffic for the target TA is sparse or non-existent (e.g. a newly-created TA before public launch).

1. Inspect the TA's corpus via the admin manage page (`/admin/ta/<ta_id>`). Note doc filenames, categories, content_titles.
2. Read 2-3 representative docs to understand the course domain (so you can author plausible student queries).
3. Hand-write 5-10 queries across failure types per the same fields above.
4. Same validate + smoke-test steps.

**Quality bar**: ~5-10 rows per TA is enough to be diagnostic; >15 is over-investment for a measurement layer. The goal isn't a comprehensive eval — it's enough rows to catch retrieval regressions when a Phase B change ships.

## Running the eval per-TA

Once rows exist for multiple TAs:

```bash
# All rows across all TAs (cross-TA aggregate)
DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/run_eval.py --out /tmp/scorecard_all.md

# One TA only
DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/run_eval.py --ta-id EgZ14pvqEYzfQRTM --out /tmp/scorecard_econ.md

# All TAs, separate scorecards — loop in shell:
for TA in EgZ14pvqEYzfQRTM <other_ta_ids>; do
  DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/run_eval.py --ta-id $TA --out /tmp/scorecard_$TA.md
done
```

The scorecard header tells you whether you're in single-TA mode or cross-TA mode (auto-detected from distinct `ta_id`s present in the run).

## Cross-cutting hygiene

- **Keep eval rows alongside the code change that needed them.** If you're shipping Cohere reranker and adding 5 eval rows for HBS2 to verify it doesn't regress: commit both in the same PR.
- **Don't seed all 25 prod TAs preemptively.** Most TAs don't need eval coverage; you'd be authoring rows you'll never run. Wait for an actual need (Phase B change about to ship, or prod failure report) before adding rows.
- **Stale rows.** If a TA's corpus changes substantially (docs added/removed/reclassified), the old eval rows may reference removed docs or test obsolete failure modes. `validate_schema.py` catches dangling `correct_doc_ids` references; recheck `forbidden_doc_ids` and `hard_negative_doc_ids` manually after a major corpus change.
