# Maize TA — Retrieval Eval Harness

This directory is the eval set + harness for Phase A of the retrieval redesign. See [attached_assets/maize-retrieval-redesign-plan.md](../attached_assets/maize-retrieval-redesign-plan.md) for full context.

**Purpose.** Provide an objective scorecard ("Recall@5 per failure type, working-case preservation, forbidden-retrieval rate") that we run against the CURRENT retriever to produce a baseline, then re-run against the refined retriever after implementation to prove the change worked without regressions.

## Files

| File | What |
|---|---|
| `maize_eval_v1.jsonl` | Labeled eval set. Each line is a JSON record per the schema in `schema.md`. Currently a sample; full ~94 rows are being labeled in chunks. |
| `schema.md` | Field definitions for `maize_eval_v1.jsonl`. The contract that the harness consumes. |
| `corpus_econ_s1117.md` | Inventory of the 30 known documents in the ECON S1117 TA, as referenced in prod logs. Used to validate that every `correct_doc_ids` / `hard_negative_doc_ids` / `forbidden_doc_ids` value points to a real doc. |
| `run_eval.py` | (TBD) — scoring runner. Loads the JSONL, calls `retrieve_context()` on each query, computes Recall@5 + per-failure-type metrics + forbidden-retrieval rate, prints a scorecard. |
| `test_retrieval_regression.py` | (TBD) — pytest hookup via DeepEval. Parametrized over the JSONL rows. Fails CI on regressions to known-good rows. |
| `baseline_scorecard_pre_refinement.md` | (TBD) — committed "before" numbers from running `run_eval.py` against the current retriever. Reference point for proving the refined retriever is an improvement. |

## Status

- [x] Schema fixed (`schema.md`)
- [x] Corpus inventory derived from prod logs (`corpus_econ_s1117.md`)
- [x] 43 prod rows + 21 synthetic rows labeled in `maize_eval_v1.jsonl` (64 total, validated clean)
- [x] Schema validation script (`validate_schema.py`)
- [x] `run_eval.py` written
- [x] `pytest` hookup (`test_retrieval_regression.py`)
- [ ] Baseline scorecard against current retriever (next: requires ECON S1117 TA indexed in local DB)

## How to run

```bash
# 1. Validate the eval set parses cleanly:
python eval/validate_schema.py

# 2. (One-time) ensure ECON S1117 TA's documents are indexed in the local DB:
#    a. Confirm the TA exists: docker exec maize_postgres_dev psql -U maize_dev -d maize_ta_dev \
#       -c "SELECT id, name FROM teaching_assistants WHERE id = 'bv1COF3YbWV28OKv';"
#    b. If not present, the TA must be created and its documents uploaded + indexed.
#       See app.py / professor.py for the upload + reindex flow.

# 3. Run the scorecard against the current retriever (baseline):
DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/run_eval.py --baseline

# 4. Run as a pytest regression suite (only working-case rows asserted; failure-type
#    rows are reported via run_eval.py, not asserted here):
DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 pytest eval/test_retrieval_regression.py -v

# 5. Smoke-test a single row while debugging:
DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/run_eval.py --row-id econ_s1117_real_typeA_01
```

## Conventions

- **Doc identifiers.** We use the document's `file_name` (which in this codebase is `Document.display_name or Document.original_filename`) as the canonical identifier in eval rows. These strings exactly match what appears in the prod `sources` log column, so retrieval output can be compared directly without normalization. Note: some doc names contain commas (e.g., `econ117_pset03- 2025, solutions`) — keep them intact; never normalize.
- **Failure type labels.** Match the parked plan exactly: A (sibling-bucket collisions), B (Roman-numeral / version-letter siblings), C (lookalike-unrelated filename match), D (problem-vs-solutions), E (cache anchoring across turns). Working cases use `null`.
- **Hard negatives.** A "hard negative" is a doc that the current system WOULD plausibly retrieve and that LOOKS right but ISN'T. Different from "forbidden" — forbidden docs (e.g., solutions when student is solving) might be retrieved by current bad behavior but should never appear post-refactor.
- **Multi-turn rows.** Use `prior_turns` array (ordered, oldest first) to give the retriever the conversation history. The harness will replay these to the retriever before evaluating the target query.
