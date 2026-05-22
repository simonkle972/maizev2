# Eval row schema (`maize_eval_v1.jsonl`)

Each line is a single JSON record. UTF-8. No trailing newline inside fields.

## Field reference

| Field | Type | Required | Description |
|---|---|---|---|
| `row_id` | string | yes | Stable unique identifier. Prefix indicates source: `econ_s1117_real_*` (prod log), `econ_s1117_syn_*` (synthetic hard-negative), `working_*` (working-case rows that must not regress). |
| `source` | enum | yes | `"prod_log"`, `"synthetic"`, or `"synthetic_working_case"`. |
| `ta_id` | string | yes | The TA the query runs against. For now always `bv1COF3YbWV28OKv` (ECON S1117). |
| `query` | string | yes | The student's message (the target turn we're evaluating retrieval for). |
| `prior_turns` | array of `{"role": "user"|"assistant", "content": str}` | yes (may be empty) | Conversation history before this query, oldest first. Empty for first-turn rows. |
| `correct_doc_ids` | array of strings | yes | The doc(s) the retriever SHOULD return. Strings match the canonical `file_name` (= `display_name or original_filename`). Must each appear in `corpus_econ_s1117.md` UNLESS the row carries `not_in_corpus: true` (see below). |
| `hard_negative_doc_ids` | array of strings | yes (may be empty) | Docs that the current bad system would plausibly retrieve and that *look* right but aren't. Used to compute "discrimination" metrics — did the new retriever pick correct over hard-negative? |
| `forbidden_doc_ids` | array of strings | yes (may be empty) | Docs that should NEVER appear in retrieved output for this query. Most commonly: solutions docs when the student is asking for help solving (Type D). A run that returns any forbidden doc fails this row. |
| `failure_type_target` | one of `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, or `null` | yes | Which failure mode this row exercises. `null` for working-case rows that must not regress. Definitions match the parked plan. |
| `not_in_corpus` | bool | no (default false) | Set to `true` when one or more `correct_doc_ids` cannot be verified against the indexed corpus (see [corpus_econ_s1117.md](corpus_econ_s1117.md) "Known gaps"). These rows are reported in a separate bucket by the harness — the failure may be an indexing problem rather than a retrieval problem, so retrieval-layer pass/fail is non-diagnostic. Resolve before declaring v1 complete. |
| `expected_intent` | object (see below) | yes | The structured intent signals the refined intent classifier should produce. Used to evaluate the intent-classifier stage independently of retrieval. |
| `notes` | string | no | Free-form. Used for "row N from initial test" provenance, the original Human Notes field from the prod log, or labeler comments. |

## `expected_intent` shape

```json
{
  "is_solution_request": false,
  "concept_or_problem": "problem",
  "document_corrected_from_prior_turn": false
}
```

| Sub-field | Type | Notes |
|---|---|---|
| `is_solution_request` | bool | True if the student is explicitly asking for the answer/solution (e.g., "show me the answer key for verification"). Default false — pedagogically dangerous to true on most rows. |
| `concept_or_problem` | enum | `"problem"` (student is solving), `"concept"` (student is asking what something means), `"both"`. |
| `document_corrected_from_prior_turn` | bool | True if this turn explicitly corrects a wrong retrieval from a prior turn ("not Lab 2, I meant PS 2"). Subsumes Phase B's mandate. |

## Example row (Type B failure from prod)

```json
{
  "row_id": "econ_s1117_real_13",
  "source": "prod_log",
  "ta_id": "bv1COF3YbWV28OKv",
  "query": "I need help with question 14 from extra problems II",
  "prior_turns": [],
  "correct_doc_ids": ["extra problems II"],
  "hard_negative_doc_ids": ["extra problems I-1-1", "extra problems I - solutions-1"],
  "forbidden_doc_ids": ["extra problems I - solutions-1"],
  "failure_type_target": "B",
  "expected_intent": {
    "is_solution_request": false,
    "concept_or_problem": "problem",
    "document_corrected_from_prior_turn": false
  },
  "notes": "row #13 from initial test (iE3yw41Vpl). Current retriever returns 'extra problems I-1-1'. Note: 'extra problems II' is the expected canonical name; the doc may not literally exist with that name in the corpus if it was indexed under a sibling alias — verify against corpus_econ_s1117.md."
}
```

## Validation rules (enforced by a sanity-check script before the harness runs)

1. `row_id` is unique across the file.
2. Every doc string in `correct_doc_ids`, `hard_negative_doc_ids`, `forbidden_doc_ids` is listed in `corpus_econ_s1117.md` (or has an explicit `not_in_corpus: true` marker indicating an indexing gap to investigate).
3. `correct_doc_ids` is non-empty (we always know what should have been retrieved; if we don't, the row isn't ready).
4. `failure_type_target` is one of the five letters or `null`; if `null`, the row's `source` must be `synthetic_working_case`.
5. No doc appears in both `correct_doc_ids` and `forbidden_doc_ids` (contradiction).
