# Eval row schema (`maize_eval_v1.jsonl`)

Each line is a single JSON record. UTF-8. No trailing newline inside fields.

## Field reference

| Field | Type | Required | Description |
|---|---|---|---|
| `row_id` | string | yes | Stable unique identifier. Prefix indicates source: `econ_s1117_real_*` (prod log), `econ_s1117_syn_*` (synthetic hard-negative), `working_*` (working-case rows that must not regress). |
| `source` | enum | yes | `"prod_log"`, `"synthetic"`, or `"synthetic_working_case"`. |
| `ta_id` | string | yes | The TA the query runs against. The eval file supports multiple TAs; existing rows are all ECON S1117 (`EgZ14pvqEYzfQRTM`). `run_eval.py --ta-id <id>` filters to one TA's rows; without the flag the harness runs all rows across all TAs (cross-TA mode). See `eval/bootstrap_new_ta.md` for how to add eval rows for a new TA. |
| `query` | string | yes | The student's message (the target turn we're evaluating retrieval for). |
| `prior_turns` | array of `{"role": "user"|"assistant", "content": str}` | yes (may be empty) | Conversation history before this query, oldest first. Empty for first-turn rows. |
| `correct_doc_ids` | array of strings | yes | The doc(s) the retriever SHOULD return. Strings match the canonical `file_name` (= `display_name or original_filename`). Must each appear in `corpus_econ_s1117.md` UNLESS the row carries `not_in_corpus: true` (see below). |
| `hard_negative_doc_ids` | array of strings | yes (may be empty) | Docs that the current bad system would plausibly retrieve and that *look* right but aren't. Used to compute "discrimination" metrics — did the new retriever pick correct over hard-negative? |
| `forbidden_doc_ids` | array of strings | yes (may be empty) | Docs that should NEVER appear in retrieved output for this query. Most commonly: solutions docs when the student is asking for help solving (Type D). A run that returns any forbidden doc fails this row. |
| `failure_type_target` | one of `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, `"F1"`, `"F2"`, `"G1"`, `"G2"`, or `null` | yes | Which failure mode this row exercises. `null` for working-case rows that must not regress. See "Failure type taxonomy" below. |
| `not_in_corpus` | bool | no (default false) | Set to `true` when one or more `correct_doc_ids` cannot be verified against the indexed corpus (see [corpus_econ_s1117.md](corpus_econ_s1117.md) "Known gaps"). These rows are reported in a separate bucket by the harness — the failure may be an indexing problem rather than a retrieval problem, so retrieval-layer pass/fail is non-diagnostic. Resolve before declaring v1 complete. |
| `forbidden_text_fragments` | array of strings | no (default `[]`) | Substrings that should NOT appear in any retrieved chunk's text. Used to score **G** failures (right doc, wrong section): if a row tests Part I retrieval, set `["Part II", "Part 2"]` here, and the harness flags `forbidden_text_hit` if any retrieved chunk contains those substrings. Case-insensitive matching. Distinct from `forbidden_doc_ids` (which is doc-level). |
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

## Failure type taxonomy

Doc-routing failures (failure surfaces at which-doc-did-we-pick layer):

| Type | Definition | Example |
|---|---|---|
| **A** | Lab vs PS — sibling docs that bucket the same way (doc_type/assignment_number collision) | Query "help with question 2 from PS2" returns `Lab2` instead of `PS2` |
| **B** | Roman-numeral siblings across *different* docs | Query about `Extra Problems II` returns `Extra Problems I` |
| **C** | Lookalike-unrelated — top-1 looks plausible but is on the wrong topic | — |
| **D** | Problem-vs-solutions — student is solving, retriever returned the solutions doc | — |
| **E** | Cache anchoring — multi-turn doc stickiness; turn-1's doc carries into turn-2 even when it shouldn't (or cache loses a previously-correct doc) | — |
| **F1** | Explicit conceptual switch — multi-turn; student explicitly names a new concept (e.g. "now explain LM curve") and retrieval should pivot at the chunk level. Tests contextualizer's `preserved_by_concept_lookup` short-circuit. | EC 112: T1 = "homework 1", T2 = "explain circular flow" — retrieval should fetch the circular-flow lecture, not stay on homework 1 |
| **F2** | Explicit document switch — multi-turn; student explicitly names a new doc (e.g. "now help with PS3", "what about lecture 2?") and retrieval should pivot at the doc level | EC 112: T1 = "circular flow", T2 = "help with questions 2-6 from homework 1" — retrieval should switch to the homework 1 doc |

Chunk-routing failures (failure surfaces at which-chunks-within-doc layer; doc-level metric would falsely register these as hits — measured via `forbidden_text_fragments`):

| Type | Definition | Example |
|---|---|---|
| **G1** | Intra-doc section confusion — right doc retrieved but chunks come from the wrong section/part of it | MGT 410: query "problem 2a from part 1 of 2024 final" → pulled the 2024 final doc, but chunks were from Part 2 |
| **G2** | Intra-doc numeric/Roman sibling — within a multi-part doc, retrieved chunks match the wrong sub-part with the same identifier | MGT 410: query "let's work on 2b" (Part I 2b) → retrieved Part II's 2b instead |

## Cache fidelity: warm vs cold mode

The harness has two modes, controlled by `--cold-cache` on `run_eval.py`:

- **Warm cache (default)**: for any row with `prior_turns`, each prior user-turn query is replayed through `retrieve_context` sequentially with a SHARED `session_id` BEFORE the target query runs. This populates the session-level doc cache the same way a real student session would. The recorded prior-turn assistant content is passed verbatim as conversation_history at each replay step — we never regenerate. Required for honestly testing the cache-anchoring failure modes (E) and explicit-switch modes (F1, F2), where the failure depends on what was cached from earlier turns.
- **Cold cache (`--cold-cache`)**: only the target query runs, with `prior_turns` as conversation_history but no warm cache state. ~3× faster on multi-turn rows; useful for quick smoke tests. Will MISS cache-state-dependent failures.

For single-turn rows (no `prior_turns`), warm and cold are equivalent.

Cost: warm-cache adds one `retrieve_context` call per prior turn. A 6-turn row becomes 7 calls instead of 1. Across the eval body, this typically takes total cost from ~$3-5 → ~$10-15 per run. Worth it for fidelity on doc-switching work — the primary failure-mode focus.

## Validation rules (enforced by a sanity-check script before the harness runs)

1. `row_id` is unique across the file.
2. Every doc string in `correct_doc_ids`, `hard_negative_doc_ids`, `forbidden_doc_ids` is listed in `corpus_econ_s1117.md` (or has an explicit `not_in_corpus: true` marker indicating an indexing gap to investigate).
3. `correct_doc_ids` is non-empty (we always know what should have been retrieved; if we don't, the row isn't ready).
4. `failure_type_target` is one of the five letters or `null`; if `null`, the row's `source` must be `synthetic_working_case`.
5. No doc appears in both `correct_doc_ids` and `forbidden_doc_ids` (contradiction).
