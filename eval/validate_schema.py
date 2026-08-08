"""Validate eval/maize_eval_v1.jsonl against the schema documented in schema.md.

Run: python eval/validate_schema.py
Exit code 0 on success, 1 on validation errors.

Checks:
  - Every line is valid JSON
  - row_id is unique across the file
  - Required fields present per schema
  - failure_type_target is one of A/B/C/D/E or null
  - If failure_type_target is null, source must be 'synthetic_working_case' OR row_id contains 'working'
  - No doc appears in both correct_doc_ids and forbidden_doc_ids on the same row
  - Reports rows flagged not_in_corpus
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED = {"row_id", "source", "ta_id", "query", "prior_turns",
            "correct_doc_ids", "hard_negative_doc_ids", "forbidden_doc_ids",
            "failure_type_target", "expected_intent"}
# Wave 2 (2026-05-31) added H/I/J/K/L for intent-classification dimensions.
# See eval/schema.md "Wave 2 intent-classification failures".
VALID_FAILURE_TYPES = {"A", "B", "C", "D", "E", "F1", "F2", "G1", "G2",
                        "H", "I", "J", "K", "L", None}
VALID_SOURCES = {"prod_log", "synthetic", "synthetic_working_case"}
VALID_EXPECTED_ACTIONS = {"retrieve", "redirect", "no_retrieval"}
VALID_INTENT_CLASSES = {"continuation", "concept_lookup", "pivot",
                         "clarification", "new", "off_topic"}


def main() -> int:
    path = Path(__file__).parent / "maize_eval_v1.jsonl"
    if not path.exists():
        print(f"ERROR: {path} not found")
        return 1

    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []

    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {lineno}: invalid JSON — {e}")
                continue
            rows.append((lineno, row))

    row_ids = Counter(row.get("row_id") for _, row in rows)
    for rid, count in row_ids.items():
        if count > 1:
            errors.append(f"row_id {rid!r} appears {count} times (must be unique)")

    failure_dist: Counter = Counter()
    source_dist: Counter = Counter()
    not_in_corpus_count = 0

    for lineno, row in rows:
        rid = row.get("row_id", f"<line {lineno}>")

        missing = REQUIRED - row.keys()
        if missing:
            errors.append(f"{rid}: missing required fields {sorted(missing)}")
            continue

        if row["source"] not in VALID_SOURCES:
            errors.append(f"{rid}: source must be one of {VALID_SOURCES}, got {row['source']!r}")

        if row["failure_type_target"] not in VALID_FAILURE_TYPES:
            errors.append(f"{rid}: failure_type_target must be one of {sorted(t for t in VALID_FAILURE_TYPES if t)} or null, got {row['failure_type_target']!r}")

        ftf = row.get("forbidden_text_fragments")
        if ftf is not None and not isinstance(ftf, list):
            errors.append(f"{rid}: forbidden_text_fragments must be a list of strings if present, got {type(ftf).__name__}")

        # Wave 2: expected_action determines whether correct_doc_ids must be populated.
        expected_action = row.get("expected_action", "retrieve")
        if expected_action not in VALID_EXPECTED_ACTIONS:
            errors.append(f"{rid}: expected_action must be one of {sorted(VALID_EXPECTED_ACTIONS)}, got {expected_action!r}")

        if not isinstance(row["correct_doc_ids"], list):
            errors.append(f"{rid}: correct_doc_ids must be a list")
        elif expected_action == "retrieve" and not row["correct_doc_ids"]:
            errors.append(f"{rid}: correct_doc_ids must be non-empty when expected_action='retrieve'")
        elif expected_action in {"redirect", "no_retrieval"} and row["correct_doc_ids"]:
            warnings.append(f"{rid}: correct_doc_ids non-empty but expected_action={expected_action!r} — HIT semantics ambiguous")

        overlap = set(row["correct_doc_ids"]) & set(row["forbidden_doc_ids"])
        if overlap:
            errors.append(f"{rid}: docs in both correct_doc_ids and forbidden_doc_ids: {sorted(overlap)}")

        if row["failure_type_target"] is None:
            looks_like_working = (
                row["source"] == "synthetic_working_case"
                or "working" in rid
            )
            if not looks_like_working:
                warnings.append(f"{rid}: failure_type_target=null but row_id/source doesn't indicate working case")

        intent = row.get("expected_intent", {})
        for field in ("is_solution_request", "concept_or_problem", "document_corrected_from_prior_turn"):
            if field not in intent:
                warnings.append(f"{rid}: expected_intent missing field {field!r}")

        # Wave 2: intent_class is optional but constrained when present.
        intent_class = intent.get("intent_class")
        if intent_class is not None and intent_class not in VALID_INTENT_CLASSES:
            errors.append(f"{rid}: expected_intent.intent_class must be one of {sorted(VALID_INTENT_CLASSES)}, got {intent_class!r}")

        # Per-bucket Wave 2 invariants (rules 8-12 in schema.md).
        ft = row["failure_type_target"]
        if ft == "K":
            if expected_action != "no_retrieval":
                errors.append(f"{rid}: failure_type_target='K' requires expected_action='no_retrieval', got {expected_action!r}")
            if intent_class != "clarification":
                errors.append(f"{rid}: failure_type_target='K' requires expected_intent.intent_class='clarification', got {intent_class!r}")
        elif ft == "L":
            if expected_action != "redirect":
                errors.append(f"{rid}: failure_type_target='L' requires expected_action='redirect', got {expected_action!r}")
            if intent_class != "off_topic":
                errors.append(f"{rid}: failure_type_target='L' requires expected_intent.intent_class='off_topic', got {intent_class!r}")
        elif ft == "H":
            if len(row["correct_doc_ids"]) < 2:
                errors.append(f"{rid}: failure_type_target='H' (multi-doc) requires len(correct_doc_ids) >= 2, got {len(row['correct_doc_ids'])}")
        elif ft == "I":
            if not intent.get("document_corrected_from_prior_turn"):
                errors.append(f"{rid}: failure_type_target='I' requires expected_intent.document_corrected_from_prior_turn=true")
            if len(row.get("prior_turns") or []) < 2:
                errors.append(f"{rid}: failure_type_target='I' requires prior_turns to contain the wrong-then-corrected exchange (>=2 messages)")
        elif ft == "J":
            cop = intent.get("concept_or_problem")
            if cop not in {"concept", "problem", "both"}:
                errors.append(f"{rid}: failure_type_target='J' requires expected_intent.concept_or_problem ∈ {{concept, problem, both}}, got {cop!r}")

        failure_dist[row["failure_type_target"]] += 1
        source_dist[row["source"]] += 1
        if row.get("not_in_corpus"):
            not_in_corpus_count += 1

    print(f"=== eval/maize_eval_v1.jsonl validation ===")
    print(f"Total rows: {len(rows)}")
    print()
    print(f"Failure type distribution:")
    for k in ["A", "B", "C", "D", "E", "F1", "F2", "G1", "G2",
              "H", "I", "J", "K", "L", None]:
        print(f"  {k!s:>6}: {failure_dist[k]}")
    print()
    print(f"Source distribution:")
    for k, v in source_dist.most_common():
        print(f"  {k:>25}: {v}")
    print()
    print(f"not_in_corpus rows: {not_in_corpus_count}")
    print()

    if warnings:
        print(f"--- {len(warnings)} warnings ---")
        for w in warnings:
            print(f"  WARN: {w}")
        print()

    if errors:
        print(f"--- {len(errors)} errors ---")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1

    print("OK — schema validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
