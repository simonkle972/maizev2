"""Convert a minimal-input CSV (per `eval/new_ta_input_template.csv`) into
validated JSONL rows appended to `eval/maize_eval_v1.jsonl`.

Workflow (see `eval/bootstrap_new_ta.md` "Minimal-input mode"):
- Human fills 4 columns per row: row_id, source, ta_id, query, correct_doc_ids
  (+ optional notes, prior_turns_json)
- Claude auto-derives the remaining 8 columns by running the live V2 retriever
  + applying pattern rules:
    * hard_negative_doc_ids — top retrieved docs minus correct
    * forbidden_doc_ids — solutions doc matching correct's assignment number
    * failure_type_target — A/B/C/D/E heuristic from the retrieval pattern
    * expected_intent — keyword heuristic on the query
    * not_in_corpus — verified against the TA's docs table
- Validates against the schema, prints a report, and (with --append) writes to
  maize_eval_v1.jsonl.

Run:
    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/csv_to_jsonl.py <input.csv> --dry-run
    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/csv_to_jsonl.py <input.csv> --append
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


JSONL_PATH = Path(__file__).parent / "maize_eval_v1.jsonl"
REQUIRED_INPUT_COLS = {"row_id", "source", "ta_id", "query", "correct_doc_ids"}
VALID_SOURCES = {"prod_log", "synthetic", "synthetic_working_case"}
VALID_FAILURE_TYPES = {"A", "B", "C", "D", "E", "F1", "F2", "G1", "G2"}

SOLUTION_REQUEST_PATTERNS = [
    r"\bshow me the answer\b",
    r"\bshow me the solution\b",
    r"\bgive me the answer\b",
    r"\bgive me the solution\b",
    r"\bwhat['s ]+ the answer\b",
    r"\banswer key\b",
]
CONCEPT_PATTERNS = [
    r"\bwhat (does|is|are) ",
    r"\bdefine\b",
    r"\bexplain\b",
    r"\bmeaning of\b",
    r"\bconcept of\b",
    r"\bdifference between\b",
]
PROBLEM_PATTERNS = [
    r"\bhelp (me )?(with|solving)\b",
    r"\bstuck on\b",
    r"\bhow do i (solve|approach)\b",
    r"\b(question|problem|part) \d+\b",
    r"\b(pset|p\.s\.|problem set|hw|lab|exam|quiz) \d+\b",
]
CORRECTION_PATTERNS = [
    r"\bnot (the )?(\w+)[,.]? i meant\b",
    r"\byou (have|got|gave) the wrong\b",
    r"\bi was asking about\b",
]


def split_pipe(s: str) -> list[str]:
    if not s or not s.strip():
        return []
    return [tok.strip() for tok in s.split("|") if tok.strip()]


def strip_ext(s: str) -> str:
    return re.sub(r"\.(pdf|docx|doc|pptx|ppt|xlsx|xls|txt|md)$", "", s, flags=re.IGNORECASE)


def parse_bool(s: str | None) -> bool | None:
    if s is None or s.strip() == "":
        return None
    return s.strip().lower() in {"true", "t", "yes", "y", "1"}


def detect_is_solution_request(query: str) -> bool:
    q = query.lower()
    for pat in SOLUTION_REQUEST_PATTERNS:
        if re.search(pat, q):
            return True
    return False


def detect_concept_or_problem(query: str) -> str:
    q = query.lower()
    concept_hit = any(re.search(p, q) for p in CONCEPT_PATTERNS)
    problem_hit = any(re.search(p, q) for p in PROBLEM_PATTERNS)
    if concept_hit and problem_hit:
        return "both"
    if concept_hit:
        return "concept"
    return "problem"  # default — most prod queries are problem-solving


def detect_correction(query: str, prior_turns: list) -> bool:
    if not prior_turns:
        return False
    q = query.lower()
    return any(re.search(p, q) for p in CORRECTION_PATTERNS)


DOC_TYPE_KEYWORDS = ["pset", "problemset", "homework", "hw", "lab", "quiz", "exam",
                     "midterm", "final", "lecture", "interactive", "ps"]


def extract_assignment_num(s: str) -> str | None:
    """Pull the assignment number out of a filename like 'econ117_pset02_2025B'."""
    m = re.search(r"(?:pset|ps|problemset|lab|hw|homework|quiz|exam|midterm|final|interactive)[\s_-]*0*(\d+)",
                  s.lower())
    return m.group(1) if m else None


def extract_doc_type(s: str) -> str | None:
    """Find a doc_type keyword (pset/lab/quiz/…) anywhere in the filename.

    Uses letter-boundaries (not \\b) because underscores are word chars in
    Python regex, so `\\bpset\\b` doesn't fire between `_` and `p` in
    'econ117_pset02_…'. We want: keyword preceded by non-letter (or start),
    followed by non-letter (digits OK).
    """
    sl = s.lower()
    # Longest match wins (pset before ps) so we don't return "ps" for "pset02"
    for kw in sorted(DOC_TYPE_KEYWORDS, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){kw}(?![a-z])", sl):
            return kw
    return None


def extract_roman(s: str) -> str | None:
    """Pull a trailing Roman numeral (I, II, III, IV) from the filename stem."""
    m = re.search(r"\b(I{1,3}V?|IV|V|VI{0,3})\b", s)
    return m.group(1) if m else None


def classify_failure_type(query: str, correct_docs: list[str], retrieved_docs: list[str],
                         prior_turns: list) -> str | None:
    """Conservative classifier: returns A/B/C/D/E or None (leave blank for human review).

    Heuristics:
      - E: prior_turns non-empty (cache anchoring is the only failure mode that *needs* prior turns)
      - D: top-1 retrieved is a "_solutions" doc and correct doc isn't
      - B: correct + top-1 share assignment_num but have different Roman numerals (extra problems I vs II)
      - A: top-1 doc differs from correct in doc_type (lab vs pset) but matches assignment_num
      - C: top-1 is clearly unrelated to correct (no shared assignment_num, no Roman match)
      - else: None (force human review)
    """
    if not retrieved_docs:
        return None
    if prior_turns:
        return "E"

    top1 = retrieved_docs[0].lower()
    correct = (correct_docs[0] if correct_docs else "").lower()

    top1_is_sol = "solution" in top1
    correct_is_sol = "solution" in correct
    if top1_is_sol and not correct_is_sol:
        return "D"

    correct_an = extract_assignment_num(correct)
    top1_an = extract_assignment_num(top1)
    correct_roman = extract_roman(correct)
    top1_roman = extract_roman(top1)

    if correct_an and top1_an and correct_an == top1_an:
        # same assignment number — distinguishing factor matters
        if correct_roman and top1_roman and correct_roman != top1_roman:
            return "B"
        # both reference assignment N but differ in doc_type prefix → A
        correct_prefix = re.split(r"[\s_-]", correct)[0]
        top1_prefix = re.split(r"[\s_-]", top1)[0]
        if correct_prefix != top1_prefix:
            return "A"
    if correct_an and top1_an and correct_an != top1_an:
        return "C"
    if not correct_an and not top1_an:
        return None
    return None


def derive_hard_negatives(retrieved_docs: list[str], correct_docs: list[str],
                          limit: int = 3) -> list[str]:
    """Top-N retrieved minus correct, capped at `limit`."""
    correct_set = {strip_ext(d).lower() for d in correct_docs}
    out = []
    for d in retrieved_docs:
        if strip_ext(d).lower() not in correct_set and d not in out:
            out.append(d)
        if len(out) >= limit:
            break
    return out


def find_solutions_doc_for(correct_doc: str, ta_doc_names: list[str]) -> str | None:
    """Find a 'solutions' doc in the corpus that matches the correct doc's
    assignment_num + doc_type. Prefers same doc_type (pset → pset_solutions,
    not quiz_solutions); falls back to any solutions doc with matching assignment_num.
    """
    an = extract_assignment_num(correct_doc)
    if not an:
        return None
    correct_type = extract_doc_type(correct_doc)
    candidates = [n for n in ta_doc_names
                  if "solution" in n.lower() and extract_assignment_num(n) == an]
    if correct_type:
        for name in candidates:
            if extract_doc_type(name) == correct_type:
                return name
    return candidates[0] if candidates else None


def run_retriever(ta_id: str, query: str, prior_turns: list) -> list[str]:
    """Run the live V2 retriever, return retrieved doc filenames (deduped, in order)."""
    from src.retriever import retrieve_context

    conv_history = []
    for turn in prior_turns:
        conv_history.append({"role": turn["role"], "content": turn["content"]})

    chunks, _ = retrieve_context(ta_id=ta_id, query=query, top_k=8,
                                 conversation_history=conv_history or None)
    seen = set()
    out = []
    for c in chunks:
        fn = c.get("file_name") or ""
        key = strip_ext(fn).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(strip_ext(fn))
    return out


def build_ta_doc_names(ta_id: str) -> list[str]:
    """All canonical doc names (display_name or original_filename, ext-stripped) for a TA."""
    from models import Document

    docs = Document.query.filter_by(ta_id=ta_id).all()
    return [strip_ext(d.display_name or d.original_filename) for d in docs]


def parse_csv_row(row: dict, lineno: int) -> tuple[dict | None, list[str]]:
    """Parse one CSV row into a partial eval row. Returns (parsed, errors)."""
    errors: list[str] = []

    rid = (row.get("row_id") or "").strip()
    if not rid or rid.startswith("EXAMPLE_"):
        return None, []  # silently skip example/blank rows

    missing = [c for c in REQUIRED_INPUT_COLS if not (row.get(c) or "").strip()]
    if missing:
        errors.append(f"line {lineno} ({rid!r}): missing required columns {missing}")
        return None, errors

    source = row["source"].strip()
    if source not in VALID_SOURCES:
        errors.append(f"line {lineno} ({rid!r}): source must be one of {VALID_SOURCES}, got {source!r}")

    try:
        prior_turns = json.loads(row.get("prior_turns_json") or "[]") or []
    except json.JSONDecodeError as e:
        errors.append(f"line {lineno} ({rid!r}): prior_turns_json is not valid JSON ({e})")
        prior_turns = []

    ftt = (row.get("failure_type_target") or "").strip() or None
    if ftt and ftt not in VALID_FAILURE_TYPES:
        errors.append(f"line {lineno} ({rid!r}): failure_type_target {ftt!r} not in {sorted(VALID_FAILURE_TYPES)}")

    parsed = {
        "row_id": rid,
        "source": source,
        "ta_id": row["ta_id"].strip(),
        "query": row["query"].strip(),
        "prior_turns": prior_turns,
        "correct_doc_ids": split_pipe(row["correct_doc_ids"]),
        "hard_negative_doc_ids_input": split_pipe(row.get("hard_negative_doc_ids", "")),
        "forbidden_doc_ids_input": split_pipe(row.get("forbidden_doc_ids", "")),
        "forbidden_text_fragments_input": split_pipe(row.get("forbidden_text_fragments", "")),
        "failure_type_target_input": ftt,
        "not_in_corpus_input": parse_bool(row.get("not_in_corpus")),
        "is_solution_request_input": parse_bool(row.get("is_solution_request")),
        "concept_or_problem_input": (row.get("concept_or_problem") or "").strip() or None,
        "document_corrected_from_prior_turn_input": parse_bool(row.get("document_corrected_from_prior_turn")),
        "notes": (row.get("notes") or "").strip(),
    }
    return parsed, errors


def autofill(parsed: dict, ta_doc_names: list[str]) -> tuple[dict, list[str]]:
    """Take a parsed CSV row + run retriever + apply heuristics → full eval row.

    Returns (final_row, per_row_notes) where per_row_notes describes what was auto-filled
    so the human can verify.
    """
    notes_log: list[str] = []

    retrieved = run_retriever(parsed["ta_id"], parsed["query"], parsed["prior_turns"])
    notes_log.append(f"retrieved top: {retrieved[:5]}")

    if parsed["hard_negative_doc_ids_input"]:
        hard_negs = parsed["hard_negative_doc_ids_input"]
        notes_log.append("hard_negatives: from CSV (kept human-provided)")
    else:
        hard_negs = derive_hard_negatives(retrieved, parsed["correct_doc_ids"], limit=3)
        notes_log.append(f"hard_negatives: auto-derived → {hard_negs}")

    if parsed["forbidden_doc_ids_input"]:
        forbidden = parsed["forbidden_doc_ids_input"]
        notes_log.append("forbidden: from CSV (kept human-provided)")
    else:
        forbidden = []
        for c in parsed["correct_doc_ids"]:
            sol = find_solutions_doc_for(c, ta_doc_names)
            if sol and sol not in forbidden and sol not in parsed["correct_doc_ids"]:
                forbidden.append(sol)
        notes_log.append(f"forbidden: auto-derived → {forbidden}")

    if parsed["failure_type_target_input"]:
        ftype = parsed["failure_type_target_input"]
        notes_log.append(f"failure_type: from CSV → {ftype}")
    else:
        ftype = classify_failure_type(parsed["query"], parsed["correct_doc_ids"],
                                      retrieved, parsed["prior_turns"])
        notes_log.append(f"failure_type: auto-classified → {ftype}")

    if parsed["is_solution_request_input"] is not None:
        is_sol = parsed["is_solution_request_input"]
    else:
        is_sol = detect_is_solution_request(parsed["query"])

    if parsed["concept_or_problem_input"]:
        cop = parsed["concept_or_problem_input"]
    else:
        cop = detect_concept_or_problem(parsed["query"])

    if parsed["document_corrected_from_prior_turn_input"] is not None:
        corrected = parsed["document_corrected_from_prior_turn_input"]
    else:
        corrected = detect_correction(parsed["query"], parsed["prior_turns"])
    notes_log.append(f"intent: is_sol={is_sol}, cop={cop}, corrected={corrected}")

    corpus_set = {n.lower() for n in ta_doc_names}
    not_in_corpus = parsed["not_in_corpus_input"]
    if not_in_corpus is None:
        missing_correct = [c for c in parsed["correct_doc_ids"] if c.lower() not in corpus_set]
        not_in_corpus = bool(missing_correct)
        if missing_correct:
            notes_log.append(f"not_in_corpus: TRUE (correct doc(s) not found in TA corpus: {missing_correct})")
        else:
            notes_log.append("not_in_corpus: false (all correct_doc_ids verified in corpus)")

    final_row: dict = {
        "row_id": parsed["row_id"],
        "source": parsed["source"],
        "ta_id": parsed["ta_id"],
        "query": parsed["query"],
        "prior_turns": parsed["prior_turns"],
        "correct_doc_ids": parsed["correct_doc_ids"],
        "hard_negative_doc_ids": hard_negs,
        "forbidden_doc_ids": forbidden,
        "failure_type_target": ftype,
        "expected_intent": {
            "is_solution_request": is_sol,
            "concept_or_problem": cop,
            "document_corrected_from_prior_turn": corrected,
        },
    }
    if parsed["forbidden_text_fragments_input"]:
        final_row["forbidden_text_fragments"] = parsed["forbidden_text_fragments_input"]
        notes_log.append(f"forbidden_text_fragments: from CSV → {parsed['forbidden_text_fragments_input']}")
    if not_in_corpus:
        final_row["not_in_corpus"] = True
    if parsed["notes"]:
        final_row["notes"] = parsed["notes"]

    return final_row, notes_log


def validate_row(row: dict, existing_row_ids: set[str]) -> list[str]:
    """Schema-level validation per `eval/schema.md`."""
    errors = []
    rid = row["row_id"]
    if rid in existing_row_ids:
        errors.append(f"{rid}: row_id already exists in maize_eval_v1.jsonl")
    if not row["correct_doc_ids"]:
        errors.append(f"{rid}: correct_doc_ids is empty")
    overlap = set(row["correct_doc_ids"]) & set(row["forbidden_doc_ids"])
    if overlap:
        errors.append(f"{rid}: docs in both correct_doc_ids and forbidden_doc_ids: {sorted(overlap)}")
    if row["failure_type_target"] is None:
        if row["source"] != "synthetic_working_case" and "working" not in rid:
            errors.append(
                f"{rid}: auto-classifier couldn't infer failure_type_target (current retriever likely "
                f"already works for this query, so no failure pattern is observable). Pre-fill "
                f"failure_type_target=A/B/C/D/E in the CSV if this row is a regression test, or rename "
                f"to use source='synthetic_working_case' if it's a working-case row."
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to the input CSV (per new_ta_input_template.csv)")
    parser.add_argument("--append", action="store_true",
                        help="Append converted rows to eval/maize_eval_v1.jsonl")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print converted rows + report without writing anywhere (default)")
    parser.add_argument("--out", type=str, default=None,
                        help="Write JSONL to this path instead of appending to maize_eval_v1.jsonl")
    args = parser.parse_args()

    if not args.append and not args.dry_run and not args.out:
        print("INFO: defaulting to --dry-run. Pass --append to write to maize_eval_v1.jsonl.")
        args.dry_run = True

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return 1

    from app import app

    existing_row_ids: set[str] = set()
    if JSONL_PATH.exists():
        with JSONL_PATH.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_row_ids.add(json.loads(line)["row_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    with app.app_context():
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)

        ta_doc_cache: dict[str, list[str]] = {}
        parse_errors: list[str] = []
        validation_errors: list[str] = []
        final_rows: list[dict] = []
        per_row_notes: dict[str, list[str]] = {}

        for lineno, row in enumerate(csv_rows, 2):  # +1 for header row
            parsed, errs = parse_csv_row(row, lineno)
            parse_errors.extend(errs)
            if not parsed:
                continue

            ta_id = parsed["ta_id"]
            if ta_id not in ta_doc_cache:
                ta_doc_cache[ta_id] = build_ta_doc_names(ta_id)

            final_row, log = autofill(parsed, ta_doc_cache[ta_id])
            verrs = validate_row(final_row, existing_row_ids | {r["row_id"] for r in final_rows})
            if verrs:
                validation_errors.extend(verrs)
                continue
            final_rows.append(final_row)
            per_row_notes[final_row["row_id"]] = log

    print(f"\n=== csv_to_jsonl conversion report ===")
    print(f"Input rows:         {len(csv_rows)}")
    print(f"Converted rows:     {len(final_rows)}")
    print(f"Parse errors:       {len(parse_errors)}")
    print(f"Validation errors:  {len(validation_errors)}")
    print()

    for r in final_rows:
        print(f"--- {r['row_id']} (ta={r['ta_id']}, failure={r['failure_type_target']}) ---")
        print(f"    query: {r['query'][:80]}")
        print(f"    correct: {r['correct_doc_ids']}")
        print(f"    hard_neg: {r['hard_negative_doc_ids']}")
        print(f"    forbidden: {r['forbidden_doc_ids']}")
        print(f"    intent: {r['expected_intent']}")
        if r.get("not_in_corpus"):
            print(f"    NOT_IN_CORPUS=True")
        for note in per_row_notes[r["row_id"]]:
            print(f"      · {note}")
        print()

    if parse_errors:
        print(f"--- {len(parse_errors)} parse errors ---")
        for e in parse_errors:
            print(f"  {e}")
        print()
    if validation_errors:
        print(f"--- {len(validation_errors)} validation errors ---")
        for e in validation_errors:
            print(f"  {e}")
        print()

    if parse_errors or validation_errors:
        print("Aborting: fix errors above before writing.")
        return 1

    if args.dry_run:
        print(f"Dry-run: would write {len(final_rows)} rows. Pass --append to write.")
        return 0

    out_path = Path(args.out) if args.out else JSONL_PATH
    mode = "a" if out_path == JSONL_PATH else "w"
    with out_path.open(mode) as f:
        for r in final_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(final_rows)} row(s) to {out_path} (mode={mode}).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
