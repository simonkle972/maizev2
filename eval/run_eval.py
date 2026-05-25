"""Eval harness for Phase A retrieval redesign.

Loads eval/maize_eval_v1.jsonl, runs each row through `retrieve_context()` in
src/retriever.py, and reports a per-failure-type scorecard. Designed to be run
twice: once against the CURRENT retriever (baseline) and once against the
REFINED retriever after the implementation gate. Comparing the two scorecards
is what proves the architectural change actually fixed Failure Types A-E
without regressing working cases.

Usage:
    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/run_eval.py
    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/run_eval.py --baseline  # also writes baseline_scorecard_pre_refinement.md
    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/run_eval.py --row-id econ_s1117_real_typeB_01  # run just one row

Prereqs: the ECON S1117 TA's documents must be indexed in the local DB the
script connects to. If not present, the script prints a helpful error.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Allow `from app import app` and `from src.retriever import ...` to resolve
# when this script is run as `python eval/run_eval.py` (Python adds the
# script's directory to sys.path, not cwd).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

EVAL_FILE = Path(__file__).parent / "maize_eval_v1.jsonl"
BASELINE_OUT = Path(__file__).parent / "baseline_scorecard_pre_refinement.md"


@dataclass
class RowResult:
    row_id: str
    failure_type: str | None
    source: str
    ta_id: str
    not_in_corpus: bool
    correct_doc_ids: list[str]
    hard_negative_doc_ids: list[str]
    forbidden_doc_ids: list[str]
    forbidden_text_fragments: list[str]
    retrieved_doc_ids: list[str]
    correct_hit_at_5: bool
    hard_negative_top1: bool
    forbidden_hit: bool
    forbidden_text_hit: bool  # chunk-level: any retrieved chunk text contains any forbidden fragment
    # Pre-rerank metrics — same logic but applied to the chunk order BEFORE
    # llm_rerank() ran. Lets us measure rerank's actual lift in isolation.
    # The diff (post − pre) is rerank "lift": positive = rerank helped,
    # negative = rerank hurt, zero = rerank had no effect on top-5 membership.
    pre_rerank_retrieved_doc_ids: list[str] = field(default_factory=list)
    correct_hit_at_5_pre_rerank: bool = False
    hard_negative_top1_pre_rerank: bool = False
    error: str | None = None
    retrieval_latency_ms: int = 0


def load_rows():
    if not EVAL_FILE.exists():
        sys.exit(f"ERROR: {EVAL_FILE} not found")
    rows = []
    with EVAL_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_row(row: dict, retrieve_context, warm_cache: bool = True) -> RowResult:
    """Evaluate one row.

    warm_cache=True (default): for multi-turn rows, replay each prior user-turn
    query through retrieve_context with a SHARED session_id BEFORE running the
    target query. This populates the session-level doc cache as in production,
    so cache_action paths (preserved_by_*, invalidated_by_*) fire faithfully.
    The recorded prior-turn assistant content is used verbatim as
    conversation_history at each replay step; we never regenerate.

    warm_cache=False: cold-cache mode. Run only the target query with
    conversation_history populated from prior_turns. Tests the contextualizer
    + retrieval but NOT cache anchoring. Faster + cheaper.

    For single-turn rows (no prior_turns), warm and cold modes are identical.
    """
    session_id = f"eval-{uuid.uuid4()}"
    prior_turns = list(row.get("prior_turns") or [])

    # Build [user, assistant] pairs from prior_turns so we can replay turn-by-turn.
    turn_pairs: list[tuple[str, str]] = []
    pending_user = None
    for t in prior_turns:
        if t.get("role") == "user":
            pending_user = t.get("content") or ""
        elif t.get("role") == "assistant" and pending_user is not None:
            turn_pairs.append((pending_user, t.get("content") or ""))
            pending_user = None
    # If there's a trailing user turn without an assistant response, drop it
    # (the row's `query` field is the *current* user turn, so a dangling user
    # turn in prior_turns would be malformed — and we'd duplicate the query).

    # Warm-cache replay — feed prior turns through retrieval to populate cache.
    if warm_cache and turn_pairs:
        cumulative_history: list[dict] = []
        for user_msg, asst_msg in turn_pairs:
            try:
                retrieve_context(
                    ta_id=row["ta_id"],
                    query=user_msg,
                    top_k=8,
                    conversation_history=list(cumulative_history),
                    session_id=session_id,
                    course_name="Introduction to Data Analysis and Econometrics",
                )
            except Exception:
                pass  # cache may be partially populated; continue to target query
            cumulative_history.append({"role": "user", "content": user_msg})
            cumulative_history.append({"role": "assistant", "content": asst_msg})

    # Target-query conversation history is the full prior_turns.
    conversation_history = list(prior_turns)

    t0 = time.time()
    pre_rerank_retrieved = []
    try:
        chunks, diagnostics = retrieve_context(
            ta_id=row["ta_id"],
            query=row["query"],
            top_k=8,
            conversation_history=conversation_history,
            session_id=session_id,
            course_name="Introduction to Data Analysis and Econometrics",
        )
        latency = int((time.time() - t0) * 1000)
        retrieved = [c.get("file_name", "") for c in (chunks or [])][:8]
        retrieved_top5 = retrieved[:5]
        retrieved_chunk_texts = [c.get("text", "") for c in (chunks or [])][:8]
        # Pre-rerank ordering — extracted from the diagnostics dict so we can
        # measure rerank lift in isolation (post − pre).
        pre_candidates = diagnostics.get("pre_rerank_candidates") or []
        pre_rerank_retrieved = [c.get("file", "") for c in pre_candidates][:8]
        error = None
    except Exception as e:
        latency = int((time.time() - t0) * 1000)
        retrieved = []
        retrieved_top5 = []
        retrieved_chunk_texts = []
        pre_rerank_retrieved = []
        error = f"{type(e).__name__}: {e}"

    correct_set = set(row["correct_doc_ids"])
    hard_neg_set = set(row.get("hard_negative_doc_ids") or [])
    forbidden_set = set(row.get("forbidden_doc_ids") or [])
    forbidden_fragments = row.get("forbidden_text_fragments") or []
    pre_rerank_top5 = pre_rerank_retrieved[:5]

    # Case-insensitive substring match across all retrieved chunk texts.
    # Used to detect G (intra-doc chunk-routing) failures — doc-level hit
    # would falsely register as a pass when chunks come from the wrong section.
    fragments_lower = [f.lower() for f in forbidden_fragments if f]
    forbidden_text_hit = bool(fragments_lower) and any(
        any(frag in (text or "").lower() for frag in fragments_lower)
        for text in retrieved_chunk_texts
    )

    return RowResult(
        row_id=row["row_id"],
        failure_type=row["failure_type_target"],
        source=row["source"],
        ta_id=row.get("ta_id", ""),
        not_in_corpus=bool(row.get("not_in_corpus")),
        correct_doc_ids=row["correct_doc_ids"],
        hard_negative_doc_ids=row.get("hard_negative_doc_ids") or [],
        forbidden_doc_ids=row.get("forbidden_doc_ids") or [],
        forbidden_text_fragments=forbidden_fragments,
        retrieved_doc_ids=retrieved,
        correct_hit_at_5=any(d in correct_set for d in retrieved_top5),
        hard_negative_top1=(retrieved[0] in hard_neg_set) if retrieved else False,
        forbidden_hit=any(d in forbidden_set for d in retrieved),
        forbidden_text_hit=forbidden_text_hit,
        pre_rerank_retrieved_doc_ids=pre_rerank_retrieved,
        correct_hit_at_5_pre_rerank=any(d in correct_set for d in pre_rerank_top5),
        hard_negative_top1_pre_rerank=(pre_rerank_retrieved[0] in hard_neg_set) if pre_rerank_retrieved else False,
        error=error,
        retrieval_latency_ms=latency,
    )


BUCKET_KEYS = ["A", "B", "C", "D", "E", "F1", "F2", "G1", "G2", "working"]


def aggregate(results: list[RowResult]) -> dict:
    """Group by failure_type and compute per-bucket metrics.

    Buckets:
      - In-corpus rows by failure_type (A, B, C, D, E, F1, F2, G1, G2, working)
      - not_in_corpus rows (reported separately — retrieval cannot pass)
    """
    in_corpus = [r for r in results if not r.not_in_corpus]
    not_in_corpus = [r for r in results if r.not_in_corpus]

    buckets: dict[str, list[RowResult]] = defaultdict(list)
    for r in in_corpus:
        key = r.failure_type or "working"
        buckets[key].append(r)

    summary = {}
    for key in BUCKET_KEYS:
        bucket = buckets.get(key, [])
        if not bucket:
            summary[key] = None
            continue
        n = len(bucket)
        post = sum(r.correct_hit_at_5 for r in bucket) / n
        pre = sum(r.correct_hit_at_5_pre_rerank for r in bucket) / n
        summary[key] = {
            "n": n,
            "correct_hit_at_5_rate": post,
            "correct_hit_at_5_pre_rerank_rate": pre,
            "rerank_lift": post - pre,  # positive = rerank helped; negative = rerank hurt
            "hard_negative_top1_rate": sum(r.hard_negative_top1 for r in bucket) / n,
            "forbidden_hit_rate": sum(r.forbidden_hit for r in bucket) / n,
            "forbidden_text_hit_rate": sum(r.forbidden_text_hit for r in bucket) / n,
            "error_count": sum(1 for r in bucket if r.error),
            "avg_latency_ms": sum(r.retrieval_latency_ms for r in bucket) / n,
        }
    summary["not_in_corpus"] = {
        "n": len(not_in_corpus),
        "retrieved_anything_relevant": sum(
            any(d in set(r.correct_doc_ids) for d in r.retrieved_doc_ids) for r in not_in_corpus
        ),
        "forbidden_hit_rate": sum(r.forbidden_hit for r in not_in_corpus) / max(len(not_in_corpus), 1),
        "note": "These rows can't pass at the retrieval layer (correct doc not verified in corpus). Reported for tracking only.",
    }
    summary["__overall__"] = {
        "total_rows": len(results),
        "in_corpus": len(in_corpus),
        "not_in_corpus": len(not_in_corpus),
        "errors": sum(1 for r in results if r.error),
        "distinct_tas": sorted({r.ta_id for r in results if r.ta_id}),
    }
    return summary


def format_scorecard(summary: dict, label: str = "Retrieval scorecard") -> str:
    lines = [f"# {label}", ""]
    overall = summary["__overall__"]
    lines.append(f"**Total rows:** {overall['total_rows']} ({overall['in_corpus']} in-corpus + "
                 f"{overall['not_in_corpus']} not-in-corpus). **Errors:** {overall['errors']}.")
    distinct_tas = overall.get("distinct_tas") or []
    if len(distinct_tas) > 1:
        lines.append(
            f"**TAs in this run:** {len(distinct_tas)} — `{'`, `'.join(distinct_tas)}`. "
            f"Cross-TA aggregate scoring; re-run with `--ta-id <id>` to scope to one TA."
        )
    elif len(distinct_tas) == 1:
        lines.append(f"**TA:** `{distinct_tas[0]}` (filtered to one TA — `--ta-id` flag in use OR file only contains rows for this TA).")
    lines.append("")
    lines.append("| Failure type | n | hit@5 pre→post (lift) | hard_neg_top1 | forbidden_hit | forbidden_text_hit | avg_latency_ms | errors |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    label_map = {
        "A": "A (Lab vs PS)",
        "B": "B (Roman numeral siblings, cross-doc)",
        "C": "C (lookalike-unrelated)",
        "D": "D (problem-vs-solutions)",
        "E": "E (cache anchoring)",
        "F1": "F1 (explicit conceptual switch)",
        "F2": "F2 (explicit document switch)",
        "G1": "G1 (intra-doc section confusion)",
        "G2": "G2 (intra-doc Roman/numeric sibling)",
        "working": "working cases",
    }
    for key in BUCKET_KEYS:
        b = summary.get(key)
        if b is None:
            lines.append(f"| {label_map[key]} | 0 | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {label_map[key]} | {b['n']} | "
            f"{b['correct_hit_at_5_pre_rerank_rate']:.0%}→{b['correct_hit_at_5_rate']:.0%} "
            f"({b['rerank_lift']:+.0%}) | "
            f"{b['hard_negative_top1_rate']:.0%} | {b['forbidden_hit_rate']:.0%} | "
            f"{b['forbidden_text_hit_rate']:.0%} | "
            f"{b['avg_latency_ms']:.0f} | {b['error_count']} |"
        )
    nic = summary["not_in_corpus"]
    lines.append("")
    lines.append(f"**Not-in-corpus bucket:** {nic['n']} rows. Retrieved a labeled-correct doc on "
                 f"{nic['retrieved_anything_relevant']}/{nic['n']} (expected 0 — these docs are unverified in the corpus). "
                 f"Forbidden-hit rate: {nic['forbidden_hit_rate']:.0%}.")
    lines.append("")
    lines.append("## Metric definitions")
    lines.append("- **hit@5 pre→post (lift)** — pre-rerank hit-rate → post-rerank hit-rate, with the rerank's contribution as a `(±X%)` delta. Positive = rerank moved correct chunks into top-5 that weren't there before. Negative = rerank pushed correct chunks out. Zero = rerank didn't affect top-5 membership.")
    lines.append("- **correct_hit@5** — fraction of rows where at least one `correct_doc_ids` appeared in retrieved top-5 (post-rerank).")
    lines.append("- **hard_negative_top1** — fraction of rows where the retrieved top-1 doc matched a known hard negative "
                 "(i.e., the current bad retrieval pattern fired).")
    lines.append("- **forbidden_hit** — fraction of rows where ANY retrieved doc was on the forbidden list "
                 "(e.g., solutions doc returned when student is solving). Lower is better; ideal = 0%.")
    lines.append("- **forbidden_text_hit** — fraction of rows where ANY retrieved chunk's text contained a forbidden substring "
                 "(e.g., \"Part II\" when the row tests Part I retrieval). Detects G failures — right doc, wrong section/part. "
                 "Lower is better; ideal = 0%.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true",
                        help="Write scorecard to baseline_scorecard_pre_refinement.md (use against the CURRENT retriever).")
    parser.add_argument("--out", type=str, default=None,
                        help="Write scorecard to a specific path (overrides --baseline default).")
    parser.add_argument("--row-id", type=str, default=None,
                        help="Run only the row with this row_id; useful for debugging a single case.")
    parser.add_argument("--ta-id", type=str, default=None,
                        help="Filter eval rows to those whose ta_id matches. Without this flag the harness "
                             "runs every row across every TA in the eval file (cross-TA mode).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N rows. Useful for smoke-testing the harness.")
    parser.add_argument("--cold-cache", action="store_true",
                        help="Disable warm-cache replay. Default is warm-cache: for multi-turn rows, "
                             "prior user-turn queries replay through retrieve_context with a shared "
                             "session_id BEFORE the target query, so session cache populates as in prod. "
                             "Without this faithful replay, cache-anchoring failure modes (E, F1, F2) "
                             "may not reproduce. Pass --cold-cache to revert to cheaper single-call "
                             "evaluation (~3x faster on multi-turn rows; useful for quick smoke tests).")
    args = parser.parse_args()

    # Set up Flask app context for DB queries.
    try:
        from app import app
        from src.retriever import retrieve_context
    except Exception as e:
        sys.exit(f"ERROR: failed to import Flask app or retriever: {type(e).__name__}: {e}\n"
                 f"Make sure DOTENV_PATH is set and dependencies are installed.")

    rows = load_rows()
    if args.row_id:
        rows = [r for r in rows if r["row_id"] == args.row_id]
        if not rows:
            sys.exit(f"ERROR: row_id {args.row_id!r} not found")
    if args.ta_id:
        rows_before = len(rows)
        rows = [r for r in rows if r.get("ta_id") == args.ta_id]
        if not rows:
            sys.exit(f"ERROR: no eval rows match ta_id={args.ta_id!r} (had {rows_before} rows before filter)")
    if args.limit:
        rows = rows[: args.limit]

    warm = not args.cold_cache
    cache_mode = "warm-cache (prior turns replayed)" if warm else "cold-cache (target query only)"
    print(f"Running {len(rows)} eval rows against the current retriever... [{cache_mode}]", flush=True)

    results: list[RowResult] = []
    with app.app_context():
        for i, row in enumerate(rows, 1):
            res = evaluate_row(row, retrieve_context, warm_cache=warm)
            results.append(res)
            status = "ERR" if res.error else ("HIT" if res.correct_hit_at_5 else "MISS")
            print(f"  [{i:3}/{len(rows)}] {status:4} {res.row_id:55} "
                  f"({res.retrieval_latency_ms}ms) "
                  f"top1={res.retrieved_doc_ids[0] if res.retrieved_doc_ids else '<none>'}",
                  flush=True)

    summary = aggregate(results)
    label = "Baseline (CURRENT retriever)" if args.baseline else "Retrieval scorecard"
    scorecard = format_scorecard(summary, label=label)
    print()
    print(scorecard)

    if args.baseline and not args.out:
        BASELINE_OUT.write_text(scorecard + "\n")
        print(f"\nBaseline written to {BASELINE_OUT}")
    elif args.out:
        Path(args.out).write_text(scorecard + "\n")
        print(f"\nScorecard written to {args.out}")

    return 1 if summary["__overall__"]["errors"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
