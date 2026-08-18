"""
Rerank agreement: how much does a reranker's ordering agree with a reference?

Why agreement rather than quality
---------------------------------
The Cohere swap is a LATENCY change. We are not trying to improve ranking — we
are trying to keep it while going faster. So the useful question is "is Cohere
equivalent to gpt-5.2?", not "is Cohere good?".

That reframing buys three things:
  - No labels needed, so it is not confined to the 92 labelled eval rows and
    inherits none of the label noise.
  - Far more signal: every position of every list contributes, versus one binary
    outcome per row.
  - The ship threshold stops being an arbitrary number. gpt-5.2 is itself
    nondeterministic, so run it twice and measure its SELF-agreement. If Cohere
    agrees with gpt-5.2 about as much as gpt-5.2 agrees with itself, the two are
    equivalent for practical purposes.

Known property, stated rather than hidden: agreement measures equivalence, not
quality. If gpt-5.2 ranks something badly and Cohere matches it, this scores as
success. For a latency-only swap that is exactly right — which is why the
pilot-replay pairwise blind grading stays as a separate backstop.

Metrics
-------
RBO (rank-biased overlap) is primary: top-weighted, so a 1<->2 swap counts for
much more than a 7<->8 swap, and it handles lists that do not contain the same
items. Both matter here — the natural "sum of rank differences" (Spearman's
footrule) scores all positions equally and breaks when the two lists are not
conjoint, which they often are not, since each reranker selects 8 from ~20
candidates.

overlap@k and top-1 agreement are reported alongside because they are immediately
interpretable in a way RBO is not.

Query set
---------
Rerank only fires on FRESH retrieval with more than top_k candidates: llm_rerank
returns early when len(chunks) <= top_k (retriever.py:807), and session-cache
hits skip reranking altogether. Multi-turn continuation turns are therefore
useless here. The set is the conceptual probes plus the session-OPENING rows of
the eval body. How many actually reranked is reported, not assumed.

Usage
-----
  # baseline: gpt-5.2 against itself (no Cohere key needed)
  DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/rerank_agreement.py --runs 2
"""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics as st
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_TA = "EgZ14pvqEYzfQRTM"


def chunk_key(c: dict) -> str:
    """
    Stable identity for a chunk across runs.

    Chunk dicts do not carry a uniform id on every retrieval path, so identity is
    a hash of the text plus the filename — robust regardless of which path built
    the list, and immune to row-id churn from re-indexing.
    """
    h = hashlib.sha1((c.get("text") or "").encode("utf-8")).hexdigest()[:16]
    return f"{c.get('file_name','?')}#{h}"


def rbo(list_a: list, list_b: list, p: float = 0.9) -> float:
    """
    Rank-biased overlap: top-weighted similarity of two ranked lists.

    p controls how sharply the top is weighted; 0.9 puts roughly 86% of the
    weight in the first 8 positions, which matches TOP_K_RERANK. Returns 1.0 for
    identical ordering, 0.0 for no shared items.

    Unlike rank correlation this does not require the lists to contain the same
    items, which matters because each reranker picks 8 from ~20 candidates and
    may select different sets.

    NORMALISATION: the RBO series only reaches 1.0 at infinite depth. Truncated at
    depth k the maximum attainable is (1 - p^k) — 0.570 for k=8, p=0.9 — so the
    raw sum must be divided by that to make "identical" score 1.0. Without this,
    identical lists and quite-different lists occupy a compressed range and the
    number is hard to read. Caught by the identical-list sanity check.
    """
    if not list_a or not list_b:
        return 0.0
    depth = max(len(list_a), len(list_b))
    seen_a, seen_b = set(), set()
    total = 0.0
    for d in range(1, depth + 1):
        if d <= len(list_a):
            seen_a.add(list_a[d - 1])
        if d <= len(list_b):
            seen_b.add(list_b[d - 1])
        overlap = len(seen_a & seen_b)
        total += (overlap / d) * (p ** (d - 1))
    max_attainable = 1.0 - (p ** depth)
    if max_attainable <= 0:
        return 0.0
    return ((1 - p) * total) / max_attainable


def overlap_at(list_a: list, list_b: list, k: int) -> float:
    a, b = set(list_a[:k]), set(list_b[:k])
    return len(a & b) / max(len(a), 1)


def load_queries(limit: int | None) -> list[dict]:
    """
    Fresh single-turn queries only — the ones where rerank actually runs.

    Session openers come from the eval body (rows with no prior_turns after
    grouping); the conceptual probes are single-turn by construction.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from run_eval import load_rows, group_into_sessions
    from concept_probe import QUERIES as CONCEPT_QUERIES

    rows = load_rows()
    openers = [s[0] for s in group_into_sessions(rows)]
    out = [{"source": "eval_opener", "ta_id": r["ta_id"], "query": r["query"],
            "row_id": r["row_id"]} for r in openers]
    out += [{"source": "concept_probe", "ta_id": DEFAULT_TA, "query": q,
             "row_id": f"concept_{i+1:02d}"} for i, q in enumerate(CONCEPT_QUERIES)]
    return out[:limit] if limit else out


def run_pass(queries: list[dict], retrieve_context, label: str) -> dict:
    """One full pass: returns {row_id: {ordered chunk keys, reranked flag, ms}}."""
    from models import db, ChatSession

    out = {}
    for i, q in enumerate(queries, 1):
        session_id = "rag" + uuid.uuid4().hex[:29]
        db.session.add(ChatSession(id=session_id, ta_id=q["ta_id"], user_id=None))
        db.session.commit()
        t0 = time.time()
        try:
            chunks, diag = retrieve_context(
                ta_id=q["ta_id"], query=q["query"], top_k=8,
                conversation_history=[], session_id=session_id, course_name="",
            )
        except Exception as e:
            print(f"    [warn] {q['row_id']}: {type(e).__name__}: {e}", flush=True)
            chunks, diag = [], {}
        ms = int((time.time() - t0) * 1000)

        out[q["row_id"]] = {
            "keys": [chunk_key(c) for c in (chunks or [])],
            "reranked": bool(diag.get("rerank_applied")),
            "method": diag.get("retrieval_method"),
            # Supplementary material performs its own LLM concept extraction and
            # appends chunks to the result, so rows where it fired are NOT a clean
            # reranker-only measurement even with the contextualizer disabled.
            "supp": bool(diag.get("supplementary_teaching_found")),
            "ms": ms,
        }
        print(f"  [{label}] {i:3}/{len(queries)} {q['row_id']:52} "
              f"rerank={'Y' if out[q['row_id']]['reranked'] else 'n'} {ms}ms", flush=True)

        try:
            db.session.rollback()
            stale = ChatSession.query.get(session_id)
            if stale:
                db.session.delete(stale)
                db.session.commit()
        except Exception:
            db.session.rollback()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=2, help="passes to make (2 = self-agreement)")
    ap.add_argument("--no-contextualizer", action="store_true",
                    help="Disable the contextualizer so effective_query is the raw query. The "
                         "contextualizer is an LLM that rewrites the query BEFORE retrieval, so a "
                         "different rewrite hands the reranker different candidates — variance a "
                         "reranker swap would not fix. Turning it off isolates the reranker as "
                         "the (near-)only remaining LLM in the path.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(Path(__file__).parent / "rerank_agreement_results.md"))
    args = ap.parse_args()

    from app import app
    from src.retriever import retrieve_context
    from config import Config

    if args.no_contextualizer:
        # Set on Config directly: config.py calls load_dotenv(override=True), so an
        # exported env var loses to the .env file.
        Config.CONTEXTUALIZER_ENABLED = False
        print("Contextualizer DISABLED — effective_query is the raw query", flush=True)

    queries = load_queries(args.limit)
    print(f"{len(queries)} queries x {args.runs} passes\n", flush=True)

    passes = []
    with app.app_context():
        for n in range(args.runs):
            passes.append(run_pass(queries, retrieve_context, f"pass {n+1}"))
            print(flush=True)

    # Only rows where rerank actually fired in EVERY pass are comparable — the
    # rest never exercised the component under test.
    comparable = [q["row_id"] for q in queries
                  if all(p.get(q["row_id"], {}).get("reranked") for p in passes)]
    reranked_any = sum(1 for q in queries
                       if any(p.get(q["row_id"], {}).get("reranked") for p in passes))

    results = []
    for rid in comparable:
        a, b = passes[0][rid]["keys"], passes[1][rid]["keys"]
        results.append({
            "row_id": rid,
            "rbo": rbo(a, b),
            "ov5": overlap_at(a, b, 5),
            "ov8": overlap_at(a, b, 8),
            "top1": 1.0 if (a and b and a[0] == b[0]) else 0.0,
            "identical": 1.0 if a[:8] == b[:8] else 0.0,
        })

    # Rows where the supplementary step fired in either pass are not a clean
    # reranker-only measurement — that step runs its own LLM concept extraction
    # and appends chunks to the result.
    clean = [r for r in results
             if not any(p.get(r["row_id"], {}).get("supp") for p in passes)]

    def m(k, rows=None):
        rows = results if rows is None else rows
        return st.mean([r[k] for r in rows]) if rows else 0.0

    print("=" * 62)
    print(f"queries: {len(queries)} | reranked in some pass: {reranked_any} | "
          f"comparable (reranked in all): {len(comparable)}")
    if args.no_contextualizer:
        print(f"contextualizer OFF | supplementary-free subset: {len(clean)}/{len(results)}")
    print("=" * 62)
    if results:
        print(f"  RBO (top-weighted)      : {m('rbo'):.3f}")
        print(f"  overlap@5               : {m('ov5'):.3f}")
        print(f"  overlap@8               : {m('ov8'):.3f}")
        print(f"  top-1 agreement         : {m('top1'):.3f}")
        print(f"  identical top-8 ordering: {m('identical'):.3f}")
        if clean and len(clean) != len(results):
            print(f"  --- supplementary-free subset (n={len(clean)}), cleanest isolation ---")
            print(f"  RBO                     : {m('rbo', clean):.3f}")
            print(f"  top-1 agreement         : {m('top1', clean):.3f}")
            print(f"  identical top-8 ordering: {m('identical', clean):.3f}")
    else:
        print("  no comparable rows — rerank never fired in all passes")

    out = Path(args.out)
    with out.open("w") as f:
        f.write("# Reranker self-agreement — gpt-5.2 vs itself\n\n")
        f.write(f"_{datetime.utcnow().isoformat(timespec='seconds')}Z — "
                f"{len(queries)} queries, {args.runs} passes_\n\n")
        f.write("Baseline for the Cohere swap. gpt-5.2 is nondeterministic, so this is the ")
        f.write("ceiling any replacement should be judged against: if Cohere agrees with ")
        f.write("gpt-5.2 about as much as gpt-5.2 agrees with itself, they are equivalent.\n\n")
        f.write(f"- queries attempted: **{len(queries)}**\n")
        f.write(f"- reranked in at least one pass: **{reranked_any}**\n")
        f.write(f"- comparable (reranked in every pass): **{len(comparable)}**\n\n")
        if results:
            f.write("| metric | value |\n|---|---|\n")
            f.write(f"| RBO (top-weighted) | **{m('rbo'):.3f}** |\n")
            f.write(f"| overlap@5 | {m('ov5'):.3f} |\n")
            f.write(f"| overlap@8 | {m('ov8'):.3f} |\n")
            f.write(f"| top-1 agreement | {m('top1'):.3f} |\n")
            f.write(f"| identical top-8 ordering | {m('identical'):.3f} |\n\n")
            worst = sorted(results, key=lambda r: r["rbo"])[:10]
            f.write("## Least stable queries\n\n| row | RBO | ov@8 | top-1 |\n|---|---|---|---|\n")
            for r in worst:
                f.write(f"| {r['row_id']} | {r['rbo']:.3f} | {r['ov8']:.2f} | "
                        f"{int(r['top1'])} |\n")
        f.write("\n_Agreement measures equivalence, not quality._\n")

    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(
        {"queries": queries, "passes": passes, "results": results}, indent=2))
    print(f"\nWrote {out} and {json_out.name}")


if __name__ == "__main__":
    main()
