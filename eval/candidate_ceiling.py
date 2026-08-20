"""
Candidate ceiling: can the correct document even reach the reranker?

Why this exists
---------------
The fair-fight vendor comparison put gpt-5.2 and cohere rerank-v4.0-pro at the
SAME 58% on retrieving the labelled-correct document. Two readings fit that:
the rerankers are equivalent, or both are pressed against the same upstream wall
and the comparison measured candidate generation rather than reranking.

Those lead to opposite next moves, so they have to be separated before any
further reranker work is worth doing.

The funnel has TWO gates, applied in sequence
---------------------------------------------
  1. hybrid_doc_search (retriever.py:1397) returns at most Config.STAGE_1_TOP_K_DOCS
     document ids -- FIVE by default -- or exactly one when the direct-match
     short-circuit fires.
  2. The chunk search is then
        filter(document_id IN those docs).order_by(cosine).limit(INITIAL_RETRIEVAL_K)
     (retriever.py:2967, :3020), so the reranker sees 20 chunks drawn only from
     those <=5 documents. If the five hold 400 chunks between them, 380 are
     discarded before the reranker is called at all.

A document can clear gate 1 and still fail gate 2: shortlisted, but every one of
its chunks ranks below 20th on cosine among the pooled candidates. With a
900-chunk textbook and a 12-chunk problem set in the same shortlist, that is not
a hypothetical.

The two gates fail for different reasons and have different fixes -- gate 1 is a
routing failure, gate 2 is crowding -- which is why they are measured apart.

What is reported
----------------
Four nested rates, each a strict subset of the one above:

  L0  correct doc exists in the corpus at all      -> a miss is a BAD LABEL
  L1  correct doc in fused_doc_ids (<=5 shortlist) -> THE CEILING, unrecoverable
  L2  correct doc has >=1 chunk in the ~20 pool    -> shortlisted but out-chunked
  L3  correct doc survives into the reranked top-8 -> the reranker dropped it

Only the L1 -> L3 span is something a reranker can influence. If L1 sits near
58%, the vendor comparison was measuring the funnel and the Cohere decision
reduces to pure latency.

No retrieval code is instrumented. Everything read here already exists in the
diagnostics dict: pre_rerank_candidates (retriever.py:3236) and hybrid_stage_1
(:2965), which carries fused_doc_ids plus the per-side bm25/dense/filename
rankings and the short-circuit record.

Usage
-----
  DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/candidate_ceiling.py
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# (stage_1_top_k_docs, initial_retrieval_k). First entry MUST be the live
# production values -- it is the row the positive control is checked against,
# and every other row is read as a delta from it.
SWEEP = [
    (5, 20),   # current
    (10, 20),  # wider document shortlist -- does gate 1 recover misses?
    (20, 20),  # wider still -- where does L1 saturate?
    (5, 50),   # wider chunk pool at today's shortlist -- gate 2 in isolation
]


def load_labelled_openers(limit: int | None = None) -> list[dict]:
    """
    Session-opening eval rows that carry a label.

    Openers only: rerank fires on FRESH retrieval, and a continuation turn served
    from the session cache never builds a candidate pool at all. Concept probes
    are excluded -- they have no correct_doc_ids, so there is nothing to locate.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from run_eval import load_rows, group_into_sessions

    openers = [s[0] for s in group_into_sessions(load_rows())]
    rows = [r for r in openers if r.get("correct_doc_ids")]
    return rows[:limit] if limit else rows


def build_filename_index() -> dict:
    """
    (ta_id, file_name) -> document_id.

    correct_doc_ids are stored as filename STEMS ('econ117_pset02_2025B_with_table')
    while fused_doc_ids are integer Document.id, so the two cannot be compared
    without this join. It is built from document_chunks rather than documents
    because that is the field the labels actually match -- documents.original_filename
    still carries the '.pdf' extension, and chunk file_name is also what
    pre_rerank_candidates reports, so one map serves both gates.

    A document with zero chunks is absent here by construction. That is correct:
    an unindexed document is unreachable, and showing up as an L0 miss is the
    honest result.
    """
    from models import db, DocumentChunk

    rows = db.session.query(
        DocumentChunk.ta_id, DocumentChunk.file_name, DocumentChunk.document_id
    ).distinct().all()
    return {(r.ta_id, r.file_name): r.document_id for r in rows}


def probe_one(retrieve_context, row: dict, fname_index: dict) -> dict:
    """One query through the real retrieval path; return the four gate outcomes."""
    from models import db, ChatSession

    ta_id = row["ta_id"]
    labels = row["correct_doc_ids"]
    # L0: which labels resolve to a real indexed document?
    correct_ids = {fname_index[(ta_id, f)] for f in labels if (ta_id, f) in fname_index}
    unresolved = [f for f in labels if (ta_id, f) not in fname_index]

    session_id = "rag" + uuid.uuid4().hex[:29]
    db.session.add(ChatSession(id=session_id, ta_id=ta_id, user_id=None))
    db.session.commit()

    t0 = time.time()
    try:
        chunks, diag = retrieve_context(
            ta_id=ta_id, query=row["query"], top_k=8,
            conversation_history=[], session_id=session_id, course_name="",
        )
        err = None
    except Exception as e:
        chunks, diag, err = [], {}, f"{type(e).__name__}: {e}"
    ms = int((time.time() - t0) * 1000)

    try:
        db.session.rollback()
        stale = ChatSession.query.get(session_id)
        if stale:
            db.session.delete(stale)
            db.session.commit()
    except Exception:
        db.session.rollback()

    stage1 = diag.get("hybrid_stage_1") or {}
    shortlist = list(stage1.get("fused_doc_ids") or [])

    # Gate 2 and the final result are reported by FILENAME: pre_rerank_candidates
    # carries 'file', not document_id, and the reranked chunks likewise.
    correct_names = {f for f in labels if (ta_id, f) in fname_index}
    pool_files = {c.get("file") for c in (diag.get("pre_rerank_candidates") or [])}
    final_files = {c.get("file_name") for c in (chunks or [])}

    # Where did the correct doc sit on each fusion input? This is what sizes a
    # shortlist widening: rank 6 on BM25 is recoverable at STAGE_1_TOP_K_DOCS=10,
    # absent from every side is not.
    # Rankings are [(doc_id, zero_based_rank)] and are NOT populated when the
    # short-circuit fires -- it returns before fusion runs (retriever.py:1670).
    def side_rank(side: str):
        ranking = dict((d, r) for d, r in (stage1.get(side) or []))
        hits = [ranking[d] for d in correct_ids if d in ranking]
        return min(hits) if hits else None

    return {
        "row_id": row["row_id"],
        "ta_id": ta_id,
        "query": row["query"],
        "labels": labels,
        "unresolved_labels": unresolved,
        "error": err,
        "ms": ms,
        "retrieval_method": diag.get("retrieval_method"),
        "short_circuit": stage1.get("short_circuit"),
        "shortlist": shortlist,
        "shortlist_size": len(shortlist),
        "pool_size": len(diag.get("pre_rerank_candidates") or []),
        "pool_doc_count": len({c.get("file") for c in (diag.get("pre_rerank_candidates") or [])}),
        "L0": bool(correct_ids),
        "L1": bool(correct_ids & set(shortlist)),
        "L2": bool(correct_names & pool_files),
        "L3": bool(correct_names & final_files),
        "rank_bm25": side_rank("bm25_ranking"),
        "rank_dense": side_rank("dense_ranking"),
        "rank_filename": side_rank("filename_ranking"),
        "reranked": bool(diag.get("rerank_applied")),
    }


def rate(rows: list, key: str) -> tuple:
    """Rate over rows with a resolvable label -- L0 misses are label rot, not misses."""
    scoped = [r for r in rows if r["L0"]]
    hit = sum(1 for r in scoped if r[key])
    return hit, len(scoped), (hit / len(scoped) if scoped else 0.0)


def fired(row: dict) -> bool:
    """
    Did the direct-match short-circuit actually fire?

    NOT a presence check: hybrid_doc_search records a short_circuit dict on the
    NON-firing path too, carrying {'fired': False, ...} plus the reason it
    declined (retriever.py:1676). Testing the dict for truthiness counts those
    declines as fires and roughly doubles the apparent rate -- which is exactly
    the mistake this function exists to prevent.
    """
    sc = row.get("short_circuit")
    return bool(sc and sc.get("fired"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-sweep", action="store_true",
                    help="Run only the live configuration, skipping the funnel sweep.")
    ap.add_argument("--out", default=str(Path(__file__).parent / "candidate_ceiling.md"))
    ap.add_argument("--from-json", metavar="PATH",
                    help="Rebuild the report from a previous run's .json instead of "
                         "re-running. The per-row records hold everything the report "
                         "derives, so a reporting fix costs nothing to reapply.")
    args = ap.parse_args()

    if args.from_json:
        report(json.loads(Path(args.from_json).read_text()), Path(args.out),
               sweep=not args.no_sweep)
        return

    from app import app
    import src.retriever as R
    from src.retriever import retrieve_context
    from config import Config

    # Contextualizer off for comparability with the vendor comparison runs; Cohere
    # as the reranker only because it is ~1s instead of ~13s. The reranker CANNOT
    # affect L0-L2: the pool is fully built before rerank() is called
    # (retriever.py:3249). It affects L3 only.
    Config.CONTEXTUALIZER_ENABLED = False
    Config.RERANKER_VENDOR = "cohere"
    Config.COHERE_RERANK_MODEL = "rerank-v4.0-pro"

    configs = SWEEP[:1] if args.no_sweep else SWEEP
    all_results = {}

    with app.app_context():
        fname_index = build_filename_index()
        rows = load_labelled_openers(args.limit)

        unresolved = [(r["row_id"], f) for r in rows for f in r["correct_doc_ids"]
                      if (r["ta_id"], f) not in fname_index]
        print(f"{len(rows)} labelled openers | filename index: {len(fname_index)} entries")
        if unresolved:
            print(f"[warn] {len(unresolved)} labels resolve to no indexed document "
                  f"(counted as L0 misses, excluded from L1-L3):")
            for rid, f in unresolved[:10]:
                print(f"         {rid}: {f!r}")
        print()

        for s1k, ik in configs:
            Config.STAGE_1_TOP_K_DOCS = s1k
            R.INITIAL_RETRIEVAL_K = ik  # module global, read at call time (:2364)
            label = f"docs={s1k},chunks={ik}"
            print(f"--- {label} ---", flush=True)
            out = []
            for i, row in enumerate(rows, 1):
                res = probe_one(retrieve_context, row, fname_index)
                out.append(res)
                flags = "".join(g if res[g] else "." for g in ("L0", "L1", "L2", "L3"))
                print(f"  {i:3}/{len(rows)} {res['row_id']:48} "
                      f"{flags.replace('L','')} shortlist={res['shortlist_size']} "
                      f"pool={res['pool_size']}/{res['pool_doc_count']}docs {res['ms']}ms",
                      flush=True)
            all_results[label] = out
            print(flush=True)

    report(all_results, Path(args.out), sweep=not args.no_sweep)


def report(all_results: dict, out: Path, sweep: bool = True) -> None:
    live = all_results[f"docs={SWEEP[0][0]},chunks={SWEEP[0][1]}"]

    print("=" * 70)
    print("CANDIDATE CEILING — live configuration")
    print("=" * 70)
    for gate, desc in [("L0", "label resolves to an indexed doc"),
                       ("L1", "correct doc in the <=5 shortlist   <-- THE CEILING"),
                       ("L2", "correct doc has a chunk in the pool"),
                       ("L3", "correct doc survives the rerank")]:
        if gate == "L0":
            h = sum(1 for r in live if r["L0"])
            print(f"  L0  {h}/{len(live)} = {h/len(live):.0%}  {desc}")
        else:
            h, n, p = rate(live, gate)
            print(f"  {gate}  {h}/{n} = {p:.0%}  {desc}")

    # The single most important split in this measurement. The short-circuit
    # bypasses fusion and returns ONE document, so on those queries the reranker
    # picks 8 chunks from a document that is already right or already wrong --
    # it has no discretion, and widening the shortlist cannot reach them either.
    ok = [r for r in live if r["L0"]]
    sc, fus = [r for r in ok if fired(r)], [r for r in ok if not fired(r)]
    print(f"\n  ROUTING PATH (of {len(ok)} rows with a resolvable label)")
    for name, grp in (("short-circuit fired", sc), ("fusion ran", fus)):
        if not grp:
            continue
        cells = "  ".join(f"{g} {rate(grp, g)[2]:.0%}" for g in ("L1", "L2", "L3"))
        print(f"    {name:22} n={len(grp):2}   {cells}")
    print(f"\n  mean shortlist size {sum(r['shortlist_size'] for r in live)/len(live):.1f} docs | "
          f"mean pool {sum(r['pool_size'] for r in live)/len(live):.1f} chunks from "
          f"{sum(r['pool_doc_count'] for r in live)/len(live):.1f} docs")

    # Nesting is EXPECTED but not guaranteed: when the candidate-doc chunk search
    # returns nothing, retrieval falls back to an unfiltered search (retriever.py:3023),
    # so a doc can enter the pool without ever making the shortlist. Report those
    # rather than asserting -- the violation is itself a finding.
    viol = [r for r in live if (r["L2"] and not r["L1"]) or (r["L3"] and not r["L2"])]
    if viol:
        print(f"\n  [note] {len(viol)} rows break L1>=L2>=L3 — check retrieval_method "
              f"for an unfiltered fallback:")
        for r in viol[:8]:
            print(f"         {r['row_id']:48} {r['retrieval_method']}")

    if sweep:
        print("\n" + "=" * 70)
        print("FUNNEL SWEEP — would a wider funnel recover the misses?")
        print("=" * 70)
        print(f"  {'config':22} {'L1 (shortlist)':>18} {'L2 (pool)':>16} {'L3 (final)':>16}"
              f"{'L1 fusion-only':>18}")
        for label, res in all_results.items():
            h1, n1, p1 = rate(res, "L1")
            h2, n2, p2 = rate(res, "L2")
            h3, n3, p3 = rate(res, "L3")
            # Short-circuit rows are inert to shortlist width by construction, so
            # they dilute the sweep. The fusion-only column is the honest read of
            # what widening buys.
            f1 = rate([r for r in res if not fired(r)], "L1")
            print(f"  {label:22} {f'{h1}/{n1} = {p1:.0%}':>18} "
                  f"{f'{h2}/{n2} = {p2:.0%}':>16} {f'{h3}/{n3} = {p3:.0%}':>16}"
                  f"{f'{f1[0]}/{f1[1]} = {f1[2]:.0%}':>18}")

    misses = [r for r in live if r["L0"] and not r["L1"]]
    if misses:
        print("\n" + "=" * 70)
        print(f"L1 MISSES ({len(misses)}) — where the correct doc actually ranked")
        print("=" * 70)
        print(f"  {'row':46} {'bm25':>6} {'dense':>6} {'fname':>6}  short-circuit")
        for r in misses:
            f = lambda v: ("-" if v is None else str(v))
            print(f"  {r['row_id']:46} {f(r['rank_bm25']):>6} {f(r['rank_dense']):>6} "
                  f"{f(r['rank_filename']):>6}  {'YES' if fired(r) else ''}")
        print("\n  ranks are 0-based within each side's own candidate pool; '-' means the")
        print("  correct doc was absent from that side entirely, or the short-circuit fired")
        print("  before fusion ran, so no per-side ranking was computed.")

    with out.open("w") as f:
        f.write("# Candidate ceiling — can the correct document reach the reranker?\n\n")
        f.write(f"_{datetime.utcnow().isoformat(timespec='seconds')}Z — "
                f"{len(live)} labelled session openers, contextualizer off_\n\n")
        f.write("Four nested gates. Only the L1→L3 span is something a reranker can "
                "influence — everything lost at L1 is unrecoverable downstream, whichever "
                "vendor is chosen.\n\n")
        f.write("| gate | question | rate |\n|---|---|---|\n")
        h0 = sum(1 for r in live if r["L0"])
        f.write(f"| L0 | label resolves to an indexed document | {h0}/{len(live)} = "
                f"{h0/len(live):.0%} |\n")
        for gate, desc in [("L1", "correct doc in the ≤5 document shortlist — **the ceiling**"),
                           ("L2", "correct doc has ≥1 chunk in the ~20-chunk pool"),
                           ("L3", "correct doc survives into the reranked top-8")]:
            h, n, p = rate(live, gate)
            f.write(f"| {gate} | {desc} | {h}/{n} = **{p:.0%}** |\n")
        f.write("\n## Routing path — the split that matters\n\n")
        f.write("The direct-match short-circuit bypasses fusion and returns exactly ONE "
                "document, so on those queries the reranker chooses 8 chunks from a document "
                "that is already right or already wrong, and a wider shortlist cannot reach "
                "them at all.\n\n")
        f.write("| path | n | L1 | L2 | L3 |\n|---|---|---|---|---|\n")
        for name, grp in (("short-circuit fired", sc), ("fusion ran", fus)):
            if not grp:
                continue
            cells = " | ".join(f"{rate(grp, g)[2]:.0%}" for g in ("L1", "L2", "L3"))
            f.write(f"| {name} | {len(grp)} | {cells} |\n")
        f.write(f"\n- mean shortlist **{sum(r['shortlist_size'] for r in live)/len(live):.1f}** docs; "
                f"mean pool **{sum(r['pool_size'] for r in live)/len(live):.1f}** chunks from "
                f"**{sum(r['pool_doc_count'] for r in live)/len(live):.1f}** docs\n\n")
        if sweep:
            f.write("## Funnel sweep\n\nThe `L1 fusion-only` column excludes short-circuit rows, "
                    "which are inert to shortlist width by construction and otherwise dilute "
                    "the effect.\n\n| config | L1 | L2 | L3 | L1 fusion-only |\n|---|---|---|---|---|\n")
            for label, res in all_results.items():
                h1, n1, p1 = rate(res, "L1")
                h2, n2, p2 = rate(res, "L2")
                h3, n3, p3 = rate(res, "L3")
                fo = rate([r for r in res if not fired(r)], "L1")
                f.write(f"| {label} | {h1}/{n1} = {p1:.0%} | {h2}/{n2} = {p2:.0%} | "
                        f"{h3}/{n3} = {p3:.0%} | {fo[0]}/{fo[1]} = {fo[2]:.0%} |\n")
            f.write("\n")
        if misses:
            f.write(f"## L1 misses ({len(misses)}) — where the correct doc ranked\n\n")
            f.write("Ranks are 0-based within each fusion side's own pool. `-` means absent "
                    "from that side, or the short-circuit returned before fusion ran.\n\n")
            f.write("| row | bm25 | dense | filename | short-circuit |\n|---|---|---|---|---|\n")
            for r in misses:
                g = lambda v: ("-" if v is None else str(v))
                f.write(f"| {r['row_id']} | {g(r['rank_bm25'])} | {g(r['rank_dense'])} | "
                        f"{g(r['rank_filename'])} | {'YES' if fired(r) else ''} |\n")
        f.write("\n_Retrieval only: no reranker judgement affects L0–L2, which are fixed "
                "before `rerank()` is called._\n")

    out.with_suffix(".json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nWrote {out} and {out.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
