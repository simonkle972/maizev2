"""
Conceptual-question probe: does book content displace the professor's material on
questions any undergraduate econometrics course would cover?

Why this exists
---------------
The 92-row eval body is overwhelmingly "help me with problem 8 from practice
problems 1" — questions about the professor's OWN artifacts, which a textbook
cannot answer. Books reached the top sources on 1 of 82 such rows. That shows
books do not intrude where they shouldn't; it says nothing about whether they
take over where they legitimately could.

These 18 questions are the other half. They are generic undergraduate
econometrics — naturally covered by both a lecture course and any of these
textbooks, so they sit in the overlap where competition actually happens, without
being sampled from either corpus (which would bias toward whichever source they
were drawn from).

Each question runs against two corpora:
  - combined   : the professor's material + the books (production shape)
  - books-only : the books alone (the control)

The control matters. Without it, "does the combined answer sound like the
textbook?" is an impression. With it, it is a comparison.

These are PROBES, not eval rows. There is no labelled correct document, so hit@5
is meaningless and adding them to maize_eval_v1.jsonl would drag the scorecard
down for no reason.

Usage
-----
  DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/concept_probe.py
  DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/concept_probe.py --no-sheet
"""
from __future__ import annotations
import argparse
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Combined = professor's material + books. Books-only = the control.
CORPORA = [
    ("combined", "EgZ14pvqEYzfQRTM"),
    ("books-only", "tLMAxWBRrycrJsWl"),
]

# Book documents in the COMBINED ta are everything above the pre-book snapshot
# boundary (68 docs, max id 82, captured before indexing). Without that boundary
# there is no way to tell book chunks from the professor's after the fact.
COMBINED_BOOK_DOC_ID_MIN = 83

QUERIES = [
    "What does OLS actually do — what is it minimising?",
    "What are the assumptions behind OLS and which ones matter most in practice?",
    "What is omitted variable bias and how do I know if I have it?",
    "What is heteroskedasticity and why is it a problem?",
    "When should I use robust standard errors instead of regular ones?",
    "What does endogeneity mean and why does it break OLS?",
    "How does an instrumental variable fix endogeneity? What makes a good instrument?",
    "What does R² actually tell me, and why shouldn't I just maximise it?",
    "How do I interpret a p-value on a regression coefficient?",
    "What is multicollinearity and does it always need fixing?",
    "How do I interpret a dummy variable coefficient?",
    "What does an interaction term mean in a regression?",
    "How do I interpret coefficients when the dependent variable is logged?",
    "What is the difference between correlation and causation in a regression context?",
    "What are fixed effects and when do I need them?",
    "How does difference-in-differences identify a causal effect?",
    "What is the difference between a standard error and a standard deviation?",
    "How do I read a confidence interval on a coefficient?",
]


def run_one(query: str, ta_id: str, retrieve_context, log_to_sheet: bool) -> dict:
    """Retrieve + generate for one query against one corpus. Single fresh turn."""
    from models import db, ChatSession, TeachingAssistant, Document
    from src.response_generator import generate_response
    from config import Config

    ta = TeachingAssistant.query.get(ta_id)
    session_id = "probe" + uuid.uuid4().hex[:27]
    db.session.add(ChatSession(id=session_id, ta_id=ta_id, user_id=None))
    db.session.commit()

    t0 = time.time()
    try:
        chunks, diag = retrieve_context(
            ta_id=ta_id, query=query, top_k=8,
            conversation_history=[], session_id=session_id,
            course_name=(ta.course_name if ta else ""),
        )
    except Exception as e:
        chunks, diag = [], {}
        print(f"    [warn] retrieval failed: {type(e).__name__}: {e}", flush=True)
    retrieval_ms = int((time.time() - t0) * 1000)

    # Which retrieved chunks came from a book? Resolved by document id against the
    # pre-book boundary; on the books-only corpus every chunk is a book by
    # construction, so the ratio there carries no information and is skipped.
    book_files, all_files = set(), []
    if ta_id == "EgZ14pvqEYzfQRTM":
        names = [c.get("file_name") for c in chunks if c.get("file_name")]
        all_files = names
        if names:
            rows = (Document.query
                    .filter(Document.ta_id == ta_id)
                    .with_entities(Document.id, Document.display_name, Document.original_filename)
                    .all())
            book_names = {(r.display_name or r.original_filename)
                          for r in rows if r.id >= COMBINED_BOOK_DOC_ID_MIN}
            book_files = {n for n in names if n in book_names}
    else:
        all_files = [c.get("file_name") for c in chunks if c.get("file_name")]

    t1 = time.time()
    answer = ""
    try:
        primary = [c for c in chunks if c.get("retrieval_role") != "teaching_material"]
        teaching = [c for c in chunks if c.get("retrieval_role") == "teaching_material"]
        parts = [f"[From: {c.get('file_name','')}]\n{c.get('text','')}" for c in primary]
        if teaching:
            parts.append("[RELEVANT TEACHING MATERIAL FROM COURSE LECTURES]")
            parts.extend(f"[From: {c.get('file_name','')}]\n{c.get('text','')}" for c in teaching)
        answer = generate_response(
            query=query,
            context="\n\n---\n\n".join(parts),
            system_prompt=(ta.system_prompt if ta else ""),
            conversation_history="",
            course_name=(ta.course_name if ta else ""),
            hybrid_mode=diag.get("hybrid_fallback_triggered", False),
            hybrid_doc_filename=diag.get("hybrid_doc_filename"),
            session_id=session_id,
        ) or ""
    except Exception as e:
        print(f"    [warn] generation failed: {type(e).__name__}: {e}", flush=True)
    generation_ms = int((time.time() - t1) * 1000)

    if log_to_sheet:
        try:
            from src.qa_logger import log_qa_entry
            log_qa_entry(
                ta_id=str(ta_id), ta_slug=(ta.slug if ta else ""),
                ta_name=(ta.name if ta else ""), course_name=(ta.course_name if ta else ""),
                session_id=session_id, query=query, answer=answer,
                sources=list(dict.fromkeys(all_files))[:3], chunk_count=len(chunks),
                latency_ms=retrieval_ms + generation_ms, retrieval_latency_ms=retrieval_ms,
                generation_latency_ms=generation_ms,
                token_count=len(answer.split()) if answer else 0,
                retrieval_diagnostics=diag, llm_model=Config.LLM_MODEL, is_anonymous=False,
            )
        except Exception as e:
            print(f"    [warn] sheet log failed: {type(e).__name__}: {e}", flush=True)

    try:
        db.session.rollback()
        stale = ChatSession.query.get(session_id)
        if stale:
            db.session.delete(stale)
            db.session.commit()
    except Exception:
        db.session.rollback()

    return {
        "answer": answer,
        "sources": list(dict.fromkeys(all_files)),
        "book_sources": sorted(book_files),
        "chunk_count": len(chunks),
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "retrieval_method": diag.get("retrieval_method"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-sheet", action="store_true", help="skip Google Sheet logging")
    ap.add_argument("--log-tab", default="concept_probe")
    ap.add_argument("--out", default=str(Path(__file__).parent / "concept_probe_results.md"))
    ap.add_argument("--limit", type=int, default=None, help="first N queries only (smoke test)")
    args = ap.parse_args()

    from app import app
    from src.retriever import retrieve_context
    from config import Config

    if not args.no_sheet:
        Config.QA_LOG_TAB_NAME = args.log_tab
        if not Config.QA_LOG_SHEET_ID:
            sys.exit("ERROR: sheet logging requested but qa_log_googlesheet is unset.")
        print(f"Sheet logging -> tab {args.log_tab!r}", flush=True)

    queries = QUERIES[: args.limit] if args.limit else QUERIES
    results: list[dict] = []

    with app.app_context():
        for qi, query in enumerate(queries, 1):
            row = {"query": query}
            for label, ta_id in CORPORA:
                print(f"[{qi:2}/{len(queries)}] {label:11} {query[:56]}", flush=True)
                row[label] = run_one(query, ta_id, retrieve_context, not args.no_sheet)
            results.append(row)

    if not args.no_sheet:
        # log_qa_entry writes on a DAEMON thread and returns before the row lands;
        # a short-lived CLI exits and kills those writes mid-flight.
        print("Waiting for sheet writes...", flush=True)
        import threading
        main_t = threading.main_thread()
        for t in threading.enumerate():
            if t is not main_t and t.is_alive():
                t.join(timeout=20)

    out = Path(args.out)
    with out.open("w") as f:
        f.write("# Conceptual-question probe — combined corpus vs books-only\n\n")
        f.write(f"_{datetime.utcnow().isoformat(timespec='seconds')}Z — "
                f"{len(queries)} questions x 2 corpora_\n\n")
        f.write("**combined** = professor's material + books (`EgZ14pvqEYzfQRTM`) · "
                "**books-only** = books alone (`tLMAxWBRrycrJsWl`), the control\n\n")
        f.write("The question this answers: on conceptual ground that BOTH corpora cover, does "
                "the combined TA answer from the professor's material or from the textbooks — "
                "and does the combined answer read like his treatment or like the book's?\n\n")

        withbook = sum(1 for r in results if r["combined"]["book_sources"])
        f.write(f"**Book sources appeared in {withbook}/{len(results)} combined-corpus answers.**\n\n")
        f.write("| # | question | combined sources | book? |\n|---|---|---|---|\n")
        for i, r in enumerate(results, 1):
            src = ", ".join(r["combined"]["sources"][:3]) or "—"
            f.write(f"| {i} | {r['query'][:60]} | {src[:70]} | "
                    f"{'YES' if r['combined']['book_sources'] else 'no'} |\n")
        f.write("\n---\n\n")

        for i, r in enumerate(results, 1):
            f.write(f"## {i}. {r['query']}\n\n")
            for label, _ in CORPORA:
                d = r[label]
                f.write(f"### {label}\n\n")
                f.write(f"*sources: {', '.join(d['sources'][:5]) or '—'}*")
                if label == "combined" and d["book_sources"]:
                    f.write(f"  ·  **book sources: {', '.join(d['book_sources'])}**")
                f.write(f"  ·  _{d['chunk_count']} chunks, {d['retrieval_method']}, "
                        f"{d['retrieval_ms']}ms + {d['generation_ms']}ms_\n\n")
                f.write((d["answer"] or "_(no answer)_") + "\n\n")
            f.write("---\n\n")

    print(f"\nWrote {out}")
    print(f"Book sources appeared in {sum(1 for r in results if r['combined']['book_sources'])}"
          f"/{len(results)} combined answers")


if __name__ == "__main__":
    main()
