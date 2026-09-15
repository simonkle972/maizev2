"""Re-index only the documents whose stored chunk text shows a known extraction defect.

Two defects are detected in `document_chunks.chunk_text`:

  glyphs  Mathematical-Alphanumeric glyphs (U+1D400..U+1D7FF), which Word/Cambria-Math
          PDFs emit doubled ("𝑓𝑓𝑋𝑋" for f_X). Fixed at extraction by
          `normalize_math_glyphs` (2026-09-13).
  split   Heading digits split by a space ("Problem 1 4", "Question 1 8"), a PyPDF2
          artefact. Fixed by the swap to pypdf (2026-09-13).

Only matching documents are re-processed: their chunks are deleted, `last_indexed_at`
is cleared, and the TA's INCREMENTAL indexer is run, so every other document's chunks
stay in place. The TA stays live: `indexing_status` is not touched and no IndexingJob
row is written, so neither the chat gate nor the admin progress UI reacts. The
affected documents are simply absent from retrieval for the minutes they take.

A code deploy alone changes nothing in stored chunks -- an extractor fix only reaches a
document when it is re-indexed. This script is the targeted alternative to a full
force re-index (which wipes every chunk and gates the chat until it finishes).

Run:
    DOTENV_PATH=.env.local python scripts/reindex_affected_docs.py              # dry run: list only
    DOTENV_PATH=.env.local python scripts/reindex_affected_docs.py --apply      # do it

Options:
    --ta-id <id>          Restrict to one TA (default: all TAs).
    --exclude-ids a,b,c   Document ids to leave alone (e.g. a 350-page textbook with
                          two affected chunks, not worth the vision cost).
    --include-pending     Proceed even if a TA already has other documents with
                          last_indexed_at IS NULL (they would be indexed too).
    --sheet-log           Keep the per-chunk index logging to Google Sheets (off by
                          default so a maintenance run does not flood the QA sheet).

On the VPS:
    cd /opt/maize && sudo -u maize ./venv/bin/python scripts/reindex_affected_docs.py --ta-id <id>
    (dry run first; then add --apply)
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Postgres ARE syntax. NB: \y is the word boundary (\b is backspace in Postgres).
GLYPH_RE = r'[\U0001D400-\U0001D7FF]'
# A literal space, not \s: the defect is a space INSIDE a number on one line
# ("Problem 1 4"). \s also matched "Part 1<newline>1.1(a)", a legitimate heading
# followed by its sub-heading, which produced four false positives on 2026-09-13.
# (?!\.\d) excludes a point value after the number: a Canvas export prints
# "Question 1 0.5 pts", which is correct text (prod, 2026-09-15).
SPLIT_RE = r'(Problem|Question|Exercise|Part|Section) \d \d(?!\.\d)\y'

FIND_SQL = """
SELECT d.ta_id, d.id, d.original_filename,
       count(*) FILTER (WHERE c.chunk_text ~ :glyph) AS glyph_chunks,
       count(*) FILTER (WHERE c.chunk_text ~ :split) AS split_chunks,
       count(*) AS total_chunks
FROM document_chunks c
JOIN documents d ON d.id = c.document_id
WHERE (:ta_id IS NULL OR d.ta_id = :ta_id)
GROUP BY d.ta_id, d.id, d.original_filename
HAVING count(*) FILTER (WHERE c.chunk_text ~ :glyph) > 0
    OR count(*) FILTER (WHERE c.chunk_text ~ :split) > 0
ORDER BY d.ta_id, d.id
"""


def find_affected(db, ta_id: str | None) -> list[dict]:
    from sqlalchemy import text
    rows = db.session.execute(text(FIND_SQL), {"glyph": GLYPH_RE, "split": SPLIT_RE, "ta_id": ta_id}).mappings().all()
    return [dict(r) for r in rows]


def print_table(rows: list[dict]) -> None:
    print(f"{'ta_id':<18} {'id':>5}  {'glyph':>5} {'split':>5} {'total':>5}  filename")
    for r in rows:
        print(f"{r['ta_id']:<18} {r['id']:>5}  {r['glyph_chunks']:>5} {r['split_chunks']:>5} {r['total_chunks']:>5}  {r['original_filename']}")
    print(f"{len(rows)} documents, {sum(r['total_chunks'] for r in rows)} chunks would be re-created")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ta-id", default=None)
    ap.add_argument("--exclude-ids", default="", help="comma-separated document ids to skip")
    ap.add_argument("--apply", action="store_true", help="actually delete chunks and re-index (default: dry run)")
    ap.add_argument("--include-pending", action="store_true")
    ap.add_argument("--sheet-log", action="store_true")
    args = ap.parse_args()
    exclude = {int(x) for x in args.exclude_ids.split(",") if x.strip()}

    from app import app
    from models import db, Document, DocumentChunk
    from config import Config
    from src.document_processor import process_and_index_documents_resumable

    if not args.sheet_log:
        Config.QA_LOG_SHEET_ID = None

    with app.app_context():
        rows = [r for r in find_affected(db, args.ta_id) if r["id"] not in exclude]
        skipped = [r for r in find_affected(db, args.ta_id) if r["id"] in exclude]
        print_table(rows)
        if skipped:
            print("excluded:", ", ".join(f"{r['id']} ({r['original_filename']})" for r in skipped))
        if not rows:
            print("Nothing to do.")
            return 0
        if not args.apply:
            print("\nDry run. Re-run with --apply to re-index these documents.")
            return 0

        by_ta: dict[str, list[dict]] = {}
        for r in rows:
            by_ta.setdefault(r["ta_id"], []).append(r)

        for ta_id, docs in by_ta.items():
            ids = [d["id"] for d in docs]
            pending = Document.query.filter(Document.ta_id == ta_id, Document.last_indexed_at.is_(None),
                                            ~Document.id.in_(ids)).count()
            if pending and not args.include_pending:
                print(f"[{ta_id}] SKIPPED: {pending} other document(s) already pending (last_indexed_at IS NULL); "
                      f"they would be indexed too. Pass --include-pending to proceed.")
                continue

            t0 = time.time()
            deleted = DocumentChunk.query.filter(DocumentChunk.document_id.in_(ids)).delete(synchronize_session=False)
            Document.query.filter(Document.id.in_(ids)).update({"last_indexed_at": None}, synchronize_session=False)
            db.session.commit()
            print(f"\n[{ta_id}] deleted {deleted} chunks from {len(ids)} documents; re-indexing...", flush=True)
            try:
                result = process_and_index_documents_resumable(ta_id, resume_from_doc_id=True)
            except Exception as e:
                db.session.rollback()
                print(f"[{ta_id}] FAILED: {type(e).__name__}: {e}. Documents {ids} are left with "
                      f"last_indexed_at NULL; the next incremental index will retry them.")
                continue
            failed = result.get("docs_failed") or []
            print(f"[{ta_id}] done in {time.time() - t0:.0f}s: {result.get('chunks_indexed')} chunks, "
                  f"{len(result.get('docs_succeeded') or [])} succeeded, {len(failed)} failed")
            for f in failed:
                print(f"    FAILED doc {f.get('doc_id')} {f.get('filename')}: {f.get('error')}")

        # Verification: the same detector over the documents just processed.
        residual = [r for r in find_affected(db, args.ta_id) if r["id"] in {r2["id"] for r2 in rows}]
        if residual:
            print("\nResidual defects after re-index (expected 0 rows):")
            print_table(residual)
        else:
            print("\nVerified: no residual glyph/split defects in the re-indexed documents.")
        return 1 if residual else 0


if __name__ == "__main__":
    sys.exit(main())
