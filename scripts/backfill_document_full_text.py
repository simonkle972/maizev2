"""Phase B latency Phase 1 (2026-08-06) — backfill Document.full_text.

Populates the new full_text cache column for docs indexed BEFORE the column
existed. Reconstructs from DocumentChunk.chunk_text rows in chunk_index order
(same source the retriever's tier-2 fallback would use at query time).

Text will have chunk-overlap redundancy at non-section-boundary transitions —
functionally equivalent to what live extraction would produce for LLM
consumption. Runs in seconds even for the full prod corpus because there are
NO vision/LLM calls (pure DB reads + one UPDATE per doc).

Per-doc cost: single SELECT + single UPDATE. For a corpus of ~200 docs, expect
total runtime <1 minute.

Idempotent: skips docs whose full_text is already set unless --force is passed.
Commits in batches of 50 to bound loss on a transient failure.

Mirrors the pattern in scripts/backfill_doc_summaries.py.

Run:
    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 venv/bin/python scripts/backfill_document_full_text.py

Optional:
    --ta-id <id>        Restrict to one TA (default: all TAs)
    --dry-run           Don't commit; print what WOULD change
    --force             Re-reconstruct even if full_text is already set
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


COMMIT_BATCH_SIZE = 50


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ta-id", type=str, default=None,
                        help="Restrict backfill to a single TA. Default: all TAs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't commit; just print what would change.")
    parser.add_argument("--force", action="store_true",
                        help="Re-reconstruct even if Document.full_text is already set.")
    args = parser.parse_args()

    from app import app
    from models import db, Document, DocumentChunk

    with app.app_context():
        doc_q = Document.query
        if args.ta_id:
            doc_q = doc_q.filter_by(ta_id=args.ta_id)
        docs = doc_q.order_by(Document.ta_id, Document.id).all()
        print(f"Reconstructing full_text for {len(docs)} document(s)...", flush=True)

        n_populated = 0
        n_skipped_already_set = 0
        n_skipped_no_chunks = 0
        n_failed = 0
        pending_in_batch = 0
        t_start = time.time()

        for i, doc in enumerate(docs, 1):
            tag = (
                f"[{i:3}/{len(docs)}] ta={doc.ta_id} doc_id={doc.id} "
                f"'{(doc.display_name or doc.original_filename)[:50]}'"
            )

            if doc.full_text and not args.force:
                print(f"  {tag} SKIP (full_text already set; pass --force to rebuild)", flush=True)
                n_skipped_already_set += 1
                continue

            chunk_rows = (DocumentChunk.query
                          .filter_by(document_id=doc.id)
                          .order_by(DocumentChunk.chunk_index)
                          .with_entities(DocumentChunk.chunk_text)
                          .all())
            if not chunk_rows:
                print(f"  {tag} SKIP (no chunks — not indexed yet)", flush=True)
                n_skipped_no_chunks += 1
                continue

            try:
                t0 = time.time()
                full_text = "\n\n".join(r.chunk_text for r in chunk_rows if r.chunk_text)
                if not full_text.strip():
                    print(f"  {tag} SKIP (chunks empty)", flush=True)
                    n_skipped_no_chunks += 1
                    continue
                latency_ms = int((time.time() - t0) * 1000)
            except Exception as e:
                print(f"  {tag} FAILED reconstruction: {type(e).__name__}: {e}", flush=True)
                n_failed += 1
                continue

            if args.dry_run:
                print(
                    f"  {tag} DRY-RUN would set full_text "
                    f"({len(full_text)} chars from {len(chunk_rows)} chunks, {latency_ms}ms)",
                    flush=True,
                )
                continue

            doc.full_text = full_text
            n_populated += 1
            pending_in_batch += 1
            print(
                f"  {tag} OK ({len(full_text)} chars from {len(chunk_rows)} chunks, {latency_ms}ms)",
                flush=True,
            )

            if pending_in_batch >= COMMIT_BATCH_SIZE:
                db.session.commit()
                print(f"  -- committed batch ({pending_in_batch} docs) --", flush=True)
                pending_in_batch = 0

        if not args.dry_run and pending_in_batch > 0:
            db.session.commit()
            print(f"  -- committed final batch ({pending_in_batch} docs) --", flush=True)

        elapsed = time.time() - t_start
        print()
        print(f"=== Backfill summary ({elapsed:.1f}s) ===", flush=True)
        print(f"  Docs populated:            {n_populated}")
        print(f"  Skipped (already set):     {n_skipped_already_set}")
        print(f"  Skipped (no chunks):       {n_skipped_no_chunks}")
        print(f"  Failed:                    {n_failed}")
        if args.dry_run:
            print(f"  (dry-run: no DB changes committed)")
        return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
