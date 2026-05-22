"""Phase A retrieval refactor — one-shot backfill for doc_role + bm25_tsvector.

Populates Document.doc_role, Document.doc_role_provenance, Document.bm25_tsvector,
and DocumentChunk.doc_role on existing already-indexed documents. New uploads
get these columns set by the worker (see src/document_processor.py — Stage 2).

Cheap path: reconstructs the doc's text by concatenating its existing
DocumentChunk.chunk_text values rather than re-extracting from file_content
or the storage_path. The text content is identical (chunks were derived from
the same source) but this is 10-100x faster and avoids needing the vision
pipeline for image-only PDFs.

Per-doc cost: ~1-2s (one LLM call for classify_doc_role + a SQL
to_tsvector call). On a 69-doc corpus, total runtime ≈ 1-2 minutes.

Idempotent: skips docs whose doc_role_provenance.source == 'professor'
(human overrides) so re-running won't clobber manual classifications.
Skips docs with no chunks (can't reconstruct text).

Run:
    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 venv/bin/python scripts/backfill_doc_role_and_bm25.py

Optional:
    --ta-id <id>        Restrict to one TA (default: all TAs)
    --dry-run           Don't commit; print what WOULD change
    --force             Reclassify even if doc_role already set
                        (still respects 'professor' provenance)
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# Project root on sys.path so `from app import app` resolves when this is
# run via `python scripts/backfill_doc_role_and_bm25.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ta-id", type=str, default=None,
                        help="Restrict backfill to a single TA. Default: all TAs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't commit; just print what would change.")
    parser.add_argument("--force", action="store_true",
                        help="Reclassify even if doc_role is already set. Still skips professor-override rows.")
    args = parser.parse_args()

    from app import app
    from models import db, Document, DocumentChunk
    from src.document_processor import classify_doc_role, sanitize_text

    with app.app_context():
        q = Document.query
        if args.ta_id:
            q = q.filter_by(ta_id=args.ta_id)
        docs = q.order_by(Document.id).all()
        print(f"Backfill: {len(docs)} document(s) selected.", flush=True)

        n_classified = 0
        n_skipped_professor = 0
        n_skipped_already_set = 0
        n_skipped_no_chunks = 0
        n_failed = 0
        t_start = time.time()

        for i, doc in enumerate(docs, 1):
            tag = f"[{i:3}/{len(docs)}] doc_id={doc.id} '{(doc.display_name or doc.original_filename)[:50]}'"

            provenance = doc.doc_role_provenance or {}
            if provenance.get("source") == "professor":
                print(f"  {tag} SKIP (professor override; doc_role={doc.doc_role})", flush=True)
                n_skipped_professor += 1
                continue

            if doc.doc_role and not args.force:
                print(f"  {tag} SKIP (already set: doc_role={doc.doc_role}; pass --force to reclassify)", flush=True)
                n_skipped_already_set += 1
                continue

            chunks = (DocumentChunk.query
                      .filter_by(document_id=doc.id)
                      .order_by(DocumentChunk.chunk_index).all())
            if not chunks:
                print(f"  {tag} SKIP (no chunks — not indexed yet)", flush=True)
                n_skipped_no_chunks += 1
                continue

            full_text = "\n\n".join(c.chunk_text for c in chunks if c.chunk_text)
            if not full_text.strip():
                print(f"  {tag} SKIP (chunks empty)", flush=True)
                n_skipped_no_chunks += 1
                continue

            try:
                t0 = time.time()
                role, confidence, rationale = classify_doc_role(full_text, doc.original_filename)
                latency_ms = int((time.time() - t0) * 1000)
            except Exception as e:
                print(f"  {tag} FAILED classify_doc_role: {type(e).__name__}: {e}", flush=True)
                n_failed += 1
                continue

            if args.dry_run:
                print(f"  {tag} DRY-RUN would set doc_role={role} (conf={confidence:.2f}, {latency_ms}ms): {rationale[:80]}", flush=True)
                continue

            doc.doc_role = role
            doc.doc_role_provenance = {
                "source": "auto",
                "confidence": confidence,
                "classified_at": datetime.utcnow().isoformat() + "Z",
                "rationale": rationale,
            }
            doc.bm25_tsvector = db.func.to_tsvector("english", sanitize_text(full_text))

            # Sync the denormalized doc_role to all chunks of this doc.
            DocumentChunk.query.filter_by(document_id=doc.id).update(
                {"doc_role": role}, synchronize_session=False
            )

            db.session.commit()
            n_classified += 1
            print(f"  {tag} OK doc_role={role} (conf={confidence:.2f}, {latency_ms}ms, {len(chunks)} chunks synced)", flush=True)

        elapsed = time.time() - t_start
        print()
        print(f"=== Backfill summary ({elapsed:.1f}s) ===", flush=True)
        print(f"  Classified:               {n_classified}")
        print(f"  Skipped (professor):      {n_skipped_professor}")
        print(f"  Skipped (already set):    {n_skipped_already_set}")
        print(f"  Skipped (no chunks):      {n_skipped_no_chunks}")
        print(f"  Failed:                   {n_failed}")
        if args.dry_run:
            print(f"  (dry-run: no DB changes committed)")
        return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
