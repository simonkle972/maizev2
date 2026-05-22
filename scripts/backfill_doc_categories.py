"""Phase A Stage 2B — one-shot backfill for doc_categories + doc_category.

Two-pass backfill:
  Pass 1: Seed TeachingAssistant.doc_categories with DEFAULT_DOC_CATEGORIES
          for any TA that doesn't already have it set. Without categories
          configured, the per-doc classifier has nothing to pick from.
  Pass 2: For every Document, classify into the parent TA's categories and
          write Document.doc_category + propagate to DocumentChunk.doc_category.

Reconstructs doc text from existing DocumentChunk.chunk_text (same trick as
the Stage 2 backfill — avoids re-running extraction).

Per-doc cost: ~1-2s. On a 69-doc corpus, total runtime ≈ 1-2 minutes.

Idempotent: skips docs whose doc_category is already set unless --force is
passed. Always re-seeds doc_categories on TAs missing it; never overwrites
an existing list (a professor may have customized it).

Run:
    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 venv/bin/python scripts/backfill_doc_categories.py

Optional:
    --ta-id <id>        Restrict to one TA (default: all TAs)
    --dry-run           Don't commit; print what WOULD change
    --force             Reclassify even if doc_category already set
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ta-id", type=str, default=None,
                        help="Restrict backfill to a single TA. Default: all TAs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't commit; just print what would change.")
    parser.add_argument("--force", action="store_true",
                        help="Reclassify even if doc_category is already set.")
    args = parser.parse_args()

    from app import app
    from models import db, Document, DocumentChunk, TeachingAssistant
    from src.document_processor import classify_doc_category, DEFAULT_DOC_CATEGORIES

    with app.app_context():
        # ---- Pass 1: seed doc_categories on TAs missing it. ----
        ta_q = TeachingAssistant.query
        if args.ta_id:
            ta_q = ta_q.filter_by(id=args.ta_id)
        tas = ta_q.all()
        print(f"Pass 1: checking doc_categories on {len(tas)} TA(s)...", flush=True)
        n_ta_seeded = 0
        for ta in tas:
            if ta.doc_categories:
                continue
            if args.dry_run:
                print(f"  [TA {ta.id}] DRY-RUN would seed doc_categories ({len(DEFAULT_DOC_CATEGORIES)} entries)", flush=True)
            else:
                ta.doc_categories = list(DEFAULT_DOC_CATEGORIES)
                print(f"  [TA {ta.id}] seeded doc_categories ({len(DEFAULT_DOC_CATEGORIES)} entries)", flush=True)
            n_ta_seeded += 1
        if not args.dry_run:
            db.session.commit()
        print(f"Pass 1 done: {n_ta_seeded} TA(s) seeded.", flush=True)
        print()

        # ---- Pass 2: classify each Document into the parent TA's categories. ----
        doc_q = Document.query
        if args.ta_id:
            doc_q = doc_q.filter_by(ta_id=args.ta_id)
        docs = doc_q.order_by(Document.ta_id, Document.id).all()
        print(f"Pass 2: classifying {len(docs)} document(s)...", flush=True)

        # Cache TA categories per TA id. In dry-run mode, fall back to the
        # default seed list so pass 2 still produces meaningful output for
        # TAs that would have been seeded.
        ta_cats_cache: dict[str, list] = {}
        for ta in tas:
            cats = ta.doc_categories or []
            if not cats and args.dry_run:
                cats = list(DEFAULT_DOC_CATEGORIES)
            ta_cats_cache[ta.id] = cats

        n_classified = 0
        n_skipped_already_set = 0
        n_skipped_no_chunks = 0
        n_skipped_no_categories = 0
        n_failed = 0
        t_start = time.time()

        for i, doc in enumerate(docs, 1):
            tag = f"[{i:3}/{len(docs)}] ta={doc.ta_id} doc_id={doc.id} '{(doc.display_name or doc.original_filename)[:50]}'"

            if doc.doc_category and not args.force:
                print(f"  {tag} SKIP (already set: doc_category={doc.doc_category}; pass --force to reclassify)", flush=True)
                n_skipped_already_set += 1
                continue

            ta_categories = ta_cats_cache.get(doc.ta_id) or []
            if not ta_categories:
                print(f"  {tag} SKIP (parent TA has no doc_categories)", flush=True)
                n_skipped_no_categories += 1
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
                slug, confidence, rationale = classify_doc_category(
                    full_text, doc.original_filename, ta_categories
                )
                latency_ms = int((time.time() - t0) * 1000)
            except Exception as e:
                print(f"  {tag} FAILED classify_doc_category: {type(e).__name__}: {e}", flush=True)
                n_failed += 1
                continue

            if args.dry_run:
                print(f"  {tag} DRY-RUN would set doc_category={slug} (conf={confidence:.2f}, {latency_ms}ms): {rationale[:80]}", flush=True)
                continue

            doc.doc_category = slug
            DocumentChunk.query.filter_by(document_id=doc.id).update(
                {"doc_category": slug}, synchronize_session=False
            )
            db.session.commit()
            n_classified += 1
            print(f"  {tag} OK doc_category={slug} (conf={confidence:.2f}, {latency_ms}ms, {len(chunks)} chunks synced)", flush=True)

        elapsed = time.time() - t_start
        print()
        print(f"=== Backfill summary ({elapsed:.1f}s) ===", flush=True)
        print(f"  TAs seeded (Pass 1):       {n_ta_seeded}")
        print(f"  Docs classified (Pass 2):  {n_classified}")
        print(f"  Skipped (already set):     {n_skipped_already_set}")
        print(f"  Skipped (no chunks):       {n_skipped_no_chunks}")
        print(f"  Skipped (no TA categories):{n_skipped_no_categories}")
        print(f"  Failed:                    {n_failed}")
        if args.dry_run:
            print(f"  (dry-run: no DB changes committed)")
        return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
