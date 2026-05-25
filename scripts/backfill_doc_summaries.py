"""Phase B Stage B10 — one-shot backfill for Document.summary + summary_embedding.

For every Document, reconstruct text from existing DocumentChunk.chunk_text
(same trick as the Stage 2 / B8 backfills — avoids re-running extraction),
call `summarize_doc` to produce a ~100-150 word summary, embed it via the
existing OpenAI embeddings client, and persist both columns.

Indexing-only TODAY: nothing reads Document.summary_embedding from retrieval
yet. Sets up the future hybrid_doc_search refactor (summary-cosine + LLM
tiebreaker replacing the current BM25+dense+filename RRF + Stage 5 short-circuit).
See `attached_assets/maize-architecture-review-2026-05-23.md` section 3.3.

Per-doc cost: ~1-3s (1 chat completion + 1 embedding call). On a 75-doc
corpus, total runtime ≈ 2-4 minutes.

Idempotent: skips docs whose summary is already set unless --force is passed.
Commits in batches of 50 to avoid losing all progress on a transient failure.

Run:
    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 venv/bin/python scripts/backfill_doc_summaries.py

Optional:
    --ta-id <id>        Restrict to one TA (default: all TAs)
    --dry-run           Don't commit; print what WOULD change
    --force             Re-summarize even if summary already set
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
                        help="Re-summarize even if Document.summary is already set.")
    args = parser.parse_args()

    from openai import OpenAI

    from app import app
    from config import Config
    from models import db, Document, DocumentChunk
    from src.document_processor import summarize_doc

    client = OpenAI(api_key=Config.OPENAI_API_KEY)

    with app.app_context():
        doc_q = Document.query
        if args.ta_id:
            doc_q = doc_q.filter_by(ta_id=args.ta_id)
        docs = doc_q.order_by(Document.ta_id, Document.id).all()
        print(f"Summarizing {len(docs)} document(s)...", flush=True)

        n_summarized = 0
        n_skipped_already_set = 0
        n_skipped_no_chunks = 0
        n_failed = 0
        pending_in_batch = 0
        t_start = time.time()

        for i, doc in enumerate(docs, 1):
            tag = f"[{i:3}/{len(docs)}] ta={doc.ta_id} doc_id={doc.id} '{(doc.display_name or doc.original_filename)[:50]}'"

            if doc.summary and not args.force:
                print(f"  {tag} SKIP (summary already set; pass --force to re-summarize)", flush=True)
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
                summary_text = summarize_doc(
                    full_text, doc.original_filename, doc.content_title or ""
                )
                if not summary_text:
                    print(f"  {tag} FAILED (summarize_doc returned empty)", flush=True)
                    n_failed += 1
                    continue
                emb_response = client.embeddings.create(
                    model=Config.EMBEDDING_MODEL,
                    input=summary_text,
                )
                embedding = emb_response.data[0].embedding
                latency_ms = int((time.time() - t0) * 1000)
            except Exception as e:
                print(f"  {tag} FAILED summarize/embed: {type(e).__name__}: {e}", flush=True)
                n_failed += 1
                continue

            preview = summary_text[:80].replace("\n", " ")
            if args.dry_run:
                print(f"  {tag} DRY-RUN would set summary ({len(summary_text)} chars, ~{len(summary_text.split())} words, {latency_ms}ms): {preview}...", flush=True)
                continue

            doc.summary = summary_text
            doc.summary_embedding = embedding
            n_summarized += 1
            pending_in_batch += 1
            print(f"  {tag} OK summary set ({len(summary_text)} chars, ~{len(summary_text.split())} words, {latency_ms}ms): {preview}...", flush=True)

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
        print(f"  Docs summarized:           {n_summarized}")
        print(f"  Skipped (already set):     {n_skipped_already_set}")
        print(f"  Skipped (no chunks):       {n_skipped_no_chunks}")
        print(f"  Failed:                    {n_failed}")
        if args.dry_run:
            print(f"  (dry-run: no DB changes committed)")
        return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
