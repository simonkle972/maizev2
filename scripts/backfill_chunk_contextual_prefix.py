"""Phase B Stage B9 — backfill DocumentChunk.contextual_prefix + re-embed.

For every DocumentChunk whose contextual_prefix is NULL (or all chunks of a doc
when --force is passed), generate a 1-2 sentence Anthropic-style context blob
situating the chunk within its parent doc, prepend it to the chunk's existing
section-enriched text, re-embed, and persist BOTH the prefix and the new
embedding atomically.

Per-chunk operation (NOT per-doc re-index): reads doc text by concatenating
existing chunks — no PDF parsing, no vision calls. Reversible: if we ever want
to revert to no-prefix embeddings, we can write a sibling script that re-embeds
with prefix='' (or just NULL the column and re-run indexing on the un-prefixed
chunk_text_enriched).

Cost note: ~$0.20 for the 75-doc ECON corpus; ~$5-10 for full prod re-index of
~5-15k chunks. OpenAI prompt-caching kicks in automatically for the full-doc
prefix once it exceeds 1024 tokens — so chunks 2+ of each doc get a discount.

Idempotent: skips chunks whose contextual_prefix is already set unless --force.
Commits every 50 chunks to limit lost progress on transient failures.

Run:
    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 venv/bin/python scripts/backfill_chunk_contextual_prefix.py

Optional:
    --ta-id <id>        Restrict to one TA (default: all TAs)
    --doc-id <int>      Restrict to one Document (overrides --ta-id)
    --limit <int>       Stop after this many chunks (useful for smoke testing)
    --dry-run           Don't commit; print what WOULD change
    --force             Re-prefix even if contextual_prefix is already set
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


def reconstruct_full_doc_text(chunks) -> str:
    """Join chunk_text in chunk_index order. Same trick as backfill_doc_summaries."""
    return "\n\n".join(c.chunk_text for c in chunks if c.chunk_text)


def reconstruct_enriched_text(chunk, doc_filename: str) -> str:
    """Rebuild the section-enriched embedding input the chunker would have produced.

    Matches the pre-B9 shape used in chunk_text_with_context (filename + section
    in square brackets) so prefix prepending matches the indexing pipeline's
    behavior exactly. If chunk_context is missing, falls back to filename-only.
    """
    if chunk.chunk_context:
        return f"[{doc_filename} > {chunk.chunk_context}] {chunk.chunk_text}"
    return f"[{doc_filename}] {chunk.chunk_text}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ta-id", type=str, default=None,
                        help="Restrict backfill to a single TA. Default: all TAs.")
    parser.add_argument("--doc-id", type=int, default=None,
                        help="Restrict to a single Document (overrides --ta-id).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many chunks; useful for smoke testing.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't commit; just print what would change.")
    parser.add_argument("--force", action="store_true",
                        help="Re-prefix even if contextual_prefix is already set.")
    args = parser.parse_args()

    from openai import OpenAI

    from app import app
    from config import Config
    from models import db, Document, DocumentChunk
    from src.document_processor import generate_contextual_prefix

    client = OpenAI(api_key=Config.OPENAI_API_KEY)

    with app.app_context():
        # Resolve which docs to process.
        doc_q = Document.query
        if args.doc_id is not None:
            doc_q = doc_q.filter(Document.id == args.doc_id)
        elif args.ta_id:
            doc_q = doc_q.filter_by(ta_id=args.ta_id)
        docs = doc_q.order_by(Document.ta_id, Document.id).all()

        total_chunks_estimate = (
            DocumentChunk.query
            .filter(DocumentChunk.document_id.in_([d.id for d in docs]))
            .count()
        )
        print(
            f"Backfilling contextual_prefix across {len(docs)} doc(s), "
            f"{total_chunks_estimate} chunk(s) total (estimate)...",
            flush=True,
        )

        n_prefixed = 0
        n_skipped_already_set = 0
        n_failed = 0
        pending_in_batch = 0
        t_start = time.time()

        for di, doc in enumerate(docs, 1):
            doc_tag = f"[doc {di:3}/{len(docs)}] ta={doc.ta_id} doc_id={doc.id} '{(doc.display_name or doc.original_filename)[:50]}'"

            chunks = (DocumentChunk.query
                      .filter_by(document_id=doc.id)
                      .order_by(DocumentChunk.chunk_index).all())
            if not chunks:
                print(f"  {doc_tag} SKIP (no chunks indexed)", flush=True)
                continue

            full_text = reconstruct_full_doc_text(chunks)
            if not full_text.strip():
                print(f"  {doc_tag} SKIP (chunks empty)", flush=True)
                continue

            print(f"  {doc_tag} processing {len(chunks)} chunk(s)...", flush=True)

            for ci, chunk in enumerate(chunks):
                if args.limit is not None and (n_prefixed + n_skipped_already_set + n_failed) >= args.limit:
                    break

                chunk_tag = f"chunk_index={chunk.chunk_index}"

                if chunk.contextual_prefix and not args.force:
                    n_skipped_already_set += 1
                    continue

                try:
                    t0 = time.time()
                    prefix = generate_contextual_prefix(
                        full_doc_text=full_text,
                        chunk_text=chunk.chunk_text,
                        filename=doc.original_filename,
                        content_title=doc.content_title or "",
                        client=client,
                    )
                    if not prefix:
                        print(f"      {chunk_tag} FAILED (generate_contextual_prefix returned empty)", flush=True)
                        n_failed += 1
                        continue

                    enriched_text = reconstruct_enriched_text(chunk, doc.original_filename)
                    embed_input = f"{prefix}\n\n{enriched_text}"
                    emb_response = client.embeddings.create(
                        model=Config.EMBEDDING_MODEL,
                        input=embed_input,
                    )
                    new_embedding = emb_response.data[0].embedding
                    latency_ms = int((time.time() - t0) * 1000)
                except Exception as e:
                    print(f"      {chunk_tag} FAILED prefix/embed: {type(e).__name__}: {e}", flush=True)
                    n_failed += 1
                    continue

                preview = prefix[:80].replace("\n", " ")
                if args.dry_run:
                    if ci < 2:  # don't flood log on dry-run
                        print(f"      {chunk_tag} DRY-RUN would set prefix ({len(prefix)} chars, {latency_ms}ms): {preview}...", flush=True)
                    n_prefixed += 1
                    continue

                chunk.contextual_prefix = prefix
                chunk.embedding = new_embedding
                n_prefixed += 1
                pending_in_batch += 1

                if pending_in_batch >= COMMIT_BATCH_SIZE:
                    db.session.commit()
                    print(f"    -- committed batch ({pending_in_batch} chunks) --", flush=True)
                    pending_in_batch = 0

            if args.limit is not None and (n_prefixed + n_skipped_already_set + n_failed) >= args.limit:
                print(f"  -- hit --limit={args.limit}; stopping --", flush=True)
                break

        if not args.dry_run and pending_in_batch > 0:
            db.session.commit()
            print(f"  -- committed final batch ({pending_in_batch} chunks) --", flush=True)

        elapsed = time.time() - t_start
        print()
        print(f"=== Backfill summary ({elapsed:.1f}s) ===", flush=True)
        print(f"  Chunks prefixed:           {n_prefixed}")
        print(f"  Skipped (already set):     {n_skipped_already_set}")
        print(f"  Failed:                    {n_failed}")
        if args.dry_run:
            print(f"  (dry-run: no DB changes committed)")
        return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
