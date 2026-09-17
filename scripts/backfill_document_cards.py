"""Phase 4 backfill: generate the document card for existing documents.

For every document (per TA, or all TAs) without a card -- or every LLM-sourced card with
--regenerate -- call src/doc_card.generate_card with the document's stored text, its filename,
the sibling documents and the course's category list; then write the card and sync the
chunks' doc_label + lexical index (no re-extraction, no re-embedding).

Dry run by default: prints a table of the cards it WOULD write, for review by hand.

Run:
    DOTENV_PATH=.env.local python scripts/backfill_document_cards.py [--ta-id X] [--regenerate] [--model M]
    DOTENV_PATH=.env.local python scripts/backfill_document_cards.py --ta-id X --apply

Professor-edited cards (card_source == 'professor') are never overwritten.
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


def _doc_text(doc, db) -> str:
    if doc.full_text:
        return doc.full_text
    from models import DocumentChunk
    rows = (db.session.query(DocumentChunk.chunk_text).filter_by(document_id=doc.id)
            .order_by(DocumentChunk.chunk_index).all())
    return "\n\n".join(r[0] for r in rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ta-id", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--regenerate", action="store_true", help="also redo existing LLM-sourced cards")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from app import app
    from models import db, Document, TeachingAssistant
    from config import Config
    from src.doc_card import generate_card, apply_card, sibling_lines, card_label, sync_chunk_identity
    from src.retriever import _docs_query, get_openai_client
    Config.QA_LOG_SHEET_ID = None

    with app.app_context():
        tas = ([TeachingAssistant.query.get(args.ta_id)] if args.ta_id
               else TeachingAssistant.query.filter(TeachingAssistant.document_count > 0).all())
        total_written = 0
        for ta in tas:
            if ta is None:
                sys.exit("ERROR: TA not found")
            docs = _docs_query(ta.id).order_by(Document.id).all()
            todo = [d for d in docs if (not d.card_title) or (args.regenerate and d.card_source != "professor")]
            if args.limit:
                todo = todo[: args.limit]
            print(f"\n=== {ta.name} ({ta.id}) — {len(docs)} documents, {len(todo)} to card, "
                  f"{sum(1 for d in docs if d.card_source == 'professor')} professor-owned ===")
            if not todo:
                continue
            cats = ta.doc_categories or []
            course = (ta.course_name or ta.name) or ""
            print(f"{'id':>4}  {'title':38} {'kind':16} {'no.':>5} {'part':8} {'term':14}  aliases / original file")
            for d in todo:
                full = Document.query.get(d.id)
                text = _doc_text(full, db)
                if not text.strip():
                    print(f"{d.id:>4}  (no text; skipped)                    {full.original_filename}")
                    continue
                t0 = time.time()
                try:
                    card = generate_card(text, full.original_filename, sibling_lines(ta.id, exclude_doc_id=d.id, docs=docs),
                                         cats, course_name=course, model=args.model)
                except Exception as e:
                    print(f"{d.id:>4}  FAILED {type(e).__name__}: {str(e)[:100]}  ({full.original_filename})")
                    continue
                print(f"{d.id:>4}  {card['title'][:38]:38} {card['kind'][:16]:16} {card['number'][:5]:>5} "
                      f"{card['part'][:8]:8} {card['term'][:14]:14}  {', '.join(card['aliases'][:6])[:60]}"
                      f"\n      file: {full.original_filename}  |  {card['reason'][:110]}  ({time.time() - t0:.1f}s)")
                if args.apply:
                    changed = apply_card(full, card)
                    if full.summary and card.get('summary'):   # summary changed: refresh its embedding
                        try:
                            emb = get_openai_client().embeddings.create(model=Config.EMBEDDING_MODEL, input=full.summary)
                            full.summary_embedding = emb.data[0].embedding
                        except Exception as e:
                            print(f"      summary embedding failed: {e}")
                    db.session.commit()
                    n = sync_chunk_identity(full, db)
                    db.session.commit()
                    total_written += 1
                    print(f"      written: '{card_label(full)}' -> {n} chunks synced")
                    # later siblings see this card
                    for k, x in enumerate(docs):
                        if x.id == full.id:
                            docs[k] = _docs_query(ta.id).filter(Document.id == full.id).first()
        print(f"\n{'Applied' if args.apply else 'Dry run'}: {total_written} cards written.")
        if not args.apply:
            print("Re-run with --apply to write them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
