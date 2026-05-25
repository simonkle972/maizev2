"""Phase B Stage B8 backfill — populate DocumentChunk.section_path from existing chunk_context.

The new section_path column was added by migration `0edbee6e8609`. New ingestion
(running through the post-B8 `chunk_text_with_context`) populates section_path
properly with multi-level hierarchy. Existing chunks have only the single-header
`chunk_context` string — this script derives a single-element section_path from
that, normalizing the format so retrieval can filter on it.

Normalization rules:
  chunk_context = "Slide 3:"            → section_path = ["Slide 3"]
  chunk_context = "Slide 7: Title text" → section_path = ["Slide 7"]
  chunk_context = "--- Page 5 ---"      → section_path = ["Page 5"]
  chunk_context = "Section II:"         → section_path = ["Section II"]
  chunk_context = "Section II - Title"  → section_path = ["Section II"]
  chunk_context = "Problem 3:"          → section_path = ["Problem 3"]
  chunk_context = "Part b:"             → section_path = ["Part b"]
  chunk_context = "" or NULL            → section_path = []

Idempotent — skips chunks whose section_path is already non-null (treats prior
backfill or post-B8 ingestion as authoritative).

Run:
    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python scripts/backfill_section_path.py

Optional:
    --ta-id <id>    Restrict backfill to one TA's chunks (default: all TAs)
    --dry-run       Don't commit; print what WOULD change
    --force         Re-derive section_path even when already set
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Same normalization shape as _HEADER_PATTERNS in src/document_processor.py.
# Order matters: first regex to match wins.
_DERIVE_PATTERNS = [
    (re.compile(r'^---\s*Page\s+(\d+)\s*---', re.IGNORECASE), lambda m: f"Page {m.group(1)}"),
    (re.compile(r'^Slide\s+(\d+)', re.IGNORECASE),            lambda m: f"Slide {m.group(1)}"),
    (re.compile(r'^Section\s+(\d+|[IVX]+)', re.IGNORECASE),   lambda m: f"Section {m.group(1).upper()}" if m.group(1).isalpha() else f"Section {m.group(1)}"),
    (re.compile(r'^Part\s+([A-Z])', re.IGNORECASE),           lambda m: f"Part {m.group(1).lower()}"),
    (re.compile(r'^Problem\s+(\d+)', re.IGNORECASE),          lambda m: f"Problem {m.group(1)}"),
    (re.compile(r'^Question\s+(\d+)', re.IGNORECASE),         lambda m: f"Question {m.group(1)}"),
    (re.compile(r'^Exercise\s+(\d+)', re.IGNORECASE),         lambda m: f"Exercise {m.group(1)}"),
]


def derive_section_path(chunk_context: str | None) -> list:
    """Return a single-element section_path list derived from a chunk_context string.

    Returns [] when the chunk_context is empty / doesn't match any known pattern.
    """
    if not chunk_context:
        return []
    text = chunk_context.strip()
    for pattern, normalizer in _DERIVE_PATTERNS:
        m = pattern.search(text)
        if m:
            return [normalizer(m)]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ta-id", type=str, default=None,
                        help="Restrict backfill to a single TA's chunks. Default: all TAs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't commit; just print what would change.")
    parser.add_argument("--force", action="store_true",
                        help="Re-derive section_path even when already set.")
    args = parser.parse_args()

    from app import app
    from models import db, DocumentChunk

    with app.app_context():
        q = DocumentChunk.query
        if args.ta_id:
            q = q.filter_by(ta_id=args.ta_id)
        if not args.force:
            # Skip chunks whose section_path is already populated. Postgres `json`
            # doesn't support `=` directly — comparing NULL is enough; chunks with
            # an empty list `[]` are also fine to skip (derive is idempotent on
            # already-set values, so re-running force=true is safe).
            q = q.filter(DocumentChunk.section_path.is_(None))
        chunks = q.order_by(DocumentChunk.id).all()
        print(f"Backfill: {len(chunks)} chunk(s) selected.", flush=True)

        n_set = 0
        n_empty = 0  # chunks where the derive returns [] (chunk_context was unmatchable)
        n_skipped = 0
        BATCH = 500

        for i, c in enumerate(chunks, 1):
            ctx = c.chunk_context or ""
            derived = derive_section_path(ctx)

            if args.dry_run:
                if i <= 20:  # cap the verbose output
                    print(f"  [{i:5}/{len(chunks)}] doc_id={c.document_id} idx={c.chunk_index} "
                          f"ctx={ctx[:30]!r:32} -> section_path={derived}", flush=True)
                if derived:
                    n_set += 1
                else:
                    n_empty += 1
                continue

            c.section_path = derived
            if derived:
                n_set += 1
            else:
                n_empty += 1

            if i % BATCH == 0:
                db.session.commit()
                print(f"  ...{i}/{len(chunks)} committed", flush=True)

        if not args.dry_run:
            db.session.commit()

        print()
        print(f"=== Backfill summary ===", flush=True)
        print(f"  Chunks scanned:                 {len(chunks)}")
        print(f"  section_path populated (non-empty): {n_set}")
        print(f"  section_path = [] (empty derive):   {n_empty}")
        print(f"  Skipped (already set):          {n_skipped}")
        if args.dry_run:
            print(f"  (dry-run: no DB changes committed)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
