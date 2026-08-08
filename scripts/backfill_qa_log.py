"""Backfill Google Sheets qa_logs_v2 from the DB chat_messages table.

Recovers rows for the period when GOOGLE_SERVICE_ACCOUNT_JSON was misquoted in
.env and Sheets writes were silently dropped. Reconstructs each row from
ChatMessage + ChatSession + TeachingAssistant. Diagnostic fields
(retrieval_diagnostics, latency numbers, chunk_count, token_count) are left
blank — they were never persisted to the DB, only to the sheet.

Run on the VPS:
    cd /opt/maize
    sudo -u maize ./venv/bin/python scripts/backfill_qa_log.py --since 2026-05-15 --dry-run
    sudo -u maize ./venv/bin/python scripts/backfill_qa_log.py --since 2026-05-15

Args:
    --since <YYYY-MM-DD or ISO datetime>   Backfill user messages created on/after this.
    --batch <N>                            Rows per Sheets append call. Default 50.
    --dry-run                              Print counts + preview; don't touch Sheets.
    --ta-id <id>                           Restrict to a single TA.
    --limit <N>                            Cap total rows written (smoke testing).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_cutoff(s: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"unparseable --since {s!r}; use YYYY-MM-DD or ISO datetime")


def build_row(user_msg, assistant_msg, session, ta, llm_model: str) -> list[str]:
    """Assemble one row matching QA_LOG_HEADERS ordering.

    Fields not persisted to DB are left as empty strings. Kept in the same
    positional order as `log_qa_entry`'s inline row at qa_logger.py:233.
    """
    sources = assistant_msg.sources if assistant_msg else None
    if isinstance(sources, list):
        sources_str = ", ".join(str(s) for s in sources)
    elif sources:
        sources_str = str(sources)
    else:
        sources_str = ""

    return [
        (user_msg.created_at.isoformat() + "Z") if user_msg.created_at else "",
        str(ta.id) if ta else "",
        ta.slug if ta else "",
        ta.name if ta else "",
        ta.course_name if ta else "",
        session.id if session else "",
        (user_msg.content or "")[:5000],
        ((assistant_msg.content if assistant_msg else "") or "")[:10000],
        sources_str,
        "",  # chunk_count
        "",  # latency_ms
        "",  # retrieval_latency_ms
        "",  # generation_latency_ms
        "",  # token_count
        "",  # total_chunks_in_ta
        "",  # filters_applied
        "",  # filter_match_count
        "",  # retrieval_method
        "",  # is_conceptual
        "",  # score_top1
        "",  # score_top8
        "",  # score_mean
        "",  # score_spread
        "",  # chunk_scores
        "",  # chunk_sources_detail
        "",  # rerank_applied
        "",  # rerank_method
        "",  # rerank_latency_ms
        "",  # llm_score_top1
        "",  # llm_score_top8
        "",  # vector_score_top1
        "",  # top_reasons
        "",  # pre_rerank_candidates
        "",  # hybrid_fallback_triggered
        "",  # hybrid_fallback_reason
        "",  # hybrid_doc_filename
        "",  # hybrid_doc_tokens
        "",  # hybrid_doc_id_method
        "",  # validation_performed
        "",  # validation_passed
        "",  # validation_expected_ref
        "",  # validation_matches_found
        llm_model,
        str(session.user_id is None) if session else "",  # is_anonymous
        "False",  # is_preview — backfilled rows are always real student traffic
        "",  # supplementary_teaching_found
        "",  # supplementary_chunk_count
        "",  # supplementary_concept_query
        "",  # supplementary_skip_reason
        "",  # contextualizer_enabled
        "",  # contextualizer_fallback
        "",  # contextualizer_latency_ms
        "",  # rewritten_query
        "",  # intent
        "",  # current_focus
        "",  # cache_action
        "",  # adversarial_short_circuit
        "",  # moderation_latency_ms
        "",  # vector_search_latency_ms
        "",  # supplementary_latency_ms
        "",  # hybrid_fetch_latency_ms
        "",  # generation_ttft_ms
        "",  # prompt_tokens_total
        "",  # prompt_tokens_cached
        "",  # paste_detected
        "",  # paste_doc
        "",  # paste_match_length
        "",  # paste_containment
        "",  # paste_longest_run
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", required=True,
                        help="Cutoff datetime (UTC). Formats: YYYY-MM-DD, YYYY-MM-DD HH:MM, ISO datetime.")
    parser.add_argument("--batch", type=int, default=50, help="Rows per Sheets append call (default 50).")
    parser.add_argument("--dry-run", action="store_true", help="Don't hit Sheets; just count + preview.")
    parser.add_argument("--ta-id", type=str, default=None, help="Restrict to a single TA.")
    parser.add_argument("--limit", type=int, default=None, help="Cap total rows written (smoke testing).")
    args = parser.parse_args()

    try:
        since_dt = parse_cutoff(args.since)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Deferred imports so --help doesn't require Flask+DB to be reachable.
    from app import app
    from config import Config
    from models import db, ChatMessage, ChatSession, TeachingAssistant
    from src.qa_logger import _get_sheets_service, QA_LOG_HEADERS, _ensure_headers_exist

    with app.app_context():
        q = (
            db.session.query(ChatMessage)
            .filter(ChatMessage.role == "user", ChatMessage.created_at >= since_dt)
            .order_by(ChatMessage.session_id, ChatMessage.created_at)
        )
        user_msgs = q.all()
        print(f"Found {len(user_msgs)} user messages since {since_dt.isoformat()}")

        session_ids = list({m.session_id for m in user_msgs})
        sessions_by_id = {
            s.id: s for s in ChatSession.query.filter(ChatSession.id.in_(session_ids)).all()
        } if session_ids else {}
        ta_ids = list({s.ta_id for s in sessions_by_id.values()})
        tas_by_id = {
            t.id: t for t in TeachingAssistant.query.filter(TeachingAssistant.id.in_(ta_ids)).all()
        } if ta_ids else {}

        if args.ta_id:
            user_msgs = [
                m for m in user_msgs
                if sessions_by_id.get(m.session_id)
                and sessions_by_id[m.session_id].ta_id == args.ta_id
            ]
            print(f"After --ta-id={args.ta_id} filter: {len(user_msgs)} user messages")

        rows: list[list[str]] = []
        skipped_no_answer = 0
        skipped_no_session = 0
        for um in user_msgs:
            session = sessions_by_id.get(um.session_id)
            if not session:
                skipped_no_session += 1
                continue
            ta = tas_by_id.get(session.ta_id)
            am = (
                ChatMessage.query
                .filter(
                    ChatMessage.session_id == um.session_id,
                    ChatMessage.role == "assistant",
                    ChatMessage.created_at > um.created_at,
                )
                .order_by(ChatMessage.created_at.asc())
                .first()
            )
            if not am:
                skipped_no_answer += 1
                continue
            rows.append(build_row(um, am, session, ta, getattr(Config, "LLM_MODEL", "") or ""))
            if args.limit and len(rows) >= args.limit:
                break

        print(
            f"Built {len(rows)} Q&A rows "
            f"(skipped: no assistant reply yet={skipped_no_answer}, orphan session={skipped_no_session})"
        )

        if args.dry_run:
            print("--dry-run: not writing to Sheets.")
            if rows:
                print("\nFirst row preview (non-empty fields only):")
                for h, v in zip(QA_LOG_HEADERS, rows[0]):
                    if v:
                        print(f"  {h}: {str(v)[:120]}")
            return 0

        service = _get_sheets_service()
        if not service:
            print("ERROR: Could not initialize Sheets service. Fix credential first.", file=sys.stderr)
            return 3
        if not _ensure_headers_exist(service, Config.QA_LOG_SHEET_ID, Config.QA_LOG_TAB_NAME):
            print("ERROR: Could not ensure sheet headers.", file=sys.stderr)
            return 4

        total = len(rows)
        for i in range(0, total, args.batch):
            chunk = rows[i:i + args.batch]
            for attempt in range(4):
                try:
                    service.spreadsheets().values().append(
                        spreadsheetId=Config.QA_LOG_SHEET_ID,
                        range=f"{Config.QA_LOG_TAB_NAME}!A:A",
                        valueInputOption="RAW",
                        insertDataOption="INSERT_ROWS",
                        body={"values": chunk},
                    ).execute()
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"  append failed (attempt {attempt + 1}/4): {e}. Retrying in {wait}s")
                    time.sleep(wait)
            else:
                print(f"ERROR: gave up on batch starting at row {i}", file=sys.stderr)
                return 5
            print(f"  wrote rows {i}..{i + len(chunk) - 1}  ({i + len(chunk)}/{total})")

        print(f"\nDONE: backfilled {total} rows into {Config.QA_LOG_TAB_NAME}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
