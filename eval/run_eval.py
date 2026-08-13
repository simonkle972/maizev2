"""Eval harness for Phase A retrieval redesign.

Loads eval/maize_eval_v1.jsonl, runs each row through `retrieve_context()` in
src/retriever.py, and reports a per-failure-type scorecard. Designed to be run
twice: once against the CURRENT retriever (baseline) and once against the
REFINED retriever after the implementation gate. Comparing the two scorecards
is what proves the architectural change actually fixed Failure Types A-E
without regressing working cases.

Usage:
    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/run_eval.py
    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/run_eval.py --baseline  # also writes baseline_scorecard_pre_refinement.md
    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/run_eval.py --row-id econ_s1117_real_typeB_01  # run just one row

Prereqs: the ECON S1117 TA's documents must be indexed in the local DB the
script connects to. If not present, the script prints a helpful error.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Allow `from app import app` and `from src.retriever import ...` to resolve
# when this script is run as `python eval/run_eval.py` (Python adds the
# script's directory to sys.path, not cwd).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

EVAL_FILE = Path(__file__).parent / "maize_eval_v1.jsonl"
BASELINE_OUT = Path(__file__).parent / "baseline_scorecard_pre_refinement.md"


@dataclass
class RowResult:
    row_id: str
    failure_type: str | None
    source: str
    ta_id: str
    not_in_corpus: bool
    correct_doc_ids: list[str]
    hard_negative_doc_ids: list[str]
    forbidden_doc_ids: list[str]
    forbidden_text_fragments: list[str]
    retrieved_doc_ids: list[str]
    correct_hit_at_5: bool
    hard_negative_top1: bool
    forbidden_hit: bool
    forbidden_text_hit: bool  # chunk-level: any retrieved chunk text contains any forbidden fragment
    # Pre-rerank metrics — same logic but applied to the chunk order BEFORE
    # llm_rerank() ran. Lets us measure rerank's actual lift in isolation.
    # The diff (post − pre) is rerank "lift": positive = rerank helped,
    # negative = rerank hurt, zero = rerank had no effect on top-5 membership.
    pre_rerank_retrieved_doc_ids: list[str] = field(default_factory=list)
    correct_hit_at_5_pre_rerank: bool = False
    hard_negative_top1_pre_rerank: bool = False
    error: str | None = None
    retrieval_latency_ms: int = 0
    # Wave 2 (2026-05-31) — intent-classification dimensions. Populated for ALL
    # rows so we can audit intent-class accuracy across the body, but the
    # per-bucket scorecard surfaces them only where they're the primary signal
    # (H/I/J/K/L). See eval/schema.md "Wave 2 intent-classification failures".
    expected_action: str = "retrieve"          # row's expected_action ("retrieve" | "redirect" | "no_retrieval")
    expected_intent_class: str | None = None   # row's expected_intent.intent_class, if labeled
    contextualizer_intent: str | None = None   # diagnostics["intent"] — what the contextualizer classified as
    adversarial_short_circuit_fired: bool = False  # diagnostics["adversarial_short_circuit"]
    retrieved_chunk_count: int = 0             # raw count of returned chunks (0 means skip-retrieval path fired)
    # Bucket-specific HIT signal — what counts as a pass for THIS row's
    # expected_action. For "retrieve" rows this equals correct_hit_at_5; for
    # "redirect" rows, equals adversarial_short_circuit_fired; for
    # "no_retrieval" rows, equals (intent == "clarification") today (proxy)
    # OR retrieved_chunk_count == 0 post-LangGraph-adaptation.
    bucket_hit: bool = False
    # Multi-doc (H) — stricter than correct_hit_at_5 (which is ANY correct in
    # top-5). all_correct_in_top_k requires EVERY correct doc to land in top-5.
    all_correct_in_top_5: bool = False
    # Concept-vs-problem (J) — did the contextualizer's classification match
    # the expected concept_or_problem label? Proxy until we expose a
    # dedicated concept_or_problem signal in diagnostics; today maps
    # contextualizer_intent="concept_lookup" → "concept", anything else → "problem".
    intent_class_match: bool = False


def load_rows():
    if not EVAL_FILE.exists():
        sys.exit(f"ERROR: {EVAL_FILE} not found")
    rows = []
    with EVAL_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _log_row_to_sheet(row: dict, chunks: list, diagnostics: dict, session_id: str,
                      latency_ms: int) -> None:
    """
    Write one eval row to the QA-log sheet so results are reviewable in the same
    place as real traffic.

    The eval calls retrieve_context directly, bypassing the chat layer where
    log_qa_entry normally fires — so without this, eval results exist only as an
    aggregate scorecard with no row-level detail to inspect.

    Retrieval-only: `answer` is empty and generation latency/token count are 0.
    Everything retrieval-side (sources, diagnostics, retrieval latency) populates
    normally.

    Point this at its own tab via `qa_log_tab_name=eval_logs` so 92 synthetic rows
    don't skew reads of real dev traffic. Never raises — a logging failure must
    not fail an eval row.
    """
    try:
        from src.qa_logger import log_qa_entry
        from models import TeachingAssistant
        from config import Config

        ta = TeachingAssistant.query.get(row["ta_id"])
        sources = list(dict.fromkeys(c.get("file_name", "") for c in (chunks or [])[:8]))[:3]

        log_qa_entry(
            ta_id=str(row["ta_id"]),
            ta_slug=(ta.slug if ta else ""),
            ta_name=(ta.name if ta else ""),
            course_name=(ta.course_name if ta else ""),
            session_id=session_id,
            query=row["query"],
            answer="",                      # retrieval-only harness
            sources=sources,
            chunk_count=len(chunks or []),
            latency_ms=latency_ms,
            retrieval_latency_ms=latency_ms,
            generation_latency_ms=0,
            token_count=0,
            retrieval_diagnostics=diagnostics,
            llm_model=Config.LLM_MODEL,
            is_anonymous=False,
        )
    except Exception as e:
        print(f"    [warn] sheet logging failed for {row.get('row_id')}: "
              f"{type(e).__name__}: {e}", flush=True)


def evaluate_row(row: dict, retrieve_context, warm_cache: bool = True,
                 log_to_sheet: bool = False) -> RowResult:
    """Evaluate one row.

    warm_cache=True (default): for multi-turn rows, replay each prior user-turn
    query through retrieve_context with a SHARED session_id BEFORE running the
    target query. This populates the session-level doc cache as in production,
    so cache_action paths (preserved_by_*, invalidated_by_*) fire faithfully.
    The recorded prior-turn assistant content is used verbatim as
    conversation_history at each replay step; we never regenerate.

    warm_cache=False: cold-cache mode. Run only the target query with
    conversation_history populated from prior_turns. Tests the contextualizer
    + retrieval but NOT cache anchoring. Faster + cheaper.

    For single-turn rows (no prior_turns), warm and cold modes are identical.
    """
    # A REAL ChatSession row is required, not just an id string. retrieve_context
    # loads the cache via ChatSession.query.get(session_id) and writes it back the
    # same way — with no row, session_context stays None, the cached path is never
    # entered, and every multi-turn row silently degrades to fresh retrieval. That
    # was the state until 2026-08-12: warm-cache mode replayed prior turns at full
    # API cost while populating nothing, so cache-anchoring modes (E, F1, F2) could
    # not reproduce despite being what warm-cache exists to test.
    # ChatSession.id is String(32), so the id must fit — the old f"eval-{uuid4()}"
    # was 41 chars and would not have inserted even if it had been attempted.
    from models import db, ChatSession

    session_id = "eval" + uuid.uuid4().hex[:28]
    prior_turns = list(row.get("prior_turns") or [])

    db.session.add(ChatSession(id=session_id, ta_id=row["ta_id"], user_id=None))
    db.session.commit()

    # Build [user, assistant] pairs from prior_turns so we can replay turn-by-turn.
    turn_pairs: list[tuple[str, str]] = []
    pending_user = None
    for t in prior_turns:
        if t.get("role") == "user":
            pending_user = t.get("content") or ""
        elif t.get("role") == "assistant" and pending_user is not None:
            turn_pairs.append((pending_user, t.get("content") or ""))
            pending_user = None
    # If there's a trailing user turn without an assistant response, drop it
    # (the row's `query` field is the *current* user turn, so a dangling user
    # turn in prior_turns would be malformed — and we'd duplicate the query).

    # Warm-cache replay — feed prior turns through retrieval to populate cache.
    if warm_cache and turn_pairs:
        cumulative_history: list[dict] = []
        for user_msg, asst_msg in turn_pairs:
            try:
                retrieve_context(
                    ta_id=row["ta_id"],
                    query=user_msg,
                    top_k=8,
                    conversation_history=list(cumulative_history),
                    session_id=session_id,
                    course_name="Introduction to Data Analysis and Econometrics",
                )
            except Exception:
                pass  # cache may be partially populated; continue to target query
            cumulative_history.append({"role": "user", "content": user_msg})
            cumulative_history.append({"role": "assistant", "content": asst_msg})

    # Target-query conversation history is the full prior_turns.
    conversation_history = list(prior_turns)

    t0 = time.time()
    pre_rerank_retrieved = []
    diagnostics: dict = {}
    try:
        chunks, diagnostics = retrieve_context(
            ta_id=row["ta_id"],
            query=row["query"],
            top_k=8,
            conversation_history=conversation_history,
            session_id=session_id,
            course_name="Introduction to Data Analysis and Econometrics",
        )
        latency = int((time.time() - t0) * 1000)
        retrieved = [c.get("file_name", "") for c in (chunks or [])][:8]
        retrieved_top5 = retrieved[:5]
        retrieved_chunk_texts = [c.get("text", "") for c in (chunks or [])][:8]
        retrieved_chunk_count = len(chunks or [])
        # Pre-rerank ordering — extracted from the diagnostics dict so we can
        # measure rerank lift in isolation (post − pre).
        pre_candidates = diagnostics.get("pre_rerank_candidates") or []
        pre_rerank_retrieved = [c.get("file", "") for c in pre_candidates][:8]
        error = None
    except Exception as e:
        latency = int((time.time() - t0) * 1000)
        retrieved = []
        retrieved_top5 = []
        retrieved_chunk_texts = []
        retrieved_chunk_count = 0
        pre_rerank_retrieved = []
        error = f"{type(e).__name__}: {e}"

    # Log before the session row is dropped, so anything reading the sheet can
    # still correlate session_id against chat_sessions if needed.
    if log_to_sheet:
        _log_row_to_sheet(row, chunks if not error else [], diagnostics, session_id, latency)

    # Drop the eval session. Retrieval is finished, so everything below is pure
    # computation over already-captured results. Leaving these behind would
    # accumulate a row per eval row per run.
    try:
        db.session.rollback()  # clear any failed transaction from the except path
        stale = ChatSession.query.get(session_id)
        if stale:
            db.session.delete(stale)
            db.session.commit()
    except Exception:
        db.session.rollback()

    correct_set = set(row["correct_doc_ids"])
    hard_neg_set = set(row.get("hard_negative_doc_ids") or [])
    forbidden_set = set(row.get("forbidden_doc_ids") or [])
    forbidden_fragments = row.get("forbidden_text_fragments") or []
    pre_rerank_top5 = pre_rerank_retrieved[:5]

    # Case-insensitive substring match across all retrieved chunk texts.
    # Used to detect G (intra-doc chunk-routing) failures — doc-level hit
    # would falsely register as a pass when chunks come from the wrong section.
    fragments_lower = [f.lower() for f in forbidden_fragments if f]
    forbidden_text_hit = bool(fragments_lower) and any(
        any(frag in (text or "").lower() for frag in fragments_lower)
        for text in retrieved_chunk_texts
    )

    # --- Wave 2: extract intent diagnostics + compute bucket_hit per expected_action ---
    expected_action = row.get("expected_action", "retrieve")
    expected_intent = row.get("expected_intent") or {}
    expected_intent_class = expected_intent.get("intent_class")
    contextualizer_intent = diagnostics.get("intent")
    adversarial_short_circuit_fired = bool(diagnostics.get("adversarial_short_circuit"))

    correct_hit_at_5 = any(d in correct_set for d in retrieved_top5)
    all_correct_in_top_5 = bool(correct_set) and correct_set.issubset(set(retrieved_top5))

    # bucket_hit — the primary HIT signal, depends on the row's expected_action.
    if expected_action == "retrieve":
        bucket_hit = correct_hit_at_5
    elif expected_action == "redirect":
        # HIT = system fired the off-topic short-circuit AND returned no chunks.
        bucket_hit = adversarial_short_circuit_fired and retrieved_chunk_count == 0
    elif expected_action == "no_retrieval":
        # Today (pre-LangGraph): proxy via contextualizer classifying as "clarification".
        # Post-LangGraph adaptation: chunks==[] is also a HIT signal once the upstream skip-gate exists.
        bucket_hit = (contextualizer_intent == "clarification") or (retrieved_chunk_count == 0)
    else:
        bucket_hit = False  # unknown expected_action, validator catches this

    # intent_class_match — used by J (concept-vs-problem) and any row that
    # carries an explicit intent_class label.
    intent_class_match = (
        expected_intent_class is not None
        and contextualizer_intent == expected_intent_class
    )

    return RowResult(
        row_id=row["row_id"],
        failure_type=row["failure_type_target"],
        source=row["source"],
        ta_id=row.get("ta_id", ""),
        not_in_corpus=bool(row.get("not_in_corpus")),
        correct_doc_ids=row["correct_doc_ids"],
        hard_negative_doc_ids=row.get("hard_negative_doc_ids") or [],
        forbidden_doc_ids=row.get("forbidden_doc_ids") or [],
        forbidden_text_fragments=forbidden_fragments,
        retrieved_doc_ids=retrieved,
        correct_hit_at_5=correct_hit_at_5,
        hard_negative_top1=(retrieved[0] in hard_neg_set) if retrieved else False,
        forbidden_hit=any(d in forbidden_set for d in retrieved),
        forbidden_text_hit=forbidden_text_hit,
        pre_rerank_retrieved_doc_ids=pre_rerank_retrieved,
        correct_hit_at_5_pre_rerank=any(d in correct_set for d in pre_rerank_top5),
        hard_negative_top1_pre_rerank=(pre_rerank_retrieved[0] in hard_neg_set) if pre_rerank_retrieved else False,
        error=error,
        retrieval_latency_ms=latency,
        expected_action=expected_action,
        expected_intent_class=expected_intent_class,
        contextualizer_intent=contextualizer_intent,
        adversarial_short_circuit_fired=adversarial_short_circuit_fired,
        retrieved_chunk_count=retrieved_chunk_count,
        bucket_hit=bucket_hit,
        all_correct_in_top_5=all_correct_in_top_5,
        intent_class_match=intent_class_match,
    )


# Wave 1 doc-routing buckets (existing) + Wave 2 intent-classification buckets
# (H/I/J/K/L added 2026-05-31 — see eval/schema.md).
BUCKET_KEYS = ["A", "B", "C", "D", "E", "F1", "F2", "G1", "G2",
               "H", "I", "J", "K", "L", "working"]


def aggregate(results: list[RowResult]) -> dict:
    """Group by failure_type and compute per-bucket metrics.

    Buckets:
      - Wave 1 doc-routing: A, B, C, D, E, F1, F2, G1, G2
      - Wave 2 intent-classification: H (multi-doc), I (doc correction),
        J (concept-vs-problem), K (followup/clarification), L (off-topic)
      - working: rows that must not regress
      - not_in_corpus: reported separately (retrieval cannot pass)

    All buckets carry the shared metrics (hit@5, forbidden_hit, etc.) plus
    bucket-specific metrics that surface in the scorecard when relevant —
    all_correct_in_top_5 for H, intent_class_match for J, and the bucket_hit
    rule which differs by expected_action for K/L.
    """
    in_corpus = [r for r in results if not r.not_in_corpus]
    not_in_corpus = [r for r in results if r.not_in_corpus]

    buckets: dict[str, list[RowResult]] = defaultdict(list)
    for r in in_corpus:
        key = r.failure_type or "working"
        buckets[key].append(r)

    summary = {}
    for key in BUCKET_KEYS:
        bucket = buckets.get(key, [])
        if not bucket:
            summary[key] = None
            continue
        n = len(bucket)
        post = sum(r.correct_hit_at_5 for r in bucket) / n
        pre = sum(r.correct_hit_at_5_pre_rerank for r in bucket) / n
        summary[key] = {
            "n": n,
            "correct_hit_at_5_rate": post,
            "correct_hit_at_5_pre_rerank_rate": pre,
            "rerank_lift": post - pre,  # positive = rerank helped; negative = rerank hurt
            "hard_negative_top1_rate": sum(r.hard_negative_top1 for r in bucket) / n,
            "forbidden_hit_rate": sum(r.forbidden_hit for r in bucket) / n,
            "forbidden_text_hit_rate": sum(r.forbidden_text_hit for r in bucket) / n,
            "error_count": sum(1 for r in bucket if r.error),
            "avg_latency_ms": sum(r.retrieval_latency_ms for r in bucket) / n,
            # Wave 2 metrics — meaningful only on certain buckets but computed
            # everywhere so the scorecard can show or hide them per-bucket.
            "bucket_hit_rate": sum(r.bucket_hit for r in bucket) / n,
            "all_correct_in_top_5_rate": sum(r.all_correct_in_top_5 for r in bucket) / n,
            "intent_class_match_rate": (
                sum(r.intent_class_match for r in bucket if r.expected_intent_class is not None)
                / max(sum(1 for r in bucket if r.expected_intent_class is not None), 1)
                if any(r.expected_intent_class is not None for r in bucket) else None
            ),
            "redirect_fired_rate": sum(r.adversarial_short_circuit_fired for r in bucket) / n,
            "chunks_returned_avg": sum(r.retrieved_chunk_count for r in bucket) / n,
        }
    summary["not_in_corpus"] = {
        "n": len(not_in_corpus),
        "retrieved_anything_relevant": sum(
            any(d in set(r.correct_doc_ids) for d in r.retrieved_doc_ids) for r in not_in_corpus
        ),
        "forbidden_hit_rate": sum(r.forbidden_hit for r in not_in_corpus) / max(len(not_in_corpus), 1),
        "note": "These rows can't pass at the retrieval layer (correct doc not verified in corpus). Reported for tracking only.",
    }
    summary["__overall__"] = {
        "total_rows": len(results),
        "in_corpus": len(in_corpus),
        "not_in_corpus": len(not_in_corpus),
        "errors": sum(1 for r in results if r.error),
        "distinct_tas": sorted({r.ta_id for r in results if r.ta_id}),
    }
    return summary


def format_scorecard(summary: dict, label: str = "Retrieval scorecard") -> str:
    lines = [f"# {label}", ""]
    overall = summary["__overall__"]
    lines.append(f"**Total rows:** {overall['total_rows']} ({overall['in_corpus']} in-corpus + "
                 f"{overall['not_in_corpus']} not-in-corpus). **Errors:** {overall['errors']}.")
    distinct_tas = overall.get("distinct_tas") or []
    if len(distinct_tas) > 1:
        lines.append(
            f"**TAs in this run:** {len(distinct_tas)} — `{'`, `'.join(distinct_tas)}`. "
            f"Cross-TA aggregate scoring; re-run with `--ta-id <id>` to scope to one TA."
        )
    elif len(distinct_tas) == 1:
        lines.append(f"**TA:** `{distinct_tas[0]}` (filtered to one TA — `--ta-id` flag in use OR file only contains rows for this TA).")
    # ---- Wave 1 doc-routing scorecard table ----
    lines.append("")
    lines.append("## Doc-routing buckets (Wave 1)")
    lines.append("")
    lines.append("| Failure type | n | hit@5 pre→post (lift) | hard_neg_top1 | forbidden_hit | forbidden_text_hit | avg_latency_ms | errors |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    label_map = {
        "A": "A (Lab vs PS)",
        "B": "B (Roman numeral siblings, cross-doc)",
        "C": "C (lookalike-unrelated)",
        "D": "D (problem-vs-solutions)",
        "E": "E (cache anchoring)",
        "F1": "F1 (explicit conceptual switch)",
        "F2": "F2 (explicit document switch)",
        "G1": "G1 (intra-doc section confusion)",
        "G2": "G2 (intra-doc Roman/numeric sibling)",
        "H": "H (multi-document intent)",
        "I": "I (document correction)",
        "J": "J (concept-vs-problem)",
        "K": "K (followup/clarification)",
        "L": "L (off-topic / redirect)",
        "working": "working cases",
    }
    wave1_keys = ["A", "B", "C", "D", "E", "F1", "F2", "G1", "G2", "working"]
    for key in wave1_keys:
        b = summary.get(key)
        if b is None:
            lines.append(f"| {label_map[key]} | 0 | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {label_map[key]} | {b['n']} | "
            f"{b['correct_hit_at_5_pre_rerank_rate']:.0%}→{b['correct_hit_at_5_rate']:.0%} "
            f"({b['rerank_lift']:+.0%}) | "
            f"{b['hard_negative_top1_rate']:.0%} | {b['forbidden_hit_rate']:.0%} | "
            f"{b['forbidden_text_hit_rate']:.0%} | "
            f"{b['avg_latency_ms']:.0f} | {b['error_count']} |"
        )

    # ---- Wave 2 intent-classification scorecard table ----
    lines.append("")
    lines.append("## Intent-classification buckets (Wave 2)")
    lines.append("")
    lines.append("| Failure type | n | bucket_hit | hit@5 (doc-routing) | all_correct_in_top_5 (H only) | intent_class_match | redirect_fired (L only) | avg_chunks_returned | avg_latency_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    wave2_keys = ["H", "I", "J", "K", "L"]
    for key in wave2_keys:
        b = summary.get(key)
        if b is None:
            lines.append(f"| {label_map[key]} | 0 | — | — | — | — | — | — | — |")
            continue
        intent_match_str = "—" if b.get("intent_class_match_rate") is None else f"{b['intent_class_match_rate']:.0%}"
        all_correct_str = f"{b['all_correct_in_top_5_rate']:.0%}" if key == "H" else "—"
        redirect_str = f"{b['redirect_fired_rate']:.0%}" if key == "L" else "—"
        lines.append(
            f"| {label_map[key]} | {b['n']} | "
            f"{b['bucket_hit_rate']:.0%} | "
            f"{b['correct_hit_at_5_rate']:.0%} | "
            f"{all_correct_str} | "
            f"{intent_match_str} | "
            f"{redirect_str} | "
            f"{b['chunks_returned_avg']:.1f} | "
            f"{b['avg_latency_ms']:.0f} |"
        )

    nic = summary["not_in_corpus"]
    lines.append("")
    lines.append(f"**Not-in-corpus bucket:** {nic['n']} rows. Retrieved a labeled-correct doc on "
                 f"{nic['retrieved_anything_relevant']}/{nic['n']} (expected 0 — these docs are unverified in the corpus). "
                 f"Forbidden-hit rate: {nic['forbidden_hit_rate']:.0%}.")
    lines.append("")
    lines.append("## Metric definitions")
    lines.append("- **hit@5 pre→post (lift)** — pre-rerank hit-rate → post-rerank hit-rate, with the rerank's contribution as a `(±X%)` delta. Positive = rerank moved correct chunks into top-5 that weren't there before. Negative = rerank pushed correct chunks out. Zero = rerank didn't affect top-5 membership.")
    lines.append("- **correct_hit@5** — fraction of rows where at least one `correct_doc_ids` appeared in retrieved top-5 (post-rerank).")
    lines.append("- **hard_negative_top1** — fraction of rows where the retrieved top-1 doc matched a known hard negative "
                 "(i.e., the current bad retrieval pattern fired).")
    lines.append("- **forbidden_hit** — fraction of rows where ANY retrieved doc was on the forbidden list "
                 "(e.g., solutions doc returned when student is solving). Lower is better; ideal = 0%.")
    lines.append("- **forbidden_text_hit** — fraction of rows where ANY retrieved chunk's text contained a forbidden substring "
                 "(e.g., \"Part II\" when the row tests Part I retrieval). Detects G failures — right doc, wrong section/part. "
                 "Lower is better; ideal = 0%.")
    lines.append("- **bucket_hit** (Wave 2) — the primary HIT signal for a row, depends on its `expected_action`: "
                 "for `retrieve` rows it equals hit@5; for `redirect` rows it requires `adversarial_short_circuit` fired AND zero chunks returned; "
                 "for `no_retrieval` rows it's a proxy via `intent == 'clarification'` today (becomes a true skip-gate metric post-LangGraph adaptation).")
    lines.append("- **all_correct_in_top_5** (H bucket) — stricter than hit@5: requires EVERY `correct_doc_ids` entry to appear in top-5, not just one. Tests whether multi-doc intent surfaces ALL needed docs.")
    lines.append("- **intent_class_match** — fraction of rows (with `expected_intent.intent_class` labeled) where the contextualizer's classification matches the label. Measures intent-classification accuracy independently of retrieval — Q1+Q2 deep-research flagged this as a literature gap; doing this puts Maize ahead of published practice.")
    lines.append("- **redirect_fired** (L bucket) — fraction of rows where `adversarial_short_circuit` fired in diagnostics, regardless of whether chunks were also returned.")
    lines.append("- **avg_chunks_returned** — average number of chunks the retriever returned. For K/L rows the IDEAL value is 0 (system should skip retrieval). Useful as a smoke check that the skip-gate is firing.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true",
                        help="Write scorecard to baseline_scorecard_pre_refinement.md (use against the CURRENT retriever).")
    parser.add_argument("--out", type=str, default=None,
                        help="Write scorecard to a specific path (overrides --baseline default).")
    parser.add_argument("--row-id", type=str, default=None,
                        help="Run only the row with this row_id; useful for debugging a single case.")
    parser.add_argument("--ta-id", type=str, default=None,
                        help="Filter eval rows to those whose ta_id matches. Without this flag the harness "
                             "runs every row across every TA in the eval file (cross-TA mode).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N rows. Useful for smoke-testing the harness.")
    parser.add_argument("--cold-cache", action="store_true",
                        help="Disable warm-cache replay. Default is warm-cache: for multi-turn rows, "
                             "prior user-turn queries replay through retrieve_context with a shared "
                             "session_id BEFORE the target query, so session cache populates as in prod. "
                             "Without this faithful replay, cache-anchoring failure modes (E, F1, F2) "
                             "may not reproduce. Pass --cold-cache to revert to cheaper single-call "
                             "evaluation (~3x faster on multi-turn rows; useful for quick smoke tests).")
    parser.add_argument("--log-to-sheet", action="store_true",
                        help="Write each row to the QA-log Google Sheet so results are reviewable "
                             "row-by-row (query, sources, intent, cache action, retrieval method) "
                             "instead of only as an aggregate scorecard. OFF by default. "
                             "Writes to the 'eval_logs' tab unless --log-tab says otherwise, so eval "
                             "rows never land in dev_logs or qa_logs_v2. Note the harness is "
                             "retrieval-only, so the answer column will be empty.")
    parser.add_argument("--log-tab", type=str, default="eval_logs",
                        help="Sheet tab for --log-to-sheet (default: eval_logs). Set explicitly at "
                             "runtime rather than via the qa_log_tab_name env var, because config.py "
                             "calls load_dotenv(override=True) — the .env file always beats an "
                             "exported variable, so env-based targeting silently writes to whatever "
                             "the file says (which is how 4 eval rows once landed in dev_logs).")
    args = parser.parse_args()

    # Set up Flask app context for DB queries.
    try:
        from app import app
        from src.retriever import retrieve_context
    except Exception as e:
        sys.exit(f"ERROR: failed to import Flask app or retriever: {type(e).__name__}: {e}\n"
                 f"Make sure DOTENV_PATH is set and dependencies are installed.")

    rows = load_rows()
    if args.row_id:
        rows = [r for r in rows if r["row_id"] == args.row_id]
        if not rows:
            sys.exit(f"ERROR: row_id {args.row_id!r} not found")
    if args.ta_id:
        rows_before = len(rows)
        rows = [r for r in rows if r.get("ta_id") == args.ta_id]
        if not rows:
            sys.exit(f"ERROR: no eval rows match ta_id={args.ta_id!r} (had {rows_before} rows before filter)")
    if args.limit:
        rows = rows[: args.limit]

    warm = not args.cold_cache
    cache_mode = "warm-cache (prior turns replayed)" if warm else "cold-cache (target query only)"
    print(f"Running {len(rows)} eval rows against the current retriever... [{cache_mode}]", flush=True)
    if args.log_to_sheet:
        from config import Config as _C
        # Set the tab on Config directly. qa_logger reads Config.QA_LOG_TAB_NAME at
        # call time, and config.py's load_dotenv(override=True) means an exported
        # env var is overwritten by the .env file — so this is the only reliable
        # way to target a tab from the harness.
        _prev_tab = _C.QA_LOG_TAB_NAME
        _C.QA_LOG_TAB_NAME = args.log_tab
        if not _C.QA_LOG_SHEET_ID:
            sys.exit("ERROR: --log-to-sheet given but no sheet configured "
                     "(qa_log_googlesheet is unset). Refusing to run.")
        print(f"Sheet logging ON -> tab {_C.QA_LOG_TAB_NAME!r} "
              f"(env default was {_prev_tab!r})", flush=True)

    def _drain_log_threads(timeout_each: float = 20.0) -> None:
        """
        Wait for qa_logger's background writers before the process exits.

        log_qa_entry starts a DAEMON thread and returns True immediately — True
        means "thread started", not "row written". A short-lived CLI exits before
        those threads finish and the writes are killed mid-flight, so without this
        the last rows of a run silently never land (and a single-row run lands
        nothing at all).
        """
        import threading
        main = threading.main_thread()
        for t in threading.enumerate():
            if t is not main and t.is_alive():
                t.join(timeout=timeout_each)

    results: list[RowResult] = []
    with app.app_context():
        for i, row in enumerate(rows, 1):
            res = evaluate_row(row, retrieve_context, warm_cache=warm,
                               log_to_sheet=args.log_to_sheet)
            results.append(res)
            status = "ERR" if res.error else ("HIT" if res.bucket_hit else "MISS")
            # For Wave 2 expected-action rows, show the diagnostic that explains the HIT/MISS
            # decision instead of top1 (which isn't meaningful for redirect/no_retrieval).
            if res.expected_action == "redirect":
                detail = f"redirect_fired={res.adversarial_short_circuit_fired} chunks={res.retrieved_chunk_count}"
            elif res.expected_action == "no_retrieval":
                detail = f"intent={res.contextualizer_intent} chunks={res.retrieved_chunk_count}"
            else:
                detail = f"top1={res.retrieved_doc_ids[0] if res.retrieved_doc_ids else '<none>'}"
            print(f"  [{i:3}/{len(rows)}] {status:4} {res.row_id:55} "
                  f"({res.retrieval_latency_ms}ms) {detail}",
                  flush=True)

    if args.log_to_sheet:
        print("Waiting for sheet writes to finish...", flush=True)
        _drain_log_threads()

    summary = aggregate(results)
    label = "Baseline (CURRENT retriever)" if args.baseline else "Retrieval scorecard"
    scorecard = format_scorecard(summary, label=label)
    print()
    print(scorecard)

    if args.baseline and not args.out:
        BASELINE_OUT.write_text(scorecard + "\n")
        print(f"\nBaseline written to {BASELINE_OUT}")
    elif args.out:
        Path(args.out).write_text(scorecard + "\n")
        print(f"\nScorecard written to {args.out}")

    return 1 if summary["__overall__"]["errors"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
