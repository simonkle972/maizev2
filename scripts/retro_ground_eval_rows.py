"""Retro-ground all eval rows in eval/maize_eval_v1.jsonl against chunk content.

For each row:
  1. LLM extracts topic keywords + structural markers from query + prior_turns.
  2. SQL searches DocumentChunk.chunk_text for those keywords scoped to row's ta_id.
  3. Ranks docs by keyword-match strength and pulls excerpts.
  4. Compares to current correct_doc_ids and assigns a verdict:
       ✓ CONFIRMED — top matches all in current correct_doc_ids
       ➕ EXPAND   — strong matches not in current correct_doc_ids
       ✏️ CORRECT  — top match not in current; current may be wrong
       ⚠️ WEAK     — no docs strongly match; manual review needed

Output: /tmp/retro_grounding_report.md

Run:
    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 venv/bin/python scripts/retro_ground_eval_rows.py
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


KEYWORD_EXTRACTION_PROMPT = """You will be given a student query (and optionally some prior conversation turns) sent to a course teaching-assistant retrieval system. Your job is to extract the topic + structural markers needed to identify which course document(s) the query is actually about.

Return JSON with three fields:
- "concept_keywords": list of topical keywords/phrases the retrieved doc should discuss (e.g. "circular flow", "Cournot quantity competition", "DCF", "covariance"). Multi-word phrases as single strings. Max 5 items.
- "structural_markers": list of doc-identifying labels (e.g. "homework 1", "problem set 2", "part 1", "exam 2024", "question 2b"). Max 5 items. Include numbers/parts that disambiguate WHICH doc/section.
- "context_keywords": (for multi-turn queries only) keywords from prior turns that anchor what this query is FOLLOWING UP on. Use [] for first-turn queries.

Return ONLY the JSON object. No preamble.

EXAMPLE QUERY: "I need help with questions 2-6 from homework 1"
EXAMPLE OUTPUT: {"concept_keywords":[],"structural_markers":["homework 1","question 2","question 3","question 4","question 5"],"context_keywords":[]}

EXAMPLE QUERY: "ok let's work on 2b next"
PRIOR TURNS: T1 "help with problem 2a from part 1 of 2024 final" / T1 assistant: "Q2a asks ..."
EXAMPLE OUTPUT: {"concept_keywords":[],"structural_markers":["problem 2b","part 1","2024 final","exam 2024"],"context_keywords":["problem 2a","part 1","2024 final"]}

QUERY: {query}
PRIOR TURNS:
{prior_turns_text}
"""


def extract_keywords(client, model, query: str, prior_turns: list) -> dict:
    if prior_turns:
        pt_lines = []
        for t in prior_turns[:6]:  # cap to last 3 turns each side
            role = t.get("role", "")
            content = (t.get("content") or "")[:300].replace("\n", " ")
            pt_lines.append(f"  {role}: {content}")
        pt_text = "\n".join(pt_lines) if pt_lines else "(none)"
    else:
        pt_text = "(none — first turn)"

    prompt = KEYWORD_EXTRACTION_PROMPT.replace("{query}", query).replace("{prior_turns_text}", pt_text)
    resp = client.chat.completions.create(
        # store=False keeps prompts and completions out of OpenAI's Application
        # State retention. Abuse-monitoring logs (up to 30 days) are separate and
        # unaffected; only org-level Zero Data Retention removes those.
        store=False,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=300,
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"concept_keywords": [], "structural_markers": [], "context_keywords": []}


def chunk_search(db, DocumentChunk, Document, ta_id: str, keywords: list[str]) -> dict:
    """For each keyword, find docs containing it. Returns {doc_filename: {kw: chunk_count, excerpts: [str]}}."""
    from sqlalchemy import func

    results = defaultdict(lambda: {"kw_matches": set(), "total_chunks": 0, "excerpts": []})

    if not keywords:
        return dict(results)

    for kw in keywords:
        kw_clean = str(kw).strip()
        if not kw_clean:
            continue
        # ILIKE substring search
        q = (db.session.query(Document.original_filename, Document.display_name, DocumentChunk.chunk_text)
             .join(DocumentChunk, DocumentChunk.document_id == Document.id)
             .filter(Document.ta_id == ta_id)
             .filter(DocumentChunk.chunk_text.ilike(f"%{kw_clean}%"))
             .limit(50))
        for fn, dn, chunk_text in q.all():
            fname = strip_ext(dn or fn)
            entry = results[fname]
            entry["kw_matches"].add(kw_clean)
            entry["total_chunks"] += 1
            if len(entry["excerpts"]) < 2:
                excerpt = excerpt_around_kw(chunk_text, kw_clean, span=160)
                entry["excerpts"].append((kw_clean, excerpt))

    return dict(results)


def strip_ext(s: str) -> str:
    return re.sub(r"\.(pdf|docx|doc|pptx|ppt|xlsx|xls|txt|md)$", "", s, flags=re.IGNORECASE)


def excerpt_around_kw(text: str, kw: str, span: int = 160) -> str:
    if not text:
        return ""
    idx = text.lower().find(kw.lower())
    if idx < 0:
        return text[:span].replace("\n", " ")
    start = max(0, idx - span // 2)
    end = min(len(text), idx + len(kw) + span // 2)
    snippet = text[start:end].replace("\n", " ").strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def rank_docs(search_results: dict, current_correct: list[str]) -> list[tuple]:
    """Return docs sorted by (num_keywords_matched DESC, total_chunks DESC)."""
    ranked = []
    for fname, info in search_results.items():
        n_kw = len(info["kw_matches"])
        n_chunks = info["total_chunks"]
        in_current = fname in current_correct
        ranked.append((fname, n_kw, n_chunks, in_current, info["excerpts"]))
    ranked.sort(key=lambda x: (-x[1], -x[2]))
    return ranked


def verdict_for_row(ranked: list, current_correct: list[str], all_keywords: list[str]) -> tuple[str, str, list[str]]:
    """Return (verdict_emoji, reasoning, proposed_correct_docs)."""
    total_kw = len([k for k in all_keywords if k.strip()])
    current_set = set(current_correct)

    if not ranked:
        return "⚠️ WEAK", "No docs match any keywords. Manual review needed.", current_correct

    strong = [r for r in ranked if r[1] >= max(2, total_kw // 2)] or ranked[:3]
    strong_filenames = [r[0] for r in strong]
    strong_set = set(strong_filenames)

    # ✓ CONFIRMED — every strong match is already in current correct
    if strong_set.issubset(current_set):
        return ("✓ CONFIRMED",
                f"All strong matches ({len(strong)}) already in current correct_doc_ids.",
                current_correct)

    # ✏️ CORRECT — top match is NOT in current (current may be wrong)
    top = ranked[0]
    if not top[3] and top[1] >= max(2, total_kw // 2):
        proposed = list(strong_set | (current_set & {f for f in strong_filenames}))
        proposed = [d for d in (strong_filenames + list(current_set - strong_set)) if d]
        return ("✏️ CORRECT",
                f"Top match `{top[0]}` ({top[1]}/{total_kw} kw, {top[2]} chunks) NOT in current correct_doc_ids. Current may be mislabeled.",
                strong_filenames)

    # ➕ EXPAND — strong matches outside current set; expand
    expand_targets = [d for d in strong_filenames if d not in current_set]
    if expand_targets:
        new_correct = list(current_correct) + expand_targets
        return ("➕ EXPAND",
                f"Strong match(es) not in current: {expand_targets!r}. Consider expanding correct_doc_ids.",
                new_correct)

    # Default: confirmed-ish
    return ("✓ CONFIRMED", "Top matches all in current correct_doc_ids.", current_correct)


def main() -> int:
    from openai import OpenAI
    from app import app
    from config import Config
    from models import db, Document, DocumentChunk

    eval_path = _PROJECT_ROOT / "eval" / "maize_eval_v1.jsonl"
    rows = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} rows from {eval_path}", flush=True)

    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    keyword_model = "gpt-4o-mini"

    out_path = Path("/tmp/retro_grounding_report.md")
    sections = []
    verdict_counter = Counter()

    with app.app_context():
        for i, r in enumerate(rows, 1):
            rid = r["row_id"]
            query = r["query"]
            prior = r.get("prior_turns") or []
            ta_id = r["ta_id"]
            current = r["correct_doc_ids"]

            print(f"[{i:3}/{len(rows)}] {rid}", flush=True)

            kws = extract_keywords(client, keyword_model, query, prior)
            all_kw = [str(k) for k in ((kws.get("concept_keywords") or []) + (kws.get("structural_markers") or []))]
            # Use context_keywords for multi-turn rows to anchor what was being discussed
            search_kw = all_kw + [str(k) for k in (kws.get("context_keywords") or [])]

            search_results = chunk_search(db, DocumentChunk, Document, ta_id, search_kw)
            ranked = rank_docs(search_results, current)
            verdict, reasoning, proposed = verdict_for_row(ranked, current, search_kw)
            verdict_counter[verdict[:1]] += 1  # tally by emoji

            # Build markdown section
            sec = [f"## {rid}",
                   f"- **TA**: `{ta_id}`",
                   f"- **Failure type**: `{r['failure_type_target']}`",
                   f"- **Source**: `{r['source']}`",
                   f"- **Query**: {query}",
                   f"- **Prior turns**: {len(prior) // 2}",
                   f"- **Current correct_doc_ids**: `{current}`",
                   f"- **Keywords**: concept={kws.get('concept_keywords')}, structural={kws.get('structural_markers')}, context={kws.get('context_keywords')}",
                   f"- **Top chunk matches**:"]
            for fname, n_kw, n_chunks, in_current, excerpts in ranked[:5]:
                mark = "✓" if in_current else " "
                sec.append(f"  - {mark} `{fname}` ({n_kw}/{len([k for k in search_kw if k.strip()])} keywords, {n_chunks} chunks)")
                for kw_matched, ex in excerpts[:1]:
                    sec.append(f"    - matched `{kw_matched}`: \"{ex}\"")
            if not ranked:
                sec.append("  - (no chunk matches found)")
            sec.append(f"- **Verdict**: {verdict}")
            sec.append(f"- **Reasoning**: {reasoning}")
            if verdict != "✓ CONFIRMED":
                sec.append(f"- **Proposed correct_doc_ids**: `{proposed}`")
            sec.append("")
            sections.append("\n".join(sec))

    header = [f"# Retro-grounding report ({len(rows)} rows)\n",
              "Verdict tally:",
              f"  - ✓ CONFIRMED: {verdict_counter.get('✓', 0)}",
              f"  - ➕ EXPAND:    {verdict_counter.get('➕', 0)}",
              f"  - ✏️ CORRECT:   {verdict_counter.get('✏', 0)}",
              f"  - ⚠️ WEAK:      {verdict_counter.get('⚠', 0)}",
              "",
              "**Action**: review each non-CONFIRMED row's chunk excerpts. If the proposed correct_doc_ids look right, confirm. If a chunk excerpt looks wrong/irrelevant, flag for re-derivation.",
              "",
              "---", ""]

    out_path.write_text("\n".join(header) + "\n\n".join(sections))
    print(f"\nWrote report to {out_path}")
    print(f"Tally: {dict(verdict_counter)}")


if __name__ == "__main__":
    sys.exit(main() or 0)
