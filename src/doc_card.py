"""The document card: one identity per document, decided once, served identically.

Phase 4 (2026-09-16). See attached_assets/maize-phase4-document-card-plan-2026-09-16.md.

Three past mistakes this module exists to prevent: metadata fields nobody reads, stages that
each compose their own version of a document's identity, and metadata the professor cannot
see or correct. So:

- `card_label(doc)`      -- THE identity string. Every stage that names a document uses it.
- `resolve_reference()`  -- THE resolver from a student's words to a document id.
- `sync_chunk_identity()`-- THE denormaliser to document_chunks (doc_label, doc_category,
                            search_tsvector), in SQL, no model call, no re-embedding; runs on
                            indexing and on every professor edit so categories flow live.
- `generate_card()`      -- the one model call at indexing that decides the card, with the
                            sibling documents in view.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

CARD_MODEL_DEFAULT = "gpt-5.6-terra"   # indexing latency is irrelevant; reasoning helps
_STOP = {"the", "a", "an", "of", "and", "for", "to", "pdf", "docx", "pptx", "xlsx"}


# --------------------------------------------------------------------------- label

def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def card_label(doc) -> str:
    """The identity string. `doc` is a Document row, a chunk row/dict, or a dict with the
    card fields. Falls back to the stored name for documents without a card yet."""
    title = (_get(doc, "card_title") or "").strip()
    if not title:
        # chunk rows carry doc_label; anything else falls back to the display/original name
        label = (_get(doc, "doc_label") or "").strip()
        if label:
            return label
        name = _get(doc, "display_name") or _get(doc, "original_filename") or _get(doc, "file_name") or ""
        return re.sub(r"\.[A-Za-z0-9]{2,5}$", "", str(name)).strip()
    part = (_get(doc, "card_part") or "").strip()
    term = (_get(doc, "card_term") or "").strip()
    label = title
    if part and part.lower() not in title.lower():
        label += f", {part}" if re.match(r"(?i)^(part|section|chapter)\b", part) else f", Part {part}"
    if term and term.lower() not in title.lower():
        label += f" ({term})"
    return label


def card_aliases(doc) -> list:
    raw = _get(doc, "card_aliases") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = [raw]
    return [str(a).strip() for a in raw if str(a).strip()]


def category_label(ta, slug: str) -> str:
    for c in (_get(ta, "doc_categories") or []):
        if c.get("slug") == slug:
            return c.get("label") or slug
    return slug or ""


# --------------------------------------------------------------------------- tokens

def _tokens(text: str) -> set:
    toks = set()
    for w in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if w in _STOP:
            continue
        toks.add(w.lstrip("0") or "0" if w.isdigit() else w)
    return toks


# --------------------------------------------------------------------------- resolver

_ROMAN = {"ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}


_LOCATOR = re.compile(r"\b(?:question|questions|problem|problems|q|exercise|exercises|page|pages|slide|slides|step|steps|item|items|no\.?)\s*#?\s*\d+[a-z]?\b", re.I)


def _number_tokens(text: str, drop_locators: bool = False) -> set:
    """Digits, years and Roman numerals -- the tokens that tell siblings apart. A bare "I"
    counts only when written in upper case as its own word. With drop_locators, numbers
    that locate a place INSIDE a document ("question 1", "problem 8", "page 3") are
    ignored: they say where in the document, not which document."""
    if drop_locators:
        text = _LOCATOR.sub(" ", text or "")
    out = set()
    for w in re.findall(r"[A-Za-z0-9]+", text or ""):
        wl = w.lower()
        if wl.isdigit():
            out.add(wl.lstrip("0") or "0")
        elif wl in _ROMAN or (w == "I"):
            out.add(wl)
    return out


def resolve_reference(ta_id: str, text: str, docs=None):
    """Map a student's words for a document ("PS3", "extra problems II", "2019 final") to a
    Document id. Returns (doc_id, label) or (None, None).

    1. Exact title / label / alias match wins.
    2. Otherwise every document is scored by how much of the reference's tokens its label +
       aliases cover (an alias contained whole in the reference adds a bonus). NUMBERS DECIDE:
       when the reference carries a number, year or Roman numeral, only documents that share
       one are candidates -- a bare "final exam" alias must never pick a year the student did
       not say. The best candidate must cover at least half the reference and be unique;
       if the only tie is a document against its own solutions, the problem document wins
       (user policy: both are valid material; the problem is what the student works on).
       Otherwise no guess: a wrong document is worse than the normal search.
    """
    if not text or not str(text).strip():
        return None, None
    ref = str(text).strip()
    ref_l = ref.lower()
    if docs is None:
        from src.retriever import _docs_query
        docs = _docs_query(ta_id).all()
    ref_nums = _number_tokens(ref, drop_locators=True)
    rt = _tokens(_LOCATOR.sub(' ', ref))   # 'question 1' locates within a document; not identity
    cands = []
    for d in docs:
        label = card_label(d)
        names = {label.lower(), (d.card_title or "").lower(), (d.display_name or "").lower(),
                 re.sub(r"\.[A-Za-z0-9]{2,5}$", "", d.original_filename or "").lower()} - {""}
        aliases = [a.lower() for a in card_aliases(d)]
        cands.append((d, label, names, aliases))
    # 1. exact
    for d, label, names, aliases in cands:
        if ref_l in names or ref_l in aliases:
            return d.id, label
    if not rt:
        return None, None
    # 2. scored, numbers must agree
    scored = []
    for d, label, names, aliases in cands:
        dt = _tokens(" ".join([label, d.card_number or "", d.card_part or "", d.card_term or ""] + aliases))
        if not dt:
            continue
        if ref_nums and not (ref_nums & _number_tokens(" ".join([label, d.card_number or "", d.card_part or "", d.card_term or ""] + aliases))):
            continue
        coverage = len(rt & dt) / len(rt)
        bonus = 0.25 if any(a and re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", ref_l) for a in aliases) else 0.0
        scored.append((round(coverage + bonus, 4), d.id, label, (d.doc_category or "").lower(), d.card_title or ""))
    if not scored:
        return None, None
    scored.sort(key=lambda x: (-x[0], x[1]))
    best = scored[0]
    if best[0] < 0.5:
        return None, None
    tied = [x for x in scored if x[0] == best[0]]
    if len(tied) == 1:
        return best[1], best[2]
    # tie-break: a document vs its own solutions -> the problem document
    def _base(title):
        t = re.sub(r"\b(solutions?|answer key|answers|key|suggested)\b", " ", title.lower())
        return " ".join(re.findall(r"[a-z0-9]+", t))
    bases = {_base(x[4]) for x in tied}
    if len(bases) == 1:
        non_sol = [x for x in tied if x[3] != "solutions" and not re.search(r"\b(solutions?|answer key)\b", x[4].lower())]
        if len(non_sol) == 1:
            return non_sol[0][1], non_sol[0][2]
    return None, None


# --------------------------------------------------------------------------- chunk sync

def sync_chunk_identity(doc, db=None) -> int:
    """Write the document's identity onto its chunks: doc_label, doc_category, and the
    chunk-level lexical index (label + aliases + section path + chunk text). Pure SQL --
    no model call, no re-embedding -- so a professor's edit or a new category flows into
    retrieval immediately. The identity prefix inside the dense embedding is the one thing
    that goes stale until the next re-index; the lexical index covers exact tokens."""
    if db is None:
        from models import db as _db
        db = _db
    from sqlalchemy import text as _sql
    label = card_label(doc)
    aliases = " ".join(card_aliases(doc))
    res = db.session.execute(_sql("""
        UPDATE document_chunks
           SET doc_label = :label,
               doc_category = :cat,
               search_tsvector = to_tsvector('english',
                   :label || ' ' || :aliases || ' ' ||
                   coalesce((SELECT string_agg(x, ' ') FROM jsonb_array_elements_text(
                       CASE WHEN jsonb_typeof(section_path::jsonb) = 'array' THEN section_path::jsonb ELSE '[]'::jsonb END) x), '')
                   || ' ' || chunk_text)
         WHERE document_id = :doc_id
    """), {"label": label[:256], "cat": doc.doc_category, "aliases": aliases, "doc_id": doc.id})
    return res.rowcount or 0


# --------------------------------------------------------------------------- generation

_CARD_PROMPT = """You are cataloguing one document from a university course so that a teaching assistant can find it when a student refers to it.

COURSE: {course_name}

THE DOCUMENT
Filename: {filename}
--- beginning of the document ---
{head}
--- end of the document ---
{tail}

THE OTHER DOCUMENTS IN THIS COURSE (title or filename · kind · number · part · term):
{siblings}

ALLOWED KINDS (use the slug):
{categories}

Decide this document's card. Return JSON with exactly these keys:
- "title": the name a student would use for it, short and specific, e.g. "Problem Set 3", "Practice Problems 1", "Extra Problems II", "Final Exam, Fall 2019", "Lecture 7: Hypothesis Testing", "Quiz 3 Solutions". Include the number in the title when there is one. Never copy a raw filename with underscores or codes.
- "kind": one slug from ALLOWED KINDS.
- "number": the document's own sequence number as a string ("3", "11-20" for a range), or "" if it has none. Copy it from the document or the filename; never infer it from the siblings. An edition, a year, a page count or a version is NOT a number.
- "part": "I", "II", "Part 2", "Section 2" etc. ONLY when the document is one part of a numbered or lettered series (Extra Problems I / II, Exam Part 1 / Part 2, Section 1 / Section 2). Otherwise "". Words like "Session", "Edition", "Version" are not parts.
- "term": the term or year if stated ("2025B", "Fall 2019", "Spring 2026"), else "".
- "aliases": every surface form a student might type for THIS document, lowercase: abbreviations ("ps3", "pset 3", "hw3"), the filename's stem, the title, number forms ("problem set 3", "problem set three"), roman/arabic variants ("part 2", "part ii"). 4 to 12 entries.
- "summary": one or two sentences on what the document covers (topics, problems), for a course overview.
- "reason": one line on how you decided the title and number, citing the document text or filename.

Rules: the siblings are there so that numbers and parts are consistent across the series (if "Practice Problems 2" exists, this is not also "Practice Problems 2" unless the document says so) and so that two files with similar names get distinct, correct titles. If the document is an answer key or solutions, say so in the title ("... Solutions"). If it is a lecture, use its lecture number or its topic. Do not invent a number.

JSON only."""


def generate_card(text: str, filename: str, siblings: list, categories: list, course_name: str = "",
                  model: str = None) -> dict:
    """One model call. `siblings` is a list of strings (one per other document); `categories`
    is the course's [{slug, label}] list. Returns the parsed card dict; raises on failure."""
    from src.retriever import _json_completion, get_openai_client
    from config import Config
    model = model or getattr(Config, "DOC_CARD_MODEL", None) or CARD_MODEL_DEFAULT
    head = (text or "")[:6000]
    tail_src = (text or "")[6000:]
    tail = ("--- last part of the document ---\n" + tail_src[-1000:]) if len(tail_src) > 200 else ""
    prompt = _CARD_PROMPT.format(
        course_name=course_name or "(unknown)", filename=filename, head=head, tail=tail,
        siblings="\n".join(f"- {s}" for s in siblings) if siblings else "- (none yet)",
        categories="\n".join(f'- "{c.get("slug")}" — {c.get("label")}' for c in (categories or [])) or "- other",
    )
    client = get_openai_client()
    r = _json_completion(client, model, prompt, 800)
    raw = r.choices[0].message.content or ""
    try:
        card = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        card = json.loads(m.group(0)) if m else {}
    valid = {c.get("slug") for c in (categories or [])}
    kind = str(card.get("kind") or "").strip()
    if valid and kind not in valid:
        kind = "other" if "other" in valid else next(iter(valid))
    aliases = card.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(",")]
    aliases = sorted({str(a).strip().lower() for a in aliases if str(a).strip()})
    number = str(card.get("number") or "").strip()[:32]
    if number.isdigit():
        number = number.lstrip("0") or "0"          # "05" -> "5" (filenames zero-pad)
    part = str(card.get("part") or "").strip()[:32]
    if part and not re.search(r"\d|\b[ivx]+\b|\b[a-h]\b", part.lower()):
        part = ""                                    # "Session", "Edition": not a part
    return {
        "title": str(card.get("title") or "").strip()[:512],
        "kind": kind,
        "number": number,
        "part": part,
        "term": str(card.get("term") or "").strip()[:64],
        "aliases": aliases[:20],
        "summary": str(card.get("summary") or "").strip(),
        "reason": str(card.get("reason") or "").strip()[:300],
        "model": model,
    }


def sibling_lines(ta_id: str, exclude_doc_id=None, docs=None) -> list:
    """The sibling list the card model sees: existing cards where present, filenames otherwise."""
    if docs is None:
        from src.retriever import _docs_query
        docs = _docs_query(ta_id).all()
    out = []
    for d in docs:
        if d.id == exclude_doc_id:
            continue
        name = card_label(d)
        out.append(f"{name} · {d.doc_category or '?'} · {d.card_number or ''} · {d.card_part or ''} · {d.card_term or ''}"
                   + (f"  [file: {d.original_filename}]" if d.card_title else ""))
    return out


def apply_card(doc, card: dict, overwrite_professor: bool = False) -> bool:
    """Write a generated card onto a Document row. A professor's edit (card_source ==
    'professor') keeps title / kind / number / part / term unless overwrite_professor;
    aliases and summary are background fields and are always refreshed."""
    professor_owned = (doc.card_source == "professor") and not overwrite_professor
    if not professor_owned:
        doc.card_title = card.get("title") or doc.card_title
        if card.get("kind"):
            doc.doc_category = card["kind"]
        doc.card_number = card.get("number") or None
        doc.card_part = card.get("part") or None
        doc.card_term = card.get("term") or None
        doc.card_source = "llm"
    doc.card_aliases = card.get("aliases") or []
    if card.get("summary"):
        doc.summary = card["summary"]
    doc.card_generated_at = datetime.utcnow()
    return not professor_owned
