"""
Analytics computation module for Maize TA.
Provides usage stats, topic clustering, and challenge identification
for the professor-facing analytics dashboard.
"""

import logging
import json
import re
import time
from datetime import datetime, timedelta
from collections import Counter

from models import db, ChatMessage, ChatSession
from config import Config

logger = logging.getLogger(__name__)

# Simple in-memory cache for topic clustering results (TTL: 24h)
_topic_cache = {}
_TOPIC_CACHE_TTL = 86400  # 24 hours


def get_usage_stats(ta_id, start_date=None, end_date=None):
    """
    Get summary statistics for a TA's chat usage.

    Returns dict with:
    - total_interactions: lifetime count of student queries
    - period_interactions: count within the date range
    - unique_sessions: distinct chat sessions in range
    - peak_day: date string of highest-volume day
    - peak_hour: string like "10-11 PM"
    """
    # Base query: student messages for this TA
    base = db.session.query(ChatMessage).join(ChatSession).filter(
        ChatSession.ta_id == ta_id,
        ChatMessage.role == 'user'
    )

    # Total lifetime interactions
    total_interactions = base.count()

    # Period-filtered query
    period_query = base
    if start_date:
        period_query = period_query.filter(ChatMessage.created_at >= start_date)
    if end_date:
        period_query = period_query.filter(ChatMessage.created_at <= end_date)

    period_interactions = period_query.count()

    # Unique sessions in range
    session_query = db.session.query(db.func.count(db.distinct(ChatSession.id))).filter(
        ChatSession.ta_id == ta_id
    )
    if start_date:
        session_query = session_query.filter(ChatSession.created_at >= start_date)
    if end_date:
        session_query = session_query.filter(ChatSession.created_at <= end_date)
    unique_sessions = session_query.scalar() or 0

    # Peak day
    peak_day_result = db.session.query(
        db.func.date(ChatMessage.created_at).label('day'),
        db.func.count(ChatMessage.id).label('cnt')
    ).join(ChatSession).filter(
        ChatSession.ta_id == ta_id,
        ChatMessage.role == 'user'
    )
    if start_date:
        peak_day_result = peak_day_result.filter(ChatMessage.created_at >= start_date)
    if end_date:
        peak_day_result = peak_day_result.filter(ChatMessage.created_at <= end_date)
    peak_day_result = peak_day_result.group_by('day').order_by(db.desc('cnt')).first()

    peak_day = None
    peak_day_count = 0
    if peak_day_result:
        peak_day = str(peak_day_result.day)
        peak_day_count = peak_day_result.cnt

    # Peak hour
    peak_hour = get_peak_hour(ta_id, start_date, end_date)

    return {
        'total_interactions': total_interactions,
        'period_interactions': period_interactions,
        'unique_sessions': unique_sessions,
        'peak_day': peak_day,
        'peak_day_count': peak_day_count,
        'peak_hour': peak_hour,
    }


def get_usage_over_time(ta_id, start_date=None, end_date=None):
    """
    Get daily interaction counts for a time-series chart.

    Returns list of {"date": "2026-03-15", "count": 12} dicts.
    """
    query = db.session.query(
        db.func.date(ChatMessage.created_at).label('day'),
        db.func.count(ChatMessage.id).label('cnt')
    ).join(ChatSession).filter(
        ChatSession.ta_id == ta_id,
        ChatMessage.role == 'user'
    )
    if start_date:
        query = query.filter(ChatMessage.created_at >= start_date)
    if end_date:
        query = query.filter(ChatMessage.created_at <= end_date)

    results = query.group_by('day').order_by('day').all()

    return [{"date": str(row.day), "count": row.cnt} for row in results]


def get_peak_hour(ta_id, start_date=None, end_date=None):
    """
    Get the most common hour of day for student queries.

    Returns string like "10-11 PM" or None if no data.
    """
    query = db.session.query(
        db.func.extract('hour', ChatMessage.created_at).label('hour'),
        db.func.count(ChatMessage.id).label('cnt')
    ).join(ChatSession).filter(
        ChatSession.ta_id == ta_id,
        ChatMessage.role == 'user'
    )
    if start_date:
        query = query.filter(ChatMessage.created_at >= start_date)
    if end_date:
        query = query.filter(ChatMessage.created_at <= end_date)

    result = query.group_by('hour').order_by(db.desc('cnt')).first()

    if not result:
        return None

    hour = int(result.hour)
    next_hour = (hour + 1) % 24

    def fmt_hour(h):
        if h == 0:
            return "12 AM"
        elif h < 12:
            return f"{h} AM"
        elif h == 12:
            return "12 PM"
        else:
            return f"{h - 12} PM"

    return f"{fmt_hour(hour)}-{fmt_hour(next_hour)}"


def get_top_challenges(ta_id, start_date=None, end_date=None, limit=15):
    """
    Identify which assignments/problems generate the most questions.
    Uses regex parsing (same as retriever's analyze_query) — no LLM needed.

    Returns list of {"document": "Homework 1", "problem": "Problem 2", "count": 47, "percentage": 18.3}
    """
    from src.retriever import extract_problem_reference

    # Fetch student messages in range
    query = db.session.query(ChatMessage.content).join(ChatSession).filter(
        ChatSession.ta_id == ta_id,
        ChatMessage.role == 'user'
    )
    if start_date:
        query = query.filter(ChatMessage.created_at >= start_date)
    if end_date:
        query = query.filter(ChatMessage.created_at <= end_date)

    messages = query.all()
    if not messages:
        return []

    # Parse each query for document and problem references
    challenge_counter = Counter()
    total_with_ref = 0

    # Document type patterns (same as retriever's analyze_query)
    doc_patterns = [
        (r'(?:homework|hw|assignment|problem\s*set|pset|ps)\s*#?\s*(\d+)', 'Homework'),
        (r'(?:lecture|lec)\s*#?\s*(\d+)', 'Lecture'),
        (r'(?:exam|midterm|final|quiz)\s*#?\s*(\d+)?', 'Exam'),
        (r'(?:reading|chapter|ch)\s*#?\s*(\d+)', 'Reading'),
    ]

    for (content,) in messages:
        query_lower = content.lower()

        # Extract document reference
        doc_label = None
        for pattern, doc_type in doc_patterns:
            match = re.search(pattern, query_lower)
            if match:
                num = match.group(1) if match.group(1) else ""
                doc_label = f"{doc_type} {num}".strip() if num else doc_type
                break

        # Extract problem reference
        prob_ref = extract_problem_reference(content)
        prob_label = None
        if prob_ref and prob_ref.get('problem_number'):
            sub = prob_ref.get('sub_part', '')
            prob_label = f"Problem {prob_ref['problem_number']}{sub or ''}"

        if doc_label or prob_label:
            key = (doc_label or "(General)", prob_label or "(General)")
            challenge_counter[key] += 1
            total_with_ref += 1

    if not challenge_counter:
        return []

    # Build result sorted by count
    results = []
    for (doc, prob), count in challenge_counter.most_common(limit):
        results.append({
            "document": doc,
            "problem": prob,
            "count": count,
            "percentage": round(count / total_with_ref * 100, 1) if total_with_ref > 0 else 0,
        })

    return results


def cluster_topics(ta_id, start_date=None, end_date=None):
    """
    Cluster student queries into 5-10 thematic topics using LLM.
    Results are cached for 24h per (ta_id, date_range).

    Returns list of {"label": "Net Present Value calculations", "count": 34, "percentage": 21.5}
    """
    # Check cache
    cache_key = f"{ta_id}:{start_date}:{end_date}"
    if cache_key in _topic_cache:
        cached_at, cached_result = _topic_cache[cache_key]
        if time.time() - cached_at < _TOPIC_CACHE_TTL:
            logger.info(f"[{ta_id}] Topic clustering cache hit for {cache_key}")
            return cached_result

    # Fetch student messages in range
    query = db.session.query(ChatMessage.content).join(ChatSession).filter(
        ChatSession.ta_id == ta_id,
        ChatMessage.role == 'user'
    )
    if start_date:
        query = query.filter(ChatMessage.created_at >= start_date)
    if end_date:
        query = query.filter(ChatMessage.created_at <= end_date)

    messages = [row.content for row in query.all()]

    if len(messages) < 3:
        return []

    # Cap at 500 queries, sample randomly if more
    import random
    if len(messages) > 500:
        messages = random.sample(messages, 500)

    # Build prompt
    numbered_queries = "\n".join(f"{i+1}. {q[:200]}" for i, q in enumerate(messages))

    prompt = f"""You are analyzing student questions from a college course. Below are {len(messages)} student queries to an AI teaching assistant. Group them into 5-10 thematic topic clusters.

For each cluster, provide:
- A descriptive, course-specific label (e.g., "Net Present Value calculations", not "Topic A")
- The count of queries in that cluster

Every query must be assigned to exactly one cluster. Return ONLY valid JSON — no markdown, no explanation:
[{{"label": "...", "count": N}}, ...]

Student queries:
{numbered_queries}"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=Config.OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cheap for clustering
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

        clusters = json.loads(raw)

        # Calculate percentages
        total = sum(c.get("count", 0) for c in clusters)
        for c in clusters:
            c["percentage"] = round(c["count"] / total * 100, 1) if total > 0 else 0

        # Sort by count descending
        clusters.sort(key=lambda x: x.get("count", 0), reverse=True)

        # Cache result
        _topic_cache[cache_key] = (time.time(), clusters)
        logger.info(f"[{ta_id}] Topic clustering complete: {len(clusters)} clusters from {len(messages)} queries")

        return clusters

    except Exception as e:
        logger.error(f"[{ta_id}] Topic clustering failed: {e}")
        return []
