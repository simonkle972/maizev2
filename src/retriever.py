import os
import logging
from openai import OpenAI
from sqlalchemy import text
from config import Config

logger = logging.getLogger(__name__)

_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
    return _openai_client

def analyze_query(query: str) -> dict:
    query_lower = query.lower()
    
    analysis = {
        "doc_type_filter": None,
        "assignment_filter": None,
        "unit_filter": None,
        "is_conceptual": False
    }
    
    import re
    
    hw_patterns = [
        r'homework\s*(\d+)',
        r'hw\s*(\d+)',
        r'assignment\s*(\d+)',
        r'problem\s*set\s*(\d+)',
        r'ps\s*(\d+)'
    ]
    
    for pattern in hw_patterns:
        match = re.search(pattern, query_lower)
        if match:
            analysis["doc_type_filter"] = "homework"
            analysis["assignment_filter"] = match.group(1)
            break
    
    exam_patterns = [
        r'(final|midterm)\s*(exam)?',
        r'exam\s*(\d+)?',
        r'quiz\s*(\d+)?'
    ]
    
    for pattern in exam_patterns:
        match = re.search(pattern, query_lower)
        if match:
            analysis["doc_type_filter"] = "exam"
            break
    
    lecture_patterns = [
        r'lecture\s*(\d+)',
        r'class\s*(\d+)',
        r'week\s*(\d+)',
        r'module\s*(\d+)',
        r'session\s*(\d+)'
    ]
    
    for pattern in lecture_patterns:
        match = re.search(pattern, query_lower)
        if match:
            analysis["doc_type_filter"] = "lecture"
            analysis["unit_filter"] = int(match.group(1))
            break
    
    conceptual_markers = [
        'what is', 'what are', 'explain', 'why', 'how does', 
        'concept', 'definition', 'meaning', 'understand',
        'difference between', 'compare', 'relationship'
    ]
    
    if any(marker in query_lower for marker in conceptual_markers):
        analysis["is_conceptual"] = True
    
    return analysis

def retrieve_context(ta_id: str, query: str, top_k: int = 8) -> tuple:
    """
    Retrieve relevant chunks for a query.
    
    Returns:
        tuple: (chunks, diagnostics)
            - chunks: list of chunk dicts with text, score, file_name, etc.
            - diagnostics: dict with retrieval metrics for logging
    """
    from models import db, DocumentChunk
    from sqlalchemy import func, literal
    from pgvector.sqlalchemy import Vector
    import json
    
    diagnostics = {
        "total_chunks_in_ta": 0,
        "filters_applied": None,
        "filter_match_count": 0,
        "retrieval_method": "unfiltered",
        "is_conceptual": False,
        "score_top1": 0.0,
        "score_top8": 0.0,
        "score_mean": 0.0,
        "score_spread": 0.0,
        "chunk_scores": [],
        "chunk_sources_detail": []
    }
    
    total_chunks = DocumentChunk.query.filter_by(ta_id=ta_id).count()
    diagnostics["total_chunks_in_ta"] = total_chunks
    
    if total_chunks == 0:
        logger.warning(f"No indexed chunks found for TA: {ta_id}")
        return [], diagnostics
    
    client = get_openai_client()
    
    response = client.embeddings.create(
        model=Config.EMBEDDING_MODEL,
        input=query
    )
    query_embedding = response.data[0].embedding
    
    query_analysis = analyze_query(query)
    diagnostics["is_conceptual"] = query_analysis.get("is_conceptual", False)
    
    base_query = db.session.query(
        DocumentChunk.chunk_text,
        DocumentChunk.file_name,
        DocumentChunk.doc_type,
        DocumentChunk.assignment_number,
        DocumentChunk.instructional_unit_number,
        DocumentChunk.instructional_unit_label,
        (1 - DocumentChunk.embedding.cosine_distance(query_embedding)).label('score')
    ).filter(DocumentChunk.ta_id == ta_id)
    
    filtered_query = base_query
    has_filters = False
    filter_description = []
    
    if query_analysis["doc_type_filter"] and query_analysis["assignment_filter"]:
        filtered_query = base_query.filter(
            DocumentChunk.doc_type == query_analysis["doc_type_filter"],
            DocumentChunk.assignment_number == query_analysis["assignment_filter"]
        )
        has_filters = True
        filter_description = [f"doc_type={query_analysis['doc_type_filter']}", f"assignment={query_analysis['assignment_filter']}"]
    elif query_analysis["doc_type_filter"] and query_analysis["unit_filter"]:
        filtered_query = base_query.filter(
            DocumentChunk.doc_type == query_analysis["doc_type_filter"],
            DocumentChunk.instructional_unit_number == query_analysis["unit_filter"]
        )
        has_filters = True
        filter_description = [f"doc_type={query_analysis['doc_type_filter']}", f"unit={query_analysis['unit_filter']}"]
    elif query_analysis["doc_type_filter"]:
        filtered_query = base_query.filter(
            DocumentChunk.doc_type == query_analysis["doc_type_filter"]
        )
        has_filters = True
        filter_description = [f"doc_type={query_analysis['doc_type_filter']}"]
    
    if has_filters:
        diagnostics["filters_applied"] = ", ".join(filter_description)
        filter_match_count = filtered_query.count()
        diagnostics["filter_match_count"] = filter_match_count
        logger.info(f"[{ta_id}] Filters applied: {diagnostics['filters_applied']}, matching chunks: {filter_match_count}")
    
    used_fallback = False
    try:
        results = filtered_query.order_by(
            DocumentChunk.embedding.cosine_distance(query_embedding)
        ).limit(top_k).all()
        
        if not results and has_filters:
            logger.info(f"[{ta_id}] No results with filter, falling back to unfiltered search")
            results = base_query.order_by(
                DocumentChunk.embedding.cosine_distance(query_embedding)
            ).limit(top_k).all()
            used_fallback = True
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        results = base_query.order_by(
            DocumentChunk.embedding.cosine_distance(query_embedding)
        ).limit(top_k).all()
        used_fallback = True
    
    if has_filters and not used_fallback:
        diagnostics["retrieval_method"] = "filtered"
    elif has_filters and used_fallback:
        diagnostics["retrieval_method"] = "fallback_unfiltered"
    else:
        diagnostics["retrieval_method"] = "unfiltered"
    
    chunks = []
    scores = []
    sources_detail = []
    
    for i, row in enumerate(results):
        score = float(row.score) if row.score else 0.0
        scores.append(score)
        
        source_info = f"{row.file_name}|{row.doc_type or 'unknown'}|unit:{row.instructional_unit_number or 'N/A'}"
        sources_detail.append(source_info)
        
        chunks.append({
            "text": row.chunk_text,
            "score": score,
            "file_name": row.file_name or "unknown",
            "doc_type": row.doc_type or "other",
            "metadata": {
                "assignment_number": row.assignment_number,
                "instructional_unit_number": row.instructional_unit_number,
                "instructional_unit_label": row.instructional_unit_label
            }
        })
    
    if scores:
        diagnostics["score_top1"] = round(scores[0], 4) if len(scores) > 0 else 0.0
        diagnostics["score_top8"] = round(scores[-1], 4) if len(scores) >= top_k else round(scores[-1], 4) if scores else 0.0
        diagnostics["score_mean"] = round(sum(scores) / len(scores), 4)
        diagnostics["score_spread"] = round(scores[0] - scores[-1], 4) if len(scores) > 1 else 0.0
        diagnostics["chunk_scores"] = [round(s, 4) for s in scores]
        diagnostics["chunk_sources_detail"] = sources_detail
    
    logger.info(f"[{ta_id}] Retrieved {len(chunks)} chunks | method={diagnostics['retrieval_method']} | scores: top1={diagnostics['score_top1']}, top8={diagnostics['score_top8']}, spread={diagnostics['score_spread']}")
    
    return chunks, diagnostics
