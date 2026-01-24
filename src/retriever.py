import os
import logging
import json
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

def rerank_chunks(query: str, chunks: list, top_k: int = 8) -> tuple:
    """
    Use LLM to rerank chunks based on relevance to query.
    
    Returns:
        tuple: (reranked_chunks, rerank_scores)
            - reranked_chunks: list of top_k chunks after reranking
            - rerank_scores: list of relevance scores from LLM
    """
    if not chunks or len(chunks) <= top_k:
        return chunks, [c.get('score', 0) for c in chunks]
    
    client = get_openai_client()
    
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        text_preview = chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text']
        chunk_summaries.append(f"[{i}] (from {chunk['file_name']}): {text_preview}")
    
    chunks_text = "\n\n".join(chunk_summaries)
    
    prompt = f"""You are a relevance scoring system. Given a student's question and a list of document chunks, score each chunk's relevance to answering the question.

STUDENT QUESTION: {query}

DOCUMENT CHUNKS:
{chunks_text}

For each chunk, provide a relevance score from 0.0 to 1.0:
- 1.0: Directly answers or contains the specific information requested
- 0.7-0.9: Highly relevant, contains related information that would help answer
- 0.4-0.6: Somewhat relevant, provides context but not direct answer
- 0.1-0.3: Tangentially related
- 0.0: Not relevant at all

Return ONLY a JSON object with chunk indices as keys and scores as values.
Example: {{"0": 0.9, "1": 0.3, "2": 0.7, ...}}

JSON response:"""

    try:
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500
        )
        
        response_text = (response.choices[0].message.content or "").strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        scores_dict = json.loads(response_text)
        
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            rerank_score = float(scores_dict.get(str(i), 0.0))
            chunk_copy = chunk.copy()
            chunk_copy['rerank_score'] = rerank_score
            chunk_copy['original_score'] = chunk.get('score', 0)
            scored_chunks.append(chunk_copy)
        
        scored_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        top_chunks = scored_chunks[:top_k]
        rerank_scores = [c['rerank_score'] for c in top_chunks]
        
        logger.info(f"Reranked {len(chunks)} chunks, selected top {top_k}. Top rerank scores: {rerank_scores[:3]}")
        
        return top_chunks, rerank_scores
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}, returning original chunks")
        return chunks[:top_k], [c.get('score', 0) for c in chunks[:top_k]]


def analyze_query(query: str, ta_id: str = None) -> dict:
    query_lower = query.lower()
    
    analysis = {
        "doc_type_filter": None,
        "assignment_filter": None,
        "unit_filter": None,
        "content_identifier_filter": None,
        "is_conceptual": False
    }
    
    import re
    
    if ta_id:
        try:
            from models import db, DocumentChunk
            known_identifiers = db.session.query(
                DocumentChunk.content_identifier
            ).filter(
                DocumentChunk.ta_id == ta_id,
                DocumentChunk.content_identifier.isnot(None),
                DocumentChunk.content_identifier != ""
            ).distinct().all()
            
            for (identifier,) in known_identifiers:
                if identifier and identifier.lower() in query_lower:
                    analysis["content_identifier_filter"] = identifier
                    logger.info(f"Detected content_identifier in query: {identifier}")
                    break
        except Exception as e:
            logger.warning(f"Error checking content identifiers: {e}")
    
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

def retrieve_context(ta_id: str, query: str, top_k: int = None) -> tuple:
    """
    Retrieve relevant chunks for a query with reranking.
    
    Flow: Retrieve TOP_K_RETRIEVAL chunks → Rerank with LLM → Return TOP_K_RERANK best chunks
    
    Returns:
        tuple: (chunks, diagnostics)
            - chunks: list of chunk dicts with text, score, file_name, etc.
            - diagnostics: dict with retrieval metrics for logging
    """
    from models import db, DocumentChunk
    from sqlalchemy import func, literal
    from pgvector.sqlalchemy import Vector
    
    initial_k = Config.TOP_K_RETRIEVAL
    final_k = top_k if top_k is not None else Config.TOP_K_RERANK
    
    diagnostics = {
        "total_chunks_in_ta": 0,
        "filters_applied": None,
        "filter_match_count": 0,
        "retrieval_method": "unfiltered",
        "is_conceptual": False,
        "initial_retrieval_k": initial_k,
        "final_k_after_rerank": final_k,
        "score_top1": 0.0,
        "score_top8": 0.0,
        "score_mean": 0.0,
        "score_spread": 0.0,
        "chunk_scores": [],
        "chunk_sources_detail": [],
        "pre_rerank_scores": [],
        "rerank_scores": [],
        "rerank_applied": False
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
    
    query_analysis = analyze_query(query, ta_id)
    diagnostics["is_conceptual"] = query_analysis.get("is_conceptual", False)
    
    base_query = db.session.query(
        DocumentChunk.chunk_text,
        DocumentChunk.file_name,
        DocumentChunk.doc_type,
        DocumentChunk.assignment_number,
        DocumentChunk.instructional_unit_number,
        DocumentChunk.instructional_unit_label,
        DocumentChunk.content_identifier,
        (1 - DocumentChunk.embedding.cosine_distance(query_embedding)).label('score')
    ).filter(DocumentChunk.ta_id == ta_id)
    
    filtered_query = base_query
    has_filters = False
    filter_description = []
    
    if query_analysis.get("content_identifier_filter"):
        content_id = query_analysis["content_identifier_filter"]
        filtered_query = base_query.filter(
            DocumentChunk.content_identifier.ilike(f"%{content_id}%")
        )
        has_filters = True
        filter_description.append(f"content_identifier={content_id}")
    
    if query_analysis["doc_type_filter"] and query_analysis["assignment_filter"]:
        if has_filters:
            filtered_query = filtered_query.filter(
                DocumentChunk.doc_type == query_analysis["doc_type_filter"],
                DocumentChunk.assignment_number == query_analysis["assignment_filter"]
            )
        else:
            filtered_query = base_query.filter(
                DocumentChunk.doc_type == query_analysis["doc_type_filter"],
                DocumentChunk.assignment_number == query_analysis["assignment_filter"]
            )
        has_filters = True
        filter_description.extend([f"doc_type={query_analysis['doc_type_filter']}", f"assignment={query_analysis['assignment_filter']}"])
    elif query_analysis["doc_type_filter"] and query_analysis["unit_filter"]:
        if has_filters:
            filtered_query = filtered_query.filter(
                DocumentChunk.doc_type == query_analysis["doc_type_filter"],
                DocumentChunk.instructional_unit_number == query_analysis["unit_filter"]
            )
        else:
            filtered_query = base_query.filter(
                DocumentChunk.doc_type == query_analysis["doc_type_filter"],
                DocumentChunk.instructional_unit_number == query_analysis["unit_filter"]
            )
        has_filters = True
        filter_description.extend([f"doc_type={query_analysis['doc_type_filter']}", f"unit={query_analysis['unit_filter']}"])
    elif query_analysis["doc_type_filter"]:
        if has_filters:
            filtered_query = filtered_query.filter(
                DocumentChunk.doc_type == query_analysis["doc_type_filter"]
            )
        else:
            filtered_query = base_query.filter(
                DocumentChunk.doc_type == query_analysis["doc_type_filter"]
            )
        has_filters = True
        filter_description.append(f"doc_type={query_analysis['doc_type_filter']}")
    
    if has_filters:
        diagnostics["filters_applied"] = ", ".join(filter_description)
        filter_match_count = filtered_query.count()
        diagnostics["filter_match_count"] = filter_match_count
        logger.info(f"[{ta_id}] Filters applied: {diagnostics['filters_applied']}, matching chunks: {filter_match_count}")
    
    used_fallback = False
    try:
        results = filtered_query.order_by(
            DocumentChunk.embedding.cosine_distance(query_embedding)
        ).limit(initial_k).all()
        
        if not results and has_filters:
            logger.info(f"[{ta_id}] No results with filter, falling back to unfiltered search")
            results = base_query.order_by(
                DocumentChunk.embedding.cosine_distance(query_embedding)
            ).limit(initial_k).all()
            used_fallback = True
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        results = base_query.order_by(
            DocumentChunk.embedding.cosine_distance(query_embedding)
        ).limit(initial_k).all()
        used_fallback = True
    
    if has_filters and not used_fallback:
        diagnostics["retrieval_method"] = "filtered"
    elif has_filters and used_fallback:
        diagnostics["retrieval_method"] = "fallback_unfiltered"
    else:
        diagnostics["retrieval_method"] = "unfiltered"
    
    initial_chunks = []
    pre_rerank_scores = []
    
    for i, row in enumerate(results):
        score = float(row.score) if row.score else 0.0
        pre_rerank_scores.append(score)
        
        initial_chunks.append({
            "text": row.chunk_text,
            "score": score,
            "file_name": row.file_name or "unknown",
            "doc_type": row.doc_type or "other",
            "content_identifier": row.content_identifier or "",
            "metadata": {
                "assignment_number": row.assignment_number,
                "instructional_unit_number": row.instructional_unit_number,
                "instructional_unit_label": row.instructional_unit_label
            }
        })
    
    diagnostics["pre_rerank_scores"] = [round(s, 4) for s in pre_rerank_scores]
    
    logger.info(f"[{ta_id}] Initial retrieval: {len(initial_chunks)} chunks (requesting {initial_k})")
    
    if len(initial_chunks) > final_k:
        reranked_chunks, rerank_scores = rerank_chunks(query, initial_chunks, final_k)
        diagnostics["rerank_applied"] = True
        diagnostics["rerank_scores"] = [round(s, 4) for s in rerank_scores]
        chunks = reranked_chunks
        logger.info(f"[{ta_id}] After reranking: {len(chunks)} chunks selected")
    else:
        chunks = initial_chunks
        diagnostics["rerank_applied"] = False
    
    final_scores = []
    sources_detail = []
    
    for chunk in chunks:
        score = chunk.get('rerank_score', chunk.get('score', 0.0))
        final_scores.append(score)
        source_info = f"{chunk['file_name']}|{chunk['doc_type']}|unit:{chunk['metadata'].get('instructional_unit_number', 'N/A')}"
        sources_detail.append(source_info)
    
    if final_scores:
        diagnostics["score_top1"] = round(final_scores[0], 4) if len(final_scores) > 0 else 0.0
        diagnostics["score_top8"] = round(final_scores[-1], 4) if len(final_scores) >= final_k else (round(final_scores[-1], 4) if final_scores else 0.0)
        diagnostics["score_mean"] = round(sum(final_scores) / len(final_scores), 4)
        diagnostics["score_spread"] = round(final_scores[0] - final_scores[-1], 4) if len(final_scores) > 1 else 0.0
        diagnostics["chunk_scores"] = [round(s, 4) for s in final_scores]
        diagnostics["chunk_sources_detail"] = sources_detail
    
    logger.info(f"[{ta_id}] Final: {len(chunks)} chunks | method={diagnostics['retrieval_method']} | reranked={diagnostics['rerank_applied']} | scores: top1={diagnostics['score_top1']}, spread={diagnostics['score_spread']}")
    
    return chunks, diagnostics
