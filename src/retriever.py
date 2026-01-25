import os
import re
import logging
from openai import OpenAI
from sqlalchemy import text
from config import Config

logger = logging.getLogger(__name__)


def tokenize_for_matching(text: str) -> set:
    """
    Tokenize text for document matching.
    Returns set of lowercase alphanumeric tokens, removing common words.
    """
    text_lower = text.lower()
    tokens = re.findall(r'[a-z0-9]+', text_lower)
    
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'of', 'in', 'on', 'at', 'to', 'for',
        'with', 'by', 'from', 'as', 'this', 'that', 'these', 'those', 'it',
        'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they',
        'help', 'please', 'understand', 'explain', 'how', 'what', 'why',
        'problem', 'question', 'answer', 'pdf', 'docx', 'xlsx', 'pptx', 'txt'
    }
    
    return set(t for t in tokens if t not in stop_words and len(t) > 1)


def find_matching_documents(query: str, ta_id: str, threshold: float = 0.4) -> list:
    """
    Find documents whose filenames match terms in the query.
    
    Args:
        query: The user's query string
        ta_id: The teaching assistant ID
        threshold: Minimum match score (0-1) to return a document
        
    Returns:
        List of dicts with 'filename', 'score', 'original_filename'
        sorted by score descending. Empty if no matches above threshold.
    """
    from models import Document
    
    documents = Document.query.filter_by(ta_id=ta_id).all()
    if not documents:
        return []
    
    query_tokens = tokenize_for_matching(query)
    if not query_tokens:
        return []
    
    matches = []
    
    for doc in documents:
        filename = doc.original_filename or doc.filename
        filename_tokens = tokenize_for_matching(filename)
        
        if not filename_tokens:
            continue
        
        overlap = query_tokens & filename_tokens
        
        if overlap:
            score = len(overlap) / len(filename_tokens)
            
            long_matches = sum(1 for t in overlap if len(t) >= 4)
            if long_matches > 0:
                score = min(1.0, score + 0.1 * long_matches)
            
            if score >= threshold:
                matches.append({
                    'filename': filename,
                    'score': round(score, 3),
                    'original_filename': doc.original_filename,
                    'matched_tokens': list(overlap)
                })
    
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    if matches:
        logger.info(f"[{ta_id}] Document matching found {len(matches)} matches: {[(m['filename'], m['score']) for m in matches[:3]]}")
    
    return matches

INITIAL_RETRIEVAL_K = 20
FINAL_K = 8


def llm_rerank(query: str, chunks: list, top_k: int = FINAL_K) -> tuple:
    """
    Rerank chunks using GPT-4o-mini for semantic relevance scoring.
    
    The LLM evaluates each chunk's relevance to the specific query,
    understanding context like "problem 2f" vs "problem 3d".
    
    Returns:
        tuple: (reranked_chunks, rerank_info)
    """
    import time
    import json
    
    if not chunks:
        return [], {"reranked": False, "reason": "no_chunks", "method": "none"}
    
    if len(chunks) <= top_k:
        return chunks, {"reranked": False, "reason": "chunks_under_limit", "method": "none"}
    
    rerank_start = time.time()
    
    preview_len = 300 if len(chunks) > 15 else 400
    
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        text_preview = chunk["text"][:preview_len].replace("\n", " ").strip()
        chunk_summaries.append(f"[{i}] {chunk['file_name']}: {text_preview}...")
    
    chunks_text = "\n\n".join(chunk_summaries)
    
    prompt = f"""You are a teaching assistant helping match student queries to course material chunks.

STUDENT QUERY: "{query}"

CANDIDATE CHUNKS (numbered 0 to {len(chunks)-1}):
{chunks_text}

TASK: Score each chunk's relevance to the SPECIFIC query on a scale of 0-10.
- Pay close attention to specific problem/question numbers (e.g., "problem 2f" means ONLY 2f, not 2d or 3f)
- Score 10 = chunk directly contains the answer or exact problem referenced
- Score 5 = chunk is related but doesn't have the specific content
- Score 0 = chunk is irrelevant

Return a JSON object with:
- "scores": array of {{"index": N, "score": N, "reason": "brief reason"}} for each chunk
- "top_indices": array of the {top_k} most relevant chunk indices in order

Example: {{"scores": [{{"index": 0, "score": 8, "reason": "Contains problem 2f setup"}}], "top_indices": [3, 0, 5, 1, 7, 2, 4, 6]}}"""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        rerank_latency_ms = int((time.time() - rerank_start) * 1000)
        
        result_text = response.choices[0].message.content or "{}"
        result = json.loads(result_text)
        
        scores_list = result.get("scores", [])
        top_indices = result.get("top_indices", [])
        
        score_map = {item["index"]: item for item in scores_list}
        
        if top_indices and len(top_indices) >= top_k:
            reranked_indices = [i for i in top_indices[:top_k] if 0 <= i < len(chunks)]
        else:
            scored = [(item["index"], item["score"]) for item in scores_list if 0 <= item.get("index", -1) < len(chunks)]
            scored.sort(key=lambda x: x[1], reverse=True)
            reranked_indices = [idx for idx, _ in scored[:top_k]]
        
        if len(reranked_indices) < top_k:
            used_indices = set(reranked_indices)
            for i in range(len(chunks)):
                if i not in used_indices:
                    reranked_indices.append(i)
                    if len(reranked_indices) >= top_k:
                        break
        
        reranked = []
        llm_scores = []
        reasons = []
        
        for idx in reranked_indices:
            chunk = chunks[idx].copy()
            score_info = score_map.get(idx, {})
            chunk["llm_relevance_score"] = score_info.get("score", 0)
            chunk["llm_reason"] = score_info.get("reason", "")
            reranked.append(chunk)
            llm_scores.append(score_info.get("score", 0))
            reasons.append(score_info.get("reason", "")[:50])
        
        vector_scores = [chunks[idx].get("score", 0) for idx in reranked_indices]
        
        rerank_info = {
            "reranked": True,
            "method": "llm",
            "initial_count": len(chunks),
            "final_count": len(reranked),
            "rerank_latency_ms": rerank_latency_ms,
            "llm_score_top1": llm_scores[0] if llm_scores else 0,
            "llm_score_top8": llm_scores[-1] if llm_scores else 0,
            "vector_score_top1": round(vector_scores[0], 4) if vector_scores else 0,
            "top_reasons": reasons[:3],
            "reranked_indices": reranked_indices[:8]
        }
        
        logger.info(f"LLM reranked {len(chunks)} -> {len(reranked)} chunks in {rerank_latency_ms}ms | top_score={llm_scores[0] if llm_scores else 0}")
        
        return reranked, rerank_info
        
    except Exception as e:
        logger.error(f"LLM rerank failed: {e}, falling back to vector order")
        rerank_latency_ms = int((time.time() - rerank_start) * 1000)
        
        return chunks[:top_k], {
            "reranked": False,
            "method": "fallback_vector",
            "reason": str(e)[:100],
            "rerank_latency_ms": rerank_latency_ms
        }


_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = Config.OPENAI_API_KEY
        if api_key:
            _openai_client = OpenAI(api_key=api_key)
        else:
            raise ValueError("OPENAI_API_KEY not configured")
    return _openai_client

def analyze_query(query: str, ta_id: str = "") -> dict:
    """
    Analyze a query to extract structured filters.
    
    Uses regex patterns for common patterns, then falls back to
    document filename matching if no structured elements found.
    """
    query_lower = query.lower()
    
    analysis = {
        "doc_type_filter": None,
        "assignment_filter": None,
        "unit_filter": None,
        "filename_filter": None,
        "filename_match_score": None,
        "filename_matched_tokens": None,
        "is_conceptual": False
    }
    
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
    
    if ta_id and not analysis["doc_type_filter"] and not analysis["assignment_filter"]:
        doc_matches = find_matching_documents(query, ta_id)
        if doc_matches:
            best_match = doc_matches[0]
            analysis["filename_filter"] = best_match["filename"]
            analysis["filename_match_score"] = best_match["score"]
            analysis["filename_matched_tokens"] = best_match.get("matched_tokens", [])
            logger.info(f"[{ta_id}] Filename match fallback: '{best_match['filename']}' (score={best_match['score']}, tokens={best_match.get('matched_tokens', [])})")
    
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
    
    initial_k = INITIAL_RETRIEVAL_K
    final_k = top_k
    
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
        "chunk_sources_detail": [],
        "rerank_applied": False,
        "rerank_info": None
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
    elif query_analysis["filename_filter"]:
        filtered_query = base_query.filter(
            DocumentChunk.file_name == query_analysis["filename_filter"]
        )
        has_filters = True
        filter_description = [f"filename={query_analysis['filename_filter']}", f"match_score={query_analysis.get('filename_match_score', 'N/A')}"]
    
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
    
    for i, row in enumerate(results):
        score = float(row.score) if row.score else 0.0
        
        initial_chunks.append({
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
    
    chunks, rerank_info = llm_rerank(query, initial_chunks, top_k=final_k)
    diagnostics["rerank_applied"] = rerank_info.get("reranked", False)
    diagnostics["rerank_info"] = rerank_info
    
    if diagnostics["rerank_applied"]:
        scores = [c.get("llm_relevance_score", c.get("score", 0.0)) for c in chunks]
    else:
        scores = [c.get("score", 0.0) for c in chunks]
    
    sources_detail = [
        f"{c['file_name']}|{c['doc_type'] or 'unknown'}|unit:{c['metadata'].get('instructional_unit_number') or 'N/A'}"
        for c in chunks
    ]
    
    if scores:
        diagnostics["score_top1"] = round(scores[0], 4) if len(scores) > 0 else 0.0
        diagnostics["score_top8"] = round(scores[-1], 4) if len(scores) >= final_k else round(scores[-1], 4) if scores else 0.0
        diagnostics["score_mean"] = round(sum(scores) / len(scores), 4)
        diagnostics["score_spread"] = round(scores[0] - scores[-1], 4) if len(scores) > 1 else 0.0
        diagnostics["chunk_scores"] = [round(s, 4) for s in scores]
        diagnostics["chunk_sources_detail"] = sources_detail
    
    logger.info(f"[{ta_id}] Retrieved {len(chunks)} chunks | method={diagnostics['retrieval_method']} | reranked={diagnostics['rerank_applied']} | scores: top1={diagnostics['score_top1']}, spread={diagnostics['score_spread']}")
    
    return chunks, diagnostics
