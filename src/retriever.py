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

def retrieve_context(ta_id: str, query: str, top_k: int = 8) -> list:
    from models import db, DocumentChunk
    
    chunk_count = DocumentChunk.query.filter_by(ta_id=ta_id).count()
    if chunk_count == 0:
        logger.warning(f"No indexed chunks found for TA: {ta_id}")
        return []
    
    client = get_openai_client()
    
    response = client.embeddings.create(
        model=Config.EMBEDDING_MODEL,
        input=query
    )
    query_embedding = response.data[0].embedding
    
    query_analysis = analyze_query(query)
    
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    
    where_clauses = ["ta_id = :ta_id"]
    params = {"ta_id": ta_id, "top_k": top_k, "embedding": embedding_str}
    
    if query_analysis["doc_type_filter"] and query_analysis["assignment_filter"]:
        where_clauses.append("doc_type = :doc_type")
        where_clauses.append("assignment_number = :assignment_number")
        params["doc_type"] = query_analysis["doc_type_filter"]
        params["assignment_number"] = query_analysis["assignment_filter"]
    elif query_analysis["doc_type_filter"] and query_analysis["unit_filter"]:
        where_clauses.append("doc_type = :doc_type")
        where_clauses.append("instructional_unit_number = :unit_number")
        params["doc_type"] = query_analysis["doc_type_filter"]
        params["unit_number"] = query_analysis["unit_filter"]
    elif query_analysis["doc_type_filter"]:
        where_clauses.append("doc_type = :doc_type")
        params["doc_type"] = query_analysis["doc_type_filter"]
    
    where_clause = " AND ".join(where_clauses)
    
    sql = text(f"""
        SELECT 
            chunk_text,
            file_name,
            doc_type,
            assignment_number,
            instructional_unit_number,
            instructional_unit_label,
            1 - (embedding <=> :embedding::vector) as score
        FROM document_chunks
        WHERE {where_clause}
        ORDER BY embedding <=> :embedding::vector
        LIMIT :top_k
    """)
    
    try:
        result = db.session.execute(sql, params)
        rows = result.fetchall()
    except Exception as e:
        logger.error(f"Vector search failed with filter: {e}")
        base_sql = text("""
            SELECT 
                chunk_text,
                file_name,
                doc_type,
                assignment_number,
                instructional_unit_number,
                instructional_unit_label,
                1 - (embedding <=> :embedding::vector) as score
            FROM document_chunks
            WHERE ta_id = :ta_id
            ORDER BY embedding <=> :embedding::vector
            LIMIT :top_k
        """)
        result = db.session.execute(base_sql, {"ta_id": ta_id, "top_k": top_k, "embedding": embedding_str})
        rows = result.fetchall()
    
    if not rows and len(where_clauses) > 1:
        logger.info("No results with filter, falling back to unfiltered search")
        base_sql = text("""
            SELECT 
                chunk_text,
                file_name,
                doc_type,
                assignment_number,
                instructional_unit_number,
                instructional_unit_label,
                1 - (embedding <=> :embedding::vector) as score
            FROM document_chunks
            WHERE ta_id = :ta_id
            ORDER BY embedding <=> :embedding::vector
            LIMIT :top_k
        """)
        result = db.session.execute(base_sql, {"ta_id": ta_id, "top_k": top_k, "embedding": embedding_str})
        rows = result.fetchall()
    
    chunks = []
    for row in rows:
        chunks.append({
            "text": row.chunk_text,
            "score": float(row.score) if row.score else 0.0,
            "file_name": row.file_name or "unknown",
            "doc_type": row.doc_type or "other",
            "metadata": {
                "assignment_number": row.assignment_number,
                "instructional_unit_number": row.instructional_unit_number,
                "instructional_unit_label": row.instructional_unit_label
            }
        })
    
    logger.info(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
    
    return chunks
