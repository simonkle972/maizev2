import os
import logging
import chromadb
from openai import OpenAI
from config import Config

logger = logging.getLogger(__name__)

_chroma_clients = {}
_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
    return _openai_client

def get_chroma_collection(ta_id: str):
    if ta_id not in _chroma_clients:
        chroma_path = os.path.join(Config.CHROMA_DB_PATH, ta_id)
        if not os.path.exists(chroma_path):
            raise ValueError(f"Index not found for TA: {ta_id}")
        
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(f"ta_{ta_id}")
        _chroma_clients[ta_id] = collection
    
    return _chroma_clients[ta_id]

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
    try:
        collection = get_chroma_collection(ta_id)
    except Exception as e:
        logger.error(f"Failed to get collection for {ta_id}: {e}")
        return []
    
    client = get_openai_client()
    
    response = client.embeddings.create(
        model=Config.EMBEDDING_MODEL,
        input=query
    )
    query_embedding = response.data[0].embedding
    
    query_analysis = analyze_query(query)
    
    where_filter = None
    
    if query_analysis["doc_type_filter"] and query_analysis["assignment_filter"]:
        where_filter = {
            "$and": [
                {"doc_type": query_analysis["doc_type_filter"]},
                {"assignment_number": query_analysis["assignment_filter"]}
            ]
        }
    elif query_analysis["doc_type_filter"] and query_analysis["unit_filter"]:
        where_filter = {
            "$and": [
                {"doc_type": query_analysis["doc_type_filter"]},
                {"instructional_unit_number": query_analysis["unit_filter"]}
            ]
        }
    elif query_analysis["doc_type_filter"]:
        where_filter = {"doc_type": query_analysis["doc_type_filter"]}
    
    try:
        if where_filter:
            results = collection.query(
                query_embeddings=[query_embedding],
                where=where_filter,
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["documents"][0]:
                logger.info(f"No results with filter, falling back to unfiltered search")
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
        else:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
    except Exception as e:
        logger.error(f"ChromaDB query failed: {e}")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
    
    chunks = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0
            
            score = 1 - distance
            
            chunks.append({
                "text": doc,
                "score": score,
                "file_name": metadata.get("file_name", "unknown"),
                "doc_type": metadata.get("doc_type", "other"),
                "metadata": metadata
            })
    
    logger.info(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
    
    return chunks
