import os
import re
import logging
import tempfile
from datetime import datetime
from openai import OpenAI
from sqlalchemy import text
from config import Config

logger = logging.getLogger(__name__)


def get_full_document_text(document_id: int) -> tuple:
    """
    Retrieve and extract full text from a document.
    
    Args:
        document_id: The document ID to retrieve
        
    Returns:
        tuple: (text, filename, token_estimate) or (None, None, 0) if failed
    """
    from models import Document
    from src.document_processor import extract_text_from_file
    
    doc = Document.query.get(document_id)
    if not doc:
        logger.warning(f"Document {document_id} not found")
        return None, None, 0
    
    text = None
    
    if doc.file_content:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{doc.file_type}") as tmp_file:
            tmp_file.write(doc.file_content)
            tmp_path = tmp_file.name
        try:
            text, _ = extract_text_from_file(tmp_path)
        finally:
            os.unlink(tmp_path)
    elif doc.storage_path and os.path.exists(doc.storage_path):
        text, _ = extract_text_from_file(doc.storage_path)
    else:
        logger.warning(f"Document {document_id} has no file_content and storage_path is missing or invalid: {doc.storage_path}")
        return None, None, 0
    
    if not text:
        logger.warning(f"Could not extract text from document {document_id} (extraction returned empty)")
        return None, None, 0
    
    token_estimate = len(text) // 4
    
    return text, doc.display_name or doc.original_filename, token_estimate


def find_solution_document(problem_doc_name: str, ta_id: str) -> tuple:
    """
    Find the corresponding solution document for a problem document.
    
    Strategy: Look for a document with "Solution" + the problem document name/number.
    For example:
    - "Practice Problems Set 1" -> "Solution to Practice Problems Set 1"
    - "Problem Set 2" -> "Solution to Problem Set 2"
    - "Homework 3" -> "Homework 3 Solutions"
    
    Args:
        problem_doc_name: The name of the problem document
        ta_id: The TA ID
        
    Returns:
        tuple: (full_text, filename, token_estimate) or (None, None, 0) if not found
    """
    from models import Document
    
    if not problem_doc_name:
        return None, None, 0
    
    problem_lower = problem_doc_name.lower()
    
    # Extract the document number/identifier for flexible matching
    # Match patterns like "Problem Set 1", "Practice Problems Set 2", "Homework 3", "PS1"
    number_match = re.search(r'(?:problem(?:s)?\s*set|homework|hw|ps|pset|practice\s*problems?\s*set?)\s*#?\s*(\d+)', problem_lower)
    doc_number = number_match.group(1) if number_match else None
    
    # Get all documents for this TA
    docs = Document.query.filter_by(ta_id=ta_id).all()
    
    solution_doc = None
    
    for doc in docs:
        doc_name = (doc.display_name or doc.original_filename or "").lower()
        
        # Check if this is a solution document
        is_solution = 'solution' in doc_name
        
        if is_solution:
            # Check if it matches our problem document
            # Method 1: Direct name containment
            # "Solution to Practice Problems Set 1" contains "Practice Problems Set 1"
            problem_name_clean = problem_lower.replace('.pdf', '').replace('.docx', '').strip()
            if problem_name_clean in doc_name:
                solution_doc = doc
                logger.info(f"[{ta_id}] Found solution doc via name containment: {doc.display_name or doc.original_filename}")
                break
            
            # Method 2: Matching document number
            if doc_number:
                sol_number_match = re.search(r'(?:problem(?:s)?\s*set|homework|hw|ps|pset|practice\s*problems?\s*set?)\s*#?\s*(\d+)', doc_name)
                if sol_number_match and sol_number_match.group(1) == doc_number:
                    solution_doc = doc
                    logger.info(f"[{ta_id}] Found solution doc via number match ({doc_number}): {doc.display_name or doc.original_filename}")
                    break
    
    if solution_doc:
        full_text, filename, token_estimate = get_full_document_text(solution_doc.id)
        return full_text, filename, token_estimate
    
    logger.info(f"[{ta_id}] No solution document found for: {problem_doc_name}")
    return None, None, 0


def identify_target_documents(chunks: list, query_analysis: dict, ta_id: str) -> tuple:
    """
    Identify which document(s) should be retrieved in full for fallback.
    
    Strategy:
    1. If there's a filename filter from query analysis, use that document
    2. If there's a doc_type and assignment_number, find matching document
    3. Search by content_title (actual document title from content, not filename)
    4. Otherwise, find the most frequently occurring document in top chunks
    
    Returns:
        tuple: (list of document IDs, identification_method string)
    """
    from models import Document
    import re
    
    if query_analysis.get("filename_filter"):
        filter_value = query_analysis["filename_filter"]
        doc = Document.query.filter_by(
            ta_id=ta_id,
            original_filename=filter_value
        ).first()
        if not doc:
            doc = Document.query.filter_by(
                ta_id=ta_id,
                display_name=filter_value
            ).first()
        if doc:
            logger.info(f"[{ta_id}] Target doc identified via filename_filter: {doc.display_name or doc.original_filename}")
            return [doc.id], "filename_filter"
    
    if query_analysis.get("doc_type_filter") and query_analysis.get("assignment_filter"):
        doc = Document.query.filter_by(
            ta_id=ta_id,
            doc_type=query_analysis["doc_type_filter"],
            assignment_number=query_analysis["assignment_filter"]
        ).first()
        if doc:
            logger.info(f"[{ta_id}] Target doc identified via metadata: {doc.original_filename}")
            return [doc.id], "metadata_filter"
    
    if query_analysis.get("doc_type_filter") and query_analysis.get("unit_filter"):
        doc = Document.query.filter_by(
            ta_id=ta_id,
            doc_type=query_analysis["doc_type_filter"],
            instructional_unit_number=query_analysis["unit_filter"]
        ).first()
        if doc:
            logger.info(f"[{ta_id}] Target doc identified via unit metadata: {doc.original_filename}")
            return [doc.id], "unit_filter"
    
    if query_analysis.get("doc_type_filter"):
        docs = Document.query.filter_by(
            ta_id=ta_id,
            doc_type=query_analysis["doc_type_filter"]
        ).all()
        if len(docs) == 1:
            logger.info(f"[{ta_id}] Target doc identified via single doc_type match: {docs[0].original_filename}")
            return [docs[0].id], "single_doc_type_match"
    
    # Strategy: Search by content_title (handles misnamed files)
    # Extract key terms from query that might match document titles
    query_lower = query_analysis.get("original_query", "").lower() if query_analysis.get("original_query") else ""
    if not query_lower and chunks:
        query_lower = ""
    
    # Look for problem set/assignment number patterns in the query
    ps_match = re.search(r'(?:problem\s*set|self[- ]?study(?:\s*problem\s*set)?)\s*#?\s*(\d+)', query_lower)
    exam_match = re.search(r'(\d{4})?\s*(?:final|midterm|exam)', query_lower)
    
    if ps_match:
        ps_number = ps_match.group(1)
        # Search content_title for matching problem set number
        docs = Document.query.filter_by(ta_id=ta_id).all()
        for doc in docs:
            if doc.content_title:
                title_lower = doc.content_title.lower()
                # Check if content_title contains the same problem set number
                title_match = re.search(r'(?:problem\s*set|self[- ]?study(?:\s*problem\s*set)?)\s*#?\s*(\d+)', title_lower)
                if title_match and title_match.group(1) == ps_number:
                    logger.info(f"[{ta_id}] Target doc identified via content_title match: '{doc.content_title}' (file: {doc.original_filename})")
                    return [doc.id], "content_title_match"
    
    year_filter = query_analysis.get("year_filter")
    if year_filter and query_analysis.get("doc_type_filter") == "exam":
        docs = Document.query.filter_by(ta_id=ta_id, doc_type="exam").all()
        for doc in docs:
            doc_name = doc.display_name or doc.original_filename
            if doc_name and year_filter in doc_name:
                logger.info(f"[{ta_id}] Target doc identified via filename year match: '{doc_name}' (year={year_filter})")
                return [doc.id], "filename_year_match"
    
    if exam_match:
        exam_year = exam_match.group(1) if exam_match.group(1) else None
        docs = Document.query.filter_by(ta_id=ta_id, doc_type="exam").all()
        if exam_year and not year_filter:
            for doc in docs:
                if doc.content_title and exam_year in doc.content_title:
                    logger.info(f"[{ta_id}] Target doc identified via content_title exam match: '{doc.content_title}'")
                    return [doc.id], "content_title_exam_match"
    
    if not chunks:
        logger.warning(f"[{ta_id}] No chunks available for document identification")
        return [], "no_chunks"
    
    doc_counts = {}
    for chunk in chunks[:8]:
        filename = chunk.get("file_name", "")
        if filename:
            doc_counts[filename] = doc_counts.get(filename, 0) + 1
    
    if not doc_counts:
        logger.warning(f"[{ta_id}] No document filenames found in chunks")
        return [], "no_filenames_in_chunks"
    
    top_filename = max(doc_counts.keys(), key=lambda k: doc_counts[k])
    
    doc = Document.query.filter_by(
        ta_id=ta_id,
        original_filename=top_filename
    ).first()
    if not doc:
        doc = Document.query.filter_by(
            ta_id=ta_id,
            display_name=top_filename
        ).first()
    
    if doc:
        logger.info(f"[{ta_id}] Target doc identified via chunk frequency: {doc.display_name or doc.original_filename}")
        return [doc.id], "chunk_frequency"
    
    logger.warning(f"[{ta_id}] Could not find document for filename: {top_filename}")
    return [], "document_not_found"


def assess_retrieval_confidence(chunks: list, rerank_info: dict) -> dict:
    """
    Assess confidence in chunk-based retrieval results.
    
    Returns a dict with:
    - is_low_confidence: bool - True if we should trigger full-doc fallback
    - reason: str - explanation of confidence assessment
    - top_score: float - highest LLM relevance score (or vector score if no rerank)
    - score_spread: float - difference between top and bottom scores
    """
    if not Config.HYBRID_RETRIEVAL_ENABLED:
        return {
            "is_low_confidence": False,
            "reason": "hybrid_disabled",
            "top_score": 0,
            "score_spread": 0
        }
    
    if not chunks:
        return {
            "is_low_confidence": True,
            "reason": "no_chunks_retrieved",
            "top_score": 0,
            "score_spread": 0
        }
    
    if not rerank_info.get("reranked", False):
        vector_scores = [c.get("score", 0) for c in chunks]
        top_vector = vector_scores[0] if vector_scores else 0
        
        if len(chunks) < 5 or top_vector < 0.75:
            return {
                "is_low_confidence": True,
                "reason": f"no_rerank_low_vector_score_{top_vector:.3f}_or_few_chunks_{len(chunks)}",
                "top_score": top_vector,
                "score_spread": 0
            }
        return {
            "is_low_confidence": False,
            "reason": "no_rerank_but_adequate_vector_scores",
            "top_score": top_vector,
            "score_spread": 0
        }
    
    llm_scores = [c.get("llm_relevance_score", 0) for c in chunks]
    top_score = llm_scores[0] if llm_scores else 0
    score_spread = (llm_scores[0] - llm_scores[-1]) if len(llm_scores) > 1 else 0
    
    is_low_confidence = False
    reason = "adequate_confidence"
    
    if top_score < Config.HYBRID_CONFIDENCE_THRESHOLD:
        is_low_confidence = True
        reason = f"top_score_{top_score}_below_threshold_{Config.HYBRID_CONFIDENCE_THRESHOLD}"
    elif score_spread < Config.HYBRID_SCORE_SPREAD_THRESHOLD and top_score < 8:
        is_low_confidence = True
        reason = f"low_spread_{score_spread}_and_moderate_top_{top_score}"
    
    return {
        "is_low_confidence": is_low_confidence,
        "reason": reason,
        "top_score": top_score,
        "score_spread": score_spread
    }


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
- NUMBER FORMAT EQUIVALENCE: Treat Roman numerals and Arabic numbers as equivalent when matching:
  * "Section 1" = "Section I", "Part 2" = "Part II", "Question 3" = "Question III"
  * "a)" = "(a)" = "a." for sub-parts
  * Match content by meaning, not exact formatting
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
            model=Config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
            response_format={"type": "json_object"},
            reasoning_effort=Config.LLM_REASONING_MEDIUM
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

def extract_problem_reference(query: str) -> dict:
    """
    Extract specific problem/question reference from query.
    
    Handles both standard and inverted patterns to support natural speech:
    
    Standard patterns:
    - "problem 2d" -> {"problem_number": "2", "sub_part": "d", "full_ref": "2d"}
    - "question 3a" -> {"problem_number": "3", "sub_part": "a", "full_ref": "3a"}  
    - "problem 7" -> {"problem_number": "7", "sub_part": None, "full_ref": "7"}
    - "part b of problem 5" -> {"problem_number": "5", "sub_part": "b", "full_ref": "5b"}
    
    Inverted patterns (sub-part mentioned first, section/problem later):
    - "question a from section 1" -> {"problem_number": "1", "sub_part": "a", "full_ref": "1a"}
    - "part c) from section 2" -> {"problem_number": "2", "sub_part": "c", "full_ref": "2c"}
    - "question d) from problem 3" -> {"problem_number": "3", "sub_part": "d", "full_ref": "3d"}
    
    Returns empty dict if no specific reference found.
    """
    query_lower = query.lower()
    
    # IMPORTANT: Pattern order matters! More specific patterns must come BEFORE generic ones.
    # Otherwise "part b of problem 5" matches "problem 5" first (without sub_part).
    
    # Pattern 1: "part d of problem 2", "part (a) of question 3" (SPECIFIC - check first)
    match = re.search(r'part\s*\(?([a-z])\)?\s*(?:of|from|in)?\s*(?:problem|question|exercise|prob|q)\s*(\d+)', query_lower)
    if match:
        sub_part = match.group(1)
        problem_num = match.group(2)
        return {
            "problem_number": problem_num,
            "sub_part": sub_part,
            "full_ref": f"{problem_num}{sub_part}"
        }
    
    # Pattern 2: INVERTED - "question a) from section 1", "part c from section 2", "question d) from problem 3"
    # Captures sub-part first, then section/problem number later (SPECIFIC - check before generic)
    match = re.search(r'(?:question|part|q)\s*\(?([a-z])\)?\s*(?:of|from|in)\s*(?:section|problem|question|part|exercise)\s*(\d+)', query_lower)
    if match:
        sub_part = match.group(1)
        problem_num = match.group(2)
        logger.info(f"Extracted inverted reference: section {problem_num}, sub-part {sub_part} -> {problem_num}{sub_part}")
        return {
            "problem_number": problem_num,
            "sub_part": sub_part,
            "full_ref": f"{problem_num}{sub_part}"
        }
    
    # Pattern 3: "section 1 question a", "section 2, part b" (section first, then sub-part)
    match = re.search(r'section\s*(\d+)\s*(?:,?\s*)?(?:question|part|q)\s*\(?([a-z])\)?', query_lower)
    if match:
        problem_num = match.group(1)
        sub_part = match.group(2)
        logger.info(f"Extracted section-first reference: section {problem_num}, sub-part {sub_part} -> {problem_num}{sub_part}")
        return {
            "problem_number": problem_num,
            "sub_part": sub_part,
            "full_ref": f"{problem_num}{sub_part}"
        }
    
    # Pattern 4: Standard - "problem 2d", "question 3a", "exercise 1b" (GENERIC - check after specific patterns)
    # IMPORTANT: Sub-part letter must be IMMEDIATELY after the number (no whitespace)
    # This prevents "Q8 I have" from matching as "8i" where "I" is the next word
    # The sub-part is optional BUT must be directly attached (no \s* before it)
    match = re.search(r'(?:problem|question|exercise|prob|q)\s*(\d+)([a-z])?(?=\s|$|\.|\,|\?|\))', query_lower)
    if match:
        problem_num = match.group(1)
        sub_part = match.group(2)
        full_ref = f"{problem_num}{sub_part}" if sub_part else problem_num
        return {
            "problem_number": problem_num,
            "sub_part": sub_part,
            "full_ref": full_ref
        }
    
    # Pattern 5: "2d", "3a" standalone with problem context words nearby
    # Sub-part must be immediately attached to number (no space between)
    if any(word in query_lower for word in ['problem', 'question', 'exercise', 'help', 'solve', 'answer', 'section']):
        match = re.search(r'(?:^|\s)(\d+)([a-z])(?=\s|$|\.|\,|\?)', query_lower)
        if match:
            problem_num = match.group(1)
            sub_part = match.group(2)
            return {
                "problem_number": problem_num,
                "sub_part": sub_part,
                "full_ref": f"{problem_num}{sub_part}"
            }
    
    return {}


def validate_chunks_contain_reference(chunks: list, problem_ref: dict) -> dict:
    """
    Validate that retrieved chunks actually contain the expected problem reference.
    
    This catches cases where the LLM reranker gives high scores to chunks from
    the wrong problem (e.g., scoring problem 3d highly when query asks for 2d).
    
    Returns:
        dict with:
        - passed: bool - True if validation passed
        - reason: str - explanation
        - matches_found: int - number of chunks containing the reference
    """
    if not problem_ref or not problem_ref.get("full_ref"):
        return {"passed": True, "reason": "no_reference_to_validate", "matches_found": 0}
    
    if not chunks:
        return {"passed": False, "reason": "no_chunks", "matches_found": 0}
    
    full_ref = problem_ref["full_ref"]
    problem_num = problem_ref["problem_number"]
    sub_part = problem_ref.get("sub_part")
    
    # Build patterns to look for in chunk text
    # We need to find evidence that the chunk is about the RIGHT problem/section
    patterns = []
    
    if sub_part:
        # Looking for "2d", "2 d", "2(d)", "2.d", "(d)" when problem/section 2 is mentioned
        # IMPORTANT: All patterns must require BOTH the section/problem number AND the sub-part
        # to avoid false positives (e.g., matching "section 1" when looking for "1a")
        patterns.extend([
            rf'(?:problem|question|exercise|section|prob|q)?\s*{problem_num}\s*[\.\(\s]*{sub_part}[\)\s\.]',  # "2d", "2.d", "2(d)"
            rf'(?:problem|question|exercise|section)\s+{problem_num}[^\d].*\({sub_part}\)',  # "problem 2 ... (d)", "section 1 ... (a)"
            rf'\({sub_part}\)[^\)]*(?:problem|question|section)?\s*{problem_num}',  # "(d) ... problem 2" (reverse order)
            rf'(?:^|\n)\s*{sub_part}\)',  # "d)" at start of line - requires sub-part presence
            rf'section\s+{problem_num}.*{sub_part}\)',  # "section 1 ... a)" - requires sub-part
            rf'section\s+{problem_num}.*\({sub_part}\)',  # "section 1 ... (a)" - requires sub-part
        ])
    else:
        # Just looking for problem/section number (no sub-part to validate)
        patterns.append(rf'(?:problem|question|exercise|section|prob|q)\s*{problem_num}(?:\s|$|\.|\,)')
    
    matches_found = 0
    matching_chunks = []
    
    for i, chunk in enumerate(chunks[:8]):  # Check top 8 chunks
        chunk_text = chunk.get("text", "").lower()
        
        for pattern in patterns:
            if re.search(pattern, chunk_text):
                matches_found += 1
                matching_chunks.append(i)
                break
    
    # Validation passes if at least one of the top chunks contains the reference
    if matches_found > 0:
        return {
            "passed": True,
            "reason": f"found_in_{matches_found}_chunks",
            "matches_found": matches_found,
            "matching_chunk_indices": matching_chunks
        }
    else:
        return {
            "passed": False,
            "reason": f"reference_{full_ref}_not_found_in_top_chunks",
            "matches_found": 0,
            "matching_chunk_indices": []
        }


def analyze_query(query: str, ta_id: str = "") -> dict:
    """
    Analyze a query to extract structured filters.
    
    Uses regex patterns for common patterns, then falls back to
    document filename matching if no structured elements found.
    
    When a specific problem reference with sub-part is detected (e.g., "section 1 question a"),
    sets requires_early_hybrid=True to route directly to full-document mode,
    bypassing the unreliable LLM reranker.
    """
    query_lower = query.lower()
    
    # Extract specific problem reference for validation
    problem_ref = extract_problem_reference(query)
    
    # Determine if this query should use early hybrid routing
    # Specific references with sub-parts (like "1a") need full document context
    # to reliably locate the exact content - chunk-based retrieval is unreliable for these
    requires_early_hybrid = bool(
        problem_ref and 
        problem_ref.get("problem_number") and 
        problem_ref.get("sub_part")
    )
    
    if requires_early_hybrid:
        logger.info(f"[{ta_id}] Early hybrid routing enabled: detected specific reference '{problem_ref.get('full_ref')}'")
    
    analysis = {
        "doc_type_filter": None,
        "assignment_filter": None,
        "unit_filter": None,
        "year_filter": None,
        "filename_filter": None,
        "filename_match_score": None,
        "filename_matched_tokens": None,
        "is_conceptual": False,
        "problem_reference": problem_ref,
        "requires_early_hybrid": requires_early_hybrid,
        "original_query": query  # For content_title matching in document identification
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
    
    year_match = re.search(r'\b(20\d{2})\b', query_lower)
    if year_match:
        analysis["year_filter"] = year_match.group(1)
        logger.info(f"[{ta_id}] Year filter extracted: {analysis['year_filter']}")
    
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


def detect_followup_query(query: str, conversation_history: list = None) -> dict:
    """
    Detect if this query is a follow-up that needs context from previous messages.
    
    Returns:
        dict with:
            - is_followup: bool
            - followup_type: str (answer_submission, clarification, continuation, pronoun_reference)
            - needs_context_enrichment: bool
    """
    query_lower = query.lower().strip()
    query_words = query_lower.split()
    
    result = {
        "is_followup": False,
        "followup_type": None,
        "needs_context_enrichment": False
    }
    
    # No history = can't be a follow-up
    if not conversation_history or len(conversation_history) == 0:
        return result
    
    # 1. Answer submission patterns
    # Match patterns at start of query OR after common prefixes like "ok", "alright", "so"
    answer_patterns = [
        r'^i\s*got\b', r'^my\s*(answer|result|solution|response)\b', r'^i\s*calculated\b',
        r'^i\s*think\s*(it|the\s*answer)\b', r'^the\s*answer\s*is\b', r'^i\s*found\b',
        r'^is\s*it\b', r'^it\s*equals?\b', r'^so\s*(it|the)\b', r'^that\s*gives?\b',
        r'^\d+\.?\d*$',  # Just a number
        r'^[a-z]\)?\.?\s*$',  # Just a letter like "b" or "c)"
        # Flexible patterns with common prefixes
        r'^(ok|okay|alright|so|well|right)\s*(,|\.|\!)?\s*i\s*(got|have|found|calculated)\b',
        r'^(ok|okay|alright|so|well|right)\s*(,|\.|\!)?\s*(my|the)\s*(answer|result|response)\b',
        # Answer with problem reference like "ok for question 8, my answer is..."
        r'^(ok|okay|alright|so|well|right)\s*(,|\.|\!)?\s*(for\s*)?(problem|question|q)\s*\d+[a-z]?\s*(,|\.|\:)?\s*(my|the|i)',
        r'\bi\s*have\s*[pqxyznm]\s*=\s*\d',  # "I have p=3" anywhere in query
        r'\bi\s*got\s*[pqxyznm]\s*=\s*\d',   # "I got x=5" anywhere in query
        r'=\s*\d+\.?\d*\s*(and|,)?\s*[pqxyznm]?\s*=?\s*\d*',  # Multiple variable assignments like "p=3 and q=510"
        r'(plugging|substituting|putting)\s*(in|it|back)',  # "plugging in" type answers
        # Answer with explicit prefix and units (e.g., "my answer is 31 minutes", "I got 5 units/min")
        r'(my\s*(answer|response|result)\s*(is|:)|i\s*(got|found|calculated))\s*\d+\.?\d*\s*(minutes?|mins?|hours?|hrs?|seconds?|secs?|units?|dollars?|percent|%)',
    ]
    for pattern in answer_patterns:
        if re.search(pattern, query_lower):
            result["is_followup"] = True
            result["followup_type"] = "answer_submission"
            result["needs_context_enrichment"] = True
            return result
    
    # 2. Clarification patterns
    clarification_patterns = [
        r'^what\s*do\s*you\s*mean\b', r'^can\s*you\s*explain\b', r'^i\s*don\'?t\s*understand\b',
        r'^why\s*(is|does|do)\b', r'^how\s*(do|does|did)\b', r'^what\s*about\b',
        r'^could\s*you\b', r'^can\s*you\s*clarify\b', r'^explain\s*more\b',
        r'^more\s*(detail|info|explanation)\b'
    ]
    for pattern in clarification_patterns:
        if re.search(pattern, query_lower):
            result["is_followup"] = True
            result["followup_type"] = "clarification"
            result["needs_context_enrichment"] = True
            return result
    
    # 3. Short query with pronouns (likely needs context)
    pronoun_refs = ['it', 'this', 'that', 'these', 'those', 'them', 'they']
    if len(query_words) <= 10:
        for pronoun in pronoun_refs:
            if pronoun in query_words:
                result["is_followup"] = True
                result["followup_type"] = "pronoun_reference"
                result["needs_context_enrichment"] = True
                return result
    
    # 4. Very short queries (likely continuation)
    if len(query_words) <= 5 and not any(kw in query_lower for kw in ['what is', 'explain', 'help with']):
        result["is_followup"] = True
        result["followup_type"] = "continuation"
        result["needs_context_enrichment"] = True
        return result
    
    # 5. Part reference without full context (e.g., "part b" or "what about 2c")
    part_patterns = [r'^(part|section|question)\s*[a-z]?\d*[a-z]?\b', r'^[a-z]?\d+[a-z]?\)?$', r'^(and|what about|now)\s*(part|section)?\s*[a-z]?\d*[a-z]?\b']
    for pattern in part_patterns:
        if re.search(pattern, query_lower):
            result["is_followup"] = True
            result["followup_type"] = "continuation"
            result["needs_context_enrichment"] = True
            return result
    
    return result


def extract_context_from_history(conversation_history: list, max_messages: int = 4) -> dict:
    """
    Extract relevant context from conversation history for query enrichment.
    
    Returns:
        dict with:
            - topic_summary: str (key topic/problem being discussed)
            - document_reference: str (any specific document mentioned)
            - problem_reference: str (any specific problem number/part)
            - last_assistant_response: str (truncated)
    """
    context = {
        "topic_summary": None,
        "document_reference": None,
        "problem_reference": None,
        "last_assistant_response": None
    }
    
    if not conversation_history:
        return context
    
    # Get recent messages (most recent first for analysis)
    recent = conversation_history[-max_messages:] if len(conversation_history) > max_messages else conversation_history
    
    # Find the most recent user question that started the topic
    for msg in reversed(recent):
        msg_content = msg.content if hasattr(msg, 'content') else str(msg)
        msg_role = msg.role if hasattr(msg, 'role') else 'unknown'
        
        if msg_role == 'user':
            # Look for problem/document references in previous user queries
            problem_match = re.search(r'(problem|question|exercise|section)\s*(\d+[a-z]?)', msg_content.lower())
            if problem_match and not context["problem_reference"]:
                context["problem_reference"] = problem_match.group(0)
            
            # Look for document references
            doc_match = re.search(r'(problem\s*set|pset|homework|hw|exam|midterm|final|lecture)\s*(\d+)?', msg_content.lower())
            if doc_match and not context["document_reference"]:
                context["document_reference"] = doc_match.group(0)
            
            # Use the first substantive user query as topic summary
            if not context["topic_summary"] and len(msg_content) > 20:
                context["topic_summary"] = msg_content[:200]
                
        elif msg_role == 'assistant' and not context["last_assistant_response"]:
            context["last_assistant_response"] = msg_content[:500]
    
    return context


def enrich_query_with_context(query: str, history_context: dict) -> str:
    """
    Enrich a follow-up query with context from conversation history.
    
    Creates an augmented query that includes relevant context for better retrieval.
    """
    enrichment_parts = []
    
    if history_context.get("document_reference"):
        enrichment_parts.append(history_context["document_reference"])
    
    if history_context.get("problem_reference"):
        enrichment_parts.append(history_context["problem_reference"])
    
    if history_context.get("topic_summary"):
        # Extract key terms from the topic summary
        topic = history_context["topic_summary"]
        enrichment_parts.append(topic)
    
    if enrichment_parts:
        enriched = f"{' '.join(enrichment_parts)} {query}"
        return enriched
    
    return query


def retrieve_context(ta_id: str, query: str, top_k: int = 8, conversation_history: list = None, session_id: str = None) -> tuple:
    """
    Retrieve relevant chunks for a query with hybrid fallback.
    
    When chunk-based retrieval shows low confidence (poor LLM rerank scores),
    falls back to retrieving the full document text for more reliable results.
    
    Session Context Caching:
        When a document is successfully retrieved with high confidence (early hybrid routing),
        the document context is cached in the session. On follow-up queries, this cached
        context is reused instead of re-searching, ensuring continuity in multi-turn conversations.
    
    Returns:
        tuple: (chunks, diagnostics)
            - chunks: list of chunk dicts with text, score, file_name, etc.
              If hybrid fallback triggered, returns a single "chunk" with full doc text
            - diagnostics: dict with retrieval metrics for logging
    """
    from models import db, DocumentChunk, ChatSession
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
        "rerank_info": None,
        "hybrid_fallback_triggered": False,
        "hybrid_fallback_reason": None,
        "hybrid_doc_filename": None,
        "hybrid_doc_tokens": 0,
        "validation_performed": False,
        "validation_passed": None,
        "validation_expected_ref": None,
        "validation_matches_found": 0,
        "followup_detected": False,
        "followup_type": None,
        "query_enriched": False,
        "original_query": query,
        "session_cache_used": False,
        "session_cache_document": None,
        "attempt_count": 0,
        "current_problem_key": None
    }
    
    # SESSION CONTEXT CACHE: Check if we have cached context from previous successful retrieval
    # This avoids re-searching for the same document on follow-up questions
    session_context = None
    if session_id:
        try:
            session = ChatSession.query.get(session_id)
            # SECURITY: Validate session belongs to this TA to prevent cross-tenant context leakage
            if session and session.ta_id == ta_id and session.active_context:
                cached_ta_id = session.active_context.get("ta_id")
                # Double-check the cached content also belongs to this TA
                if cached_ta_id == ta_id or cached_ta_id is None:  # Allow legacy caches without ta_id
                    session_context = session.active_context
                    logger.info(f"[{ta_id}] Session has cached context: {session_context.get('document_filename', 'unknown')}")
                else:
                    logger.warning(f"[{ta_id}] Cached context belongs to different TA ({cached_ta_id}), ignoring")
        except Exception as e:
            logger.warning(f"[{ta_id}] Failed to load session context: {e}")
    
    # FOLLOW-UP DETECTION AND QUERY ENRICHMENT
    # Detect if this is a follow-up query that needs context from conversation history
    followup_info = detect_followup_query(query, conversation_history)
    diagnostics["followup_detected"] = followup_info["is_followup"]
    diagnostics["followup_type"] = followup_info["followup_type"]
    
    # CONVERSATION-BASED ATTEMPT TRACKING
    # Instead of relying on regex to detect answer submissions, count student exchanges
    # in conversation history as a proxy for how many attempts have been made.
    # The LLM itself will determine if the student is submitting an answer.
    if session_context and conversation_history:
        problem_key = session_context.get("problem_reference", "unknown_problem")
        attempt_counts = session_context.get("attempt_counts", {})
        
        # Count student messages in conversation history as exchange count
        student_messages = [m for m in conversation_history if getattr(m, 'role', None) == "user" or (isinstance(m, dict) and m.get("role") == "user")]
        exchange_count = len(student_messages)
        
        attempt_counts[problem_key] = exchange_count
        diagnostics["attempt_count"] = exchange_count
        diagnostics["current_problem_key"] = problem_key
        logger.info(f"[{ta_id}] Conversation exchange #{exchange_count} for problem '{problem_key}'")
        
        session_context["attempt_counts"] = attempt_counts
    
    # If follow-up detected, enrich the query with context from history
    # BUT only if history actually contains a problem/document reference worth enriching with
    effective_query = query
    if followup_info["needs_context_enrichment"] and conversation_history:
        history_context = extract_context_from_history(conversation_history)
        
        # Only enrich if we found useful context (problem or document reference)
        # This prevents over-enriching standalone short queries
        has_useful_context = (
            history_context.get("problem_reference") or 
            history_context.get("document_reference") or
            history_context.get("topic_summary")
        )
        
        if has_useful_context:
            enriched_query = enrich_query_with_context(query, history_context)
            
            if enriched_query != query:
                logger.info(f"[{ta_id}] Follow-up detected ({followup_info['followup_type']}), enriching query")
                logger.info(f"[{ta_id}] Original: '{query}' -> Enriched: '{enriched_query[:100]}...'")
                effective_query = enriched_query
                diagnostics["query_enriched"] = True
                diagnostics["enriched_query"] = enriched_query[:200]
        else:
            logger.info(f"[{ta_id}] Follow-up detected but no useful context in history, skipping enrichment")
    
    # USE SESSION CACHE FOR CONVERSATIONAL CONTINUITY
    # When there's cached document context AND conversation history, use the cache directly.
    # This is NOT gated on regex follow-up detection - the LLM naturally understands
    # conversational context (answer submissions, clarifications, etc.) without rigid patterns.
    if session_context and session_context.get("document_content") and conversation_history:
        # Check if user is asking about something new (topic switch detection)
        # Look for explicit new problem/document references that differ from cached context
        query_lower = query.lower()
        cached_problem = session_context.get("problem_reference", "").lower() if session_context.get("problem_reference") else ""
        
        # Extract problem numbers for comparison (normalize "Q8", "question 8", "problem 8" all to "8")
        def extract_problem_number(text):
            """Extract just the problem number from various formats."""
            if not text:
                return None
            # Match patterns like "q8", "question 8", "problem 8", "8", "q8a", "8a"
            match = re.search(r'(?:q|question|problem|exercise|section)?\s*(\d+)([a-z])?', text.lower())
            if match:
                return match.group(1) + (match.group(2) or "")  # e.g., "8" or "8a"
            return None
        
        # Detect topic switch: user mentions a DIFFERENT problem/document than what's cached
        problem_match = re.search(r'(problem|question|exercise|section|q)\s*(\d+[a-z]?)', query_lower)
        doc_match = re.search(r'(problem\s*set|pset|homework|hw|exam|midterm|final|lecture|unit|week)\s*(\d+)?', query_lower)
        
        is_topic_switch = False
        
        # Topic switch detection: only switch if the student explicitly references a DIFFERENT
        # problem number or document. Short conversational messages (answers, clarifications)
        # should NOT trigger a switch.
        if problem_match:
            new_problem_num = extract_problem_number(problem_match.group(0))
            cached_problem_num = extract_problem_number(cached_problem)
            
            if new_problem_num and cached_problem_num and new_problem_num != cached_problem_num:
                is_topic_switch = True
                logger.info(f"[{ta_id}] Topic switch: cached problem='{cached_problem_num}', new='{new_problem_num}'")
        
        if not is_topic_switch and doc_match:
            cached_doc = session_context.get("document_filename", "").lower()
            new_doc_num = doc_match.group(2) if doc_match.group(2) else ""
            cached_doc_num_match = re.search(r'(\d+)', cached_doc)
            cached_doc_num = cached_doc_num_match.group(1) if cached_doc_num_match else ""
            
            if new_doc_num and cached_doc_num and new_doc_num != cached_doc_num:
                is_topic_switch = True
                logger.info(f"[{ta_id}] Topic switch: cached doc='{cached_doc_num}', new='{new_doc_num}'")
        
        if not is_topic_switch:
            # Use cached context - no need to re-search
            logger.info(f"[{ta_id}] Using cached session context for follow-up (document: {session_context.get('document_filename')})")
            
            diagnostics["session_cache_used"] = True
            diagnostics["session_cache_document"] = session_context.get("document_filename")
            diagnostics["hybrid_fallback_triggered"] = True
            diagnostics["hybrid_fallback_reason"] = "session_cache"
            diagnostics["hybrid_doc_filename"] = session_context.get("document_filename")
            
            # Save updated attempt counts to session (was already incremented earlier)
            if session_id and session_context.get("attempt_counts"):
                try:
                    session = ChatSession.query.get(session_id)
                    if session and session.ta_id == ta_id:
                        session.active_context = dict(session_context)
                        db.session.commit()
                except Exception as e:
                    logger.warning(f"[{ta_id}] Failed to save attempt count: {e}")
            
            # Fetch solution document for answer validation when there's enough conversation
            # context (2+ student messages means at least one Q&A exchange has occurred,
            # so the student has had a chance to attempt the problem).
            # This avoids exposing solution content on the very first help request.
            combined_content = session_context.get("document_content", "")
            solution_added = False
            
            student_messages = [m for m in conversation_history if getattr(m, 'role', None) == "user" or (isinstance(m, dict) and m.get("role") == "user")]
            if len(student_messages) >= 2:
                problem_doc_name = session_context.get("document_filename", "")
                solution_text, solution_filename, solution_tokens = find_solution_document(problem_doc_name, ta_id)
                
                if solution_text:
                    logger.info(f"[{ta_id}] Including solution document '{solution_filename}' for answer verification (exchange #{len(student_messages)})")
                    combined_content = f"=== PROBLEM DOCUMENT: {problem_doc_name} ===\n\n{combined_content}\n\n=== SOLUTION DOCUMENT (for answer verification): {solution_filename} ===\n\n{solution_text}"
                    diagnostics["solution_doc_added"] = True
                    diagnostics["solution_doc_filename"] = solution_filename
                    solution_added = True
                else:
                    logger.info(f"[{ta_id}] No solution document found - LLM will use problem context only")
                    diagnostics["solution_doc_added"] = False
            else:
                logger.info(f"[{ta_id}] Early conversation (exchange #{len(student_messages)}) - solution doc not yet included")
                diagnostics["solution_doc_added"] = False
            
            cached_chunk = {
                "text": combined_content,
                "score": 1.0,
                "file_name": session_context.get("document_filename", "cached document"),
                "chunk_index": 0,
                "doc_type": session_context.get("doc_type"),
                "problem_reference": session_context.get("problem_reference"),
                "solution_included": solution_added
            }
            return [cached_chunk], diagnostics
        else:
            # User is switching topics - clear the cache and reset attempt counts
            logger.info(f"[{ta_id}] Topic switch detected, clearing session cache and resetting attempts")
            diagnostics["attempt_count"] = 0  # Reset for new problem
            diagnostics["current_problem_key"] = None
            if session_id:
                try:
                    session = ChatSession.query.get(session_id)
                    # SECURITY: Only clear cache for this TA's session
                    if session and session.ta_id == ta_id:
                        session.active_context = None
                        db.session.commit()
                except Exception as e:
                    logger.warning(f"[{ta_id}] Failed to clear session cache: {e}")
    
    total_chunks = DocumentChunk.query.filter_by(ta_id=ta_id).count()
    diagnostics["total_chunks_in_ta"] = total_chunks
    
    if total_chunks == 0:
        logger.warning(f"No indexed chunks found for TA: {ta_id}")
        return [], diagnostics
    
    client = get_openai_client()
    
    # Use the enriched query for embedding to get better semantic search results
    response = client.embeddings.create(
        model=Config.EMBEDDING_MODEL,
        input=effective_query
    )
    query_embedding = response.data[0].embedding
    
    # IMPORTANT: Use ORIGINAL query for analyze_query to preserve filename matching and filters
    # Enriched query is only for semantic search (embeddings), not for filter extraction
    query_analysis = analyze_query(query, ta_id)
    diagnostics["is_conceptual"] = query_analysis.get("is_conceptual", False)
    
    # EARLY HYBRID ROUTING: For specific problem references (e.g., "section 1 question a"),
    # skip the unreliable LLM reranker and go directly to full-document mode.
    # This is more reliable for pinpoint queries where we need to find exact content.
    if query_analysis.get("requires_early_hybrid") and Config.HYBRID_RETRIEVAL_ENABLED:
        problem_ref = query_analysis.get("problem_reference", {})
        logger.info(f"[{ta_id}] Early hybrid routing: skipping reranker for specific reference '{problem_ref.get('full_ref')}'")
        
        # Identify target document using query filters (year, doc_type, etc.)
        target_doc_ids, id_method = identify_target_documents([], query_analysis, ta_id)
        diagnostics["hybrid_doc_id_method"] = id_method
        
        if target_doc_ids:
            doc_id = target_doc_ids[0]
            full_text, filename, token_estimate = get_full_document_text(doc_id)
            
            if full_text and token_estimate <= Config.HYBRID_MAX_DOC_TOKENS:
                logger.info(f"[{ta_id}] Early hybrid: using full document '{filename}' ({token_estimate} tokens)")
                
                diagnostics["hybrid_fallback_triggered"] = True
                diagnostics["hybrid_fallback_reason"] = f"early_routing_specific_ref_{problem_ref.get('full_ref')}"
                diagnostics["hybrid_doc_filename"] = filename
                diagnostics["hybrid_doc_tokens"] = token_estimate
                diagnostics["retrieval_method"] = "early_hybrid_full_doc"
                diagnostics["validation_expected_ref"] = problem_ref.get("full_ref")
                
                hybrid_chunks = [{
                    "text": full_text,
                    "score": 10.0,
                    "file_name": filename,
                    "doc_type": "exam",  # Will be from document metadata in practice
                    "metadata": {},
                    "is_full_document": True,
                    "llm_relevance_score": 10.0,
                    "llm_reason": f"Early hybrid routing for specific reference '{problem_ref.get('full_ref')}'"
                }]
                
                logger.info(f"[{ta_id}] Early hybrid complete | doc={filename} | tokens={token_estimate}")
                
                # CACHE TO SESSION: Save this successful retrieval for follow-up queries
                if session_id:
                    try:
                        session = ChatSession.query.get(session_id)
                        # SECURITY: Only cache if session belongs to this TA
                        if session and session.ta_id == ta_id:
                            # Preserve existing attempt_counts from session_context if available
                            existing_attempts = session_context.get("attempt_counts", {}) if session_context else {}
                            session.active_context = {
                                "ta_id": ta_id,  # Store ta_id for cross-tenant security validation
                                "document_filename": filename,
                                "document_content": full_text,
                                "problem_reference": problem_ref.get("full_ref") if problem_ref else None,
                                "doc_type": "problem_set",
                                "cached_at": datetime.utcnow().isoformat(),
                                "attempt_counts": existing_attempts
                            }
                            db.session.commit()
                            logger.info(f"[{ta_id}] Cached document context for session: {filename}")
                    except Exception as e:
                        logger.warning(f"[{ta_id}] Failed to cache session context: {e}")
                
                return hybrid_chunks, diagnostics
            elif token_estimate > Config.HYBRID_MAX_DOC_TOKENS:
                logger.warning(f"[{ta_id}] Document too large for early hybrid: {token_estimate} tokens, falling back to chunk retrieval")
            elif not full_text:
                logger.warning(f"[{ta_id}] Failed to extract text for early hybrid, falling back to chunk retrieval")
        else:
            logger.warning(f"[{ta_id}] Early hybrid: could not identify target document (method={id_method}), falling back to chunk retrieval")
    
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
    
    if query_analysis["year_filter"]:
        year = query_analysis["year_filter"]
        filtered_query = filtered_query.filter(
            DocumentChunk.file_name.contains(year)
        )
        has_filters = True
        filter_description.append(f"year={year}")
        logger.info(f"[{ta_id}] Year filter applied: {year}")
    
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
    
    pre_rerank_candidates = []
    for i, chunk in enumerate(initial_chunks):
        text_preview = chunk["text"][:200].replace("\n", " ").replace("\t", " ").strip()
        pre_rerank_candidates.append({
            "idx": i,
            "file": chunk["file_name"],
            "score": round(chunk["score"], 4),
            "text": text_preview
        })
    diagnostics["pre_rerank_candidates"] = pre_rerank_candidates
    
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
    
    # Post-retrieval validation: check if chunks contain the expected problem reference
    problem_ref = query_analysis.get("problem_reference", {})
    validation_result = {"passed": True, "reason": "no_reference_to_validate", "matches_found": 0}
    
    if problem_ref and problem_ref.get("full_ref"):
        validation_result = validate_chunks_contain_reference(chunks, problem_ref)
        diagnostics["validation_performed"] = True
        diagnostics["validation_passed"] = validation_result["passed"]
        diagnostics["validation_expected_ref"] = problem_ref.get("full_ref")
        diagnostics["validation_matches_found"] = validation_result["matches_found"]
        
        if not validation_result["passed"]:
            logger.warning(f"[{ta_id}] Validation FAILED: expected reference '{problem_ref['full_ref']}' not found in top chunks")
    
    confidence = assess_retrieval_confidence(chunks, rerank_info)
    
    # Trigger hybrid fallback if: low confidence OR validation failed
    should_trigger_hybrid = confidence["is_low_confidence"] or (
        diagnostics["validation_performed"] and not diagnostics["validation_passed"]
    )
    
    if should_trigger_hybrid:
        # Determine the reason for triggering hybrid
        if not validation_result["passed"] and diagnostics["validation_performed"]:
            trigger_reason = f"validation_failed_{validation_result['reason']}"
            logger.info(f"[{ta_id}] Hybrid triggered by VALIDATION FAILURE: expected '{problem_ref.get('full_ref')}' not in chunks")
        else:
            trigger_reason = confidence["reason"]
            logger.info(f"[{ta_id}] Hybrid triggered by LOW CONFIDENCE: {confidence['reason']} (top_score={confidence['top_score']}, spread={confidence['score_spread']})")
        
        target_doc_ids, id_method = identify_target_documents(chunks, query_analysis, ta_id)
        diagnostics["hybrid_doc_id_method"] = id_method
        
        if target_doc_ids:
            doc_id = target_doc_ids[0]
            full_text, filename, token_estimate = get_full_document_text(doc_id)
            
            if full_text and token_estimate <= Config.HYBRID_MAX_DOC_TOKENS:
                logger.info(f"[{ta_id}] Hybrid fallback: using full document '{filename}' ({token_estimate} tokens)")
                
                diagnostics["hybrid_fallback_triggered"] = True
                diagnostics["hybrid_fallback_reason"] = trigger_reason
                diagnostics["hybrid_doc_filename"] = filename
                diagnostics["hybrid_doc_tokens"] = token_estimate
                diagnostics["retrieval_method"] = "hybrid_full_doc"
                
                hybrid_chunks = [{
                    "text": full_text,
                    "score": 10.0,
                    "file_name": filename,
                    "doc_type": chunks[0].get("doc_type", "other") if chunks else "other",
                    "metadata": chunks[0].get("metadata", {}) if chunks else {},
                    "is_full_document": True,
                    "llm_relevance_score": 10.0,
                    "llm_reason": "Full document fallback due to low chunk confidence"
                }]
                
                # CACHE THE DOCUMENT for follow-up queries
                # This ensures answer submissions and clarification questions use the same document
                if session_id:
                    try:
                        session = ChatSession.query.get(session_id)
                        # SECURITY: Only cache if session belongs to this TA
                        if session and session.ta_id == ta_id:
                            existing_attempts = session_context.get("attempt_counts", {}) if session_context else {}
                            session.active_context = {
                                "ta_id": ta_id,
                                "document_filename": filename,
                                "document_content": full_text,
                                "problem_reference": problem_ref.get("full_ref") if problem_ref else None,
                                "doc_type": chunks[0].get("doc_type", "other") if chunks else "other",
                                "cached_at": datetime.utcnow().isoformat(),
                                "attempt_counts": existing_attempts
                            }
                            db.session.commit()
                            logger.info(f"[{ta_id}] Cached document context for session (hybrid fallback): {filename}")
                    except Exception as e:
                        logger.warning(f"[{ta_id}] Failed to cache session context in hybrid fallback: {e}")
                
                logger.info(f"[{ta_id}] Hybrid fallback complete | doc={filename} | tokens={token_estimate}")
                return hybrid_chunks, diagnostics
            elif token_estimate > Config.HYBRID_MAX_DOC_TOKENS:
                logger.warning(f"[{ta_id}] Document too large for hybrid fallback: {token_estimate} tokens > {Config.HYBRID_MAX_DOC_TOKENS}")
                diagnostics["hybrid_fallback_reason"] = f"doc_too_large_{token_estimate}_tokens"
            elif not full_text:
                logger.warning(f"[{ta_id}] Failed to extract text from document {doc_id}")
                diagnostics["hybrid_fallback_reason"] = f"extraction_failed_doc_{doc_id}"
        else:
            logger.warning(f"[{ta_id}] No target document identified for hybrid fallback (method={id_method})")
            diagnostics["hybrid_fallback_reason"] = f"no_target_doc_{id_method}"
    
    logger.info(f"[{ta_id}] Retrieved {len(chunks)} chunks | method={diagnostics['retrieval_method']} | reranked={diagnostics['rerank_applied']} | scores: top1={diagnostics['score_top1']}, spread={diagnostics['score_spread']}")
    
    # CACHE DOCUMENT CONTEXT for follow-up queries (standard chunk retrieval path)
    # Only cache when retrieval is confident (not low confidence) AND validation passed
    # This prevents caching wrong document context that would mislead follow-ups
    should_cache_chunks = (
        session_id and 
        chunks and 
        not diagnostics.get("session_cache_used") and
        not confidence["is_low_confidence"] and  # Only cache when confident
        (not diagnostics.get("validation_performed") or diagnostics.get("validation_passed"))  # And validation passed (if performed)
    )
    
    if should_cache_chunks:
        try:
            # Get the primary document from top chunk
            top_doc = chunks[0].get("file_name", "")
            if top_doc:
                session = ChatSession.query.get(session_id)
                # SECURITY: Only cache if session belongs to this TA
                if session and session.ta_id == ta_id:
                    # Concatenate chunk texts as the cached content
                    combined_content = "\n\n---\n\n".join([c.get("text", "") for c in chunks])
                    existing_attempts = session_context.get("attempt_counts", {}) if session_context else {}
                    session.active_context = {
                        "ta_id": ta_id,
                        "document_filename": top_doc,
                        "document_content": combined_content,
                        "problem_reference": problem_ref.get("full_ref") if problem_ref else None,
                        "doc_type": chunks[0].get("doc_type", "other"),
                        "cached_at": datetime.utcnow().isoformat(),
                        "attempt_counts": existing_attempts,
                        "cache_source": "chunk_retrieval"  # Track that this came from chunks, not full doc
                    }
                    db.session.commit()
                    logger.info(f"[{ta_id}] Cached document context for session (chunk retrieval): {top_doc}")
        except Exception as e:
            logger.warning(f"[{ta_id}] Failed to cache session context in chunk retrieval: {e}")
    
    return chunks, diagnostics
