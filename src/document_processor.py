import os
import logging
import json
import time
import re
from datetime import datetime
from config import Config
from sqlalchemy.exc import OperationalError, DBAPIError

logger = logging.getLogger(__name__)

def extract_section_headers(text: str) -> list:
    """
    Extract problem/section headers from document text with their positions.
    Returns list of (header_content_start, header_text) tuples, sorted by position.
    
    The position returned is the start of the actual header content (group 1),
    not the newline/start anchor, so boundary splitting includes the header.
    
    Matches patterns like:
    - "Problem 1: Title"
    - "Problem 2 (5 points)"
    - "Question 3:"
    - "Section I - Title"
    - "Part A:"
    """
    headers = []
    
    patterns = [
        r'(?:^|\n)(Problem\s+\d+[:\s][^\n]{0,60})',
        r'(?:^|\n)(Question\s+\d+[:\s][^\n]{0,60})',
        r'(?:^|\n)(Section\s+(?:\d+|[IVX]+)[:\s\-][^\n]{0,60})',
        r'(?:^|\n)(Part\s+[A-Z][:\s][^\n]{0,60})',
        r'(?:^|\n)(Exercise\s+\d+[:\s][^\n]{0,60})',
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            header_text = match.group(1).strip()
            header_text = re.sub(r'\s+', ' ', header_text)
            if len(header_text) > 80:
                header_text = header_text[:77] + "..."
            # Use start of group 1 (actual header content), not match.start() (which includes newline)
            header_content_start = match.start(1)
            headers.append((header_content_start, header_text))
    
    headers.sort(key=lambda x: x[0])
    return headers

def get_context_for_position(headers: list, position: int) -> str:
    """
    Get the most recent header that appears before the given position.
    Returns the header text or empty string if no header precedes this position.
    """
    context = ""
    for header_pos, header_text in headers:
        if header_pos <= position:
            context = header_text
        else:
            break
    return context

def chunk_text_with_context(text: str, chunk_size: int = 800, overlap: int = 200, doc_filename: str = "") -> list:
    """
    Chunk text with boundary-aware splitting at section headers.
    Forces chunk breaks at section boundaries (Problem X, Question X, etc.) so chunks
    don't span multiple problems/sections.
    Returns list of dicts with 'text' (enriched) and 'original_text' (raw).
    """
    headers = extract_section_headers(text)
    
    # Cache sorted header positions once
    sorted_header_positions = sorted(pos for pos, _ in headers)
    
    if len(text) <= chunk_size:
        context = get_context_for_position(headers, 0)
        enriched = f"[{doc_filename} > {context}] {text}" if context else f"[{doc_filename}] {text}"
        return [{"text": enriched, "original_text": text, "context": context}]
    
    chunks = []
    start = 0
    header_idx = 0  # Track position in sorted headers
    
    while start < len(text):
        # Advance header index past any headers at or before start
        while header_idx < len(sorted_header_positions) and sorted_header_positions[header_idx] <= start:
            header_idx += 1
        next_boundary = sorted_header_positions[header_idx] if header_idx < len(sorted_header_positions) else None
        
        # Calculate tentative end position
        end = start + chunk_size
        
        # Track if we're breaking at a boundary
        breaking_at_boundary = False
        
        # If there's a section boundary within this chunk, force break there
        if next_boundary is not None and next_boundary < end and next_boundary > start:
            # End this chunk at the boundary position (not including the header)
            end = next_boundary
            breaking_at_boundary = True
        elif end < len(text):
            # No boundary in range - use natural break points as before
            break_points = [
                text.rfind('\n\n', start, end),
                text.rfind('. ', start, end),
                text.rfind('\n', start, end),
                text.rfind(' ', start, end)
            ]
            for bp in break_points:
                if bp > start + chunk_size // 2:
                    end = bp + 1
                    break
        
        # Guard against zero-length chunks or no progress
        if end <= start:
            end = min(start + chunk_size, len(text))
        
        chunk_text = text[start:end].strip()
        
        if chunk_text:
            # Get context for this chunk position
            context = get_context_for_position(headers, start)
            if context:
                enriched = f"[{doc_filename} > {context}] {chunk_text}"
            else:
                enriched = f"[{doc_filename}] {chunk_text}"
            chunks.append({
                "text": enriched,
                "original_text": chunk_text,
                "context": context
            })
            
            # Move to next position
            if breaking_at_boundary:
                # CRITICAL: Start exactly at the boundary with NO overlap
                # This ensures the next chunk gets the correct section context
                start = next_boundary
            else:
                start = end - overlap
                if start < 0:
                    start = 0
        else:
            # Empty chunk - advance to avoid infinite loop
            if next_boundary is not None and next_boundary > start:
                start = next_boundary
            else:
                start = end if end > start else start + 1
        
        if start >= len(text):
            break
    
    return chunks

def db_commit_with_retry(db, max_retries=3, delay=1.0):
    """Commit database changes with retry logic for connection issues."""
    for attempt in range(max_retries):
        try:
            db.session.commit()
            return True
        except (OperationalError, DBAPIError) as e:
            db.session.rollback()
            if attempt < max_retries - 1:
                logger.warning(f"Database commit failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay * (attempt + 1))
            else:
                logger.error(f"Database commit failed after {max_retries} attempts: {e}")
                raise
    return False

def sanitize_text(text: str) -> str:
    """Remove null bytes and other problematic characters that PostgreSQL cannot store."""
    if not text:
        return ""
    text = text.replace('\x00', '')
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    return text

def extract_text_from_file(file_path: str) -> tuple:
    """Extract text from file. Returns (text, page_count) - page_count is 0 for non-PDFs."""
    ext = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else ''
    page_count = 0
    
    try:
        if ext == 'pdf':
            text, page_count = extract_pdf(file_path)
        elif ext == 'docx':
            text = extract_docx(file_path)
        elif ext == 'doc':
            text = extract_doc(file_path)
        elif ext in ('xlsx', 'xls'):
            text = extract_excel(file_path)
        elif ext == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        elif ext in ('pptx', 'ppt'):
            text = extract_pptx(file_path)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return "", 0
        
        return sanitize_text(text), page_count
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return "", 0

def extract_pdf(file_path: str) -> tuple:
    """Extract PDF text and return (text, page_count)."""
    text, page_count = _extract_pdf_pypdf2(file_path)
    if text and len(text.strip()) > 100:
        return text, page_count
    
    logger.info("PyPDF2 extraction insufficient, trying pdfplumber...")
    text, page_count = _extract_pdf_pdfplumber(file_path)
    if text and len(text.strip()) > 100:
        return text, page_count
    
    return "", 0

def _extract_pdf_pdfplumber(file_path: str) -> tuple:
    """Extract PDF text using pdfplumber with total time limit. Returns (text, page_count)."""
    try:
        import pdfplumber
        
        text_parts = []
        start_time = time.time()
        max_total_time = 120
        
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages):
                page_start = time.time()
                try:
                    if time.time() - start_time > max_total_time:
                        logger.warning(f"pdfplumber exceeded {max_total_time}s total, stopping at page {page_num}")
                        break
                    
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                    
                    page_time = time.time() - page_start
                    if page_time > 10:
                        logger.info(f"pdfplumber page {page_num + 1}/{total_pages} took {page_time:.1f}s")
                        
                except Exception as page_e:
                    logger.warning(f"pdfplumber failed on page {page_num + 1}: {page_e}")
                    continue
                    
        return "\n\n".join(text_parts), total_pages
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}")
        return "", 0

def _extract_pdf_pypdf2(file_path: str) -> tuple:
    """Extract PDF text using PyPDF2. Returns (text, page_count)."""
    try:
        from PyPDF2 import PdfReader
        
        reader = PdfReader(file_path)
        text_parts = []
        page_count = len(reader.pages)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts), page_count
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")
        return "", 0

def extract_docx(file_path: str) -> str:
    """
    Extract text from DOCX files, preserving list numbering (a, b, c, 1, 2, 3),
    tables, and document structure.
    
    Uses docx2python for comprehensive extraction that preserves:
    - Numbered/bulleted list prefixes
    - Table content
    - Text boxes
    - Footnotes/endnotes
    """
    try:
        from docx2python import docx2python
        
        with docx2python(file_path) as doc:
            text_parts = []
            
            if doc.body:
                body_text = _flatten_docx2python_content(doc.body)
                if body_text.strip():
                    text_parts.append(body_text)
            
            if doc.footnotes:
                footnotes_text = _flatten_docx2python_content(doc.footnotes)
                if footnotes_text.strip():
                    text_parts.append("\n--- Footnotes ---\n" + footnotes_text)
            
            if doc.endnotes:
                endnotes_text = _flatten_docx2python_content(doc.endnotes)
                if endnotes_text.strip():
                    text_parts.append("\n--- Endnotes ---\n" + endnotes_text)
            
            result = "\n\n".join(text_parts)
            
            if result.strip():
                logger.info(f"Successfully extracted DOCX using docx2python: {len(result)} chars")
                return result
            
    except Exception as e:
        logger.warning(f"docx2python extraction failed, falling back to python-docx: {e}")
    
    try:
        from docx import Document
        
        doc = Document(file_path)
        text_parts = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                prefix = _get_paragraph_list_prefix(para)
                if prefix:
                    text_parts.append(f"{prefix} {para.text}")
                else:
                    text_parts.append(para.text)
        
        for table in doc.tables:
            table_text = _extract_table_text(table)
            if table_text.strip():
                text_parts.append(table_text)
        
        result = "\n\n".join(text_parts)
        logger.info(f"Extracted DOCX using python-docx fallback: {len(result)} chars")
        return result
        
    except Exception as e:
        logger.error(f"DOCX extraction failed completely: {e}")
        return ""


def _flatten_docx2python_content(content) -> str:
    """
    Recursively flatten docx2python nested list structure into text.
    docx2python returns deeply nested lists representing document structure.
    """
    if isinstance(content, str):
        return content.strip()
    
    if isinstance(content, list):
        parts = []
        for item in content:
            flattened = _flatten_docx2python_content(item)
            if flattened:
                parts.append(flattened)
        return "\n".join(parts)
    
    return ""


def _get_paragraph_list_prefix(para) -> str:
    """
    Extract list numbering prefix from a paragraph using python-docx XML parsing.
    Returns prefix like 'a)', '1.', 'i)' or empty string if not a list item.
    """
    try:
        if para._element.pPr is None:
            return ""
        
        numPr = para._element.pPr.numPr
        if numPr is None or numPr.numId is None:
            return ""
        
        return ""
        
    except Exception:
        return ""


def _extract_table_text(table) -> str:
    """Extract text from a Word table, preserving structure."""
    try:
        rows_text = []
        for row in table.rows:
            cells_text = []
            for cell in row.cells:
                cell_content = cell.text.strip()
                if cell_content:
                    cells_text.append(cell_content)
            if cells_text:
                rows_text.append(" | ".join(cells_text))
        
        if rows_text:
            return "Table:\n" + "\n".join(rows_text)
        return ""
    except Exception:
        return ""


def extract_doc(file_path: str) -> str:
    """
    Extract text from older .doc files (pre-2007 Word format).
    Uses antiword command-line tool or falls back to textract.
    """
    import subprocess
    
    try:
        result = subprocess.run(
            ['antiword', file_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            logger.info(f"Extracted .doc using antiword: {len(result.stdout)} chars")
            return result.stdout
    except FileNotFoundError:
        logger.warning("antiword not installed, trying catdoc")
    except Exception as e:
        logger.warning(f"antiword extraction failed: {e}")
    
    try:
        result = subprocess.run(
            ['catdoc', file_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            logger.info(f"Extracted .doc using catdoc: {len(result.stdout)} chars")
            return result.stdout
    except FileNotFoundError:
        logger.warning("catdoc not installed")
    except Exception as e:
        logger.warning(f"catdoc extraction failed: {e}")
    
    logger.warning(f"Could not extract .doc file: {file_path}. Install antiword or catdoc.")
    return ""

def extract_excel(file_path: str) -> str:
    import pandas as pd
    
    xl = pd.ExcelFile(file_path)
    text_parts = []
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        text_parts.append(f"Sheet: {sheet_name}\n{df.to_string()}")
    return "\n\n".join(text_parts)

def extract_pptx(file_path: str) -> str:
    try:
        from pptx import Presentation
        prs = Presentation(file_path)
        text_parts = []
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text)
            if slide_text:
                text_parts.append(f"Slide {slide_num}:\n" + "\n".join(slide_text))
        return "\n\n".join(text_parts)
    except ImportError:
        logger.warning("python-pptx not installed, skipping PowerPoint file")
        return ""

def extract_metadata_with_llm(text: str, filename: str) -> dict:
    from openai import OpenAI
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    preview = text[:3000] if len(text) > 3000 else text
    
    prompt = f"""Analyze this course document and extract metadata. The filename is: {filename}

Document preview:
{preview}

Extract the following information as JSON:
{{
    "doc_type": "homework" | "exam" | "lecture" | "reading" | "syllabus" | "other",
    "assignment_number": "1" | "2" | "3" | null (if applicable),
    "instructional_unit_number": 1 | 2 | 3 | null (lecture/class/week number if mentioned),
    "instructional_unit_label": "lecture" | "class" | "week" | "module" | "session" | null,
    "course_code": "MGT404" | null (if visible),
    "year": "2024" | "2025" | null (if mentioned),
    "is_solutions": true | false (whether this contains solutions/answers),
    "content_title": "The actual document title as written in the content (e.g., 'Self-Study Problem Set #2', 'Final Exam 2024'). Extract from headers/title text, not filename."
}}

IMPORTANT: Classify by the PRIMARY document type, not whether it has solutions.
- "exam_solutions.pdf" or "final_exam_with_solutions.pdf" → doc_type: "exam", is_solutions: true
- "homework_answers.pdf" or "problem_set_solutions.pdf" → doc_type: "homework", is_solutions: true
- "lecture_notes.pdf" → doc_type: "lecture"

Return ONLY valid JSON, no other text."""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            metadata = json.loads(content)
            return metadata
        except Exception as e:
            logger.warning(f"LLM metadata extraction attempt {attempt+1} failed: {e}")
            if attempt == 2:
                logger.error(f"All LLM metadata extraction attempts failed, returning defaults")
                return {
                    "doc_type": "other",
                    "assignment_number": None,
                    "instructional_unit_number": None,
                    "instructional_unit_label": None,
                    "course_code": None,
                    "year": None,
                    "is_solutions": False,
                    "content_title": None
                }

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> list:
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            break_points = [
                text.rfind('\n\n', start, end),
                text.rfind('. ', start, end),
                text.rfind('\n', start, end),
                text.rfind(' ', start, end)
            ]
            for bp in break_points:
                if bp > start + chunk_size // 2:
                    end = bp + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    
    return chunks

def process_and_index_documents(ta_id: str, progress_callback=None) -> dict:
    import tempfile
    from openai import OpenAI
    
    from models import db, Document, TeachingAssistant, DocumentChunk
    from flask import current_app
    from src.qa_logger import log_index_batch
    
    logger.info(f"[{ta_id}] Starting indexing process...")
    
    ta = db.session.get(TeachingAssistant, ta_id)
    ta_slug = ta.slug if ta else "unknown"
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    logger.info(f"[{ta_id}] Clearing existing chunks...")
    DocumentChunk.query.filter_by(ta_id=ta_id).delete()
    db_commit_with_retry(db)
    logger.info(f"[{ta_id}] Cleared existing chunks")
    
    doc_ids = [d.id for d in db.session.query(Document.id).filter_by(ta_id=ta_id).all()]
    total_docs = len(doc_ids)
    
    if total_docs == 0:
        raise ValueError("No documents found for this TA")
    
    logger.info(f"[{ta_id}] Found {total_docs} documents to process: {doc_ids}")
    
    all_chunk_data = []
    index_log_entries = []
    
    for doc_idx, doc_id in enumerate(doc_ids):
        doc = db.session.get(Document, doc_id)
        if not doc:
            logger.warning(f"[{ta_id}] Document {doc_id} not found, skipping")
            continue
        
        logger.info(f"[{ta_id}] Processing document [{doc.id}]: {doc.original_filename} ({doc_idx + 1}/{total_docs})")
        
        if progress_callback and total_docs > 0:
            progress = int((doc_idx / total_docs) * 50)
            progress_callback(ta_id, progress)
        
        text = None
        page_count = 0
        
        logger.info(f"[{ta_id}] [{doc.id}] Extracting text...")
        if doc.file_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{doc.file_type}") as tmp_file:
                tmp_file.write(doc.file_content)
                tmp_path = tmp_file.name
            try:
                text, page_count = extract_text_from_file(tmp_path)
            finally:
                os.unlink(tmp_path)
        elif os.path.exists(doc.storage_path):
            text, page_count = extract_text_from_file(doc.storage_path)
        else:
            logger.warning(f"[{ta_id}] [{doc.id}] No file content available - document needs to be re-uploaded")
            continue
        
        if not text:
            logger.warning(f"[{ta_id}] [{doc.id}] No text extracted")
            continue
        
        raw_text_length = len(text)
        logger.info(f"[{ta_id}] [{doc.id}] Extracted {raw_text_length} chars from {page_count} pages")
        
        logger.info(f"[{ta_id}] [{doc.id}] Extracting metadata with LLM...")
        metadata = extract_metadata_with_llm(text, doc.original_filename)
        doc.doc_type = metadata.get("doc_type")
        doc.assignment_number = metadata.get("assignment_number")
        doc.instructional_unit_number = metadata.get("instructional_unit_number")
        doc.instructional_unit_label = metadata.get("instructional_unit_label")
        doc.content_title = metadata.get("content_title")
        doc.extraction_metadata = metadata
        doc.metadata_extracted = True
        logger.info(f"[{ta_id}] [{doc.id}] Saving metadata...")
        db_commit_with_retry(db)
        logger.info(f"[{ta_id}] [{doc.id}] Metadata saved")
        
        # Log headers found for diagnostic purposes
        headers_found = extract_section_headers(text)
        headers_summary = "; ".join([f"{h[1][:40]}" for h in headers_found[:5]])
        logger.info(f"[{ta_id}] [{doc.id}] Found {len(headers_found)} headers: {headers_summary}")
        
        chunks = chunk_text_with_context(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP, doc.original_filename)
        total_chunks = len(chunks)
        
        for i, chunk_data in enumerate(chunks):
            all_chunk_data.append({
                "ta_id": ta_id,
                "document_id": doc.id,
                "chunk_index": i,
                "chunk_text": chunk_data["original_text"],
                "chunk_text_enriched": chunk_data["text"],
                "context": chunk_data.get("context", ""),
                "doc_type": doc.doc_type or "other",
                "assignment_number": doc.assignment_number or "",
                "instructional_unit_number": doc.instructional_unit_number or 0,
                "instructional_unit_label": doc.instructional_unit_label or "",
                "file_name": doc.original_filename
            })
            
            index_log_entries.append({
                "ta_id": ta_id,
                "ta_slug": ta_slug,
                "file_name": doc.original_filename,
                "doc_type": doc.doc_type or "other",
                "total_pages": page_count,
                "raw_text_length": raw_text_length,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "chunk_text_length": len(chunk_data["original_text"]),
                "chunk_context": chunk_data.get("context", ""),
                "chunk_text_preview": chunk_data["original_text"][:300],
                "enriched_text_preview": chunk_data["text"][:300],
                "has_embedding": False,
                "status": "pending",
                "error_message": "",
                "headers_found": headers_summary if i == 0 else ""  # Only log on first chunk
            })
    
    if not all_chunk_data:
        raise ValueError("No text content found in any documents")
    
    logger.info(f"[{ta_id}] Document processing complete. Total chunks to embed: {len(all_chunk_data)}")
    
    try:
        all_embeddings = []
        batch_size = 100
        total_batches = (len(all_chunk_data) + batch_size - 1) // batch_size
        
        logger.info(f"[{ta_id}] Starting embedding generation ({total_batches} batches)...")
        
        for batch_idx, i in enumerate(range(0, len(all_chunk_data), batch_size)):
            batch_texts = [c["chunk_text_enriched"] for c in all_chunk_data[i:i+batch_size]]
            
            if progress_callback and total_batches > 0:
                progress = 50 + int((batch_idx / total_batches) * 40)
                progress_callback(ta_id, progress)
            
            logger.info(f"[{ta_id}] Embedding batch {batch_idx + 1}/{total_batches}...")
            response = client.embeddings.create(
                model=Config.EMBEDDING_MODEL,
                input=batch_texts
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            logger.info(f"[{ta_id}] Embedded batch {batch_idx + 1}/{total_batches} ({len(all_embeddings)} total)")
        
        if progress_callback:
            progress_callback(ta_id, 90)
        
        logger.info(f"[{ta_id}] Embeddings complete. Storing chunks in database...")
        
        db_batch_size = 500
        total_db_batches = (len(all_chunk_data) + db_batch_size - 1) // db_batch_size
        
        for batch_idx, i in enumerate(range(0, len(all_chunk_data), db_batch_size)):
            batch_end = min(i + db_batch_size, len(all_chunk_data))
            
            if progress_callback and total_db_batches > 0:
                progress = 90 + int((batch_idx / total_db_batches) * 10)
                progress_callback(ta_id, progress)
            
            logger.info(f"[{ta_id}] Creating chunk objects for batch {batch_idx + 1}/{total_db_batches}...")
            for j in range(i, batch_end):
                chunk_data = all_chunk_data[j]
                chunk_obj = DocumentChunk(
                    ta_id=chunk_data["ta_id"],
                    document_id=chunk_data["document_id"],
                    chunk_index=chunk_data["chunk_index"],
                    chunk_text=sanitize_text(chunk_data["chunk_text"]),
                    chunk_context=chunk_data.get("context", "")[:256] if chunk_data.get("context") else None,
                    doc_type=chunk_data["doc_type"],
                    assignment_number=chunk_data["assignment_number"],
                    instructional_unit_number=chunk_data["instructional_unit_number"],
                    instructional_unit_label=chunk_data["instructional_unit_label"],
                    file_name=chunk_data["file_name"],
                    embedding=all_embeddings[j]
                )
                db.session.add(chunk_obj)
            
            logger.info(f"[{ta_id}] Committing batch {batch_idx + 1}/{total_db_batches} to database...")
            db_commit_with_retry(db)
            logger.info(f"[{ta_id}] Stored batch {batch_idx + 1}/{total_db_batches}: chunks {i+1}-{batch_end} of {len(all_chunk_data)}")
        
        logger.info(f"[{ta_id}] Indexing complete! Total chunks: {len(all_chunk_data)}")
        
        for entry in index_log_entries:
            entry["has_embedding"] = True
            entry["status"] = "success"
        
        logger.info(f"[{ta_id}] Logging {len(index_log_entries)} index entries to Google Sheets...")
        log_index_batch(index_log_entries)
        
        return {"chunks_indexed": len(all_chunk_data)}
        
    except Exception as e:
        logger.error(f"[{ta_id}] Indexing failed during embedding/storage: {e}")
        
        for entry in index_log_entries:
            entry["has_embedding"] = False
            entry["status"] = "error"
            entry["error_message"] = str(e)[:500]
        
        logger.info(f"[{ta_id}] Logging {len(index_log_entries)} failed index entries to Google Sheets...")
        log_index_batch(index_log_entries)
        
        raise
