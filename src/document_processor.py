import os
import logging
import json
import time
from datetime import datetime
from config import Config
from sqlalchemy.exc import OperationalError, DBAPIError

logger = logging.getLogger(__name__)

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

def extract_text_from_file(file_path: str) -> str:
    ext = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else ''
    
    try:
        if ext == 'pdf':
            text = extract_pdf(file_path)
        elif ext in ('docx', 'doc'):
            text = extract_docx(file_path)
        elif ext in ('xlsx', 'xls'):
            text = extract_excel(file_path)
        elif ext == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        elif ext in ('pptx', 'ppt'):
            text = extract_pptx(file_path)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return ""
        
        return sanitize_text(text)
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""

def extract_pdf(file_path: str) -> str:
    text = _extract_pdf_pypdf2(file_path)
    if text and len(text.strip()) > 100:
        return text
    
    logger.info("PyPDF2 extraction insufficient, trying pdfplumber...")
    text = _extract_pdf_pdfplumber(file_path)
    if text and len(text.strip()) > 100:
        return text
    
    return ""

def _extract_pdf_pdfplumber(file_path: str) -> str:
    """Extract PDF text using pdfplumber with total time limit."""
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
                    
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}")
        return ""

def _extract_pdf_pypdf2(file_path: str) -> str:
    try:
        from PyPDF2 import PdfReader
        
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")
        return ""

def extract_docx(file_path: str) -> str:
    from docx import Document
    
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)

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
    "doc_type": "homework" | "exam" | "lecture" | "solutions" | "reading" | "syllabus" | "other",
    "assignment_number": "1" | "2" | "3" | null (if applicable),
    "instructional_unit_number": 1 | 2 | 3 | null (lecture/class/week number if mentioned),
    "instructional_unit_label": "lecture" | "class" | "week" | "module" | "session" | null,
    "course_code": "MGT404" | null (if visible),
    "year": "2024" | "2025" | null (if mentioned),
    "is_solutions": true | false (whether this contains solutions/answers)
}}

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
                    "is_solutions": False
                }

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list:
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
    
    logger.info(f"[{ta_id}] Starting indexing process...")
    
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
        
        logger.info(f"[{ta_id}] [{doc.id}] Extracting text...")
        if doc.file_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{doc.file_type}") as tmp_file:
                tmp_file.write(doc.file_content)
                tmp_path = tmp_file.name
            try:
                text = extract_text_from_file(tmp_path)
            finally:
                os.unlink(tmp_path)
        elif os.path.exists(doc.storage_path):
            text = extract_text_from_file(doc.storage_path)
        else:
            logger.warning(f"[{ta_id}] [{doc.id}] No file content available - document needs to be re-uploaded")
            continue
        
        if not text:
            logger.warning(f"[{ta_id}] [{doc.id}] No text extracted")
            continue
        
        logger.info(f"[{ta_id}] [{doc.id}] Extracted {len(text)} chars")
        
        logger.info(f"[{ta_id}] [{doc.id}] Extracting metadata with LLM...")
        metadata = extract_metadata_with_llm(text, doc.original_filename)
        doc.doc_type = metadata.get("doc_type")
        doc.assignment_number = metadata.get("assignment_number")
        doc.instructional_unit_number = metadata.get("instructional_unit_number")
        doc.instructional_unit_label = metadata.get("instructional_unit_label")
        doc.extraction_metadata = metadata
        doc.metadata_extracted = True
        logger.info(f"[{ta_id}] [{doc.id}] Saving metadata...")
        db_commit_with_retry(db)
        logger.info(f"[{ta_id}] [{doc.id}] Metadata saved")
        
        chunks = chunk_text(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        
        for i, chunk in enumerate(chunks):
            all_chunk_data.append({
                "ta_id": ta_id,
                "document_id": doc.id,
                "chunk_index": i,
                "chunk_text": chunk,
                "doc_type": doc.doc_type or "other",
                "assignment_number": doc.assignment_number or "",
                "instructional_unit_number": doc.instructional_unit_number or 0,
                "instructional_unit_label": doc.instructional_unit_label or "",
                "file_name": doc.original_filename
            })
    
    if not all_chunk_data:
        raise ValueError("No text content found in any documents")
    
    logger.info(f"[{ta_id}] Document processing complete. Total chunks to embed: {len(all_chunk_data)}")
    
    all_embeddings = []
    batch_size = 100
    total_batches = (len(all_chunk_data) + batch_size - 1) // batch_size
    
    logger.info(f"[{ta_id}] Starting embedding generation ({total_batches} batches)...")
    
    for batch_idx, i in enumerate(range(0, len(all_chunk_data), batch_size)):
        batch_texts = [c["chunk_text"] for c in all_chunk_data[i:i+batch_size]]
        
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
    
    return {"chunks_indexed": len(all_chunk_data)}
