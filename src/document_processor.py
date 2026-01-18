import os
import logging
import json
from datetime import datetime
from config import Config

logger = logging.getLogger(__name__)

def extract_text_from_file(file_path: str) -> str:
    ext = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else ''
    
    try:
        if ext == 'pdf':
            return extract_pdf(file_path)
        elif ext in ('docx', 'doc'):
            return extract_docx(file_path)
        elif ext in ('xlsx', 'xls'):
            return extract_excel(file_path)
        elif ext == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext in ('pptx', 'ppt'):
            return extract_pptx(file_path)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return ""
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""

def extract_pdf(file_path: str) -> str:
    from PyPDF2 import PdfReader
    
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n\n".join(text_parts)

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

def process_and_index_documents(ta_id: str) -> dict:
    import chromadb
    from openai import OpenAI
    
    from models import db, Document, TeachingAssistant
    from flask import current_app
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    docs_dir = f"data/courses/{ta_id}/docs"
    
    if not os.path.exists(docs_dir):
        raise ValueError(f"Documents directory not found: {docs_dir}")
    
    chroma_path = os.path.join(Config.CHROMA_DB_PATH, ta_id)
    os.makedirs(chroma_path, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    
    try:
        chroma_client.delete_collection(f"ta_{ta_id}")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name=f"ta_{ta_id}",
        metadata={"hnsw:space": "cosine"}
    )
    
    documents = Document.query.filter_by(ta_id=ta_id).all()
    
    all_chunks = []
    all_embeddings = []
    all_ids = []
    all_metadatas = []
    
    for doc in documents:
        logger.info(f"Processing document: {doc.original_filename}")
        
        text = extract_text_from_file(doc.storage_path)
        if not text:
            logger.warning(f"No text extracted from {doc.original_filename}")
            continue
        
        if not doc.metadata_extracted:
            metadata = extract_metadata_with_llm(text, doc.original_filename)
            doc.doc_type = metadata.get("doc_type")
            doc.assignment_number = metadata.get("assignment_number")
            doc.instructional_unit_number = metadata.get("instructional_unit_number")
            doc.instructional_unit_label = metadata.get("instructional_unit_label")
            doc.extraction_metadata = metadata
            doc.metadata_extracted = True
            db.session.commit()
        else:
            metadata = doc.extraction_metadata or {}
        
        chunks = chunk_text(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc.id}_{i}"
            chunk_metadata = {
                "ta_id": ta_id,
                "document_id": doc.id,
                "file_name": doc.original_filename,
                "chunk_index": i,
                "doc_type": doc.doc_type or "other",
                "assignment_number": doc.assignment_number or "",
                "instructional_unit_number": doc.instructional_unit_number or 0,
                "instructional_unit_label": doc.instructional_unit_label or ""
            }
            
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append(chunk_metadata)
    
    if not all_chunks:
        raise ValueError("No text content found in any documents")
    
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = all_chunks[i:i+batch_size]
        
        response = client.embeddings.create(
            model=Config.EMBEDDING_MODEL,
            input=batch_texts
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        ids=all_ids,
        metadatas=all_metadatas
    )
    
    logger.info(f"Indexed {len(all_chunks)} chunks for TA {ta_id}")
    
    return {"chunks_indexed": len(all_chunks)}
