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
        r'(?:^|\n)(Slide\s+\d+[:\s]?[^\n]{0,60})',
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
        text = _supplement_pdf_with_figures(file_path, text)
        return text, page_count

    logger.info("PyPDF2 extraction insufficient, trying pdfplumber...")
    text, page_count = _extract_pdf_pdfplumber(file_path)
    if text and len(text.strip()) > 100:
        text = _supplement_pdf_with_figures(file_path, text)
        return text, page_count

    logger.info("Text extraction insufficient - attempting vision-based extraction for image/handwritten PDF...")
    text, page_count = _extract_pdf_vision(file_path)
    if text and len(text.strip()) > 50:
        return text, page_count  # vision already described figures — skip supplement

    return "", 0


def _supplement_pdf_with_figures(file_path: str, text: str) -> str:
    """
    Run a figure-only vision pass over every PDF page and splice descriptions
    into the already-extracted text under the matching '--- Page N ---' marker.
    Pages with no figures return 'No figures' and are left untouched.
    """
    try:
        import base64
        import re as _re
        from io import BytesIO
        from pdf2image import convert_from_path
        from openai import OpenAI

        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        images = convert_from_path(file_path, dpi=200, fmt='jpeg')
        logger.info(f"Figure supplement: rendering {len(images)} pages for {file_path}")

        supplemented = text
        for page_num, img in enumerate(images, 1):
            try:
                img_buffer = BytesIO()
                img.save(img_buffer, format='JPEG', quality=85)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

                response = client.chat.completions.create(
                    model=Config.VISION_MODEL,
                    max_tokens=400,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"This is page {page_num} of a document. "
                                    "Describe ONLY charts, figures, diagrams, and images visible on this page. "
                                    "Include axis labels, trends, key data values, and notable features. "
                                    "Do NOT transcribe text that is clearly readable as prose. "
                                    "If there are no charts or figures on this page, reply exactly: No figures"
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }]
                )

                desc = response.choices[0].message.content or ""
                desc = desc.strip()
                if not desc or desc.lower().startswith("no figures"):
                    continue

                # Splice description after the matching "--- Page N ---" marker
                marker = f"--- Page {page_num} ---"
                if marker in supplemented:
                    supplemented = supplemented.replace(
                        marker,
                        f"{marker}\n[FIGURE: {desc}]",
                        1
                    )
                    logger.info(f"Figure supplement: added figure description for page {page_num}")
                else:
                    # Page marker missing (e.g. page had no text) — append at end
                    supplemented += f"\n\n{marker}\n[FIGURE: {desc}]"
                    logger.info(f"Figure supplement: appended figure description for page {page_num} (no marker found)")

            except Exception as page_e:
                logger.warning(f"Figure supplement failed on page {page_num}: {page_e}")
                continue

        return supplemented

    except ImportError as e:
        logger.warning(f"Figure supplement unavailable - missing dependency: {e}")
        return text
    except Exception as e:
        logger.warning(f"Figure supplement failed: {e}")
        return text


def _extract_pdf_vision(file_path: str) -> tuple:
    """
    Extract content from image-heavy/handwritten PDFs using GPT-4o vision.
    Converts each page to an image and sends to GPT-4o for transcription.
    Returns (text, page_count).
    """
    try:
        import base64
        from io import BytesIO
        from pdf2image import convert_from_path
        from openai import OpenAI

        client = OpenAI(api_key=Config.OPENAI_API_KEY)

        images = convert_from_path(file_path, dpi=200, fmt='jpeg')
        page_count = len(images)
        logger.info(f"Vision extraction: converted {page_count} pages to images")

        if page_count == 0:
            return "", 0

        max_pages = 50
        if page_count > max_pages:
            logger.warning(f"Vision extraction: PDF has {page_count} pages, limiting to {max_pages}")
            images = images[:max_pages]

        text_parts = []
        for page_num, img in enumerate(images, 1):
            try:
                img_buffer = BytesIO()
                img.save(img_buffer, format='JPEG', quality=85)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                img_size_kb = len(img_buffer.getvalue()) / 1024
                logger.info(f"Vision extraction: processing page {page_num}/{len(images)} ({img_size_kb:.0f}KB)")

                response = client.chat.completions.create(
                    model=Config.VISION_MODEL,
                    max_tokens=1500,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"This is page {page_num} of a lecture document. "
                                    "Transcribe ALL visible content including:\n"
                                    "- Handwritten text (preserve exact wording)\n"
                                    "- Printed text\n"
                                    "- Mathematical equations and formulas (use LaTeX notation)\n"
                                    "- Diagrams and illustrations (describe them in [DIAGRAM: ...] tags)\n"
                                    "- Tables (preserve structure)\n"
                                    "- Labels, annotations, and margin notes\n\n"
                                    "For mathematical content, use LaTeX notation like $x^2$ or $$\\int f(x) dx$$.\n"
                                    "Preserve the logical flow and structure of the content. "
                                    "If text is unclear, provide your best interpretation with [unclear] markers."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }],
                )

                page_text = response.choices[0].message.content
                if page_text and page_text.strip():
                    text_parts.append(f"--- Page {page_num} ---\n{page_text.strip()}")
                    logger.info(f"Vision extraction: page {page_num} yielded {len(page_text)} chars")
                else:
                    logger.warning(f"Vision extraction: page {page_num} returned empty content")

            except Exception as page_e:
                logger.warning(f"Vision extraction failed on page {page_num}: {page_e}")
                continue

        if not text_parts:
            return "", 0

        full_text = "\n\n".join(text_parts)
        logger.info(f"Vision extraction complete: {len(full_text)} chars from {len(text_parts)}/{page_count} pages")
        return full_text, page_count

    except ImportError as e:
        logger.warning(f"Vision extraction unavailable - missing dependency: {e}")
        return "", 0
    except Exception as e:
        logger.error(f"Vision extraction failed: {e}")
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
                        text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
                    
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
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                text_parts.append(f"--- Page {page_num} ---\n{text}")
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
        import base64
        from io import BytesIO
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE
        from openai import OpenAI

        prs = Presentation(file_path)
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        text_parts = []
        raster_types = {'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'}

        def _iter_shapes(shapes):
            for shape in shapes:
                yield shape
                if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                    yield from _iter_shapes(shape.shapes)

        for slide_num, slide in enumerate(prs.slides, 1):
            slide_text = []
            figure_descriptions = []

            for shape in _iter_shapes(slide.shapes):
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text)

                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    try:
                        img = shape.image
                        if img.content_type not in raster_types:
                            continue
                        img_base64 = base64.b64encode(img.blob).decode()
                        response = client.chat.completions.create(
                            model=Config.VISION_MODEL,
                            max_tokens=300,
                            messages=[{
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            f"Describe this chart, figure, or image from slide {slide_num} of a lecture. "
                                            "Be specific and concise — 2–4 sentences. "
                                            "Include axis labels, trends, key values, and any notable features."
                                        )
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{img.content_type};base64,{img_base64}",
                                            "detail": "high"
                                        }
                                    }
                                ]
                            }]
                        )
                        desc = response.choices[0].message.content
                        if desc and desc.strip():
                            figure_descriptions.append(f"[FIGURE: {desc.strip()}]")
                            logger.info(f"PPTX: described image on slide {slide_num}")
                    except Exception as img_e:
                        logger.warning(f"PPTX: failed to describe image on slide {slide_num}: {img_e}")

                if shape.shape_type == MSO_SHAPE_TYPE.CHART:
                    logger.info(f"PPTX: found CHART shape on slide {slide_num}")
                    try:
                        chart = shape.chart
                        title = chart.chart_title.text_frame.text if chart.has_title else ""
                        chart_type = str(chart.chart_type)
                        series_data = []
                        for plot in chart.plots:
                            for series in plot.series:
                                try:
                                    vals = [v for v in (series.values or []) if v is not None]
                                    series_data.append({"name": series.name or "Series", "values": vals[:20]})
                                except Exception:
                                    pass
                        prompt = (
                            f"Describe this chart from slide {slide_num} of a lecture in 2-4 sentences. "
                            f"Chart type: {chart_type}. Title: '{title}'. Series data: {series_data}. "
                            "Include the key trend, axis interpretation, and any notable data points."
                        )
                        response = client.chat.completions.create(
                            model=Config.VISION_MODEL,
                            max_tokens=300,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        desc = response.choices[0].message.content
                        if desc and desc.strip():
                            figure_descriptions.append(f"[CHART: {desc.strip()}]")
                            logger.info(f"PPTX: described chart on slide {slide_num}")
                    except Exception as chart_e:
                        import traceback
                        logger.warning(f"PPTX: failed to describe chart on slide {slide_num}: {chart_e}\n{traceback.format_exc()}")

            slide_parts = []
            if slide_text:
                slide_parts.append("\n".join(slide_text))
            slide_parts.extend(figure_descriptions)
            if not slide_parts:
                slide_parts.append("[Visual content only — chart or figure description unavailable]")

            text_parts.append(f"Slide {slide_num}:\n" + "\n".join(slide_parts))

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
    "content_title": "The actual document title as written in the content (e.g., 'Self-Study Problem Set #2', 'Final Exam 2024'). Extract from headers/title text, not filename.",
    "section_numbering_style": "arabic" | "roman" | "mixed" | null (how major sections/problems are numbered: "1, 2, 3" = arabic, "I, II, III" = roman)
}}

IMPORTANT: Classify by the PRIMARY document type, not whether it has solutions.
- "exam_solutions.pdf" or "final_exam_with_solutions.pdf" → doc_type: "exam", is_solutions: true
- "homework_answers.pdf" or "problem_set_solutions.pdf" → doc_type: "homework", is_solutions: true
- "lecture_notes.pdf" → doc_type: "lecture"

NUMBERING STYLE: Look at how the document labels its main sections/problems:
- If sections are "Section 1", "Problem 2", "Question 3" → section_numbering_style: "arabic"
- If sections are "Section I", "Problem II", "Part III" → section_numbering_style: "roman"
- If mixed or unclear → section_numbering_style: "mixed"

Return ONLY valid JSON, no other text."""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,
                reasoning_effort=Config.LLM_REASONING_MEDIUM
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
                    "content_title": None,
                    "section_numbering_style": None
                }


def extract_metadata_from_file_content(file_content: bytes, file_type: str, original_filename: str) -> dict:
    """
    Extract document metadata from file content at upload time.
    This runs LLM classification immediately so admins can review/edit before indexing.
    
    Returns dict with: doc_type, assignment_number, instructional_unit_number, content_title, etc.
    """
    import tempfile
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            text, _ = extract_text_from_file(tmp_path)
        finally:
            os.unlink(tmp_path)
        
        if not text or len(text.strip()) < 50:
            logger.warning(f"Could not extract sufficient text from {original_filename} for metadata extraction")
            return {
                "doc_type": None,
                "assignment_number": None,
                "instructional_unit_number": None,
                "instructional_unit_label": None,
                "content_title": None,
                "extraction_success": False
            }
        
        metadata = extract_metadata_with_llm(text, original_filename)
        metadata["extraction_success"] = True
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting metadata from {original_filename}: {e}")
        return {
            "doc_type": None,
            "assignment_number": None,
            "instructional_unit_number": None,
            "instructional_unit_label": None,
            "content_title": None,
            "extraction_success": False
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
        
        # Only update fields if they're currently empty - preserve human-edited values
        if not doc.doc_type:
            doc.doc_type = metadata.get("doc_type")
        if not doc.assignment_number:
            doc.assignment_number = metadata.get("assignment_number")
        if doc.instructional_unit_number is None:
            doc.instructional_unit_number = metadata.get("instructional_unit_number")
        if not doc.instructional_unit_label:
            doc.instructional_unit_label = metadata.get("instructional_unit_label")
        if not doc.content_title:
            doc.content_title = metadata.get("content_title")
        
        # Always store full extraction metadata for reference
        doc.extraction_metadata = metadata
        doc.metadata_extracted = True
        logger.info(f"[{ta_id}] [{doc.id}] Saving metadata (preserving human edits)...")
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
                "file_name": doc.display_name or doc.original_filename
            })
            
            index_log_entries.append({
                "ta_id": ta_id,
                "ta_slug": ta_slug,
                "file_name": doc.display_name or doc.original_filename,
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
        
        indexed_doc_ids = list(set(chunk["document_id"] for chunk in all_chunk_data))
        from datetime import datetime
        if indexed_doc_ids:
            now = datetime.utcnow()
            # Set BOTH timestamps to same value - prevents SQLAlchemy onupdate from setting updated_at later
            updated_count = Document.query.filter(Document.id.in_(indexed_doc_ids)).update(
                {"last_indexed_at": now, "updated_at": now},
                synchronize_session=False
            )
            db.session.commit()
            logger.info(f"[{ta_id}] Updated last_indexed_at for {updated_count} documents (expected {len(indexed_doc_ids)})")
        
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


def process_and_index_documents_resumable(ta_id: str, progress_callback=None, resume_from_doc_id=None, job_start_time=None) -> dict:
    """
    Resumable version of document indexing that can continue from where it left off.
    Uses document.last_indexed_at as the completion marker - if a document has chunks
    stored, it has last_indexed_at set and will be skipped on resume.
    """
    import tempfile
    from openai import OpenAI
    from datetime import datetime
    
    from models import db, Document, TeachingAssistant, DocumentChunk
    from flask import current_app
    from src.qa_logger import log_index_batch
    
    is_resume = bool(resume_from_doc_id)
    logger.info(f"[{ta_id}] Starting {'resumable' if is_resume else 'fresh'} indexing process...")
    
    ta = db.session.get(TeachingAssistant, ta_id)
    ta_slug = ta.slug if ta else "unknown"
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    all_doc_ids = [d.id for d in db.session.query(Document.id).filter_by(ta_id=ta_id).order_by(Document.id).all()]
    total_docs = len(all_doc_ids)
    
    if total_docs == 0:
        raise ValueError("No documents found for this TA")
    
    if is_resume:
        already_indexed_docs = db.session.query(Document.id, Document.last_indexed_at, Document.updated_at).filter(
            Document.ta_id == ta_id,
            Document.last_indexed_at.isnot(None)
        ).all()
        already_indexed_doc_ids = set(
            d.id for d in already_indexed_docs 
            if d.last_indexed_at and (d.updated_at is None or d.updated_at <= d.last_indexed_at)
        )
        doc_ids = [d for d in all_doc_ids if d not in already_indexed_doc_ids]
        docs_already_processed = len(already_indexed_doc_ids)
        logger.info(f"[{ta_id}] Resuming: {docs_already_processed} docs already indexed (have chunks and not modified since), {len(doc_ids)} remaining")
    else:
        logger.info(f"[{ta_id}] Fresh indexing - clearing existing chunks and reset last_indexed_at...")
        DocumentChunk.query.filter_by(ta_id=ta_id).delete()
        Document.query.filter_by(ta_id=ta_id).update({"last_indexed_at": None}, synchronize_session=False)
        db_commit_with_retry(db)
        logger.info(f"[{ta_id}] Cleared existing chunks and reset document states")
        doc_ids = all_doc_ids
        docs_already_processed = 0
    
    logger.info(f"[{ta_id}] Found {len(doc_ids)} documents to process: {doc_ids}")
    
    total_chunks_created = 0
    all_index_log_entries = []
    docs_with_content = 0
    
    for doc_idx, doc_id in enumerate(doc_ids):
        absolute_doc_idx = docs_already_processed + doc_idx
        
        doc = db.session.get(Document, doc_id)
        if not doc:
            logger.warning(f"[{ta_id}] Document {doc_id} not found, skipping")
            continue
        
        logger.info(f"[{ta_id}] Processing document [{doc.id}]: {doc.original_filename} ({absolute_doc_idx + 1}/{total_docs})")
        
        if progress_callback and total_docs > 0:
            progress = int((absolute_doc_idx / total_docs) * 80)
            progress_callback(ta_id, progress, docs_processed=absolute_doc_idx)
        
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
        
        if not doc.doc_type:
            doc.doc_type = metadata.get("doc_type")
        if not doc.assignment_number:
            doc.assignment_number = metadata.get("assignment_number")
        if doc.instructional_unit_number is None:
            doc.instructional_unit_number = metadata.get("instructional_unit_number")
        if not doc.instructional_unit_label:
            doc.instructional_unit_label = metadata.get("instructional_unit_label")
        if not doc.content_title:
            doc.content_title = metadata.get("content_title")
        
        doc.extraction_metadata = metadata
        doc.metadata_extracted = True
        logger.info(f"[{ta_id}] [{doc.id}] Saving metadata (preserving human edits)...")
        db_commit_with_retry(db)
        logger.info(f"[{ta_id}] [{doc.id}] Metadata saved")
        
        headers_found = extract_section_headers(text)
        headers_summary = "; ".join([f"{h[1][:40]}" for h in headers_found[:5]])
        logger.info(f"[{ta_id}] [{doc.id}] Found {len(headers_found)} headers: {headers_summary}")
        
        chunks = chunk_text_with_context(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP, doc.original_filename)
        num_chunks = len(chunks)
        
        doc_chunk_data = []
        doc_log_entries = []
        for i, chunk_data in enumerate(chunks):
            doc_chunk_data.append({
                "chunk_index": i,
                "chunk_text": chunk_data["original_text"],
                "chunk_text_enriched": chunk_data["text"],
                "context": chunk_data.get("context", ""),
                "doc_type": doc.doc_type or "other",
                "assignment_number": doc.assignment_number or "",
                "instructional_unit_number": doc.instructional_unit_number or 0,
                "instructional_unit_label": doc.instructional_unit_label or "",
                "file_name": doc.display_name or doc.original_filename
            })
            
            doc_log_entries.append({
                "ta_id": ta_id,
                "ta_slug": ta_slug,
                "file_name": doc.display_name or doc.original_filename,
                "doc_type": doc.doc_type or "other",
                "total_pages": page_count,
                "raw_text_length": raw_text_length,
                "chunk_index": i,
                "total_chunks": num_chunks,
                "chunk_text_length": len(chunk_data["original_text"]),
                "chunk_context": chunk_data.get("context", ""),
                "chunk_text_preview": chunk_data["original_text"][:300],
                "enriched_text_preview": chunk_data["text"][:300],
                "has_embedding": False,
                "status": "pending",
                "error_message": "",
                "headers_found": headers_summary if i == 0 else ""
            })
        
        try:
            logger.info(f"[{ta_id}] [{doc.id}] Embedding {num_chunks} chunks for this document...")
            chunk_texts = [c["chunk_text_enriched"] for c in doc_chunk_data]
            
            doc_embeddings = []
            batch_size = 100
            for batch_start in range(0, len(chunk_texts), batch_size):
                batch_texts_slice = chunk_texts[batch_start:batch_start+batch_size]
                response = client.embeddings.create(
                    model=Config.EMBEDDING_MODEL,
                    input=batch_texts_slice
                )
                doc_embeddings.extend([item.embedding for item in response.data])
            
            logger.info(f"[{ta_id}] [{doc.id}] Storing {num_chunks} chunks...")
            for i, chunk_item in enumerate(doc_chunk_data):
                chunk_obj = DocumentChunk(
                    ta_id=ta_id,
                    document_id=doc.id,
                    chunk_index=chunk_item["chunk_index"],
                    chunk_text=sanitize_text(chunk_item["chunk_text"]),
                    chunk_context=chunk_item.get("context", "")[:256] if chunk_item.get("context") else None,
                    doc_type=chunk_item["doc_type"],
                    assignment_number=chunk_item["assignment_number"],
                    instructional_unit_number=chunk_item["instructional_unit_number"],
                    instructional_unit_label=chunk_item["instructional_unit_label"],
                    file_name=chunk_item["file_name"],
                    embedding=doc_embeddings[i]
                )
                db.session.add(chunk_obj)
            
            now = datetime.utcnow()
            doc.last_indexed_at = now
            doc.updated_at = now
            db_commit_with_retry(db)
            
            total_chunks_created += num_chunks
            docs_with_content += 1
            logger.info(f"[{ta_id}] [{doc.id}] Document fully indexed with {num_chunks} chunks")
            
            for entry in doc_log_entries:
                entry["has_embedding"] = True
                entry["status"] = "success"
            all_index_log_entries.extend(doc_log_entries)
                
        except Exception as doc_error:
            logger.error(f"[{ta_id}] [{doc.id}] Failed to embed/store chunks: {doc_error}")
            db.session.rollback()
            for entry in doc_log_entries:
                entry["has_embedding"] = False
                entry["status"] = "error"
                entry["error_message"] = str(doc_error)[:500]
            all_index_log_entries.extend(doc_log_entries)
            raise
        
        if progress_callback:
            progress_callback(ta_id, int(((absolute_doc_idx + 1) / total_docs) * 100), 
                            docs_processed=absolute_doc_idx + 1,
                            chunks_created=total_chunks_created)
    
    if docs_with_content == 0:
        if is_resume:
            logger.info(f"[{ta_id}] No new documents to process - resumption complete")
            return {"chunks_indexed": 0}
        raise ValueError("No text content found in any documents")
    
    logger.info(f"[{ta_id}] Indexing complete! Total chunks: {total_chunks_created}")
    
    if all_index_log_entries:
        logger.info(f"[{ta_id}] Logging {len(all_index_log_entries)} index entries to Google Sheets...")
        log_index_batch(all_index_log_entries)
    
    return {"chunks_indexed": total_chunks_created}
