import os
import logging
import secrets
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_from_directory, Response, stream_with_context
from config import Config
from models import db, TeachingAssistant, Document, ChatSession, ChatMessage, DocumentChunk

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET") or secrets.token_hex(32)

app.config["SQLALCHEMY_DATABASE_URI"] = Config.DATABASE_URL
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 180,
    "pool_pre_ping": True,
    "pool_timeout": 30,
    "pool_size": 5,
    "max_overflow": 10,
}

db.init_app(app)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def landing():
    return render_template('landing.html')

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin')
@admin_required
def admin_panel():
    return render_template('admin.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if session.get('admin_logged_in'):
        return redirect(url_for('admin_panel'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            return render_template('admin_login.html', error='Please enter both username and password')
        
        valid_username = Config.ADMIN_USERNAME
        valid_password = Config.ADMIN_PASSWORD
        
        if not valid_username or not valid_password:
            return render_template('admin_login.html', error='Admin credentials not configured')
        
        if username == valid_username and password == valid_password:
            session['admin_logged_in'] = True
            session.permanent = True
            return redirect(url_for('admin_panel'))
        else:
            return render_template('admin_login.html', error='Invalid username or password')
    
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

def admin_api_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/api/tas', methods=['GET'])
@admin_api_required
def list_tas():
    tas = TeachingAssistant.query.filter_by(is_active=True).all()
    return jsonify([{
        "id": ta.id,
        "slug": ta.slug,
        "name": ta.name,
        "course_name": ta.course_name,
        "is_indexed": ta.is_indexed,
        "document_count": ta.document_count,
        "created_at": ta.created_at.isoformat() if ta.created_at else None,
        "indexed_at": ta.indexed_at.isoformat() if ta.indexed_at else None,
        "indexing_status": ta.indexing_status
    } for ta in tas])

@app.route('/admin/api/tas', methods=['POST'])
@admin_api_required
def create_ta():
    data = request.json
    ta_id = secrets.token_urlsafe(12)
    slug = data.get("slug", "").strip().lower().replace(" ", "-")
    
    if not slug:
        return jsonify({"error": "Slug is required"}), 400
    
    existing = TeachingAssistant.query.filter_by(slug=slug, is_active=True).first()
    if existing:
        return jsonify({"error": "Slug already exists"}), 400
    
    ta = TeachingAssistant(
        id=ta_id,
        slug=slug,
        name=data.get("name", "New TA"),
        course_name=data.get("course_name", ""),
        system_prompt=data.get("system_prompt", TeachingAssistant.system_prompt.default.arg)
    )
    
    db.session.add(ta)
    db.session.commit()
    
    os.makedirs(f"data/courses/{ta_id}/docs", exist_ok=True)
    
    return jsonify({
        "id": ta_id,
        "slug": slug,
        "name": ta.name,
        "course_name": ta.course_name
    })

@app.route('/admin/api/tas/<ta_id>', methods=['GET'])
@admin_api_required
def get_ta(ta_id):
    ta = TeachingAssistant.query.get(ta_id)
    if not ta:
        return jsonify({"error": "TA not found"}), 404
    
    documents = [{
        "id": doc.id,
        "filename": doc.original_filename,
        "file_type": doc.file_type,
        "doc_type": doc.doc_type,
        "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None
    } for doc in ta.documents]
    
    return jsonify({
        "id": ta.id,
        "slug": ta.slug,
        "name": ta.name,
        "course_name": ta.course_name,
        "system_prompt": ta.system_prompt,
        "is_indexed": ta.is_indexed,
        "document_count": ta.document_count,
        "documents": documents,
        "created_at": ta.created_at.isoformat() if ta.created_at else None,
        "indexed_at": ta.indexed_at.isoformat() if ta.indexed_at else None,
        "indexing_status": ta.indexing_status,
        "indexing_error": ta.indexing_error,
        "indexing_progress": ta.indexing_progress
    })

@app.route('/admin/api/tas/<ta_id>', methods=['PUT'])
@admin_api_required
def update_ta(ta_id):
    ta = TeachingAssistant.query.get(ta_id)
    if not ta:
        return jsonify({"error": "TA not found"}), 404
    
    data = request.json
    
    if "slug" in data:
        import re
        new_slug = re.sub(r'\s+', '-', data["slug"].strip().lower())
        if not new_slug:
            return jsonify({"error": "Slug cannot be empty"}), 400
        if not re.match(r'^[a-z0-9-]+$', new_slug):
            return jsonify({"error": "URL path can only contain lowercase letters, numbers, and hyphens"}), 400
        if new_slug != ta.slug:
            existing = TeachingAssistant.query.filter_by(slug=new_slug, is_active=True).first()
            if existing:
                return jsonify({"error": "This URL path is already in use"}), 400
            ta.slug = new_slug
    
    if "name" in data:
        ta.name = data["name"]
    if "course_name" in data:
        ta.course_name = data["course_name"]
    if "system_prompt" in data:
        ta.system_prompt = data["system_prompt"]
    
    db.session.commit()
    return jsonify({"success": True, "slug": ta.slug})

@app.route('/admin/api/tas/<ta_id>', methods=['DELETE'])
@admin_api_required
def delete_ta(ta_id):
    ta = TeachingAssistant.query.get(ta_id)
    if not ta:
        return jsonify({"error": "TA not found"}), 404
    
    try:
        chroma_path = os.path.join(Config.CHROMA_DB_PATH, ta_id)
        if os.path.exists(chroma_path):
            import shutil
            shutil.rmtree(chroma_path)
    except Exception as e:
        logger.warning(f"Could not delete ChromaDB for {ta_id}: {e}")
    
    db.session.delete(ta)
    db.session.commit()
    return jsonify({"success": True})

@app.route('/admin/api/cleanup-slug', methods=['POST'])
@admin_api_required
def cleanup_orphaned_slug():
    """Force-delete any TA with a given slug (including inactive ones)."""
    data = request.json
    slug = data.get("slug", "").strip().lower()
    
    if not slug:
        return jsonify({"error": "Slug is required"}), 400
    
    tas = TeachingAssistant.query.filter_by(slug=slug).all()
    if not tas:
        return jsonify({"error": "No TA found with that slug"}), 404
    
    deleted_count = 0
    for ta in tas:
        try:
            chroma_path = os.path.join(Config.CHROMA_DB_PATH, ta.id)
            if os.path.exists(chroma_path):
                import shutil
                shutil.rmtree(chroma_path)
        except Exception as e:
            logger.warning(f"Could not delete ChromaDB for {ta.id}: {e}")
        
        db.session.delete(ta)
        deleted_count += 1
    
    db.session.commit()
    return jsonify({"success": True, "deleted_count": deleted_count})

@app.route('/admin/api/tas/<ta_id>/upload', methods=['POST'])
@admin_api_required
def upload_document(ta_id):
    ta = TeachingAssistant.query.get(ta_id)
    if not ta:
        return jsonify({"error": "TA not found"}), 404
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    original_filename = file.filename
    file_ext = original_filename.rsplit('.', 1)[-1].lower() if '.' in original_filename else ''
    
    allowed_extensions = {'pdf', 'docx', 'doc', 'xlsx', 'xls', 'txt', 'pptx', 'ppt'}
    if file_ext not in allowed_extensions:
        return jsonify({"error": f"File type .{file_ext} not supported"}), 400
    
    safe_filename = f"{secrets.token_urlsafe(8)}_{original_filename}"
    storage_path = f"data/courses/{ta_id}/docs/{safe_filename}"
    
    file_content = file.read()
    file_size = len(file_content)
    
    doc = Document(
        ta_id=ta_id,
        filename=safe_filename,
        original_filename=original_filename,
        file_type=file_ext,
        file_size=file_size,
        storage_path=storage_path,
        file_content=file_content
    )
    
    db.session.add(doc)
    db.session.flush()
    
    ta.document_count = Document.query.filter_by(ta_id=ta_id).count()
    ta.is_indexed = False
    ta.indexing_status = None
    ta.indexing_error = None
    db.session.commit()
    
    return jsonify({
        "success": True,
        "document_id": doc.id,
        "filename": original_filename
    })

@app.route('/admin/api/tas/<ta_id>/documents/<int:doc_id>', methods=['DELETE'])
@admin_api_required
def delete_document(ta_id, doc_id):
    doc = Document.query.filter_by(id=doc_id, ta_id=ta_id).first()
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    
    if os.path.exists(doc.storage_path):
        os.remove(doc.storage_path)
    
    ta = TeachingAssistant.query.get(ta_id)
    
    DocumentChunk.query.filter_by(document_id=doc_id).delete()
    
    db.session.delete(doc)
    db.session.flush()
    
    ta.document_count = Document.query.filter_by(ta_id=ta_id).count()
    ta.is_indexed = False
    ta.indexing_status = None
    ta.indexing_error = None
    db.session.commit()
    
    return jsonify({"success": True})

def update_indexing_progress(ta_id, progress):
    """Update indexing progress for a TA (called from document_processor)."""
    with app.app_context():
        ta = TeachingAssistant.query.get(ta_id)
        if ta:
            ta.indexing_progress = progress
            db.session.commit()

def run_indexing_task(ta_id):
    """Background task to run document indexing."""
    with app.app_context():
        ta = TeachingAssistant.query.get(ta_id)
        if not ta:
            return
        
        try:
            ta.indexing_status = 'running'
            ta.indexing_error = None
            ta.indexing_progress = 0
            ta.is_indexed = False
            db.session.commit()
            
            from src.document_processor import process_and_index_documents
            result = process_and_index_documents(ta_id, progress_callback=update_indexing_progress)
            
            ta = TeachingAssistant.query.get(ta_id)
            ta.is_indexed = True
            ta.indexed_at = datetime.utcnow()
            ta.indexing_status = 'completed'
            ta.indexing_progress = 100
            db.session.commit()
            logger.info(f"Indexing completed for {ta_id}: {result.get('chunks_indexed', 0)} chunks")
            
        except Exception as e:
            logger.error(f"Indexing failed for {ta_id}: {e}")
            ta = TeachingAssistant.query.get(ta_id)
            if ta:
                ta.indexing_status = 'failed'
                ta.indexing_error = str(e)
                ta.is_indexed = False
                db.session.commit()

@app.route('/admin/api/tas/<ta_id>/reindex', methods=['POST'])
@admin_api_required
def reindex_ta(ta_id):
    ta = TeachingAssistant.query.get(ta_id)
    if not ta:
        return jsonify({"error": "TA not found"}), 404
    
    if ta.document_count == 0:
        return jsonify({"error": "No documents to index"}), 400
    
    if ta.indexing_status == 'running':
        return jsonify({"error": "Indexing is already in progress"}), 400
    
    thread = threading.Thread(target=run_indexing_task, args=(ta_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({"success": True, "message": "Indexing started in the background"})

@app.route('/admin/api/tas/<ta_id>/indexing-status', methods=['GET'])
@admin_api_required
def get_indexing_status(ta_id):
    ta = TeachingAssistant.query.get(ta_id)
    if not ta:
        return jsonify({"error": "TA not found"}), 404
    
    return jsonify({
        "status": ta.indexing_status,
        "progress": ta.indexing_progress,
        "error": ta.indexing_error,
        "is_indexed": ta.is_indexed,
        "indexed_at": ta.indexed_at.isoformat() if ta.indexed_at else None
    })

@app.route('/admin/api/tas/<ta_id>/reset-indexing', methods=['POST'])
@admin_api_required
def reset_indexing_status(ta_id):
    ta = TeachingAssistant.query.get(ta_id)
    if not ta:
        return jsonify({"error": "TA not found"}), 404
    
    ta.indexing_status = None
    ta.indexing_progress = 0
    ta.indexing_error = None
    db.session.commit()
    
    return jsonify({"success": True, "message": "Indexing status reset"})

@app.route('/admin/api/test-qa-logging', methods=['GET'])
@admin_api_required
def test_qa_logging():
    from src.qa_logger import test_connection
    result = test_connection()
    return jsonify(result)

@app.route('/<slug>')
def ta_chat(slug):
    ta = TeachingAssistant.query.filter_by(slug=slug, is_active=True).first()
    if not ta:
        return render_template('404.html'), 404
    
    return render_template('chat.html', ta=ta)

@app.route('/<slug>/api/chat', methods=['POST'])
def chat_api(slug):
    ta = TeachingAssistant.query.filter_by(slug=slug, is_active=True).first()
    if not ta:
        return jsonify({"error": "TA not found"}), 404
    
    if not ta.is_indexed:
        return jsonify({"error": "This teaching assistant is not ready yet. Please check back later."}), 400
    
    data = request.json
    query = data.get('query', '').strip()
    session_id = data.get('session_id', '')
    
    if not query:
        return jsonify({"error": "Query required"}), 400
    
    if not session_id:
        session_id = secrets.token_urlsafe(16)
        chat_session = ChatSession(id=session_id, ta_id=ta.id)
        db.session.add(chat_session)
        db.session.commit()
    else:
        chat_session = ChatSession.query.filter_by(id=session_id, ta_id=ta.id).first()
        if not chat_session:
            session_id = secrets.token_urlsafe(16)
            chat_session = ChatSession(id=session_id, ta_id=ta.id)
            db.session.add(chat_session)
            db.session.commit()
    
    try:
        from src.retriever import retrieve_context
        from src.response_generator import generate_response
        
        recent_messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.created_at.desc()).limit(10).all()
        conversation_history = list(reversed(recent_messages))
        
        chunks = retrieve_context(ta.id, query, top_k=8)
        
        context = "\n\n---\n\n".join([
            f"[From: {c['file_name']}]\n{c['text']}" 
            for c in chunks
        ])
        
        history_text = ""
        if conversation_history:
            history_parts = []
            for msg in conversation_history[-6:]:
                role = "Student" if msg.role == "user" else "Assistant"
                history_parts.append(f"{role}: {msg.content[:300]}...")
            history_text = "\n".join(history_parts)
        
        response_text = generate_response(
            query=query,
            context=context,
            system_prompt=ta.system_prompt,
            conversation_history=history_text,
            course_name=ta.course_name
        )
        
        user_message = ChatMessage(session_id=session_id, role="user", content=query)
        assistant_message = ChatMessage(
            session_id=session_id, 
            role="assistant", 
            content=response_text,
            sources=[c['file_name'] for c in chunks[:3]]
        )
        db.session.add(user_message)
        db.session.add(assistant_message)
        chat_session.last_activity = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            "response": response_text,
            "session_id": session_id,
            "sources": [c['file_name'] for c in chunks[:3]]
        })
        
    except Exception as e:
        logger.error(f"Chat error for {slug}: {e}")
        return jsonify({"error": "An error occurred processing your question. Please try again."}), 500

@app.route('/<slug>/api/chat/stream', methods=['POST'])
def chat_stream_api(slug):
    import json
    
    ta = TeachingAssistant.query.filter_by(slug=slug, is_active=True).first()
    if not ta:
        return jsonify({"error": "TA not found"}), 404
    
    if not ta.is_indexed:
        return jsonify({"error": "This teaching assistant is not ready yet. Please check back later."}), 400
    
    data = request.json
    query = data.get('query', '').strip()
    session_id = data.get('session_id', '')
    
    if not query:
        return jsonify({"error": "Query required"}), 400
    
    if not session_id:
        session_id = secrets.token_urlsafe(16)
        chat_session = ChatSession(id=session_id, ta_id=ta.id)
        db.session.add(chat_session)
        db.session.commit()
    else:
        chat_session = ChatSession.query.filter_by(id=session_id, ta_id=ta.id).first()
        if not chat_session:
            session_id = secrets.token_urlsafe(16)
            chat_session = ChatSession(id=session_id, ta_id=ta.id)
            db.session.add(chat_session)
            db.session.commit()
    
    user_message = ChatMessage(session_id=session_id, role="user", content=query)
    db.session.add(user_message)
    db.session.commit()
    
    ta_id = ta.id
    ta_slug = ta.slug
    ta_name = ta.name
    ta_system_prompt = ta.system_prompt
    ta_course_name = ta.course_name
    
    def generate():
        import time
        from src.qa_logger import log_qa_entry
        
        start_time = time.time()
        retrieval_latency_ms = 0
        generation_latency_ms = 0
        chunk_count = 0
        sources = []
        full_response = ""
        
        try:
            from src.retriever import retrieve_context
            from src.response_generator import generate_response_stream
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching course materials...'})}\n\n"
            
            recent_messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.created_at.desc()).limit(10).all()
            conversation_history = list(reversed(recent_messages))
            
            history_text = ""
            if conversation_history:
                history_parts = []
                for msg in conversation_history[-6:]:
                    role = "Student" if msg.role == "user" else "Assistant"
                    history_parts.append(f"{role}: {msg.content[:300]}...")
                history_text = "\n".join(history_parts)
            
            retrieval_start = time.time()
            chunks = retrieve_context(ta_id, query, top_k=8)
            retrieval_latency_ms = int((time.time() - retrieval_start) * 1000)
            chunk_count = len(chunks)
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing relevant content...'})}\n\n"
            
            context = "\n\n---\n\n".join([
                f"[From: {c['file_name']}]\n{c['text']}" 
                for c in chunks
            ])
            
            sources = [c['file_name'] for c in chunks[:3]]
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"
            
            generation_start = time.time()
            for token in generate_response_stream(
                query=query,
                context=context,
                system_prompt=ta_system_prompt,
                conversation_history=history_text,
                course_name=ta_course_name
            ):
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            generation_latency_ms = int((time.time() - generation_start) * 1000)
            
            chat_session_update = ChatSession.query.get(session_id)
            assistant_message = ChatMessage(
                session_id=session_id, 
                role="assistant", 
                content=full_response,
                sources=sources
            )
            db.session.add(assistant_message)
            if chat_session_update:
                chat_session_update.last_activity = datetime.utcnow()
            db.session.commit()
            
            total_latency_ms = int((time.time() - start_time) * 1000)
            token_count = len(full_response.split())
            
            log_qa_entry(
                ta_id=str(ta_id),
                ta_slug=ta_slug,
                ta_name=ta_name,
                course_name=ta_course_name,
                session_id=session_id,
                query=query,
                answer=full_response,
                sources=sources,
                chunk_count=chunk_count,
                latency_ms=total_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                token_count=token_count
            )
            
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming chat error for {slug}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'message': 'An error occurred processing your question.'})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
