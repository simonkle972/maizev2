import os
import logging
import secrets
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_from_directory
from config import Config
from models import db, TeachingAssistant, Document, ChatSession, ChatMessage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET") or secrets.token_hex(32)

app.config["SQLALCHEMY_DATABASE_URI"] = Config.DATABASE_URL
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

db.init_app(app)

with app.app_context():
    db.create_all()

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/admin')
def admin_panel():
    return render_template('admin.html')

@app.route('/admin/api/tas', methods=['GET'])
def list_tas():
    auth = request.headers.get('Authorization', '')
    if auth != f"Bearer {Config.ADMIN_SECRET_KEY}":
        return jsonify({"error": "Unauthorized"}), 401
    
    tas = TeachingAssistant.query.filter_by(is_active=True).all()
    return jsonify([{
        "id": ta.id,
        "slug": ta.slug,
        "name": ta.name,
        "course_name": ta.course_name,
        "is_indexed": ta.is_indexed,
        "document_count": ta.document_count,
        "created_at": ta.created_at.isoformat() if ta.created_at else None,
        "indexed_at": ta.indexed_at.isoformat() if ta.indexed_at else None
    } for ta in tas])

@app.route('/admin/api/tas', methods=['POST'])
def create_ta():
    auth = request.headers.get('Authorization', '')
    if auth != f"Bearer {Config.ADMIN_SECRET_KEY}":
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    ta_id = secrets.token_urlsafe(12)
    slug = data.get("slug", "").strip().lower().replace(" ", "-")
    
    if not slug:
        return jsonify({"error": "Slug is required"}), 400
    
    existing = TeachingAssistant.query.filter_by(slug=slug).first()
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
def get_ta(ta_id):
    auth = request.headers.get('Authorization', '')
    if auth != f"Bearer {Config.ADMIN_SECRET_KEY}":
        return jsonify({"error": "Unauthorized"}), 401
    
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
        "indexed_at": ta.indexed_at.isoformat() if ta.indexed_at else None
    })

@app.route('/admin/api/tas/<ta_id>', methods=['PUT'])
def update_ta(ta_id):
    auth = request.headers.get('Authorization', '')
    if auth != f"Bearer {Config.ADMIN_SECRET_KEY}":
        return jsonify({"error": "Unauthorized"}), 401
    
    ta = TeachingAssistant.query.get(ta_id)
    if not ta:
        return jsonify({"error": "TA not found"}), 404
    
    data = request.json
    if "name" in data:
        ta.name = data["name"]
    if "course_name" in data:
        ta.course_name = data["course_name"]
    if "system_prompt" in data:
        ta.system_prompt = data["system_prompt"]
    
    db.session.commit()
    return jsonify({"success": True})

@app.route('/admin/api/tas/<ta_id>', methods=['DELETE'])
def delete_ta(ta_id):
    auth = request.headers.get('Authorization', '')
    if auth != f"Bearer {Config.ADMIN_SECRET_KEY}":
        return jsonify({"error": "Unauthorized"}), 401
    
    ta = TeachingAssistant.query.get(ta_id)
    if not ta:
        return jsonify({"error": "TA not found"}), 404
    
    ta.is_active = False
    db.session.commit()
    return jsonify({"success": True})

@app.route('/admin/api/tas/<ta_id>/upload', methods=['POST'])
def upload_document(ta_id):
    auth = request.headers.get('Authorization', '')
    if auth != f"Bearer {Config.ADMIN_SECRET_KEY}":
        return jsonify({"error": "Unauthorized"}), 401
    
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
    
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    file.save(storage_path)
    
    file_size = os.path.getsize(storage_path)
    
    doc = Document(
        ta_id=ta_id,
        filename=safe_filename,
        original_filename=original_filename,
        file_type=file_ext,
        file_size=file_size,
        storage_path=storage_path
    )
    
    db.session.add(doc)
    ta.document_count = ta.documents.count() + 1
    ta.is_indexed = False
    db.session.commit()
    
    return jsonify({
        "success": True,
        "document_id": doc.id,
        "filename": original_filename
    })

@app.route('/admin/api/tas/<ta_id>/documents/<int:doc_id>', methods=['DELETE'])
def delete_document(ta_id, doc_id):
    auth = request.headers.get('Authorization', '')
    if auth != f"Bearer {Config.ADMIN_SECRET_KEY}":
        return jsonify({"error": "Unauthorized"}), 401
    
    doc = Document.query.filter_by(id=doc_id, ta_id=ta_id).first()
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    
    if os.path.exists(doc.storage_path):
        os.remove(doc.storage_path)
    
    ta = TeachingAssistant.query.get(ta_id)
    
    db.session.delete(doc)
    ta.document_count = max(0, ta.document_count - 1)
    ta.is_indexed = False
    db.session.commit()
    
    return jsonify({"success": True})

@app.route('/admin/api/tas/<ta_id>/reindex', methods=['POST'])
def reindex_ta(ta_id):
    auth = request.headers.get('Authorization', '')
    if auth != f"Bearer {Config.ADMIN_SECRET_KEY}":
        return jsonify({"error": "Unauthorized"}), 401
    
    ta = TeachingAssistant.query.get(ta_id)
    if not ta:
        return jsonify({"error": "TA not found"}), 404
    
    if ta.document_count == 0:
        return jsonify({"error": "No documents to index"}), 400
    
    try:
        from src.document_processor import process_and_index_documents
        result = process_and_index_documents(ta_id)
        
        ta.is_indexed = True
        ta.indexed_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({"success": True, "chunks_indexed": result.get("chunks_indexed", 0)})
    except Exception as e:
        logger.error(f"Indexing failed for {ta_id}: {e}")
        return jsonify({"error": str(e)}), 500

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
