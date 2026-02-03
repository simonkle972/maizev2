import os
import logging
import secrets
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_from_directory, Response, stream_with_context
from config import Config
from models import db, TeachingAssistant, Document, ChatSession, ChatMessage, DocumentChunk, Institution, IndexingJob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET") or secrets.token_hex(32)

app.config["SQLALCHEMY_DATABASE_URI"] = Config.DATABASE_URL
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 600,
    "pool_pre_ping": True,
    "pool_timeout": 60,
    "pool_size": 5,
    "max_overflow": 10,
    "connect_args": {
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }
}

db.init_app(app)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/demo-request', methods=['POST'])
def demo_request():
    """Handle demo request form submissions and send email notification."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    institution = data.get('institution', '').strip()
    course = data.get('course', '').strip()
    message = data.get('message', '').strip()
    
    if not name or not email:
        return jsonify({"error": "Name and email are required"}), 400
    
    email_body = f"""
New Demo Request for Maize

Name: {name}
Email: {email}
Institution: {institution}
Course: {course or 'Not specified'}

Message:
{message or 'No additional message'}

---
Submitted at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
"""
    
    smtp_host = os.environ.get('SMTP_HOST')
    smtp_port = int(os.environ.get('SMTP_PORT', 587))
    smtp_user = os.environ.get('SMTP_USER')
    smtp_pass = os.environ.get('SMTP_PASS')
    
    if smtp_host and smtp_user and smtp_pass:
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_user
            msg['To'] = 'simon.kleffner@yale.edu'
            msg['Subject'] = f'Maize Demo Request: {name} from {institution}'
            msg.attach(MIMEText(email_body, 'plain'))
            
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            
            logger.info(f"Demo request email sent for {email}")
        except Exception as e:
            logger.error(f"Failed to send demo request email: {e}")
            logger.info(f"Demo request (email failed): {name} <{email}> from {institution}")
    else:
        logger.info(f"Demo request (no SMTP configured): {name} <{email}> from {institution}")
        logger.info(f"Course: {course or 'N/A'}, Message: {message or 'N/A'}")
    
    return jsonify({"success": True, "message": "Demo request received"})

@app.route('/health')
def health_check():
    return 'OK', 200

@app.route('/')
def landing():
    return redirect('/mgt422')

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

@app.route('/admin/api/institutions', methods=['GET'])
@admin_api_required
def list_institutions():
    institutions = Institution.query.order_by(Institution.name).all()
    return jsonify([{
        "id": inst.id,
        "name": inst.name,
        "customer_id": inst.customer_id,
        "notes": inst.notes,
        "ta_count": inst.teaching_assistants.count(),
        "created_at": inst.created_at.isoformat() if inst.created_at else None
    } for inst in institutions])

@app.route('/admin/api/institutions', methods=['POST'])
@admin_api_required
def create_institution():
    data = request.json
    name = data.get("name", "").strip()
    
    if not name:
        return jsonify({"error": "Name is required"}), 400
    
    customer_id = data.get("customer_id", "").strip() or None
    if customer_id:
        existing = Institution.query.filter_by(customer_id=customer_id).first()
        if existing:
            return jsonify({"error": "Customer ID already exists"}), 400
    
    inst = Institution(
        name=name,
        customer_id=customer_id,
        notes=data.get("notes", "").strip() or None
    )
    
    db.session.add(inst)
    db.session.commit()
    
    return jsonify({
        "id": inst.id,
        "name": inst.name,
        "customer_id": inst.customer_id,
        "notes": inst.notes,
        "ta_count": 0,
        "created_at": inst.created_at.isoformat() if inst.created_at else None
    })

@app.route('/admin/api/institutions/<int:inst_id>', methods=['GET'])
@admin_api_required
def get_institution(inst_id):
    inst = Institution.query.get(inst_id)
    if not inst:
        return jsonify({"error": "Institution not found"}), 404
    
    return jsonify({
        "id": inst.id,
        "name": inst.name,
        "customer_id": inst.customer_id,
        "notes": inst.notes,
        "ta_count": inst.teaching_assistants.count(),
        "created_at": inst.created_at.isoformat() if inst.created_at else None,
        "updated_at": inst.updated_at.isoformat() if inst.updated_at else None
    })

@app.route('/admin/api/institutions/<int:inst_id>', methods=['PUT'])
@admin_api_required
def update_institution(inst_id):
    inst = Institution.query.get(inst_id)
    if not inst:
        return jsonify({"error": "Institution not found"}), 404
    
    data = request.json
    
    if "name" in data:
        name = data["name"].strip()
        if not name:
            return jsonify({"error": "Name cannot be empty"}), 400
        inst.name = name
    
    if "customer_id" in data:
        customer_id = data["customer_id"].strip() or None
        if customer_id and customer_id != inst.customer_id:
            existing = Institution.query.filter_by(customer_id=customer_id).first()
            if existing:
                return jsonify({"error": "Customer ID already exists"}), 400
        inst.customer_id = customer_id
    
    if "notes" in data:
        inst.notes = data["notes"].strip() or None
    
    db.session.commit()
    
    return jsonify({
        "id": inst.id,
        "name": inst.name,
        "customer_id": inst.customer_id,
        "notes": inst.notes,
        "ta_count": inst.teaching_assistants.count(),
        "created_at": inst.created_at.isoformat() if inst.created_at else None,
        "updated_at": inst.updated_at.isoformat() if inst.updated_at else None
    })

@app.route('/admin/api/institutions/<int:inst_id>', methods=['DELETE'])
@admin_api_required
def delete_institution(inst_id):
    inst = Institution.query.get(inst_id)
    if not inst:
        return jsonify({"error": "Institution not found"}), 404
    
    if inst.teaching_assistants.count() > 0:
        return jsonify({"error": "Cannot delete institution with associated TAs. Remove TAs first or reassign them."}), 400
    
    db.session.delete(inst)
    db.session.commit()
    
    return jsonify({"success": True})

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
        "indexing_status": ta.indexing_status,
        "institution_id": ta.institution_id,
        "institution_name": ta.institution.name if ta.institution else None,
        "last_activity_at": ta.last_activity_at.isoformat() if ta.last_activity_at else None
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
    
    institution_id = data.get("institution_id")
    if institution_id:
        institution_id = int(institution_id)
        inst = Institution.query.get(institution_id)
        if not inst:
            return jsonify({"error": "Institution not found"}), 400
    
    ta = TeachingAssistant(
        id=ta_id,
        slug=slug,
        name=data.get("name", "New TA"),
        course_name=data.get("course_name", ""),
        system_prompt=data.get("system_prompt", TeachingAssistant.system_prompt.default.arg),
        institution_id=institution_id if institution_id else None
    )
    
    db.session.add(ta)
    db.session.commit()
    
    os.makedirs(f"data/courses/{ta_id}/docs", exist_ok=True)
    
    return jsonify({
        "id": ta_id,
        "slug": slug,
        "name": ta.name,
        "course_name": ta.course_name,
        "institution_id": ta.institution_id,
        "institution_name": ta.institution.name if ta.institution else None
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
        "display_name": doc.display_name or doc.original_filename,
        "file_type": doc.file_type,
        "doc_type": doc.doc_type,
        "unit_number": doc.instructional_unit_number,
        "assignment_number": doc.assignment_number,
        "content_title": doc.content_title,
        "metadata_extracted": doc.metadata_extracted,
        "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        "last_indexed_at": doc.last_indexed_at.isoformat() if doc.last_indexed_at else None,
        "needs_reindex": (doc.updated_at and doc.last_indexed_at and doc.updated_at > doc.last_indexed_at) or (doc.updated_at and not doc.last_indexed_at)
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
        "indexing_progress": ta.indexing_progress,
        "institution_id": ta.institution_id,
        "institution_name": ta.institution.name if ta.institution else None,
        "last_activity_at": ta.last_activity_at.isoformat() if ta.last_activity_at else None
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
    if "institution_id" in data:
        institution_id = data["institution_id"]
        if institution_id is None or institution_id == "":
            ta.institution_id = None
        else:
            institution_id = int(institution_id)
            inst = Institution.query.get(institution_id)
            if not inst:
                return jsonify({"error": "Institution not found"}), 400
            ta.institution_id = institution_id
    
    db.session.commit()
    return jsonify({
        "success": True, 
        "slug": ta.slug,
        "institution_id": ta.institution_id,
        "institution_name": ta.institution.name if ta.institution else None
    })

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
    from src.document_processor import extract_metadata_from_file_content
    
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
    
    metadata = extract_metadata_from_file_content(file_content, file_ext, original_filename)
    
    display_name = original_filename
    if file_ext:
        display_name = original_filename.rsplit('.', 1)[0]
    
    doc = Document(
        ta_id=ta_id,
        filename=safe_filename,
        original_filename=original_filename,
        display_name=display_name,
        file_type=file_ext,
        file_size=file_size,
        storage_path=storage_path,
        file_content=file_content,
        doc_type=metadata.get("doc_type"),
        assignment_number=metadata.get("assignment_number"),
        instructional_unit_number=metadata.get("instructional_unit_number"),
        instructional_unit_label=metadata.get("instructional_unit_label"),
        content_title=metadata.get("content_title"),
        metadata_extracted=metadata.get("extraction_success", False),
        extraction_metadata=metadata
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
        "filename": original_filename,
        "display_name": doc.display_name,
        "doc_type": doc.doc_type,
        "unit_number": doc.instructional_unit_number,
        "content_title": doc.content_title,
        "metadata_extracted": doc.metadata_extracted
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


@app.route('/admin/api/tas/<ta_id>/documents/<int:doc_id>', methods=['PATCH'])
@admin_api_required
def update_document_metadata(ta_id, doc_id):
    """Update document metadata (display_name, doc_type, unit_number) before indexing."""
    doc = Document.query.filter_by(id=doc_id, ta_id=ta_id).first()
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    
    data = request.json
    
    if "display_name" in data:
        doc.display_name = data["display_name"].strip() if data["display_name"] else doc.original_filename
    
    if "doc_type" in data:
        valid_doc_types = ["homework", "exam", "lecture", "reading", "syllabus", "other"]
        if data["doc_type"] in valid_doc_types or data["doc_type"] is None:
            doc.doc_type = data["doc_type"]
    
    if "unit_number" in data:
        if data["unit_number"] is None or data["unit_number"] == "":
            doc.instructional_unit_number = None
        else:
            try:
                doc.instructional_unit_number = int(data["unit_number"])
            except (ValueError, TypeError):
                pass
    
    if "assignment_number" in data:
        doc.assignment_number = data["assignment_number"] if data["assignment_number"] else None
    
    ta = TeachingAssistant.query.get(ta_id)
    if ta:
        ta.is_indexed = False
        ta.indexing_status = None
    
    db.session.commit()
    
    return jsonify({
        "success": True,
        "document": {
            "id": doc.id,
            "display_name": doc.display_name,
            "doc_type": doc.doc_type,
            "unit_number": doc.instructional_unit_number,
            "assignment_number": doc.assignment_number
        }
    })


def update_indexing_progress(ta_id, progress, job_id=None, docs_processed=None, chunks_created=None):
    """Update indexing progress for a TA (called from document_processor)."""
    with app.app_context():
        ta = TeachingAssistant.query.get(ta_id)
        if ta:
            ta.indexing_progress = progress
            db.session.commit()
        
        if job_id:
            job = IndexingJob.query.get(job_id)
            if job:
                if docs_processed is not None:
                    job.docs_processed = docs_processed
                if chunks_created is not None:
                    job.chunks_created = chunks_created
                db.session.commit()

def run_indexing_task(ta_id, job_id=None, is_resume=False):
    """Background task to run document indexing with job tracking for resumption."""
    import traceback
    
    with app.app_context():
        logger.info(f"[{ta_id}] Background indexing task started (job_id={job_id}, is_resume={is_resume})")
        
        ta = TeachingAssistant.query.get(ta_id)
        if not ta:
            logger.error(f"[{ta_id}] TA not found, aborting indexing")
            return
        
        job = None
        if job_id:
            job = IndexingJob.query.get(job_id)
        
        if not job:
            job = IndexingJob(
                ta_id=ta_id,
                status='running',
                started_at=datetime.utcnow(),
                total_docs=ta.document_count
            )
            db.session.add(job)
            db.session.commit()
            job_id = job.id
        else:
            job.status = 'running'
            job.started_at = datetime.utcnow()
            db.session.commit()
        
        try:
            ta.indexing_status = 'running'
            ta.indexing_error = None
            if not is_resume:
                ta.indexing_progress = 0
                ta.is_indexed = False
            db.session.commit()
            logger.info(f"[{ta_id}] Set status to running, starting document processing...")
            
            from src.document_processor import process_and_index_documents_resumable
            
            def progress_with_job(ta_id, progress, docs_processed=None, chunks_created=None):
                update_indexing_progress(ta_id, progress, job_id, docs_processed, chunks_created)
            
            result = process_and_index_documents_resumable(
                ta_id, 
                progress_callback=progress_with_job,
                resume_from_doc_id=is_resume  # Pass is_resume flag to trigger resume logic
            )
            
            db.session.expire_all()
            ta = TeachingAssistant.query.get(ta_id)
            ta.is_indexed = True
            ta.indexed_at = datetime.utcnow()
            ta.indexing_status = 'completed'
            ta.indexing_progress = 100
            db.session.commit()
            
            job = IndexingJob.query.get(job_id)
            if job:
                job.status = 'completed'
                job.completed_at = datetime.utcnow()
                job.chunks_created = result.get('chunks_indexed', 0)
                db.session.commit()
            
            logger.info(f"[{ta_id}] Indexing completed successfully: {result.get('chunks_indexed', 0)} chunks")
            
        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()
            logger.error(f"[{ta_id}] Indexing failed with error: {error_msg}")
            logger.error(f"[{ta_id}] Traceback:\n{tb}")
            
            try:
                db.session.rollback()
                db.session.expire_all()
                ta = TeachingAssistant.query.get(ta_id)
                if ta:
                    ta.indexing_status = 'failed'
                    ta.indexing_error = error_msg[:500]
                    ta.is_indexed = False
                    db.session.commit()
                
                job = IndexingJob.query.get(job_id)
                if job:
                    job.status = 'failed'
                    job.error_message = error_msg[:500]
                    db.session.commit()
                    
                logger.info(f"[{ta_id}] Set indexing status to failed")
            except Exception as e2:
                logger.error(f"[{ta_id}] Failed to update status after error: {e2}")


def resume_interrupted_indexing_jobs():
    """Check for interrupted indexing jobs and resume them on startup.
    Uses atomic status update to prevent multiple workers from resuming the same job."""
    with app.app_context():
        try:
            orphaned_tas = TeachingAssistant.query.filter(
                TeachingAssistant.indexing_status == 'running'
            ).all()
            
            for ta in orphaned_tas:
                active_job = IndexingJob.query.filter(
                    IndexingJob.ta_id == ta.id,
                    IndexingJob.status.in_(['running', 'resuming', 'pending'])
                ).first()
                
                if not active_job:
                    logger.info(f"[{ta.id}] Found orphaned TA with indexing_status='running' but no active job - creating resume job...")
                    now = datetime.utcnow()
                    new_job = IndexingJob(
                        ta_id=ta.id,
                        status='resuming',
                        started_at=now,
                        total_docs=ta.document_count,
                        docs_processed=0,
                        chunks_created=0,
                        created_at=now,
                        updated_at=now
                    )
                    db.session.add(new_job)
                    db.session.commit()
                    
                    thread = threading.Thread(
                        target=run_indexing_task, 
                        args=(ta.id, new_job.id, True)
                    )
                    thread.daemon = True
                    thread.start()
                    logger.info(f"[{ta.id}] Started resume job {new_job.id} for orphaned TA")
            
            interrupted_jobs = IndexingJob.query.filter(
                IndexingJob.status.in_(['running', 'resuming'])
            ).all()
            
            for job in interrupted_jobs:
                result = db.session.execute(
                    db.text("""
                        UPDATE indexing_jobs 
                        SET status = 'resuming', updated_at = NOW()
                        WHERE id = :job_id AND status = 'running'
                        RETURNING id
                    """),
                    {"job_id": job.id}
                )
                db.session.commit()
                
                row = result.fetchone()
                if row:
                    logger.info(f"[{job.ta_id}] Acquired lock on interrupted indexing job {job.id}, resuming...")
                    
                    thread = threading.Thread(
                        target=run_indexing_task, 
                        args=(job.ta_id, job.id, True)
                    )
                    thread.daemon = True
                    thread.start()
                elif job.status == 'resuming':
                    stale_threshold_seconds = 300
                    if job.updated_at and (datetime.utcnow() - job.updated_at).total_seconds() > stale_threshold_seconds:
                        stale_result = db.session.execute(
                            db.text("""
                                UPDATE indexing_jobs 
                                SET status = 'resuming', updated_at = NOW()
                                WHERE id = :job_id 
                                  AND status = 'resuming' 
                                  AND updated_at < NOW() - INTERVAL '5 minutes'
                                RETURNING id
                            """),
                            {"job_id": job.id}
                        )
                        db.session.commit()
                        stale_row = stale_result.fetchone()
                        if stale_row:
                            logger.info(f"[{job.ta_id}] Re-acquired stale job {job.id} (was in 'resuming' for over 5 minutes)...")
                            thread = threading.Thread(
                                target=run_indexing_task, 
                                args=(job.ta_id, job.id, True)
                            )
                            thread.daemon = True
                            thread.start()
                        else:
                            logger.info(f"[{job.ta_id}] Job {job.id} stale check failed (likely claimed by another worker)")
                    else:
                        logger.info(f"[{job.ta_id}] Job {job.id} is actively being resumed by another worker, skipping")
                else:
                    logger.info(f"[{job.ta_id}] Job {job.id} status changed, skipping")
                
        except Exception as e:
            logger.error(f"Error checking for interrupted indexing jobs: {e}")
            import traceback
            logger.error(traceback.format_exc())

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
    
    active_jobs = IndexingJob.query.filter(
        IndexingJob.ta_id == ta_id,
        IndexingJob.status.in_(['running', 'resuming', 'pending'])
    ).all()
    for job in active_jobs:
        job.status = 'cancelled'
    if active_jobs:
        db.session.commit()
        logger.info(f"[{ta_id}] Cancelled {len(active_jobs)} existing indexing jobs before starting fresh")
    
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
        import time
        from src.retriever import retrieve_context
        from src.response_generator import generate_response
        from src.qa_logger import log_qa_entry
        
        start_time = time.time()
        
        recent_messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.created_at.desc()).limit(10).all()
        conversation_history = list(reversed(recent_messages))
        
        retrieval_start = time.time()
        chunks, retrieval_diagnostics = retrieve_context(ta.id, query, top_k=8, conversation_history=conversation_history, session_id=session_id)
        retrieval_latency_ms = int((time.time() - retrieval_start) * 1000)
        
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
        
        generation_start = time.time()
        hybrid_mode = retrieval_diagnostics.get("hybrid_fallback_triggered", False)
        hybrid_doc_filename = retrieval_diagnostics.get("hybrid_doc_filename")
        query_reference = retrieval_diagnostics.get("validation_expected_ref")
        
        response_text = generate_response(
            query=query,
            context=context,
            system_prompt=ta.system_prompt,
            conversation_history=history_text,
            course_name=ta.course_name,
            hybrid_mode=hybrid_mode,
            hybrid_doc_filename=hybrid_doc_filename,
            query_reference=query_reference
        )
        generation_latency_ms = int((time.time() - generation_start) * 1000)
        
        user_message = ChatMessage(session_id=session_id, role="user", content=query)
        sources = [c['file_name'] for c in chunks[:3]]
        assistant_message = ChatMessage(
            session_id=session_id, 
            role="assistant", 
            content=response_text,
            sources=sources
        )
        db.session.add(user_message)
        db.session.add(assistant_message)
        chat_session.last_activity = datetime.utcnow()
        db.session.commit()
        
        total_latency_ms = int((time.time() - start_time) * 1000)
        token_count = len(response_text.split())
        
        log_qa_entry(
            ta_id=str(ta.id),
            ta_slug=ta.slug,
            ta_name=ta.name,
            course_name=ta.course_name,
            session_id=session_id,
            query=query,
            answer=response_text,
            sources=sources,
            chunk_count=len(chunks),
            latency_ms=total_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            token_count=token_count,
            retrieval_diagnostics=retrieval_diagnostics
        )
        
        return jsonify({
            "response": response_text,
            "session_id": session_id,
            "sources": sources
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
            chunks, retrieval_diagnostics = retrieve_context(ta_id, query, top_k=8, conversation_history=conversation_history, session_id=session_id)
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
            
            hybrid_mode = retrieval_diagnostics.get("hybrid_fallback_triggered", False)
            hybrid_doc_filename = retrieval_diagnostics.get("hybrid_doc_filename")
            query_reference = retrieval_diagnostics.get("validation_expected_ref")
            attempt_count = retrieval_diagnostics.get("attempt_count", 0)
            
            generation_start = time.time()
            for token in generate_response_stream(
                query=query,
                context=context,
                system_prompt=ta_system_prompt,
                conversation_history=history_text,
                course_name=ta_course_name,
                hybrid_mode=hybrid_mode,
                hybrid_doc_filename=hybrid_doc_filename,
                query_reference=query_reference,
                attempt_count=attempt_count
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
                token_count=token_count,
                retrieval_diagnostics=retrieval_diagnostics
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

with app.app_context():
    db.create_all()
    threading.Timer(2.0, resume_interrupted_indexing_jobs).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
