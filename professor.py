"""
Professor blueprint for Maize TA.
Handles TA creation, management, document uploads, and billing.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user
from functools import wraps
from datetime import datetime
import secrets
import os
import threading

from models import db, TeachingAssistant, EnrollmentLink, Enrollment, Document, Institution, DocumentChunk, IndexingJob
from config import Config
from utils.stripe_helpers import (
    create_checkout_session,
    create_ta_from_checkout,
    pause_ta_subscription,
    resume_ta_subscription,
    create_customer_portal_session
)
from utils.email import send_contact_sales_email

professor_bp = Blueprint('professor', __name__, url_prefix='/professor')


def professor_required(f):
    """Decorator to require professor role."""
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if current_user.role != 'professor':
            if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"error": "Access denied. Professors only."}), 403
            flash('Access denied. Professors only.', 'error')
            return redirect(url_for('auth.professor_login'))
        return f(*args, **kwargs)
    return decorated_function


def professor_owns_ta(f):
    """Decorator to verify professor owns the TA."""
    @wraps(f)
    @login_required
    def decorated_function(ta_id, *args, **kwargs):
        ta = TeachingAssistant.query.get_or_404(ta_id)
        if ta.professor_id != current_user.id:
            if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"error": "Access denied. You do not own this TA."}), 403
            flash('Access denied. You do not own this TA.', 'error')
            return redirect(url_for('professor.dashboard'))
        return f(ta_id, *args, **kwargs)
    return decorated_function


@professor_bp.route('/dashboard')
@professor_required
def dashboard():
    """Professor dashboard showing all owned TAs."""
    tas = TeachingAssistant.query.filter_by(
        professor_id=current_user.id
    ).order_by(TeachingAssistant.created_at.desc()).all()

    # Calculate stats
    total_students = sum(ta.current_enrollment_count for ta in tas)
    active_tas = sum(1 for ta in tas if ta.is_available)

    return render_template('professor/dashboard.html',
                         tas=tas,
                         total_students=total_students,
                         active_tas=active_tas,
                         total_monthly_cost=current_user.total_monthly_cost,
                         billing_tiers=Config.BILLING_TIERS)


@professor_bp.route('/ta/create', methods=['GET', 'POST'])
@professor_required
def create_ta():
    """Create new TA with Stripe Checkout."""
    if request.args.get('canceled') == 'true':
        flash('Payment canceled. Your TA was not created.', 'info')

    if request.method == 'POST':
        ta_name = request.form.get('ta_name', '').strip()
        course_name = request.form.get('course_name', '').strip()
        tier = request.form.get('tier')

        # Validation
        if not all([ta_name, course_name, tier]):
            flash('TA name, course name, and tier are required', 'error')
            return render_template('professor/create_ta.html', billing_tiers=Config.BILLING_TIERS)

        # Handle custom tier - map to tier3 and send notification
        is_custom_tier = False
        if tier == 'custom':
            is_custom_tier = True
            tier = 'tier3'  # Use tier3 (Large Course, 250 students, $19.99/mo)
        elif tier not in Config.BILLING_TIERS:
            flash('Invalid billing tier', 'error')
            return render_template('professor/create_ta.html', billing_tiers=Config.BILLING_TIERS)

        # Use default system prompt
        system_prompt = "You are a helpful teaching assistant for this course. Help students understand course concepts by explaining clearly and guiding them through problems without giving direct answers."

        # Create Stripe Checkout session
        base_url = request.url_root.rstrip('/')
        checkout_url, error = create_checkout_session(
            current_user, tier, ta_name, course_name, system_prompt, base_url
        )

        if error:
            flash(f'Error creating checkout session: {error}', 'error')
            return render_template('professor/create_ta.html', billing_tiers=Config.BILLING_TIERS)

        # Send custom tier notification email if requested
        if is_custom_tier:
            from utils.email import send_custom_tier_notification
            institution_name = current_user.institution.name if current_user.institution else 'Unknown'
            send_custom_tier_notification(
                f"{current_user.first_name} {current_user.last_name}",
                current_user.email,
                institution_name,
                ta_name,
                course_name
            )

        return redirect(checkout_url)

    return render_template('professor/create_ta.html', billing_tiers=Config.BILLING_TIERS)


@professor_bp.route('/ta/create/success')
@professor_required
def create_ta_success():
    """Handle successful TA creation after Stripe Checkout."""
    session_id = request.args.get('session_id')
    if not session_id:
        flash('Invalid session', 'error')
        return redirect(url_for('professor.dashboard'))

    # Create TA from checkout session
    ta, link, error = create_ta_from_checkout(session_id)

    if error:
        flash(f'Error creating TA: {error}', 'error')
        return redirect(url_for('professor.dashboard'))

    flash(f'TA "{ta.name}" created successfully!', 'success')
    return redirect(url_for('professor.manage_ta', ta_id=ta.id))


@professor_bp.route('/ta/<ta_id>')
@professor_required
@professor_owns_ta
def manage_ta(ta_id):
    """Manage specific TA."""
    ta = TeachingAssistant.query.get_or_404(ta_id)

    # Get enrollment link
    link = EnrollmentLink.query.filter_by(ta_id=ta_id, is_active=True).first()

    # Get enrollments
    enrollments = Enrollment.query.filter_by(ta_id=ta_id).order_by(Enrollment.enrolled_at.desc()).all()

    # Get documents
    documents = Document.query.filter_by(ta_id=ta_id).order_by(Document.uploaded_at.desc()).all()

    enrollment_url = None
    if link:
        enrollment_url = url_for('auth.student_signup', token=link.token, _external=True)

    tier_info = Config.BILLING_TIERS.get(ta.billing_tier, {})

    return render_template('professor/manage_ta.html',
                         ta=ta,
                         enrollment_url=enrollment_url,
                         link=link,
                         enrollments=enrollments,
                         documents=documents,
                         tier_info=tier_info)


@professor_bp.route('/ta/<ta_id>/pause', methods=['POST'])
@professor_required
@professor_owns_ta
def pause_ta(ta_id):
    """Pause TA and stop billing."""
    ta = TeachingAssistant.query.get_or_404(ta_id)

    if ta.is_paused:
        flash('TA is already paused', 'info')
        return redirect(url_for('professor.manage_ta', ta_id=ta_id))

    if not ta.requires_billing:
        # Admin TAs don't have billing to pause
        ta.is_paused = True
        ta.paused_at = datetime.utcnow()
        db.session.commit()
        flash('TA paused successfully', 'success')
        return redirect(url_for('professor.manage_ta', ta_id=ta_id))

    success, error = pause_ta_subscription(ta)

    if error:
        flash(f'Error pausing TA: {error}', 'error')
    else:
        flash('TA paused successfully. Billing has been stopped.', 'success')

    return redirect(url_for('professor.manage_ta', ta_id=ta_id))


@professor_bp.route('/ta/<ta_id>/resume', methods=['POST'])
@professor_required
@professor_owns_ta
def resume_ta(ta_id):
    """Resume paused TA and restart billing."""
    ta = TeachingAssistant.query.get_or_404(ta_id)

    if not ta.is_paused:
        flash('TA is not paused', 'info')
        return redirect(url_for('professor.manage_ta', ta_id=ta_id))

    if not ta.requires_billing:
        # Admin TAs don't have billing to resume
        ta.is_paused = False
        ta.paused_at = None
        db.session.commit()
        flash('TA resumed successfully', 'success')
        return redirect(url_for('professor.manage_ta', ta_id=ta_id))

    success, error = resume_ta_subscription(ta)

    if error:
        flash(f'Error resuming TA: {error}', 'error')
    else:
        flash('TA resumed successfully. Billing has been restarted.', 'success')

    return redirect(url_for('professor.manage_ta', ta_id=ta_id))


@professor_bp.route('/settings')
@professor_required
def settings():
    """Professor account settings."""
    tas = TeachingAssistant.query.filter_by(professor_id=current_user.id).all()

    active_tas = [ta for ta in tas if ta.is_available and ta.requires_billing]
    paused_tas = [ta for ta in tas if ta.is_paused and ta.requires_billing]

    return render_template('professor/settings.html',
                         active_tas=active_tas,
                         paused_tas=paused_tas,
                         total_monthly_cost=current_user.total_monthly_cost,
                         billing_tiers=Config.BILLING_TIERS)


@professor_bp.route('/settings/billing')
@professor_required
def billing():
    """Detailed billing page."""
    tas = TeachingAssistant.query.filter_by(
        professor_id=current_user.id,
        requires_billing=True
    ).order_by(TeachingAssistant.created_at.desc()).all()

    return render_template('professor/billing.html',
                         tas=tas,
                         total_monthly_cost=current_user.total_monthly_cost,
                         billing_tiers=Config.BILLING_TIERS)


@professor_bp.route('/settings/billing/portal')
@professor_required
def billing_portal():
    """Redirect to Stripe Customer Portal."""
    base_url = request.url_root.rstrip('/')
    portal_url, error = create_customer_portal_session(current_user, base_url)

    if error:
        flash(f'Error accessing billing portal: {error}', 'error')
        return redirect(url_for('professor.billing'))

    return redirect(portal_url)


@professor_bp.route('/contact-sales', methods=['POST'])
@professor_required
def contact_sales():
    """Send contact sales request for custom tier."""
    message = request.form.get('message', '').strip()
    student_count = request.form.get('student_count', '')

    if not message:
        flash('Please provide a message', 'error')
        return redirect(url_for('professor.create_ta'))

    institution_name = current_user.institution.name if current_user.institution else 'Unknown'

    success, error = send_contact_sales_email(
        f"{current_user.first_name} {current_user.last_name}",
        current_user.email,
        institution_name,
        message,
        student_count
    )

    if error:
        flash('Failed to send request. Please try again later.', 'error')
    else:
        flash('Request sent! We will contact you shortly.', 'success')

    return redirect(url_for('professor.create_ta'))


# Document Management Routes

@professor_bp.route('/ta/<ta_id>/upload', methods=['POST'])
@professor_required
@professor_owns_ta
def upload_document(ta_id):
    """Upload document to TA."""
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
    db.session.commit()

    return jsonify({
        "success": True,
        "document": {
            "id": doc.id,
            "filename": doc.original_filename,
            "display_name": doc.display_name,
            "file_type": doc.file_type,
            "file_size": doc.file_size,
            "doc_type": doc.doc_type,
            "instructional_unit_number": doc.instructional_unit_number
        }
    })


@professor_bp.route('/ta/<ta_id>/documents/<int:doc_id>', methods=['PATCH'])
@professor_required
@professor_owns_ta
def update_document_metadata(ta_id, doc_id):
    """Update document metadata (display_name, doc_type, unit_number)."""
    doc = Document.query.filter_by(id=doc_id, ta_id=ta_id).first()
    if not doc:
        return jsonify({"error": "Document not found"}), 404

    data = request.get_json()

    if 'display_name' in data:
        doc.display_name = data['display_name']

    if 'doc_type' in data:
        doc.doc_type = data['doc_type'] if data['doc_type'] else None

    if 'instructional_unit_number' in data:
        value = data['instructional_unit_number']
        doc.instructional_unit_number = int(value) if value and value != '' else None

    db.session.commit()

    return jsonify({"success": True})


@professor_bp.route('/ta/<ta_id>/documents/<int:doc_id>', methods=['DELETE'])
@professor_required
@professor_owns_ta
def delete_document(ta_id, doc_id):
    """Delete document."""
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


@professor_bp.route('/ta/<ta_id>/reindex', methods=['POST'])
@professor_required
@professor_owns_ta
def reindex_ta(ta_id):
    """Trigger indexing for TA."""
    from app import run_indexing_task

    ta = TeachingAssistant.query.get(ta_id)
    if not ta:
        return jsonify({"error": "TA not found"}), 404

    if ta.document_count == 0:
        return jsonify({"error": "No documents to index"}), 400

    if ta.indexing_status == 'running':
        return jsonify({"error": "Indexing is already in progress"}), 400

    # Cancel any existing jobs
    active_jobs = IndexingJob.query.filter(
        IndexingJob.ta_id == ta_id,
        IndexingJob.status.in_(['running', 'resuming', 'pending'])
    ).all()
    for job in active_jobs:
        job.status = 'cancelled'
    if active_jobs:
        db.session.commit()

    # Start indexing in background thread
    thread = threading.Thread(target=run_indexing_task, args=(ta_id,))
    thread.daemon = True
    thread.start()

    return jsonify({"success": True, "message": "Indexing started in the background"})


@professor_bp.route('/ta/<ta_id>/indexing-status', methods=['GET'])
@professor_required
@professor_owns_ta
def indexing_status(ta_id):
    """Get current indexing status for TA."""
    ta = TeachingAssistant.query.get(ta_id)
    if not ta:
        return jsonify({"error": "TA not found"}), 404

    return jsonify({
        "status": ta.indexing_status,
        "progress": ta.indexing_progress or 0,
        "error": ta.indexing_error,
        "is_indexed": ta.is_indexed,
        "indexed_at": ta.indexed_at.isoformat() if ta.indexed_at else None
    })


@professor_bp.route('/ta/<ta_id>/delete', methods=['POST'])
@professor_required
@professor_owns_ta
def delete_ta(ta_id):
    """
    Completely delete a TA and all associated data.
    Requires double-confirmation from frontend.
    """
    ta = TeachingAssistant.query.get_or_404(ta_id)

    try:
        import shutil
        import logging
        logger = logging.getLogger(__name__)

        # 1. Cancel Stripe subscription (if exists)
        if ta.requires_billing and ta.stripe_subscription_id:
            from utils.stripe_helpers import cancel_ta_subscription
            success, error = cancel_ta_subscription(ta)
            if not success:
                logger.warning(f"Failed to cancel Stripe subscription for TA {ta_id}: {error}")
                # Continue with deletion anyway

        # 2. Delete ChromaDB directory
        chroma_path = os.path.join(Config.CHROMA_DB_PATH, ta_id)
        if os.path.exists(chroma_path):
            try:
                shutil.rmtree(chroma_path)
            except Exception as e:
                logger.warning(f"Could not delete ChromaDB for {ta_id}: {e}")

        # 3. Delete document files from disk
        docs_path = f"data/courses/{ta_id}/docs"
        if os.path.exists(docs_path):
            try:
                shutil.rmtree(docs_path)
            except Exception as e:
                logger.warning(f"Could not delete document files for {ta_id}: {e}")

        # 4. Manual cleanup of non-cascading relationships
        DocumentChunk.query.filter_by(ta_id=ta_id).delete()
        Enrollment.query.filter_by(ta_id=ta_id).delete()
        EnrollmentLink.query.filter_by(ta_id=ta_id).delete()
        IndexingJob.query.filter_by(ta_id=ta_id).delete()

        # 5. Delete the TA (cascades Documents, ChatSessions, ChatMessages)
        ta_name = ta.name
        db.session.delete(ta)
        db.session.commit()

        flash(f'TA "{ta_name}" has been permanently deleted.', 'success')
        return jsonify({"success": True})

    except Exception as e:
        db.session.rollback()
        import logging
        logging.getLogger(__name__).error(f"Error deleting TA {ta_id}: {e}")
        return jsonify({"error": str(e)}), 500
