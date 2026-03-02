"""
Professor blueprint for Maize TA.
Handles TA creation, management, document uploads, and billing.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, session
from flask_login import login_required, current_user
from functools import wraps
from datetime import datetime
import secrets
import os
import threading

from models import db, TeachingAssistant, EnrollmentLink, Enrollment, Document, Institution, InstitutionDomain, DocumentChunk, IndexingJob
from utils.validators import match_institution_by_email
from config import Config
from utils.stripe_helpers import (
    create_publish_checkout_session,
    activate_ta_from_checkout,
    pause_ta_subscription,
    resume_ta_subscription,
    create_customer_portal_session,
    generate_slug
)
from utils.email import send_contact_sales_email, send_support_request_email
from extensions import limiter

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
    """Create new TA as a draft (no payment required)."""
    if request.method == 'POST':
        if not current_user.email_verified:
            flash('Please verify your email before creating a TA. Check your inbox for a verification email.', 'warning')
            return redirect(url_for('professor.dashboard'))

        ta_name = request.form.get('ta_name', '').strip()
        course_name = request.form.get('course_name', '').strip()

        if not ta_name or not course_name:
            flash('TA name and course name are required', 'error')
            return render_template('professor/create_ta.html')

        from utils.stripe_helpers import generate_slug
        ta_id = secrets.token_urlsafe(12)
        slug = generate_slug(ta_name)
        system_prompt = "You are a helpful teaching assistant for this course. Help students understand course concepts by explaining clearly and guiding them through problems without giving direct answers."

        ta = TeachingAssistant(
            id=ta_id,
            slug=slug,
            name=ta_name,
            course_name=course_name,
            system_prompt=system_prompt,
            professor_id=current_user.id,
            institution_id=current_user.institution_id,
            status='draft',
            requires_billing=True,
            allow_anonymous_chat=False,
        )
        db.session.add(ta)
        db.session.commit()

        flash(f'TA "{ta.name}" created! Upload your course materials, then publish when ready.', 'success')
        return redirect(url_for('professor.manage_ta', ta_id=ta.id))

    return render_template('professor/create_ta.html')


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
        enrollment_url = url_for('auth0.enroll_via_link', token=link.token, _external=True)

    tier_info = Config.BILLING_TIERS.get(ta.billing_tier, {})

    today = datetime.utcnow().strftime('%Y-%m-%d')
    test_chat_used = session.get(f'tc_{ta_id}_{today}', 0)

    return render_template('professor/manage_ta.html',
                         ta=ta,
                         enrollment_url=enrollment_url,
                         link=link,
                         enrollments=enrollments,
                         documents=documents,
                         tier_info=tier_info,
                         billing_tiers=Config.BILLING_TIERS,
                         test_chat_used=test_chat_used)


@professor_bp.route('/ta/<ta_id>/publish', methods=['GET', 'POST'])
@professor_required
@professor_owns_ta
def publish_ta(ta_id):
    """Tier selection and Stripe checkout to publish a draft TA."""
    ta = TeachingAssistant.query.get_or_404(ta_id)

    if ta.status != 'draft':
        flash('This TA is already published.', 'info')
        return redirect(url_for('professor.manage_ta', ta_id=ta_id))

    if request.args.get('canceled') == 'true':
        flash('Payment canceled. Your TA is still in draft.', 'info')

    if request.method == 'POST':
        tier = request.form.get('tier')
        is_custom_tier = (tier == 'custom')
        if is_custom_tier:
            tier = 'tier3'
        elif tier not in Config.BILLING_TIERS:
            flash('Invalid billing tier', 'error')
            return render_template('professor/publish.html', ta=ta, billing_tiers=Config.BILLING_TIERS)

        base_url = request.url_root.rstrip('/')
        checkout_url, error = create_publish_checkout_session(ta, tier, base_url)

        if error:
            flash(f'Error starting checkout: {error}', 'error')
            return render_template('professor/publish.html', ta=ta, billing_tiers=Config.BILLING_TIERS)

        if is_custom_tier:
            from utils.email import send_custom_tier_notification
            institution_name = current_user.institution.name if current_user.institution else 'Unknown'
            send_custom_tier_notification(
                f"{current_user.first_name} {current_user.last_name}",
                current_user.email,
                institution_name,
                ta.name,
                ta.course_name
            )

        return redirect(checkout_url)

    return render_template('professor/publish.html', ta=ta, billing_tiers=Config.BILLING_TIERS)


@professor_bp.route('/ta/<ta_id>/publish/success')
@professor_required
@professor_owns_ta
def publish_ta_success(ta_id):
    """Handle successful publish after Stripe Checkout."""
    session_id = request.args.get('session_id')
    if not session_id:
        flash('Invalid session', 'error')
        return redirect(url_for('professor.manage_ta', ta_id=ta_id))

    ta, link, error = activate_ta_from_checkout(session_id)

    if error:
        flash(f'Error publishing TA: {error}', 'error')
        return redirect(url_for('professor.manage_ta', ta_id=ta_id))

    flash(f'"{ta.name}" is now live! Share the enrollment link with your students.', 'success')
    return redirect(url_for('professor.manage_ta', ta_id=ta.id))


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
        ta.status = 'paused'
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
        ta.status = 'active'
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


@professor_bp.route('/ta/<ta_id>/change-tier', methods=['GET', 'POST'])
@professor_required
@professor_owns_ta
def change_ta_tier(ta_id):
    """Show tier selection page; handle tier change (upgrade or downgrade) on POST."""
    ta = TeachingAssistant.query.get_or_404(ta_id)

    if not ta.requires_billing or ta.status == 'draft':
        flash('Tier changes are not available for this TA.', 'error')
        return redirect(url_for('professor.manage_ta', ta_id=ta_id))

    if request.method == 'POST':
        new_tier = request.form.get('tier')

        if new_tier == 'custom':
            from utils.email import send_custom_tier_notification
            institution_name = current_user.institution.name if current_user.institution else 'Unknown'
            send_custom_tier_notification(
                f"{current_user.first_name} {current_user.last_name}",
                current_user.email,
                institution_name,
                ta.name,
                ta.course_name,
            )
            flash('Thanks! Our team will reach out within 24 hours to discuss custom pricing.', 'success')
            return redirect(url_for('professor.manage_ta', ta_id=ta_id))

        if new_tier not in Config.BILLING_TIERS:
            flash('Invalid billing tier.', 'error')
            return redirect(url_for('professor.change_ta_tier', ta_id=ta_id))

        if new_tier == ta.billing_tier:
            flash('That is already your current plan.', 'info')
            return redirect(url_for('professor.manage_ta', ta_id=ta_id))

        # Downgrade guard: cannot drop below current enrollment count
        new_cap = Config.BILLING_TIERS[new_tier]['max_students']
        current_count = ta.current_enrollment_count
        if new_cap < current_count:
            flash(
                f'Cannot downgrade: {current_count} students are enrolled but the '
                f'{Config.BILLING_TIERS[new_tier]["name"]} tier only supports {new_cap}.',
                'error'
            )
            return redirect(url_for('professor.change_ta_tier', ta_id=ta_id))

        from utils.stripe_helpers import change_ta_subscription
        success, error = change_ta_subscription(ta, new_tier)

        if success:
            flash(f'Plan changed to {Config.BILLING_TIERS[new_tier]["name"]}.', 'success')
        else:
            flash(f'Plan change failed: {error}', 'error')

        return redirect(url_for('professor.manage_ta', ta_id=ta_id))

    # GET — render tier selection page
    enrollment_count = ta.current_enrollment_count
    return render_template('professor/change_tier.html',
                         ta=ta,
                         billing_tiers=Config.BILLING_TIERS,
                         enrollment_count=enrollment_count)


@professor_bp.route('/ta/<ta_id>/test-chat/stream', methods=['POST'])
@limiter.limit("20 per day")
@professor_required
@professor_owns_ta
def test_chat_stream(ta_id):
    """
    Professor test chat — streams TA response for the professor's own testing.
    Logs to QA analytics with is_preview=True so it can be filtered later.
    Does not require TA to be active/published.
    Guards: must be indexed and not currently indexing.
    """
    from flask import stream_with_context, Response
    from src.qa_logger import log_qa_entry
    import json
    import time

    ta = TeachingAssistant.query.get_or_404(ta_id)

    if not ta.is_indexed:
        return jsonify(error="Documents must be indexed before testing."), 400
    if ta.indexing_status == 'running':
        return jsonify(error="Indexing is in progress. Please wait until it completes."), 400

    data = request.get_json(silent=True) or {}
    query = (data.get('query') or '').strip()
    conversation_history = (data.get('conversation_history') or '')

    if not query:
        return jsonify(error="Query is required."), 400

    session_id = f"prof_test_{secrets.token_hex(8)}"

    def generate():
        from src.retriever import retrieve_context
        from src.response_generator import generate_response_stream
        total_start = time.time()
        try:
            yield f"data: {json.dumps({'type': 'status', 'content': 'Searching course materials...'})}\n\n"

            retrieval_start = time.time()
            chunks, retrieval_diagnostics = retrieve_context(
                ta.id, query, top_k=8, conversation_history=conversation_history
            )
            retrieval_latency_ms = int((time.time() - retrieval_start) * 1000)
            chunk_count = len(chunks)

            context = "\n\n---\n\n".join([
                f"[From: {c['file_name']}]\n{c['text']}" for c in chunks
            ])
            sources = [c['file_name'] for c in chunks[:3]]

            hybrid_mode = retrieval_diagnostics.get("hybrid_fallback_triggered", False)
            hybrid_doc_filename = retrieval_diagnostics.get("hybrid_doc_filename")
            query_reference = retrieval_diagnostics.get("validation_expected_ref")
            attempt_count = retrieval_diagnostics.get("attempt_count", 0)

            yield f"data: {json.dumps({'type': 'status', 'content': 'Generating response...'})}\n\n"

            if sources:
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            full_response = ''
            token_count = 0
            gen_start = time.time()
            for token in generate_response_stream(
                query=query,
                context=context,
                system_prompt=ta.system_prompt or '',
                conversation_history=conversation_history,
                course_name=ta.course_name or '',
                hybrid_mode=hybrid_mode,
                hybrid_doc_filename=hybrid_doc_filename,
                query_reference=query_reference,
                attempt_count=attempt_count,
            ):
                full_response += token
                token_count += 1
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            generation_latency_ms = int((time.time() - gen_start) * 1000)
            total_latency_ms = int((time.time() - total_start) * 1000)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            log_qa_entry(
                ta_id=str(ta.id),
                ta_slug=ta.slug or '',
                ta_name=ta.name,
                course_name=ta.course_name or '',
                session_id=session_id,
                query=query,
                answer=full_response,
                sources=sources,
                chunk_count=chunk_count,
                latency_ms=total_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                token_count=token_count,
                retrieval_diagnostics=retrieval_diagnostics,
                llm_model=Config.LLM_MODEL,
                is_anonymous=False,
                is_preview=True,
            )

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    # Increment session-based day counter so page reload shows correct count
    today = datetime.utcnow().strftime('%Y-%m-%d')
    session[f'tc_{ta_id}_{today}'] = session.get(f'tc_{ta_id}_{today}', 0) + 1

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@professor_bp.route('/settings')
@professor_required
def settings():
    """Professor account settings."""
    tas = TeachingAssistant.query.filter_by(professor_id=current_user.id).all()

    active_tas = [ta for ta in tas if ta.status == 'active' and ta.requires_billing]
    paused_tas = [ta for ta in tas if ta.status == 'paused' and ta.requires_billing]

    return render_template('professor/settings.html',
                         active_tas=active_tas,
                         paused_tas=paused_tas,
                         total_monthly_cost=current_user.total_monthly_cost,
                         billing_tiers=Config.BILLING_TIERS)


@professor_bp.route('/settings/billing')
@professor_required
def billing():
    """Detailed billing page."""
    tas = TeachingAssistant.query.filter(
        TeachingAssistant.professor_id == current_user.id,
        TeachingAssistant.requires_billing == True,
        TeachingAssistant.status != 'draft'
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


@professor_bp.route('/api/institutions')
@professor_required
def institutions_search():
    """JSON endpoint for institution autocomplete. Returns up to 10 matches."""
    q = request.args.get('q', '').strip()
    if len(q) < 2:
        return jsonify([])
    results = Institution.query.filter(
        Institution.name.ilike(f'%{q}%')
    ).order_by(Institution.name).limit(10).all()
    return jsonify([{'id': i.id, 'name': i.name, 'country': i.country or ''} for i in results])


@professor_bp.route('/onboarding', methods=['GET', 'POST'])
@professor_required
def onboarding():
    """Post-signup onboarding: collect name and institution for new Auth0 professors."""
    if current_user.onboarding_complete:
        return redirect(url_for('professor.dashboard'))

    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        institution_id = request.form.get('institution_id', '').strip()
        new_institution_name = request.form.get('new_institution_name', '').strip()
        auto_verified = request.form.get('auto_verified', 'false') == 'true'

        if not first_name or not last_name:
            flash('First and last name are required.', 'error')
        elif not institution_id and not new_institution_name:
            flash('Please select or create an institution.', 'error')
        else:
            if institution_id == 'new' and new_institution_name:
                domain = current_user.email.split('@')[-1].lower()
                institution = Institution(name=new_institution_name, email_domain=domain)
                db.session.add(institution)
                db.session.flush()
                auto_verified = False  # User-created institution cannot be auto-verified
            elif institution_id and institution_id != 'new':
                institution = Institution.query.get(int(institution_id))
            else:
                flash('Invalid institution selection.', 'error')
                suggested = match_institution_by_email(current_user.email)
                return render_template('professor/onboarding.html', suggested_institution=suggested)

            current_user.first_name = first_name
            current_user.last_name = last_name
            current_user.institution_id = institution.id
            current_user.onboarding_complete = True
            # Always verify server-side — catches both confirmed suggestions and
            # manual re-selection of the matched institution
            email_match = match_institution_by_email(current_user.email)
            if email_match and email_match.id == institution.id:
                current_user.institution_verified = True
                current_user.verification_domain = current_user.email.split('@')[-1].lower()
            else:
                current_user.institution_verified = False
                current_user.verification_domain = None
            db.session.commit()
            return redirect(url_for('professor.dashboard'))

    suggested = match_institution_by_email(current_user.email)
    return render_template('professor/onboarding.html', suggested_institution=suggested)


@professor_bp.route('/profile', methods=['GET', 'POST'])
@professor_required
def profile():
    """Professor profile page — view and edit name, institution, and verification status."""
    if request.method == 'POST':
        action = request.form.get('action', 'save')

        if action == 'verify':
            # Verify affiliation via a secondary institutional email
            verification_email = request.form.get('verification_email', '').strip().lower()
            if not verification_email or '@' not in verification_email:
                flash('Please enter a valid email address.', 'error')
            elif not current_user.institution:
                flash('Please set your institution first.', 'error')
            else:
                matched = match_institution_by_email(verification_email)
                if matched and matched.id == current_user.institution_id:
                    current_user.institution_verified = True
                    current_user.verification_domain = verification_email.split('@')[-1].lower()
                    db.session.commit()
                    flash('Affiliation verified!', 'success')
                elif matched:
                    flash(
                        f'That email is associated with {matched.name}, not your current institution. '
                        'Change your institution first if you want to switch.',
                        'error'
                    )
                else:
                    flash('Email domain not recognized — verification unsuccessful.', 'error')
            return redirect(url_for('professor.profile'))

        # Default: save profile
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        institution_id = request.form.get('institution_id', '').strip()
        new_institution_name = request.form.get('new_institution_name', '').strip()

        if not first_name or not last_name:
            flash('First and last name are required.', 'error')
        elif not institution_id and not new_institution_name:
            flash('Please select or create an institution.', 'error')
        else:
            if institution_id == 'new' and new_institution_name:
                domain = current_user.email.split('@')[-1].lower()
                institution = Institution(name=new_institution_name, email_domain=domain)
                db.session.add(institution)
                db.session.flush()
            elif institution_id and institution_id != 'new':
                institution = Institution.query.get(int(institution_id))
            else:
                flash('Invalid institution selection.', 'error')
                return render_template('professor/profile.html')

            current_user.first_name = first_name
            current_user.last_name = last_name

            # If institution changed, reset verification and re-check via primary email
            if institution.id != current_user.institution_id:
                current_user.institution_id = institution.id
                current_user.institution_verified = False
                current_user.verification_domain = None
                # Auto-verify if primary email matches the new institution
                matched = match_institution_by_email(current_user.email)
                if matched and matched.id == institution.id:
                    current_user.institution_verified = True
                    current_user.verification_domain = current_user.email.split('@')[-1].lower()
            else:
                current_user.institution_id = institution.id

            db.session.commit()
            flash('Profile updated successfully.', 'success')
            return redirect(url_for('professor.profile'))

    return render_template('professor/profile.html')


@professor_bp.route('/support', methods=['GET', 'POST'])
@professor_required
def support():
    """Support request form — sends email to simon@getmaize.ai."""
    if request.method == 'POST':
        subject_line = request.form.get('subject', '').strip()
        message = request.form.get('message', '').strip()

        if not subject_line or not message:
            flash('Please fill in both fields.', 'error')
        else:
            institution_name = current_user.institution.name if current_user.institution else 'Unknown'
            send_support_request_email(
                f"{current_user.first_name} {current_user.last_name}",
                current_user.email,
                institution_name,
                subject_line,
                message
            )
            flash("Message sent — we'll get back to you shortly.", 'success')
            return redirect(url_for('professor.support'))

    return render_template('professor/support.html')


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

    allowed_extensions = {
        'pdf', 'docx', 'doc', 'xlsx', 'xls', 'txt', 'pptx', 'ppt',
        'png', 'jpg', 'jpeg', 'gif', 'webp',
        'md', 'py', 'json', 'csv', 'ipynb',
    }
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
