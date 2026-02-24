"""
Student blueprint for Maize TA.
Handles student dashboard, enrolled TAs, and chat access.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, Response, stream_with_context
from functools import wraps
from datetime import datetime
import secrets
import logging

from models import db, Enrollment, TeachingAssistant, ChatSession, ChatMessage
from auth_student import current_student

logger = logging.getLogger(__name__)

student_bp = Blueprint('student', __name__, url_prefix='/student')


def student_required(f):
    """
    Decorator to require student authentication.
    Redirects to /student/login on auth failure.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_student:
            # Not logged in as student
            if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"error": "Student authentication required"}), 401
            flash('Please log in as a student to access this page.', 'error')
            return redirect(url_for('auth.student_login'))

        if not current_student.is_active:
            flash('Your account is inactive. Please contact support.', 'error')
            from auth_student import logout_student
            logout_student()
            return redirect(url_for('auth.student_login'))

        return f(*args, **kwargs)
    return decorated_function


def student_enrolled_in_ta(f):
    """Decorator to verify student is enrolled in the TA."""
    @wraps(f)
    @student_required
    def decorated_function(ta_id, *args, **kwargs):
        enrollment = Enrollment.query.filter_by(
            student_id=current_student.id,
            ta_id=ta_id
        ).first()

        if not enrollment:
            flash('You are not enrolled in this course.', 'error')
            return redirect(url_for('student.dashboard'))

        return f(ta_id, *args, **kwargs)
    return decorated_function


@student_bp.route('/dashboard')
@student_bp.route('/dashboard/<ta_id>')
@student_required
def dashboard(ta_id=None):
    """Student dashboard showing all enrolled TAs with embedded chat."""
    # Load all enrolled TAs
    enrollments = Enrollment.query.filter_by(
        student_id=current_student.id
    ).all()

    tas_with_sessions = []
    for enrollment in enrollments:
        ta = enrollment.ta

        # Skip inactive or paused TAs
        if not ta.is_active or ta.is_paused:
            continue

        # Get last session for this TA
        last_session = ChatSession.query.filter_by(
            ta_id=ta.id,
            user_id=current_student.id
        ).order_by(ChatSession.created_at.desc()).first()

        # Determine the timestamp for sorting
        if last_session:
            last_message_time = last_session.last_activity or last_session.created_at
            is_new = False
        else:
            last_message_time = enrollment.enrolled_at
            is_new = True

        tas_with_sessions.append({
            'ta': ta,
            'enrollment': enrollment,
            'last_session': last_session,
            'last_message_time': last_message_time,
            'is_new': is_new
        })

    # Sort by most recent first
    tas_with_sessions.sort(key=lambda x: x['last_message_time'], reverse=True)

    # Handle selected TA
    selected_ta = None
    if ta_id:
        selected_ta = TeachingAssistant.query.get_or_404(ta_id)
        # Verify enrollment
        enrollment = Enrollment.query.filter_by(
            student_id=current_student.id,
            ta_id=ta_id
        ).first()

        if not enrollment:
            flash('You are not enrolled in this course.', 'error')
            return redirect(url_for('student.dashboard'))

        # Update first accessed timestamp if needed
        if not enrollment.first_accessed_at:
            enrollment.first_accessed_at = datetime.utcnow()
            db.session.commit()

    return render_template('student/dashboard.html',
                         tas=tas_with_sessions,
                         selected_ta=selected_ta)


@student_bp.route('/ta/<ta_id>/access')
@student_required
@student_enrolled_in_ta
def access_ta(ta_id):
    """Record first access to TA and redirect to chat."""
    ta = TeachingAssistant.query.get_or_404(ta_id)

    # Check if TA is available
    if not ta.is_available:
        flash('This course is currently unavailable.', 'error')
        return redirect(url_for('student.dashboard'))

    # Update first accessed timestamp
    enrollment = Enrollment.query.filter_by(
        student_id=current_student.id,
        ta_id=ta_id
    ).first()

    if enrollment and not enrollment.first_accessed_at:
        enrollment.first_accessed_at = datetime.utcnow()
        db.session.commit()

    # Redirect to dashboard with this TA selected
    return redirect(url_for('student.dashboard', ta_id=ta_id))


@student_bp.route('/ta/<ta_id>/sessions')
@student_required
@student_enrolled_in_ta
def ta_sessions(ta_id):
    """View chat history for a TA."""
    ta = TeachingAssistant.query.get_or_404(ta_id)

    sessions = ChatSession.query.filter_by(
        ta_id=ta_id,
        user_id=current_student.id
    ).order_by(ChatSession.last_activity.desc()).all()

    return render_template('student/sessions.html',
                         ta=ta,
                         sessions=sessions)


@student_bp.route('/ta/<ta_id>/api/chat/stream', methods=['POST'])
@student_required
@student_enrolled_in_ta
def chat_stream(ta_id):
    """Authenticated streaming chat API for students."""
    import json

    ta = TeachingAssistant.query.get_or_404(ta_id)

    # Check if TA is available
    if not ta.is_available:
        return jsonify({"error": "This course is currently unavailable."}), 400

    if not ta.is_indexed:
        return jsonify({"error": "This teaching assistant is not ready yet. Please check back later."}), 400

    data = request.json
    query = data.get('query', '').strip()
    session_id = data.get('session_id', '')

    if not query:
        return jsonify({"error": "Query required"}), 400

    # Create or get chat session
    if not session_id:
        session_id = secrets.token_urlsafe(16)
        chat_session = ChatSession(id=session_id, ta_id=ta.id, user_id=current_student.id)
        db.session.add(chat_session)
        db.session.commit()
    else:
        chat_session = ChatSession.query.filter_by(id=session_id, ta_id=ta.id).first()
        if not chat_session:
            session_id = secrets.token_urlsafe(16)
            chat_session = ChatSession(id=session_id, ta_id=ta.id, user_id=current_student.id)
            db.session.add(chat_session)
            db.session.commit()

    # Save user message
    user_message = ChatMessage(session_id=session_id, role="user", content=query)
    db.session.add(user_message)
    db.session.commit()

    # Store TA info for use in generator
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
            from src.response_generator import generate_response_stream, escape_hash_in_latex

            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching course materials...'})}\n\n"

            # Get recent conversation history
            recent_messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.created_at.desc()).limit(10).all()
            conversation_history = list(reversed(recent_messages))

            history_text = ""
            if conversation_history:
                history_parts = []
                for msg in conversation_history[-6:]:
                    role = "Student" if msg.role == "user" else "Assistant"
                    history_parts.append(f"{role}: {msg.content[:300]}...")
                history_text = "\n".join(history_parts)

            # Retrieve context
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

            # Get hybrid mode info
            hybrid_mode = retrieval_diagnostics.get("hybrid_fallback_triggered", False)
            hybrid_doc_filename = retrieval_diagnostics.get("hybrid_doc_filename")
            query_reference = retrieval_diagnostics.get("validation_expected_ref")
            attempt_count = retrieval_diagnostics.get("attempt_count", 0)

            # Generate streaming response
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

            # Sanitize LaTeX # characters before saving
            full_response = escape_hash_in_latex(full_response)

            # Save assistant message
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

            # Log QA entry
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
            logger.error(f"Streaming chat error for TA {ta_id}: {e}")
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
