"""
Student blueprint for Maize TA.
Handles student dashboard, enrolled TAs, and chat access.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from functools import wraps
from datetime import datetime
import logging

from models import db, Enrollment, TeachingAssistant, ChatSession
from auth_student import current_student
from config import Config

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

        # Skip unavailable TAs (draft or paused)
        if not ta.is_available:
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
    if ta_id and not current_student.email_verified:
        flash('Please verify your email to access TAs.', 'warning')
        return redirect(url_for('student.dashboard'))
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
    """Authenticated streaming chat API for enrolled students. Delegates to shared helper."""
    from src.chat_streaming import stream_chat_response

    if not current_student.email_verified:
        return jsonify({"error": "Please verify your email address to use the AI TA. Check your inbox for a verification email."}), 403

    ta = TeachingAssistant.query.get_or_404(ta_id)

    if not ta.is_available:
        return jsonify({"error": "This course is currently unavailable."}), 400

    if not ta.is_indexed:
        return jsonify({"error": "This teaching assistant is not ready yet. Please check back later."}), 400

    data = request.json
    query = data.get('query', '').strip()
    session_id = data.get('session_id', '')

    if not query:
        return jsonify({"error": "Query required"}), 400

    return stream_chat_response(
        ta=ta,
        query=query,
        session_id=session_id,
        user_id=current_student.id,
        is_anonymous=False,
    )
