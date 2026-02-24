"""
Student authentication helpers.
Parallel session management for students, separate from Flask-Login's professor/admin sessions.

This module provides:
- login_student() - Sets session['_student_id'] without touching professor sessions
- logout_student() - Clears student session only
- current_student - Flask global proxy (like current_user but for students)
- get_current_student() - Loads student from session with caching
"""

from flask import session, g
from datetime import datetime
from werkzeug.local import LocalProxy


def login_student(user, remember=True):
    """
    Log in a student by setting session['_student_id'].
    Does NOT interfere with professor/admin sessions (session['_user_id']).

    Args:
        user: User object with role='student'
        remember: Whether to set session as permanent (default True)

    Raises:
        ValueError: If user is not a student
    """
    if user.role != 'student':
        raise ValueError(f"Cannot login non-student user as student. Role: {user.role}")

    session['_student_id'] = user.id
    session.permanent = remember
    session.modified = True  # Force session save

    # Update last login timestamp
    from models import db
    user.last_login = datetime.utcnow()
    db.session.commit()


def logout_student():
    """
    Log out the current student by removing session['_student_id'].
    Does NOT affect professor/admin sessions (session['_user_id']).
    """
    session.pop('_student_id', None)
    session.modified = True

    # Clear cached student from flask.g
    if hasattr(g, '_current_student'):
        delattr(g, '_current_student')


def get_current_student():
    """
    Load the current student from session['_student_id'].
    Returns None if no student is logged in.

    Cached in flask.g to avoid repeated database queries per request.
    Automatically verifies that the loaded user is actually a student.
    """
    if not hasattr(g, '_current_student'):
        student_id = session.get('_student_id')

        if student_id:
            from models import User
            user = User.query.get(int(student_id))

            # Verify role (safety check against data corruption)
            if user and user.role == 'student':
                g._current_student = user
            else:
                # Data corruption: student_id points to non-student or deleted user
                logout_student()
                g._current_student = None
        else:
            g._current_student = None

    return g._current_student


# Create a proxy object that behaves like current_user but for students
# This allows templates and routes to use `current_student` naturally
current_student = LocalProxy(get_current_student)
