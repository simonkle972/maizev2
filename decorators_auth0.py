from functools import wraps
from flask import session, redirect, url_for, flash, request, jsonify, g
from models import User

def auth0_login_required(f):
    """Require Auth0 authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user_id = session.get('auth0_user_id')
        if not user_id:
            if request.is_json:
                return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for('auth0.login'))

        g.current_user = User.query.get(user_id)
        if not g.current_user:
            session.clear()
            return redirect(url_for('auth0.login'))

        return f(*args, **kwargs)
    return decorated

def auth0_role_required(role):
    """Require specific role (professor, student, admin)."""
    def decorator(f):
        @wraps(f)
        @auth0_login_required
        def decorated(*args, **kwargs):
            if g.current_user.role != role:
                if request.is_json:
                    return jsonify({"error": f"Access denied. {role.capitalize()}s only."}), 403
                flash(f'Access denied. {role.capitalize()}s only.', 'error')
                return redirect(url_for('landing'))
            return f(*args, **kwargs)
        return decorated
    return decorator

def auth0_professor_required(f):
    """Require professor role."""
    return auth0_role_required('professor')(f)

def auth0_student_required(f):
    """Require student role."""
    return auth0_role_required('student')(f)

def auth0_admin_required(f):
    """Require admin role."""
    return auth0_role_required('admin')(f)
