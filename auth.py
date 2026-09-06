"""
Authentication blueprint for Maize TA.

Only the admin login lives here. Professors and students authenticate through
Auth0 (see auth_auth0.py); their local email/password routes, the signup routes
and the password reset flow were removed once Auth0 became the real login path.
Admin is deliberately NOT on Auth0 -- it authenticates against User.check_password
so we retain a way in that does not depend on an external provider.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, current_user

from models import User
from auth_student import logout_student
from extensions import limiter

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/logout')
def logout():
    """
    Log out current user (professor or admin).
    For student logout, use /student/logout instead.
    """
    logout_user()
    flash('You have been logged out.', 'success')

    # Check for next URL parameter (validate it's a relative path to prevent open redirect)
    from urllib.parse import urlparse
    next_url = request.args.get('next')
    if next_url and urlparse(next_url).netloc == '':
        return redirect(next_url)

    return redirect(url_for('landing'))


@auth_bp.route('/student/logout')
def student_logout():
    """
    Log out student session only.
    Professor sessions are unaffected (parallel session system).
    """
    logout_student()
    flash('You have been logged out.', 'success')
    return redirect(url_for('landing'))


@auth_bp.route('/admin/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def admin_login():
    """
    Admin-only login page.
    IGNORES student and professor sessions - admins can log in even if others are logged in.
    Hidden route - not linked from public pages, only accessible by direct URL.
    """
    # Check if admin is already logged in (IGNORE student/professor sessions)
    if current_user.is_authenticated and current_user.role == 'admin':
        return redirect(url_for('admin_panel'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Email and password are required', 'error')
            return render_template('auth/login.html', role='admin')

        # Find user by email
        user = User.query.filter_by(email=email).first()

        if not user or not user.check_password(password):
            flash('Invalid email or password', 'error')
            return render_template('auth/login.html', role='admin')

        # Verify user is an admin
        if user.role != 'admin':
            flash('Access denied. Admin credentials required.', 'error')
            return render_template('auth/login.html', role='admin')

        # Check if account is active
        if not user.is_active:
            flash('Your account is inactive. Please contact support.', 'error')
            return render_template('auth/login.html', role='admin')

        # Log in admin (Flask-Login)
        login_user(user, remember=True)

        flash('Welcome back, admin!', 'success')
        return redirect(url_for('admin_panel'))

    # Show admin login form
    return render_template('auth/login.html', role='admin')
