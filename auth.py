"""
Authentication blueprint for Maize TA.
Handles login, signup, password reset for professors and students.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, current_user
from datetime import datetime, timedelta
import secrets

from models import db, User, Institution, Enrollment, EnrollmentLink, PasswordResetToken, TeachingAssistant
from auth_student import current_student, login_student, logout_student
from utils.validators import (
    validate_edu_email,
    validate_professor_email,
    validate_student_email,
    suggest_institution,
    validate_password_strength
)
from utils.email import send_password_reset_email, send_welcome_email

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """
    Deprecated unified login route.
    Redirects to role-specific login pages for backward compatibility.
    """
    # If someone tries to POST (legacy form submission), detect role and redirect
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()

        if email:
            user = User.query.filter_by(email=email).first()

            if user:
                # Redirect to appropriate role-specific login page
                if user.role == 'student':
                    flash('Please use the student login page.', 'info')
                    return redirect(url_for('auth.student_login'))
                elif user.role == 'professor':
                    flash('Please use the professor login page.', 'info')
                    return redirect(url_for('auth.professor_login'))
                elif user.role == 'admin':
                    flash('Please use the admin login page.', 'info')
                    return redirect(url_for('auth.admin_login'))

        flash('Invalid email or account not found.', 'error')

    # For GET requests, just redirect to landing page
    flash('Please use the role-specific login page.', 'info')
    return redirect(url_for('landing'))


@auth_bp.route('/professor/signup', methods=['GET', 'POST'])
def professor_signup():
    """Professor registration page."""
    if current_user.is_authenticated:
        if current_user.role == 'professor':
            return redirect(url_for('professor.dashboard'))
        elif current_user.role == 'admin':
            # Allow admin to view the signup form for testing
            pass
        elif current_user.role == 'student':
            return redirect(url_for('student.dashboard'))

    suggested_institution = None

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        institution_id = request.form.get('institution_id')
        new_institution_name = request.form.get('new_institution_name', '').strip()

        # Validation
        if not all([email, password, confirm_password, first_name, last_name]):
            flash('All fields are required', 'error')
            return render_template('auth/professor_signup.html')

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('auth/professor_signup.html')

        # Validate .edu email
        is_valid, error = validate_edu_email(email, user_role='professor')
        if not is_valid:
            flash(error, 'error')
            return render_template('auth/professor_signup.html')

        # Validate password strength
        is_valid, error = validate_password_strength(password)
        if not is_valid:
            flash(error, 'error')
            return render_template('auth/professor_signup.html')

        # Check if email already exists
        if User.query.filter_by(email=email).first():
            flash('An account with this email already exists', 'error')
            return render_template('auth/professor_signup.html')

        # Handle institution
        if institution_id and institution_id != 'new':
            institution = Institution.query.get(int(institution_id))
        elif new_institution_name:
            # Create new institution
            domain = email.split('@')[-1].lower()
            institution = Institution(
                name=new_institution_name,
                email_domain=domain
            )
            db.session.add(institution)
            db.session.flush()
        else:
            flash('Please select or create an institution', 'error')
            return render_template('auth/professor_signup.html')

        # Validate professor email matches institution domain
        is_valid, error = validate_professor_email(email, institution)
        if not is_valid:
            flash(error, 'error')
            return render_template('auth/professor_signup.html')

        # Create professor user
        user = User(
            email=email,
            first_name=first_name,
            last_name=last_name,
            role='professor',
            institution_id=institution.id
        )
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        # Send welcome email (optional, don't block on failure)
        try:
            send_welcome_email(email, f"{first_name} {last_name}", 'professor')
        except:
            pass

        # Auto-login the new professor
        login_user(user, remember=True)
        flash('Account created successfully! Welcome to Maize TA.', 'success')
        return redirect(url_for('professor.dashboard'))

    # GET request - check if we can suggest an institution
    email_preview = request.args.get('email', '')
    if email_preview and '@' in email_preview:
        suggested_institution = suggest_institution(email_preview)

    institutions = Institution.query.order_by(Institution.name).all()
    return render_template('auth/professor_signup.html',
                         institutions=institutions,
                         suggested_institution=suggested_institution)


@auth_bp.route('/student/signup/<token>', methods=['GET', 'POST'])
def student_signup(token):
    """Student signup via enrollment link OR auto-enroll existing student."""
    link = EnrollmentLink.query.filter_by(token=token).first_or_404()

    # Check if link is valid
    if not link.is_valid:
        if link.is_full:
            flash('This course has reached maximum capacity.', 'error')
        elif not link.is_active:
            flash('This enrollment link is no longer active.', 'error')
        elif link.expires_at and datetime.utcnow() > link.expires_at:
            flash('This enrollment link has expired.', 'error')
        return redirect(url_for('landing'))

    ta = link.ta

    # Store enrollment token in session for login flow (if user clicks "Already have account?")
    session['pending_enrollment_token'] = token

    # If user is already logged in as a student, auto-enroll them
    # Note: Professor sessions are separate, so this only checks student session
    if current_student:
        # Check if already enrolled
        existing_enrollment = Enrollment.query.filter_by(
            student_id=current_student.id,
            ta_id=ta.id
        ).first()

        if existing_enrollment:
            flash("You're already enrolled in this course.", 'info')
            return redirect(url_for('student.dashboard'))

        # Atomic capacity check and enrollment
        with db.session.begin_nested():
            # Re-fetch link with lock
            link = EnrollmentLink.query.with_for_update().get(link.id)
            if link.is_full:
                flash("Course capacity reached.", 'error')
                return redirect(url_for('student.dashboard'))

            # Create enrollment
            enrollment = Enrollment(
                student_id=current_student.id,
                ta_id=ta.id,
                enrollment_token=token
            )
            db.session.add(enrollment)
            link.current_enrollments += 1

        db.session.commit()
        flash("Successfully enrolled! Course added to your dashboard.", 'success')
        return redirect(url_for('student.dashboard'))

    # New student signup
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()

        # Validation
        if not all([email, password, confirm_password, first_name, last_name]):
            flash('All fields are required', 'error')
            return render_template('auth/student_signup.html', link=link, ta=ta)

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('auth/student_signup.html', link=link, ta=ta)

        # Validate .edu email
        is_valid, error = validate_edu_email(email, user_role='student')
        if not is_valid:
            flash(error, 'error')
            return render_template('auth/student_signup.html', link=link, ta=ta)

        # Validate student email matches institution domain
        is_valid, error = validate_student_email(email, ta)
        if not is_valid:
            flash(error, 'error')
            return render_template('auth/student_signup.html', link=link, ta=ta)

        # Validate password strength
        is_valid, error = validate_password_strength(password)
        if not is_valid:
            flash(error, 'error')
            return render_template('auth/student_signup.html', link=link, ta=ta)

        # Check if email already exists
        if User.query.filter_by(email=email).first():
            flash('An account with this email already exists. Please log in instead.', 'error')
            # Store enrollment link token in session to complete enrollment after login
            session['pending_enrollment_token'] = token
            return redirect(url_for('auth.student_login'))

        # Atomic capacity check and user creation
        with db.session.begin_nested():
            # Re-fetch link with lock
            link = EnrollmentLink.query.with_for_update().get(link.id)
            if link.is_full:
                flash("Course capacity reached.", 'error')
                return redirect(url_for('landing'))

            # Create student user
            user = User(
                email=email,
                first_name=first_name,
                last_name=last_name,
                role='student',
                institution_id=ta.institution_id
            )
            user.set_password(password)
            db.session.add(user)
            db.session.flush()

            # Create enrollment
            enrollment = Enrollment(
                student_id=user.id,
                ta_id=ta.id,
                enrollment_token=token
            )
            db.session.add(enrollment)
            link.current_enrollments += 1

        db.session.commit()

        # Send welcome email (optional)
        try:
            send_welcome_email(email, f"{first_name} {last_name}", 'student')
        except:
            pass

        # Log in the newly created student (parallel session - does not affect professor sessions)
        login_student(user, remember=True)

        flash('Account created successfully! Welcome to the course.', 'success')
        return redirect(url_for('student.dashboard'))

    return render_template('auth/student_signup.html', link=link, ta=ta)


@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Request password reset."""
    if current_user.is_authenticated:
        flash('You are already logged in.', 'info')
        return redirect(url_for('landing'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()

        user = User.query.filter_by(email=email).first()

        if user:
            # Generate reset token
            token = secrets.token_urlsafe(32)
            reset_token = PasswordResetToken(
                user_id=user.id,
                token=token,
                expires_at=datetime.utcnow() + timedelta(hours=24)
            )
            db.session.add(reset_token)
            db.session.commit()

            # Send reset email
            base_url = request.url_root.rstrip('/')
            success, error = send_password_reset_email(email, token, base_url)

            if not success:
                flash('Failed to send reset email. Please try again later.', 'error')
                return render_template('auth/forgot_password.html')

        # Always show success message (don't reveal if email exists)
        flash('If an account exists with that email, you will receive password reset instructions.', 'success')
        return redirect(url_for('landing'))

    return render_template('auth/forgot_password.html')


@auth_bp.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Reset password with token."""
    if current_user.is_authenticated:
        flash('You are already logged in.', 'info')
        return redirect(url_for('landing'))

    reset_token = PasswordResetToken.query.filter_by(token=token).first_or_404()

    if not reset_token.is_valid:
        flash('This password reset link is invalid or has expired.', 'error')
        return redirect(url_for('auth.forgot_password'))

    if request.method == 'POST':
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not password or not confirm_password:
            flash('Both fields are required', 'error')
            return render_template('auth/reset_password.html', token=token)

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('auth/reset_password.html', token=token)

        # Validate password strength
        is_valid, error = validate_password_strength(password)
        if not is_valid:
            flash(error, 'error')
            return render_template('auth/reset_password.html', token=token)

        # Update password
        user = reset_token.user
        user.set_password(password)
        reset_token.is_used = True
        db.session.commit()

        flash('Password reset successfully! Please log in with your new password.', 'success')

        # Redirect to role-specific login page
        if user.role == 'student':
            return redirect(url_for('auth.student_login'))
        elif user.role == 'professor':
            return redirect(url_for('auth.professor_login'))
        elif user.role == 'admin':
            return redirect(url_for('auth.admin_login'))
        else:
            return redirect(url_for('landing'))  # Unknown role - redirect to home

    return render_template('auth/reset_password.html', token=token)


@auth_bp.route('/logout')
def logout():
    """
    Log out current user (professor or admin).
    For student logout, use /student/logout instead.
    """
    logout_user()
    flash('You have been logged out.', 'success')

    # Check for next URL parameter
    next_url = request.args.get('next')
    if next_url:
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


@auth_bp.route('/student/login', methods=['GET', 'POST'])
def student_login():
    """
    Student-only login page.
    IGNORES professor sessions - students can log in even if professor is logged in.
    """
    # Check if student is already logged in (IGNORE professor session)
    if current_student:
        return redirect(url_for('student.dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Email and password are required', 'error')
            return render_template('auth/login.html', role='student')

        # Find user by email
        user = User.query.filter_by(email=email).first()

        if not user or not user.check_password(password):
            flash('Invalid email or password', 'error')
            return render_template('auth/login.html', role='student')

        # Verify user is a student
        if user.role != 'student':
            flash('This login is for students only. Please use the professor login page.', 'error')
            return render_template('auth/login.html', role='student')

        # Check if account is active
        if not user.is_active:
            flash('Your account is inactive. Please contact support.', 'error')
            return render_template('auth/login.html', role='student')

        # Log in student (parallel session - does not affect professor sessions)
        login_student(user, remember=True)

        # Check if there's a pending enrollment from signup link
        pending_token = session.pop('pending_enrollment_token', None)
        if pending_token:
            # Process pending enrollment
            link = EnrollmentLink.query.filter_by(token=pending_token).first()
            if link and link.is_valid:
                # Check if already enrolled
                existing_enrollment = Enrollment.query.filter_by(
                    student_id=user.id,
                    ta_id=link.ta_id
                ).first()

                if not existing_enrollment:
                    # Atomic capacity check and enrollment
                    try:
                        with db.session.begin_nested():
                            # Re-fetch link with lock
                            link = EnrollmentLink.query.with_for_update().get(link.id)
                            if link and not link.is_full:
                                # Create enrollment
                                enrollment = Enrollment(
                                    student_id=user.id,
                                    ta_id=link.ta_id,
                                    enrollment_token=pending_token
                                )
                                db.session.add(enrollment)
                                link.current_enrollments += 1

                        db.session.commit()
                        flash('Welcome back! Successfully enrolled in the course.', 'success')
                        return redirect(url_for('student.dashboard'))
                    except:
                        db.session.rollback()

        flash('Welcome back!', 'success')
        return redirect(url_for('student.dashboard'))

    # Show student login form
    return render_template('auth/login.html', role='student')


@auth_bp.route('/professor/login', methods=['GET', 'POST'])
def professor_login():
    """
    Professor-only login page.
    IGNORES student sessions - professors can log in even if student is logged in.
    """
    # Check if professor is already logged in (IGNORE student session)
    if current_user.is_authenticated and current_user.role == 'professor':
        return redirect(url_for('professor.dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Email and password are required', 'error')
            return render_template('auth/login.html', role='professor')

        # Find user by email
        user = User.query.filter_by(email=email).first()

        if not user or not user.check_password(password):
            flash('Invalid email or password', 'error')
            return render_template('auth/login.html', role='professor')

        # Verify user is a professor
        if user.role != 'professor':
            flash('This login is for professors only. Please use the student login page.', 'error')
            return render_template('auth/login.html', role='professor')

        # Check if account is active
        if not user.is_active:
            flash('Your account is inactive. Please contact support.', 'error')
            return render_template('auth/login.html', role='professor')

        # Log in professor (Flask-Login)
        login_user(user, remember=True)

        flash('Welcome back!', 'success')
        return redirect(url_for('professor.dashboard'))

    # Show professor login form
    return render_template('auth/login.html', role='professor')


@auth_bp.route('/admin/login', methods=['GET', 'POST'])
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
