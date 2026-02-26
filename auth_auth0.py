from flask import Blueprint, redirect, url_for, session, request, flash, render_template
from flask_login import login_user, logout_user
from auth0_integration import oauth
from auth_student import login_student, logout_student
from models import db, User, EnrollmentLink, Enrollment

auth0_bp = Blueprint('auth0', __name__, url_prefix='/auth0')


@auth0_bp.route('/login')
def login():
    """Initiate Auth0 login."""
    role = request.args.get('role', 'student')
    session['auth0_role'] = role
    redirect_uri = url_for('auth0.callback', _external=True)
    return oauth.auth0.authorize_redirect(redirect_uri=redirect_uri, prompt='login')


@auth0_bp.route('/callback')
def callback():
    """Handle Auth0 callback."""
    token = oauth.auth0.authorize_access_token()
    userinfo = token.get('userinfo', {})

    auth0_sub = userinfo.get('sub')
    email = userinfo.get('email')

    if not auth0_sub or not email:
        flash('Authentication failed: missing user information.', 'error')
        return redirect(url_for('landing'))

    # Find or create User
    user = User.query.filter_by(auth0_sub=auth0_sub).first()

    if not user:
        # Check if email exists (link accounts)
        user = User.query.filter_by(email=email).first()
        if user:
            user.auth0_sub = auth0_sub
        else:
            # Create new user
            role = session.pop('auth0_role', 'student')
            first_name = userinfo.get('given_name', email.split('@')[0])
            last_name = userinfo.get('family_name', '')

            user = User(
                email=email,
                auth0_sub=auth0_sub,
                role=role,
                first_name=first_name,
                last_name=last_name,
                password_hash='auth0'
            )
            db.session.add(user)
        db.session.commit()
    else:
        session.pop('auth0_role', None)

    # Always sync email_verified status from Auth0
    user.email_verified = userinfo.get('email_verified', False)
    db.session.commit()

    # Store in session — satisfy both Flask-Login (professors) and student session system
    access_token = token.get('access_token')  # capture before session.clear()
    logout_user()
    logout_student()
    session.clear()
    session['auth0_user_id'] = user.id
    session['auth0_access_token'] = access_token
    login_user(user)
    if user.role == 'student':
        login_student(user)

    # Redirect based on role
    if user.role == 'professor':
        return redirect(url_for('professor.dashboard'))
    elif user.role == 'student':
        if user.enrollments:
            return redirect(url_for('student.dashboard'))
        else:
            return redirect(url_for('auth0.enroll_prompt'))
    else:
        return redirect(url_for('admin_panel'))


@auth0_bp.route('/enroll', methods=['GET'])
def enroll_prompt():
    """Prompt student to enter enrollment token."""
    if 'auth0_user_id' not in session:
        return redirect(url_for('auth0.login', role='student'))
    return render_template('auth/enroll.html')


@auth0_bp.route('/enroll', methods=['POST'])
def enroll_submit():
    """Complete enrollment with token."""
    user_id = session.get('auth0_user_id')
    if not user_id:
        return redirect(url_for('auth0.login', role='student'))

    user = User.query.get(user_id)
    token = request.form.get('token', '').strip()

    if not token:
        flash('Please enter an enrollment token.', 'error')
        return redirect(url_for('auth0.enroll_prompt'))

    link = EnrollmentLink.query.filter_by(token=token).first()

    if not link or not link.is_valid:
        flash('Invalid or expired enrollment token.', 'error')
        return redirect(url_for('auth0.enroll_prompt'))

    # Check if already enrolled
    existing = Enrollment.query.filter_by(
        student_id=user.id,
        ta_id=link.ta_id
    ).first()

    if existing:
        flash('You are already enrolled in this course.', 'info')
        return redirect(url_for('student.dashboard'))

    # Atomic enrollment with capacity check
    with db.session.begin_nested():
        link = EnrollmentLink.query.with_for_update().get(link.id)
        if link.is_full:
            flash('Course capacity has been reached.', 'error')
            return redirect(url_for('auth0.enroll_prompt'))

        enrollment = Enrollment(
            student_id=user.id,
            ta_id=link.ta_id,
            enrollment_token=token
        )
        db.session.add(enrollment)
        link.current_enrollments += 1

    db.session.commit()
    flash('Successfully enrolled!', 'success')
    return redirect(url_for('student.dashboard'))


@auth0_bp.route('/refresh-verification')
def refresh_verification():
    """Re-sync email_verified from Auth0 without a full re-login."""
    import requests as req
    from config import Config
    from flask_login import current_user

    user_id = session.get('auth0_user_id')
    access_token = session.get('auth0_access_token')

    role = current_user.role if current_user.is_authenticated else 'student'
    dashboard = url_for('professor.dashboard') if role == 'professor' else url_for('student.dashboard')

    if not user_id or not access_token:
        return redirect(url_for('auth0.login', role=role))

    try:
        resp = req.get(
            f'https://{Config.AUTH0_DOMAIN}/userinfo',
            headers={'Authorization': f'Bearer {access_token}'},
            timeout=5
        )
        if resp.status_code == 401:
            flash('Session expired. Please log in again to update your verification status.', 'warning')
            return redirect(url_for('auth0.login', role=role))

        if resp.ok:
            userinfo = resp.json()
            user = User.query.get(user_id)
            if user:
                user.email_verified = userinfo.get('email_verified', False)
                db.session.commit()
                if not user.email_verified:
                    flash('Your email is still not verified. Check your inbox for the verification link and try again.', 'warning')
    except Exception:
        flash('Could not reach the verification service. Please log out and back in.', 'warning')

    return redirect(dashboard)


@auth0_bp.route('/logout')
def logout():
    """Log out from Auth0 and clear session."""
    from flask_login import current_user
    from urllib.parse import quote
    from config import Config
    role = current_user.role if current_user.is_authenticated else 'professor'
    logout_user()
    logout_student()
    session.clear()
    domain = Config.AUTH0_DOMAIN
    client_id = Config.AUTH0_CLIENT_ID
    return_to = url_for('auth0.login', role=role, _external=True)
    return redirect(f'https://{domain}/v2/logout?returnTo={quote(return_to, safe="")}&client_id={client_id}')
