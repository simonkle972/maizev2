"""
Email validation utilities for .edu domain and institution domain matching.
"""

from models import Institution


def validate_edu_email(email, user_role=None):
    """
    Validate that email is from a .edu domain.

    Args:
        email: Email address to validate
        user_role: User role ('admin', 'professor', 'student')

    Returns:
        Tuple of (is_valid, error_message)
    """
    if user_role == 'admin':
        return True, None  # Admins exempt from .edu requirement

    if not email or '@' not in email:
        return False, "Invalid email address"

    domain = email.split('@')[-1].lower()
    if not domain.endswith('.edu'):
        return False, "Only .edu email addresses are allowed"

    return True, None


def validate_professor_email(email, institution):
    """
    Validate that professor email matches institution domain.

    Args:
        email: Professor's email address
        institution: Institution object

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not institution or not institution.email_domain:
        return True, None  # No institution domain restriction

    domain = email.split('@')[-1].lower()
    institution_domain = institution.email_domain.lower()

    if domain != institution_domain:
        return False, f"Email must be from {institution_domain}"

    return True, None


def validate_student_email(email, ta):
    """
    Validate that student email matches institution domain.

    Args:
        email: Student's email address
        ta: TeachingAssistant object

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Admin TAs don't enforce domain restrictions
    if not ta.requires_billing:
        return True, None

    institution = ta.institution
    if not institution or not institution.email_domain:
        return True, None  # No institution = no restriction

    domain = email.split('@')[-1].lower()
    institution_domain = institution.email_domain.lower()

    if domain != institution_domain:
        return False, f"Email must be from {institution_domain}"

    return True, None


def suggest_institution(email):
    """
    Suggest existing institution based on email domain.

    Args:
        email: Email address

    Returns:
        Institution object or None
    """
    if not email or '@' not in email:
        return None

    domain = email.split('@')[-1].lower()
    return Institution.query.filter_by(email_domain=domain).first()


def validate_password_strength(password):
    """
    Validate password meets minimum requirements.

    Requirements:
    - At least 8 characters
    - At least 1 uppercase letter
    - At least 1 lowercase letter
    - At least 1 number

    Args:
        password: Password to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters long"

    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"

    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"

    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"

    return True, None
