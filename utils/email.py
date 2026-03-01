"""
Email utilities for sending password reset and notification emails.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import Config


def send_email(to_email, subject, html_body, text_body=None):
    """
    Send an email using SMTP configuration.

    Args:
        to_email: Recipient email address
        subject: Email subject
        html_body: HTML content of the email
        text_body: Plain text alternative (optional)

    Returns:
        Tuple of (success: bool, error_message: str or None)
    """
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = Config.SMTP_USER
        msg['To'] = to_email

        # Add text and HTML parts
        if text_body:
            part1 = MIMEText(text_body, 'plain')
            msg.attach(part1)

        part2 = MIMEText(html_body, 'html')
        msg.attach(part2)

        # Send via SMTP
        with smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT) as server:
            server.starttls()
            server.login(Config.SMTP_USER, Config.SMTP_PASS)
            server.send_message(msg)

        return True, None

    except Exception as e:
        return False, str(e)


def send_password_reset_email(user_email, reset_token, base_url):
    """
    Send password reset email to user.

    Args:
        user_email: User's email address
        reset_token: Password reset token
        base_url: Base URL of the application (e.g., http://localhost:5000)

    Returns:
        Tuple of (success: bool, error_message: str or None)
    """
    reset_link = f"{base_url}/reset-password/{reset_token}"

    subject = "Reset Your Maize TA Password"

    html_body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <h2 style="color: #4CAF50;">Password Reset Request</h2>
            <p>You requested to reset your password for Maize TA.</p>
            <p>Click the link below to reset your password:</p>
            <p>
                <a href="{reset_link}"
                   style="display: inline-block; padding: 10px 20px; background-color: #4CAF50;
                          color: white; text-decoration: none; border-radius: 5px;">
                    Reset Password
                </a>
            </p>
            <p>Or copy and paste this link into your browser:</p>
            <p style="color: #666; font-size: 12px;">{reset_link}</p>
            <p><strong>This link will expire in 24 hours.</strong></p>
            <p>If you didn't request a password reset, you can safely ignore this email.</p>
            <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
            <p style="font-size: 12px; color: #999;">
                Maize TA - AI Teaching Assistant Platform
            </p>
        </body>
    </html>
    """

    text_body = f"""
Password Reset Request

You requested to reset your password for Maize TA.

Click the link below to reset your password:
{reset_link}

This link will expire in 24 hours.

If you didn't request a password reset, you can safely ignore this email.

---
Maize TA - AI Teaching Assistant Platform
    """

    return send_email(user_email, subject, html_body, text_body)


def send_contact_sales_email(professor_name, professor_email, institution_name, message, student_count=None):
    """
    Send contact sales request to admin.

    Args:
        professor_name: Professor's name
        professor_email: Professor's email address
        institution_name: Institution name
        message: Custom message from professor
        student_count: Requested student count (optional)

    Returns:
        Tuple of (success: bool, error_message: str or None)
    """
    subject = f"Custom Tier Request from {professor_name} ({institution_name})"

    html_body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <h2 style="color: #4CAF50;">New Custom Tier Request</h2>
            <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold; background-color: #f9f9f9;">
                        Professor Name
                    </td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{professor_name}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold; background-color: #f9f9f9;">
                        Email
                    </td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{professor_email}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold; background-color: #f9f9f9;">
                        Institution
                    </td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{institution_name}</td>
                </tr>
                {f'''<tr>
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold; background-color: #f9f9f9;">
                        Requested Student Count
                    </td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{student_count}</td>
                </tr>''' if student_count else ''}
            </table>
            <h3>Message:</h3>
            <p style="padding: 15px; background-color: #f9f9f9; border-left: 4px solid #4CAF50;">
                {message}
            </p>
            <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
            <p style="font-size: 12px; color: #999;">
                This email was sent from Maize TA Contact Sales form.
            </p>
        </body>
    </html>
    """

    # Send to admin email (use SMTP_USER as admin contact)
    return send_email(Config.SMTP_USER, subject, html_body)


def send_custom_tier_notification(professor_name, professor_email, institution_name, ta_name, course_name):
    """
    Send custom tier notification to sales team when professor requests custom tier.

    Args:
        professor_name: Professor's name
        professor_email: Professor's email address
        institution_name: Institution name
        ta_name: TA name being created
        course_name: Course name

    Returns:
        Tuple of (success: bool, error_message: str or None)
    """
    subject = f"Custom Tier Request - {ta_name} ({institution_name})"

    html_body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <h2 style="color: #f59e0b;">Custom Tier Request (250+ Students)</h2>
            <p>A professor has requested a custom tier for their TA during the creation process.</p>

            <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold; background-color: #f9f9f9;">
                        Professor Name
                    </td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{professor_name}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold; background-color: #f9f9f9;">
                        Email
                    </td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{professor_email}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold; background-color: #f9f9f9;">
                        Institution
                    </td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{institution_name}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold; background-color: #f9f9f9;">
                        TA Name
                    </td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{ta_name}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold; background-color: #f9f9f9;">
                        Course Name
                    </td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{course_name}</td>
                </tr>
            </table>

            <div style="background-color: #fef3c7; padding: 15px; border-left: 4px solid #f59e0b; margin: 20px 0;">
                <p style="margin: 0;"><strong>Action Required:</strong></p>
                <p style="margin: 5px 0 0 0;">
                    The TA has been created with the Large tier ($29.99/month, 250-student cap) as a temporary measure.
                    Please contact the professor within 24 hours to discuss custom pricing for their larger class.
                </p>
            </div>

            <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
            <p style="font-size: 12px; color: #999;">
                This notification was automatically generated by Maize TA when a professor selected the "Custom (250+ students)" tier.
            </p>
        </body>
    </html>
    """

    text_body = f"""
Custom Tier Request - {ta_name}

Professor: {professor_name}
Email: {professor_email}
Institution: {institution_name}
TA Name: {ta_name}
Course: {course_name}

The TA has been created with the Large tier ($29.99/month, 250-student cap) as a temporary measure.
Please contact the professor within 24 hours to discuss custom pricing for their larger class.

---
Maize TA - Automated Notification
    """

    # Send to simon@getmaize.ai
    return send_email("simon@getmaize.ai", subject, html_body, text_body)


def send_welcome_email(user_email, user_name, user_role):
    """
    Send welcome email to new user.

    Args:
        user_email: User's email address
        user_name: User's full name
        user_role: User's role ('professor' or 'student')

    Returns:
        Tuple of (success: bool, error_message: str or None)
    """
    subject = "Welcome to Maize TA!"

    role_specific_content = ""
    if user_role == "professor":
        role_specific_content = """
            <p>As a professor, you can now:</p>
            <ul>
                <li>Create AI teaching assistants for your courses</li>
                <li>Upload course materials and assignments</li>
                <li>Generate enrollment links for your students</li>
                <li>Monitor student engagement and TA usage</li>
            </ul>
            <p>Get started by creating your first teaching assistant in the dashboard.</p>
        """
    else:  # student
        role_specific_content = """
            <p>As a student, you now have access to:</p>
            <ul>
                <li>24/7 AI teaching assistance for your courses</li>
                <li>Interactive help with assignments and concepts</li>
                <li>Personalized explanations and guidance</li>
            </ul>
            <p>Visit your dashboard to see your enrolled courses and start chatting with your TAs.</p>
        """

    html_body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <h2 style="color: #4CAF50;">Welcome to Maize TA, {user_name}!</h2>
            <p>Your account has been successfully created.</p>
            {role_specific_content}
            <p>
                <a href="http://localhost:5000/login"
                   style="display: inline-block; padding: 10px 20px; background-color: #4CAF50;
                          color: white; text-decoration: none; border-radius: 5px; margin: 10px 0;">
                    Go to Dashboard
                </a>
            </p>
            <p>If you have any questions or need help, don't hesitate to contact us.</p>
            <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
            <p style="font-size: 12px; color: #999;">
                Maize TA - AI Teaching Assistant Platform
            </p>
        </body>
    </html>
    """

    return send_email(user_email, subject, html_body)
