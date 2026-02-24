#!/usr/bin/env python3
"""
Admin migration script - creates admin user in the database.
Usage: python migrate_admin.py
"""

import os
from dotenv import load_dotenv
from app import app
from models import db, User

# Load local environment
dotenv_path = os.getenv('DOTENV_PATH', '.env.local')
load_dotenv(dotenv_path)


def migrate_admin():
    """Create admin user if it doesn't exist."""
    with app.app_context():
        # Admin credentials (from seed_admin.py)
        admin_email = 'dev_admin@maize.local'
        admin_password = 'dev_password_123'

        # Check if admin already exists
        existing_admin = User.query.filter_by(email=admin_email).first()

        if existing_admin:
            print(f"✅ Admin user already exists: {existing_admin.email}")
            print(f"   Role: {existing_admin.role}")
            print(f"   ID: {existing_admin.id}")
            return

        # Create admin user
        admin = User(
            email=admin_email,
            first_name='Admin',
            last_name='User',
            role='admin',
            institution_id=None,  # Admins don't need institution
            is_active=True
        )
        admin.set_password(admin_password)

        db.session.add(admin)
        db.session.commit()

        print("🔐 Admin user created successfully!")
        print(f"   Email: {admin_email}")
        print(f"   Password: {admin_password}")
        print(f"   Role: admin")
        print(f"   ID: {admin.id}")
        print()
        print("⚠️  IMPORTANT: Change the admin password after first login!")
        print("   You can log in at: http://localhost:5000/login")


if __name__ == '__main__':
    migrate_admin()
