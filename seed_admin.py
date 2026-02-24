#!/usr/bin/env python3
"""
Seed script to create local admin user.
Usage: export DOTENV_PATH=.env.local && python seed_admin.py
"""

import os
from dotenv import load_dotenv
from app import app
from models import db, User

# Load local environment
dotenv_path = os.getenv('DOTENV_PATH', '.env.local')
load_dotenv(dotenv_path)

def seed_admin():
    """Create local admin user if it doesn't exist."""
    with app.app_context():
        admin_email = "simon.kleffner98@gmail.com"

        # Check if admin already exists
        existing_admin = User.query.filter_by(email=admin_email, role='admin').first()

        if existing_admin:
            print(f"✅ Admin user already exists: {admin_email}")
            print(f"   Login at: http://localhost:5000/admin/login")
            return

        # Create admin user
        admin = User(
            email=admin_email,
            role='admin',
            first_name='Simon',
            last_name='Kleffner',
            is_active=True
        )
        admin.set_password('@KLEFFNER98')

        db.session.add(admin)
        db.session.commit()

        print("🔐 Local admin user created successfully!")
        print(f"   Email: {admin_email}")
        print("   Password: @KLEFFNER98")
        print(f"\n✅ Login at: http://localhost:5000/admin/login")

if __name__ == '__main__':
    seed_admin()
