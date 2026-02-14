#!/usr/bin/env python3
"""
Seed script to create local admin user.
Usage: python seed_admin.py
"""

import os
from dotenv import load_dotenv
from app import app

# Load local environment
dotenv_path = os.getenv('DOTENV_PATH', '.env.local')
load_dotenv(dotenv_path)

def seed_admin():
    """Create local admin user if it doesn't exist."""
    with app.app_context():
        # For now, just set config values
        # When you add User model with auth, this will create actual users

        print("🔐 Setting up local admin credentials...")

        # Temporarily set admin config (until User model exists)
        os.environ['admin_id'] = 'dev_admin'
        os.environ['admin_pw'] = 'dev_password_123'
        os.environ['ADMIN_SECRET_KEY'] = 'dev-secret-key-2024'

        print("✅ Admin credentials configured:")
        print("   Username: dev_admin")
        print("   Password: dev_password_123")
        print("")
        print("⚠️  Note: These credentials are temporary until User model is added.")
        print("   When auth is implemented, this script will create actual database users.")

if __name__ == '__main__':
    seed_admin()
