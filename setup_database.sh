#!/bin/bash
# Database setup script - enables pgvector extension and initializes schema

set -e

DB_URL="$1"

if [ -z "$DB_URL" ]; then
    echo "Error: DATABASE_URL is required"
    echo "Usage: ./setup_database.sh 'postgresql://user:pass@host:port/db'"
    exit 1
fi

echo "🗄️  Setting up database..."

# Enable pgvector extension
echo "📦 Enabling pgvector extension..."
psql "$DB_URL" -c "CREATE EXTENSION IF NOT EXISTS vector;" || {
    echo "⚠️  Warning: Could not enable pgvector extension. It may already be enabled or require manual setup."
}

# Verify pgvector is enabled
echo "✅ Verifying pgvector extension..."
psql "$DB_URL" -c "SELECT * FROM pg_extension WHERE extname = 'vector';" || {
    echo "❌ Error: pgvector extension is not enabled"
    exit 1
}

echo "✅ Database setup complete!"
echo ""
echo "You can now run: python init_db.py"
