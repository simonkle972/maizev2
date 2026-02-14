#!/bin/bash
# Local development runner for Maize TA

set -e

echo "🚀 Starting Maize TA in local development mode..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Start PostgreSQL container if not running
echo "🐳 Starting PostgreSQL container..."
docker-compose up -d postgres

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
until docker exec maize_postgres_dev pg_isready -U maize_dev -d maize_ta_dev > /dev/null 2>&1; do
    sleep 1
done
echo "✅ Database is ready!"

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3.11 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Set environment file
export DOTENV_PATH=.env.local

# Run database migrations
echo "🔄 Running database migrations..."
flask db upgrade

# Start Flask development server
echo ""
echo "🌐 Starting Flask dev server on http://localhost:5000"
echo "   To create admin user, run: python seed_admin.py"
echo "   Press Ctrl+C to stop"
echo ""
python app.py
