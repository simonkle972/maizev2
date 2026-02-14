# Maize TA Development Guide

## Prerequisites

- Python 3.11
- Docker Desktop for Mac
- Git

## First Time Setup

1. **Install Docker Desktop:**
   ```bash
   # Download from https://www.docker.com/products/docker-desktop
   # Verify installation
   docker --version
   docker-compose --version
   ```

2. **Clone and set up repository:**
   ```bash
   cd ~/Desktop/Maize\ TA/Maize-Blueprint-V2

   # Create Python virtual environment
   python3.11 -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt

   # Install poppler for PDF processing
   brew install poppler
   ```

3. **Start Docker database:**
   ```bash
   docker-compose up -d postgres
   ```

4. **Configure local environment:**
   - `.env.local` is already created (with DATABASE_URL, OPENAI_API_KEY)
   - Run migrations: `export DOTENV_PATH=.env.local && flask db upgrade`
   - Seed admin user: `python seed_admin.py`

5. **Start app:**
   ```bash
   ./run_local.sh
   ```

6. **Access app:**
   - Open http://localhost:5000
   - Admin: http://localhost:5000/admin (login via seed_admin.py credentials)

## Daily Development Workflow

1. **Start environment:**
   ```bash
   ./run_local.sh
   ```

2. **Create feature branch:**
   ```bash
   git checkout main
   git pull
   git checkout -b feature/your-feature-name
   ```

3. **Make changes and test:**
   - Edit code
   - Test on http://localhost:5000
   - Check logs in terminal

4. **Database changes (if needed):**
   ```bash
   # After modifying models.py
   export DOTENV_PATH=.env.local
   flask db migrate -m "Description of schema change"
   flask db upgrade
   ```

5. **Commit and push:**
   ```bash
   git add .
   git commit -m "Clear description of changes"
   git push -u origin feature/your-feature-name
   ```

6. **Create Pull Request:**
   ```bash
   # Using GitHub CLI
   gh pr create --title "Feature: Description" --body "Details..."

   # Or visit GitHub UI
   ```

7. **Merge and deploy:**
   ```bash
   # After PR is merged on GitHub
   git checkout main
   git pull

   # Deploy to production (on VPS)
   cd /opt/maize
   sudo -u maize git pull
   sudo -u maize ./venv/bin/flask db upgrade
   sudo systemctl restart maize
   ```

## Testing Checklist Before Merging PR

- [ ] App starts without errors (`./run_local.sh`)
- [ ] Admin panel accessible
- [ ] Create institution/TA works
- [ ] Document upload works (PDF, Excel, Word)
- [ ] Indexing completes successfully
- [ ] Chat generates responses
- [ ] LaTeX renders correctly
- [ ] No JavaScript console errors
- [ ] Database migrations run successfully
- [ ] No secrets committed to git

## Useful Commands

### Docker
```bash
# Start database
docker-compose up -d postgres

# Stop database
docker-compose down

# Reset database (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d postgres
export DOTENV_PATH=.env.local
flask db upgrade

# View database logs
docker-compose logs -f postgres

# Connect to database
docker exec -it maize_postgres_dev psql -U maize_dev -d maize_ta_dev
```

### Database Migrations
```bash
export DOTENV_PATH=.env.local

# Create migration after model changes
flask db migrate -m "Add user authentication tables"

# Apply migrations
flask db upgrade

# Rollback last migration
flask db downgrade

# View migration history
flask db history
```

### Git
```bash
# View all branches
git branch -a

# Delete merged feature branch
git branch -d feature/old-feature

# View changes before merging
git diff main..feature/your-feature

# Sync main with remote
git checkout main
git pull origin main
```

## Troubleshooting

### Database Connection Issues
```bash
# Check if Docker container is running
docker-compose ps

# Restart Docker container
docker-compose restart postgres

# View container logs
docker-compose logs postgres
```

### Port 5000 Already in Use
```bash
# Find and kill process
lsof -ti:5000 | xargs kill -9
```

### Migrations Out of Sync
```bash
# On production: check current migration
flask db current

# On local: match production state
flask db upgrade

# If migrations conflict, resolve manually or recreate migrations/
```

### Reset Everything
```bash
# Stop Docker
docker-compose down -v

# Delete virtual environment
rm -rf venv

# Recreate from scratch
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
docker-compose up -d postgres
export DOTENV_PATH=.env.local
flask db upgrade
python seed_admin.py
./run_local.sh
```

## Feature Branch Naming Convention

```
feature/auth           - New authentication system
feature/billing        - Billing integration
fix/latex-rendering    - Bug fix for LaTeX
refactor/retriever     - Code refactoring
docs/api-endpoints     - Documentation updates
```

## Production Deployment

After merging PR to main:

```bash
# On VPS
cd /opt/maize
sudo -u maize git pull
sudo -u maize ./venv/bin/flask db upgrade  # Run migrations
sudo systemctl restart maize

# Verify
curl https://getmaize.ai/health
```

## Environment Variables

### Local (.env.local)
- `DATABASE_URL`: Docker PostgreSQL connection
- `OPENAI_API_KEY`: Shared with production (counts toward quota)
- `SESSION_SECRET`: Local session secret
- Admin credentials managed via seed_admin.py

### Production (.env on VPS)
- `DATABASE_URL`: Vultr managed PostgreSQL
- `OPENAI_API_KEY`: Production API key
- `SESSION_SECRET`: Production session secret
- `admin_id`, `admin_pw`, `ADMIN_SECRET_KEY`: Production admin credentials

**IMPORTANT:** Never commit `.env` files to git. They are excluded in `.gitignore`.
