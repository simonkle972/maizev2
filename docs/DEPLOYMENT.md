# Deployment Guide

## Production Deployment

### Deploy Process
```bash
# On VPS (after git push)
cd /opt/maize
sudo -u maize git pull
sudo -u maize ./venv/bin/flask db upgrade  # If database changes
systemctl restart maize
systemctl status maize  # Verify

# Check application logs
journalctl -u maize -f
```

### Environment Configuration

The system uses **DOTENV_PATH** to switch between environments:
- **Local**: `export DOTENV_PATH=.env.local` (uses Docker PostgreSQL)
- **Production**: `.env` (default, uses Vultr managed PostgreSQL)

**CRITICAL**: Never commit `.env` or `.env.local` files. They are gitignored.

## Git Workflow

Feature branch model:
```bash
# 1. Create feature branch
git checkout -b feature/description

# 2. Develop and test locally
./run_local.sh

# 3. Commit and push
git add .
git commit -m "Clear description"
git push -u origin feature/description

# 4. Create PR on GitHub
gh pr create --title "Feature: description" --body "Details..."

# 5. After merge, deploy to production
# (on VPS)
cd /opt/maize
sudo -u maize git pull
sudo -u maize ./venv/bin/flask db upgrade
systemctl restart maize
```

**Main branch always represents production state.**

## Security Notes

- Never commit API keys or credentials (use .env files, gitignored)
- Production .env stays on VPS only, never in git history
- SMTP passwords use app-specific passwords, not main account password
- Session secrets auto-generated if not provided
- Admin credentials should be rotated regularly (currently in .env)

## Dependencies

### System Requirements

Critical system requirements:
- **poppler**: Required for pdf2image (PDF OCR fallback)
  - macOS: `brew install poppler`
  - Linux: `apt-get install poppler-utils`
- **Python 3.9+**: Tested with 3.9.6
- **Docker**: For local PostgreSQL development
- **PostgreSQL 16 with pgvector**: Production and local

See [requirements.txt](../requirements.txt) for Python packages.

## Email Configuration

Contact form at `/api/demo-request` requires SMTP configuration:
```bash
# .env or .env.local
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password  # Generate in Google Account settings
```
