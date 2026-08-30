#!/bin/bash
# Deployment script for Maize Teaching Assistant on Vultr VPS
# Run this script on your VPS after initial setup

set -e

echo "🚀 Starting Maize deployment..."

# Variables
DOMAIN="getmaize.ai"
APP_DIR="/opt/maize"

# ---- Required configuration -------------------------------------------------
# Secrets are NEVER hardcoded in this file: it is committed to a public
# repository. Supply them in the environment and run with `sudo -E`, which
# preserves them across the sudo boundary:
#
#   export OPENAI_API_KEY='sk-...'
#   export DATABASE_URL='postgresql://user:password@host:port/dbname'
#   export ADMIN_ID='your-admin-username'
#   export ADMIN_PW='your-admin-password'
#   sudo -E ./deploy.sh
#
# SESSION_SECRET and ADMIN_SECRET_KEY are generated here if not supplied.
REQUIRED_VARS=(OPENAI_API_KEY DATABASE_URL ADMIN_ID ADMIN_PW)
missing=()
for v in "${REQUIRED_VARS[@]}"; do
    [ -n "${!v:-}" ] || missing+=("$v")
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "ERROR: missing required environment variables: ${missing[*]}" >&2
    echo "See the comment block at the top of this script." >&2
    exit 1
fi

DB_URL="$DATABASE_URL"
SESSION_SECRET="${SESSION_SECRET:-$(openssl rand -hex 32)}"
ADMIN_SECRET_KEY="${ADMIN_SECRET_KEY:-$(openssl rand -hex 32)}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}📦 Installing system dependencies...${NC}"
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv nginx postgresql-client git

echo -e "${YELLOW}👤 Creating maize user...${NC}"
sudo useradd -m -s /bin/bash maize || echo "User already exists"

echo -e "${YELLOW}📁 Setting up application directory...${NC}"
sudo mkdir -p $APP_DIR
sudo mkdir -p /var/log/maize
sudo mkdir -p /var/run/maize
sudo chown -R maize:maize $APP_DIR /var/log/maize /var/run/maize

echo -e "${YELLOW}📥 Cloning repository...${NC}"
cd $APP_DIR
sudo -u maize git clone https://github.com/simonkle972/maizev2.git .

echo -e "${YELLOW}🐍 Setting up Python virtual environment...${NC}"
sudo -u maize python3 -m venv venv
sudo -u maize ./venv/bin/pip install --upgrade pip
sudo -u maize ./venv/bin/pip install -r requirements.txt

echo -e "${YELLOW}⚙️  Configuring environment variables...${NC}"
# Never regenerate an existing .env. On a live host it holds configuration this
# script does not know about (Auth0, Stripe, SMTP, and any hand-added keys), and
# overwriting it would silently revert production to this script's assumptions.
if [ -f "$APP_DIR/.env" ]; then
    echo -e "${YELLOW}   $APP_DIR/.env already exists — leaving it untouched.${NC}"
    echo -e "${YELLOW}   Move it aside first if you genuinely want to regenerate it.${NC}"
else
    umask 027
    cat > "$APP_DIR/.env" << EOF
OPENAI_API_KEY=$OPENAI_API_KEY
DATABASE_URL=$DATABASE_URL
SESSION_SECRET=$SESSION_SECRET
admin_id=$ADMIN_ID
admin_pw=$ADMIN_PW
ADMIN_SECRET_KEY=$ADMIN_SECRET_KEY
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASS=
EOF
    # The app loads this file directly at import time as the `maize` user, so it
    # must stay group-readable. 600 locks the app out of its own config.
    chown root:maize "$APP_DIR/.env"
    chmod 640 "$APP_DIR/.env"
fi

echo -e "${YELLOW}🗄️  Setting up database (enabling pgvector)...${NC}"
chmod +x setup_database.sh
./setup_database.sh "$DB_URL"

echo -e "${YELLOW}🗄️  Initializing database schema...${NC}"
sudo -u maize ./venv/bin/python init_db.py

echo -e "${YELLOW}⚙️  Configuring systemd service...${NC}"
sudo cp maize.service.template /etc/systemd/system/maize.service
sudo systemctl daemon-reload
sudo systemctl enable maize
sudo systemctl start maize

echo -e "${YELLOW}🌐 Configuring Nginx...${NC}"
sudo sed "s/YOUR_DOMAIN/$DOMAIN/g" nginx.conf.template > /tmp/maize.conf
sudo mv /tmp/maize.conf /etc/nginx/sites-available/maize
sudo ln -sf /etc/nginx/sites-available/maize /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

echo -e "${YELLOW}🔒 Setting up SSL certificate...${NC}"
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo -e "${GREEN}Your app should be running at https://$DOMAIN${NC}"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status maize    # Check app status"
echo "  sudo systemctl restart maize   # Restart app"
echo "  sudo journalctl -u maize -f    # View logs"
echo "  sudo tail -f /var/log/maize/error.log  # View error logs"
