#!/bin/bash
# Deployment script for Maize Teaching Assistant on Vultr VPS
# Run this script on your VPS after initial setup

set -e

echo "🚀 Starting Maize deployment..."

# Variables
DOMAIN="getmaize.ai"
APP_DIR="/opt/maize"
DB_URL="postgres://vultradmin:AVNS_6V1BH0tYL23lFjsjGRL@vultr-prod-72d325ef-c651-4219-8d54-35da77e71244-vultr-prod-9472.vultrdb.com:16751/defaultdb"

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
SESSION_SECRET=$(openssl rand -hex 32)
sudo -u maize cat > $APP_DIR/.env << EOF
OPENAI_API_KEY=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
DATABASE_URL=postgresql://vultradmin:AVNS_6V1BH0tYL23lFjsjGRL@vultr-prod-72d325ef-c651-4219-8d54-35da77e71244-vultr-prod-9472.vultrdb.com:16751/defaultdb
SESSION_SECRET=$SESSION_SECRET
admin_id=simonkleffner
admin_pw=@KLEFFNER98
ADMIN_SECRET_KEY=maize-admin-2024
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASS=
EOF

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
