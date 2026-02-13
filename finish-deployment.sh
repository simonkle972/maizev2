#!/bin/bash
# Finish deployment script - completes nginx and SSL setup

set -e

echo "🌐 Configuring nginx..."

# Create HTTP-only nginx config
cat > /tmp/maize-nginx << 'EOF'
server {
    listen 80;
    server_name getmaize.ai;

    client_max_body_size 50M;

    access_log /var/log/nginx/maize_access.log;
    error_log /var/log/nginx/maize_error.log;

    location /static {
        alias /opt/maize/static;
        expires 30d;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
EOF

sudo mv /tmp/maize-nginx /etc/nginx/sites-available/maize
sudo ln -sf /etc/nginx/sites-available/maize /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

echo "✅ Testing nginx configuration..."
sudo nginx -t

echo "🔄 Restarting nginx..."
sudo systemctl restart nginx

echo "🔒 Installing certbot and getting SSL certificate..."
sudo apt-get update
sudo apt-get install -y certbot python3-certbot-nginx

echo "🔐 Getting SSL certificate for getmaize.ai..."
sudo certbot --nginx -d getmaize.ai --email simon@getmaize.ai --agree-tos --no-eff-email --redirect --non-interactive

echo "✅ Deployment complete!"
echo ""
echo "🎉 Your app is live at: https://getmaize.ai"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status maize       # Check app status"
echo "  sudo systemctl restart maize      # Restart app"
echo "  sudo journalctl -u maize -f       # View logs"
echo "  sudo systemctl restart nginx      # Restart nginx"
