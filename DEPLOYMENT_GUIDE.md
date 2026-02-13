# Maize Teaching Assistant - Vultr Deployment Guide

This guide will walk you through deploying your Maize application on Vultr VPS with a managed PostgreSQL database.

## Prerequisites
- Vultr account (sign up at https://www.vultr.com)
- Domain name configured (we'll point it to your VPS)
- GitHub repository already set up at https://github.com/simonkle972/maizev2.git

## Step 1: Set Up Vultr Managed PostgreSQL Database

1. **Log into Vultr Dashboard**
   - Go to https://my.vultr.com

2. **Create Managed Database**
   - Click "Databases" in the left sidebar
   - Click "Deploy Database"
   - Choose:
     - **Database Engine**: PostgreSQL 15 or 16
     - **Server Location**: Choose closest to you (e.g., New York, London)
     - **Plan**: Start with the smallest plan ($15/month - 1GB RAM)
     - **Label**: `maize-db`
   - Click "Deploy Now"

3. **Enable pgvector Extension**
   - Wait for database to deploy (3-5 minutes)
   - Click on your database
   - Go to "Settings" or "Advanced" tab
   - Look for "Extensions" or use the connection info to connect via psql
   - Run this command:
     ```sql
     CREATE EXTENSION IF NOT EXISTS vector;
     ```

4. **Get Connection String**
   - In your database dashboard, find the connection details
   - Copy the **PostgreSQL connection string** - it looks like:
     ```
     postgresql://user:password@host-abc123.vultr.com:16752/defaultdb?sslmode=require
     ```
   - **Save this - you'll need it for the deployment script**

## Step 2: Set Up Vultr VPS Instance

1. **Deploy VPS**
   - Click "Compute" in sidebar → "Deploy Server"
   - Choose:
     - **Server Type**: Cloud Compute
     - **Location**: Same as database (for best performance)
     - **Server Image**: Ubuntu 22.04 LTS
     - **Plan**: Start with $12/month (2GB RAM, 1 vCPU)
     - **Additional Features**: Enable "Auto Backups" (recommended)
     - **Server Hostname**: `maize-app`
   - Click "Deploy Now"

2. **Note Server IP Address**
   - Wait for server to deploy (1-2 minutes)
   - Copy the **IP address** shown in the dashboard

3. **Access Your VPS**
   - Use SSH to connect:
     ```bash
     ssh root@YOUR_SERVER_IP
     ```
   - Password will be in the Vultr dashboard or emailed to you

## Step 3: Configure Domain

1. **Point Domain to VPS**
   - Go to your domain registrar (GoDaddy, Namecheap, etc.)
   - Add an **A Record**:
     - Name: `@` (or subdomain like `app`)
     - Value: Your VPS IP address
     - TTL: 300 (or default)
   - Wait 5-30 minutes for DNS to propagate

2. **Verify DNS**
   ```bash
   dig your-domain.com
   # or
   nslookup your-domain.com
   ```

## Step 4: Deploy Application

1. **Connect to VPS**
   ```bash
   ssh root@YOUR_SERVER_IP
   ```

2. **Download Deployment Script**
   ```bash
   wget https://raw.githubusercontent.com/simonkle972/maizev2/main/deploy.sh
   chmod +x deploy.sh
   ```

3. **Edit Deployment Script**
   ```bash
   nano deploy.sh
   ```
   - Update these variables at the top:
     - `DOMAIN`: Your actual domain name
     - `DB_URL`: The PostgreSQL connection string from Step 1

4. **Run Deployment**
   ```bash
   ./deploy.sh
   ```
   - This will take 5-10 minutes
   - It will install all dependencies, set up nginx, SSL, and start your app

5. **Verify Deployment**
   - Check if app is running:
     ```bash
     sudo systemctl status maize
     ```
   - Visit your domain: `https://your-domain.com`

## Step 5: Verify Everything Works

1. **Test Health Endpoint**
   ```bash
   curl https://your-domain.com/health
   ```
   Should return: `{"status":"healthy"}`

2. **Test Admin Login**
   - Go to `https://your-domain.com/admin`
   - Login with:
     - Username: `simonkleffner`
     - Password: `@KLEFFNER98`

3. **Check Logs**
   ```bash
   # Application logs
   sudo journalctl -u maize -f

   # Nginx logs
   sudo tail -f /var/log/nginx/maize_error.log

   # Gunicorn logs
   sudo tail -f /var/log/maize/error.log
   ```

## Useful Management Commands

```bash
# Restart application
sudo systemctl restart maize

# Stop application
sudo systemctl stop maize

# Start application
sudo systemctl start maize

# View logs
sudo journalctl -u maize -f

# Pull latest code and restart
cd /opt/maize
sudo -u maize git pull
sudo systemctl restart maize

# Renew SSL certificate (happens automatically, but manual command)
sudo certbot renew
```

## Troubleshooting

### App won't start
```bash
# Check logs
sudo journalctl -u maize -n 100

# Check if port 8000 is in use
sudo netstat -tulpn | grep 8000

# Restart the service
sudo systemctl restart maize
```

### Database connection issues
```bash
# Test database connection
psql "YOUR_DATABASE_URL"

# Check if pgvector extension is installed
psql "YOUR_DATABASE_URL" -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### Nginx issues
```bash
# Test nginx configuration
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx

# Check nginx logs
sudo tail -f /var/log/nginx/maize_error.log
```

## Security Recommendations

1. **Change root password**
   ```bash
   passwd
   ```

2. **Set up SSH keys** (disable password auth)
   ```bash
   # On your local machine
   ssh-copy-id root@YOUR_SERVER_IP

   # Then disable password auth
   sudo nano /etc/ssh/sshd_config
   # Set: PasswordAuthentication no
   sudo systemctl restart sshd
   ```

3. **Set up firewall**
   ```bash
   sudo ufw allow 22/tcp    # SSH
   sudo ufw allow 80/tcp    # HTTP
   sudo ufw allow 443/tcp   # HTTPS
   sudo ufw enable
   ```

4. **Enable automatic security updates**
   ```bash
   sudo apt-get install unattended-upgrades
   sudo dpkg-reconfigure -plow unattended-upgrades
   ```

## Updating Your Application

When you push new code to GitHub:

```bash
# SSH into VPS
ssh root@YOUR_SERVER_IP

# Pull latest code
cd /opt/maize
sudo -u maize git pull

# Restart application
sudo systemctl restart maize
```

## Cost Estimate

- **Managed Database**: $15/month (1GB RAM)
- **VPS**: $12/month (2GB RAM)
- **Domain**: $10-15/year
- **Total**: ~$27/month + domain

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review logs: `sudo journalctl -u maize -f`
3. Verify database connection
4. Check Vultr service status

---

**Ready to deploy?** Start with Step 1 and work through each section carefully!
