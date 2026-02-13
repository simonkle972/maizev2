"""Gunicorn configuration file for production deployment."""
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 5

# Logging
accesslog = "/var/log/maize/access.log"
errorlog = "/var/log/maize/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "maize-app"

# Server mechanics
daemon = False
pidfile = "/var/run/maize/maize.pid"
user = None
group = None
tmp_upload_dir = None

# SSL (if needed later)
# keyfile = "/path/to/key.pem"
# certfile = "/path/to/cert.pem"
